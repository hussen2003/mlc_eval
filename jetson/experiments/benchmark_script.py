#!/usr/bin/env python3
"""
Benchmark llama.cpp vs MLC-LLM on Jetson using Vulkan.

Metrics:
  - TTFT: time to first streamed output token/content chunk, in seconds
  - TPS: generated output tokens per second, measured after TTFT

Output files:
  - results/raw_results.jsonl
  - results/results.csv
  - results/summary.csv
  - results/run_metadata.json

Assumptions:
  - llama.cpp is already built with Vulkan.
  - MLC-LLM is already installed and usable with Vulkan.
  - Both servers expose an OpenAI-compatible /v1/chat/completions endpoint.
  - Python package "requests" is installed.

Install dependency:
  python3 -m pip install requests

Edit MODEL_CONFIG before running.
"""

import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


RUNS_PER_MODEL = 10
PROMPT = "hi"
MAX_TOKENS = 64
TEMPERATURE = 0.0
REQUEST_TIMEOUT_SECONDS = 300
SERVER_START_TIMEOUT_SECONDS = 180
RESULTS_DIR = Path("results")


# If reproducing, edit these paths/model IDs for your setup.
MODEL_CONFIG = [
    {
        "model_label": "tinyllama-1.1b-chat-q4",
        "llama_cpp": {
            "enabled": True,
            "server_bin": "/home/nvidia/sdcard/llama-src/llama.cpp/build-vulkan/bin/llama-server",
            "model_path": "/home/nvidia/sdcard/llama-src/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "host": "127.0.0.1",
            "port": 8081,
            "ngl": 20,
        },
        "mlc_llm": {
            "enabled": True,
            "model": "HF://mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
            "host": "127.0.0.1",
            "port": 9091,
            "extra_args": [],
        },
    },
    {
        "model_label": "qwen2.5-0.5b-instruct-q4",
        "llama_cpp": {
            "enabled": True,
            "server_bin": "/home/nvidia/sdcard/llama-src/llama.cpp/build-vulkan/bin/llama-server",
            "model_path": "/home/nvidia/sdcard/llama-src/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "host": "127.0.0.1",
            "port": 8082,
            "ngl": 20,
        },
        "mlc_llm": {
            "enabled": True,
            "model": "HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
            "host": "127.0.0.1",
            "port": 9092,
            "extra_args": [],
        },
    },
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def approx_token_count(text: str) -> int:
    """
    Lightweight tokenizer approximation for output TPS comparison.
    This avoids depending on model-specific tokenizers.
    """
    if not text:
        return 0
    pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(1, len(pieces))


def make_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_server(base_url: str, timeout_s: int) -> None:
    deadline = time.perf_counter() + timeout_s
    last_error = None

    endpoints = [
        f"{base_url}/health",
        f"{base_url}/v1/models",
    ]

    while time.perf_counter() < deadline:
        for url in endpoints:
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if 200 <= response.status < 500:
                        return
            except Exception as exc:
                last_error = exc
        time.sleep(1)

    raise RuntimeError(f"Server did not become ready at {base_url}. Last error: {last_error}")


def start_llama_cpp_server(cfg: Dict) -> subprocess.Popen:
    cmd = [
        cfg["server_bin"],
        "-m",
        cfg["model_path"],
        "-ngl",
        str(cfg["ngl"]),
        "--host",
        cfg["host"],
        "--port",
        str(cfg["port"]),
    ]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )


def start_mlc_llm_server(cfg: Dict) -> subprocess.Popen:
    cmd = [
        "mlc_llm",
        "serve",
        cfg["model"],
        "--device",
        "vulkan",
        "--host",
        cfg["host"],
        "--port",
        str(cfg["port"]),
    ]

    cmd.extend(cfg.get("extra_args", []))

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )


def stop_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return

    if proc.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def read_process_output(proc: subprocess.Popen, max_lines: int = 80) -> str:
    if proc.stdout is None:
        return ""

    lines = []
    try:
        while len(lines) < max_lines:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip())
    except Exception:
        pass

    return "\n".join(lines)


def stream_chat_completion(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[float, float, str, int]:
    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    start_t = time.perf_counter()
    first_token_t = None
    end_t = None
    output_parts: List[str] = []

    with requests.post(
        url,
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue

            line = raw_line.strip()
            if not line:
                continue

            if not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()

            if data == "[DONE]":
                end_t = time.perf_counter()
                break

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = event.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content", "")

            if content:
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                output_parts.append(content)

    if first_token_t is None:
        first_token_t = time.perf_counter()

    if end_t is None:
        end_t = time.perf_counter()

    output_text = "".join(output_parts)
    output_tokens = approx_token_count(output_text)

    ttft_s = first_token_t - start_t
    decode_s = max(end_t - first_token_t, 1e-9)
    tps = output_tokens / decode_s

    return ttft_s, tps, output_text, output_tokens


def run_backend_benchmark(
    backend_name: str,
    model_label: str,
    server_proc: subprocess.Popen,
    base_url: str,
    served_model_name: str,
    runs: int,
) -> List[Dict]:
    records = []

    for run_idx in range(1, runs + 1):
        try:
            ttft_s, tps, output_text, output_tokens = stream_chat_completion(
                base_url=base_url,
                model_name=served_model_name,
                prompt=PROMPT,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            record = {
                "timestamp": now_iso(),
                "backend": backend_name,
                "model_label": model_label,
                "run": run_idx,
                "prompt": PROMPT,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "ttft_s": ttft_s,
                "tps": tps,
                "output_tokens_approx": output_tokens,
                "output_text": output_text,
                "status": "ok",
                "error": "",
            }

        except Exception as exc:
            record = {
                "timestamp": now_iso(),
                "backend": backend_name,
                "model_label": model_label,
                "run": run_idx,
                "prompt": PROMPT,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "ttft_s": "",
                "tps": "",
                "output_tokens_approx": "",
                "output_text": "",
                "status": "error",
                "error": repr(exc),
            }

        records.append(record)

        print(
            f"{backend_name} | {model_label} | run {run_idx}/{runs} | "
            f"status={record['status']} | ttft={record['ttft_s']} | tps={record['tps']}",
            flush=True,
        )

        time.sleep(1)

    return records


def write_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: List[Dict]) -> None:
    if not records:
        return

    fieldnames = [
        "timestamp",
        "backend",
        "model_label",
        "run",
        "prompt",
        "max_tokens",
        "temperature",
        "ttft_s",
        "tps",
        "output_tokens_approx",
        "status",
        "error",
        "output_text",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def summarize(records: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple[str, str], List[Dict]] = {}

    for record in records:
        if record.get("status") != "ok":
            continue

        key = (record["backend"], record["model_label"])
        groups.setdefault(key, []).append(record)

    summary_rows = []

    for (backend, model_label), rows in sorted(groups.items()):
        ttft_values = [float(r["ttft_s"]) for r in rows]
        tps_values = [float(r["tps"]) for r in rows]

        summary_rows.append(
            {
                "backend": backend,
                "model_label": model_label,
                "successful_runs": len(rows),
                "ttft_mean_s": statistics.mean(ttft_values),
                "ttft_median_s": statistics.median(ttft_values),
                "ttft_stdev_s": statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0.0,
                "tps_mean": statistics.mean(tps_values),
                "tps_median": statistics.median(tps_values),
                "tps_stdev": statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0,
            }
        )

    return summary_rows


def write_summary_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return

    fieldnames = [
        "backend",
        "model_label",
        "successful_runs",
        "ttft_mean_s",
        "ttft_median_s",
        "ttft_stdev_s",
        "tps_mean",
        "tps_median",
        "tps_stdev",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path) -> None:
    metadata = {
        "created_at": now_iso(),
        "runs_per_model": RUNS_PER_MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "metric_notes": {
            "ttft_s": "Wall-clock time from HTTP request start to first streamed content chunk.",
            "tps": "Approximate output tokens divided by time from first streamed content chunk to stream completion.",
            "token_count": "Approximate regex-based token count. Replace with model tokenizer for exact token accounting.",
        },
        "model_config": MODEL_CONFIG,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_one_server_block(
    backend_name: str,
    model_label: str,
    proc: subprocess.Popen,
    host: str,
    port: int,
    served_model_name: str,
) -> List[Dict]:
    base_url = f"http://{host}:{port}"

    try:
        wait_for_server(base_url, SERVER_START_TIMEOUT_SECONDS)
    except Exception:
        server_output = read_process_output(proc)
        stop_process(proc)
        raise RuntimeError(
            f"{backend_name} server failed to become ready for {model_label}.\n"
            f"Server output:\n{server_output}"
        )

    try:
        return run_backend_benchmark(
            backend_name=backend_name,
            model_label=model_label,
            server_proc=proc,
            base_url=base_url,
            served_model_name=served_model_name,
            runs=RUNS_PER_MODEL,
        )
    finally:
        stop_process(proc)


def main() -> int:
    make_results_dir()

    all_records: List[Dict] = []

    for model_cfg in MODEL_CONFIG:
        model_label = model_cfg["model_label"]

        llama_cfg = model_cfg["llama_cpp"]
        if llama_cfg.get("enabled", True):
            print(f"Starting llama.cpp Vulkan server for {model_label}", flush=True)
            proc = start_llama_cpp_server(llama_cfg)

            records = run_one_server_block(
                backend_name="llama.cpp-vulkan",
                model_label=model_label,
                proc=proc,
                host=llama_cfg["host"],
                port=llama_cfg["port"],
                served_model_name=llama_cfg["model_path"],
            )
            all_records.extend(records)

        mlc_cfg = model_cfg["mlc_llm"]
        if mlc_cfg.get("enabled", True):
            print(f"Starting MLC-LLM Vulkan server for {model_label}", flush=True)
            proc = start_mlc_llm_server(mlc_cfg)

            records = run_one_server_block(
                backend_name="mlc-llm-vulkan",
                model_label=model_label,
                proc=proc,
                host=mlc_cfg["host"],
                port=mlc_cfg["port"],
                served_model_name=mlc_cfg["model"],
            )
            all_records.extend(records)

    raw_path = RESULTS_DIR / "raw_results.jsonl"
    csv_path = RESULTS_DIR / "results.csv"
    summary_path = RESULTS_DIR / "summary.csv"
    metadata_path = RESULTS_DIR / "run_metadata.json"

    write_jsonl(raw_path, all_records)
    write_csv(csv_path, all_records)
    write_summary_csv(summary_path, summarize(all_records))
    write_metadata(metadata_path)

    print(f"Wrote {raw_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
