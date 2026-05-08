# mlc_eval

Benchmarking and evaluation tools for running local LLMs on Mac using MLC-LLM and Ollama, plus a web demo.

---

## Benchmarks

Both benchmarks use **Llama-3.2-1B-Instruct** and include a warmup call so model load time is excluded from the results, giving accurate real-world TTFT and TPS numbers.

### Prerequisites

```bash
conda create -n MLC python=3.11 -y
conda activate MLC
conda install -c conda-forge clang git git-lfs -y
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
pip install ollama
```

For Ollama, you also need the Ollama app installed:

```bash
brew install ollama
ollama serve         # run in a separate terminal
ollama pull llama3.2:1b
```

### bench.py — MLC-LLM

Runs inference via MLC-LLM using Metal on Apple Silicon.

```bash
python bench.py
```

### bench2.py — Ollama

Runs inference via Ollama.

```bash
python bench2.py
```

### Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token (ms) — latency before output starts |
| TPS | Tokens per second — generation speed after first token |
| Total time | Full end-to-end response time |

---

## Web Demo

A lightweight web app demo.

```bash
cd web
python -m http.server 8000
```

Then open your browser at [http://localhost:8000](http://localhost:8000).

---

## iOS/MLC Demo

See the `ios/MLC Demo` folder.
