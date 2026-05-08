# mlc_eval

A cross-platform benchmark suite for local Large Language Model (LLM) inference. The project measures Time To First Token (TTFT) and Tokens Per Second (TPS) across four deployment targets:

1. NVIDIA Jetson AGX Xavier 16 GB, using llama.cpp and MLC-LLM, both built against Vulkan.
2. Web browsers, using WebLLM and WebGPU
3. macOS/iOS on Apple Silicon, using MLC-LLM (Metal), Ollama, and SwiftUI client that calls a remote MLC-LLM server.

Every benchmark warms up the engine before timing so that model-load cost is excluded from the reported numbers.

---

## Table of Contents

- [Repository Layout](#repository-layout)
- [Metrics](#metrics)
- [Platform 1: NVIDIA Jetson AGX Xavier (Vulkan)](#platform-1-nvidia-jetson-agx-xavier-vulkan)
- [Platform 2: Web Browser (WebLLM)](#platform-2-web-browser-webllm)
- [Platform 3: macOS/iOS](#platform-3-macosios)
- [Full Dependency Summary](#full-dependency-summary)
- [Reproducing Results](#reproducing-results)

---

## Repository Layout

```
mlc_eval/
    README.md
    bench.py                              # macOS MLC-LLM (Metal) benchmark
    bench2.py                             # macOS Ollama benchmark

    jetson/
        setup/
            llama-cpp.md                  # Build llama.cpp with Vulkan on Jetson
            mlc-llm-vulkan.md             # Build TVM Unity + MLC-LLM with Vulkan on Jetson
        experiments/
            benchmark_script.py           # Runs both backends and writes raw + summary results
            figures_creation_script.py    # Generates plots from the result files
            results/
                raw_results.jsonl         # One JSON record per run
                results.csv               # Same data in CSV form
                summary.csv               # Mean, median, stdev per (backend, model)
                run_metadata.json         # Hardware, software, and run parameters

    web/
        index.html                        # WebLLM benchmark page
        index.js                          # Engine setup, streaming, TTFT/TPS capture
        styles.css

    ios/
        MLC Demo/                         # Xcode project for the SwiftUI client
```

---

## Metrics

| Metric     | Definition                                                           |
|------------|----------------------------------------------------------------------|
| TTFT       | Time from request submission to the first streamed content chunk.    |
| TPS        | Output tokens divided by the elapsed time after the first token.     |
| Total time | End-to-end response time, including TTFT and decode.                 |

The Jetson script uses a regex-based approximate token count so that the comparison between backends does not depend on a model-specific tokenizer. The macOS scripts count streamed chunks directly. Numbers are comparable within a script, not across scripts.

---

## Platform 1: NVIDIA Jetson AGX Xavier (Vulkan)

Contains both the install guides and a benchmarking harness that drives two independent OpenAI-compatible servers (llama.cpp's `llama-server` and `mlc_llm serve`), runs the same prompt 10 times per backend, and writes structured result files plus matplotlib figures.

### Target Hardware

- NVIDIA Jetson AGX Xavier 16 GB Developer Kit.
- JetPack 5.x flashed onto the device.

### Why Vulkan and Not CUDA

JetPack 5.x ships CUDA 11.4. The current MLC-LLM prebuilt CUDA wheels target newer CUDA versions (12.8 and above) and ROCm targets, so they will not load on a JetPack 5.x Xavier. The setup uses Vulkan as the common compatibility path, and the same backend is used for llama.cpp so that the two systems are compared on equal footing.

### Installing the Two Backends

Follow the two setup guides in order. They are designed to be run from a clean Jetson shell and include verification steps:

1. `jetson/setup/llama-cpp.md` builds `llama.cpp` with `-DGGML_VULKAN=ON`, downloads a TinyLlama GGUF model, and verifies CLI plus server inference.
2. `jetson/setup/mlc-llm-vulkan.md` creates a `mlc-vulkan` conda env, builds TVM Unity with Vulkan enabled (CUDA explicitly off), and then builds MLC-LLM against that TVM.

After both guides complete, you should have:

```
~/llama-src/llama.cpp/build-vulkan/bin/llama-server      # built from source
mlc_llm                                                   # CLI on PATH inside the mlc-vulkan conda env
```

### Configuring the Benchmark Script

`jetson/experiments/benchmark_script.py` is self-contained, but the `MODEL_CONFIG` block at the top of the file points at hardcoded paths that match the tested machine. You must edit it before running. The fields you need to change:

- `llama_cpp.server_bin` -- absolute path to your built `llama-server`.
- `llama_cpp.model_path` -- absolute path to a `.gguf` model on disk.
- `mlc_llm.model` -- the `HF://` model identifier you want to serve.
- `host` and `port` -- pick free ports on your Jetson. The defaults are `8081`, `8082`, `9091`, `9092`.
- `ngl` -- number of model layers to offload to the GPU. Lower this if you hit Vulkan memory errors.

The harness defaults to 10 runs per (backend, model) pair, the prompt `"hi"`, `max_tokens=64`, and `temperature=0.0`. These constants live near the top of the file and can be edited.

### Installing the Benchmark Script's Python Dependencies

Inside the `mlc-vulkan` conda environment:

```bash
python3 -m pip install requests
```

Only `requests` is needed at run time, because the script uses `mlc_llm` and `llama-server` as subprocesses, not as Python libraries.

### Running the Benchmark

```bash
conda activate mlc-vulkan
cd jetson/experiments
python3 benchmark_script.py
```

For each model entry in `MODEL_CONFIG`, the script:

1. Starts `llama-server` with Vulkan offload, waits for `/health` or `/v1/models` to respond, runs 10 streamed `/v1/chat/completions` requests, then terminates the server.
2. Starts `mlc_llm serve --device vulkan` for the same logical model, waits for it to come up, runs 10 streamed requests, then terminates it.

Output is written into `results/` next to the script:

| File                | Contents                                                                |
|---------------------|-------------------------------------------------------------------------|
| `raw_results.jsonl` | One JSON line per run, including timestamp, TTFT, TPS, and output text. |
| `results.csv`       | The same data in CSV form.                                              |
| `summary.csv`       | Mean, median, and stdev per (backend, model) pair.                      |
| `run_metadata.json` | Run parameters and the model configuration used.                        |

### Generating Figures

After the benchmark completes, generate plots from the CSVs:

```bash
pip install pandas matplotlib
python3 figures_creation_script.py
```

This writes seven PNG files into `results/figures/`:

- `ttft_bar_mean.png` and `tps_bar_mean.png` -- mean with stdev error bars per backend, grouped by model.
- `ttft_boxplot.png` and `tps_boxplot.png` -- distribution across the 10 runs.
- `ttft_runs_line.png` and `tps_runs_line.png` -- per-run trajectory.
- `ttft_vs_tps_scatter.png` -- joint distribution.

The plotting script also prints a textual summary to stdout.

### Existing Results

A complete run is committed in `jetson/experiments/results/`. It was produced on Jetson AGX Xavier 16 GB with JetPack 5.x using TinyLlama 1.1B q4 and Qwen2.5 0.5B q4 against both backends, 10 runs each, prompt `"hi"`, `max_tokens=64`, `temperature=0.0`. See `run_metadata.json` for the exact configuration.

---

## Platform 2: Web Browser (WebLLM)

The `web/` folder is a static HTML demo that loads `@mlc-ai/web-llm` from a CDN and benchmarks any of six prebuilt MLC models running entirely in-browser on WebGPU.

### Browser Requirements

- A WebGPU-capable browser. Recent versions of Chrome, Edge, and Chrome Canary qualify on most platforms. Safari Technology Preview and Firefox Nightly also work in many configurations.
- A GPU with enough memory for the chosen model. The script lists `vram_required_MB` for each entry. Plan for at least 4 GB of free GPU memory for the smallest q3 model and 6 GB or more for the larger ones.
- The site must be served over HTTP (or HTTPS), not opened with `file://`. Browsers disable WebGPU and ES module imports for `file://` URLs.

### Available Models

`web/index.js` exposes three model pairs at q4 and q3 quantization each:

- Llama-3.1-8B-Instruct (q4 and q3)
- gemma-2-9b-it (q4 and q3)
- Mistral-7B-Instruct-v0.2 (q4 and q3)

The q4 entries use prebuilt `.wasm` libraries served from the WebLLM CDN. The q3 entries reference `http://localhost:8000/...wasm`, meaning you must place those compiled WebAssembly libraries in the `web/` directory yourself or change the URLs to point at hosted copies. If the q3 wasm files are not present locally, only the q4 entries will load.

### Running the Demo

From the repository root:

```bash
cd web
python -m http.server 8000
```

Open `http://localhost:8000` in a WebGPU-capable browser. The page has three steps:

1. Pick a model from the dropdown and click `Load Model`. The first load downloads weights into the browser's storage and can take several minutes on the larger models.
2. Type a benchmark prompt and press `Send`. The response streams in.
3. Each completed turn appends a row to the `Benchmark Results` table with the model name, prompt, response, TTFT in seconds, and decode TPS read from WebLLM's runtime stats.

### Reproducing Web Numbers

WebGPU performance varies a lot by browser, GPU, driver version, and even between page loads (because the in-browser engine can JIT-compile differently after the first call). Best practice:

1. Use a single browser version and pin it.
2. Close other tabs.
3. For each model, send the same prompt three to five times and report the mean of the later runs, since the first response after model load is cold.

---

## Platform 3: macOS/iOS

The two scripts in the repository root benchmark the same model (Llama-3.2-1B-Instruct, q4 quantization where applicable) against two different runtimes on macOS.

### Hardware / OS Requirements

- Apple Silicon Mac (M1, M2, M3, or M4 family).
- macOS 13 Ventura or later.
- At least 8 GB of unified memory free at run time.
- About 1 GB of free disk for the quantized model weights.

### Software Prerequisites

- Miniconda or Anaconda.
- Homebrew (only required for the Ollama path).
- Git and Git LFS.

### Environment Setup

Create the conda environment and install MLC-LLM nightly wheels plus the Ollama Python client:

```bash
conda create -n MLC python=3.11 -y
conda activate MLC
conda install -c conda-forge clang git git-lfs -y
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
pip install ollama
```

For the Ollama path you also need the Ollama server running locally:

```bash
brew install ollama
ollama serve              # leave running in its own terminal
ollama pull llama3.2:1b
```

### Running bench.py (MLC-LLM, Metal)

```bash
conda activate MLC
python bench.py
```

The script downloads `HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC` on first run, performs a warmup chat call so that Metal kernels and weights are cached, and then runs a single timed prompt:

```
"Explain the theory of relativity in simple terms."
```

It prints TTFT in milliseconds, the streamed token count, TPS, and total wall-clock time, then terminates the engine cleanly.

### Running bench2.py (Ollama)

```bash
conda activate MLC
python bench2.py
```

This benchmark uses the same prompt against the `llama3.2:1b` model served by your local Ollama process, so make sure `ollama serve` is running and the model has been pulled.

### Reproducing macOS Numbers

Each script uses a deterministic prompt and a single timed turn, so variance between runs comes mostly from system load. To produce comparable numbers:

1. Close other GPU and CPU heavy applications.
2. Plug the Mac into AC power and disable Low Power Mode.
3. Run each script three to five times, discard the first run as a system-cache warmup, and report the mean of the remaining runs.

### iOS/MLC Demo

See the `ios/MLC` Demo folder.

---

## Full Dependency Summary

| Component                   | Required                                                                          |
|-----------------------------|-----------------------------------------------------------------------------------|
| macOS bench.py              | Python 3.11, mlc-llm-nightly-cpu, mlc-ai-nightly-cpu, clang, git, git-lfs          |
| macOS bench2.py             | Python 3.11, ollama (Python client), Ollama daemon, llama3.2:1b model              |
| Jetson llama.cpp            | build-essential, cmake, ninja, libvulkan-dev, vulkan-tools, glslang-dev, python3   |
| Jetson MLC-LLM              | The above plus rustup, conda, llvmdev, clang (conda-forge), TVM Unity from source  |
| Jetson benchmark_script.py  | requests                                                                           |
| Jetson figures script       | pandas, matplotlib                                                                 |
| Web demo                    | Any HTTP server, WebGPU-capable browser, optional local q3 wasm libraries          |
| iOS app                     | Xcode 15 or later, iOS 17 or later, network access to an MLC-LLM server            |

---

## Reproducing Results

To reproduce the numbers under `jetson/experiments/results/` end to end:

1. Flash a Jetson AGX Xavier 16 GB with JetPack 5.x.
2. Pick a single `nvpmodel` profile (for example MAXN) and lock the fan curve. Do not change either between runs.
3. Follow `jetson/setup/llama-cpp.md` exactly through Section 11 (the inference test).
4. Follow `jetson/setup/mlc-llm-vulkan.md` exactly through Section 9 (the import check).
5. Download the same two models referenced in the existing `run_metadata.json`:
   - TinyLlama 1.1B Chat v1.0 Q4_K_M GGUF for llama.cpp.
   - `HF://mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC` for MLC-LLM.
   - Qwen 2.5 0.5B Instruct q4_k_m GGUF for llama.cpp.
   - `HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC` for MLC-LLM.
6. Edit the paths and ports in `MODEL_CONFIG` at the top of `benchmark_script.py` to match your filesystem.
7. Activate the `mlc-vulkan` conda environment, install `requests`, and run the script from `jetson/experiments/`.
8. Run `figures_creation_script.py` to regenerate the plots.

The exact numerical values will not match to the third decimal place because of thermal variance and driver differences, but the qualitative ranking (MLC-LLM Vulkan faster than llama.cpp Vulkan on TTFT and TPS for both models) is reproducible.

---
