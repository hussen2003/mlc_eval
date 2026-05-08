# Install llama.cpp on Jetson AGX Xavier 16 GB Using Vulkan

This document describes the setup path for building and running `llama.cpp` on a Jetson AGX Xavier 16 GB using the Vulkan backend.

The target system is assumed to be running JetPack 5.x. JetPack 5.x ships with CUDA 11.4, but this setup intentionally does not use CUDA. The goal is to reproduce a Vulkan-based `llama.cpp` installation.

## Target Hardware

```text
Jetson AGX Xavier 16 GB
````

## Target Software Assumptions

```text
JetPack 5.x
Ubuntu-based Jetson Linux environment
Vulkan backend
No CUDA backend
No Conda required
```

## Directory Layout

This setup uses the following working directory layout:

```text
llama-src/
└── llama.cpp/
```

## 1. Install System Dependencies

Update the package index.

```bash
sudo apt update
```

Install required build tools and Python tooling.

```bash
sudo apt install -y \
  build-essential git cmake ninja-build pkg-config \
  python3 python3-pip python3-dev
```

Install Vulkan development and diagnostic packages.

```bash
sudo apt install -y \
  libvulkan-dev vulkan-tools \
  spirv-tools \
  spirv-headers \
  glslang-dev
```

## 2. Verify Vulkan Is Available

Before building `llama.cpp`, confirm that the Jetson can see a Vulkan-capable device.

```bash
vulkaninfo | grep -Ei "deviceName|driverName|apiVersion"
```

Expected result: the command should print Vulkan device, driver, and API information.

Example output will vary by Jetson software version, but it should include fields similar to:

```text
deviceName
driverName
apiVersion
```

If this command fails, stop here and fix Vulkan visibility before continuing.

## 3. Create Source Directory

Create a workspace directory.

```bash
mkdir -p llama-src
cd llama-src
```

## 4. Clone llama.cpp

Clone the upstream `llama.cpp` repository.

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

Initialize any required submodules.

```bash
git submodule update --init --recursive
```

At this point, the working directory should be:

```text
llama-src/llama.cpp
```

## 5. Configure llama.cpp With Vulkan Enabled

Create a dedicated build directory for the Vulkan build.

```bash
mkdir -p build-vulkan
cd build-vulkan
```

Configure the project with CMake and enable the Vulkan backend.

```bash
cmake .. -G Ninja \
  -DGGML_VULKAN=ON
```

This build configuration intentionally enables Vulkan and does not enable CUDA.

## 6. Build llama.cpp

Build using Ninja.

```bash
ninja -j"$(nproc)"
```

If the Jetson runs out of memory during compilation, reduce the number of parallel build jobs.

```bash
ninja -j2
```

If memory pressure still occurs, use a single job.

```bash
ninja -j1
```

## 7. Verify Build Output

After the build completes, list the generated binaries.

```bash
ls -lh bin
```

Expected binaries may include:

```text
llama-cli
llama-server
llama-bench
llama-quantize
```

The exact binaries can vary depending on the `llama.cpp` version.

If `bin/llama-cli` is not present, locate executable files with:

```bash
find . -maxdepth 3 -type f -executable | sort
```

## 8. Verify llama.cpp CLI

Run the CLI help command.

```bash
./bin/llama-cli --help
```

Expected result: the command should print the available `llama-cli` options.

If this works, the Vulkan-enabled `llama.cpp` build completed successfully.

## 9. Install Python Helper Tools for Model Download

Install the Hugging Face CLI helper.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install huggingface-hub
```

Confirm the command is available.

```bash
huggingface-cli --help
```

If the shell cannot find `huggingface-cli`, use the Python module form instead:

```bash
python3 -m huggingface_hub.commands.huggingface_cli --help
```

## 10. Download a Small GGUF Test Model

`llama.cpp` runs GGUF model files. Use a small quantized model for the first reproducibility test.

From the `build-vulkan` directory, create a model directory one level above the build directory:

```bash
mkdir -p ../models
```

Download a small GGUF model.

```bash
huggingface-cli download \
  TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --local-dir ../models \
  --local-dir-use-symlinks False
```

After download, verify the model file exists.

```bash
ls -lh ../models
```

Expected file:

```text
tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## 11. Run a Vulkan Inference Test

From the `build-vulkan` directory, run:

```bash
./bin/llama-cli \
  -m ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Explain what llama.cpp is in one paragraph." \
  -n 128 \
  -ngl 99
```

The `-ngl 99` option requests GPU offload for as many model layers as possible.

If the command fails because of memory pressure, reduce GPU offload:

```bash
./bin/llama-cli \
  -m ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Explain what llama.cpp is in one paragraph." \
  -n 128 \
  -ngl 20
```

If it still fails, reduce further:

```bash
./bin/llama-cli \
  -m ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Explain what llama.cpp is in one paragraph." \
  -n 128 \
  -ngl 10
```

Expected result: the model should generate text from the prompt.

## 12. Run llama.cpp Server With Vulkan

From the `build-vulkan` directory, run:

```bash
./bin/llama-server \
  -m ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
```

In another terminal on the Jetson, test the server health endpoint.

```bash
curl http://127.0.0.1:8080/health
```

Expected result: the server should return a health response.

If `-ngl 99` causes memory errors, stop the server and retry with a lower value.

```bash
./bin/llama-server \
  -m ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -ngl 20 \
  --host 0.0.0.0 \
  --port 8080
```

## 13. Reproducibility Checklist

A successful setup should satisfy all of the following checks.

Vulkan is visible:

```bash
vulkaninfo | grep -Ei "deviceName|driverName|apiVersion"
```

The repository exists:

```bash
test -d ~/llama-src/llama.cpp || test -d llama-src/llama.cpp
```

The Vulkan build directory exists:

```bash
test -d build-vulkan
```

The CLI works:

```bash
./bin/llama-cli --help
```

The test model exists:

```bash
ls -lh ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Inference works:

```bash
./bin/llama-cli \
  -m ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Hello from Jetson AGX Xavier." \
  -n 64 \
  -ngl 20
```

## 14. Troubleshooting

### `vulkaninfo` does not show a device

Run the full command:

```bash
vulkaninfo
```

If it fails, reinstall the Vulkan packages:

```bash
sudo apt install -y \
  libvulkan-dev vulkan-tools \
  spirv-tools \
  spirv-headers \
  glslang-dev
```

If Vulkan still does not work, confirm that the Jetson graphics stack is correctly installed through JetPack.

Do not continue until `vulkaninfo` works.

### CMake cannot find Vulkan

Install the Vulkan development package.

```bash
sudo apt install -y libvulkan-dev vulkan-tools
```

Then recreate the build directory.

```bash
cd ~/llama-src/llama.cpp

rm -rf build-vulkan
mkdir -p build-vulkan
cd build-vulkan

cmake .. -G Ninja \
  -DGGML_VULKAN=ON
```

### Build runs out of memory

Reduce the number of parallel jobs.

```bash
ninja -j2
```

or:

```bash
ninja -j1
```

On Jetson AGX Xavier 16 GB, this is a common issue during native compilation.

### `llama-cli` is not found

Check the build output.

```bash
ls -lh bin
```

If the binary is not under `bin`, search for executables.

```bash
find . -maxdepth 3 -type f -executable | sort
```

Use the actual path printed by the search command.

### Model download fails

Install or update the Hugging Face Hub CLI.

```bash
python3 -m pip install --upgrade huggingface-hub
```

Then retry the download.

If the model repository or file name is unavailable, use another compatible `.gguf` model.

### Inference fails with GPU memory errors

Lower the GPU layer offload value.

Start with:

```bash
-ngl 20
```

If that fails, try:

```bash
-ngl 10
```

If that still fails, omit `-ngl` to test CPU-only execution.

```bash
./bin/llama-cli \
  -m ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Hello" \
  -n 64
```

## 15. Full Reproduction Command Sequence

The following command sequence reproduces the Vulkan setup from a clean shell.

```bash
sudo apt update

sudo apt install -y \
  build-essential git cmake ninja-build pkg-config \
  python3 python3-pip python3-dev

sudo apt install -y \
  libvulkan-dev vulkan-tools \
  spirv-tools \
  spirv-headers \
  glslang-dev

vulkaninfo | grep -Ei "deviceName|driverName|apiVersion"

mkdir -p llama-src
cd llama-src

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

git submodule update --init --recursive

mkdir -p build-vulkan
cd build-vulkan

cmake .. -G Ninja \
  -DGGML_VULKAN=ON

ninja -j"$(nproc)"

./bin/llama-cli --help

python3 -m pip install --upgrade pip
python3 -m pip install huggingface-hub

mkdir -p ../models

huggingface-cli download \
  TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --local-dir ../models \
  --local-dir-use-symlinks False

ls -lh ../models

./bin/llama-cli \
  -m ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Explain what llama.cpp is in one paragraph." \
  -n 128 \
  -ngl 20
```

## 16. Final Expected State

After completing this setup, the Jetson AGX Xavier should have:

```text
llama.cpp cloned from source
llama.cpp built with GGML_VULKAN=ON
llama-cli available
llama-server available
A small GGUF model downloaded
Successful text generation through the Vulkan backend
```
