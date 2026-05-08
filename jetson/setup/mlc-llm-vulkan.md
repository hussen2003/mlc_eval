# Install MLC LLM on Jetson AGX Xavier 16 GB Using Vulkan

This guide documents how to install MLC LLM on a Jetson AGX Xavier 16 GB using a Vulkan-based build path.

The target system is assumed to be running JetPack 5.x. JetPack 5.x ships with CUDA 11.4, while current MLC LLM prebuilt Python packages are aligned with newer accelerator stacks such as CUDA 12.8, CUDA 13, ROCm 6.1, ROCm 6.2, and Vulkan. Because of that CUDA version mismatch, this setup uses Vulkan instead of CUDA.

## Target Environment

Hardware:

```text
Jetson AGX Xavier 16 GB
````

Software assumptions:

```text
JetPack 5.x
CUDA 11.4
Conda available
Python 3.11 environment
Vulkan build path
```

Directory layout used in this guide:

```text
mlc-src/
├── tvm-unity/
└── mlc-llm/
```

## 1. Install System Dependencies

Update apt and install the required build and Vulkan packages.

```bash
sudo apt update

sudo apt install -y \
  build-essential git cmake ninja-build \
  python3-dev libvulkan-dev vulkan-tools \
  spirv-tools \
  spirv-headers \
  glslang-dev
```

## 2. Install Rust

Install Rust using `rustup`.

```bash
curl https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"
```

Verify Rust installation.

```bash
cargo --version
rustc --version
```

Both commands should print installed versions.

## 3. Create Conda Environment

Create and activate a dedicated Conda environment for the Vulkan MLC LLM build.

```bash
conda create -n mlc-vulkan python=3.11 pip -y
conda activate mlc-vulkan
```

Install Python and build dependencies from `conda-forge`.

```bash
conda install -c conda-forge -y \
  cmake ninja \
  numpy \
  packaging \
  psutil \
  typing-extensions \
  llvmdev \
  clang \
  libxml2 \
  zlib \
  zstd \
  pytest
```

## 4. Verify Vulkan Device Visibility

Check that Vulkan is available and that the Jetson GPU is visible through Vulkan.

```bash
vulkaninfo | grep -Ei "deviceName|driverName|apiVersion"
```

Expected result: the command should print Vulkan device, driver, and API information.

If this command fails, fix Vulkan visibility before continuing.

## 5. Create Source Directory

Create a workspace for the TVM Unity and MLC LLM source builds.

```bash
mkdir -p mlc-src
cd mlc-src
```

## 6. Build TVM Unity With Vulkan Enabled

Clone TVM Unity recursively.

```bash
git clone --recursive https://github.com/apache/tvm tvm-unity
cd tvm-unity
```

Create the build directory and copy the default CMake configuration.

```bash
mkdir -p build
cp cmake/config.cmake build/config.cmake
```

Edit the CMake configuration file.

```bash
nano build/config.cmake
```

Change or add the following lines:

```cmake
set(USE_LLVM "<PATH TO CONDA_ENVS>/conda-envs/mlc-vulkan/bin/llvm-config")
set(USE_VULKAN ON)
set(USE_CUDA OFF)
set(USE_CUDNN OFF)
set(USE_CUBLAS OFF)
set(USE_CUTLASS OFF)
set(USE_ROCM OFF)
```

Replace the placeholder path with the actual path to `llvm-config` inside the `mlc-vulkan` Conda environment.

You can find the correct path with:

```bash
which llvm-config
```

For example:

```cmake
set(USE_LLVM "/home/ubuntu/miniforge3/envs/mlc-vulkan/bin/llvm-config")
```

Build TVM.

```bash
cd build
cmake .. -G Ninja
ninja -j"$(nproc)"
```

Install TVM into the active Conda environment in editable mode.

```bash
cd ..
python -m pip install -e .
```

## 7. Verify TVM Vulkan Support

Run the following Python check.

```bash
python - <<'PY'
import tvm

print("TVM Python:", tvm.__file__)
print("TVM library:", tvm.base._LIB)
print("Vulkan exists:", tvm.vulkan().exist)
print("Runtime enabled Vulkan:", tvm.runtime.enabled("vulkan"))
PY
```

Expected output should include:

```text
Vulkan exists: True
Runtime enabled Vulkan: True
```

If either value is `False`, TVM was not built with working Vulkan support or Vulkan is not visible on the system.

## 8. Build MLC LLM

Return to the source directory.

```bash
cd ../
```

If you are currently inside `mlc-src/tvm-unity`, this should place you back in:

```text
mlc-src/
```

Clone MLC LLM recursively.

```bash
git clone --recursive https://github.com/mlc-ai/mlc-llm.git
cd mlc-llm
```

Create and enter the build directory if it does not already exist.

```bash
mkdir -p build
cd build
```

Generate the CMake configuration.

```bash
python ../cmake/gen_cmake_config.py
```

Build MLC LLM.

```bash
cmake .. -G Ninja
ninja -j"$(nproc)"
```

Install MLC LLM into the active Conda environment in editable mode.

```bash
cd ..
python -m pip install -e .
```

## 9. Verify MLC LLM Installation

Check that the Python package imports.

```bash
python -c "import mlc_llm; print(mlc_llm)"
```

Check the CLI.

```bash
mlc_llm --help
```

Both commands should complete without errors.

## 10. Notes

This setup intentionally disables CUDA-related TVM build flags:

```cmake
set(USE_CUDA OFF)
set(USE_CUDNN OFF)
set(USE_CUBLAS OFF)
set(USE_CUTLASS OFF)
```

This is because Jetson AGX Xavier on JetPack 5.x uses CUDA 11.4, while current MLC LLM prebuilt CUDA packages target newer CUDA versions. Vulkan is used as the compatibility path.

## 11. Troubleshooting

### `vulkaninfo` does not show a device

Run:

```bash
vulkaninfo
```

If it fails completely, confirm that Vulkan packages are installed:

```bash
sudo apt install -y libvulkan-dev vulkan-tools
```

Also confirm that the Jetson graphics stack is correctly installed through JetPack.

### `tvm.vulkan().exist` returns `False`

Recheck `build/config.cmake` inside `tvm-unity` and confirm:

```cmake
set(USE_VULKAN ON)
```

Then rebuild TVM:

```bash
cd mlc-src/tvm-unity/build
cmake .. -G Ninja
ninja -j"$(nproc)"
cd ..
python -m pip install -e .
```

### `tvm.runtime.enabled("vulkan")` returns `False`

This usually means TVM was not compiled with Vulkan runtime support. Confirm that the TVM build used the edited config file at:

```text
mlc-src/tvm-unity/build/config.cmake
```

Then rerun CMake and Ninja.

### `llvm-config` path is wrong

Inside the active Conda environment, run:

```bash
which llvm-config
```

Use that exact path in:

```cmake
set(USE_LLVM "/absolute/path/to/mlc-vulkan/bin/llvm-config")
```

### Build runs out of memory

Jetson AGX Xavier 16 GB may run out of memory during parallel compilation. Reduce the number of jobs:

```bash
ninja -j2
```

or:

```bash
ninja -j1
```

### Conda environment is not active

Before building or installing, confirm:

```bash
conda activate mlc-vulkan
which python
which pip
```

Both `python` and `pip` should resolve inside the `mlc-vulkan` environment.

## 12. Full Command Sequence

The following is a condensed command sequence. Edit the TVM CMake config manually at the indicated step.

```bash
sudo apt update

sudo apt install -y \
  build-essential git cmake ninja-build \
  python3-dev libvulkan-dev vulkan-tools \
  spirv-tools \
  spirv-headers \
  glslang-dev

curl https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"

cargo --version
rustc --version

conda create -n mlc-vulkan python=3.11 pip -y
conda activate mlc-vulkan

conda install -c conda-forge -y \
  cmake ninja \
  numpy \
  packaging \
  psutil \
  typing-extensions \
  llvmdev \
  clang \
  libxml2 \
  zlib \
  zstd \
  pytest

vulkaninfo | grep -Ei "deviceName|driverName|apiVersion"

mkdir -p mlc-src
cd mlc-src

git clone --recursive https://github.com/apache/tvm tvm-unity
cd tvm-unity

mkdir -p build
cp cmake/config.cmake build/config.cmake

# Edit build/config.cmake:
# set(USE_LLVM "<absolute path from: which llvm-config>")
# set(USE_VULKAN ON)
# set(USE_CUDA OFF)
# set(USE_CUDNN OFF)
# set(USE_CUBLAS OFF)
# set(USE_CUTLASS OFF)
# set(USE_ROCM OFF)

cd build
cmake .. -G Ninja
ninja -j"$(nproc)"

cd ..
python -m pip install -e .

python - <<'PY'
import tvm

print("TVM Python:", tvm.__file__)
print("TVM library:", tvm.base._LIB)
print("Vulkan exists:", tvm.vulkan().exist)
print("Runtime enabled Vulkan:", tvm.runtime.enabled("vulkan"))
PY

cd ../

git clone --recursive https://github.com/mlc-ai/mlc-llm.git
cd mlc-llm

mkdir -p build
cd build

python ../cmake/gen_cmake_config.py

cmake .. -G Ninja
ninja -j"$(nproc)"

cd ..
python -m pip install -e .

python -c "import mlc_llm; print(mlc_llm)"
mlc_llm --help
```
