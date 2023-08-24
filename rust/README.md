# Rust inference

## Run with docker

Pull docker `rust:latest` and run

``` bash
docker pull rust:latest
docker run --name mnist_cls.rust -v /path/to/workdir:/workspace rust:latest
```

## Setup

``` bash
mkdir external_libraries && cd external_libraries
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip && rm libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
export LIBTORCH=/workspace/rust/mnist_cls/external_libraries/libtorch
# LIBTORCH_INCLUDE must contains `include` directory.
export LIBTORCH_INCLUDE=/workspace/rust/mnist_cls/external_libraries/libtorch
# LIBTORCH_LIB must contains `lib` directory.
export LIBTORCH_LIB=/workspace/rust/mnist_cls/external_libraries/libtorch
export LD_LIBRARY_PATH=/workspace/rust/mnist_cls/external_libraries/libtorch/lib:$LD_LIBRARY_PATH
```