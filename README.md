# DeepGen
## 1.介绍
该项目为重构项目，项目详细介绍请转至原项目地址[GeepGen(main-branch)](https://github.com/DeepGenGroup/DeepGen)或者[GeepGen(dev-branch)](https://github.com/DeepGenGroup/DeepGen/tree/dev_graph)

## 2.构建llvm
海光DCU，需要设置dtk 24.04.1；修改代码~/rocm-llvm-project/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
```c++
llvm::cl::init(llvm::AMDGPU::AMDHSA_COV6),
// to
llvm::cl::init(llvm::AMDGPU::AMDHSA_COV4),
```
编译构建llvm
```sh
git clone https://github.com/DeepGenGroup/rocm-llvm-project.git
git checkout deepgen-dev

cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX=~/llvm-install \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"

ninja install
```

## 3.构建项目
修改项目根目录CMake文件：
```cmake
# llvm/mlir 安装路径
set(LLVM_INSTALL_DIR "/home/xiebaokang/software/install/rocm-llvm-install")
```
编译构建deepgen
```sh
# 当前在项目目录下
mkdir build & cd build

cmake -G Ninja .. -DCOMPILE_AS_PYMODULE=OFF -DEBUG_MODE=ON
# DCOMPILE_AS_PYMODULE(ON)：绑定python（DeepGen/src/main.cc）
# DCOMPILE_AS_PYMODULE(OFF)：C++测试（DeepGen/src/test.cc）
```

## 4.补充
该项目已完成：
1. kernel naive express -> MLIR(affine/memref/...)
2. nvvm/rocdl + mlir-llvm -> hsaco/cubin
未完成：
1. optimize