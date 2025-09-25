cmake_minimum_required(VERSION 3.15.0)
project(KernelCodeGen LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(LLVM_INSTALL_DIR "~/rocm-llvm-install")

set(USER_PTXAS_PATH "/usr/local/cuda-12.4/bin/ptxas") 
set(USER_LLD_PATH "/opt/dtk/llvm/bin/ld.lld")
set(PTX_DUMP_PATH "${CMAKE_SOURCE_DIR}/.dump/ass.ptx") 
set(GCN_DUMP_PATH "${CMAKE_SOURCE_DIR}/.dump/ass.s") 
# set(STABLEHLO_SRC_DIR "~/software/rocm-stablehlo")
# set(STABLEHLO_BUILD_DIR "~/software/rocm-stablehlo/build")
# set(STABLEHLO_INSATLL_DIR "/home/xiebaokang/software/install/stablehlo-install/lib")

find_package(Python COMPONENTS Interpreter Development)
if(Python_FOUND)
    message(STATUS "=== Found Python: ${Python_EXECUTABLE}")
    message(STATUS "=== Python Include: ${Python_INCLUDE_DIRS}")
    message(STATUS "=== Python Libraries: ${Python_LIBRARIES}")
endif()

# option(USE_STABLEHLO_EMBEDDED "Use embedded stableHLO(ON) or external stableHLO(OFF)" ON)
option(COMPILE_AS_PYMODULE "Compile KernelCodeGen to DynamicLib or Exe" OFF)
option(DEBUG_MODE "Enable debug log" ON)
# close some warnings
add_compile_options(
  -fexceptions
  -Wno-unused-function
  -Wno-unused-variable
  -Wno-unused-result
  -Wno-sign-compare
  -Wno-unused-but-set-variable
  -Wno-return-local-addr
  -Wno-parentheses
  -Wno-cast-qual
  -Wno-unused-but-set-parameter
  -Wno-deprecated-declarations
  -Wno-unused-value
  )

configure_file(config.h.in ${CMAKE_SOURCE_DIR}/include/config.h)

set(MLIR_DIR ${LLVM_INSTALL_DIR}/lib/cmake/mlir)
find_package(MLIR REQUIRED)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
# llvm and mlir include files
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# project inlcude files
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/build/include)
include_directories(${PROJECT_BINARY_DIR}/include)
# stablehlo include files
# include_directories(${STABLEHLO_SRC_DIR})
# include_directories(${STABLEHLO_BUILD_DIR})

link_directories(${LLVM_INSTALL_DIR}/lib)
add_definitions(${LLVM_DEFINITIONS})
message("add-def:${LLVM_DEFINITIONS}")
if(DEBUG_MODE)
  add_definitions(-DKCG_DEBUG)
  add_compile_options(-g)
endif()

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(include)
add_subdirectory(src/lib)
add_subdirectory(src)
add_subdirectory(_temptest)