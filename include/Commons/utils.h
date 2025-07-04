#pragma once

#ifndef _utils_h_
#define _utils_h_
// std
#include <iostream>
#include <map>
#include <vector>
#include <memory>
#include <type_traits>
// mlir
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
// pass
#include "mlir/Pass/Pass.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// mlir builder
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/TypeUtilities.h"
// other dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
// llvm
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
// commons include
#include "Commons/enums.h"
#include "Commons/defines.h"


#include "mlir/IR/Verifier.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

namespace DeepGen {

#ifdef KCG_DEBUG
#define LOG_DEBUG(message,mod)  \
{\
  llvm::outs() << message;llvm::outs().flush(); mod.dump();\
}
#else
#define LOG_DEBUG(message,mod)  ;
#endif

// ========================== about type func ===============================

inline mlir::Type getDType(mlir::OpBuilder builder, DType dtype) {
  // get mlir type
  if (dtype == DType::FLOAT16) return builder.getF16Type();
  if (dtype == DType::FLOAT32) return builder.getF32Type();
  if (dtype == DType::INT16) return builder.getIntegerType(16);
  if (dtype == DType::INT32) return builder.getIntegerType(32);
  if (dtype == DType::INT64) return builder.getIntegerType(64);
  assert(false && "getDType:: Unsupported Type!");
  return nullptr;
}

inline mlir::MemRefType getMemType(std::vector<int64_t> shape, mlir::Type arith_type, MemorySpace mem_space) {
  // create a mem type
  int ms = static_cast<int>(mem_space);
  return mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape), arith_type, {}, ms);
}

// =========================================================================

// =========================== about attr func =============================

template <typename T>
void setOpAttr(mlir::Operation* operation, const std::string& key, T&& val) {
  // set op attr
  mlir::OpBuilder b(operation->getContext());
  mlir::Attribute attr;
  if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
    attr = b.getStringAttr(std::forward<T>(val));
  } else if constexpr (std::is_same_v<std::decay_t<T>, int32_t>) {
    attr = b.getI32IntegerAttr(val);
  } else if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
    attr = b.getIntegerAttr(b.getI1Type(), val);
  } else {
    assert(false && "Unsupported attribute type");
  }
  operation->setAttr(key, attr);
}

template <typename T>
void setOpAttrArray(mlir::Operation* operation, const std::string& key, const std::vector<T>& val) {
  // set op attr
  mlir::OpBuilder b(operation->getContext());
  llvm::SmallVector<mlir::Attribute> attr_array;
  for (auto&& elem : val) {
    if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
      attr_array.push_back(b.getStringAttr(elem));
    } else if constexpr (std::is_same_v<std::decay_t<T>, int32_t>) {
      attr_array.push_back(b.getI32IntegerAttr(elem));
    } else if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
      attr_array.push_back(b.getIntegerAttr(b.getI1Type(), elem));
    } else {
      assert(false && "Unsupported array element type");
    }
  }
  operation->setAttr(key, b.getArrayAttr(attr_array));
}

template <typename T>
T getOpAttr(mlir::Operation* operation, const std::string& key) {
  // get op attr
  mlir::Attribute attr = operation->getAttr(key);
  if (!attr) {
    assert(false && "Attribute not found");
    return T{};
  } 
  if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr)) {
      return strAttr.getValue().str();
    }
  } else if constexpr (std::is_same_v<std::decay_t<T>, int32_t> || std::is_same_v<std::decay_t<T>, bool>) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
      return static_cast<T>(intAttr.getInt());
    }
  } 
  assert(false && "Attribute type mismatch");
  return T{};
}

template <typename T>
std::vector<T> getOpAttrArray(mlir::Operation* operation, const std::string& key) {
  // get op attr
  std::vector<T> result;
  mlir::Attribute attr = operation->getAttr(key);
  if (!attr) {
    assert(false && "Attribute not found");
    return T{};
  } 
  if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (mlir::Attribute elemAttr : arrayAttr) {
      if constexpr (std::is_same_v<T, std::string>) {
        if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elemAttr)) {
          result.push_back(strAttr.getValue().str());
        }
      } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, bool>) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(elemAttr)) {
          result.push_back(static_cast<T>(intAttr.getInt()));
        }
      } else {
        assert(false && "Unsupported array element type");
      }
    }
    return result;
  }
  assert(false && "Attribute type mismatch");
  return T{};
}

// ==========================================================================
inline std::string getenv(const char *name) {
  //获取环境变量
  const char *cstr = std::getenv(name);
  if (!cstr)
    return "";
  std::string result(cstr);
  return result;
}



}

#endif