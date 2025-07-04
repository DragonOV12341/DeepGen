#pragma once
#ifndef _Lowering_h_
#define _Lowering_h_

#include "Commons/utils.h"
// lowering llvm
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

using namespace mlir;
namespace DeepGen {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createVectorToLLVMPass(unsigned indexBitWidth);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGlobalShmSetZeroPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLLVMFuncOpAddGPUAttrPass(Target target);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGPUToROCDLOrNVVMPass(Target target, unsigned indexBitwidth);
  
}

#endif