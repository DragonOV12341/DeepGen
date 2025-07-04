#pragma once
#ifndef _Transform_h_
#define _Transform_h_

#include "Commons/utils.h"
#include "Transforms/ExprUtils.h"
#include "Transforms/MemUtils.h"
// transform
#include "mlir/Dialect/Affine/LoopUtils.h"


using namespace mlir;
namespace DeepGen {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createParallelToGPUPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createCombineMemrefPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> ReplaceAllocToGetglobalPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAmendAllocaOpAddrSpacePass(Target target);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAffineUnrollPass();

}

#endif