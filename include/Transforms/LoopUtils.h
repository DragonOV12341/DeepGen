#pragma once
#ifndef _LoopUtils_h_
#define _LoopUtils_h_

#include "Commons/utils.h"

namespace DeepGen {
namespace LoopUtil {


struct NestedLoopResult {
    llvm::SmallVector<mlir::affine::AffineForOp> loops;
    llvm::SmallVector<mlir::Value> ivs;
};

NestedLoopResult createNestedLoops(
  mlir::OpBuilder& builder,
  llvm::SmallVector<int64_t> lowerBounds,
  llvm::SmallVector<int64_t> upperBounds,
  llvm::SmallVector<int64_t> steps,
  llvm::ArrayRef<std::string> keys = {},
  llvm::ArrayRef<std::string> vals = {}
);

}
}

#endif