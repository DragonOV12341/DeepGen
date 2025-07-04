#pragma once
#ifndef _MemUtils_h_
#define _MemUtils_h_

#include "Commons/utils.h"

using namespace mlir;
namespace DeepGen {
namespace MemUtil {

template<typename LoadOrStoreOp>
void amendLoadOrStoreOp(LoadOrStoreOp &op, AffineMap map, llvm::SmallVector<Value> operands, Value buf) {
  // 修改 load 或 store op 的 map/operands/buf
  mlir::OpBuilder b(op);
  if constexpr (std::is_same_v<std::decay_t<LoadOrStoreOp>, affine::AffineLoadOp>) {
    auto newLoadOp = b.create<affine::AffineLoadOp>(op.getLoc(), buf, map, operands);
    op.getResult().replaceAllUsesWith(newLoadOp.getResult());
  
  } else if constexpr (std::is_same_v<std::decay_t<LoadOrStoreOp>, affine::AffineStoreOp>) {
    b.create<affine::AffineStoreOp>(op.getLoc(), op.getValue(), buf, map, operands);

  } else if constexpr (std::is_same_v<std::decay_t<LoadOrStoreOp>, affine::AffineVectorLoadOp>) {
    auto newVectorLoadOp = b.create<affine::AffineVectorLoadOp>(op.getLoc(), op.getVectorType(), buf, map, operands);
    op.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());

  } else if constexpr (std::is_same_v<std::decay_t<LoadOrStoreOp>, affine::AffineVectorStoreOp>) {
    b.create<affine::AffineVectorStoreOp>(op.getLoc(), op.getValue(), buf, map, operands);
  }
  op.erase();
}

Value createAllocOp(OpBuilder builder, const std::vector<int64_t>& shape, Type dtype, MemorySpace space, int align, std::string bufDesc="");

}
}

#endif