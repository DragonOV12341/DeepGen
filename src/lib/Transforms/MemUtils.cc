
#include "Transforms/MemUtils.h"

using namespace mlir;
namespace DeepGen {
namespace MemUtil {

Value createAllocOp(OpBuilder builder, const std::vector<int64_t>& shape, Type dtype, MemorySpace space, int align, std::string bufDesc) {
  // 创建allocaOp
  Value allocVal;
  Location loc = builder.getUnknownLoc();
  auto bufferType = MemRefType::get(shape, dtype, {}, static_cast<int>(space));
  if (space == MemorySpace::local) {
    auto reg = builder.create<memref::AllocaOp>(loc, bufferType);
    if (align != 0) {
      reg.setAlignment(align);
    }
    if (!bufDesc.empty()) {
      setOpAttr(reg, BUF_DESC, bufDesc);
    }
    allocVal = reg.getResult();
  } else {
    auto sm = builder.create<memref::AllocOp>(loc, bufferType);
    if (align != 0) {
      sm.setAlignment(align);
    }
    if (!bufDesc.empty()) {
      setOpAttr(sm, BUF_DESC, bufDesc);
    }
    allocVal = sm.getResult();
  }
  return allocVal;
}

}
}