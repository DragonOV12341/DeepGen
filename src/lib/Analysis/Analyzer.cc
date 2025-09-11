#include "Analysis/Analyzer.h"


namespace DeepGen {
namespace Analyzer {


int getThreadsPerCTA(mlir::ModuleOp module) {
  // 根据func属性获取cta的thread个数
  int threadNum = 1;
  for (auto &op : module.getBody()->getOperations()) {
    if (auto funcOp = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(op)) {
      if (!funcOp->hasAttr(FUNC_KERNEL_TYPE)) continue;
      auto blockDims = funcOp->getAttrOfType<mlir::DenseI32ArrayAttr>(FUNC_BLOCK_DIM);
      for (size_t i=0; i<blockDims.size(); i++) {
        threadNum *= blockDims[i];
      }
      return threadNum;
    }
  }
  return threadNum;
}

llvm::SmallVector<mlir::Operation*> 
getOperationByAttr(mlir::Operation *outer_op, const std::string& key, const std::string& val) {
  // 根据属性获取operation
  llvm::SmallVector<mlir::Operation*> ops;
  outer_op->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *inner_op) {
    if (getOpAttr<std::string>(inner_op, key) == val) {
      ops.push_back(inner_op);
    }
  });
  return ops;
}

}
}