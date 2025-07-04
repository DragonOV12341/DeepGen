
#include "Transforms/LoopUtils.h"

namespace DeepGen {
namespace LoopUtil {

NestedLoopResult createNestedLoops(
  mlir::OpBuilder& builder,
  llvm::SmallVector<int64_t> lowerBounds,
  llvm::SmallVector<int64_t> upperBounds,
  llvm::SmallVector<int64_t> steps,
  llvm::ArrayRef<std::string> keys,
  llvm::ArrayRef<std::string> vals
) {
  // 根据loop的信息创建嵌套的loops
  if (!lowerBounds.size()) return NestedLoopResult({{}, {}});
  llvm::SmallVector<int64_t> outer{lowerBounds[0], upperBounds[0], steps[0]};
  lowerBounds.erase(lowerBounds.begin());
  upperBounds.erase(upperBounds.begin());
  steps.erase(steps.begin());
  // create for
  llvm::SmallVector<mlir::Value> allIvs;
  auto loopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv_, mlir::ValueRange iterArgs) {
    allIvs.push_back(iv_);
    mlir::affine::buildAffineLoopNest(b, loc, lowerBounds, upperBounds, steps,
      [&](mlir::OpBuilder &bb, mlir::Location loc, mlir::ValueRange ivs) {
        for (auto iv : ivs) { allIvs.push_back(iv); }
      });
    b.create<mlir::affine::AffineYieldOp>(loc);
  };
  mlir::Location loc_ = builder.getUnknownLoc();
  auto outerLoop = builder.create<mlir::affine::AffineForOp>(loc_, outer[0], outer[1], outer[2], mlir::ValueRange({}), loopBody);
  // collect for
  int index = 0;
  llvm::SmallVector<mlir::affine::AffineForOp> loops;
  outerLoop.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp fop) {
    if (vals.size() > index) {
      setOpAttr(fop, keys[index], vals[index]);
      index++;
    }
    loops.push_back(fop);
  });
  builder.setInsertionPointToStart(loops[loops.size()-1].getBody());
  return NestedLoopResult({loops, allIvs});
}

}
}