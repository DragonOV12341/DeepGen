
#include "Transforms/ExprUtils.h"

using namespace mlir;
namespace DeepGen {
namespace ExprUtil {

AffineMap linearizeIndex(OpBuilder builder, AffineMap map, int64_t start_idx, const std::vector<int64_t>& shape) {
  // index [z, y, x]  shape {b, m, n}  start_idx  => newmap {start_idx + z * (m * n) + y * n + x}
  auto oldExprs = map.getResults();
  AffineExpr expr = builder.getAffineConstantExpr(0);
  for (size_t i=0; i<oldExprs.size(); i++) {
    int stride = 1;
    for (size_t j=i+1; j<shape.size(); j++) {
      stride *= shape[j];
    }
    expr = expr + oldExprs[i] * stride;
  }
  expr = start_idx + expr;
  return AffineMap::get(map.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), builder.getContext());
}

}
}