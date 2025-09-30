#ifndef Deepgen_utils_h
#define Deepgen_utils_h
#include "Dialect/Deepgen/IR/Dialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mlir {
namespace deepgen{


affine::AffineForOp createPipelinedForOp(OpBuilder &builder, Location loc, Value lb, Value ub,
    int64_t step, affine::AffineForOp::BodyBuilderFn bodyBuilderFn = nullptr);
affine::AffineForOp createPersistentForOp(OpBuilder &builder, Location loc, Value lb, Value ub,
    int64_t step, affine::AffineForOp::BodyBuilderFn bodyBuilderFn = nullptr);

}}

#endif // Deepgen_utils_h