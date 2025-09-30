#include "Dialect/Deepgen/Utils/Utils.h"

namespace mlir {
namespace deepgen{

static affine::AffineForOp
buildAffineLoopFromConstants(OpBuilder &builder, Location loc, int64_t lb,
                             int64_t ub, int64_t step,
                             affine::AffineForOp::BodyBuilderFn bodyBuilderFn) {
  return builder.create<affine::AffineForOp>(loc, lb, ub, step,
                                     /*iterArgs=*/std::nullopt, bodyBuilderFn);
}

static affine::AffineForOp
buildAffineLoopFromValues(OpBuilder &builder, Location loc, Value lb, Value ub,
                          int64_t step,
                          affine::AffineForOp::BodyBuilderFn bodyBuilderFn, const std::string& attr) {
  std::optional<int64_t> lbConst = getConstantIntValue(lb);
  std::optional<int64_t> ubConst = getConstantIntValue(ub);
  const std::string name = "type";
  if (lbConst && ubConst){
    auto ret = buildAffineLoopFromConstants(builder, loc, lbConst.value(), ubConst.value(), step, bodyBuilderFn);
    // ret->setAttr(name, builder.getStringAttr(attr));
    return ret;
  }
  else{
    auto ret = builder.create<affine::AffineForOp>(loc, lb, builder.getDimIdentityMap(), ub,
                                     builder.getDimIdentityMap(), step,
                                     /*iterArgs=*/std::nullopt, bodyBuilderFn);
    
    // ret->setAttr(name, builder.getStringAttr(attr));
    return ret;
  }
  
}

affine::AffineForOp buildAffineLoop(
    OpBuilder &builder, Location loc, Value lb, Value ub,
    int64_t step, const std::string& attr, 
    affine::AffineForOp::BodyBuilderFn bodyBuilderFn) {
  return buildAffineLoopFromValues(builder,loc,lb,ub, step, bodyBuilderFn, attr);
}

affine::AffineForOp createPipelinedForOp(OpBuilder &builder, Location loc, Value lb, Value ub,
    int64_t step, affine::AffineForOp::BodyBuilderFn bodyBuilderFn){
    return buildAffineLoop(builder,loc,lb,ub,step,"pipelined",bodyBuilderFn);
}

affine::AffineForOp createPersistentForOp(OpBuilder &builder, Location loc, Value lb, Value ub,
    int64_t step, affine::AffineForOp::BodyBuilderFn bodyBuilderFn){
    return buildAffineLoop(builder,loc,lb,ub,step,"persistent",bodyBuilderFn);
}

}  // namespace deepgen
}  // namespace mlir
