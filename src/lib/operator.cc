#include "operator.h"

namespace DeepGen {

// ====================================== dot kernel  ======================================
bool Dot::verify(ArgTensor A, ArgTensor B, ArgTensor C) {
  // 默认 A[B, M, K] - B[B, K, N] - C[B, M, N]
  if (A.rank == B.rank && A.rank == C.rank) {
    // dtype
    if (A.dtype == B.dtype && A.dtype == C.dtype) {
      int32_t rank = A.rank;
      // transpose
      if (A.istran && B.istran && A.shape[rank-2] == B.shape[rank-1] && // A-k == B-k
        A.shape[rank-1] == C.shape[rank-2] && C.shape[rank-1] == B.shape[rank-2]) {  // A-m == C-m & B-n == C-n
        return true;
      } else if (!A.istran && B.istran && A.shape[rank-1] == B.shape[rank-1] &&  // A-k == B-k
        A.shape[rank-2] == C.shape[rank-2] && C.shape[rank-1] == B.shape[rank-1]) {  // A-m == C-m & B-n == C-n
        return true;
      } else if (A.istran && !B.istran && A.shape[rank-2] == B.shape[rank-2] &&  // A-k == B-k
        A.shape[rank-1] == C.shape[rank-2] && C.shape[rank-1] == B.shape[rank-1]) {  // A-m == C-m & B-n == C-n
        return true;
      } else if (!A.istran && !B.istran && A.shape[rank-1] == B.shape[rank-2] &&  // A-k == B-k
        A.shape[rank-2] == C.shape[rank-2] && C.shape[rank-1] == B.shape[rank-1]) {  // A-m == C-m & B-n == C-n
        return true;
      }
      llvm::errs() << "The k/m/n dimensions are not equal.\n";
      return false;
    }
    llvm::errs() << "The dtype of Tensor is ont equal.\n";
    return false;
  } 
  llvm::errs() << "The rank of Tensor is ont equal.\n";
  return false;
}

mlir::func::FuncOp Dot::buildKernel(mlir::ModuleOp module, int kernel_num, ArgTensor A, ArgTensor B, ArgTensor C) {
  // create dot cpu kernel
  if (!verify(A, B, C))
    assert(false && "Dot verify failed.");
  std::string kernel_name = KERNEL_DOT + std::string("_") + std::to_string(kernel_num);
  mlir::OpBuilder builder(module);
  builder.setInsertionPointToEnd(module.getBody());
  KernelInfo info = Operator::parseKernelInfo(builder, kernel_name, KERNEL_DOT, {A, B, C});
  // create func op
  mlir::func::FuncOp funcOp = Operator::buildFuncOp(builder, info);
  mlir::ValueRange operands = funcOp.getArguments();
  // nest batch for
  int64_t k = A.shape[A.rank-1];
  if (A.istran) k = A.shape[A.rank-2];
  llvm::SmallVector<int64_t> batch(A.shape.begin(), A.shape.end()-2);
  llvm::SmallVector<int64_t> mn(C.shape.end()-2, C.shape.end()), lbs(batch.size(), 0), steps(batch.size(), 1);
  llvm::SmallVector<std::string> vals(batch.size(), BATCH), keys(batch.size(), FOR_PARA_DESC);
  auto result = LoopUtil::createNestedLoops(builder, lbs, batch, steps, keys, vals);
  // dot nest for
  mlir::SmallVector<int64_t> lbs_(mn.size(), 0), steps_(mn.size(), 1);
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lbs_, mn, steps_,
    [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange ivs) {
      mlir::Value row = ivs[0], col = ivs[1];
      auto zero = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(getDType(b, C.dtype), 0));

      auto kLoopBody = [&](mlir::OpBuilder &bb, mlir::Location loc_, mlir::Value kiv, mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);

        auto indexA = Operator::getIndexes(result.ivs, {row, kiv}, A.istran);
        auto indexB = Operator::getIndexes(result.ivs, {kiv, col}, B.istran);

        auto ld_a = bb.create<mlir::affine::AffineLoadOp>(loc_, /*A*/operands[0], mlir::ValueRange(indexA));
        auto ld_b = bb.create<mlir::affine::AffineLoadOp>(loc_, /*B*/operands[1], mlir::ValueRange(indexB));
        auto mul = bb.create<mlir::arith::MulFOp>(loc_, ld_a, ld_b);
        auto add = bb.create<mlir::arith::AddFOp>(loc_, mul, iterArgs[0]);
        bb.create<mlir::affine::AffineYieldOp>(loc_, add.getResult());
      };
      auto forK = b.create<mlir::affine::AffineForOp>(loc, /*lb*/0, k, 1, mlir::ValueRange({zero.getResult()}), kLoopBody);
      setOpAttr(forK, FOR_PARA_DESC, std::string(REDUCE));
      auto indexC = Operator::getIndexes(result.ivs, {row, col}, false);
      b.create<mlir::affine::AffineStoreOp>(loc, forK.getResult(0), /*C*/operands[2], mlir::ValueRange(indexC));
  });
  // para attr
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
    if (!forOp->getAttr(FOR_PARA_DESC)) 
      setOpAttr(forOp, FOR_PARA_DESC, std::string(PARALLEL));
  });
  return funcOp;
}
// =========================================================================================

// ==================================== element kernel  ====================================
std::map<ElementWiseMode, std::string> ElementWise::mode_name = {
    {ElementWiseMode::Exp, "Exp"},
    {ElementWiseMode::Exp2, "Exp2"},
};

bool ElementWise::verify(ArgTensor input, ArgTensor output) {
  // verify element wise kernel
  if (input.rank == output.rank && input.dtype == output.dtype) {
    int32_t rank = input.rank;
    if (input.istran && input.shape[rank-1] == output.shape[rank-2] && // input-row == output-row
      input.shape[rank-2] == output.shape[rank-1]) { // input-col == output-col
      return true;
    } else if (!input.istran && input.shape[rank-2] == output.shape[rank-2] &&  // input-row == output-row
      input.shape[rank-1] == output.shape[rank-1]) // input-col == output-col
      return true;
  }
  llvm::errs() << "The dtype/rank/row/col of Tensor is ont equal.\n";
  return false;
}

mlir::Value ElementWise::createModeOp(mlir::OpBuilder b, mlir::Value operand, ElementWiseMode mode) {
  // create mode op of element wise
  mlir::Value st;
  switch (mode) {
  case ElementWiseMode::Exp:
    st = b.create<mlir::math::ExpOp>(b.getUnknownLoc(), operand);
    break;
  case ElementWiseMode::Exp2:
    st = b.create<mlir::math::Exp2Op>(b.getUnknownLoc(), operand);
    break;
  default:
    assert(false && "mode op failed");
    break;
  }
  return st;
}

mlir::func::FuncOp ElementWise::buildKernel(mlir::ModuleOp module, int kernel_num, ArgTensor input, ArgTensor output, ElementWiseMode mode) {
  // creaet element wise cpu kernel
  if (!verify(input, output)) 
    assert(false && "Element Wise verify failed.");
  std::string kernel_name = KERNEL_ELEMENTWISE + std::string("_") + mode_name[mode] + std::string("_") + std::to_string(kernel_num);
  mlir::OpBuilder builder(module);
  builder.setInsertionPointToEnd(module.getBody());
  KernelInfo info = Operator::parseKernelInfo(builder, kernel_name, KERNEL_ELEMENTWISE, {input, output});
  // create func op
  mlir::func::FuncOp funcOp = Operator::buildFuncOp(builder, info);
  mlir::ValueRange operands = funcOp.getArguments();
  // nest batch for
  int64_t r = output.shape[output.rank-2], c = output.shape[output.rank-1];
  llvm::SmallVector<int64_t> batch(input.shape.begin(), input.shape.end()-2);
  llvm::SmallVector<int64_t> lbs(batch.size(), 0), steps(batch.size(), 1);
  llvm::SmallVector<std::string> keys(batch.size(), FOR_PARA_DESC), vals(batch.size(), BATCH);
  auto result = LoopUtil::createNestedLoops(builder, lbs, batch, steps, keys, vals);
  // dot nest for
  mlir::SmallVector<int64_t> lbs_(2, 0), steps_(2, 1), ubs{r, c};
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lbs_, ubs, steps_,
    [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange ivs) {
      mlir::Value row = ivs[0], col = ivs[1];
      auto index1 = Operator::getIndexes(result.ivs, {row, col}, input.istran);
      auto ld = b.create<mlir::affine::AffineLoadOp>(loc, /*input*/operands[0], mlir::ValueRange(index1));
      auto st = createModeOp(b, ld, mode);
      auto index2 = Operator::getIndexes(result.ivs, {row, col}, false);
      b.create<mlir::affine::AffineStoreOp>(loc, st, /*output*/operands[1], mlir::ValueRange(index2));
  });
  // para attr
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
    if (!forOp->getAttr(FOR_PARA_DESC)) 
      setOpAttr(forOp, FOR_PARA_DESC, std::string(PARALLEL));
  });
  return funcOp;
}
// =========================================================================================

mlir::ModuleOp KernelGenerator::loadMLIRFile(const std::string& filePath) {
  // read mlir file
  mlir::OwningOpRef<mlir::ModuleOp> mod = parseSourceFile<mlir::ModuleOp>(filePath, context.get());
  return *mod;
}

}