#include "Transforms/Transform.h"

using namespace mlir;
namespace DeepGen {

// ===================================================================
//                  affine parallel to gpu index        
// ===================================================================
// 将affine的parallelOp 转成Gpu的block/threadIdOp表示，func添加grid/block size作为属性
struct ParallelToGPULowering : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineParallelOp parallelOp, PatternRewriter &rewriter) const final {
    constexpr gpu::Dimension dims[] = {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
    // 替换 parallelOp 为 gpu::BlockIdOp
    std::vector<int32_t> ubs;
    auto gpuidx = getOpAttr<std::string>(parallelOp, GPU_INDEX);
    auto upperBounds = parallelOp.getUpperBoundsMap().getConstantResults();
    SmallVector<Value, 3> ids;
    for (unsigned i = 0; i < parallelOp.getNumDims(); ++i) {
      if (gpuidx == THREADIDX) {
        auto threadId = rewriter.create<gpu::ThreadIdOp>(parallelOp.getLoc(), dims[i]);
        ids.push_back(threadId);
      } else {
        auto blockId = rewriter.create<gpu::BlockIdOp>(parallelOp.getLoc(), dims[i]);
        ids.push_back(blockId);
      }
      ubs.push_back(static_cast<int32_t>(upperBounds[i]));
    }
    // 将func设置block和thread的上界属性
    func::FuncOp funcOp = nullptr;
    Operation* parentOp = parallelOp->getParentOp();
    while (parentOp) {
      if (funcOp = mlir::dyn_cast<func::FuncOp>(parentOp)) { break; }
      parentOp = parentOp->getParentOp();
    }
    if (funcOp == nullptr) {
      llvm::errs() << "The ParentOp of scf::ParallelOp must is FuncOp!\n";
      assert(false);
    }
    if (gpuidx == THREADIDX) {
      setOpAttrArray(funcOp, FUNC_GRID_DIM, ubs);
    } else {
      setOpAttrArray(funcOp, FUNC_BLOCK_DIM, ubs);
    }
    
    // 替换使用循环变量的操作
    auto ivs = parallelOp.getIVs();
    for (unsigned i = 0; i < ivs.size(); ++i) {
      ivs[i].replaceAllUsesWith(ids[i]);
    }

    // 内层操作移出内层 p  collect op
    SmallVector<Operation *, 4> opsToMove;
    for (Operation &op : parallelOp.getBody()->getOperations()) {
      if (!dyn_cast<affine::AffineYieldOp>(op)) {
        opsToMove.push_back(&op);
      }
    }
    // 内层操作移出内层 p 
    Operation *tempOp = ids.back().getDefiningOp();
    for (Operation *op : opsToMove) {
      op->moveAfter(tempOp);
      tempOp = op;
    }
    rewriter.eraseOp(parallelOp);
    return success();
  }
};

// 将affine的parallelOp 转成Gpu的block/threadIdOp表示的pass
struct ParallelToGPUPass : public PassWrapper<ParallelToGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelToGPUPass)
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    // target.addIllegalOp<gpu::BlockIdOp, gpu::ThreadIdOp>();
    target.addIllegalOp<affine::AffineParallelOp>();
    target.addLegalDialect<gpu::GPUDialect>();

    patterns.add<ParallelToGPULowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))){
      return signalPassFailure();
    }

  }
};

// =====================================================================
//    Multiple allocOp/allocaOp are merged into one allocOp/allocaOp
// =====================================================================
// 将alloc和alloca操作合并，生成一个memref
struct CombineMemrefPass : public PassWrapper<CombineMemrefPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CombineMemrefPass)

  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    for (Operation &op : module.getBody()->getOperations()) {
      if (auto funcOp = mlir::dyn_cast<func::FuncOp>(&op)) {
        combineAllocOrAllocaOp<memref::AllocOp>(funcOp);
        // combineAllocOrAllocaOp<memref::AllocaOp>(funcOp);
      }
    }
  }

  template<typename AllocOrAllocaOp>
  void combineAllocOrAllocaOp(func::FuncOp &funcOp) {
    int64_t memSize = 0;
    llvm::DenseMap<AllocOrAllocaOp, int64_t> start_idxs;
    AllocOrAllocaOp firstOp = nullptr;
    MemRefType type;
    // 记录 allocop的mem尺寸起始位置
    funcOp.walk<WalkOrder::PreOrder>([&](AllocOrAllocaOp allocOp) {
      if (memSize == 0) {
        firstOp = allocOp;
      }
      start_idxs.try_emplace(allocOp, memSize);
      type = mlir::dyn_cast<MemRefType>(allocOp.getResult().getType());
      int64_t offset = 1;
      for (auto shape : type.getShape()) {
        offset *= shape;
      }
      memSize += offset;
    });
    if (memSize) {
      OpBuilder b(firstOp);
      auto memSpace = static_cast<MemorySpace>(type.getMemorySpaceAsInt());
      auto new_buf = MemUtil::createAllocOp(b, {memSize}, type.getElementType(), memSpace, BUF_ALIGN_16B);
      for (const auto& pair : start_idxs) {
        Value buf = pair.first->getResult(0);
        auto t = mlir::dyn_cast<MemRefType>(buf.getType());
        auto shape = t.getShape();  // 每次alloc的shape
        SmallVector<Operation *> users;
        for (auto user : buf.getUsers()) {  // collect users
          users.push_back(user);
        }
        // auto users = result.getUsers();
        for (auto user : users) {
          if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(user)) {
            auto map = ExprUtil::linearizeIndex(b, loadOp.getAffineMap(), pair.second, shape);
            MemUtil::amendLoadOrStoreOp(loadOp, map, loadOp.getMapOperands(), new_buf);

          } else if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(user)) {
            auto map = ExprUtil::linearizeIndex(b, storeOp.getAffineMap(), pair.second, shape);
            MemUtil::amendLoadOrStoreOp(storeOp, map, storeOp.getMapOperands(), new_buf);

          } else if (auto vectorLoadOp = mlir::dyn_cast<affine::AffineVectorLoadOp>(user)) {
            auto map = ExprUtil::linearizeIndex(b, vectorLoadOp.getAffineMap(), pair.second, shape);
            MemUtil::amendLoadOrStoreOp(vectorLoadOp, map, vectorLoadOp.getMapOperands(), new_buf);

          } else if (auto vectorStoreOp = mlir::dyn_cast<affine::AffineVectorStoreOp>(user)) {
            auto map = ExprUtil::linearizeIndex(b, vectorStoreOp.getAffineMap(), pair.second, shape);
            MemUtil::amendLoadOrStoreOp(vectorStoreOp, map, vectorStoreOp.getMapOperands(), new_buf);
          }
        }
        pair.first->erase();
      }
    }
  }
};

// =====================================================================
//              replace memref::AllocOp to LLVM::GlobalOp
// =====================================================================
// replace alloc<shared> to getGlobalOp
struct ReplaceAllocOpToGetGlobalOp : public PassWrapper<ReplaceAllocOpToGetGlobalOp, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceAllocOpToGetGlobalOp)
   ReplaceAllocOpToGetGlobalOp() = default;
   void runOnOperation() override {
     auto module = getOperation();
    int i = 0;
     std::vector<MemRefType> memTypesToAdd {};
     module.walk<WalkOrder::PreOrder>([&](memref::AllocOp allocOp) {
      auto memspace = allocOp.getResult().getType().getMemorySpaceAsInt();
      if(memspace == (int)MemorySpace::shared){
        OpBuilder builder(allocOp);
        OpBuilder b(module);
        b.setInsertionPointToStart(module.getBody());
        std::string buf_name = "smem_" + std::to_string(i);
        auto globalOp = b.create<memref::GlobalOp>(
          b.getUnknownLoc(),
          buf_name,
          b.getStringAttr("public"),
          allocOp.getResult().getType(),
          Attribute(),
          false,
          IntegerAttr()
          );
        globalOp.setAlignment(BUF_ALIGN_16B);  // 对齐到 4*sizeof(float) 字节，以增加访问效率
        auto newop = builder.create<memref::GetGlobalOp>(
          builder.getUnknownLoc(),allocOp.getResult().getType(), buf_name);
        allocOp.getResult().replaceAllUsesWith(newop);
        allocOp.erase();
        ++i;
      }
     });
   }
}; 

// ===================================================================
//                  amend memerf alloca Addrspace       
// ===================================================================
// 将memref.alloca的地址空间进行修改，local=0为cuda/loacl=5为rocm
struct AmendAllocaOpAddrSpace : public OpRewritePattern<memref::AllocaOp> {
  AmendAllocaOpAddrSpace(MLIRContext *context, Target target)
    : OpRewritePattern(context), target(target) {}

  LogicalResult matchAndRewrite(memref::AllocaOp allocaOp, PatternRewriter &rewriter) const override {
    MemRefType originalType = allocaOp.getType();
    int requiredSpace = (target == Target::ROCm) ? 5 : 0;
    if (static_cast<int>(originalType.getMemorySpaceAsInt()) == requiredSpace) {
      return failure();
    }
    MLIRContext *ctx = allocaOp.getContext();
    Attribute memorySpaceAttr = IntegerAttr::get(IntegerType::get(ctx, 64), requiredSpace);
    MemRefType newType = MemRefType::get(originalType.getShape(), originalType.getElementType(), 
                                         originalType.getLayout(),memorySpaceAttr);

    rewriter.setInsertionPoint(allocaOp);
    auto newAlloca = rewriter.create<memref::AllocaOp>(allocaOp.getLoc(), newType);
    rewriter.replaceOp(allocaOp, newAlloca.getResult());
    return success();
  }
  private:
    Target target;
};

struct AmendAllocaOpAddrSpacePass : public PassWrapper<AmendAllocaOpAddrSpacePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AmendAllocaOpAddrSpacePass)

  AmendAllocaOpAddrSpacePass(Target target) : target(target) {}
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AmendAllocaOpAddrSpace>(&getContext(), target);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
  private:
    Target target;
};

// ====================================================================
//                         Affine ForOp Unroll
// ====================================================================
// affine 循环展开
struct AffineUnrollPass : public PassWrapper<AffineUnrollPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineUnrollPass)

  void runOnOperation() override {
    getOperation().walk([&] (affine::AffineForOp forOp){

      if (auto unrollName = forOp->getAttr(AFFINE_LOOP)) {
        auto unrollAttr = mlir::dyn_cast<mlir::StringAttr>(unrollName);
        if (unrollAttr.getValue().str() == "unroll") {
          int32_t unroll_num = getOpAttr<int32_t>(forOp, AFFINE_UNROLL_NUM);
          int64_t ub = forOp.getConstantUpperBound();
          int64_t lb = forOp.getConstantLowerBound();
          int64_t step = forOp.getStep().getLimitedValue();
          int64_t loopNum =  (ub - lb) / step;
          if (loopNum > unroll_num) {
            auto ret = mlir::affine::loopUnrollJamByFactor(forOp, unroll_num);
            if(failed(ret)){
              return signalPassFailure();
            }
          } else {
            auto ret = mlir::affine::loopUnrollFull(forOp);
            if(failed(ret)){
              return signalPassFailure();
            }
          }
        }
      }
    });
  }
};

// =============================================================================
std::unique_ptr<OperationPass<ModuleOp>> createParallelToGPUPass() {
  return std::make_unique<ParallelToGPUPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createCombineMemrefPass() {
  return std::make_unique<CombineMemrefPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> ReplaceAllocToGetglobalPass() {
  return std::make_unique<ReplaceAllocOpToGetGlobalOp>();
}

std::unique_ptr<OperationPass<ModuleOp>> createAmendAllocaOpAddrSpacePass(Target target) {
  return std::make_unique<AmendAllocaOpAddrSpacePass>(target);
}

std::unique_ptr<OperationPass<ModuleOp>> createAffineUnrollPass() {
  return std::make_unique<AffineUnrollPass>();
}
  
}