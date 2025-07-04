#include "Conversion/Lowering.h"

using namespace mlir;
namespace DeepGen {

// =====================================================================
//                  Vecotr Dialect To LLVM Dialect
// =====================================================================
// 将memref lowering到llvm上，因为 passes.h.inc中的base类没有提供可以选择indexBitWidth的options，所以自己写了一个
struct VectorToLLVMPass : public PassWrapper<VectorToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorToLLVMPass)

  VectorToLLVMPass(unsigned indexBitWidth_=32) : indexBitWidth(indexBitWidth_) {};

  unsigned indexBitWidth;

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    LowerToLLVMOptions options(&getContext());
    options.overrideIndexBitwidth(indexBitWidth);
    bool force32BitVectorIndices = false;
    if (indexBitWidth == 32) {
      force32BitVectorIndices = true;
    }

    LLVMTypeConverter converter(&getContext(), options);
    mlir::populateVectorToLLVMConversionPatterns(converter, patterns, false, force32BitVectorIndices);
    // mlir::populateVectorToLLVMMatrixConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

// =====================================================================
//               Change the size of LLVM::GlobalOp to 0
// =====================================================================
// 将globalshm的尺寸修改为0
struct SetShmSizeZeroPass : public PassWrapper<SetShmSizeZeroPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetShmSizeZeroPass)

  SetShmSizeZeroPass() {};

  void runOnOperation() override {
    auto mod = getOperation();
    std::vector<mlir::LLVM::GlobalOp> globalOps = {};
    mod.walk([&](mlir::LLVM::GlobalOp op) {
      auto type = op.getGlobalType();
      auto arrTy = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type);
      auto newType = mlir::LLVM::LLVMArrayType::get(arrTy.getElementType(),0);
      auto builder = mlir::OpBuilder(op);
      //  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type, bool isConstant, Linkage linkage, StringRef name, Attribute value, uint64_t alignment = 0, unsigned addrSpace = 0, bool dsoLocal = false, bool thread_local_ = false, SymbolRefAttr comdat = {}, ArrayRef<NamedAttribute> attrs = {});
      uint64_t align = 0;
      if(op.getAlignment()){
        align = op.getAlignment().value();
      }
      auto newOp = builder.create<mlir::LLVM::GlobalOp>(builder.getUnknownLoc(),
        newType,
        op.getConstant(),
        op.getLinkage(),
        op.getSymName(),
        op.getValueAttr(),
        align,
        op.getAddrSpace(),
        op.getDsoLocal(),
        op.getThreadLocal_()
      );
      auto useRange = op.getSymbolUses(mod);
      auto result = op.replaceAllSymbolUses(newOp.getSymNameAttr(),mod);
      assert(result.succeeded() && "setshmsizezeroPass failed");
      op.erase();
      globalOps.push_back(newOp);
    });
    if (globalOps.size()) {
      SmallVector<Operation*> gtps;
      mod.walk([&](mlir::LLVM::AddressOfOp op) {
        auto ptr = mlir::dyn_cast_if_present<mlir::LLVM::LLVMPointerType>(op.getRes().getType());
        if(ptr.getAddressSpace() == (int)MemorySpace::shared) {
          // auto parentOp = mlir::dyn_cast<LLVM::LLVMFuncOp>(op->getParentOp());
          // if (!parentOp) assert(false);
          // auto dataflowTypeAttr = mlir::dyn_cast<mlir::StringAttr>(parentOp->getAttr("func.dataflow.type"));
          // std::string type_ = dataflowTypeAttr.getValue().str();
          for (auto user : op.getResult().getUsers()) {
            if (auto getptr = mlir::dyn_cast<LLVM::GEPOp>(user)) {
              getptr.replaceAllUsesWith(op.getResult());
              gtps.push_back(user);
            }
          }
        }
      });
      for (auto gtp : gtps) gtp->erase();
    }
  }
};

// ===================================================================
//         LLVMFuncOp add attribute (nvvm.kernel, nvvm.maxnid)
// ===================================================================
// LLVMFunc 添加 nvvm.kernel 和 nvvm.maxnid 属性给func
struct LLVMFuncOpAddGPUAttrPass : public PassWrapper<LLVMFuncOpAddGPUAttrPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMFuncOpAddGPUAttrPass)
  explicit LLVMFuncOpAddGPUAttrPass(Target target_) : target(target_) {};
  private:
    Target target;
  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    OpBuilder builder(module);
    module.walk<WalkOrder::PreOrder>([&](LLVM::LLVMFuncOp funcOp) {
      if (target == Target::CUDA) {
        // funcOp->setAttr(CUDA_KERNEL, builder.getIntegerAttr(builder.getI1Type(), 1));
        setOpAttr(funcOp, CUDA_KERNEL, true);
      } else {
        // funcOp->setAttr(ROCM_KERNEL, builder.getIntegerAttr(builder.getI1Type(), 1));
        setOpAttr(funcOp, ROCM_KERNEL, true);
      }
    });
  }
};

// ===================================================================
//                 gpu index to nvvm/rocdl index 
// ===================================================================
// 将 GUP 的IdOp转成 rocdl/nvvm的IdOp，读取func的attr加到新的IdOp上
template <typename Op, typename XOp, typename YOp, typename ZOp>
struct GPUIndexIntrinsicOpLowering : public ConvertOpToLLVMPattern<Op> {
private:
  unsigned indexBitwidth;
  StringRef boundsAttrName;

public:
  explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<Op>(typeConverter),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        boundsAttrName("") {}

  explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter &typeConverter,
                                       StringRef boundsAttrName)
      :  ConvertOpToLLVMPattern<Op>(typeConverter),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        boundsAttrName(boundsAttrName) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    Operation *newOp;
    switch (op.getDimension()) {
    case gpu::Dimension::x:
      newOp = rewriter.create<XOp>(loc, IntegerType::get(context, indexBitwidth));
      break;
    case gpu::Dimension::y:
      newOp = rewriter.create<YOp>(loc, IntegerType::get(context, indexBitwidth));
      break;
    case gpu::Dimension::z:
      newOp = rewriter.create<ZOp>(loc, IntegerType::get(context, indexBitwidth));
      break;
    }

    Operation *function;
    if (auto Func = op->template getParentOfType<func::FuncOp>())
      function = Func;
    if (auto llvmFunc = op->template getParentOfType<LLVM::LLVMFuncOp>())
      function = llvmFunc;
    if (!boundsAttrName.empty() && function) {
      if (auto attr = function->template getAttrOfType<DenseI32ArrayAttr>(boundsAttrName)) {
        int32_t maximum = attr[static_cast<uint32_t>(op.getDimension())];
        std::vector<int32_t> range{0, maximum};
        setOpAttrArray(newOp, RANGE, range);
        // newOp->setAttr("range", rewriter.getDenseI32ArrayAttr({0, maximum}));
      }
    }
    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    }
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct GPUShuffleOpToROCDLLowering : public ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern<gpu::ShuffleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value initShflValue = adaptor.getValue();
    Type shflType = initShflValue.getType();
    // TODO: Add support for non 32-bit shuffle values.
    if (!shflType.isIntOrFloat() || shflType.getIntOrFloatBitWidth() != 32)
      return rewriter.notifyMatchFailure(op, "only 32-bit int/float types are supported");

    const unsigned indexBitwidth = getTypeConverter()->getIndexTypeBitwidth();

    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    Value zero_ = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value minus1 = rewriter.create<arith::ConstantIntOp>(loc, -1, 32);
    Value mbcntLo = rewriter.create<ROCDL::MbcntLoOp>(loc, int32Type, ValueRange{minus1, zero_});
    Value srcLaneId = rewriter.create<ROCDL::MbcntHiOp>(loc, int32Type, ValueRange{minus1, mbcntLo});  // tid % warpsize

    Value width = adaptor.getWidth();  // 这是自己设置的16
    Value one = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 1);
    Value negwidth = rewriter.create<LLVM::SubOp>(loc, int32Type, width, one);
    Value maskAnd = rewriter.create<LLVM::AndOp>(loc, int32Type, srcLaneId, negwidth);  // -64
    Value add = rewriter.create<LLVM::AddOp>(loc, int32Type, maskAnd, adaptor.getOffset());

    // Value width = adaptor.getWidth();  // 这是自己设置的16
    // Value zero = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 0);
    // Value negwidth = rewriter.create<LLVM::SubOp>(loc, int32Type, zero, width);
    // Value add = rewriter.create<LLVM::AddOp>(loc, int32Type, srcLaneId, width);
    // Value widthOrZeroIfOutside = rewriter.create<LLVM::AndOp>(loc, int32Type, add, negwidth);  // -64
    Value dstLane;
    // TODO: Add support for gpu::ShuffleMode::UP and gpu::ShuffleMode::DOWN.
    // TODO: Use ds_swizzle for XOR when step/offsets are constants for better
    // perf.
    switch (op.getMode()) {
    case gpu::ShuffleMode::DOWN:
      dstLane = rewriter.create<LLVM::AddOp>(loc, int32Type, srcLaneId, adaptor.getOffset());  // tid % warpsize + offset
      break;
    case gpu::ShuffleMode::XOR:
      dstLane = rewriter.create<LLVM::XOrOp>(loc, int32Type, srcLaneId, adaptor.getOffset());
      break;
    case gpu::ShuffleMode::IDX:
      dstLane = adaptor.getOffset();
      break;
    default:
      return failure();
    }
    // "tid % warpsize + offset < -64  == false"
    Value isActiveSrcLane = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, add, width);
    Value selectDstLane = rewriter.create<LLVM::SelectOp>(loc, isActiveSrcLane, srcLaneId, dstLane); // tid % warpsize + offset / tid % warpsize
    Value two = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 2);
    Value dwordAlignedDstLane = rewriter.create<LLVM::ShlOp>(loc, int32Type, selectDstLane, two);  // dstlane * 4
    if (shflType.isF32()) {
      initShflValue = rewriter.create<LLVM::BitcastOp>(loc, int32Type, initShflValue);
    }
    Value shflValue = rewriter.create<ROCDL::DsBpermuteOp>(loc, int32Type, dwordAlignedDstLane, initShflValue);
    if (shflType.isF32()) {
      shflValue = rewriter.create<LLVM::BitcastOp>(loc, shflType, shflValue);
    }
    rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
    return success();
  }
};

struct GPUShuffleOpToNVVMLowering : public ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern<gpu::ShuffleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto valueTy = adaptor.getValue().getType();
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    auto predTy = IntegerType::get(rewriter.getContext(), 1);

    // Value one = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 1);
    Value minusOne = rewriter.create<LLVM::ConstantOp>(loc, int32Type, -1);
    // Value thirtyTwo = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 32);
    // Value numLeadInactiveLane = rewriter.create<LLVM::SubOp>(loc, int32Type, thirtyTwo, adaptor.getWidth());
    // Bit mask of active lanes: `(-1) >> (32 - activeWidth)`.
    // Value activeMask = rewriter.create<LLVM::LShrOp>(loc, int32Type, minusOne, numLeadInactiveLane);
    // Value maskAndClamp;
    // if (op.getMode() == gpu::ShuffleMode::UP) {
    //   // Clamp lane: `32 - activeWidth`
    //   maskAndClamp = numLeadInactiveLane;
    // } else {
    //   // Clamp lane: `activeWidth - 1`
    //   maskAndClamp = rewriter.create<LLVM::SubOp>(loc, int32Type, adaptor.getWidth(), one);
    // }
    Value segmaskAndClamp;
    auto constOp = adaptor.getWidth().getDefiningOp<arith::ConstantOp>();
    auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue());
    auto width = intAttr.getInt();
    // llvm::outs() << "witdh: " << width;
    if (width < 32) {
      segmaskAndClamp = rewriter.create<LLVM::ConstantOp>(loc, int32Type, ((32 - width) << 8) + 31);
    } else {
      segmaskAndClamp = rewriter.create<LLVM::ConstantOp>(loc, int32Type, width - 1);;
    }
    
    bool predIsUsed = !op->getResult(1).use_empty();
    UnitAttr returnValueAndIsValidAttr = nullptr;
    Type resultTy = valueTy;
    if (predIsUsed) {
      returnValueAndIsValidAttr = rewriter.getUnitAttr();
      resultTy = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {valueTy, predTy});
    }
    NVVM::ShflKind nvvmMode;
    switch (op.getMode()) {
      case gpu::ShuffleMode::XOR:
        nvvmMode = NVVM::ShflKind::bfly;
        break;
      case gpu::ShuffleMode::UP:
        nvvmMode =  NVVM::ShflKind::up;
        break;
      case gpu::ShuffleMode::DOWN:
        nvvmMode =  NVVM::ShflKind::down;
        break;
      case gpu::ShuffleMode::IDX:
        nvvmMode =  NVVM::ShflKind::idx;
        break;
      default:
        return failure();
    }
    // Value shfl = rewriter.create<NVVM::ShflOp>(loc, resultTy, activeMask, adaptor.getValue(), adaptor.getOffset(),
        // maskAndClamp, nvvmMode, returnValueAndIsValidAttr);
    Value shfl = rewriter.create<NVVM::ShflOp>(loc, resultTy, minusOne, adaptor.getValue(), adaptor.getOffset(),
        segmaskAndClamp, nvvmMode, returnValueAndIsValidAttr);
    if (predIsUsed) {
      Value shflValue = rewriter.create<LLVM::ExtractValueOp>(loc, shfl, 0);
      Value isActiveSrcLane = rewriter.create<LLVM::ExtractValueOp>(loc, shfl, 1);
      rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
    } else {
      rewriter.replaceOp(op, {shfl, nullptr});
    }
    return success();
  }
};

// 将gpu barrier转成rocdl的barrier
struct GPUBarrierToROCDLLowering : public OpRewritePattern<gpu::BarrierOp> {
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp brOp, PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ROCDL::BarrierOp>(brOp);
    return success();
  }
};

// 将gpu barrier转成NVVM的barrier0
struct GPUBarrierToNVVMLowering : public OpRewritePattern<gpu::BarrierOp> {
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp brOp, PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<NVVM::Barrier0Op>(brOp);
    return success();
  }
};

// 将上述 3 个重写加到这个pass中
struct GPUToROCDLOrNVVMPass : public PassWrapper<GPUToROCDLOrNVVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUToROCDLOrNVVMPass)

  explicit GPUToROCDLOrNVVMPass(Target target_, unsigned indexBitwidth_) : 
                                target(target_), indexBitwidth(indexBitwidth_) {};
  private:
    Target target;
    unsigned indexBitwidth;
    StringRef amdgcnDataLayout =
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
    "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:"
    "32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:"
    "64-S32-A5-G1-ni:7:8:9";

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ROCDL::ROCDLDialect>();
    registry.insert<NVVM::NVVMDialect>();
  }
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget Codetarget(getContext());
    LowerToLLVMOptions options(&getContext());
    if (target == Target::ROCm) {
      options.dataLayout = llvm::DataLayout(amdgcnDataLayout);
    }
    options.overrideIndexBitwidth(indexBitwidth);
    LLVMTypeConverter typeConverter(&getContext(), options);

    Codetarget.addIllegalDialect<gpu::GPUDialect>();
    Codetarget.addLegalDialect<LLVM::LLVMDialect, ROCDL::ROCDLDialect, NVVM::NVVMDialect>();

    if (target == Target::ROCm) {
      Codetarget.addLegalDialect<arith::ArithDialect>();
      // populateGpuToROCDLConversionPatterns(typeConverter, patterns, gpu::amd::Runtime::HIP);
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, ROCDL::BlockIdXOp, 
                                               ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>>(typeConverter, FUNC_GRID_DIM);
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                               ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>>(typeConverter, FUNC_BLOCK_DIM);
      patterns.add<GPUShuffleOpToROCDLLowering>(typeConverter);
      patterns.add<GPUBarrierToROCDLLowering>(&getContext());
      populateMathToROCDLConversionPatterns(typeConverter, patterns);  // exp仅支持fp64/16
    } else if (target == Target::CUDA) {
      // populateGpuToNVVMConversionPatterns(typeConverter, patterns, 10);
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, NVVM::BlockIdXOp, 
                                               NVVM::BlockIdYOp, NVVM::BlockIdZOp>>(typeConverter, FUNC_GRID_DIM);
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp,
                                               NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>>(typeConverter, FUNC_BLOCK_DIM);
      patterns.add<GPUShuffleOpToNVVMLowering>(typeConverter);
      patterns.add<GPUBarrierToNVVMLowering>(&getContext());
      populateLibDeviceConversionPatterns(typeConverter, patterns, /*benefit*/10);  // 大部分只支持fp32/fp16
    }

    if (failed(applyPartialConversion(getOperation(), Codetarget, std::move(patterns)))){
      return signalPassFailure();
    }
  }
};


// =============================================================================
std::unique_ptr<OperationPass<ModuleOp>> createVectorToLLVMPass(unsigned indexBitWidth) {
  return std::make_unique<VectorToLLVMPass>(indexBitWidth);
}

std::unique_ptr<OperationPass<ModuleOp>> createGlobalShmSetZeroPass() {
  return std::make_unique<SetShmSizeZeroPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createLLVMFuncOpAddGPUAttrPass(Target target) {
  return std::make_unique<LLVMFuncOpAddGPUAttrPass>(target);
}

std::unique_ptr<OperationPass<ModuleOp>> createGPUToROCDLOrNVVMPass(Target target, unsigned indexBitwidth) {
  return std::make_unique<GPUToROCDLOrNVVMPass>(target, indexBitwidth);
}

}