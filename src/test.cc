#include "compiler.h"
#include "operator.h"
#include "Python.h"

using namespace DeepGen;

int test() {
  KernelGenerator generator;        // 没有外部导入module
  DGCompiler compiler(Target::CUDA, "90");
  // create dot kernel
  ArgTensor A{{1, 32, 1024, 128}, 4, DType::FLOAT16, ArgType::Input, false};
  ArgTensor B{{1, 32, 128, 1024}, 4, DType::FLOAT16, ArgType::Input, false};
  ArgTensor C{{1, 32, 1024, 1024}, 4, DType::FLOAT16, ArgType::Output, false};
  auto dot0 = generator.create<Dot>(A, B, C);  // funcOp
  // LOG_DEBUG("============ CPU DOT0\n", dot0);
  // // create elementwise kernel
  // ArgTensor input{{1, 32, 1024, 1024}, DType::FLOAT16, ArgType::Input, 4, false};
  // ArgTensor output{{1, 32, 1024, 1024}, DType::FLOAT16, ArgType::Output, 4, false};
  // auto exp0 = generator.create<ElementWise>(input, output, ElementWiseMode::Exp);  // funcOp
  // LOG_DEBUG("============ CPU EXP0\n", exp0);
  // get mlir module
  auto module = generator.getModule();  // module
  LOG_DEBUG("============ CPU MODULE\n", module);
  Config tile_cfg = {{"BLOCK_TILE_Y", 128}, {"BLOCK_TILE_X", 128}};
  // compiler.mapping(module, );
  return 0;
}

std::string nvgpu_tma_test() {
  KernelGenerator generator;        // 没有外部导入module
  DGCompiler compiler(Target::CUDA, "90");

  auto module = generator.getModule();
  mlir::OpBuilder builder(module);
  builder.setInsertionPointToEnd(module.getBody());

  // TensorMapDescriptorType A
  mlir::MLIRContext *context = module.getContext();
  mlir::Type elem_type = getDType(builder, DType::FLOAT16);
  auto AmemType = getMemType({1024, 1024}, elem_type, MemorySpace::global);
  auto AMapDesc = mlir::nvgpu::TensorMapDescriptorType::get(context, AmemType, 
                                    mlir::nvgpu::TensorMapSwizzleKind::SWIZZLE_128B, 
                                    mlir::nvgpu::TensorMapL2PromoKind::L2PROMO_256B,
                                    mlir::nvgpu::TensorMapOOBKind::OOB_NAN,
                                    mlir::nvgpu::TensorMapInterleaveKind::INTERLEAVE_NONE);
  // TensorMapDescriptorType B
  auto BmemType = getMemType({1024, 1024}, elem_type, MemorySpace::global);
  auto BMapDesc = mlir::nvgpu::TensorMapDescriptorType::get(context, BmemType, 
                                                            mlir::nvgpu::TensorMapSwizzleKind::SWIZZLE_128B, 
                                                            mlir::nvgpu::TensorMapL2PromoKind::L2PROMO_256B,
                                                            mlir::nvgpu::TensorMapOOBKind::OOB_NAN,
                                                            mlir::nvgpu::TensorMapInterleaveKind::INTERLEAVE_NONE);
  // std::vector<mlir::Type> funcType{AmemType, BmemType};
  std::vector<mlir::Type> funcType{AMapDesc, BMapDesc};
  mlir::Location loc = builder.getUnknownLoc();
  auto functionType = builder.getFunctionType(mlir::TypeRange(funcType), mlir::TypeRange({}));
  auto funcOp = builder.create<mlir::func::FuncOp>(loc, llvm::StringRef("tma_test_kernel"), functionType);
  auto& region = funcOp->getRegion(0);
  if (!region.hasOneBlock()) {
    region.emplaceBlock();
  }
  auto& body =  funcOp.front(); //? region.front()  : ;
  llvm::SmallVector<mlir::Location> locs(funcType.size(), loc);
  body.addArguments(funcType, locs);
  auto& entryBlock = funcOp.front();
  builder.setInsertionPointToStart(&entryBlock);
  builder.create<mlir::func::ReturnOp>(loc);
  builder.setInsertionPointToStart(&entryBlock);
  mlir::ValueRange operands = funcOp.getArguments();

  // func body
  // constant
  auto idx1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto idx0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto idx4 = builder.create<mlir::arith::ConstantIndexOp>(loc, 4);
  auto mask = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getIntegerType(32), builder.getI32IntegerAttr(0xffffffff));
  auto true_ = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(builder.getI1Type(), 1));
  // blockidx/threadidx
  auto bid = builder.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);
  auto tid = builder.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  // create if
  mlir::AffineExpr expr = builder.getAffineDimExpr(0);
  auto set_ = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr - 128}), llvm::ArrayRef<bool>({true}));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(loc, set_, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp.getThenBlock());
  // tensormap prefetch
  builder.create<mlir::nvgpu::TmaPrefetchOp>(loc, operands[0], true_);
  builder.create<mlir::nvgpu::TmaPrefetchOp>(loc, operands[1], true_);
  builder.setInsertionPointAfter(ifOp);
  // warp sync
  builder.create<mlir::NVVM::SyncWarpOp>(loc, mask);

  // create shared memroy
  auto smA = MemUtil::createAllocOp(builder, {128, 128}, elem_type, MemorySpace::shared, 1024);
  auto smB = MemUtil::createAllocOp(builder, {128, 128}, elem_type, MemorySpace::shared, 1024);

  // create mbarrizer
  auto sm = gpu::AddressSpaceAttr::get(context, gpu::GPUDialect::getWorkgroupAddressSpace());
  auto mbr_type = mlir::nvgpu::MBarrierGroupType::get(context, sm, 1);
  auto empty_mbr = builder.create<mlir::nvgpu::MBarrierCreateOp>(loc, mbr_type);
  auto full_mbr = builder.create<mlir::nvgpu::MBarrierCreateOp>(loc, mbr_type);
  // init mbarrizer
  auto ifOp1 = builder.create<mlir::affine::AffineIfOp>(loc, set_, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp1.getThenBlock());
  auto init_full_br = builder.create<mlir::nvgpu::MBarrierInitOp>(loc, full_mbr, idx1, idx0, true_);
  auto init_empty_br = builder.create<mlir::nvgpu::MBarrierInitOp>(loc, empty_mbr, idx4, idx0, true_);
  // fence.proxy.async.shared::cta;
  auto proxy_kind = mlir::NVVM::ProxyKindAttr::get(context, mlir::NVVM::ProxyKind::async_shared);
  auto sm_space = mlir::NVVM::SharedSpaceAttr::get(context, mlir::NVVM::SharedSpace::shared_cta);
  builder.create<mlir::NVVM::FenceProxyOp>(loc, proxy_kind, sm_space);
  builder.setInsertionPointAfter(ifOp1);
  builder.create<mlir::NVVM::Barrier0Op>(loc);
  // producer and consumer
  mlir::AffineExpr expr1 = builder.getAffineDimExpr(0);
  auto set1 = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr - 128}), llvm::ArrayRef<bool>({false}));
  auto ifOp2 = builder.create<mlir::affine::AffineIfOp>(loc, set1, mlir::ValueRange{tid}, true);
  builder.setInsertionPointToStart(ifOp2.getThenBlock());
  // producer set max reg
  auto action = mlir::NVVM::SetMaxRegisterActionAttr::get(context, mlir::NVVM::SetMaxRegisterAction::decrease);
  auto reg_count = mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), 40);
  builder.create<mlir::NVVM::SetMaxRegisterOp>(loc, reg_count, action);
  // tma copy if
  auto ifOp3 = builder.create<mlir::affine::AffineIfOp>(loc, set_, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp3.getThenBlock());
  // tma copy 
  // builder.create<mlir::nvgpu::TmaAsyncLoadOp>(loc, smA, init_full_br, operands[0], );

  builder.setInsertionPointToStart(ifOp2.getElseBlock());
  // consumer
  auto action1 = mlir::NVVM::SetMaxRegisterActionAttr::get(context, mlir::NVVM::SetMaxRegisterAction::decrease);
  auto reg_count1 = mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), 232);
  builder.create<mlir::NVVM::SetMaxRegisterOp>(loc, reg_count1, action1);

  builder.setInsertionPointAfter(ifOp2);
  builder.create<mlir::NVVM::SyncWarpOp>(loc, mask);

  // print
  llvm::outs() << module << "\n";

  // transform and lowering
  compiler.transform(module);
  compiler.lowering(module);
  llvm::outs() << module << "\n";
  return "";
}

// create llvm funcOp
mlir::LLVM::LLVMFuncOp buildLLVMFunction(mlir::OpBuilder &builder, mlir::StringRef kernel_name, unsigned arg_num) {
  mlir::MLIRContext *context = builder.getContext();
  // === 函数参数类型列表 ===
  llvm::SmallVector<Type, 3> argTypes;
  for(unsigned i=0; i<arg_num; i++) {
    argTypes.push_back(mlir::LLVM::LLVMPointerType::get(context));
  }
  // === 构造函数类型 ===
  auto voidTy = mlir::LLVM::LLVMVoidType::get(context);
  auto funcType = mlir::LLVM::LLVMFunctionType::get(voidTy, argTypes, /*isVarArg=*/false);
  // === 创建函数操作 ===
  auto funcOp = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), kernel_name, funcType);
  // func attr
  const auto trueType = mlir::IntegerType::get(context, 1, mlir::IntegerType::Unsigned);
  funcOp->setAttr(mlir::NVVM::NVVMDialect::getKernelFuncAttrName(), builder.getIntegerAttr(trueType, 1));
  // func args attr
  const auto i32_type = mlir::IntegerType::get(context, 32);
  const auto byteType = mlir::IntegerType::get(context, 8);
  const auto arrayType = mlir::LLVM::LLVMArrayType::get(context, byteType, 128);
  for (unsigned i=0; i<funcOp.getNumArguments(); ++i) {
    funcOp.setArgAttr(i, mlir::LLVM::LLVMDialect::getByValAttrName(), mlir::TypeAttr::get(arrayType));
    funcOp.setArgAttr(i, mlir::NVVM::NVVMDialect::getGridConstantAttrName(), mlir::UnitAttr::get(context));
    funcOp.setArgAttr(i, mlir::LLVM::LLVMDialect::getAlignAttrName(), mlir::IntegerAttr::get(i32_type, 64));
  }
  // === 添加函数体（空）===
  auto &entryBlock = *funcOp.addEntryBlock(builder);
  builder.setInsertionPointToStart(&entryBlock);
  builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
  builder.setInsertionPointToStart(&entryBlock);
  return funcOp;
}

// create smem buffer
mlir::Value buildSmemForLLVMFunc(mlir::LLVM::LLVMFuncOp funcOp, uint64_t alignment=1024) {
  mlir::OpBuilder builder(funcOp);
  mlir::Location loc = builder.getUnknownLoc();
  builder.setInsertionPoint(funcOp);
  auto globOp = builder.create<mlir::LLVM::GlobalOp>(
                  loc, 
                  mlir::LLVM::LLVMArrayType::get(builder.getF16Type(), 0), 
                  false, 
                  LLVM::Linkage::External,
                  "smem", 
                  mlir::Attribute(),
                  /*alignment=*/alignment,
                  /*addrSpace=*/3);
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  auto smem_ptr = builder.create<mlir::LLVM::AddressOfOp>(loc, globOp);
  return smem_ptr;
}

// create tidx and bidx
std::pair<mlir::Value, mlir::Value> buildTidXAndBidXOps(mlir::OpBuilder &builder) {
  mlir::Location loc = builder.getUnknownLoc();
  // mlir::Value tidx = builder.create<mlir::NVVM::ThreadIdXOp>(loc, builder.getI32Type());
  // mlir::Value bidx = builder.create<mlir::NVVM::BlockIdXOp>(loc, builder.getI32Type());
  mlir::Value tidx = builder.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  mlir::Value bidx = builder.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);
  return std::make_pair(tidx, bidx);
}

// remapping bid
std::pair<mlir::Value, mlir::Value> remappingBidXToYX(mlir::OpBuilder &builder, mlir::Value bid, int64_t bx_num, int64_t BM, int64_t BN) {
  mlir::AffineExpr bid_expr = builder.getAffineDimExpr(0);
  mlir::AffineExpr by_expr = bid_expr.floorDiv(bx_num) * BM;
  mlir::AffineExpr bx_expr = (bid_expr % bx_num) * BN;
  mlir::Location loc = builder.getUnknownLoc();
  auto by = builder.create<mlir::affine::AffineApplyOp>(loc, mlir::ArrayRef<mlir::AffineExpr>({by_expr}), mlir::ValueRange({bid}));
  auto bx = builder.create<mlir::affine::AffineApplyOp>(loc, mlir::ArrayRef<mlir::AffineExpr>({bx_expr}), mlir::ValueRange({bid}));
  return std::make_pair(by, bx);
}

// create prefetch tensormap operation
void buildPrefetchTensorMap(mlir::OpBuilder &builder, llvm::SmallVector<mlir::Value> tm_ptrs, mlir::Value tid, int64_t idx) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::AffineExpr expr = builder.getAffineDimExpr(0);
  auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr - idx}), llvm::ArrayRef<bool>({true}));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(loc, set, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp.getThenBlock());
  for (auto tm_ptr : tm_ptrs) {
    builder.create<mlir::NVVM::PrefetchTensorMapOp>(loc, tm_ptr, mlir::Value());
  }
  builder.setInsertionPointAfter(ifOp);
  auto mask = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getIntegerType(32), builder.getI32IntegerAttr(0xffffffff));
  builder.create<mlir::NVVM::SyncWarpOp>(loc, mask);
}

// get elemenet ptr
mlir::Value getSmemPtr(mlir::OpBuilder &builder, mlir::Value smem_ptr, mlir::Type result_type, int64_t start) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type i32Type = builder.getI32Type();
  mlir::Value index = builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, mlir::IntegerAttr::get(builder.getIndexType(), start));
  auto sm_ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext(), 3);
  auto ptr = builder.create<mlir::LLVM::GEPOp>(loc, sm_ptr_type, result_type, smem_ptr, mlir::ValueRange{index});
  return ptr;
}

// init mbr
void initSmemMbarrier(
  mlir::OpBuilder &builder, 
  llvm::SmallVector<mlir::Value> mbr_ptrs, 
  llvm::SmallVector<uint32_t> counts, 
  mlir::Value tid, 
  int64_t idx) {
  assert(counts.size() == mbr_ptrs.size());
  mlir::MLIRContext *context = builder.getContext();
  mlir::Location loc = builder.getUnknownLoc();
  mlir::AffineExpr expr = builder.getAffineDimExpr(0);
  auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr - idx}), llvm::ArrayRef<bool>({true}));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(loc, set, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp.getThenBlock());
  for (int i=0; i<counts.size(); i++) {
    mlir::Value count = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), counts[i]);
    builder.create<mlir::NVVM::MBarrierInitSharedOp>(loc, mbr_ptrs[i], count, mlir::Value());
  }
  auto pta = mlir::NVVM::ProxyKindAttr::get(context, mlir::NVVM::ProxyKind::async_shared);
  auto ssa = mlir::NVVM::SharedSpaceAttr::get(context, mlir::NVVM::SharedSpace::shared_cta);
  builder.create<mlir::NVVM::FenceProxyOp>(loc, pta, ssa);
  builder.setInsertionPointAfter(ifOp);
  builder.create<mlir::NVVM::Barrier0Op>(loc);
}

// inline set max registers ptx code
mlir::Operation* inlineSetMaxRegPtx(mlir::OpBuilder &builder, int64_t regCount, mlir::NVVM::SetMaxRegisterAction action) {
  mlir::Location loc = builder.getUnknownLoc();
  std::string ptx_code = "setmaxnreg.dec.sync.aligned.u32 $0;";
  if (action == mlir::NVVM::SetMaxRegisterAction::increase) {
    ptx_code = "setmaxnreg.inc.sync.aligned.u32 $0;";
  } 
  auto reg_count = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), regCount);
  return builder.create<mlir::NVVM::InlinePtxOp>(loc, mlir::ValueRange(), mlir::ValueRange({reg_count}), 
    builder.getStringAttr(ptx_code), mlir::Value());

}

// create comsumer and producer
std::pair<mlir::Operation*, mlir::Operation*> buildComsumerAndProducerBlock(
  mlir::OpBuilder &builder, 
  mlir::Value tid,
  int32_t comsumer_thread_num,
  uint32_t comsumer_reg_count, 
  uint32_t producer_reg_count) {
  
  mlir::MLIRContext *context = builder.getContext();
  mlir::Location loc = builder.getUnknownLoc();
  mlir::AffineExpr expr = builder.getAffineDimExpr(0);
  auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr - comsumer_thread_num}), llvm::ArrayRef<bool>({false}));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(loc, set, mlir::ValueRange{tid}, true);
  builder.setInsertionPointToStart(ifOp.getThenBlock());
  // producer block
  // auto de_attr = mlir::NVVM::SetMaxRegisterActionAttr::get(context, );
  // auto op1 = builder.create<mlir::NVVM::SetMaxRegisterOp>(loc, producer_reg_count, mlir::NVVM::SetMaxRegisterAction::decrease);
  auto op1 = inlineSetMaxRegPtx(builder, producer_reg_count, mlir::NVVM::SetMaxRegisterAction::decrease);
  builder.setInsertionPointToStart(ifOp.getElseBlock());
  // comsumer block
  // auto ac_attr = mlir::NVVM::SetMaxRegisterActionAttr::get(context, );
  // auto op2 = builder.create<mlir::NVVM::SetMaxRegisterOp>(loc, comsumer_reg_count, mlir::NVVM::SetMaxRegisterAction::increase);
  auto op2 = inlineSetMaxRegPtx(builder, comsumer_reg_count, mlir::NVVM::SetMaxRegisterAction::increase);
  return std::make_pair(op1, op2);
}

// tma load
void TMALoad(mlir::OpBuilder &builder, mlir::Value tid, mlir::Value by, mlir::Value bx, mlir::Value mbr1, mlir::Value mbr2, 
             mlir::Value sma_ptr, mlir::Value smb_ptr, mlir::Value tma_desc_a, mlir::Value tma_desc_b,
             int32_t idx, int64_t K, int64_t BM, int64_t BN, int64_t BK) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Location loc = builder.getUnknownLoc();
  mlir::AffineExpr expr = builder.getAffineDimExpr(0);
  auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr - idx}), llvm::ArrayRef<bool>({true}));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(loc, set, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp.getThenBlock());
  // for k
  auto ticks = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), 0x989680);
  auto one_idx = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), 1);
  auto forK = builder.create<mlir::affine::AffineForOp>(loc, /*lb*/0, K, BK, mlir::ValueRange(), 
    [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value kiv, mlir::ValueRange iterArgs) {
      // wait
      auto kiv_ = builder.create<mlir::UnrealizedConversionCastOp>(loc, builder.getI32Type(), kiv);
      auto add = builder.create<mlir::LLVM::AddOp>(loc, one_idx.getResult(), kiv_.getResult(0));
      auto phase = builder.create<mlir::LLVM::AndOp>(loc, add, one_idx);
      builder.create<mlir::NVVM::MBarrierTryWaitParitySharedOp>(loc, mbr1, phase, ticks);
      // tma copy a
      auto l1hint = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64Type(), 0x1000000000000000);
      auto by_ = builder.create<mlir::UnrealizedConversionCastOp>(loc, builder.getI32Type(), by);
      builder.create<mlir::NVVM::CpAsyncBulkTensorGlobalToSharedClusterOp>(
        loc, sma_ptr, tma_desc_a, mlir::ValueRange({kiv_.getResult(0), by_.getResult(0)}), mbr2, 
        mlir::ValueRange(), mlir::Value(), l1hint.getResult(), mlir::Value());
      // tma copy b
      auto bx_ = builder.create<mlir::UnrealizedConversionCastOp>(loc, builder.getI32Type(), bx);
      builder.create<mlir::NVVM::CpAsyncBulkTensorGlobalToSharedClusterOp>(
        loc, smb_ptr, tma_desc_b, mlir::ValueRange({kiv_.getResult(0), bx_.getResult(0)}), mbr2, 
        mlir::ValueRange(), mlir::Value(), l1hint.getResult(), mlir::Value());
      // arrive_expect_tx
      int64_t transction_bytes = (BM * BK + BN * BK) * /*half*/2;
      auto tbs = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), transction_bytes);
      builder.create<mlir::NVVM::MBarrierArriveExpectTxSharedOp>(loc, mbr2, tbs.getResult(), mlir::Value());
      b.create<mlir::affine::AffineYieldOp>(loc);
    });
}

void llvmTest() {
  KernelGenerator generator;        // 没有外部导入module
  DGCompiler compiler(Target::CUDA, "90");

  auto module = generator.getModule();
  mlir::OpBuilder builder(module);
  mlir::Location loc = builder.getUnknownLoc();
  builder.setInsertionPointToStart(module.getBody());
  mlir::MLIRContext *context = module.getContext();
  // create func
  auto funcOp = buildLLVMFunction(builder, "kernel", 3);
  mlir::ValueRange operands = funcOp.getArguments();
  // create smem buffer
  auto smem_ptr = buildSmemForLLVMFunc(funcOp);
  // create tid and bid
  auto [tid, bid] = buildTidXAndBidXOps(builder);
  auto [by, bx] = remappingBidXToYX(builder, bid, /*4096/128*/32, /*BM*/128, /*BN*/128);
  // create prefetch
  llvm::SmallVector<mlir::Value> tm_args(operands);
  buildPrefetchTensorMap(builder, tm_args, tid, /*thread index*/128);
  // get element ptr
  auto sma_ptr = getSmemPtr(builder, smem_ptr, builder.getF16Type(), 0);
  auto smb_ptr = getSmemPtr(builder, smem_ptr, builder.getF16Type(), 8192);
  auto smc_ptr = getSmemPtr(builder, smem_ptr, builder.getF16Type(), 16384);
  auto mbr_ptr = getSmemPtr(builder, smem_ptr, builder.getF16Type(), 24576);

  auto full_mbr_ptr = getSmemPtr(builder, mbr_ptr, builder.getI64Type(), 0);
  auto empty_mbr_ptr = getSmemPtr(builder, mbr_ptr, builder.getI64Type(), 1);
  // init mbarrier
  llvm::SmallVector<mlir::Value> mbr_ptrs{full_mbr_ptr, empty_mbr_ptr};
  llvm::SmallVector<uint32_t> counts{1, 8};
  initSmemMbarrier(builder, mbr_ptrs, counts, tid, /*thread index*/128);
  // create producer and comsumer
  auto [pos1, pos2] = buildComsumerAndProducerBlock(builder, tid, /*c_th_num*/128, /*c_reg_count*/232, /*p_reg_count*/40);
  // tma copy
  builder.setInsertionPointAfter(pos1);
  TMALoad(builder, tid, by, bx, empty_mbr_ptr, full_mbr_ptr, sma_ptr, smb_ptr, 
          operands[0], operands[1], 128, /*K*/4096, /*BM*/128, /*BN*/128, /*BK*/64);
  
  llvm::outs() << module << "\n";
}

void fileLowering() {
  // 
  KernelGenerator generator;        // 没有外部导入module
  DGCompiler compiler(Target::CUDA, "90");
  auto path = generator.loadMLIRFile("/home/xiebaokang/projects/cuda/test/test.mlir", false);
  llvm::outs() << path << "\n";
}

void loadAndStoreTest() {
  KernelGenerator generator;        // 没有外部导入module
  DGCompiler compiler(Target::CUDA, "90");

  auto module = generator.getModule();
  mlir::OpBuilder builder(module);
  mlir::Location loc = builder.getUnknownLoc();
  builder.setInsertionPointToStart(module.getBody());
  mlir::MLIRContext *context = module.getContext();
  // create func
  auto funcOp = buildLLVMFunction(builder, "kernel", 2);
  mlir::ValueRange operands = funcOp.getArguments();
  // create smem buffer
  auto smem_ptr = buildSmemForLLVMFunc(funcOp);
  // create tid and bid
  auto [tid, bid] = buildTidXAndBidXOps(builder);
  auto [by, bx] = remappingBidXToYX(builder, bid, /*gridN*/32, /*BM*/128, /*BN*/128);
  auto by_ = builder.create<mlir::UnrealizedConversionCastOp>(loc, builder.getI32Type(), by);
  auto bx_ = builder.create<mlir::UnrealizedConversionCastOp>(loc, builder.getI32Type(), bx);
  // prefetch tenormap
  buildPrefetchTensorMap(builder, {operands[0], operands[1]}, tid, 0);
  // init mbarrier
  auto sm_ptr = getSmemPtr(builder, smem_ptr, builder.getF16Type(), 0);  // 0-16384
  auto mbr_ptr = getSmemPtr(builder, smem_ptr, builder.getF16Type(), 16384);  // 16384
  initSmemMbarrier(builder, {mbr_ptr}, {1}, tid, /*thread index*/0);
  // tma load
  auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({builder.getAffineDimExpr(0)}), llvm::ArrayRef<bool>({true}));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(loc, set, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp.getThenBlock());
  // load  a
  auto l1hint = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64Type(), 0x1000000000000000);
  // ptr smem
  builder.create<mlir::NVVM::CpAsyncBulkTensorGlobalToSharedClusterOp>(
    loc, sm_ptr, operands[0], mlir::ValueRange({bx_.getResult(0), by_.getResult(0)}), mbr_ptr, 
    mlir::ValueRange(), mlir::Value(), l1hint.getResult(), mlir::Value());
  // arrive expect
  int64_t transction_bytes = (128 * 128) * /*half*/2;
  auto tbs = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), transction_bytes);
  builder.create<mlir::NVVM::MBarrierArriveExpectTxSharedOp>(loc, mbr_ptr, tbs.getResult(), mlir::Value());
  builder.setInsertionPointAfter(ifOp);
  // wait
  auto ifOp1 = builder.create<mlir::affine::AffineIfOp>(loc, set, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp1.getThenBlock());
  auto ticks = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), 0x989680);
  auto phase = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), 0);
  builder.create<mlir::NVVM::MBarrierTryWaitParitySharedOp>(loc, mbr_ptr, phase, ticks);
  // tma store
  builder.create<mlir::NVVM::CpAsyncBulkTensorSharedCTAToGlobalOp>(
    loc, operands[1], sm_ptr, mlir::ValueRange({bx_.getResult(0), by_.getResult(0)}), mlir::Value());
  builder.setInsertionPointAfter(ifOp1);
  builder.create<mlir::NVVM::Barrier0Op>(loc);
  llvm::outs() << module << "\n";
}

int main() {
  // llvmTest();
  // nvgpu_tma_test();
  fileLowering();
  // loadAndStoreTest();
  return 0;
}