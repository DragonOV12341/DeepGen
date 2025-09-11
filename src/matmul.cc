#include "compiler.h"
#include "operator.h"
#include "Python.h"

using namespace DeepGen;

// create llvm funcOp
mlir::LLVM::LLVMFuncOp buildLLVMFunction(mlir::OpBuilder &builder, mlir::StringRef kernel_name, const std::vector<std::string>& arg_types) {
  mlir::MLIRContext *context = builder.getContext();
  // === 函数参数类型列表 ===
  llvm::SmallVector<Type, 3> argTypes;
  for(unsigned i=0; i<arg_types.size(); i++) {
    if (arg_types[i] == "tensorDesc") {
      argTypes.push_back(mlir::LLVM::LLVMPointerType::get(context));
    } else {
      argTypes.push_back(mlir::LLVM::LLVMPointerType::get(context, 1));
    }
  }
  // === 构造函数类型 ===
  auto voidTy = mlir::LLVM::LLVMVoidType::get(context);
  auto funcType = mlir::LLVM::LLVMFunctionType::get(voidTy, argTypes, /*isVarArg=*/false);
  auto funcOp = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), kernel_name, funcType);
  // func attr
  const auto trueType = mlir::IntegerType::get(context, 1, mlir::IntegerType::Unsigned);
  funcOp->setAttr(mlir::NVVM::NVVMDialect::getKernelFuncAttrName(), builder.getIntegerAttr(trueType, 1));
  // func args attr
  for (unsigned i=0; i<funcOp.getNumArguments(); ++i) {
    if (arg_types[i] == "tensorDesc") {
      const auto i32_type = mlir::IntegerType::get(context, 32);
      const auto byteType = mlir::IntegerType::get(context, 8);
      const auto arrayType = mlir::LLVM::LLVMArrayType::get(context, byteType, 128);
      funcOp.setArgAttr(i, mlir::LLVM::LLVMDialect::getByValAttrName(), mlir::TypeAttr::get(arrayType));
      funcOp.setArgAttr(i, mlir::NVVM::NVVMDialect::getGridConstantAttrName(), mlir::UnitAttr::get(context));
      funcOp.setArgAttr(i, mlir::LLVM::LLVMDialect::getAlignAttrName(), mlir::IntegerAttr::get(i32_type, 64));
    }
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
  mlir::Type type = mlir::LLVM::LLVMArrayType::get(builder.getF16Type(), 0);
  auto globOp = builder.create<mlir::LLVM::GlobalOp>(
                loc, type, false,  LLVM::Linkage::External, "smem", mlir::Attribute(), /*align=*/alignment, /*memSpace=*/3);
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  auto smem_ptr = builder.create<mlir::LLVM::AddressOfOp>(loc, globOp);
  return smem_ptr;
}

// create tidx and bidx
std::pair<mlir::Value, mlir::Value> buildTidAndBidOps(mlir::OpBuilder &builder) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Value tidx = builder.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  mlir::Value bidx = builder.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);
  return std::make_pair(tidx, bidx);
}

// get id
template <typename T>
std::pair<mlir::Value, mlir::Value> getIdx(mlir::OpBuilder &builder) {
  mlir::Location loc = builder.getUnknownLoc();
  auto ssa_id = builder.create<T>(loc, builder.getI32Type());
  auto id = builder.create<mlir::UnrealizedConversionCastOp>(loc, builder.getIndexType(), ssa_id.getResult());
  return std::make_pair(ssa_id.getResult(), id.getResult(0));
}

// get warp group id
std::pair<mlir::Value, mlir::Value> getWarpGroupIdx(mlir::OpBuilder &builder, mlir::Value tid) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Value i7 = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), 7);
  auto ssa_wgid = builder.create<mlir::LLVM::LShrOp>(loc, tid, i7);
  auto wgid = builder.create<mlir::UnrealizedConversionCastOp>(loc, builder.getIndexType(), ssa_wgid.getResult());
  return std::make_pair(ssa_wgid.getResult(), wgid.getResult(0));
}

// remapping idx
template <int64_t LayoutY, int64_t LayoutX>
llvm::SmallVector<mlir::Value> mappingIdx(mlir::OpBuilder &builder, mlir::Value idx, mlir::ArrayRef<int64_t> offset={}, bool is_ssa=true) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::ArrayRef<int64_t> layout = mlir::ArrayRef<int64_t>({LayoutY, LayoutX});
  auto delineOp = builder.create<mlir::affine::AffineDelinearizeIndexOp>(loc, idx, layout);
  auto deline_idxs = delineOp.getResults();
  llvm::SmallVector<mlir::Value> new_idxs;
  for (unsigned i=0; i<deline_idxs.size(); ++i) {
    mlir::Value new_idx = deline_idxs[i];
    if (offset.size()) {
      assert(layout.size() == offset.size());
      mlir::AffineExpr expr = builder.getAffineDimExpr(0) * offset[i];
      auto applyOp = builder.create<mlir::affine::AffineApplyOp>(loc, mlir::ArrayRef<mlir::AffineExpr>({expr}), mlir::ValueRange({new_idx}));
      new_idx = applyOp.getResult();
    }
    if (is_ssa)
      new_idx = builder.create<mlir::UnrealizedConversionCastOp>(loc, builder.getI32Type(), new_idx).getResult(0);
    new_idxs.push_back(new_idx);
  }
  return new_idxs;
}

// create prefetch tensormap operation
void buildPrefetchTensorMap(mlir::OpBuilder &builder, llvm::SmallVector<mlir::Value> tensor_desc_ptrs, mlir::Value tid, int64_t idx) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::AffineExpr expr = builder.getAffineDimExpr(0);
  auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr - idx}), llvm::ArrayRef<bool>({true}));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(loc, set, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp.getThenBlock());
  for (auto td_ptr : tensor_desc_ptrs) {
    builder.create<mlir::NVVM::PrefetchTensorMapOp>(loc, td_ptr, mlir::Value());
  }
  builder.setInsertionPointAfter(ifOp);
  auto mask = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getIntegerType(32), builder.getI32IntegerAttr(0xffffffff));
  builder.create<mlir::NVVM::SyncWarpOp>(loc, mask);
}

// get elemenet ptr
mlir::Value getSmemPtr(mlir::OpBuilder &builder, mlir::Value smem_ptr, mlir::Type elem_type, int64_t idx) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type i32Type = builder.getI32Type();
  mlir::Value index = builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, mlir::IntegerAttr::get(builder.getIndexType(), idx));
  auto sm_ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext(), 3);
  auto ptr = builder.create<mlir::LLVM::GEPOp>(loc, sm_ptr_type, elem_type, smem_ptr, mlir::ValueRange{index});
  return ptr;
}

// get registers ptr
template <int64_t BM, int64_t BN, int64_t BK, int64_t WM, int64_t WN, int64_t MMA_M, int64_t MMA_N, int64_t MMA_K>
mlir::Value getAccumulator(mlir::OpBuilder &builder, mlir::Type elem_type) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::MLIRContext *context = builder.getContext();
  // create llvm struct
  unsigned numMembers;
  if (elem_type.isF32() || elem_type.isInteger(32))
    numMembers = WN / 2;
  else if (elem_type.isF16())
    numMembers = WN / 4;
  else
    llvm_unreachable("unsupported type for warpgroup accumulator");
  llvm::SmallVector<mlir::Type> innerStructBody;
  for (unsigned i = 0; i < numMembers; i++)
    innerStructBody.push_back(builder.getI32Type());
  auto innerStructType = mlir::LLVM::LLVMStructType::getLiteral(context, innerStructBody);
  llvm::SmallVector<mlir::Type> structBody;
  for (int i=0; i<BM; i+=WM)
    for (int j=0; j<BN; j+=WN)
      structBody.push_back(innerStructType);
  auto packStructType = mlir::LLVM::LLVMStructType::getLiteral(context, structBody);
  // init acc
  mlir::Type elemType = mlir::cast<mlir::LLVM::LLVMStructType>(packStructType.getBody().front()).getBody().front();
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(loc, elemType, builder.getZeroAttr(elemType));
  mlir::Value packStruct = builder.create<mlir::LLVM::PoisonOp>(loc, packStructType);
  llvm::SmallVector<mlir::Value> innerStructs;
  // Unpack the structs and set all values to zero
  for (auto [idx, s] : llvm::enumerate(packStructType.getBody())) {
    auto structType = mlir::cast<mlir::LLVM::LLVMStructType>(s);
    mlir::Value structValue = builder.create<mlir::LLVM::ExtractValueOp>(loc, packStruct, idx);
    for (unsigned i=0; i<structType.getBody().size(); ++i) {
      structValue = builder.create<mlir::LLVM::InsertValueOp>(loc, structType, structValue, zero, mlir::ArrayRef<int64_t>({i}));
    }
    innerStructs.push_back(structValue);
  }
  // Pack the inner structs into a single struct
  for (auto [idx, matrix] : llvm::enumerate(innerStructs)) {
    packStruct = builder.create<mlir::LLVM::InsertValueOp>(loc, packStruct.getType(), packStruct, matrix, idx);
  }
  return packStruct;
}

// init mbr
void initSmemMbarrier(mlir::OpBuilder &builder, llvm::DenseMap<mlir::Value, uint32_t> mbrs, mlir::Value tid, int64_t idx) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Location loc = builder.getUnknownLoc();
  mlir::AffineExpr expr = builder.getAffineDimExpr(0);
  auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr - idx}), llvm::ArrayRef<bool>({true}));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(loc, set, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp.getThenBlock());
  for (auto &[mbr_ptr, count] : mbrs) {
    mlir::Value count_val = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), count);
    builder.create<mlir::NVVM::MBarrierInitSharedOp>(loc, mbr_ptr, count_val, mlir::Value());
  }
  auto pta = mlir::NVVM::ProxyKindAttr::get(context, mlir::NVVM::ProxyKind::async_shared);
  auto ssa = mlir::NVVM::SharedSpaceAttr::get(context, mlir::NVVM::SharedSpace::shared_cta);
  builder.create<mlir::NVVM::FenceProxyOp>(loc, pta, ssa);
  builder.setInsertionPointAfter(ifOp);
  builder.create<mlir::NVVM::Barrier0Op>(loc);
}

// tma load
void TMALoad(
  mlir::OpBuilder &builder, 
  llvm::SmallVector<mlir::Value> srcs, 
  llvm::SmallVector<mlir::Value> dsts, 
  llvm::SmallVector<mlir::ValueRange> coordinates,
  mlir::Value mbarrier_ptr,
  int64_t transction_bytes,
  mlir::Value tid, 
  int64_t idx
) {
  assert(srcs.size() == dsts.size() && dsts.size() == coordinates.size());
  mlir::Location loc = builder.getUnknownLoc();
  mlir::AffineExpr expr = builder.getAffineDimExpr(0);
  auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr - idx}), llvm::ArrayRef<bool>({true}));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(loc, set, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp.getThenBlock());

  auto l2hint = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64Type(), 0x1000000000000000);
  for (int i=0; i<srcs.size(); i++) {
    builder.create<mlir::NVVM::CpAsyncBulkTensorGlobalToSharedClusterOp>(
      loc, dsts[i], srcs[i], coordinates[i], mbarrier_ptr, mlir::ValueRange(), mlir::Value(), l2hint.getResult(), mlir::Value());
  }
  // arrive expect
  auto tbs = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), transction_bytes);
  builder.create<mlir::NVVM::MBarrierArriveExpectTxSharedOp>(loc, mbarrier_ptr, tbs.getResult(), mlir::Value());
  // // else
  // builder.setInsertionPointToStart(ifOp.getElseBlock());
  // auto token_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  // builder.create<mlir::NVVM::MBarrierArriveSharedOp>(loc, token_type, mbarrier_ptr);
  builder.setInsertionPointAfter(ifOp);
}


// make smem desc
mlir::Value makeSmemDesc(OpBuilder &builder, mlir::Value smem_ptr, SwizzleMode swizzle_mode) {
  unsigned layout =
    (swizzle_mode == SwizzleMode::SWIZZLE_128B)  ? 128
    : (swizzle_mode == SwizzleMode::SWIZZLE_64B) ? 64
    : (swizzle_mode == SwizzleMode::SWIZZLE_32B) ? 32
                                                : 1;
  unsigned swizzle =
    (swizzle_mode == SwizzleMode::SWIZZLE_128B)  ? 1
    : (swizzle_mode == SwizzleMode::SWIZZLE_64B) ? 2
    : (swizzle_mode == SwizzleMode::SWIZZLE_32B) ? 3
                                                : 0;
  
  auto ti64 = builder.getIntegerType(64);
  mlir::Location loc = builder.getUnknownLoc();
  auto makeConst = [&](uint64_t index) -> Value {
    return builder.create<mlir::LLVM::ConstantOp>(loc, ti64, builder.getI64IntegerAttr(index));
  };
  auto shiftLeft = [&](mlir::Value value, unsigned shift) -> Value {
    return builder.create<mlir::LLVM::ShlOp>(loc, ti64, value, makeConst(shift));
  };
  auto shiftRight = [&](mlir::Value value, unsigned shift) -> Value {
    return builder.create<mlir::LLVM::LShrOp>(loc, ti64, value, makeConst(shift));
  };
  auto insertBit = [&](mlir::Value desc, mlir::Value val, int startBit) {
    return builder.create<mlir::LLVM::OrOp>(loc, ti64, desc, shiftLeft(val, startBit));
  };

  mlir::Value basePtr = builder.create<mlir::LLVM::PtrToIntOp>(loc, ti64, smem_ptr);
  mlir::Value basePtr14bit = shiftRight(shiftLeft(basePtr, 46), 50);
  mlir::Value strideDim = makeConst((layout << 3) >> 4);

  int startSwizzleBit = 62, startOffsetBit = 49, startStrideBit = 32, startLeadBit = 16, startBaseAddrBit = 0;
  Value dsc = makeConst(0);
  // // [62,64)  swizzle type
  dsc = insertBit(dsc, makeConst(layout), startSwizzleBit);
  // // [49,52)  base_offset
  dsc = insertBit(dsc, makeConst(0), startOffsetBit);
  // // [32,46)  stride
  dsc = insertBit(dsc, strideDim, startStrideBit);
  // // [16,30)  leading dimension  / 因为smem_ptr已经设置了偏移了，所以这个就不需要设置了
  dsc = insertBit(dsc, makeConst(0), startLeadBit);
  // // [0,14)   start_address
  dsc = insertBit(dsc, basePtr14bit, startBaseAddrBit);
  return dsc;
}

// get elemet ptr
mlir::Value getElementPtr(OpBuilder &builder, mlir::Value old_ptr, mlir::Type elem_type, mlir::AffineExpr expr, const llvm::SmallVector<mlir::Value>& operands) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::MLIRContext *context = builder.getContext();
  auto offset = builder.create<mlir::affine::AffineApplyOp>(loc, mlir::ArrayRef<mlir::AffineExpr>({expr}), mlir::ValueRange(operands));
  auto offset_ = builder.create<mlir::UnrealizedConversionCastOp>(loc, builder.getI32Type(), offset.getResult());
  auto elem_ptr = builder.create<mlir::LLVM::GEPOp>(loc, old_ptr.getType(), elem_type, old_ptr, mlir::ValueRange{offset_.getResult(0)});
  return elem_ptr;
}

// wgmma
template <int64_t BM, int64_t BN, int64_t BK, int64_t WM, int64_t WN, int64_t MMA_M, int64_t MMA_N, int64_t MMA_K>
mlir::Value wgmma(
  mlir::OpBuilder &builder, 
  mlir::Value sma_ptr, 
  mlir::Value smb_ptr, 
  NVVM::MMALayout layout_a, 
  NVVM::MMALayout layout_b, 
  mlir::Value accums, 
  mlir::Value mbarrier, 
  mlir::Value wg_y, 
  mlir::Value wg_x, 
  mlir::Value phase
) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::MLIRContext *context = builder.getContext();
  // wait
  auto ticks = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), 0x989680);
  auto i1 = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), 1);
  auto i4 = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), 4);
  phase = builder.create<mlir::LLVM::LShrOp>(loc, phase, i4);
  // auto add = builder.create<mlir::LLVM::AddOp>(loc, phase, i1);
  auto phase_ = builder.create<mlir::LLVM::AndOp>(loc, phase, i1);
  builder.create<mlir::NVVM::MBarrierTryWaitParitySharedOp>(loc, mbarrier, phase_, ticks);
  // wgmma
  int idx = 0;
  // auto pos = builder.saveInsertionPoint();
  // builder.setInsertionPointAfter(accums.getDefiningOp());
  mlir::Value result = builder.create<mlir::LLVM::PoisonOp>(loc, accums.getType());
  // builder.restoreInsertionPoint(pos);
  for (int block_iter_y=0; block_iter_y<BM; block_iter_y+=WM) {
    for (int block_iter_x=0; block_iter_x<BN; block_iter_x+=WN) {
      builder.create<mlir::NVVM::WgmmaFenceAlignedOp>(loc);
      mlir::Value acc = builder.create<mlir::LLVM::ExtractValueOp>(loc, accums, idx);
      for (int iter_k=0; iter_k<BK; iter_k+=MMA_K) {
        mlir::AffineExpr wgid_expr = builder.getAffineDimExpr(0);
        mlir::AffineExpr k_expr = builder.getAffineConstantExpr(iter_k);
        mlir::AffineExpr expr_y = (layout_a == NVVM::MMALayout::row) ? (block_iter_y + wgid_expr * MMA_M) : k_expr;
        mlir::AffineExpr expr_x = (layout_a == NVVM::MMALayout::row) ? k_expr : (block_iter_y + wgid_expr * MMA_M);
        int64_t stride = (layout_a == NVVM::MMALayout::row) ? BK : BM;
        auto sma_start_ptr = getElementPtr(builder, sma_ptr, builder.getF16Type(), expr_y * stride + expr_x, {wg_y});
        expr_y = (layout_b == NVVM::MMALayout::row) ? k_expr : (block_iter_x + wgid_expr * MMA_N);
        expr_x = (layout_b == NVVM::MMALayout::row) ? (block_iter_x + wgid_expr * MMA_N) : k_expr;
        stride = (layout_b == NVVM::MMALayout::row) ? BN : BK;
        auto smb_start_ptr = getElementPtr(builder, smb_ptr, builder.getF16Type(), expr_y * stride + expr_x, {wg_x});

        auto desc_a = makeSmemDesc(builder, sma_start_ptr, SwizzleMode::SWIZZLE_NONE);
        auto desc_b = makeSmemDesc(builder, smb_start_ptr, SwizzleMode::SWIZZLE_NONE);

        auto shape = NVVM::MMAShapeAttr::get(context, MMA_M, MMA_N, MMA_K);
        auto type = NVVM::WGMMATypesAttr::get(context, NVVM::WGMMATypes::f16);
        auto scale_out = NVVM::WGMMAScaleOutAttr::get(context, NVVM::WGMMAScaleOut::one);
        auto scale_in = NVVM::WGMMAScaleInAttr::get(context, NVVM::WGMMAScaleIn::one);
        auto layout_a_ = NVVM::MMALayoutAttr::get(context, layout_a);
        auto layout_b_ = NVVM::MMALayoutAttr::get(context, layout_b);
        auto overflow = NVVM::MMAIntOverflowAttr::get(context, NVVM::MMAIntOverflow::wrapped);
        auto wgmmaOp = builder.create<mlir::NVVM::WgmmaMmaAsyncOp>(
          loc, acc.getType(), acc, desc_a, desc_b, shape, 
          type, type, type, scale_out, scale_in, scale_in, 
          layout_a_, layout_b_, overflow);
        acc = wgmmaOp.getResult();
      }
      result = builder.create<mlir::LLVM::InsertValueOp>(loc, accums.getType(), result, acc, idx++);
      builder.create<mlir::NVVM::WgmmaGroupSyncAlignedOp>(loc);
      builder.create<mlir::NVVM::WgmmaWaitGroupSyncOp>(loc, mlir::IntegerAttr::get(builder.getI64Type(), 0));
    }
  }
  // arrive
  return result;
}

// store registers to smem
template <int64_t BM, int64_t BN, int64_t WM, int64_t WN, int64_t MMA_M, int64_t MMA_N>
void StoreToSMem(mlir::OpBuilder &builder, mlir::Value reg, mlir::Value smem, mlir::Value wgy, mlir::Value wgx, mlir::Value wid, mlir::Value lid) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::MLIRContext *context = builder.getContext();
  auto item = mlir::cast<mlir::LLVM::LLVMStructType>(reg.getType()).getBody().front();
  auto reg_count = mlir::cast<mlir::LLVM::LLVMStructType>(item).getBody().size();

  auto wids = mappingIdx<2, 4>(builder, wid, mlir::ArrayRef<int64_t>(), /*ssa*/false);   // wids[0] == warpid / 4 (wx) ...
  auto lids = mappingIdx<2, 16>(builder, lid, mlir::ArrayRef<int64_t>(), /*ssa*/false);  // lids[0] == laneid / 16 (lx) ...

  mlir::AffineExpr wgy_expr = builder.getAffineDimExpr(0);
  mlir::AffineExpr wgx_expr = builder.getAffineDimExpr(1);
  mlir::AffineExpr wy_expr = builder.getAffineDimExpr(2);
  mlir::AffineExpr ly_expr = builder.getAffineDimExpr(3);
  mlir::AffineExpr lx_expr = builder.getAffineDimExpr(4);

  int idx = 0;
  for (int i=0; i<BM; i+=WM) {
    for (int j=0; j<BN; j+=WN) {
      mlir::Value acc = builder.create<mlir::LLVM::ExtractValueOp>(loc, reg, idx);
      for (int wave=0; wave<reg_count/4; wave++) {
        // get smem ptr
        mlir::AffineExpr off1 = (i + wgy_expr * MMA_M) * BN + (j + wgx_expr * MMA_N);
        mlir::AffineExpr off2 = (wy_expr * 16 + ly_expr) * MMA_N + (wave * 16 + lx_expr * 8);
        auto new_smem = getElementPtr(builder, smem, builder.getF16Type(), off1 + off2, {wgy, wgx, wids[1], lids[1], lids[0]});
        // get elem
        llvm::SmallVector<mlir::Value> elems;
        for (int num=0; num<4; num++) {
          int elem_idx = wave * 4 + num;
          mlir::Value elem = builder.create<mlir::LLVM::ExtractValueOp>(loc, acc, elem_idx);
          elems.push_back(elem);
        }
        // stmatrix
        builder.create<mlir::NVVM::StMatrixOp>(loc, new_smem, mlir::ValueRange(elems), mlir::NVVM::MMALayout::row);
      } 
      idx++;
    }
  }
  auto pta = mlir::NVVM::ProxyKindAttr::get(context, mlir::NVVM::ProxyKind::async_shared);
  auto ssa = mlir::NVVM::SharedSpaceAttr::get(context, mlir::NVVM::SharedSpace::shared_cta);
  builder.create<mlir::NVVM::FenceProxyOp>(loc, pta, ssa);
  builder.create<mlir::NVVM::Barrier0Op>(loc);
}

// TMA Store
void TMAStore(mlir::OpBuilder &builder, mlir::Value src, mlir::Value dst, mlir::ValueRange coordinates, mlir::Value tid, int64_t idx) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::AffineExpr expr = builder.getAffineDimExpr(0);
  auto set = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({expr - idx}), llvm::ArrayRef<bool>({true}));
  auto ifOp = builder.create<mlir::affine::AffineIfOp>(loc, set, mlir::ValueRange{tid}, false);
  builder.setInsertionPointToStart(ifOp.getThenBlock());

  builder.create<mlir::NVVM::CpAsyncBulkTensorSharedCTAToGlobalOp>(loc, dst, src, coordinates, mlir::Value());
  // arrive expect
  auto cmt = builder.create<mlir::NVVM::CpAsyncBulkCommitGroupOp>(loc);
  builder.setInsertionPointAfter(ifOp);
  auto mask = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getIntegerType(32), builder.getI32IntegerAttr(0xffffffff));
  builder.create<mlir::NVVM::SyncWarpOp>(loc, mask);
}


template<
  int64_t M, int64_t N, int64_t K, 
  int64_t BM, int64_t BN, int64_t BK,
  int64_t WM, int64_t WN,  // wave_m / wave_n
  int64_t MMA_M, int64_t MMA_N, int64_t MMA_K
>
void matmul() {
  KernelGenerator generator;
  DGCompiler compiler(Target::CUDA, "90");

  constexpr int64_t grid_x = N / BN;
  constexpr int64_t grid_y = M / BM;
  constexpr int64_t wg_layout_y = WM / MMA_M;
  constexpr int64_t wg_layout_x = WN / MMA_N;

  auto module = generator.getModule();
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(module);
  mlir::Location loc = builder.getUnknownLoc();
  builder.setInsertionPointToStart(module.getBody());

  auto funcOp = buildLLVMFunction(builder, "kernel", {"tensorDesc", "tensorDesc", "tensorDesc"});
  mlir::ValueRange operands = funcOp.getArguments();

  auto smem_ptr = buildSmemForLLVMFunc(funcOp);

  auto [ssa_wid, wid] = getIdx<mlir::NVVM::WarpIdOp>(builder);
  auto [ssa_bid, bid] = getIdx<mlir::NVVM::BlockIdXOp>(builder);
  auto [ssa_tid, tid] = getIdx<mlir::NVVM::ThreadIdXOp>(builder);
  auto [ssa_lid, lid] = getIdx<mlir::NVVM::LaneIdOp>(builder);
  auto [sss_wgid, wgid] = getWarpGroupIdx(builder, ssa_tid);

  // auto [tid, bid] = buildTidAndBidOps(builder);

  auto bids = mappingIdx<grid_y, grid_x>(builder, bid, mlir::ArrayRef<int64_t>({BM, BN}));
  auto wgids = mappingIdx<wg_layout_y, wg_layout_x>(builder, wgid, mlir::ArrayRef<int64_t>(), /*ssa*/false);
  mlir::Value wgy = wgids[0], wgx = wgids[1];

  buildPrefetchTensorMap(builder, {operands[0], operands[1], operands[2]}, tid, /*idx*/0);
  // sm size 66560 byte
  auto sma_ptr = getSmemPtr(builder, smem_ptr, builder.getF16Type(), 0);  // 0-8192 fp16
  auto smb_ptr = getSmemPtr(builder, smem_ptr, builder.getF16Type(), BM*BK);  // 8192-16384 fp16
  auto smc_ptr = getSmemPtr(builder, smem_ptr, builder.getF16Type(), BM*BK + BN*BK);  // 16384-32768 fp16
  auto mbr_ptr = getSmemPtr(builder, smem_ptr, builder.getF16Type(), BM*BK + BN*BK + BM*BN);  // 32768-33280 fp16

  // acc reg
  auto accs = getAccumulator<BM, BN, BK, WM, WN, MMA_M, MMA_N, MMA_K>(builder, builder.getF16Type());

  initSmemMbarrier(builder, {{mbr_ptr, 1}}, tid, 0);

  // mlir::Value matrixD = nullptr;
  auto forKOp = builder.create<mlir::affine::AffineForOp>(loc, 0, K, BK, mlir::ValueRange({accs}), 
    [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value kiv, mlir::ValueRange iterArgs) {
      auto bk = b.create<mlir::UnrealizedConversionCastOp>(loc, b.getI32Type(), kiv).getResult(0);
      auto accums = iterArgs[0];
      // load
      auto coord_a = mlir::ValueRange({bk, bids[0]});
      auto coord_b = mlir::ValueRange({bids[1], bk});
      int64_t transction_bytes = (BM * BK + BN * BK) * 2;
      TMALoad(b, {operands[0], operands[1]}, {sma_ptr, smb_ptr}, {coord_a, coord_b}, mbr_ptr, transction_bytes, tid, /*idx*/0);
      // compute
      auto item = wgmma<BM, BN, BK, WM, WN, MMA_M, MMA_N, MMA_K>(b, sma_ptr, smb_ptr, NVVM::MMALayout::row, NVVM::MMALayout::row, accums, mbr_ptr, wgy, wgx, bk);

      b.create<mlir::affine::AffineYieldOp>(l, item);
    });
  builder.create<mlir::NVVM::Barrier0Op>(loc);
  // store reg to smem, using stmatrix
  StoreToSMem<BM, BN, WM, WN, MMA_M, MMA_N>(builder, forKOp.getResults()[0], smc_ptr, wgy, wgx, wid, lid);
  // tma store
  TMAStore(builder, smc_ptr, operands[2], mlir::ValueRange(bids), tid, /*idx*/0);

  llvm::outs() << module << "\n";
}

int main() {
  matmul<
    4096, 4096, 4096, 
    128, 128, 16,
    128, 128,
    64, 128, 16
  >();
  return 0;
}