#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/raw_ostream.h"
#include "Dialect/Deepgen/IR/Dialect.h"
#include <mlir/IR/Attributes.h>
#include "Dialect/Deepgen/Utils/Utils.h"

using namespace mlir;
using namespace mlir::deepgen;

int main(int argc, char **argv)
{
    MLIRContext ctx;
    ctx.loadDialect<func::FuncDialect, arith::ArithDialect, deepgen::DeepgenDialect, memref::MemRefDialect, tensor::TensorDialect>();

    // 创建 OpBuilder
    OpBuilder builder(&ctx);
    auto mod = builder.create<ModuleOp>(builder.getUnknownLoc());

    // 设置插入点
    builder.setInsertionPointToEnd(mod.getBody());

    // 创建 func
    auto i32 = builder.getI32Type();
    auto tensorTy = mlir::RankedTensorType::get({1,5},i32);

    std::vector<int64_t> shapeA = {32, 32};
    std::vector<int64_t> griddim = {1,1,1};
    std::vector<int64_t> blockdim = {128,1,1};
    auto memrefTypeA = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shapeA), i32);
    

    mlir::TypeRange intypes = llvm::ArrayRef<Type>({i32,i32,tensorTy,memrefTypeA});
    mlir::TypeRange outtypes = llvm::ArrayRef<Type>({});

    auto funcType = builder.getFunctionType(intypes, outtypes);
    auto griddimAttr = builder.getDenseI64ArrayAttr(griddim);
    auto blockdimAttr = builder.getDenseI64ArrayAttr(blockdim);
    auto func = builder.create<deepgen::KernelFuncOp>(builder.getUnknownLoc(), "test", funcType, griddimAttr, blockdimAttr);

    // 添加基本块
    auto entry = func.addEntryBlock();
    auto args = entry->getArguments();

    // 设置插入点
    builder.setInsertionPointToEnd(entry);

    // 创建 arith.addi
    auto addi = builder.create<arith::AddIOp>(builder.getUnknownLoc(), args[0], args[1]);
    auto bid = builder.create<deepgen::BlockIdOp>(builder.getUnknownLoc(),  deepgen::Dimension::x );
    auto tid = builder.create<deepgen::ThreadIdOp>(builder.getUnknownLoc(), deepgen::Dimension::x ) ;
    auto mem = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), memrefTypeA);
    auto mem2 = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), memrefTypeA);
    // 创建 func.return
    builder.create<deepgen::SetLayoutOp>(builder.getUnknownLoc(), mem->getResult(0).getType(), mem->getResult(0), deepgen::LayoutKindAttr::get(&ctx, deepgen::LayoutKind::Normal));
    builder.create<deepgen::FillOp>(builder.getUnknownLoc(),  mem, addi.getResult());
    // auto loadRes=  builder.create<deepgen::LoadOp>(builder.getUnknownLoc(),  mem.getResult(), mlir::ValueRange({addi.getResult()}), true ) ;
    auto r = builder.create<deepgen::WgGemmOp>(builder.getUnknownLoc(), mem,mem,mem,mem );
//   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value source, ArrayRef<int64_t> staticLow, ArrayRef<int64_t> staticHigh, ValueRange low, ValueRange high, bool nofold = false, ArrayRef<NamedAttribute> attrs = {});
    std::vector<int64_t> low = {2};
    std::vector<int64_t> high = {3};

    std::vector<mlir::Value> low_v = {args[0]};
    std::vector<mlir::Value> high_v = {args[1]};
    mlir::ValueRange lvr = low_v;
    mlir::ValueRange hvr = high_v;
    // auto padded = builder.create<mlir::tensor::PadOp>(builder.getUnknownLoc(),tensorTy, args[2], low,high,lvr, hvr);
    builder.create<deepgen::CopyOp>(builder.getUnknownLoc(), mem,mem2, builder.getBoolAttr(true));
    std::vector<int64_t> threads = {128,128,128,128,128,128};
    auto bar_list= builder.create<deepgen::CreateListOfMBarrierOp>(builder.getUnknownLoc(), threads);
    auto bar = builder.create<deepgen::GetMBarrierOp>(builder.getUnknownLoc(), bar_list ,args[0]);
    auto parity = builder.create<arith::ConstantOp>(builder.getUnknownLoc(), builder.getI1Type(), builder.getBoolAttr(0) );
    builder.create<deepgen::MBarrierWaitParityOp>(builder.getUnknownLoc(), bar, parity );
    builder.create<deepgen::NotifyMBarrierArrivedOp>(builder.getUnknownLoc(), bar);
    
    std::vector<int64_t> steps = {1};
    affine::AffineForOp::BodyBuilderFn loopBuiler = [&](OpBuilder & builder, Location loc, Value v, ValueRange vr)->void 
    {
        // auto r = builder.create<deepgen::WgGemmOp>(builder.getUnknownLoc(), mem,mem,mem,mem );
        auto addi_temp = builder.create<arith::AddIOp>(loc, args[0], args[1]);
    };
    // auto forop = deepgen::createPipelinedForOp(builder,builder.getUnknownLoc(),args[0],args[1],1,nullptr);
    auto ret = builder.create<deepgen::ReturnOp>(builder.getUnknownLoc(), ValueRange({}));
    mod->print(llvm::outs());
    return 0;
}
