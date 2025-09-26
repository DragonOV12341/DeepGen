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

using namespace mlir;
using namespace mlir::deepgen;

int main(int argc, char **argv)
{
    MLIRContext ctx;
    ctx.loadDialect<func::FuncDialect, arith::ArithDialect, deepgen::DeepgenDialect, memref::MemRefDialect>();

    // 创建 OpBuilder
    OpBuilder builder(&ctx);
    auto mod = builder.create<ModuleOp>(builder.getUnknownLoc());

    // 设置插入点
    builder.setInsertionPointToEnd(mod.getBody());

    // 创建 func
    auto i32 = builder.getI32Type();
    std::vector<int64_t> shapeA = {32, 32};
    std::vector<int64_t> griddim = {1,1,1};
    std::vector<int64_t> blockdim = {128,1,1};
    auto memrefTypeA = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shapeA), i32);
    

    mlir::TypeRange intypes = llvm::ArrayRef<Type>({i32,i32,memrefTypeA});
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
    // 创建 func.return
    builder.create<deepgen::SetLayoutOp>(builder.getUnknownLoc(), mem->getResult(0).getType(), mem->getResult(0), deepgen::LayoutKindAttr::get(&ctx, deepgen::LayoutKind::Normal));
    auto ret = builder.create<deepgen::ReturnOp>(builder.getUnknownLoc(), ValueRange({}));
    mod->print(llvm::outs());
    return 0;
}
