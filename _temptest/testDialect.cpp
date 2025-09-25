#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"
#include "Dialect/Deepgen/IR/Dialect.h"
#include <mlir/IR/Attributes.h>

using namespace mlir;
using namespace mlir::deepgen;

int main(int argc, char ** argv) {
  MLIRContext ctx;
  ctx.loadDialect<func::FuncDialect, arith::ArithDialect, mlir::deepgen::DeepgenDialect>();

  // 创建 OpBuilder
  OpBuilder builder(&ctx);
  auto mod = builder.create<ModuleOp>(builder.getUnknownLoc());

  // 设置插入点
  builder.setInsertionPointToEnd(mod.getBody());
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ArrayAttr gridDim, ::mlir::ArrayAttr blockDim, ::mlir::ValueRange arguments);

 
    // 创建 gridDim 和 blockDim 的整数数组属性
    // [32, 1, 1]
    mlir::SmallVector<mlir::Attribute, 3> gridDimValues;
    gridDimValues.push_back(builder.getI32IntegerAttr(32));
    gridDimValues.push_back(builder.getI32IntegerAttr(1));
    gridDimValues.push_back(builder.getI32IntegerAttr(1));
    mlir::ArrayAttr gridDimAttr = builder.getArrayAttr(gridDimValues);

    // [128, 32, 1]
    mlir::SmallVector<mlir::Attribute, 3> blockDimValues;
    blockDimValues.push_back(builder.getI32IntegerAttr(128));
    blockDimValues.push_back(builder.getI32IntegerAttr(32));
    blockDimValues.push_back(builder.getI32IntegerAttr(1));
    mlir::ArrayAttr blockDimAttr = builder.getArrayAttr(blockDimValues);

    // 调用 builder.create<KernelOp> 来构建操作
    // 注意：这里的 KernelOp 是一个占位符，您需要根据您的Dialect和操作类型进行替换
    mlir::Operation* kernelOp = builder.create<mlir::deepgen::KernelOp>(
        builder.getUnknownLoc(),     // location
        gridDimAttr,
        blockDimAttr,
        mlir::ValueRange()
    );
    mod.dump();

}
