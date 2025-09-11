#pragma once
#ifndef _Analyzer_h_
#define _Analyzer_h_

#include "Commons/utils.h"

namespace DeepGen {
namespace Analyzer {

// 根据func属性获取cta的thread个数
int getThreadsPerCTA(mlir::ModuleOp module);

// 根据属性获取operation
llvm::SmallVector<mlir::Operation*> 
getOperationByAttr(mlir::Operation *outer_op, const std::string& key, const std::string& val);

// 根据判断并行的affinefor所对应的维度

}
}
#endif