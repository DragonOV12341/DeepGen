#pragma once
#ifndef _ExprUtils_h_
#define _ExprUtils_h_

#include "Commons/utils.h"

using namespace mlir;
namespace DeepGen {
namespace ExprUtil {

AffineMap linearizeIndex(OpBuilder builder, AffineMap map, int64_t start_idx, const std::vector<int64_t>& shape);

}
}

#endif