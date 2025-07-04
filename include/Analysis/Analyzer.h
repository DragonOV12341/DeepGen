#pragma once
#ifndef _Analyzer_h_
#define _Analyzer_h_

#include "Commons/utils.h"

namespace DeepGen {
namespace Analyzer {

int getThreadsPerCTA(mlir::ModuleOp module);

}
}
#endif