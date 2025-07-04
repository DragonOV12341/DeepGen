#include "compiler.h"
#include "operator.h"
#include "Python.h"

using namespace DeepGen;

int main() {
  KernelGenerator generator;        // 没有外部导入module
  DGCompiler compiler(Target::CUDA, "90");
  // create kernel
  ArgTensor A{{1, 32, 1024, 128}, DType::FLOAT16, ArgType::INPUT, 4, false};
  ArgTensor B{{1, 32, 128, 1024}, DType::FLOAT16, ArgType::INPUT, 4, false};
  ArgTensor C{{1, 32, 1024, 1024}, DType::FLOAT16, ArgType::OUTPUT, 4, false};
  auto module = generator.create<Dot>(A, B, C);
  LOG_DEBUG("Dot\n", module);
  return 0;
}