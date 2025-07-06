#include "compiler.h"
#include "operator.h"
#include "Python.h"

using namespace DeepGen;

int main() {
  KernelGenerator generator;        // 没有外部导入module
  DGCompiler compiler(Target::CUDA, "90");
  // create dot kernel
  ArgTensor A{{1, 32, 1024, 128}, DType::FLOAT16, ArgType::INPUT, 4, false};
  ArgTensor B{{1, 32, 128, 1024}, DType::FLOAT16, ArgType::INPUT, 4, false};
  ArgTensor C{{1, 32, 1024, 1024}, DType::FLOAT16, ArgType::OUTPUT, 4, false};
  auto dot0 = generator.create<Dot>(A, B, C);  // funcOp
  LOG_DEBUG("============ CPU DOT0\n", dot0);
  // create elementwise kernel
  ArgTensor input{{1, 32, 1024, 1024}, DType::FLOAT16, ArgType::INPUT, 4, false};
  ArgTensor output{{1, 32, 1024, 1024}, DType::FLOAT16, ArgType::OUTPUT, 4, false};
  auto exp0 = generator.create<ElementWise>(input, output, ElementWiseMode::Exp);  // funcOp
  LOG_DEBUG("============ CPU EXP0\n", exp0);
  // get mlir module
  auto module = generator.getModule();  // module
  LOG_DEBUG("============ CPU MODULE\n", module);
  return 0;
}