#pragma once

#ifndef _enums_h_
#define _enums_h_

namespace DeepGen {

enum class Target {
  CUDA = 0,
  ROCm = 1,
};

enum class MemorySpace {
  global = 1,
  shared = 3,
  // local = 5,
  local = 0,
  constant = 4,
  unallocated = 7,
  inplace = 6,
};

enum class DType {
  FLOAT16 = 2,
  FLOAT32 = 4,
  INT16 = 0,
  INT32 = 1,
  INT64 = 3,
};

enum class ArgType {
  OUTPUT = 0,
  INPUT = 1,
};

enum class ElementWiseMode {
  Exp = 0,
  Exp2 = 1,
};

enum class ReduceMode {
  Sum = 0,
  Max = 1,
};

enum class BinaryMode {
  Add = 0,
  Sub = 1,
  Mul = 2,
  Div = 3,
};

}
#endif