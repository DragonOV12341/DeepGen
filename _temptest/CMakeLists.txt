
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})


add_executable(testdialect testDialect.cpp)
target_link_libraries(
    testdialect
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
  DeepgenIR
  DeepgenIRUtils
)
