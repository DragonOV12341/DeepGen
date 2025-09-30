add_mlir_dialect_library(DeepgenIRUtils
    Utils.cpp

  ADDITIONAL_HEADER_DIRS
  Dialect/Deepgen/IR

  DEPENDS
  DeepgenInc

  LINK_LIBS PUBLIC
  MLIRAffineDialect
)

