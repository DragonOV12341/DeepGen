#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "Dialect/Deepgen/IR/Dialect.h"
#include "Dialect/Deepgen/IR/Types.h"


#define GET_OP_CLASSES
#include "Dialect/Deepgen/IR/Ops.cpp.inc"

// enum attribute definitions
#include "Dialect/Deepgen/IR/OpsEnums.cpp.inc"

