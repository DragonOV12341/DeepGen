#include "Dialect/Deepgen/IR/Types.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "Dialect/Deepgen/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::deepgen;

#define GET_TYPEDEF_CLASSES
#include "Dialect/Deepgen/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void DeepgenDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Deepgen/IR/Types.cpp.inc"
      >();
}

Type PointerType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  Type pointeeType;
  if (parser.parseType(pointeeType))
    return Type();

  int addressSpace = 1;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(addressSpace))
      return Type();
  }

  if (parser.parseGreater())
    return Type();

  return PointerType::get(pointeeType, addressSpace);
}

void PointerType::print(AsmPrinter &printer) const {
  printer << "<" << getPointeeType() << ", " << getAddressSpace() << ">";
}

namespace mlir {

namespace deepgen {

unsigned getPointeeBitWidth(Type type) {
  auto pointeeType = getPointeeType(type);
  if (auto tensorTy = mlir::dyn_cast<RankedTensorType>( pointeeType))
    return tensorTy.getElementType().getIntOrFloatBitWidth();
  return pointeeType.getIntOrFloatBitWidth();
}

Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorTy = mlir::dyn_cast<RankedTensorType>(type) )
    return RankedTensorType::get(tensorTy.getShape(), i1Type,
                                 tensorTy.getEncoding());
  return i1Type;
}

Type getPointeeType(Type type) {
  if (auto tensorTy = mlir::dyn_cast<RankedTensorType>(type)) {
    // Tensor of pointers
    auto shape = tensorTy.getShape();
    auto ptrType = mlir::dyn_cast<PointerType>( tensorTy.getElementType()) ;
    Type pointeeType = ptrType.getPointeeType();
    return RankedTensorType::get(shape, pointeeType, tensorTy.getEncoding());
  } else if (auto ptrType = mlir::dyn_cast<PointerType>(type)) {
    // scalar pointer
    Type pointeeType = ptrType.getPointeeType();
    return pointeeType;
  }
  return type;
}

Type getI32SameShape(Type type) {
  auto i32Type = IntegerType::get(type.getContext(), 32);
  if (auto tensorTy = mlir::dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(tensorTy.getShape(), i32Type,
                                 tensorTy.getEncoding());
  return i32Type;
}

Type getPointerTypeSameShape(Type type) {
  if (auto tensorTy = mlir::dyn_cast<RankedTensorType>(type)) {
    Type elementType = tensorTy.getElementType();
    auto shape = tensorTy.getShape();
    PointerType ptrType = PointerType::get(elementType, 1);
    return RankedTensorType::get(shape, ptrType, tensorTy.getEncoding());
  } else {
    return PointerType::get(type, 1);
  }
}

Type getPointerType(Type type) { return PointerType::get(type, 1); }

bool isTensorPointerType(Type type) {
  if (auto ptrType = mlir::dyn_cast<PointerType>(type))
    return mlir::isa<RankedTensorType>(ptrType.getPointeeType());
  return false;
}

bool isTensorOrTensorPointerType(Type type) {
  return mlir::isa<RankedTensorType>(type) || isTensorPointerType(type);
}

Type getElementTypeOfTensorPointerType(Type type) {
  if (auto ptrType = mlir::dyn_cast<PointerType>(type))
    if (auto tensorTy = mlir::dyn_cast<RankedTensorType>(ptrType.getPointeeType()))
      return tensorTy.getElementType();
  return {};
}

} // namespace triton

} // namespace mlir
