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

namespace mlir {
namespace deepgen{


void KernelFuncOp::build(
    ::mlir::OpBuilder &builder, 
    ::mlir::OperationState &state, 
    StringRef name, FunctionType type, 
    DenseI64ArrayAttr griddimAttr, DenseI64ArrayAttr blockdimAttr, 
    ArrayRef<NamedAttribute> attrs,
    ArrayRef<DictionaryAttr> argAttrs  
)
{
    state.addAttribute(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
    state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
    state.addAttribute("griddim",griddimAttr);
    state.addAttribute("blockdim",blockdimAttr);
    state.attributes.append(attrs.begin(), attrs.end());

    state.addRegion();
    if (argAttrs.empty()){
        return;
    }
    assert(type.getNumInputs() == argAttrs.size());
    call_interface_impl::addArgAndResultAttrs(
        builder, state, argAttrs, /*resultAttrs=*/{},
        getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}


ParseResult KernelFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void KernelFuncOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}



// -- ReturnOp --
LogicalResult ReturnOp::verify() {
  auto function = cast<KernelFuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}


}  // end namespace
}  // end namespace