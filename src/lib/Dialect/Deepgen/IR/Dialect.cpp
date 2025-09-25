#include "Dialect/Deepgen/IR/Dialect.h"
#include "Dialect/Deepgen/IR/Types.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "Dialect/Deepgen/IR/AttrInterfaces.h.inc"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/Transforms/InliningUtils.h"
#include "Dialect/Deepgen/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::deepgen;

//===----------------------------------------------------------------------===//
// TritonDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

// namespace {
// struct DeepgenInlinerInterface : public DialectInlinerInterface {
//   using DialectInlinerInterface::DialectInlinerInterface;

//   bool isLegalToInline(Operation *call, Operation *callable,
//                        bool wouldBeCloned) const final {
//     auto funcOp = dyn_cast<triton::FuncOp>(callable);
//     if (!funcOp)
//       return true;
//     if (funcOp->hasAttr("noinline"))
//       return !funcOp->getAttrOfType<BoolAttr>("noinline").getValue();
//     return true;
//   }

//   bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
//                        IRMapping &valueMapping) const final {
//     return true;
//   }

//   bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
//                        IRMapping &) const final {
//     return true;
//   }
//   //===--------------------------------------------------------------------===//
//   // Transformation Hooks
//   //===--------------------------------------------------------------------===//

//   /// Handle the given inlined terminator by replacing it with a new operation
//   /// as necessary.
//   void handleTerminator(Operation *op, Block *newDest) const final {
//     // Only return needs to be handled here.
//     auto returnOp = dyn_cast<triton::ReturnOp>(op);
//     if (!returnOp)
//       return;

//     // Replace the return with a branch to the dest.
//     OpBuilder builder(op);
//     builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest,
//                                        returnOp.getOperands());
//     op->erase();
//   }

//   /// Handle the given inlined terminator by replacing it with a new operation
//   /// as necessary.
//   void handleTerminator(Operation *op,
//                         ArrayRef<Value> valuesToRepl) const  {
//     // Only return needs to be handled here.
//     auto returnOp = cast<triton::ReturnOp>(op);

//     // Replace the values directly with the return operands.
//     assert(returnOp.getNumOperands() == valuesToRepl.size());
//     for (const auto &it : llvm::enumerate(returnOp.getOperands()))
//       valuesToRepl[it.index()].replaceAllUsesWith(it.value());
//   }
// };
// } // namespace

void DeepgenDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "Dialect/Deepgen/IR/Ops.cpp.inc"
      >();

  // We can also add interface here.
//   addInterfaces<DeepgenInlinerInterface>();
}

Operation *DeepgenDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
