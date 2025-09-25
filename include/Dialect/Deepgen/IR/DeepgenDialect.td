#ifndef DEEPGEN_DIALECT
#define DEEPGEN_DIALECT

include "mlir/IR/OpBase.td"

def Deepgen_Dialect : Dialect {
  let name = "deepgen";

  let cppNamespace = "::mlir::deepgen";

  let summary = "The deepgen IR in MLIR";

  let description = [{
    Deepgen Dialect.

    Dependent Dialects:
      * Arith:
        * addf, addi, andi, cmpf, cmpi, divf, fptosi, ...
      * Math:
        * exp, sin, cos, log, ...
      * StructuredControlFlow:
        * for, if, while, yield, condition
      * ControlFlow:
        * br, cond_br
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "math::MathDialect",
    "scf::SCFDialect",
    "cf::ControlFlowDialect"
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

include "Dialect/Deepgen/IR/DeepgenTypes.td"


#endif // TRITON_DIALECT
