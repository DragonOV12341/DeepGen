#ifndef Deepgen_OPS
#define Deepgen_OPS

include "Dialect/Deepgen/IR/DeepgenDialect.td"
include "Dialect/Deepgen/IR/DeepgenTypes.td"
include "Dialect/Deepgen/IR/DeepgenAttrDefs.td"
include "Dialect/Deepgen/IR/DeepgenInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/Interfaces/CastInterfaces.td" // CastOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/CastInterfaces.td" // CastOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface

//
// Op Base
//
class Deepgen_Op<string mnemonic, list<Trait> traits = []> :
    Op<Deepgen_Dialect, mnemonic, !listconcat(traits, [])> {
}

def KernelFuncOp : Deepgen_Op<"kernel_func",[
  AffineScope, IsolatedFromAbove, OpAsmOpInterface, FunctionOpInterface, CallableOpInterface
]> {
    let summary = "定义KernelFunction ";

    let arguments = (ins 
      SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      DenseI64ArrayAttr : $griddim,
      DenseI64ArrayAttr : $blockdim,
      OptionalAttr<StrAttr>:$sym_visibility,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
      "StringRef":$name, "FunctionType":$type,
      "DenseI64ArrayAttr" : $griddimAttr,
      "DenseI64ArrayAttr" : $blockdimAttr,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
    >];
    let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }

    static LogicalResult verifyInvariants(KernelFuncOp op) {
        // 验证 griddim 属性 (如果存在) 必须是长度为 3 的数组
        if (op.getGriddim().size() != 3){
          return op.emitOpError("griddim attribute must be an array of length 3");
        }
        // 验证 blockdim 属性 (如果存在) 必须是长度为 3 的数组
        if (op.getBlockdim().size() != 3){
          return op.emitOpError("blockdim attribute must be an array of length 3");
        }
      return success();
    }
  }];

    let hasCustomAssemblyFormat = 1;
}


def ReturnOp : Deepgen_Op<"return", [Pure, HasParent<"KernelFuncOp">, /*MemRefsNormalizable, */ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `tt.return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    deepgen.kernel @foo() : (i32, f8) {
      ...
      deepgen.return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}


class GPU_IndexOp<string mnemonic, list<Trait> traits = []> :
    Deepgen_Op<mnemonic, !listconcat(traits, [
        Pure,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>])>,
    Arguments<(ins DimAxisAttr:$dimension,
                   OptionalAttr<IndexAttr>:$upper_bound)>, Results<(outs Index)> {
  let assemblyFormat = "$dimension (`upper_bound` $upper_bound^)? attr-dict";
  let extraClassDefinition = [{
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
      auto dimStr = stringifyDimension(getDimensionAttr().getValue());
      auto opName = getOperationName();
      opName.consume_front("deepgen.");
      SmallString<8> resultName({opName, dimStr});
      setNameFn(getResult(),resultName);
    }
  }];
  let builders = [
    OpBuilder<(ins "::mlir::deepgen::Dimension":$dimension), [{
      build($_builder, $_state, dimension, /*upperBound=*/nullptr);
    }]>,
    OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::deepgen::Dimension":$dimension), [{
      build($_builder, $_state, resultType, dimension, /*upperBound=*/nullptr);
    }]>
  ];
}

def BlockIdOp : GPU_IndexOp<"blockId"> {
  let description = [{
    Returns the block id, i.e. the index of the current block within the grid
    along the x, y, or z `dimension`.

    Example:

    ```mlir
    %bIdY = deepgen.block_id y
    ```

    If `upper_bound` is set, or if one can be inferred from `known_grid_size`-type
    annotations in context, executions where the block index in `dimension` would
    be greater than or equal to that bound cause undefined behavior. `upper_bound`
    takes priority over bounds inferrable from context.

    There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).
  }];
}

def ThreadIdOp : GPU_IndexOp<"threadId"> {
  let description = [{
    Returns the thread id, i.e. the index of the current thread within the block
    along the x, y, or z `dimension`.

    Example:

    ```mlir
    %tIdX = deepgen.thread_id x
    ```

    If `upper_bound` is set, or if one can be inferred from `known_block_size`-type
    annotations in context, executions where the thread index would be greater
    than or equal to that bound cause undefined behavior.

    There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).
  }];
}




// def AllocSharedOp : Deepgen_Op<"alloc_shared", [
//     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
// ]> {
//   let summary = "Allocates a memory region on GPU shared memory";
//   let description = [{
//     The `deepgen.alloc_shared` operation allocates a memory region on the GPU's
//     shared memory. The dimensions of the allocated memory are specified by
//     the operands. This operation is typically used inside a GPU launch.
//   }];

//   // 输入是可变参数的 index，表示 memref 的维度
//   let arguments = (ins
//     "Type":$elem_type,          // 新增的操作数，表示元素类型
//     Variadic<Index>:$dynamicSizes
//   );

//   // 返回类型是一个 memref
//   let results = (outs AnyMemRef:$memref_result);
//   let assemblyFormat = [{
//     `(`$dynamicSizes`)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
//   }];
// }



// def AllocLocalOp : Deepgen_Op<"alloc_local",[]> {
//   let summary = "Allocates a memory region on GPU local memory";
//   let description = [{
//     The `deepgen.alloc_local` operation allocates a memory region on the GPU's
//     thread local memory. The dimensions of the allocated memory are specified by
//     the operands. This operation is typically used inside a GPU launch.
//   }];

//   // 输入是可变参数的 index，表示 memref 的维度
//   let arguments = (ins
//     Type:$elem_type,          // 元素类型
//     Variadic<IntegerOrIndexType>:$dynamic_dims  // size(任意维度)
//   );

//   // 返回类型是一个 memref
//   let results = (outs AnyMemRef:$memref_result);
// }



// 定义一个 setLayout 操作
def SetLayoutOp : Deepgen_Op<"set_layout", [
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType,
    Pure]> {

  // 输入操作数: 一个 memref
  let arguments = (ins AnyRankedOrUnrankedMemRef:$inputMemref, LayoutAttr:$layout);

  // 输出结果: 一个新的 memref，具有与输入相同的类型
  let results = (outs 
    AnyRankedOrUnrankedMemRef:$memref
  );

  let assemblyFormat = "$inputMemref type($inputMemref) attr-dict `:` type($memref) `with` $layout";

  let builders = [
    OpBuilder<(ins "Value":$inputMemref, "::mlir::deepgen::LayoutKind":$layout), [{
      build($_builder, $_state, inputMemref, layout );
    }]>
  ];

}


def FillOp : Deepgen_Op<"fill",[
]> {
    let summary = "fill value to memref buffer";
    let arguments = (ins
        AnyTypeOf<[I32,I64,F32,F64]> : $fill_value,         // 新增的操作数，表示填充值(0 也包括其中，故不设置 clearop)
        Arg<AnyRankedOrUnrankedMemRef, "the memref to fill", [MemWriteAt<0, FullEffect>]>:$target
    ); 
    // let assemblyFormat = "$target `,` $fill_value `(` attr-dict `)` `:` type($target)";
    let assemblyFormat = "$fill_value type($fill_value) `to` $target  attr-dict `:` type($target)";
}
/****
// def LoadOp : Deepgen_Op<"load",[
//     DeclareOpInterfaceMethods<MemoryEffectOpInterface>
// ]> {
//     let summary = "load src data to dest ";  // 同步或者异步
//     let arguments = (ins
//         AnyMemRef:$memref,
//         Variadic<Index>:$indices
//     );
//     let results = (outs AnyType:$result);
//     let has  = [
//         MemoryEffects<[Read<"memref">]>
//     ];
// }

// def StoreOp : Deepgen_Op<"store",[    
//     DeclareOpInterfaceMethods<MemoryEffectOpInterface>
// ]> {
//     let summary = "store data to dest ";  // 同步或者异步
//     let arguments = (ins
//         AnyType:$value,
//         AnyMemRef:$memref,
//         Variadic<Index>:$indices
//     );
//     let has = [
//         MemoryEffects<[Write<"memref">]>
//     ];
// }

def CopyOp : Deepgen_Op<"copy",[
    DeclareOpInterfaceMethods<MemoryEffectOpInterface>
]> {
    let summary = "copy src data to dest ";  // 同步或者异步
    let arguments = (ins
        AnyMemRef:$memref_src,
        Variadic<Index>:$indices_src
        AnyMemRef:$memref_dst,
        Variadic<Index>:$indices_dst,
        Index:$num_bytes        // 表示要拷贝的字节数
    );
    let attributes = (ins
        DefaultValuedAttr<BoolAttr, "false">:$isAsync  //  同步或者异步
    );
    let has  = [
        MemoryEffects<[Read<"memref_src">, Write<"memref_dst">]>
    ];
}


def WgGemmOp : GPU_Op<"wg_gemm", [
    DeclareOpInterfaceMethods<MemoryEffectOpInterface>
]> {
  let summary = "Performs a warpgroup-level matrix-matrix multiply-accumulate.";
  let description = [{
    The `gpu.wgmma_gemm` operation performs a WGMMA matrix multiply-accumulate
    operation using Tensor Cores. It operates on matrix fragments loaded from
    memrefs. The dimensions and types are specified by the op's attributes.

    The operation computes: `D = C + A * B`
  }];

  let arguments = (ins
    AnyMemRef:$opA,
    AnyMemRef:$opB,
    AnyMemRef:$opC,
    AnyMemRef:$opD
  );
  
  let attributes = (ins
    // I64ArrayAttr:$matrix_shape,  // E.g., [16, 16, 16] for 16x16x16 shape
    I64Attr:$m_dim,
    I64Attr:$n_dim,
    I64Attr:$k_dim,
    StringAttr:$layoutA,         // "row_major" or "col_major"
    StringAttr:$layoutB          // "row_major" or "col_major"
  );

  let has = [
    MemoryEffects<[Read<"opA">, Read<"opB">, Read<"opC">, Write<"opD">]>
  ];
}

def ParallelForOp : Deepgen_Op<"parallel_for",[]> {
    let summary = "可以安全拆分到多个axis上并行的并行for loop （类似 gpu.parallel）";
    let arguments = (ins
      Variadic<IntegerOrIndexType>:$upperBounds
    )
    let results = (
      /* no results for this loop op */
    );
    // Regions: The loop body
    let regions = (
      // The loop body region. It should accept one `index` operand for each dimension.
      Region<1>:$body
    );

    // The traits of the op.
    let traits = [
      NoSideEffects
    ];
}

def PersistentForOp : Deepgen_Op<"persist_for",[]> {
    let summary = "一组线程处理大量数据时，需要多个wave完成这个任务。单个线程在每个wave中会映射到不同的位置load/store/calculate数据 （即：建立一个for w : wave）";
}

def PipelinedForOp : Deepgen_Op<"pipelined_for",[]> {
    let summary = "串行for loop, 可以进行 softpipeline优化。会带有一些attr指示后续如何优化。本身只是产出一个简单for";
    let arguments = (ins
      Variadic<IntegerOrIndexType>:$lowerBounds,
      Variadic<IntegerOrIndexType>:$upperBounds,
      Variadic<IntegerOrIndexType>:$steps
    )
    let attributes = (ins
      I32:$num_stages
    );
    let results = (
      /* no results for this loop op */
    );
    // Regions: The loop body
    let regions = (
      // The loop body region. It should accept one `index` operand for each dimension.
      Region<1>:$body
    );

    // The traits of the op.
    let traits = [
      NoSideEffects
    ];
}

def WgSpecializedOp : Deepgen_Op<"warpgroup_spec",[
  {
      // This op specifies that a task is to be executed by a single
      // warp group with a specific ID.
      // A warp group is a group of 128 threads.
  }
]> {
    let summary = "128个thread组成一个 warpgroup. 该op指定某个warpgroup的所有线程完成特定任务 ： 即 ws(1){do something ... } <=> if(tid>=128 && tid < 128+128){do sth}";
    // 输入操作数：一个整数或索引类型的值，代表 warp group 的 ID
    let arguments = (ins 
      AnyIntegerOrIndexType:$warp_group_id
    );

    // 操作结果：这个 op 本身不产生返回值，主要用于其副作用（执行代码）
    let results = (
      /* no results */
    );

    // 区域：用于封装要执行的任务
    let regions = (
      // 该区域包含一个基本块，其参数为 warp group 的 ID
      Region<1>:$body
    );

    // Traits：表示这个操作没有副作用
    let traits = [
      NoSideEffects
    ];
}


def BarrierArriveOp : Deepgen_Op<"barrier_arrive",[]> {
    let summary = "标记barrier到达了 ";
}

def BarrierWaitOp : Deepgen_Op<"barrier_wait",[]> {
    let summary = "等待barrier到达";

}


*****/

#endif // Deepgen_OPS
