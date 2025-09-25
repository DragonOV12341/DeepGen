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
    Op<Deepgen_Dialect, mnemonic, traits> {
}

def KernelOp : Deepgen_Op<"kernel",[
    IsolatedFromAbove
]> {
    let summary = "定义KernelFunction ";
    let arguments = (ins 
        I64ArrayAttr:$gridDim,
        I64ArrayAttr:$blockDim,
        Variadic<AnyType>:$arguments
    );

    // 内核的主体，一个包含指令的区域
    let regions = (region SizedRegion<1>);  
    
    /** usage: 
    // 顶层 Op，与 func.func 同级
    deepgen.kernel @my_add_kernel(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>)
        {gridDim = [32, 1, 1], blockDim = [32, 1, 1]} {

        // 内核的 body，这里可以放置你的计算指令
        // 例如，使用 gpu.thread_id 来获取线程ID，进行并行计算
        %x = deepgen.thread_id x
        %val0 = memref.load %arg0[%x] : memref<1024xf32>
        %val1 = memref.load %arg1[%x] : memref<1024xf32>
        %sum = arith.addf %val0, %val1 : f32
        memref.store %sum, %arg2[%x] : memref<1024xf32>
        deepgen.return
    }

    func.func @main() {
        ...
    }
    **/
}

/*
def ReturnOp : Deepgen_Op<"return", [Pure, HasParent<"KernelDefineOp">, /*MemRefsNormalizable, */ReturnLike, Terminator]> {
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

def BlockIdOp : Deepgen_Op<"block_id", [
    NoSideEffects,  // 这个操作没有副作用，因为它只是读取一个值
    Pure,           // 它的结果只依赖于其输入（虽然这里没有输入，但其值是确定的）
    IsolatedFromAbove  // 此op不能从parent区域之外获取ssa值。
]> {
  let summary = "Get the block ID for the current thread block.";
  let description = [{
    The `block_id` operation returns the ID of the current thread block
    in the specified dimension (x, y, or z). The value returned is an index.
  }];
  
  // 属性，用于指定维度
  let attributes = (ins StringEnumAttr<"dimension", ["x", "y", "z"]>:$dimension);
  
  // 返回类型为 index
  let results = (outs Index);
  /** usage:
  deepgen.kernel @mykernel(){gridDim = [32, 1, 1], blockDim = [32, 1, 1]} {
    %blockIdX = deepgen.block_id x
  }
  **/

}

def ThreadIdOp : Deepgen_Op<"thread_id", [
    NoSideEffects,  // 这个操作没有副作用，因为它只是读取一个值
    Pure,           // 它的结果只依赖于其输入（虽然这里没有输入，但其值是确定的）
    IsolatedFromAbove  // 此op不能从parent区域之外获取ssa值。
]> {
  let summary = "Get the thread ID for the current thread block.";
  let description = [{
    The `thread_id` operation returns the ID of the current thread block
    in the specified dimension (x, y, or z). The value returned is an index.
  }];
  
  // 属性，用于指定维度
  let attributes = ( StringEnumAttr<"dimension", ["x", "y", "z"]>:$dimension);
  
  // 返回类型为 index
  let results = (outs Index);
  /** usage:
  deepgen.kernel @mykernel(){gridDim = [32, 1, 1], blockDim = [32, 1, 1]} {
    %blockIdX = deepgen.thread_id x
  }
  **/

}



def AllocSharedOp : Deepgen_Op<"alloc_shared", [
    DeclareOpInterfaceMethods<MemoryEffectOpInterface>,
    MemoryEffects<[Allocate]>
]> {
  let summary = "Allocates a memory region on GPU shared memory";
  let description = [{
    The `deepgen.alloc_shared` operation allocates a memory region on the GPU's
    shared memory. The dimensions of the allocated memory are specified by
    the operands. This operation is typically used inside a GPU launch.
  }];

  // 输入是可变参数的 index，表示 memref 的维度
  let arguments = (ins
    Type:$elem_type,          // 新增的操作数，表示元素类型
    Variadic<IntegerOrIndexType>:$dynamic_dims
  );

  // 返回类型是一个 memref
  let results = (outs AnyMemRef:$memref_result);
}

def AllocLocalOp : Deepgen_Op<"alloc_local",[]> {
  let summary = "Allocates a memory region on GPU local memory";
  let description = [{
    The `deepgen.alloc_local` operation allocates a memory region on the GPU's
    thread local memory. The dimensions of the allocated memory are specified by
    the operands. This operation is typically used inside a GPU launch.
  }];

  // 输入是可变参数的 index，表示 memref 的维度
  let arguments = (ins
    Type:$elem_type,          // 元素类型
    Variadic<IntegerOrIndexType>:$dynamic_dims  // size(任意维度)
  );

  // 返回类型是一个 memref
  let results = (outs AnyMemRef:$memref_result);
}

// 定义一个 setLayout 操作
def SetLayoutOp : Op<"set_layout",
    [{
      // This op modifies the layout of a memref without changing its logical shape.
      // It returns a new memref with the specified layout.
    }]> {

  // 输入操作数: 一个 memref
  let arguments = (ins 
    AnyMemRef:$input
  );

  // 输入属性: 一个枚举值，表示布局模式
  let attributes = ( 
    LayoutAttr:$layout_mode
  );

  // 输出结果: 一个新的 memref，具有与输入相同的类型
  let results = (
    AnyMemRef
  );

  // Traits:
  // 1. SameOperandsAndResultShape: 确保输入和输出的memref形状相同。
  // 2. SameOperandsAndResultElementType: 确保输入和输出的元素类型相同。
  let traits = [
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType,
    NoSideEffects
  ];

  // 验证器
  let verifier = "return verifySetLayoutOp(*this);";
}

def FillOp : Deepgen_Op<"fill",[
    DeclareOpInterfaceMethods<MemoryEffectOpInterface>
]> {
    let summary = "fill value to memref buffer";
    let arguments = (ins
        AnyType:$fill_value,         // 新增的操作数，表示填充值(0 也包括其中，故不设置 clearop)
        AnyMemRef:$memref,
    );
    let has = [
        MemoryEffects<[Write<"memref">]>
    ];
}

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

*/
#endif // Deepgen_OPS
