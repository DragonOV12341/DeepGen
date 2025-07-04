#pragma once

#ifndef _defines_h_
#define _defines_h_

namespace DeepGen {

// func attrs
#define FUNC_STATE         "func.state"
#define FUNC_OUTPUT_NUM    "func.output.num"
#define FUNC_PARA_DIM      "func.para.dim"
#define FUNC_ARG_TRAN      "func.arg.tran"
#define FUNC_KERNEL_TYPE   "func.kernel.type"
#define FUNC_GRID_DIM      "func.grid.dim"
#define FUNC_BLOCK_DIM     "func.block.dim"


// for attrs
#define FOR_DESC         "for.desc"
#define FOR_PARA_DESC    "for.para.desc"

// lowering
#define ROCM_KERNEL          "rocdl.kernel"
#define CUDA_KERNEL          "nvvm.kernel"
#define RANGE                "range"
#define GPU_INDEX            "gpu.index"
#define AFFINE_LOOP          "affine.loop"
#define AFFINE_UNROLL_NUM    "affine.unroll.num"
#define BUF_DESC             "buf.desc"
#define EXTERN_LIB           "extern.lib"

// value
#define BATCH            "batch"
#define PARALLEL         "parallel"
#define REDUCE           "reduce"
#define BLOCKIDX         "blockIdx"
#define THREADIDX        "threadIdx"

// kernel type
#define KERNEL_DOT            "Dot"
#define KERNEL_ELEMENTWISE    "Elementwise"
#define KERNEL_BINARY         "Binary"
#define KERNEL_REDUCE         "Reduce"
#define KERNEL_BROADCAST      "Broadcast"

// hong
#define BUF_ALIGN_16B        16
#define INDEX_BIT_WIDTH      32

}
#endif