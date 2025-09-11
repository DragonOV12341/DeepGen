#pragma once
#ifndef _compiler_h_
#define _compiler_h_

// lowering
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

// mine
#include "Commons/utils.h"
#include "Transforms/Transform.h"
#include "Conversion/Lowering.h"
#include "Targets/Translation.h"

namespace DeepGen {
  
class DGCompiler {
  
  public:
    DGCompiler(Target target, const std::string& arch) : target(target), arch(arch) {}
    DGCompiler(const DGCompiler& other) : DGCompiler(other.target, other.arch) {}
    DGCompiler() {}

    void setPaltform(Target tg, const std::string& ac) {
      this->target = tg;
      this->arch = ac;
    }
  
  private:
    Target target;
    std::string arch;
  
  public:
    bool fuseing(mlir::ModuleOp& mod/*, graph */);  // forop fuse
    bool mapping(mlir::ModuleOp& mod, Config tile_cfg);  // mapping forop to parallelop
    bool optimize(mlir::ModuleOp& mod, Config opt_cfg);  // kernel optimization passes set

    bool transform(mlir::ModuleOp& mod);
    bool lowering(mlir::ModuleOp& mod);
    std::string translate(mlir::ModuleOp& mod);
};

}

#endif