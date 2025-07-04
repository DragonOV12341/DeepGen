#include "compiler.h"


namespace DeepGen {

bool DGCompiler::transform(mlir::ModuleOp& mod) {
  // dialect optimize
  mlir::PassManager pm(mod.getContext());
  pm.addPass(createParallelToGPUPass());
  pm.addPass(createCombineMemrefPass());
  pm.addPass(ReplaceAllocToGetglobalPass());
  pm.addPass(createAmendAllocaOpAddrSpacePass(this->target));
  pm.addNestedPass<func::FuncOp>(affine::createAffineLoopInvariantCodeMotionPass());
  pm.addNestedPass<func::FuncOp>(affine::createAffineLoopNormalizePass());
  pm.addPass(createAffineUnrollPass());
  pm.addPass(mlir::createCSEPass());  // 冗余消除
  pm.addPass(mlir::createSymbolDCEPass());  // 死代码消除/化简
  if (mlir::failed(pm.run(mod)))
    return false;
  return true;
}

bool DGCompiler::lowering(mlir::ModuleOp& mod) {
  // lowering
  // == lowering to other dialect ==
  mlir::PassManager pm1(mod.getContext());
  // affine to scf/vector
  pm1.addPass(mlir::createLowerAffinePass());
  pm1.addNestedPass<func::FuncOp>(mlir::createLoopInvariantCodeMotionPass());
  pm1.addPass(mlir::createCanonicalizerPass());         // 代数简化、死代码消除、冗余操作合并
  pm1.addPass(mlir::createCSEPass());                   // 冗余消除
  pm1.addPass(mlir::createSymbolDCEPass());             // 死代码消除/化简
  // scf to cf
  pm1.addPass(mlir::createSCFToControlFlowPass());
  if (mlir::failed(pm1.run(mod)))
    return false;
  
  // == lowering to llvm  ==
  mlir::PassManager pm2(mod.getContext());
  // cf to llvm
  ConvertControlFlowToLLVMPassOptions cfOptions;
  cfOptions.indexBitwidth = INDEX_BIT_WIDTH;
  pm2.addPass(mlir::createConvertControlFlowToLLVMPass(cfOptions));
  // vector to llvm
  pm2.addPass(createVectorToLLVMPass(INDEX_BIT_WIDTH));
  // memref to llvm
  FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
  memrefOptions.indexBitwidth = INDEX_BIT_WIDTH;
  // memrefOptions.useAlignedAlloc = true;
  pm2.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(memrefOptions));
  pm2.addPass(createGlobalShmSetZeroPass());
  // func to llvm
  ConvertFuncToLLVMPassOptions funcOptions;
  funcOptions.indexBitwidth = INDEX_BIT_WIDTH;
  funcOptions.useBarePtrCallConv = true;
  pm2.addPass(mlir::createConvertFuncToLLVMPass(funcOptions));
  pm2.addPass(createLLVMFuncOpAddGPUAttrPass(target));  // llvmfuncOp add nvvm/rocdl.kernel or nvvm.maxnid
  // gpu to rocdl/nvvm
  pm2.addPass(createGPUToROCDLOrNVVMPass(this->target, INDEX_BIT_WIDTH));
  // math to llvm
  pm2.addPass(mlir::createConvertMathToLLVMPass());  // ConvertMathToLLVMPassOptions options.approximateLog1p 精度换性能(true)
  // arith to llvm
  ArithToLLVMConversionPassOptions arithOptions;
  arithOptions.indexBitwidth = INDEX_BIT_WIDTH;
  pm2.addPass(mlir::createArithToLLVMConversionPass(arithOptions));
  // simipfy
  pm2.addPass(mlir::createCanonicalizerPass());
  pm2.addPass(mlir::createCSEPass());
  pm2.addPass(mlir::createSymbolDCEPass());
  if (mlir::failed(pm2.run(mod)))
    return false;
  return true;
}

std::string DGCompiler::translate(mlir::ModuleOp& mod) {
  // mlir -> llvm -> bin
  if (target == Target::ROCm) {
    const int wavesPerEU = 0;
    std::string llvmIR = std::move(translateMLIRToLLVMIR(mod, target, wavesPerEU));
    // llvm::outs() << " =========== after LLVM IR ============\n";
    // llvm::outs() << llvmIR << "\n";
    const std::string gfx_triple{"amdgcn-amd-amdhsa"};
    // const std::string gfx_features{"+code-object-v4"};
    const std::string gfx_features{""};
    return generateAmdgcnAndHsacoFromLLIRFile(llvmIR, "gfx" + arch, gfx_triple, gfx_features);
  } else {
    std::string llvmIR = std::move(translateMLIRToLLVMIR(mod, target));
    // llvm::outs() << " =========== after LLVM IR ============\n";
    // llvm::outs() << llvmIR << "\n";
    // const int capability = CUDA_CAP;
    const int version = 83;
    auto paths = generatePTXAndCubinFromLLIRFile(llvmIR, std::stoi(arch), version);
    return paths.second;
  }
}

}