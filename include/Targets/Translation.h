#pragma once
#ifndef _Translation_h_
#define _Translation_h_

#include <dlfcn.h>
#include <random>
#include <iterator>
#include <cstdio>
#include <signal.h>
#include <exception>
#include <filesystem>
#include <cstdlib>
#include <optional>
#include <string>
#include <sstream>
#include <fstream>
#include <initializer_list>
#include <climits>
#include <cfloat>
#include <mutex>
#include <stack>
#include <unordered_map>

#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"

#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"

#include "Commons/utils.h"
#include "Analysis/Analyzer.h"

namespace DeepGen {

std::string generateAmdgcnAndHsacoFromLLIRFile(
        const std::string& llir,
        const std::string& gfx_arch,
        const std::string& gfx_triple,
        const std::string& gfx_features);

std::tuple<std::string, std::string> translateLLVMIRToHSACO(
        const std::string llvmIR, 
        std::string gfx_arch, 
        std::string gfx_triple, 
        std::string gfx_features);

std::pair<std::string, std::string> generatePTXAndCubinFromLLIRFile(const std::string llvmIR, int capability, int version);

std::string translateMLIRToLLVMIR(mlir::ModuleOp module, Target target, const int wavesPerEU=0);

}
#endif