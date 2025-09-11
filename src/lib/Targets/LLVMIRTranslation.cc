#include "Targets/Translation.h"

namespace {
using namespace llvm;

static std::optional<OptimizationLevel> mapToLevel(unsigned optLevel, unsigned sizeLevel) {
  switch (optLevel) {
  case 0:
    return OptimizationLevel::O0;
  case 1:
    return OptimizationLevel::O1;
  case 2:
    switch (sizeLevel) {
    case 0:
      return OptimizationLevel::O2;
    case 1:
      return OptimizationLevel::Os;
    case 2:
      return OptimizationLevel::Oz;
    }
    break;
  case 3:
    return OptimizationLevel::O3;
  }
  return std::nullopt;
}

static std::function<Error(Module *)> makeOptimizingPipeline(unsigned optLevel, unsigned sizeLevel, TargetMachine *targetMachine) {
  return [optLevel, sizeLevel, targetMachine](Module *m) -> Error {
    std::optional<OptimizationLevel> ol = mapToLevel(optLevel, sizeLevel);
    if (!ol) {
      return make_error<StringError>(
          formatv("invalid optimization/size level {0}/{1}", optLevel, sizeLevel).str(),
          inconvertibleErrorCode());
    }
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PipelineTuningOptions tuningOptions;
    tuningOptions.LoopUnrolling = true;
    tuningOptions.LoopInterleaving = true;
    tuningOptions.LoopVectorization = true;
    tuningOptions.SLPVectorization = true;

    PassBuilder pb(targetMachine, tuningOptions);

    std::string pluginFile = getenv("AMDGCN_INSTRUMENTATION_LIB");
    if (!pluginFile.empty()) {
        llvm::errs() << "Adding AMDGCN instrumentation pass to pipeline" << "\n";
        auto passPlugin = llvm::PassPlugin::Load(pluginFile);
        if (!passPlugin) {
                llvm::Error Err = passPlugin.takeError();
                llvm::errs() << "ERROR: " << Err << "\n";
                consumeError(std::move(Err));
        }
        passPlugin->registerPassBuilderCallbacks(pb);
    }

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm;
    pb.registerVectorizerStartEPCallback(
        [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
          fpm.addPass(InstCombinePass());
        });
    mpm.addPass(pb.buildPerModuleDefaultPipeline(*ol));
    mpm.run(*m, mam);
    return Error::success();
  };
}
}




using namespace llvm;
namespace DeepGen {

struct NVVMMetadata {
  llvm::SmallVector<int, 3> maxntid;
  bool isKernel{};
};

// Add the nvvm related metadata to LLVM IR.
static void amendLLVMFunc(llvm::Function *func, const NVVMMetadata &metadata, Target target, const int threadsPerCTA, const int wavesPerEU) {
  auto *module = func->getParent();
  auto &ctx = func->getContext();
  if (!metadata.maxntid.empty()) {
    auto maxntid = llvm::to_vector(llvm::map_range(metadata.maxntid, [&](int value) {
      return llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), llvm::APInt(32, value));
    }));
    SmallVector<llvm::Metadata *> md_args = {llvm::ValueAsMetadata::get(func)};
    if (maxntid.size() > 0) {
      md_args.push_back(llvm::MDString::get(ctx, "maxntidx"));
      md_args.push_back(llvm::ValueAsMetadata::get(maxntid[0]));
    }
    if (maxntid.size() > 1) {
      md_args.push_back(llvm::MDString::get(ctx, "maxntidy"));
      md_args.push_back(llvm::ValueAsMetadata::get(maxntid[1]));
    }
    if (maxntid.size() > 2) {
      md_args.push_back(llvm::MDString::get(ctx, "maxntidz"));
      md_args.push_back(llvm::ValueAsMetadata::get(maxntid[2]));
    }
    module->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(llvm::MDNode::get(ctx, md_args));
  }

  if (metadata.isKernel) {
    switch (target) {
    case Target::CUDA: {
      llvm::Metadata *mdArgs[] = {
        llvm::ValueAsMetadata::get(func), llvm::MDString::get(ctx, "kernel"),
        llvm::ValueAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
      module->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(llvm::MDNode::get(ctx, mdArgs));
    } break;
    case Target::ROCm: {
      func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      func->addFnAttr("amdgpu-flat-work-group-size", "1, " + std::to_string(threadsPerCTA));
      if (wavesPerEU > 0)
        func->addFnAttr("amdgpu-waves-per-eu", std::to_string(wavesPerEU));
      func->addFnAttr("denormal-fp-math-f32", "preserve-sign");
      func->addFnAttr("amdgpu-unsafe-fp-atomics", "true");
      for (unsigned I = 0; I < func->arg_size(); ++I) {
        Argument &Arg = *func->getArg(I);
        // Check for incompatible attributes.
        if (Arg.hasByRefAttr() || Arg.hasNestAttr()) break;
        Arg.addAttr(llvm::Attribute::InReg);
      }
    } break;
    }
  }
}

static void extractNVVMMetadata(mlir::ModuleOp module, llvm::DenseMap<llvm::StringRef, NVVMMetadata> *dic) {
  for (auto op : module.getOps<mlir::LLVM::LLVMFuncOp>()) {
    NVVMMetadata meta;
    bool hasMetadata{};
    // maxntid
    if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("nvvm.maxntid")) {
      llvm::transform(attr.getAsValueRange<mlir::IntegerAttr>(), std::back_inserter(meta.maxntid),
                      [](llvm::APInt value) { return value.getZExtValue(); });
      hasMetadata = true;
    }
    // kernel
    if (op->hasAttr(CUDA_KERNEL) || op->hasAttr(ROCM_KERNEL)) {
      meta.isKernel = true;
      hasMetadata = true;
    }
    if (hasMetadata){
      dic->try_emplace(op.getNameAttr().strref(), std::move(meta));
    }
  }
}

static std::map<std::string, std::string> getExternLibs(mlir::ModuleOp module) {
  std::map<std::string, std::string> externLibs;
  SmallVector<mlir::LLVM::LLVMFuncOp> funcs;
  module.walk([&](mlir::LLVM::LLVMFuncOp func){
    if (func.isExternal())
      funcs.push_back(func); 
  });

  for (auto &func : funcs) {
    if (func.getOperation()->hasAttr("libname")) {
      auto name = mlir::dyn_cast<mlir::StringAttr>(func.getOperation()->getAttr("libname"));
      auto path = mlir::dyn_cast<mlir::StringAttr>(func.getOperation()->getAttr("libpath"));
        if (name) {
          std::string libName = name.str();
          externLibs[libName] = path.str();
        }
      }
    }

  if (module.getOperation()->hasAttr(EXTERN_LIB)) {
    auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(module.getOperation()->getAttr(EXTERN_LIB));
    for (auto &attr : dict) {
      externLibs[attr.getName().strref().trim().str()] = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()).strref().trim().str();
    }
  }

  if (!funcs.empty()) {
    static const std::string libdevice = "libdevice";
        // first search for environmental path
    std::string env_path = getenv("LIBDEVICE_PATH");
    if (!env_path.empty()) {
      externLibs.try_emplace(libdevice, env_path);
      return externLibs;
    }
    namespace fs = std::filesystem;
    static const auto this_library_path = [] {
    Dl_info fileinfo;
    if (dladdr(reinterpret_cast<void *>(&getExternLibs), &fileinfo) == 0) {
      return std::filesystem::path();
    }
    return std::filesystem::path(fileinfo.dli_fname);
    }();
    static const auto runtime_path = this_library_path.parent_path().parent_path()
                                     / "third_party" / "cuda" / "lib" / "libdevice.10.bc";
    if (fs::exists(runtime_path)) {
      externLibs.try_emplace(libdevice, runtime_path.string());
    } else {
      static const auto this_file_path = std::filesystem::path(__FILE__);
      static const auto compiletime_path = this_file_path.parent_path().parent_path().parent_path().parent_path()
                                           / "third_party" / "cuda" / "lib" / "libdevice.10.bc";
      if (!fs::exists(compiletime_path)) {
        std::string error_msg = "Can't find libdevice at neither " + runtime_path.string() + " nor " + compiletime_path.string();
        llvm::report_fatal_error(error_msg.c_str());
      }
      externLibs.try_emplace(libdevice, compiletime_path.string());
    }
  }
  return externLibs;
}

static void linkLibdevice(llvm::Module &module) {
  // std::cout << "linkLibdevice" << std::endl;
  // please check https://llvm.org/docs/NVPTXUsage.html#reflection-parameters
  // this will enable fast math path in libdevice
  // for example, when enable nvvm-reflect-ftz, sqrt.approx.f32 will change to
  // sqrt.approx.ftz.f32
  auto &ctx = module.getContext();
  llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
  llvm::Metadata *mdFour = llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(i32, 4));
  llvm::Metadata *mdName = llvm::MDString::get(ctx, "nvvm-reflect-ftz");
  llvm::Metadata *mdOne = llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(i32, 1));
  llvm::MDNode *reflect = llvm::MDNode::get(ctx, {mdFour, mdName, mdOne});
  module.addModuleFlag(reflect);
}

static bool linkExternLib(llvm::Module &module, llvm::StringRef name, llvm::StringRef path, Target target) {
  llvm::SMDiagnostic err;
  auto &ctx = module.getContext();

  auto extMod = llvm::parseIRFile(path, err, ctx);
  if (!extMod) {
    llvm::errs() << "Failed to load " << path;
    return true;
  }
  extMod->setTargetTriple(module.getTargetTriple());
  extMod->setDataLayout(module.getDataLayout());

  if (llvm::Linker::linkModules(module, std::move(extMod), llvm::Linker::Flags::LinkOnlyNeeded)) {
    llvm::errs() << "Failed to link " << path;
    return true;
  }
  // check if ROCM
  if (Target::CUDA == target) {
    if (name == "libdevice") {
      linkLibdevice(module);
    }
        // else {
        //   assert(false && "unknown extern lib: ");
        // }
  }
  return false;
}


std::string translateMLIRToLLVMIR(mlir::ModuleOp module, Target target, const int wavesPerEU) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::DLTIDialect, mlir::func::FuncDialect>();
  registerAllToLLVMIRTranslations(registry);
  module.getContext()->appendDialectRegistry(registry);
  llvm::outs() << module << "\n";

  llvm::DenseMap<llvm::StringRef, NVVMMetadata> nvvmMetadata;
  extractNVVMMetadata(module, &nvvmMetadata);

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return "";
  }
  // llvm::outs() << *llvmModule << "\n";
  auto externLibs = getExternLibs(module);
  for (auto &lib : externLibs) {
    if (linkExternLib(*llvmModule, lib.first, lib.second, target))
      return nullptr;
  }

  const int threadsPerCTA = Analyzer::getThreadsPerCTA(module);
  for (auto &func : llvmModule->functions()) {
    auto it = nvvmMetadata.find(func.getName());
    if (it != nvvmMetadata.end()) {
      amendLLVMFunc(&func, it->second, target, threadsPerCTA, wavesPerEU);   //wavesPerEU 每个EU上可以运行的wavefront数量
    }
  }
  
  // auto optPipeline = makeOptimizingPipeline(/*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  // if (auto err = optPipeline(llvmModule.get())) {
  //   llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
  //   return "";
  // }  
  std::string str{""};
  llvm::raw_string_ostream os(str);
  llvmModule->print(os, nullptr);
  os.flush();
  return str;
}

}

