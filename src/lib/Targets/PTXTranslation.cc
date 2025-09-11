#include "config.h"
#include "Targets/Translation.h"

using namespace mlir;
namespace DeepGen {

static void initLLVM() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}

static bool findAndReplace(std::string &str, const std::string &begin, const std::string &end, const std::string &target) {
  size_t startReplace = str.find(begin);
  if (startReplace == std::string::npos)
    return false;
  size_t endReplace = str.find(end, startReplace);
  if (endReplace == std::string::npos)
    return false;
  str.replace(startReplace, endReplace + 1 - startReplace, target);
  return true;
}

std::string translateLLVMIRToPTX(llvm::Module &module, int cc, int version) {
  // LLVM version in use may not officially support target hardware.
  // Supported versions for LLVM 14 are here:
  // https://github.com/llvm/llvm-project/blob/f28c006a5895fc0e329fe15fead81e37457cb1d1/clang/include/clang/Basic/BuiltinsNVPTX.def
  int maxPTX = std::min(80, version);
  int maxCC = std::min(90, cc);
  // options
  auto options = llvm::cl::getRegisteredOptions();
  auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options["nvptx-short-ptr"]);
  assert(shortPtr);
  shortPtr->setValue(true);
  std::string sm = cc == 90 ? "sm_90a" : "sm_" + std::to_string(cc);
  // max PTX version
  int ptxMajor = maxPTX / 10;
  int ptxMinor = maxPTX % 10;
  // create
  std::string triple = "nvptx64-nvidia-cuda";
  std::string proc = sm;
  std::string layout = "";
  std::string features = "";
  
  // std::string features = "+ptx" + std::to_string(maxPTX);
  for (llvm::Function &f : module.functions()) {
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  }
  initLLVM();
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());
  pm.run(module);
  
  // module.print(llvm::outs(), nullptr);

  // create machine
  llvm::Triple tr;
  tr.setArch(llvm::Triple::ArchType::nvptx64);
  tr.setVendor(llvm::Triple::VendorType::NVIDIA);
  tr.setOS(llvm::Triple::OSType::CUDA);
  tr.setEnvironment(llvm::Triple::EnvironmentType::GNU);
  module.setTargetTriple(tr);
  
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  llvm::TargetOptions opt;
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  llvm::TargetMachine *machine = target->createTargetMachine(
      module.getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive);
  // set data layout
  if (layout.empty())
    module.setDataLayout(machine->createDataLayout());
  else
    module.setDataLayout(layout);
  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    for (llvm::Function &f : module.functions())
      f.addFnAttr(llvm::Attribute::AlwaysInline);
    llvm::legacy::PassManager pass;
    // emit
    machine->addPassesToEmitFile(pass, pstream, nullptr, llvm::CodeGenFileType::AssemblyFile);
    pass.run(module);
  }
  // post-process
  findAndReplace(result, ".version", "\n", ".version " + std::to_string(ptxMajor) + "." +
                 std::to_string(ptxMinor) + "\n");
  findAndReplace(result, ".target", "\n", ".target " + sm + "\n");
  while (findAndReplace(result, "\t// begin inline asm", "\n", ""))
    ;
  while (findAndReplace(result, "\t// end inline asm", "\n", ""))
    ;
  return result;
}

std::string translate_llvmir_to_ptx(const std::string llvmIR, int capability, int version) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> module = llvm::parseIR(buffer->getMemBufferRef(), error, context);
  auto ptxCode = translateLLVMIRToPTX(*module, capability, version);

  // save ptx code

  llvm::SmallString<64> fsrc;
  llvm::sys::fs::createTemporaryFile("compile-ptx-src", "", fsrc);
  std::string ptxSrc = std::string(fsrc) + ".ptx";
  std::ofstream ofs(ptxSrc);
  ofs << ptxCode << std::endl;
  ofs.close();

#ifdef KCG_DEBUG
  std::string ptxPath = PTX_DUMP_PATH;
  std::ofstream ofss(ptxPath);
  ofss << ptxCode << std::endl;
  ofss.close();
#endif
  // std::cout << "==== ptx code: \n" << ptxCode << "\n";
  return ptxSrc;
}

std::string compile_ptx_to_cubin(const std::string &ptxPath, const std::string &ptxasPath, int capability) {
  // compile ptx with ptxas
  llvm::SmallString<64> flog;
  llvm::sys::fs::createTemporaryFile("compile-ptx-log", "", flog);
  auto name =  ptxPath.substr(0, ptxPath.size()-3);
  std::string fbin = std::string(name) + "cubin";
  // llvm::FileRemover logRemover(flog);
  // llvm::FileRemover binRemover(fbin);
  const char *_flog = flog.c_str();
  std::string cmd;
  int err;
  cmd = ptxasPath + " -v --gpu-name=sm_" + std::to_string(capability) + (capability == 90 ? "a " : " ") +
        ptxPath + " -o " + fbin + " 2> " + _flog;
  // llvm::outs() << cmd << "\n";
  err = system(cmd.c_str());
  if (err != 0) {
    err >>= 8;
    std::ifstream _log(_flog);
    std::string log(std::istreambuf_iterator<char>(_log), {});
    if (err == 255) {
      throw std::runtime_error("Internal DeepGen PTX codegen error: \n" + log);
    } else if (err == 128 + SIGSEGV) {
      throw std::runtime_error("Please run `ptxas " + ptxPath + "` to confirm that this is a bug in `ptxas`\n" + log);
    } else {
      throw std::runtime_error("`ptxas` failed with error code " + std::to_string(err) + ": \n" + log);
    }
    return "";
  }
  if (remove(ptxPath.c_str()) == 0){
#ifdef KCG_DEBUG
    std::cout << "file deleted : " << ptxPath << std::endl;
#endif
  }
  else{
    perror("file del error"); // 打印错误信息
  }
  return fbin;

}

std::pair<std::string, std::string> generatePTXAndCubinFromLLIRFile(const std::string llvmIR, int capability, int version) {
  std::string ptxasPath = USER_PTXAS_PATH;
  // std::string ptxPath = "/home/xiebaokang/projects/mymlir/DeepGen/_tmp/test.ptx";
  if (!std::filesystem::exists(ptxasPath)) {
    std::cout << "[FatalError] ptxas not found : " << ptxasPath << std::endl;
    std::abort();
  }
  std::string ptxPath = translate_llvmir_to_ptx(llvmIR, capability, version);
  std::string cabinPath = compile_ptx_to_cubin(ptxPath, ptxasPath, capability);
  return std::make_pair(ptxPath, cabinPath);
}

}