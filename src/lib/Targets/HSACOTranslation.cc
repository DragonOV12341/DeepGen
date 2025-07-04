#include "config.h"
#include <filesystem>
#include "Targets/Translation.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace DeepGen {

std::unique_ptr<llvm::TargetMachine>
initialize_module(llvm::Module *module, const std::string &triple,
                    const std::string &proc, const std::string &features)
{
    // verify and store llvm
    llvm::legacy::PassManager pm;
    pm.add(llvm::createVerifierPass());
    pm.run(*module);

    llvm::Triple tr;
    tr.setArch(llvm::Triple::ArchType::amdgcn);
    tr.setVendor(llvm::Triple::VendorType::AMD);
    tr.setOS(llvm::Triple::OSType::AMDHSA);
    tr.setEnvironment(llvm::Triple::EnvironmentType::UnknownEnvironment);
    module->setTargetTriple(tr);

    std::string error;
    auto target =
        llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
    if (target == nullptr)
    {
        llvm::errs() << "LookupTarget fail: " << error << '\n';
        return nullptr;
    }
    llvm::TargetOptions opt;
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    opt.UnsafeFPMath = false;
    opt.NoInfsFPMath = false;
    opt.NoNaNsFPMath = true;
    llvm::TargetMachine *machine = target->createTargetMachine(
        module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
        std::nullopt, llvm::CodeGenOptLevel::Aggressive);

    module->setDataLayout(machine->createDataLayout());

    for (llvm::Function &f : module->functions())
        f.addFnAttr(llvm::Attribute::AlwaysInline);

    return std::unique_ptr<llvm::TargetMachine>(machine);
}

void init_llvm()
{
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
}

std::string generate_amdgcn_assembly(llvm::Module *module,
                                        const std::string &triple,
                                        const std::string &proc,
                                        const std::string &features)
{
  auto machine = initialize_module(module, triple, proc, features);

  if (machine == nullptr) {
    assert(false && "generate_amdgcn_assembly error!");
    return "";
  }

  llvm::SmallVector<char, 0> buffer;
  llvm::legacy::PassManager pass;
  llvm::raw_svector_ostream stream(buffer);

  // emit
  machine->addPassesToEmitFile(pass, stream, nullptr,
                                  llvm::CodeGenFileType::AssemblyFile);
  pass.run(*module);
  std::string amdgcn(buffer.begin(), buffer.end());
  return amdgcn;
}

std::string generate_hsaco(llvm::Module *module, const std::string &triple,
                            const std::string &proc,
                            const std::string &features)
{
  auto machine = initialize_module(module, triple, proc, features);  
  // create unique dir for kernel's binary and hsaco
  std::error_code ec;
  std::string kernel_name_base = "dg_kernel";

  llvm::SmallString<64> fsrc;
  llvm::sys::fs::createTemporaryFile(kernel_name_base, "", fsrc);
  std::string dump_path = std::string(fsrc);

  std::filesystem::path tmp = std::filesystem::temp_directory_path();
  std::filesystem::path kernel_dir_base(kernel_name_base);
  llvm::SmallString<256> unique_dir;
  ec = llvm::sys::fs::createUniqueDirectory((tmp / kernel_dir_base).string(), unique_dir);
  if (ec) {
    std::cerr << "Directory for " << kernel_name_base << " was not created. error code: " << ec << std::endl;
  }
  std::filesystem::path kernel_dir(unique_dir.data());
  std::string kernel_name = kernel_dir.stem();
  // Save GCN ISA binary.
  std::filesystem::path isa_binary(kernel_name + ".o");
  std::string isabin_path;
  if (!dump_path.empty())
    isabin_path = (dump_path / isa_binary).string();
  else
    isabin_path = (kernel_dir / isa_binary).string();
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
  if (ec) {
    llvm::errs() << isabin_path << " was not created. error code: " << ec.category().name() << ':' << ec.value() << '\n';
  }

  // Write out bitcode
  std::filesystem::path bitcode_filename(kernel_name + ".bc");
  std::string bitcode_path;
  if (!dump_path.empty())
    bitcode_path = (dump_path / bitcode_filename).string();
  else
    bitcode_path = (kernel_dir / bitcode_filename).string();
  std::unique_ptr<llvm::raw_fd_ostream> bitecode_fs(new llvm::raw_fd_ostream(bitcode_path, ec, llvm::sys::fs::OF_Text));
  if (ec) {
    llvm::errs() << bitcode_path << " was not created. error code: " << ec.category().name()
                      << ':' << ec.value() << '\n';
  }

  llvm::WriteBitcodeToFile(*module, *bitecode_fs);
  // emit
  llvm::legacy::PassManager pass;
  machine->addPassesToEmitFile(pass, *isabin_fs, nullptr, llvm::CodeGenFileType::ObjectFile);
  pass.run(*module);

  // module->print(llvm::outs(), nullptr);
  // generate HASCO file
  std::filesystem::path hsaco(kernel_name + ".hsaco");
  std::string hsaco_path = (kernel_dir / hsaco).string();
  std::string error_message;

  // Check in triton/third_party/rocm/llvm/bin first.  For whls this will be the
  // correct location. If not found, go back to using ROCM_PATH or /opt/rocm
  static const auto this_library_path = []
  {
    Dl_info fileinfo;
    if (dladdr(reinterpret_cast<void *>(generate_hsaco), &fileinfo) == 0) {
      return std::filesystem::path();
    }
    return std::filesystem::path(fileinfo.dli_fname);
  }();

  std::string lld_path = USER_LLD_PATH;
  if (!std::filesystem::exists(lld_path)) {
    std::string rocm_path = getenv("ROCM_PATH");
    auto ROCM_DEFAULT_DIR = "/opt/dtk";
    lld_path = (rocm_path.empty()) ? ROCM_DEFAULT_DIR : rocm_path;
    lld_path += "/llvm/bin/ld.lld";
    if (!std::filesystem::exists(lld_path)){
      std::cout << "[FatalError] ld.lld not found" << std::endl;
      std::abort();
    }
  }
  int lld_result = llvm::sys::ExecuteAndWait(lld_path,
                                  {lld_path, "-flavor", "gnu",
                                  "-shared", "-o", hsaco_path, isabin_path},
                                  std::nullopt, {}, 0, 0, &error_message);
  if (lld_result) {
    llvm::errs() << "ld.lld execute fail: " << '\n' << error_message << "Code: " << lld_result << '\n';
  }
  isabin_fs->close();
  bitecode_fs->close();
  if (remove(bitcode_path.c_str()) == 0) {
#ifdef KCG_DEBUG
    std::cout << "file deleted: " << bitcode_path << std::endl;
#endif
  } else {
    perror("file deleted error"); // 打印错误信息
  }
  if (remove(isabin_path.c_str()) == 0) {
#ifdef KCG_DEBUG
    std::cout << "file deleted: " << isabin_path << std::endl;
#endif
  } else {
    perror("file deleted error"); // 打印错误信息
  }
  return hsaco_path;
}


std::tuple<std::string, std::string>
llir_to_amdgcn_and_hsaco(llvm::Module *module, std::string gfx_arch, std::string gfx_triple, std::string gfx_features) {
  init_llvm();
  // verify and store llvm
  auto module_obj = llvm::CloneModule(*module);
  if (!module_obj) {
    llvm::errs() << "Error: clonging LLIR failed\n";
  }
  auto amdgcn = generate_amdgcn_assembly(module, gfx_triple, gfx_arch, gfx_features);
  auto hsaco_path = generate_hsaco(module_obj.get(), gfx_triple, gfx_arch, gfx_features);
  return std::make_tuple(amdgcn, hsaco_path);
}


std::tuple<std::string, std::string> translateLLVMIRToHSACO(
  const std::string llvmIR, 
  std::string gfx_arch, 
  std::string gfx_triple, 
  std::string gfx_features) 
{
  llvm::LLVMContext context;
  std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> module = llvm::parseIR(buffer->getMemBufferRef(), error, context);
  auto hsacoCode = llir_to_amdgcn_and_hsaco(module.get(), gfx_arch, gfx_triple, gfx_features);
  return hsacoCode;
}


std::string generateAmdgcnAndHsacoFromLLIRFile(
  const std::string &llvmIR,
  const std::string &gfx_arch,
  const std::string &gfx_triple,
  const std::string &gfx_features) 
{
  auto [amdgcn, hsacoPath] = translateLLVMIRToHSACO(llvmIR, gfx_arch, gfx_triple, gfx_features);
#ifdef KCG_DEBUG
  std::string amdgcnPath = GCN_DUMP_PATH;
  std::ofstream outasm(amdgcnPath);
  outasm << amdgcn << std::endl;
  outasm.close();
#endif
  return hsacoPath;
}

}