#pragma once
#ifndef _operator_h_
#define _operator_h_

#include "Commons/utils.h"
#include "Transforms/LoopUtils.h"

namespace DeepGen {

struct ArgTensor {
  // tensor data struct
  std::vector<int64_t> shape;
  DeepGen::DType dtype;
  DeepGen::ArgType argtype;
  int32_t rank;
  bool istran;
};

struct KernelInfo {
  // kernel info struct
  std::string kernel_name;
  std::string kernel_type;
  std::vector<mlir::Type> arg_type;
  std::vector<bool> istrans;
  int32_t output_num;
};

// operator base class
template <typename T>
struct Operator {
  template <typename... Args>
  static mlir::func::FuncOp buildKernel(mlir::ModuleOp module, int kernel_num, Args &&...args) {
    return T::buildKernel(module, kernel_num, std::forward<Args>(args)...);
  }

  static std::vector<mlir::Value> getIndexes(const llvm::SmallVector<mlir::Value>& bs, const std::vector<mlir::Value>& ivs, bool istran) {
    // 因为有转置，所以ivs取数的索引也需要转置
    std::vector<mlir::Value> indexes(bs.begin(), bs.end());
    if (istran) {
      indexes.push_back(ivs[0]);
      indexes.push_back(ivs[1]);
    } else {
      indexes.push_back(ivs[1]);
      indexes.push_back(ivs[0]);
    }
    return indexes;
  }

  static KernelInfo parseKernelInfo(mlir::OpBuilder b, const std::string& kn, const std::string& kt, const std::vector<ArgTensor>& args) {
    // parse kernel info 
    int32_t output_num = 0;
    std::vector<mlir::Type> arg_type;
    std::vector<bool> istrans;
    for (auto arg: args) {
      auto mem_type = getMemType(arg.shape, getDType(b, arg.dtype), MemorySpace::global);
      arg_type.push_back(mem_type);
      istrans.push_back(arg.istran);
      if (arg.argtype == ArgType::OUTPUT) output_num++;
    }
    KernelInfo info = {kn, kt, arg_type, istrans, output_num};
    return info;
  }

  static mlir::func::FuncOp buildFuncOp(mlir::OpBuilder& builder, KernelInfo info) {
    // create func Op
    mlir::Location loc = builder.getUnknownLoc();
    auto functionType = builder.getFunctionType(mlir::TypeRange(info.arg_type), mlir::TypeRange({}));
    auto funcOp = builder.create<mlir::func::FuncOp>(loc, llvm::StringRef(info.kernel_name), functionType);
    auto& region = funcOp->getRegion(0);
    if (!region.hasOneBlock()) {
      region.emplaceBlock();
    }
    auto& body =  funcOp.front(); //? region.front()  : ;
    llvm::SmallVector<mlir::Location> locs(info.arg_type.size(), loc);
    body.addArguments(info.arg_type, locs);
    // func add attrs
    setOpAttr(funcOp, FUNC_STATE, std::string("cpu"));
    setOpAttr(funcOp, FUNC_KERNEL_TYPE, info.kernel_type);
    setOpAttr(funcOp, FUNC_OUTPUT_NUM, info.output_num);
    setOpAttrArray(funcOp, FUNC_ARG_TRAN, info.istrans);
    
    auto& entryBlock = funcOp.front();
    builder.setInsertionPointToStart(&entryBlock);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&entryBlock);
    return funcOp;
  }
};

// Dot kernel
struct Dot : Operator<Dot> {
  static bool verify(ArgTensor A, ArgTensor B, ArgTensor C);

  static mlir::func::FuncOp buildKernel(mlir::ModuleOp module, int kernel_num, ArgTensor A, ArgTensor B, ArgTensor C);
};

// ElementWise kernel
struct ElementWise : Operator<ElementWise> {
  static std::map<ElementWiseMode, std::string> mode_name;

  static mlir::Value createModeOp(mlir::OpBuilder b, mlir::Value operand, ElementWiseMode mode);

  static bool verify(ArgTensor input, ArgTensor output);

  static mlir::func::FuncOp buildKernel(mlir::ModuleOp module, int kernel_num, ArgTensor input, ArgTensor output, ElementWiseMode mode);
};


// KernelGenerator class
class KernelGenerator {
  public:
    KernelGenerator() : context(std::make_unique<mlir::MLIRContext>()) {
      this->initCount();
      this->initContext();
      mlir::OpBuilder builder(context.get());
      this->module = mlir::ModuleOp::create(builder.getUnknownLoc());
    };
    KernelGenerator(mlir::ModuleOp mod) : module(mod), context(mod.getContext()) {
      this->initCount();
      this->connect(mod);
    };

    void connect(mlir::ModuleOp module);

    mlir::ModuleOp loadMLIRFile(const std::string& filePath);

    mlir::ModuleOp getModule() {
      return this->module;
    }

    template <typename OperatorType, typename... Args> 
    mlir::func::FuncOp create(Args &&...args) {
      int tmp_count;
      if constexpr (std::is_same_v<std::decay_t<OperatorType>, Dot>) {
        tmp_count = count[KERNEL_DOT];
        count[KERNEL_DOT]++;
      } else if constexpr (std::is_same_v<std::decay_t<OperatorType>, ElementWise>) {
        tmp_count = count[KERNEL_ELEMENTWISE];
        count[KERNEL_ELEMENTWISE]++;
      } else {
        assert(false && "Unsupported create kernel.");
      }
      return OperatorType::buildKernel(this->module, tmp_count, std::forward<Args>(args)...);
    }

  private:
    mlir::ModuleOp module;
    std::unique_ptr<mlir::MLIRContext> context;
    std::map<std::string, int> count;  // operator count

    void initCount() {
      count.emplace(KERNEL_DOT, 0);
      count.emplace(KERNEL_ELEMENTWISE, 0);
    }

    void initContext() {
      context->getOrLoadDialect<mlir::affine::AffineDialect>();
      context->getOrLoadDialect<mlir::memref::MemRefDialect>();
      context->getOrLoadDialect<mlir::func::FuncDialect>();
      context->getOrLoadDialect<mlir::arith::ArithDialect>();
      context->getOrLoadDialect<mlir::gpu::GPUDialect>();
      context->getOrLoadDialect<mlir::vector::VectorDialect>();
      context->getOrLoadDialect<mlir::scf::SCFDialect>();
      context->getOrLoadDialect<mlir::math::MathDialect>();
      context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
      context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    }
};

}

#endif