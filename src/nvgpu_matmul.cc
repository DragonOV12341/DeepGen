#include "compiler.h"
#include "operator.h"
#include "Python.h"

using namespace DeepGen;


int main() {
  KernelGenerator generator;
  DGCompiler compiler(Target::CUDA, "90");
  auto module = generator.getModule();
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(module);
  mlir::Location loc = builder.getUnknownLoc();
  builder.setInsertionPointToStart(module.getBody());

  mlir::MemRefType mem_type = mlir::MemRefType::get(llvm::ArrayRef<int64_t>({2048, 2048}), builder.getF16Type(), {}, 1);
  auto memOp_a = builder.create<mlir::memref::AllocOp>(loc, mem_type);
  auto memOp_b = builder.create<mlir::memref::AllocOp>(loc, mem_type);

  mlir::UnrankedMemRefType unrank_mem_type = mlir::UnrankedMemRefType::get(builder.getF16Type(), 1);
  auto unrank_memOp_a = builder.create<mlir::memref::CastOp>(loc, unrank_mem_type, memOp_a);
  auto unrank_memOp_b = builder.create<mlir::memref::CastOp>(loc, unrank_mem_type, memOp_b);
  
  auto i64 = builder.create<mlir::arith::ConstantIndexOp>(loc, 64);
  auto i128 = builder.create<mlir::arith::ConstantIndexOp>(loc, 128);

  mlir::MemRefType mem_type_sm_a = mlir::MemRefType::get(llvm::ArrayRef<int64_t>({128, 64}), builder.getF16Type(), {}, 3);
  mlir::Type desc_type_a = mlir::nvgpu::TensorMapDescriptorType::get(context, mem_type_sm_a, 
                                mlir::nvgpu::TensorMapSwizzleKind::SWIZZLE_128B, 
                                mlir::nvgpu::TensorMapL2PromoKind::L2PROMO_256B,
                                mlir::nvgpu::TensorMapOOBKind::OOB_NAN,
                                mlir::nvgpu::TensorMapInterleaveKind::INTERLEAVE_NONE);
  auto tensor_desc_a = builder.create<mlir::nvgpu::TmaCreateDescriptorOp>(loc, desc_type_a, unrank_memOp_a, mlir::ValueRange({i128, i64}));
  llvm::outs() << tensor_desc_a.getType().getTensor().getDimSize(0) << "\n";

  mlir::MemRefType mem_type_sm_b = mlir::MemRefType::get(llvm::ArrayRef<int64_t>({64, 128}), builder.getF16Type(), {}, 3);
  mlir::Type desc_type_b = mlir::nvgpu::TensorMapDescriptorType::get(context, mem_type_sm_b, 
                                mlir::nvgpu::TensorMapSwizzleKind::SWIZZLE_128B, 
                                mlir::nvgpu::TensorMapL2PromoKind::L2PROMO_256B,
                                mlir::nvgpu::TensorMapOOBKind::OOB_NAN,
                                mlir::nvgpu::TensorMapInterleaveKind::INTERLEAVE_NONE);
  auto tensor_desc_b = builder.create<mlir::nvgpu::TmaCreateDescriptorOp>(loc, desc_type_b, unrank_memOp_b, mlir::ValueRange({i64, i128}));

  mlir::Type wg_mat_type_a = mlir::nvgpu::WarpgroupMatrixDescriptorType::get(context, mem_type_sm_a);
  mlir::Type wg_mat_type_b = mlir::nvgpu::WarpgroupMatrixDescriptorType::get(context, mem_type_sm_b);
  auto mem_sm_Op_a = builder.create<mlir::memref::AllocOp>(loc, mem_type_sm_a);
  auto mem_sm_Op_b = builder.create<mlir::memref::AllocOp>(loc, mem_type_sm_b);
  auto sm_a_desc = builder.create<mlir::nvgpu::WarpgroupGenerateDescriptorOp>(loc, wg_mat_type_a, mem_sm_Op_a, tensor_desc_a);
  auto sm_b_desc = builder.create<mlir::nvgpu::WarpgroupGenerateDescriptorOp>(loc, wg_mat_type_b, mem_sm_Op_b, tensor_desc_b);

  auto result_shape_type = mlir::VectorType::get(mlir::ArrayRef<int64_t>({128, 128}), builder.getF16Type());
  auto wg_acc_type = mlir::nvgpu::WarpgroupAccumulatorType::get(context, result_shape_type);
  auto acc = builder.create<mlir::nvgpu::WarpgroupMmaInitAccumulatorOp>(loc, wg_acc_type);

  auto res = builder.create<mlir::nvgpu::WarpgroupMmaOp>(loc, wg_acc_type, sm_a_desc, sm_b_desc, mlir::IntegerAttr(), mlir::UnitAttr(), mlir::UnitAttr(), acc);

  llvm::outs() << "type: " << res.getMatrixC().getType() << "\n";

  mlir::PassManager pm(context);
  pm.addPass(mlir::createConvertNVGPUToNVVMPass());
  if (mlir::failed(pm.run(module)))
    return 0;

  // compiler.transform(module);
  // compiler.lowering(module);

  llvm::outs() << module << "\n";
  return 0;
}