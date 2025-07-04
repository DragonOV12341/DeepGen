file(GLOB_RECURSE analysis_src ./Analysis/*.cc)
file(GLOB_RECURSE conversion_src ./Conversion/*.cc)
file(GLOB_RECURSE transforms_src ./Transforms/*.cc)
file(GLOB_RECURSE targets_src ./Targets/*.cc)

add_library(dg_runtime 
            ${analysis_src}
            ${conversion_src}
            ${transforms_src}
            ${targets_src}
            compiler.cc
            operator.cc
            )


#target_compile_options(dg_runtime PRIVATE -frtti)
# MLIR有自己的一套RTTI，需要关掉才能正确继承MLIR中类
target_compile_options(dg_runtime PUBLIC -fno-rtti)

set(LLVM_LINK_COMPONENTS
        Core
        Support
        nativecodegen
        OrcJIT
        )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

target_link_libraries(dg_runtime
        PUBLIC
        ${dialect_libs}
        ${conversion_libs}
        ${translation_libs}
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIRNVVMToLLVMIRTranslation
        MLIRToLLVMIRTranslationRegistration
        MLIRTargetLLVMIRImport
        MLIRTargetLLVMIRExport
        MLIRFuncToLLVM
        )