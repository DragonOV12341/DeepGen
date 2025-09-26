#ifndef TRITON_ATTR_DEFS
#define TRITON_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

def MemSpaceEnumAttr : I32EnumAttr<
    "MemSpaceEnum", "memory space enum",
    [
        I32EnumAttrCase<"GLOBAL",0, "global" >,
        I32EnumAttrCase<"SHARED",1, "shared" >,
        I32EnumAttrCase<"LOCAL", 2, "local"   >
    ]>{
        let cppNamespace = "::mlir::deepgen";
    }

def MemSyncScopeAttr : I32EnumAttr<
    "MemSyncScope", "",
    [
      I32EnumAttrCase<"DEVICE", 1, "gpu_device_level">,
      I32EnumAttrCase<"BLOCK", 2, "block_level">,
      I32EnumAttrCase<"SYSTEM", 3, "sys_level">,
    ]> {
    let cppNamespace = "::mlir::deepgen";
}

def DimAxisAttr : I32EnumAttr<
    "Dimension", "",
    [
        I32EnumAttrCase<"x", 0, "x">,
        I32EnumAttrCase<"y", 1, "y">,
        I32EnumAttrCase<"z", 2, "z">,
    ]> {
    let cppNamespace = "::mlir::deepgen";
}

def LayoutAttr : I32EnumAttr<
    "LayoutKind", "buffer布局为swizzled或普通",
    [
        I32EnumAttrCase<"Normal", 0, "normal_layout">,
        I32EnumAttrCase<"Swizzled", 1, "swizzle_layout">
    ]> {
    let cppNamespace = "::mlir::deepgen";
}


#endif
