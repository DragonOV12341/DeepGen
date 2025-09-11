def compute_logic(warpid, laneid, warp_group_id):
    # r1 对应 warpid
    # r3 对应 laneid
    # r4 对应 warp_group_id
    
    # shr.s32 %r152, %r1, 31;  // 有符号右移31位，获取符号位扩展
    r152 = warpid >> 31  # 对于32位有符号数，右移31位会得到符号位(0或-1)
    
    # shr.u32 %r153, %r152, 30;  // 无符号右移30位
    # 注意：Python中移位前先转换为无符号表示
    r153 = (r152 & 0xFFFFFFFF) >> 30  # 结果只能是0或1
    
    # add.s32 %r154, %r1, %r153;
    r154 = warpid + r153
    
    # and.b32 %r155, %r154, -4;  // -4的二进制是111...1100，清除最后两位
    r155 = r154 & -4
    
    # sub.s32 %r156, %r1, %r155;
    r156 = warpid - r155
    
    # setp.lt.s32 %p5, %r156, 0;  // 判断是否小于0
    p5 = 1 if (r156 < 0) else 0
    
    # add.s32 %r157, %r156, 4;
    r157 = r156 + 4
    
    # selp.b32 %r158, %r157, %r156, %p5;  // 根据p5选择值
    r158 = r157 if p5 else r156
    
    # and.b32 %r159, %r3, 15;  // 取laneid的低4位
    r159 = laneid & 15
    
    # shl.b32 %r160, %r4, 13;  // 左移13位
    r160 = warp_group_id << 13
    
    # shl.b32 %r161, %r158, 11;  // 左移11位
    r161 = r158 << 11
    
    # shl.b32 %r162, %r159, 7;  // 左移7位
    r162 = r159 << 7
    
    # or.b32 %r163, %r161, %r162;  // 按位或
    r163 = r161 | r162
    
    # add.s32 %r164, %r160, %r163;
    r164 = r160 + r163
    
    # shr.u32 %r165, %r3, 1;  // 无符号右移1位
    r165 = (laneid & 0xFFFFFFFF) >> 1  # 处理无符号右移
    
    # and.b32 %r166, %r165, 8;  // 保留第3位(0-based)
    r166 = r165 & 8
    
    # or.b32 %r167, %r164, %r166;  // 按位或
    r167 = r164 | r166
    
    # shl.b32 %r168, %r167, 1;  // 左移1位
    r168 = r167 << 1
    
    # mov.b32 %r169, smem;  // 假设共享内存基地址为0
    smem_base = 0  # 实际应用中会是真实的共享内存地址
    r169 = smem_base
    
    # add.s32 %r170, %r169, %r168;
    r170 = r169 + r168
    
    # add.s32 %r112, %r170, 8192;
    r112 = r170 + 8192
    
    print
    
    return {
        'r112': r112,
        'r158': r158,
        'r167': r167,
        'r170': r170
    }

def compute(warpid, laneid, warp_group_id):
    wgy = warp_group_id
    wgx = 0
    lx = laneid // 16
    ly = laneid % 16
    wy = warpid % 4
    
    offset = (wgy * 64) * 128 + (wgx * 128) + (wy * 16 + ly) * 128 + lx * 8

    print(f"offset: {offset}")
    print(f"addr:{4096 + offset}")

# 示例调用
if __name__ == "__main__":
    # 传入示例参数
    for i in range(256):
      warpid = i // 32
      laneid = i % 32
      warp_group_id = i // 128
      result = compute_logic(warpid=warpid, laneid=laneid, warp_group_id=warp_group_id)
      print(f"计算结果: {result}")
      compute(warpid=warpid, laneid=laneid, warp_group_id=warp_group_id)