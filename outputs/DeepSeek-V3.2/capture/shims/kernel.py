# -*- coding: utf-8 -*-
"""kernel.py 等价替换件（shim）· DeepSeek-V3.2

毛坯 `inputs/DeepSeek-V3.2/inference/kernel.py` 的三个 tilelang kernel 在本机
无法运行（缺 tilelang 依赖，见 audit_report.missing_deps），本文件用纯 torch
基础算子实现数学等价替换，经 spec.sys_path 遮蔽同名 `kernel` 模块 +
spec.shims[].rebind 重绑 model 模块全局名接管全部调用点。

对应 IR contract.replacements 条目：
  - act_quant  → ActQuantShim
  - fp8_gemm   → Fp8GemmShim
  - fp8_index  → Fp8IndexShim

数学语义均逐行对照原 kernel 源码（kernel.py）：
  - act_quant   kernel.py L36-84（fn L87-111）：末维按 128 分块，
    amax → clamp(min 1e-4) → (ue8m0 时) 2^ceil(log2(amax/448))，
    y = clamp(x/s, ±448) cast fp8；scale 形状 [..., N/128] fp32。
  - fp8_gemm    kernel.py L114-168（fn L171-196）：
    C[m,n] = Σ_k a[m,k]·sa[m,⌊k/128⌋]·b[n,k]·sb[⌊n/128⌋,⌊k/128⌋]，
    fp32 累加，输出 cast 默认 dtype（等价于双侧按块反量化后的 fp32 matmul）。
  - fp8_index   kernel.py L199-251（fn L254-274，语义见 docstring L269-272）：
    o[b,m,n] = ks[b,n] · Σ_h relu(Σ_d q[b,m,h,d]·k[b,n,d]) · qs[b,m,h]，fp32 输出。

数值纪律：全部中间计算升 float32；输出 dtype 对齐调用处约定；
clamp 边界（±448 / amax 下限 1e-4）与源码逐一对齐。
"""
import torch
from typing import Optional, Tuple

BLOCK_SIZE = 128
FP8_MIN = -448.0
FP8_MAX = 448.0


def _fast_round_scale_ue8m0(amax: torch.Tensor) -> torch.Tensor:
    """ue8m0 尺度：s = 2^ceil(log2(amax))，与 kernel.py fast_round_scale /
    fast_log2_ceil / fast_pow2（L20-33）位运算逐位等价的纯 torch 实现。

    fast_log2_ceil(x) = exp(x)-127 + (man != 0)（对正的规格化 fp32 即 ceil(log2(x))）；
    fast_pow2(e) = (e+127) << 23 重新解释为 fp32。
    前置条件：amax > 0（act_quant 语义里 amax 已被 clamp 到 ≥ 1e-4）。
    """
    bits = amax.float().view(torch.int32)
    exp = (torch.bitwise_right_shift(bits, 23) & 0xFF) - 127
    man = torch.bitwise_and(bits, (1 << 23) - 1)
    ceil_log2 = exp + (man != 0).to(torch.int32)          # ceil(log2(amax))
    bits_s = (ceil_log2 + 127) << 23
    return bits_s.view(torch.float32)


def _block_amax(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """末维按 block_size 分块求绝对值最大值（fp32）。末维必须能被 block_size 整除。"""
    n = x.size(-1)
    assert n % block_size == 0, (
        f"末维必须能被 block_size 整除（{n} % {block_size} != 0）")
    xf = x.float()
    lead_shape = xf.shape[:-1]
    xf = xf.reshape(-1, n // block_size, block_size)
    return xf.abs().amax(dim=-1).reshape(*lead_shape, n // block_size)


class ActQuantShim(torch.nn.Module):
    """act_quant 等价替换件（kernel.py L87-111 + kernel L36-84）。

    forward(x, block_size=128, scale_fmt=None) -> (fp8 同形, fp32 scale [..., N/block])
    与原实现一致：要求 x contiguous、末维整除 block_size。
    """

    def forward(self, x: torch.Tensor, block_size: int = 128,
                scale_fmt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.is_contiguous(), "输入张量必须是连续的（对齐原实现 L102 assert）"
        amax = _block_amax(x, block_size)                     # L68 reduce_absmax
        amax = torch.clamp(amax, min=1e-4)                    # L70
        if scale_fmt:                                         # L71-72（ue8m0）
            s = _fast_round_scale_ue8m0(amax / FP8_MAX)
        else:                                                 # L74
            s = (amax * (1.0 / FP8_MAX)).float()
        xf = x.float().reshape(*s.shape[:-1], -1)             # [.., N] 平铺到块
        s_exp = s.repeat_interleave(block_size, dim=-1)       # 尺度沿末维展开
        y = torch.clamp(xf / s_exp, FP8_MIN, FP8_MAX)         # L76-78
        y = y.reshape(x.shape).to(torch.float8_e4m3fn)
        return y, s


class Fp8GemmShim(torch.nn.Module):
    """fp8_gemm 等价替换件（kernel.py L114-168 + fn L171-196）。

    forward(a, a_scale, b, b_scale) -> [*, N]
      C[m,n] = Σ_k a[m,k]·sa[m,⌊k/128⌋]·b[n,k]·sb[⌊n/128⌋,⌊k/128⌋]
    实现结构上等价于：双侧按 128 块反量化到 fp32 后做 matmul，再 cast 默认 dtype。
    a 任意前置维（调用处传入 [B,S,K] 等），b [N,K]，K 必须整除 128（sa/sb 布局约定）。
    """

    def forward(self, a: torch.Tensor, a_scale: torch.Tensor,
                b: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
        k = a.size(-1)
        assert k % BLOCK_SIZE == 0, f"K 必须整除 {BLOCK_SIZE}（当前 {k}）"
        assert b.size(-1) == k, f"b 末维 {b.size(-1)} 与 a 末维 {k} 不一致"
        a2 = a.reshape(-1, k).float()                         # [M, K]
        b2 = b.float()                                        # [N, K]
        sa = a_scale.float().repeat_interleave(BLOCK_SIZE, dim=-1)   # [M, K]
        nb = b_scale.size(-2)
        sb = b_scale.float()                                  # [ceil(N/128), K/128]
        sb_exp = sb.repeat_interleave(BLOCK_SIZE, dim=-2).repeat_interleave(BLOCK_SIZE, dim=-1)
        b_deq = b2 * sb_exp[: b2.size(-2)]                    # [N, K] 逐块反量化
        c = a2 * sa                                           # A 反量化
        c = torch.matmul(c, b_deq.transpose(0, 1))            # [M, N] fp32 累加
        return c.reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


class Fp8IndexShim(torch.nn.Module):
    """fp8_index 等价替换件（kernel.py L199-251 + fn L254-274）。

    forward(q, q_s, k, k_s) -> [B, S, T] fp32
      o[b,m,n] = ks[b,n] · Σ_h relu(Σ_d q[b,m,h,d]·k[b,n,d]) · qs[b,m,h]
    说明：第三形参 q_s 实收 Indexer L479 的 weights（已并入 softmax_scale 与
    q_scale），逐点相乘语义一致；k_s 按 d 维 128 分块（ID=128 时每行 1 块）。
    """

    def forward(self, q: torch.Tensor, q_s: torch.Tensor,
                k: torch.Tensor, k_s: torch.Tensor) -> torch.Tensor:
        b, s, ih, id_ = q.shape
        t = k.shape[1]
        qf = q.float().reshape(b * s, ih, id_)                # [B*S, IH, ID]
        kf = k.float().reshape(b, t, id_)                     # [B, T, ID]
        nb = k_s.shape[-1]                                    # d 维分块数
        id_pad = nb * BLOCK_SIZE
        if id_pad != id_:  # 理论上 ID 整除 128；防御性补齐
            kf = torch.nn.functional.pad(kf, (0, id_pad - id_))
            qf = torch.nn.functional.pad(qf, (0, id_pad - id_))
        kf = kf.reshape(b, t, nb, BLOCK_SIZE)
        qf = qf.reshape(b, s, ih, nb, BLOCK_SIZE)
        # 每块点积：dot[b,s,h,g,t] = Σ_d q[b,s,h,g,d]·k[b,t,g,d]
        dot = torch.einsum("bshgd,btgd->bshgt", qf, kf)       # [B, S, IH, NB, T]
        dot = torch.relu(dot)                                 # L241
        # Σ_h relu(块内点积) × q_s → [B, S, NB, T]
        o = (dot * q_s.reshape(b, s, ih, nb, 1)).sum(dim=2)   # [B, S, NB, T]
        o = (o * k_s.float().reshape(b, 1, nb, t)).sum(dim=2)  # [B, S, T]（L247）
        return o


# ---------------------------------------------------------------- 模块级别名
# 毛坯 model.py L10 `from kernel import act_quant, fp8_gemm, fp8_index` 要求
# kernel 模块导出这三个名字（否则导入失败）。此处提供单例实例；真实调用会被
# spec.shims[].rebind 重绑到挂树实例接管（挂树=图上可见，重绑=调用接管）。
act_quant = ActQuantShim()
fp8_gemm = Fp8GemmShim()
fp8_index = Fp8IndexShim()
