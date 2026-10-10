# -*- coding: utf-8 -*-
"""shim 等价性自检 · DeepSeek-V3.2（shim-development §4 要求留痕）

验证策略（原 kernel 本机完全不可运行，无 tilelang）：
  全部采用「对拍数学定义」级——为每个 shim 写一份与 shim 结构不同的独立
  参考实现（loop / 朴素公式 / popcount 矩阵），随机输入 + 边界用例对拍。
运行：.venv/bin/python outputs/DeepSeek-V3.2/capture/shims/selftest.py
结果抄录至 IR contract.replacements[].evidence。
"""
import sys
import pathlib
import torch

SHIM_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SHIM_DIR))
from kernel import ActQuantShim, Fp8GemmShim, Fp8IndexShim  # noqa: E402
import fast_hadamard_transform as fht  # noqa: E402

torch.manual_seed(42)
PASS = []


def record(name, ok, detail):
    PASS.append((name, ok, detail))
    print(f"  [{'通过' if ok else '失败'}] {name}: {detail}")


# ---------------------------------------------------------------- act_quant
def ref_act_quant(x, block_size, ue8m0):
    """独立参考实现：分块 loop + torch.log2/ceil 数学定义（与 shim 位运算结构不同）。"""
    n = x.size(-1)
    xf = x.float()
    lead = xf.shape[:-1]
    blocks = xf.reshape(-1, block_size)
    nb = blocks.size(0)
    amax = blocks.abs().amax(dim=-1)
    amax = torch.clamp(amax, min=1e-4)
    # 源码 L74 语义为「乘以 fp8_max_inv」（非除法），位级语义以此为准
    s = amax * (1.0 / 448.0)
    if ue8m0:
        s = torch.pow(2.0, torch.ceil(torch.log2(s)))
    y = torch.clamp(blocks / s.unsqueeze(-1), -448.0, 448.0)
    y = y.reshape(x.shape).to(torch.float8_e4m3fn)
    return y, s.reshape(*lead, n // block_size)


def test_act_quant():
    shim = ActQuantShim()
    for fmt in (None, "ue8m0"):
        for shape in [(4, 256), (2, 3, 128), (8, 128)]:
            x = (torch.randn(*shape) * 30).to(torch.bfloat16)
            y1, s1 = shim(x.contiguous(), 128, fmt)
            y2, s2 = ref_act_quant(x.contiguous(), 128, fmt == "ue8m0")
            same_y = torch.equal(y1, y2)
            same_s = torch.allclose(s1, s2, rtol=0, atol=0)
            record(f"act_quant fmt={fmt} shape={shape}", same_y and same_s,
                   f"y 一致={same_y} scale 一致={same_s}")
    # 边界：全零 / amax 恰为 448 / 极小值
    x0 = torch.zeros(2, 128, dtype=torch.bfloat16)
    y0, s0 = shim(x0, 128, "ue8m0")
    record("act_quant 边界·全零", torch.equal(y0.float(), torch.zeros(2, 128)) and
           torch.all(s0 > 0), f"y 全零={torch.equal(y0.float(), torch.zeros(2, 128))}, s={s0.flatten()[:2].tolist()}")
    x1 = torch.full((1, 128), 448.0, dtype=torch.bfloat16)
    y1, s1 = shim(x1, 128, "ue8m0")
    record("act_quant 边界·amax=448", torch.equal(s1, torch.ones(1, 1)) and
           torch.all(y1.float() <= 448.0),
           f"s={s1.flatten().tolist()}（2^0=1）, y max={y1.float().max().item()}")
    x2 = torch.full((1, 128), 1e-4, dtype=torch.bfloat16)
    y2, s2 = shim(x2, 128, "ue8m0")
    expect_s = 2.0 ** torch.ceil(torch.log2(torch.tensor(1e-4 / 448.0)))
    record("act_quant 边界·amax=1e-4(下限)", torch.allclose(s2, expect_s.reshape(1, 1), rtol=0, atol=0),
           f"s={s2.flatten().tolist()} 期望={expect_s.item():.3e}")


# ---------------------------------------------------------------- fp8_gemm
def ref_fp8_gemm(a, sa, b, sb):
    """独立参考实现：按 k 块循环累加（与 shim 的整体反量化+matmul 结构不同）。"""
    m, k = a.shape
    n = b.shape[0]
    acc = torch.zeros(m, n, dtype=torch.float32)
    nb_k = k // 128
    sb_n = sb.repeat_interleave(128, dim=0)[:n]           # [N, K/128] 逐 n 行展开（不足截断）
    for g in range(nb_k):
        a_blk = a[:, g * 128:(g + 1) * 128].float() * sa[:, g:g + 1]
        b_blk = b[:, g * 128:(g + 1) * 128].float() * sb_n[:, g:g + 1]
        acc += a_blk @ b_blk.transpose(0, 1)
    return acc.to(torch.get_default_dtype())


def test_fp8_gemm():
    shim = Fp8GemmShim()
    for (m, k, n) in [(5, 256, 384), (1, 128, 576), (64, 512, 2048)]:
        a = (torch.randn(m, k) * 3).clamp(-448, 448).to(torch.float8_e4m3fn)
        b = (torch.randn(n, k)).clamp(-448, 448).to(torch.float8_e4m3fn)
        sa = (torch.rand(m, k // 128) * 0.01 + 0.001)
        sb = (torch.rand((n + 127) // 128, k // 128) * 0.01 + 0.001)
        c1 = shim(a, sa, b, sb)
        c2 = ref_fp8_gemm(a, sa, b, sb)
        err = (c1.float() - c2.float()).abs().max().item()
        rel = err / max(1.0, c2.float().abs().max().item())
        record(f"fp8_gemm M={m} K={k} N={n}", rel < 1e-5,
               f"shape={tuple(c1.shape)} dtype={c1.dtype} 最大绝对误差={err:.3e} 相对={rel:.2e}")


# ---------------------------------------------------------------- fp8_index
def ref_fp8_index(q, qs, k, ks):
    """独立参考实现：按 docstring 语义（kernel.py L269-272）的朴素四重循环。"""
    b, s, ih, id_ = q.shape
    t = k.shape[1]
    out = torch.zeros(b, s, t, dtype=torch.float32)
    for bi in range(b):
        for mi in range(s):
            for ni in range(t):
                total = 0.0
                for hi in range(ih):
                    dot = 0.0
                    for di in range(id_):
                        dot += float(q[bi, mi, hi, di]) * float(k[bi, ni, di])
                    total += max(0.0, dot) * float(qs[bi, mi, hi, 0])
                out[bi, mi, ni] = total * float(ks[bi, ni, 0])
    return out


def test_fp8_index():
    shim = Fp8IndexShim()
    b, s, ih, id_, t = 2, 3, 4, 128, 5
    q = (torch.randn(b, s, ih, id_) * 5).clamp(-448, 448).to(torch.float8_e4m3fn)
    k = (torch.randn(b, t, id_) * 5).clamp(-448, 448).to(torch.float8_e4m3fn)
    qs = torch.rand(b, s, ih, 1) * 0.1
    ks = torch.rand(b, t, id_ // 128) * 0.01
    o1 = shim(q, qs, k, ks)
    o2 = ref_fp8_index(q, qs, k, ks)
    err = (o1 - o2).abs().max().item()
    record(f"fp8_index B={b} S={s} IH={ih} ID={id_} T={t}", torch.allclose(o1, o2, rtol=1e-5, atol=1e-6),
           f"shape={tuple(o1.shape)} dtype={o1.dtype} 最大误差={err:.3e}")


# ---------------------------------------------------------------- hadamard
def ref_hadamard_matrix(n):
    """独立参考实现：popcount 符号公式 H[i,j] = (-1)^popcount(i & j)。"""
    idx = torch.arange(n)
    pc = torch.tensor([[bin(i & j).count('1') for j in range(n)] for i in range(n)])
    return torch.where(pc % 2 == 0, 1.0, -1.0)


def test_hadamard():
    for n in (8, 128):
        x = torch.randn(4, n, dtype=torch.bfloat16)
        scale = n ** -0.5
        y1 = fht.hadamard_transform(x, scale)
        h = ref_hadamard_matrix(n)
        y2 = (x.float() @ h) * scale
        err = (y1.float() - y2).abs().max().item()
        record(f"hadamard_transform N={n}", err < 1e-2,
               f"shape={tuple(y1.shape)} dtype={y1.dtype} 最大误差={err:.3e}（bf16 量化噪声内）")
        # 正交性：H/√N 应为正交阵（y @ Hᵀ/√N ≈ x）
        x_back = fht.hadamard_transform(y1, scale)
        ortho_err = (x_back.float() - x.float()).abs().max().item()
        record(f"hadamard_transform 正交性 N={n}", ortho_err < 1e-1,
               f"往返最大误差={ortho_err:.3e}")


if __name__ == "__main__":
    print("◆ shim 等价性自检开始")
    test_act_quant()
    test_fp8_gemm()
    test_fp8_index()
    test_hadamard()
    n_fail = sum(1 for _, ok, _ in PASS if not ok)
    print(f"◆ 自检完成：{len(PASS)} 项，失败 {n_fail} 项")
    sys.exit(1 if n_fail else 0)
