# -*- coding: utf-8 -*-
"""Kimi-K3 shim 等价对拍（为 outputs/Kimi-K3/capture/shims/README.md 留验证痕）。

对拍等级（如实声明）：数学定义级 —— 参考实现由 shim 头注的数学定义独立手写
（与 shim 代码不同的实现路径：显式 per-channel/head 循环替代 einsum/conv1d），
验证"实现符合其声称的数学定义"，不验证"与 fla 原生 Triton 核数值一致"（fla 未安装，无法核对）。

R1 einops.rearrange 子集 vs 手写循环参考
R2a ShortConvolution vs 逐通道循环参考
R2b FusedRMSNormGated vs 公式参考
R2c chunk_kda / fused_recurrent_kda vs 逐步递推参考（含模型真实旗标组合）
R3  flash_attention_2_cpu_equiv vs 毛坯自带 eager_attention_forward（同输入对拍）
"""
import importlib
import math
import sys
import types

import torch

WS = "/Volumes/Agent/workspace/projects/model-insight/model-insight-0928"
SHIMS = f"{WS}/outputs/Kimi-K3/capture/shims"
MODEL_DIR = f"{WS}/inputs/Kimi-K3"

sys.path.insert(0, SHIMS)

torch.manual_seed(0)
FP32 = dict(rtol=1e-4, atol=1e-5)
BF16 = dict(rtol=2e-2, atol=3e-2)

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# ===========================================================================
# R3/R2 依赖：加载 hooks（P1 别名 + R3 注册），再取毛坯 eager_attention_forward
# ===========================================================================
spec = importlib.util.spec_from_file_location(
    "kk3_hooks", f"{WS}/outputs/Kimi-K3/capture/hooks.py")
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)
hooks._MODEL_DIR = MODEL_DIR
hooks._apply_compat_patches()          # P1-P4 + R3 注册（含被测的等价函数）

from fla.modules import FusedRMSNormGated, ShortConvolution      # noqa: E402 (shim)
from fla.ops.kda import chunk_kda, fused_recurrent_kda           # noqa: E402 (shim)

lin = importlib.import_module("kimi_k3_pkg.modeling_kimi_linear")
eager_attention_forward = lin.eager_attention_forward
fa2_equiv = hooks.flash_attention_2_cpu_equiv


# ===========================================================================
# R1 einops.rearrange —— 手写参考（切片/转置组合）
# ===========================================================================
import einops  # shim（sys.path 首位）

# 1) 拆分 'b t (h d) -> b t h d'：参考 = view+无置换
x = torch.randn(2, 5, 24)
ref = x.reshape(2, 5, 8, 3)
check("R1 拆分 (b t (h d))", torch.equal(einops.rearrange(x, "b t (h d) -> b t h d", d=3), ref))

# 2) 合并 'b h d -> b (h d)'：参考 = flatten
y = torch.randn(2, 5, 24)
ref2 = y.reshape(2, 120)
check("R1 合并 (b (h d))", torch.equal(einops.rearrange(y, "b h d -> b (h d)"), ref2))

# 3) 省略号 '... (h d) -> ... h d'：参考 = reshape 前导维展平后拆
z = torch.randn(2, 3, 24)
ref3 = z.reshape(2, 3, 8, 3)   # einops 语义：... 逐维保留，拆分只作用于最后一维
check("R1 省略号 (... (h d))",
      torch.equal(einops.rearrange(z, "... (h d) -> ... h d", d=3), ref3))

# 4) 置换 'b t h d -> b h t d'
w = torch.randn(2, 5, 8, 3)
ref4 = w.permute(0, 2, 1, 3)
check("R1 置换", torch.equal(einops.rearrange(w, "b t h d -> b h t d"), ref4))

# 5) 毛坯实际模式1：'b t (h d) -> b h t d'（KDA q/k 重排，modeling:605-606）
m1_in = torch.randn(2, 7, 96)
ref5 = m1_in.reshape(2, 7, 3, 32).permute(0, 2, 1, 3)
check("R1 毛坯模式1 (b t (h d) -> b h t d)",
      torch.equal(einops.rearrange(m1_in, "b t (h d) -> b h t d", d=32), ref5))

# 6) 毛坯实际模式2：'b t h d -> b t (h d)'（KDA 输出合并，modeling:658）
m2_in = torch.randn(2, 7, 3, 32)
ref6 = m2_in.reshape(2, 7, 96)
check("R1 毛坯模式2 (b t h d -> b t (h d))",
      torch.equal(einops.rearrange(m2_in, "b t h d -> b t (h d)"), ref6))


# ===========================================================================
# R2a ShortConvolution vs 逐通道循环参考
# ===========================================================================
def shortconv_ref(x, weight, bias, cache=None, act="silu", k=4):
    """逐通道循环：y[b,d,t] = act(sum_k w[d,0,k]*xpad[b,d,t+k] + bias[d])。
    x (B,L,D)  weight (D,1,K)  bias (D,)  cache (B,D,K-1) | None
    返回 (y (B,L,D), final (B,D,K-1) 原始输入尾部)。"""
    B, L, D = x.shape
    pad = cache if cache is not None else x.new_zeros(B, D, k - 1)
    xpad = torch.cat([pad, x.transpose(1, 2)], dim=-1).float()   # (B,D,K-1+L)
    y = torch.empty(B, D, L, dtype=torch.float32)
    for d in range(D):
        for t in range(L):
            # 逐通道内积（fp32 张量运算，与 shim 的 F.conv1d 路径不同实现）
            y[:, d, t] = (weight[d, 0, :].float() * xpad[:, d, t:t + k]).sum(-1) + bias[d].float()
    y = y.to(x.dtype)
    final = xpad[:, :, -(k - 1):]
    if act == "silu":
        y = torch.nn.functional.silu(y.float()).to(x.dtype)
    return y.transpose(1, 2), final            # 转回 (B,L,D)


for dt, tol, tag in ((torch.float32, FP32, "fp32"), (torch.bfloat16, BF16, "bf16")):
    conv = ShortConvolution(hidden_size=16, kernel_size=4, activation="silu", bias=True).to(dt)
    conv.eval()
    x = torch.randn(2, 7, 16, dtype=dt)
    y, final = conv(x, output_final_state=True)
    yr, fr = shortconv_ref(x, conv.conv1d.weight.detach(), conv.conv1d.bias.detach())
    d1, d2 = maxdiff(y, yr), maxdiff(final, fr)
    check(f"R2a ShortConv 无缓存 {tag}", d1 < tol["atol"] + tol["rtol"] * yr.abs().max().item() and d2 < 1e-5,
          f"y_maxdiff={d1:.2e} final_maxdiff={d2:.2e}")
    # 带 cache（decode 续算）
    cache = torch.randn(2, 16, 3, dtype=dt)
    y2, final2 = conv(x, cache=cache, output_final_state=True)
    yr2, fr2 = shortconv_ref(x, conv.conv1d.weight.detach(), conv.conv1d.bias.detach(), cache=cache)
    d3, d4 = maxdiff(y2, yr2), maxdiff(final2, fr2)
    check(f"R2a ShortConv 带 cache {tag}", d3 < tol["atol"] + tol["rtol"] * yr2.abs().max().item() and d4 < 1e-5,
          f"y_maxdiff={d3:.2e} final_maxdiff={d4:.2e}")


# ===========================================================================
# R2b FusedRMSNormGated vs 公式参考
# ===========================================================================
for dt, tol, tag in ((torch.float32, FP32, "fp32"), (torch.bfloat16, BF16, "bf16")):
    norm = FusedRMSNormGated(hidden_size=32, eps=1e-6, activation="sigmoid").to(dt)
    norm.eval()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32).abs() + 0.5)
    x = torch.randn(2, 5, 32, dtype=dt)
    o = torch.randn(2, 5, 32, dtype=dt)
    y = norm(x, o)
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight.float()
           * torch.sigmoid(o.float())).to(dt)
    d = maxdiff(y, ref)
    check(f"R2b FusedRMSNormGated {tag}", d < tol["atol"] + tol["rtol"] * ref.abs().max().item(),
          f"maxdiff={d:.2e}")


# ===========================================================================
# R2c chunk_kda / fused_recurrent_kda vs 逐步递推参考（独立手写，fp64 数值锚）
# ===========================================================================
def kda_ref_fp64(q, k, v, g, beta, A_log, dt_bias, initial_state,
                 l2n=True, gate=True, beta_sig=True, safe_gate=False, lower_bound=None):
    """q,k (B,T,H,K)  v (B,T,H,V)  g_raw (B,T,H,K)  beta_raw (B,T,H)
    A_log (H,)  dt_bias (H,K)。全部在 float64 重算（数值锚）。"""
    B, T, H, K = q.shape
    V = v.shape[-1]
    q = q.double(); k = k.double(); v = v.double()
    if l2n:
        q = q / q.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        k = k / k.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    beta = torch.sigmoid(beta.double()) if beta_sig else beta.double()
    if gate:
        gf = g.double() + dt_bias.double().view(H, K)
        gf = -torch.exp(A_log.double()).view(H, 1) * torch.nn.functional.softplus(gf)
        if safe_gate and lower_bound is not None:
            gf = torch.clamp(gf, min=float(lower_bound))
    else:
        gf = g.double()
    decay = torch.exp(gf)
    state = initial_state.double().clone() if initial_state is not None else torch.zeros(B, H, K, V, dtype=torch.float64)
    outs = []
    for t in range(T):
        # 显式逐 head 循环（与 shim 的 einsum 路径不同实现）
        S_all = state * decay[:, t].unsqueeze(-1)
        new_state = torch.empty_like(S_all)
        o_t = torch.empty(B, H, V, dtype=torch.float64)
        for h in range(H):
            S = S_all[:, h]                                    # (B,K,V)
            u = torch.einsum("bk,bkv->bv", k[:, t, h].double(), S)
            delta = beta[:, t, h].unsqueeze(-1) * (v[:, t, h].double() - u)
            S = S + k[:, t, h].double().unsqueeze(-1) * delta.unsqueeze(-2)
            new_state[:, h] = S
            o_t[:, h] = torch.einsum("bk,bkv->bv", q[:, t, h].double(), S)
        state = new_state
        outs.append(o_t)
    return torch.stack(outs, dim=1), state                     # (B,T,H,V) fp64, (B,H,K,V) fp64


B, T, H, K, V = 2, 6, 3, 8, 8
FLAGS = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
             use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-5.0,
             output_final_state=True, transpose_state_layout=True)
A_log = torch.randn(H) * 0.3
dt_bias = torch.randn(H, K) * 0.5

# fp32 数学级对拍（把 shim 输出当 fp32 → 与 fp64 锚比较，容差按 fp32 累积放宽）
q = torch.randn(B, T, H, K); k = torch.randn(B, T, H, K); v = torch.randn(B, T, H, V)
g = torch.randn(B, T, H, K); beta = torch.randn(B, T, H)
o, st = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
                  initial_state=None, **FLAGS)
o64, st64 = kda_ref_fp64(q, k, v, g, beta, A_log, dt_bias.view(H, K), None,
                         l2n=True, gate=True, beta_sig=True, safe_gate=True, lower_bound=-5.0)
do, ds = maxdiff(o, o64), maxdiff(st, st64)
scale_o = o64.abs().max().item()
check("R2c chunk_kda 模型旗标 fp32-vs-fp64锚",
      do < 1e-3 * scale_o + 1e-4 and ds < 1e-2 * st64.abs().max().item(),
      f"o_maxdiff={do:.2e} (|o|max={scale_o:.2f}) state_maxdiff={ds:.2e}")

# bf16 dtype 纪律（模型实际 dtype）：与 fp64 锚比较，容差 = bf16 舍入 + fp32 累积
qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
gb, bb = g.bfloat16(), beta.bfloat16()
ob, stb = chunk_kda(q=qb, k=kb, v=vb, g=gb, beta=bb, A_log=A_log, dt_bias=dt_bias,
                    initial_state=None, **FLAGS)
do_b = maxdiff(ob, o64)
ds_b = maxdiff(stb, st64)
check("R2c chunk_kda 模型旗标 bf16-vs-fp64锚",
      do_b < 3e-2 * scale_o and ds_b < 5e-2 * st64.abs().max().item(),
      f"o_maxdiff={do_b:.2e} state_maxdiff={ds_b:.2e}（bf16 舍入量级）")

# fused_recurrent（decode 调用形态：不传 safe_gate → shim 默认 False，无截断）
FL_D = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, output_final_state=True,
            transpose_state_layout=True)
od, std = fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
                              initial_state=None, **FL_D)
od64, std64 = kda_ref_fp64(q, k, v, g, beta, A_log, dt_bias.view(H, K), None,
                           l2n=True, gate=True, beta_sig=True, safe_gate=False, lower_bound=None)
dod = maxdiff(od, od64)
check("R2c fused_recurrent decode形态(safe_gate缺省) fp32-vs-fp64锚",
      dod < 1e-3 * od64.abs().max().item() + 1e-4,
      f"o_maxdiff={dod:.2e}（确认 decode 路径无 clamp 与旗标一致）")

# 初态非零（decode 续算语义）
init = torch.randn(B, H, K, V)
o_i, st_i = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
                      initial_state=init, **FLAGS)
o_i64, st_i64 = kda_ref_fp64(q, k, v, g, beta, A_log, dt_bias.view(H, K), init,
                             l2n=True, gate=True, beta_sig=True, safe_gate=True, lower_bound=-5.0)
doi = maxdiff(o_i, o_i64)
check("R2c chunk_kda 带初态 fp32-vs-fp64锚",
      doi < 1e-3 * o_i64.abs().max().item() + 1e-4,
      f"o_maxdiff={doi:.2e}")


# ===========================================================================
# R3 flash_attention_2_cpu_equiv vs 毛坯 eager_attention_forward
# ===========================================================================
class FakeMod:
    def __init__(self, n_kv_groups, causal=True):
        self.num_key_value_groups = n_kv_groups
        self.is_causal = causal
        self.training = False


def bottom_right_mask(B, H, Sq, Sk, dtype):
    qpos = torch.arange(Sq).unsqueeze(-1)
    kpos = torch.arange(Sk).unsqueeze(0)
    causal = kpos <= qpos + (Sk - Sq)
    m = torch.zeros(B, 1, Sq, Sk, dtype=dtype)
    return m.masked_fill(~causal, float("-inf")).expand(B, H, Sq, Sk)


for dt, tol, tag in ((torch.float32, FP32, "fp32"), (torch.bfloat16, BF16, "bf16")):
    mod = FakeMod(2)
    q = torch.randn(2, 4, 6, 8, dtype=dt) * 0.5
    k = torch.randn(2, 2, 6, 8, dtype=dt) * 0.5
    v = torch.randn(2, 2, 6, 8, dtype=dt) * 0.5
    scaling = 8 ** -0.5
    mask = bottom_right_mask(2, 4, 6, 6, dt)
    out_e, probs_e = eager_attention_forward(mod, q, k, v, mask, scaling=scaling)
    out_m, _ = fa2_equiv(mod, q, k, v, attention_mask=None, scaling=scaling)  # 内建因果掩码
    d = maxdiff(out_e, out_m)
    check(f"R3 GQA prefill 因果 {tag}", d < tol["atol"] + tol["rtol"] * out_e.abs().max().item(),
          f"maxdiff={d:.2e}")

# GQA prefill 无因果（双向）：eager mask=None vs equiv is_causal=False
mod = FakeMod(2, causal=False)
q = torch.randn(2, 4, 6, 8) * 0.5
k = torch.randn(2, 2, 6, 8) * 0.5
v = torch.randn(2, 2, 6, 8) * 0.5
out_e, _ = eager_attention_forward(mod, q, k, v, None, scaling=8 ** -0.5)
out_m, _ = fa2_equiv(mod, q, k, v, attention_mask=None, scaling=8 ** -0.5)
d = maxdiff(out_e, out_m)
check("R3 无掩码（mask=None）fp32", d < 1e-5, f"maxdiff={d:.2e}")

# decode Sq=1（缓存续算，past=16）：全可见
mod = FakeMod(1)
q = torch.randn(2, 8, 1, 8) * 0.5
k = torch.randn(2, 8, 17, 8) * 0.5
v = torch.randn(2, 8, 17, 8) * 0.5
out_e, _ = eager_attention_forward(mod, q, k, v, None, scaling=8 ** -0.5)
out_m, _ = fa2_equiv(mod, q, k, v, attention_mask=None, scaling=8 ** -0.5)
d = maxdiff(out_e, out_m)
check("R3 decode Sq=1 全可见 fp32", d < 1e-5, f"maxdiff={d:.2e}")

# n_rep=1（MLA 实际 num_key_value_groups=1）
mod = FakeMod(1)
q = torch.randn(2, 8, 6, 8) * 0.5
k = torch.randn(2, 8, 6, 8) * 0.5
v = torch.randn(2, 8, 6, 8) * 0.5
mask = bottom_right_mask(2, 8, 6, 6, torch.float32)
out_e, _ = eager_attention_forward(mod, q, k, v, mask, scaling=8 ** -0.5)
out_m, _ = fa2_equiv(mod, q, k, v, attention_mask=None, scaling=8 ** -0.5)
d = maxdiff(out_e, out_m)
check("R3 n_rep=1 prefill 因果 fp32", d < 1e-5, f"maxdiff={d:.2e}")

# ===========================================================================
# 汇总
# ===========================================================================
n_pass = sum(1 for _, ok, _ in results if ok)
n_all = len(results)
print(f"\n=== 对拍汇总：{n_pass}/{n_all} 通过 ===")
if n_pass < n_all:
    sys.exit(1)
