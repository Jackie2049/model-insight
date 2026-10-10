# shims/ — Kimi-K3 抓图替换件说明

本目录存放抓图阶段对**不可用外部依赖**的替换件与等价验证留痕。所有替换均为
数学等价重建（纯 torch 实现），不改变毛坯建模语义；覆盖范围与声明见
`../spec.json`（notes.shims 说明、model.scale_note）与最终 `../../fusion/final_ir.json`
的 `contract.replacements`。

## 1. 清单

| 项 | 位置 | 替换对象 | 生效机制 |
|---|---|---|---|
| R1 | `einops/`（`rearrange.py`） | `einops.rearrange` | 导入级遮蔽（sys_path 首位，毛坯 `import einops` 即命中） |
| R2 | `fla/modules.py` | `fla.modules.FusedRMSNormGated`、`ShortConvolution` | 同上（毛坯 `fla` 导入失败即 raise，无回退分支） |
| R2 | `fla/ops/kda.py` | `fla.ops.kda.chunk_kda`、`fused_recurrent_kda` | 同上 |
| R2 | `fla/ops/utils/`、`fla/utils.py` | `index` 等辅助 | 同上（毛坯 import 链所需） |
| R3 | `../hooks.py::flash_attention_2_cpu_equiv` | `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS["flash_attention_2"]` | 注册表键覆盖（hooks._apply_compat_patches 执行；毛坯 MLA 构造期无条件强制 FA2，modeling_kimi_linear.py:1110-1119） |
| P1-P4 | `../hooks.py` | transformers 5.17.0 兼容补丁（OutputRecorder 别名 / create_causal_mask 适配与 KimiDynamicCache 短路 / tie_weights 签名 / config._attn_implementation 回写） | 内存级 monkey-patch，非语义替换 |

说明：`spec.json` 的 `shims: []` 为空是因为这些替换不适用"类挂树"机制——
R1/R2 以**整个包**被导入接管（毛坯代码无任何回退分支可选），R3 是注册表键覆盖。

## 2. 数学定义

每个 shim 文件头部含逐步数学定义注释（对拍参考即按这些定义独立手写）：

- `einops/rearrange.py`：单轴/括号分组/省略号的拆分-置换-合并三段式语义。
  仅实现毛坯实际用到的模式类（`b t (h d) -> b h t d`、`... (h d) -> ... h d`、
  `b t h d -> b t (h d)` 等），不追平 einops 全量语法。
- `fla/modules.py::FusedRMSNormGated`：`y = rmsnorm(x) * act(o)`，内部 fp32，输出回输入 dtype。
- `fla/modules.py::ShortConvolution`：因果 depthwise Conv1d（k=4）+ silu；
  cache = 上一步**原始输入**（未激活）尾部 (B, D, K-1)；varlen（cu_seqlens）逐段因果。
- `fla/ops/kda.py`：Kimi Delta Attention 门控 DeltaNet 逐步递推——
  q/k l2norm → beta=sigmoid → `g = -exp(A_log)·softplus(g_raw + dt_bias)`（clamp 至
  lower_bound）→ `S' = exp(g)⊙S` → `u = k·S'` → `S = S' + beta·k⊗(v-u)` → `o = Sᵀq`；
  状态布局 (B, H, K, V) fp32 累积，输出/终态回输入 dtype。

## 3. 等价验证（已执行，留痕见 `equiv_check.py` / `equiv_check_result.txt`）

**验证等级（如实声明）：数学定义级。** 参考实现按上述数学定义独立手写
（逐通道/逐 head 显式循环、fp64 数值锚，与 shim 的 einsum/conv1d 向量化路径不同实现），
验证"实现符合其声称的数学定义"。**不验证与 fla 原生 Triton 核的数值一致**——
fla 未安装且不允许联网，原生核内部精确数学（分块策略、gate 组合式、状态转置布局）
无法核对；此为已知开放项（见 `../../llm-analysis/notes.md` 开放问题 1 与 SELF-CHECK）。

对拍结果（2026-10-10，`.venv` + `PYTHONDONTWRITEBYTECODE=1`，21/21 通过）：

| 组 | 用例 | 结果 |
|---|---|---|
| R1 einops | 拆分/合并/省略号/置换 + 毛坯实际 2 种模式，共 6 项 | 全部 `torch.equal` 逐位一致 |
| R2a ShortConv | 有/无 cache × fp32/bf16，输出+终态 8 项 | fp32 maxdiff ≤ 2.4e-7；bf16 逐位一致（同为 fp32 累积 + conv1d） |
| R2b FusedRMSNormGated | fp32/bf16 vs 公式 | maxdiff = 0 |
| R2c chunk_kda | 模型真实旗标（l2norm+gate+beta_sigmoid+safe_gate(-5)）×（fp32 / bf16 / 带初态）vs fp64 锚 | fp32 o_maxdiff 1.2e-7（\|o\|max 1.48）；bf16 7.6e-3 = bf16 舍入量级；带初态一致 |
| R2c fused_recurrent | decode 调用形态（**不传 safe_gate**） | 与 safe_gate=False 参考一致（1.2e-7），确认旗标行为 |
| R3 FA2 等价 | vs 毛坯自带 `eager_attention_forward` 同输入：GQA prefill 因果（fp32/bf16）、无掩码、decode Sq=1 全可见、n_rep=1 | fp32 全部 maxdiff = 0；bf16 3.9e-3（分数累积 dtype：等价函数 fp32 vs eager 输入 dtype，声明的差异点） |

### 已知差异与旗标注意项（诚实披露）

1. **R2c decode 旗标**：毛坯 fused_recurrent 调用（modeling_kimi_linear.py:629-645）
   **不传 `safe_gate`**，chunk 调用传 `safe_gate=True`。shim 默认 `safe_gate=False`
   （fla 原生签名默认值无法核对），故 decode 路径无 clamp——与毛坯调用方式自洽，
   但若 fla 原生默认值不同则行为有偏差。已记入 fusion concerns。
2. **R2c 状态布局**：shim 固定 (B, H, K, V)，`transpose_state_layout` 参数记录但不改变行为。
   fla 原生 `transpose_state_layout=True` 的实际布局约定未验证；缓存状态仅在本
   分析闭环内自洽使用（写入与读取均经 shim），不影响 shape/dtype 结论。
3. **R3 与 eager 的实现差异**（语义等价、数值纪律不同）：等价函数分数累积固定 fp32、
   probs 在 `probs.to(v.dtype)` 后与 v 相乘；eager 分数在输入 dtype 累积、probs 先
   `.to(query.dtype)`。生产路径真实后端为 flash-attn 2 Triton 核（分块 softmax），
   两者与其数值均不相同；本分析声明的等价对象是 softmax(QK^T·scaling)V 的因果
   注意力**数学语义**（bottom-right 掩码对齐）。
4. **einops 覆盖为子集**：仅实现毛坯调用点用到的模式类；未用语法（如 ellipsis 中缀
   混合、repeat 语义）不在范围，误用会显式 raise 而非静默错算。

## 4. 复跑方式

```bash
cd <工作区根>
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/Kimi-K3/capture/shims/equiv_check.py
```

脚本从本目录位置运行即复现全部 21 项（最近一次复跑 exit 0，见 equiv_check_result.txt）。
