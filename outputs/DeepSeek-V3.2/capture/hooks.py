# -*- coding: utf-8 -*-
"""抓图 hooks · DeepSeek-V3.2（spec.json 的 hooks 字段指向本文件）

make_model(ctx)：读 671B 配置 json + 声明式缩水覆盖 → ModelArgs → Transformer，
  并做「量化一致」的确定性随机初始化（fp8 权重与其 128 块 scale 数学自洽，
  规避 torch.empty 垃圾位产生 NaN/Inf 毒化前向）。未加载真实权重（任务即
  随机初始化分析 shape/dtype，已声明进 IR scale_note）。
make_inputs(scenario, model, ctx)：macro 场景构造 (tokens, start_pos)；
  micro 场景按 target 模块签名构造块级输入（h/residual/freqs_cis/mask）。

缩水声明（671B 全量 → 抓图实例化，分支维度保真、规模维度缩水）：
  | 字段 | 671B | 抓图 | 说明 |
  |---|---|---|---|
  | vocab_size | 129280 | 1024 | 规模缩水 |
  | n_layers | 61 | 4 | 层 0-2 稠密 + 层 3 MoE，保留真实层型边界 |
  | n_heads | 128 | 8 | 规模缩水（头结构不变：nope128/rope64/v128） |
  | n_routed_experts | 256 | 32 | 8 组×每组 4 专家，保真分组限制逻辑 |
  | inter_dim | 18432 | 1024 | 规模缩水 |
  | moe_inter_dim | 2048 | 256 | 规模缩水 |
  | index_n_heads | 64 | 8 | 规模缩水（index_head_dim=128 不变） |
  | index_topk | 2048 | 8 | 使 V=min(topk,T) 走 topk 侧，覆盖稀疏分支 |
  | max_batch_size | 8 | 1 | 规模缩水 |
  | max_seq_len | 16384 | 512 | 配合 original_seq_len=256 触发 yarn/mscale 分支 |
  | original_seq_len | 4096 | 256 | 同上（max_seq_len > original_seq_len 分支） |
  保留不变：n_dense_layers=3、n_activated_experts=8、n_expert_groups=8、
  n_limited_groups=4、score_func=sigmoid、route_scale=2.5、q_lora_rank=1536、
  kv_lora_rank=512、qk_nope=128/qk_rope=64/v=128、index_head_dim=128、
  rope 参数（theta 10000/factor 40/beta 32,1）、mscale=1.0、dtype=fp8、
  scale_fmt=ue8m0、dim=7168（保真 Gate 的 dim==7168 bias 分支）。
"""
import json
import math
from pathlib import Path

import torch

SEED_WEIGHTS = 20261010          # 权重初始化独立种子（不依赖构造期 RNG 顺序）
BLOCK = 128
FP8_MAX = 448.0


def _ue8m0_round(scale: torch.Tensor) -> torch.Tensor:
    """s → 2^ceil(log2(s))，与 kernel.py fast_round_scale 位运算语义一致。"""
    bits = scale.float().view(torch.int32)
    exp = (torch.bitwise_right_shift(bits, 23) & 0xFF) - 127
    man = torch.bitwise_and(bits, (1 << 23) - 1)
    ceil_log2 = exp + (man != 0).to(torch.int32)
    return ((ceil_log2 + 127) << 23).view(torch.float32)


def _init_fp8_weight(weight: torch.nn.Parameter, scale_param: torch.nn.Parameter, gen):
    """fp8 权重与 128 块 scale 的量化一致初始化：
    先生成 fp32 权重 → 逐 128×128 块 amax → ue8m0 尺度 → clamp(±448) 量化。
    维度不整除 128 时按 ceil 补齐（与 Linear.__init__ L185-186 的 ceil 布局一致）。"""
    o, i = weight.shape
    w32 = torch.randn(o, i, generator=gen, dtype=torch.float32) * 0.02
    so, si = (o + BLOCK - 1) // BLOCK, (i + BLOCK - 1) // BLOCK
    po, pi = so * BLOCK - o, si * BLOCK - i
    wp = torch.nn.functional.pad(w32, (0, pi, 0, po))
    blk = (wp.view(so, BLOCK, si, BLOCK).permute(0, 2, 1, 3)
             .reshape(-1, BLOCK * BLOCK))
    amax = blk.abs().amax(dim=-1).clamp(min=1e-4).view(so, si)
    scale = _ue8m0_round(amax / FP8_MAX) if scale_param is not None else None
    scale_exp = (scale.repeat_interleave(BLOCK, dim=-2)
                     .repeat_interleave(BLOCK, dim=-1))[:o, :i]
    wq = torch.clamp(w32 / scale_exp, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    weight.data.copy_(wq)
    scale_param.data.copy_(scale)


def _init_weights(model, seed: int):
    """遍历参数做确定性初始化；RMSNorm/LayerNorm 的 ones/zeros 权重保持不动。"""
    from model import RMSNorm, LayerNorm
    gen = torch.Generator().manual_seed(seed)
    for mod in model.modules():
        if isinstance(mod, (RMSNorm, LayerNorm)):
            continue
        fp8_weight = getattr(mod, "weight", None)
        if (isinstance(fp8_weight, torch.nn.Parameter)
                and fp8_weight.dtype == torch.float8_e4m3fn):
            _init_fp8_weight(fp8_weight, mod.scale, gen)
        for name, p in list(mod.named_parameters(recurse=False)):
            if name == "scale" and fp8_weight is not None:
                continue                      # 已随 fp8 weight 分支处理
            if name == "weight" and isinstance(fp8_weight, torch.nn.Parameter):
                continue
            with torch.no_grad():
                p.copy_((torch.randn(*p.shape, generator=gen, dtype=torch.float32)
                         * 0.02).to(p.dtype))


def make_model(ctx):
    """按 671B 配置 + 缩水覆盖构造 Transformer（完全接管构造）。"""
    cfg = json.loads((ctx.model_dir / "inference" / "config_671B_v3.2.json").read_text())
    shrink = {
        "vocab_size": 1024, "n_layers": 4, "n_heads": 8,
        "n_routed_experts": 32, "inter_dim": 1024, "moe_inter_dim": 256,
        "index_n_heads": 8, "index_topk": 8,
        "max_batch_size": 1, "max_seq_len": 512, "original_seq_len": 256,
    }
    cfg.update(shrink)
    ctx.log(f"构造参数（671B json + 缩水覆盖）：{cfg}")

    import model as model_mod          # 延迟导入：sys_path 遮蔽与 init_distributed 已就绪
    args = model_mod.ModelArgs(**cfg)
    m = model_mod.Transformer(args)
    _init_weights(m, SEED_WEIGHTS)
    ctx.log("已完成量化一致随机初始化（未加载真实权重，任务声明内）")
    return m


def _causal_mask(seqlen: int) -> torch.Tensor:
    return torch.full((seqlen, seqlen), float("-inf")).triu_(1).to(torch.bfloat16)


def make_inputs(scenario, model, ctx):
    """macro：prefill → (tokens[B,64], 0)；decode → (tokens[B,1], 64)。
    micro：按 target 模块的 forward 签名构造块级小输入。"""
    job = ctx.current_job
    kind = job.get("kind", "macro")

    if kind == "micro":
        target = job["target"]
        sub = model.get_submodule(target)
        bsz, seqlen = 1, 64
        dim = getattr(sub, "dim", None)
        if dim is None and hasattr(sub, "attn"):
            dim = sub.attn.dim
        if dim is None:
            raise ValueError(f"无法推断 micro target {target!r} 的输入维度")
        h = (torch.randn(bsz, seqlen, dim, dtype=torch.bfloat16) * 0.5)
        freqs_cis = model.freqs_cis[0:seqlen]
        mask = _causal_mask(seqlen)
        if hasattr(sub, "attn") and hasattr(sub, "ffn"):        # Block
            residual = (torch.randn(bsz, seqlen, dim, dtype=torch.bfloat16) * 0.5)
            return (h, residual, 0, freqs_cis, mask)
        if hasattr(sub, "indexer"):                             # MLA
            return (h, 0, freqs_cis, mask)
        return (h,)                                             # MLP / MoE 单输入

    vocab = model.head.out_features
    if scenario == "prefill":
        tokens = torch.randint(0, vocab, (1, 64))
        return (tokens, 0)
    if scenario == "decode":
        tokens = torch.randint(0, vocab, (1, 1))
        return (tokens, 64)
    raise ValueError(f"未知场景：{scenario!r}")
