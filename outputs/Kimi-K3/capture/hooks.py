# -*- coding: utf-8 -*-
"""Kimi-K3 抓图钩子（hooks.py）

职责（无模型语义改动，全部为环境适配与输入工厂）：
1. transformers 5.17.0 与毛坯（按 4.56 编写）的兼容补丁 4 项：
   - P1 OutputRecorder 迁移别名（transformers.utils.generic → output_capturing）
   - P2 create_causal_mask 形参/掩码机适配（input_embeds 改名、cache_position 剥除、
       KimiDynamicCache 非 Cache 协议时短路返回 None——等价 FA2 无显式掩码语义，
       因果性由注意力核保证；含 padding 的掩码场景不在本分析范围）
   - P3 tie_weights 签名兼容（5.17 init_weights 传 recompute_mapping=False）
   - P4 config._attn_implementation 回写（PretrainedConfig.__init__ 会重置子类先设值）
2. 等价替换（ replacements 声明见 spec.json / 最终 IR）：
   - R1 einops.rearrange 子集 shim（capture/shims/einops/，sys_path 导入级遮蔽）
   - R2 fla 包 shim：FusedRMSNormGated / ShortConvolution / chunk_kda /
       fused_recurrent_kda / prepare_* / tensor_cache（capture/shims/fla/）
   - R3 语言侧 FA2 纯 torch 等价函数注册（本机无 flash_attn 内核，注册表接管）
3. 缩水实例化配置（分支维度保真，规模缩水，声明见 spec.model.scale_note）。
4. make_inputs：三个场景 + 各 micro 作业的输入工厂；decode 系作业先用
   未追踪 prefill 构建 KimiDynamicCache 再返回输入。

毛坯目录只读：所有补丁均为内存级，不落盘、不写 __pycache__。
"""
import importlib
import math
import sys
import types

_PATCHED = False
_PKG_NAME = "kimi_k3_pkg"

# ---------------------------------------------------------------------------
# R3：语言侧 FA2 等价函数（纯 torch，CPU 可用；数学定义 softmax(QK^T·scaling)V，
# bottom-right 因果掩码；等价性说明见 spec.json 的 shims_notes）
# ---------------------------------------------------------------------------
def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    return x.repeat_interleave(dim=1, repeats=n_rep)


def flash_attention_2_cpu_equiv(module, query, key, value, attention_mask=None,
                                dropout=0.0, scaling=None, is_causal=None,
                                cu_seq_lens_q=None, cu_seq_lens_k=None,
                                max_length_q=None, max_length_k=None, **kwargs):
    import torch
    B, H, Sq, D = query.shape
    Sk = key.shape[2]
    scaling = scaling if scaling is not None else 1.0 / math.sqrt(D)
    n_rep = module.num_key_value_groups if hasattr(module, "num_key_value_groups") else 1
    k = _repeat_kv(key, n_rep)
    v = _repeat_kv(value, n_rep)
    scores = torch.matmul(query.float(), k.float().transpose(-2, -1)) * scaling
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        else:
            scores = scores + attention_mask.to(scores.dtype)
    elif getattr(module, "is_causal", False) and Sq > 1:
        # bottom-right 对齐因果掩码（decode Sq=1 时自然全可见）
        qpos = torch.arange(Sq, device=query.device).unsqueeze(-1)
        kpos = torch.arange(Sk, device=query.device).unsqueeze(0)
        causal = kpos <= qpos + (Sk - Sq)
        scores = scores.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs.to(v.dtype), v)
    return out.transpose(1, 2).contiguous(), None


def _apply_compat_patches():
    """兼容补丁（幂等）：必须在导入毛坯模块前后按序执行，见模块 docstring。"""
    global _PATCHED
    if _PATCHED:
        return
    import torch
    # P1：OutputRecorder 已从 transformers.utils.generic 迁移至 output_capturing
    from transformers.utils import output_capturing
    import transformers.utils.generic as tg
    if not hasattr(tg, "OutputRecorder"):
        tg.OutputRecorder = output_capturing.OutputRecorder

    # 合成包挂载毛坯目录（毛坯用相对导入，必须以包形式导入；目录只读不可加 __init__.py）
    mdir = str(_MODEL_DIR)
    pkg = types.ModuleType(_PKG_NAME)
    pkg.__path__ = [mdir]
    sys.modules[_PKG_NAME] = pkg

    lin = importlib.import_module(f"{_PKG_NAME}.modeling_kimi_linear")

    # P2：create_causal_mask 适配（毛坯按 4.56 形参调用：input_embeds/cache_position）
    from transformers import masking_utils
    _orig_ccm = masking_utils.create_causal_mask

    def create_causal_mask_compat(*args, **kwargs):
        if "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        kwargs.pop("cache_position", None)   # 5.17 已无此形参（由 past_key_values 推导）
        pkv = kwargs.get("past_key_values")
        if pkv is not None and not hasattr(pkv, "get_query_offset"):
            # KimiDynamicCache 非 transformers Cache 协议对象：掩码机无法处理，
            # 短路返回 None（等价 FA2 无显式掩码；因果性由注意力核保证）
            return None
        return _orig_ccm(*args, **kwargs)

    lin.create_causal_mask = create_causal_mask_compat   # 模型命名空间重绑

    k3 = importlib.import_module(f"{_PKG_NAME}.modeling_kimi_k3")

    # P3：tie_weights 签名兼容（5.17 init_weights → tie_weights(recompute_mapping=False)）
    _orig_tie = k3.KimiK3ForConditionalGeneration.tie_weights

    def _tie_compat(self, *args, **kwargs):
        return _orig_tie(self)

    k3.KimiK3ForConditionalGeneration.tie_weights = _tie_compat

    # R3：FA2 等价函数注册（语言侧构造期强制 flash_attention_2，见
    # modeling_kimi_linear.py:1110-1119；注册表为 MutableMapping 可写）
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_2_cpu_equiv

    _PATCHED = True


_MODEL_DIR = None

# ---------------------------------------------------------------------------
# 缩水实例化配置（分支维度保真；规模缩水逐项见 spec.model.scale_note）
# ---------------------------------------------------------------------------
TEXT_CFG = dict(
    model_type="kimi_linear", vocab_size=1000, hidden_size=512,
    intermediate_size=1024, num_hidden_layers=4,
    num_attention_heads=8, num_key_value_heads=8,
    hidden_act="situ", activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
    rms_norm_eps=1e-5, pad_token_id=0, use_cache=True, initializer_range=0.02,
    first_k_dense_replace=1, moe_layer_freq=1,
    num_experts=4, num_experts_per_token=2, num_shared_experts=2,
    moe_intermediate_size=128, routed_expert_hidden_size=256,
    latent_moe_use_norm=True, moe_renormalize=True,
    moe_router_activation_func="sigmoid", routed_scaling_factor=1.0,
    num_expert_group=1, topk_group=1, use_grouped_topk=True,
    q_lora_rank=64, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8,
    v_head_dim=16, mla_use_nope=True, mla_use_output_gate=True,
    attn_res_block_size=2,
    linear_attn_config=dict(
        head_dim=32, num_heads=8, short_conv_kernel_size=4,
        kda_layers=[1, 2, 3], full_attn_layers=[4],
        use_full_rank_gate=True, gate_lower_bound=-5.0,
    ),
    max_position_embeddings=4096, tie_word_embeddings=False,
)
VISION_CFG = dict(
    patch_size=14, init_pos_emb_height=8, init_pos_emb_width=8, init_pos_emb_time=2,
    pos_emb_type="divided_fixed", vt_num_attention_heads=4, vt_num_hidden_layers=2,
    vt_hidden_size=128, vt_intermediate_size=256, merge_kernel_size=[2, 2],
    merge_type="sd2_tpool", _attn_implementation="eager",   # 视觉走毛坯自带 eager 回退
    mm_projector_type="patchmergerv2", mm_hidden_size=128,
    projector_hidden_act="gelu", projector_ln_eps=1e-5,
    qkv_hidden_size=128, norm_type="rmsnorm", attn_bias=False,
    patch_embed_proj_bias=False, mlp_type="mlp2", linear_bias=False,
    activation_func="gelu_pytorch_tanh", pos_emb_interpolation_mode="bilinear",
    text_hidden_size=512,
)


def make_model(ctx):
    """完全接管构造：兼容补丁 → 缩水配置 → 实例化（bf16，seed 由驱动设定）。"""
    import torch
    global _MODEL_DIR
    _MODEL_DIR = ctx.model_dir
    _apply_compat_patches()

    k3 = importlib.import_module(f"{_PKG_NAME}.modeling_kimi_k3")
    cfg = k3.KimiK3Config(text_config=dict(TEXT_CFG), vision_config=dict(VISION_CFG),
                          media_placeholder_token_id=998, pad_token_id=999,
                          ignore_index=-100)
    # P4：5.17 构造期会重置 attn 实现——显式回写（视觉走毛坯自带 eager 回退）
    cfg._attn_implementation_internal = "eager"
    cfg.vision_config._attn_implementation_internal = "eager"

    model = k3.KimiK3ForConditionalGeneration(cfg)
    ctx.log(f"缩水模型构造完成：参数量 "
            f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M，"
            f"dtype={next(model.parameters()).dtype}")
    return model


# ---------------------------------------------------------------------------
# 输入工厂（按 job id 分派；宏作业按场景）
# ---------------------------------------------------------------------------
def _grid_thws(ctx):
    import torch
    # 图1 (t=1,h=8,w=8)=64 patch（=pos_emb 尺寸，直取分支）；图2 (t=2,h=2,w=2)=8 patch
    #（≠8×8，双线性插值分支）→ L=72
    return torch.tensor([[1, 8, 8], [2, 2, 2]])


def _vision_macro_inputs(model, ctx):
    import torch
    grid = _grid_thws(ctx)
    pixel_values = torch.randn(72, 3, 14, 14, dtype=torch.bfloat16)
    B, S = 2, 12
    input_ids = torch.randint(10, 900, (B, S))
    input_ids[0, 3] = 998   # 每行 1 个占位符（数量必须等于图片数）
    input_ids[1, 5] = 998
    attention_mask = torch.ones(B, S, dtype=torch.long)
    return dict(input_ids=input_ids, pixel_values=pixel_values, grid_thws=grid,
                attention_mask=attention_mask, use_cache=True)


def _text_prefill_inputs(model, ctx):
    import torch
    return dict(input_ids=torch.randint(10, 900, (2, 16)),
                attention_mask=torch.ones(2, 16, dtype=torch.long),
                use_cache=True)


def _build_text_cache(model, ctx):
    """未追踪 prefill 构建 KimiDynamicCache（decode 系作业的前置）。"""
    import torch
    inputs = _text_prefill_inputs(model, ctx)
    with torch.no_grad():
        out = model(**inputs)
    return out.past_key_values


def make_inputs(scenario, model, ctx):
    """返回 tuple（位置输入，按各 forward 签名的位置顺序排列）。
    注：torchvista 仅按 tuple 展开调用（model(*inputs)），dict 不可用。
    """
    import torch
    job_id = ctx.current_job.get("id", "")
    H = 512   # 缩水 hidden

    # ---- 宏作业（根 forward 签名位置：input_ids, pixel_values, grid_thws,
    #      attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache）----
    if job_id == "macro_prefill_vision":
        grid = _grid_thws(ctx)
        pixel_values = torch.randn(72, 3, 14, 14, dtype=torch.bfloat16)
        B, S = 2, 12
        input_ids = torch.randint(10, 900, (B, S))
        input_ids[0, 3] = 998   # 每行 1 个占位符（数量必须等于图片数）
        input_ids[1, 5] = 998
        return (input_ids, pixel_values, grid,
                torch.ones(B, S, dtype=torch.long),
                None, None, None, None, True)
    if job_id == "macro_prefill_text":
        return (torch.randint(10, 900, (2, 16)), None, None,
                torch.ones(2, 16, dtype=torch.long),
                None, None, None, None, True)
    if job_id == "macro_decode":
        cache = _build_text_cache(model, ctx)      # 未追踪 prefill → KimiDynamicCache
        return (torch.randint(10, 900, (2, 1)), None, None,
                torch.ones(2, 17, dtype=torch.long),
                None, cache, None, None, True)

    # ---- 视觉 micro（prefill_vision）----
    if job_id == "micro_vit_patch_embed":
        return (torch.randn(72, 3, 14, 14, dtype=torch.bfloat16), _grid_thws(ctx))
    if job_id == "micro_vit_pos_emb":
        # (2,4) 网格 ≠ 8×8 → 双线性插值分支；t=1 不走时间 repeat
        return (torch.randn(8, 128, dtype=torch.bfloat16),
                torch.tensor([[1, 2, 4]]))
    if job_id == "micro_vit_layer0":
        grid = _grid_thws(ctx)
        cu = torch.cat([torch.zeros(1, dtype=torch.int32),
                        grid.prod(dim=1).cumsum(dim=0).to(torch.int32)])
        freqs = model.vision_tower.encoder.rope_2d.get_freqs_cis(
            grid_thws=grid, device="cpu")
        return (torch.randn(72, 128, dtype=torch.bfloat16), cu, 64, freqs)
    if job_id == "micro_projector":
        # 逐图 list[(n_i, 4, Cv)]：图1 16 token、图2 4 token
        return ([torch.randn(16, 4, 128, dtype=torch.bfloat16),
                 torch.randn(4, 4, 128, dtype=torch.bfloat16)],)

    # ---- 语言层 micro（prefill_text）----
    # 层 forward 位置签名：(hidden_states, attention_mask, position_ids,
    #   past_key_values, output_attentions, use_cache, block_residual)
    if job_id == "micro_layer_kda0":       # KDA + dense（layer 0）
        # 全模型中 block_residual 由 KimiLinearModel.forward 初始化为 (T,0,H) 空张量
        #（modeling_kimi_linear.py:1188-1192），独立跑层保持一致
        return (torch.randn(2, 16, H, dtype=torch.bfloat16),
                None, None, None, False, False,
                torch.zeros(32, 0, H, dtype=torch.bfloat16))
    if job_id in ("micro_layer_mla3", "micro_layer_kda_moe1"):
        # 独立跑层需给 block_residual（全模型中由 layer 0 生成）
        return (torch.randn(2, 16, H, dtype=torch.bfloat16),
                None, None, None, False, False,
                torch.zeros(32, 1, H, dtype=torch.bfloat16))
    # KDA forward 位置签名：(hidden_states, attention_mask, cache_params)
    if job_id == "micro_attn_kda":         # KDA chunk 模式
        return (torch.randn(2, 16, H, dtype=torch.bfloat16), None, None)
    # MLA forward 位置签名：(hidden_states, attention_mask, position_ids, past_key_values)
    if job_id == "micro_attn_mla":         # MLA 无缓存
        return (torch.randn(2, 16, H, dtype=torch.bfloat16),)
    if job_id == "micro_attn_kda_decode":  # KDA fused_recurrent 模式
        cache = _build_text_cache(model, ctx)
        return (torch.randn(2, 1, H, dtype=torch.bfloat16), None, cache)
    if job_id == "micro_attn_mla_decode":  # MLA 缓存续算
        cache = _build_text_cache(model, ctx)
        return (torch.randn(2, 1, H, dtype=torch.bfloat16), None, None, cache)
    if job_id == "micro_moe_gate":
        return (torch.randn(2, 16, H, dtype=torch.bfloat16),)
    if job_id == "micro_expert0":          # 单专家 (n, El=256)
        return (torch.randn(5, 256, dtype=torch.bfloat16),)
    if job_id == "micro_mlp_dense":        # layer 0 dense FFN
        return (torch.randn(2, 16, H, dtype=torch.bfloat16),)
    if job_id == "micro_embed":
        return (torch.randint(10, 900, (2, 16)),)
    if job_id == "micro_norm":
        return (torch.randn(2, 16, H, dtype=torch.bfloat16),)
    if job_id == "micro_lm_head":
        return (torch.randn(2, 16, H, dtype=torch.bfloat16),)

    raise ValueError(f"hooks.make_inputs：未识别的作业 {job_id!r}（scenario={scenario!r}）")
