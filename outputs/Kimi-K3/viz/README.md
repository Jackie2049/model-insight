# Kimi-K3 算子分析 · 可视化页面导览（viz/）

本目录是**阶段五（S4b）交付**：把融合**定稿**信息（等价替换标记 + 融合状态徽标 +
融合面板）注入 TorchVista 抓图页面后的 HTML 副本。抓图原件与逐节点实测数据
保留在 [`../capture/`](../capture/)（`*.measured.json`），融合结论见
[`../fusion/final_ir.json`](../fusion/final_ir.json)（attempt 3 定稿，ir_schema.py
校验通过）与 [`../fusion/diff_report.md`](../fusion/diff_report.md)（含复核记录）。

- 模型：Kimi-K3（`KimiK3ForConditionalGeneration` = MoonViT3d 视觉塔 +
  PatchMergerMLPV2 投影器 + KimiLinearForCausalLM 93 层 KDA/MLA 混合语言模型；
  抓图为缩水实例化——分支维度保真、规模缩水，逐项声明见各页面右上角
  融合面板与 `final_ir.json` 的 `model.scale_note`）
- 场景三支：`prefill_vision`（图文前填）/ `prefill_text`（纯文本前填）/ `decode`（单步解码）
- 生成命令（可重跑，幂等）：

  ```bash
  .venv/bin/python agent-kit_ops-analysis/scripts/render_fusion.py \
      --html outputs/Kimi-K3/capture \
      --ir   outputs/Kimi-K3/fusion/final_ir.json \
      --spec outputs/Kimi-K3/capture/spec.json
  ```

## 入口与推荐阅读顺序

1. **`macro_prefill_vision.html`** —— 全模型含图前填：视觉塔 → 投影器 → 图文合并 →
   语言层 → lm_head 的完整链路，看"结构全貌"从这里开始。
2. **`macro_prefill_text.html`** —— 纯文本前填：dense layer0 + MoE 层 + attn_res 机制。
3. **`macro_decode.html`** —— 单步解码（缓存续算）：KDA fused_recurrent、MLA cache、
   短卷积缓存读写、attn_res 增量路径。
4. 逐层细看用 micro 页面（下表），每页聚焦一个模块/层型，节点少、便于对照
   `*.measured.json` 逐记录核对。

## 页面清单（20 个）

| 页面 | 场景 | 覆盖内容 / 看点 |
|---|---|---|
| `macro_prefill_vision.html` | prefill_vision | 全模型：视觉塔 2 层（rope 预计算 262144 表 + 逐图切片）、tpool 合并、投影器、图文合并、4 语言层（KDA 双支路：文本 (2,27) + 视觉 unpad (1,39)）、attn_res、末层 norm、lm_head |
| `macro_prefill_text.html` | prefill_text | 纯文本 (2,16)：embed、dense layer0（SituAndMul I=1024）、MoE 层（路由 fp32/top-2/专家分派）、attn_res、lm_head |
| `macro_decode.html` | decode | 缓存续算 (2,1)：KDA fused_recurrent、conv cache (2,256,3)、MLA cache 拼接、attn_res (T,2,H) 归一化打分 |
| `micro_vit_patch_embed.html` | prefill_vision | patch_embed：Conv2d 输入 (72,3,14,14) → view (72,128) + pos_emb 加法 |
| `micro_vit_pos_emb.html` | prefill_vision | pos_emb 双线性插值分支：(2,4) 网格 (1,128,8,8)→(1,128,2,4)、时间 sincos repeat |
| `micro_vit_layer0.html` | prefill_vision | 视觉单层全链：norm0（原生 rms_norm 融合算子）、wqkv view/unbind、apply_rope 复数旋转、eager 注意力（bool 掩码 + QK^T/√d）、wo、MLP2、norm1 |
| `micro_projector.html` | prefill_vision | 投影器：list[(n,4,128)] → view/concat (20,512) → proj.0/gelu/proj.2 → post_norm → split |
| `micro_layer_kda0.html` | prefill_text | KDA+dense 整层（layer0）：q/k/v 投影、短卷积、f_a/f_b/beta、chunk_kda、o_norm、o_proj、dense FFN、block_residual 空张量起步 |
| `micro_layer_kda_moe1.html` | prefill_text | KDA+MoE 整层（layer1）：含路由 gate、专家分派簿记（argsort/scatter/切片）、routed_norm、共享专家 |
| `micro_layer_mla3.html` | prefill_text | MLA+MoE 整层（layer3）：q/kv 低秩拆装、q_a/kv_a layernorm、kv_b、FA2 等价注意力（无显式掩码）、sigmoid 门控、o_proj |
| `micro_attn_kda.html` | prefill_text | KDA 注意力单模块（chunk 模式）(2,16) |
| `micro_attn_mla.html` | prefill_text | MLA 注意力单模块（无缓存）(2,16) |
| `micro_attn_kda_decode.html` | decode | KDA fused_recurrent 模式 + conv cache 读写（(2,256,3)+(2,256,1)→(2,256,4)→conv→切片） |
| `micro_attn_mla_decode.html` | decode | MLA 缓存续算（key/value 拼接 + 注意力） |
| `micro_moe_gate.html` | prefill_text | MoE 路由：hidden.float()/weight.float() → fp32 logits (T,4) → sigmoid → top-2 → renormalize |
| `micro_expert0.html` | prefill_text | 单专家 (5,256)：w1/w3 → cat (5,256)→(5,256)→ situ → w2 |
| `micro_mlp_dense.html` | prefill_text | dense FFN：gate/up (2,16,1024) → cat (2,16,2048) → situ → down |
| `micro_embed.html` | prefill_text | embed_tokens 查表 (2,16)→(2,16,512) |
| `micro_norm.html` | prefill_text | KimiRMSNorm 的 fp32 内部算子链（float/pow/mean/add/rsqrt/mul/to/weight mul） |
| `micro_lm_head.html` | prefill_text | lm_head (2,16,512)→(2,16,1000) |

## 融合面板读法（每页右上角）

- **模型与缩水声明**：`model.name` + `model.scale_note`（分支保真逐条 + 规模缩水逐项
  数值对照，含 proj 中间维 512 的修正说明）。
- **状态计数**：定稿分布 **verified 64 / llm_only 5 / mismatch 0 / conflict 0**
  （attempt 3 三问归因后，42 个机械 mismatch 全部裁决为 verified，归因见
  `../fusion/diff_report.md` 复核记录）。
- **等价替换清单**：4 项——R1 einops rearrange 子集 / R2 fla 六组件（FusedRMSNormGated、
  ShortConvolution、chunk_kda、fused_recurrent_kda 等）/ R3 语言侧 FA2 CPU 等价函数 /
  P transformers 5.17.0 兼容补丁，每项含数学等价说明与验证等级
  （全部为"对拍数学定义"级，留痕见 `../capture/shims/README.md`）。
- **不一致 / 未测清单**：mismatch 为空；未测（llm_only）5 个节点列出路径。

## 节点标记读法

- **绿描边**：verified（该显示名在本场景的全部实测记录与逻辑一致；按显示名聚合，
  同名实例全部打标，折叠/展开重绘后由 MutationObserver 自动重打）。
- **无红虚线的说明（重要）**：红虚线是"类挂树 shim 节点"的标记，本模型的替换件均为
  **导入级/注册级**（`spec.shims=[]`，见 spec.notes），图中不存在挂树的 shim 节点，故
  无红虚线目标——这是忠实输出而非遗漏。替换件在图中的具象体现：
  - R2 fla shim：`ShortConvolution` 模块标签（q/k/v_conv1d）及其域内 `conv1d`、
    `silu` 算子；o_norm 的 fp32 归一化 + sigmoid 门控算子链；
  - R3 FA2 等价函数：MLA 图中的 fp32 `matmul/softmax/transpose` 算子（函数调用本身
    不登记为算子节点）；
  - R1 einops shim：KDA 域内的 `view/permute/reshape` 算子。
  对应关系已写入 `final_ir.json` 的 `contract.replacements`（各节点 branch_note 亦注明）。
- **llm_only 节点**：面板有"仅逻辑分析、尚未运行验证"提示（未执行 ≠ 不存在）：
  `vision_tower.encoder.{j}.mlp` / `.activation`（gelu 算子未被抓图引擎登记，仅有源码
  与形状链旁证）、`vision_tower.encoder.rope_2d`（普通函数无自有 scope，encoder 记录
  为旁证）、`merge`（根域记录无法逐条区分归因）、`lm.cache`（非 Module 状态容器，
  无前向算子）。

## 交互与场景注意点

- 图支持缩放/平移、模块折叠展开；打标随重绘自动更新。
- `prefill_vision` 语言层为**双支路**：文本 (2,27) 与视觉 unpad (1,39) 共享同一层模板，
  同名算子会出现两套 shape——这是 `attention_mask` 非 None 时的 unpad 展平分支
  （`modeling_kimi_linear.py:566-574`），非形状矛盾。
- `decode` 页短卷积出现 (2,1,256) 与 (2,256,·) 两种布局：前者是模块 I/O 域 (B,S,P)，
  后者是卷积域 (B,P,K')，dim1 语义不同。
- 规模为缩水实例化（H=512、4 层、4 专家等），**数值结论不外推**；全量对照见
  `final_ir.json` → `model.config_snapshot` → `config_json_全量对照`。

> 事后补充（kit 维护者）：`index.html` 为聚合面板，由 `render_index.py` 生成
> （场景/视图切换、iframe 预览、分析摘要），浏览各页面从它进入即可。
