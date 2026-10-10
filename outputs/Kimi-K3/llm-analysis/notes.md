# S2a 逻辑分析推导笔记 · Kimi-K3

时间：2026-10-10（本笔记为逻辑推导引擎的独立原始记录，融合后不可回改；修正一律以追加条目方式记录）

## 一、阅读清单（S2a 阶段实际读过的材料）

- agent-kit_ops-analysis/SKILL.md（操作总纲）
- agent-kit_ops-analysis/prompts/logical-analysis.md（本阶段规程）
- outputs/Kimi-K3/audit/audit_report.json（S0 静态审计产物）
- inputs/Kimi-K3/ 下：config.json、DRIVE.md、configuration_kimi_k3.py、modeling_kimi_linear.py（全文 1315 行）、modeling_kimi_k3.py（全文 1318 行）
- 其余毛坯文件（kimi_k3_processor.py、kimi_k3_vision_processing.py、media_utils.py、tokenization_kimi.py、encoding_k3.py、generation_config.json、model.safetensors.index.json、README.md、*.json 配置）仅作通读/定位，不进入建模前向路径，未逐行推导。

声明：本阶段未读任何 capture/ 产物（当时 capture/ 尚不存在）。

## 二、模型总体结构（源码事实）

入口类 `KimiK3ForConditionalGeneration`（modeling_kimi_k3.py:893，config.json architectures 也指向它），
是一个 LLaVA 式三段 VLM：

1. **视觉塔** `vision_tower = MoonViT3dPretrainedModel`（modeling_kimi_k3.py:649）
   - `patch_embed = MoonVision3dPatchEmbed`（:286）：`proj` 为 Conv2d(3→1024, k=14, s=14, bias=False)
     （:308-312），输入已是处理器切好的 patch 序列 (L, 3, 14, 14)，卷积后 view 成 (L, 1024)
     （:335）；`pos_emb = Learnable2DInterpPosEmbDivided_fixed`（:233）做 2D 可学习位置嵌入 +
     时间维 sincos（:247-253），按 grid_thws 逐图插值/平铺后加到 x 上（:260-283）。
   - `encoder = MoonViT3dEncoder`（:567）：`rope_2d = Rope2DPosEmbRepeated`（:341，无 forward，
     get_freqs_cis 产出 complex64 频率表 :378-434）；27 个 `MoonViTEncoderLayer`（:461）；
     `final_layernorm` RMSNorm（:587-591）。encoder.forward 内由 grid_thws 计算
     cu_seqlens（:603-610），逐层 packed varlen 前向（:611-615）。
   - 每个 MoonViTEncoderLayer：norm0(RMSNorm) → attention_qkvpacked（wqkv 一次投影 3×qkv
     （:502-505, :519-528），unbind 成 q/k/v（:528），apply_rope 复数旋转（:172-193），
     VL_VISION_ATTENTION_FUNCTIONS[attn_implementation]（:136-139, :532）→ wo）→ 残差 →
     norm1(RMSNorm) → MLP2（fc0→act→fc1, :437-458）→ 残差。
   - 视觉塔 forward 末端：`tpool_patch_merger`（:621-646，普通函数非 Module），merge_type
     'sd2_tpool'：时间维 mean 池化 + 2×2 空间合并，输出 list[每图 (h/2·w/2, 4, C)]（:709-713）。
2. **投影器** `mm_projector = PatchMergerMLPV2`（:783）：list 输入 concat 成 (N, 4·1024=4096)
   → Linear(4096→4096, bias=False) → GELU → Linear(4096→7168, bias=False) → post_norm
   RMSNorm(7168)（:790-815），split 回 per-image list。
3. **图文合并** `_merge_input_ids_with_image_features`（:958-1090，普通方法）：占位符 token
   （media_placeholder_token_id=163605）按 feature_lengths 展开成图像特征，产出
   final_embedding / final_attention_mask / position_ids；pad 位置清零（:1082-1085）。
4. **语言模型** `language_model = KimiLinearForCausalLM`（modeling_kimi_linear.py:1236）
   → `model = KimiLinearModel`（:1090）：
   - `embed_tokens` nn.Embedding(163840, 7168, padding_idx=163839)（:1096-1097）
   - 93 个 `KimiDecoderLayer`（:877）；最终 `norm` RMSNorm（:1100）；
     use_attn_residuals=True（attn_res_block_size=12, config.json:26）→
     `output_attn_res_norm`/`output_attn_res_proj`（:1105-1108）。
   - lm_head Linear(7168→163840, bias=False)（:1247-1248），tie_word_embeddings=False。
5. **注意力残差机制**（use_attn_residuals=True 时每层走 `_forward_attn_residual` :973-1046）：
   prefix_sum 跨层累加；每当 layer_idx % 12 == 0 把 prefix_sum 追加进 block_residual（:995-998）；
   每层注意力前后各一次 `_apply_attn_res`（:1075-1088）：把 prefix_sum 与 block_residual
   序列 concat 后做 RMS 风格归一化，用 norm.weight × proj.weight（Linear 7168→1）算逐块
   softmax 注意力权重，加权求和。模型 forward 结束后再做一次 output_attn_res（:1215-1217, :1226-1233）。

## 三、层型统计与分支判定（关键：is_kda_layer 的 +1 语义）

`KimiLinearConfig.is_kda_layer(idx)` = `(idx+1) in kda_layers`（configuration_kimi_k3.py:152-156）
—— linear_attn_config 里两张清单是 **1-indexed**（清单号 = layer_idx + 1）。据此（config.json）：

- `kda_layers`（1-indexed）[1,2,3,5,...,91] 共 69 项 → **KDA 层 layer_idx ∈ {0,1,2,4,5,6,7,...,91}**
  （即除 3,7,...,92 外的全部；69 层）→ self_attn = KimiDeltaAttention。
- `full_attn_layers`（1-indexed）[4,8,...,92,93] 共 24 项 → **MLA 层 layer_idx ∈ {3,7,11,...,92}**
  （93 为越界冗余项，对应 layer_idx=92；24 层）→ self_attn = KimiMLAAttention。
- 69 + 24 = 93 ✓ 全覆盖，无歧义层。
- FFN 分支（modeling_kimi_linear.py:893-900）：layer_idx >= first_k_dense_replace(1) 且
  layer_idx % moe_layer_freq(1) == 0 → MoE。故 **layer 0 = dense KimiMLP，layer 1..92 = KimiSparseMoeBlock**。
  组合出 3 种层型：KDA+dense(仅 layer 0)、KDA+MoE(68 层)、MLA+MoE(24 层)。
- KimiSparseMoeBlock.forward 断言 `not self.training`（:824-827）——训练态 MoE 未实现，
  本分析一律 eval 模式。

## 四、全量配置 → 维度映射（config.json 推导）

语言侧 KimiLinearConfig：
- vocab_size=163840, hidden_size H=7168, num_hidden_layers=93
- MLA：q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
  v_head_dim=128 → q_head_dim=192；num_attention_heads=num_key_value_heads=96（n_rep=1）；
  mla_use_nope=True, mla_use_output_gate=True；scaling=192^-0.5（:359）
- KDA（linear_attn_config）：num_heads Hk=96, head_dim Dk=128（q/k/v 投影宽 12288），
  short_conv_kernel_size=4, use_full_rank_gate=True（g_proj 7168→12288）,
  gate_lower_bound=-5.0
- MoE：num_experts=896, top-16, moe_intermediate_size=3072, routed_expert_hidden_size=3584
  （latent MoE：down 7168→3584 → 专家 3584→3072→3584 → norm → up 3584→7168;
   latent_moe_use_norm=True）, num_shared_experts=2（shared KimiMLP intermediate=3072×2=6144）,
  routed_scaling_factor=1.0, moe_router_activation_func=sigmoid, moe_renormalize=True,
  num_expert_group=1（不分组）
- dense FFN：intermediate_size=33792；hidden_act='situ'（SituAndMul, beta=4.0,
  linear_beta=25.0；:64-82）
- rms_norm_eps=1e-5；max_position_embeddings=1048576

视觉侧 KimiK3VisionConfig：
- vt_hidden_size=1024, vt_num_attention_heads=12, qkv_hidden_size=1536 → head_dim=1536/12=128,
  vt_num_hidden_layers=27, vt_intermediate_size=4096（MLP2 1024→4096→1024）
- patch_size=14, init_pos_emb 64×64×4 帧, pos_emb_type=divided_fixed, 插值 bilinear
- merge_kernel_size=(2,2), merge_type=sd2_tpool；norm_type=rmsnorm；attn_bias/linear_bias/
  patch_embed_proj_bias 均 False；activation_func=gelu_pytorch_tanh（PytorchGELUTanh）
- mm_projector：patchmergerv2, mm_hidden_size=1024, text_hidden_size=7168, projector_ln_eps=1e-5
- 顶层：media_placeholder_token_id=163605, pad_token_id=163839, ignore_index=-100

## 五、外部依赖在路径上的判定（shim 需求的先决判断）

audit missing_deps：PIL、einops、fla、flash_attn、tiktoken。

| 依赖 | 调用点 | 是否在建模前向路径 | 判定 |
|---|---|---|---|
| einops | modeling_kimi_linear.py:31（rearrange，KDA 内多处 :570/:602-607/:655/:658） | **是** | 需 shim（rearrange 子集） |
| fla | :47-51（FusedRMSNormGated/ShortConvolution/chunk_kda/fused_recurrent_kda/prepare_*/tensor_cache） | **是**（KDA 69 层必经） | 需 shim 包 |
| flash_attn | modeling_kimi_k3.py:53-56 | 视觉：FA2 不可用时回退 flash_attn_varlen_func=None → `multihead_attention` 会崩；但毛坯自带 eager 回退（VL_VISION_ATTENTION_FUNCTIONS['eager']=eager_attention :102-139）。语言侧 MLA 强制走 ALL_ATTENTION_FUNCTIONS['flash_attention_2']（modeling_kimi_linear.py:1110-1117 无条件覆写为 FA2） | 视觉走毛坯自带 eager 回退（构造期令 vision config._attn_implementation='eager'）；语言侧需在 ALL_ATTENTION_FUNCTIONS 注册纯 torch FA2 等价函数 |
| PIL | kimi_k3_vision_processing.py:9、media_utils.py:9 | 否（图像预处理，非建模 forward） | 不替换 |
| tiktoken | tokenization_kimi.py:7-8 | 否（分词器） | 不替换 |

try/except 回退 3 处判定：
1. modeling_kimi_k3.py:36-41 PytorchGELUTanh 回退 → transformers 5.17 有 PytorchGELUTanh，走 try 分支，无需处理。
2. modeling_kimi_linear.py:46-53 fla 导入失败即 raise → 必须 shim fla 包（无毛坯内回退）。
3. tokenization_kimi.py:13 相对/绝对导入回退 → 与建模前向无关。

## 六、transformers 5.17.0 环境适配的预判（capture 侧实施要点，逻辑侧先记录）

（说明：以下为读 transformers 5.17.0 库源码/探针所得的环境事实，供 capture 阶段使用；
 不属于对毛坯本征行为的推断。）
- `transformers.utils.generic` 已无 OutputRecorder（移至 transformers.utils.output_capturing），
  modeling_kimi_linear.py:44 的导入会失败 → 兼容补丁。
- `create_causal_mask` 形参为 `inputs_embeds` 且无 `cache_position`；毛坯传
  `input_embeds`/`cache_position`（:1173-1180）→ 兼容包装。且其内部需要
  `past_key_values.get_query_offset`，KimiDynamicCache（自定义类 :120-223）不满足 Cache 协议
  → 掩码机需短路返回 None（FA2 语义本就无显式掩码，因果性由注意力核保证）。
- `PretrainedConfig.__init__` 会重置子类先设的 `_attn_implementation` → 构造模型前需对
  config 实例显式回写。5.17 的 ALL_ATTENTION_FUNCTIONS 为 MutableMapping，可注册自定义实现。
- `post_init→init_weights→tie_weights(recompute_mapping=False)` 与毛坯顶层
  `tie_weights()`（:945-946）签名冲突 → 内存级包装。
- MLA 的 FA2 dispatch 被无条件强制（:1110-1119），故语言侧注意力后端实测图中的
  "flash_attention_2" 节点实际执行的是注册的纯 torch 等价函数（等价替换，须写入 replacements）。
- 毛坯 config.json 内含 compressed-tensors mxfp4 量化配置，但毛坯建模代码不实现反量化，
  本分析随机初始化、省略该字段 → 权重全部为普通 nn.Linear 参数，dtype 跟随默认 dtype。

## 七、dtype 推导要点

- 全模型默认 dtype：config.json dtype=bfloat16 → 实例化时 torch 默认 dtype 设为 bfloat16，
  所有 nn.Linear/Embedding/Conv2d 参数与激活默认 bf16。
- 显式升 fp32 的位置（保持/改变 dtype 的算子语义证据）：
  - KimiRMSNorm：x.float() 计算后 cast 回输入 dtype（:232-236）
  - SituAndMul：gate/up 切半后 .to(float32)，返回 cast 回 x.dtype（:76-82）
  - KimiMoEGate：logits = F.linear(hidden.float(), weight.float())（:707-709），sigmoid/softmax
    路径 fp32，topk_weight 随 fp32；moe_infer 末尾 mul_ topk_weight 后 .type(new_x.dtype)（:867-873）
  - KDA：beta = b_proj(...).float()（:603）；chunk_kda 内部（fla 核）常规 fp32 累积，
    输出回 bf16
  - _apply_attn_res：v_float=v.float()，softmax fp32，输出 .to(v.dtype)（:1080-1088）
  - apply_rope：view_as_complex 要求 float32（xq.float()，:187-193），freqs_cis complex64，
    输出 type_as 回 bf16
  - eager_attention（视觉回退）：softmax dtype=float32 再 .to(q.dtype)（:127-128）
- KimiDynamicCache 的 key/value 随层 dtype（bf16）。

## 八、主数据流 shape 骨架（全量配置，符号化）

视觉（单图 t,h,w；多图则 L=Σt·h·w）：
- pixel_values (L, 3, 14, 14) → conv → (L,1024,1,1) → view (L,1024)
- pos_emb 逐图 (t·h·w, 1024) 加上 → (L,1024)
- rope freqs_cis：(Σt·h·w, dim/2=64) complex64
- 每层：norm (L,1024) → wqkv (L,4608) → view (L,3,12,128) → unbind q/k/v (L,12,128)
  → rope → varlen attn（packed (L,12,128)，非 batch 维）→ wo (L,1024)
- final_layernorm (L,1024) → tpool → list[(h/2·w/2, 4, 1024)] → projector concat
  (N,4096) → (N,7168) → post_norm → split list[(n_i,7168)]，N=Σh/2·w/2

合并：inputs_embeds (B,S,7168) + 图像特征展开 → final_embedding (B,S',7168)（S'≥S），
final_attention_mask (B,S') bool，position_ids (B,S')。

语言层 i（hidden_states (B,S',7168)）：
- input_layernorm → (B,S',7168)
- KDA 层：q/k/v_proj (B,S',12288) → ShortConv (B,S',12288)（cache (B,12288,3)）→
  rearrange (B,S',96,128)；g=f_b(f_a(x)) (B,S',96,128)；beta=b_proj(x).float() (B,S',96)；
  chunk_kda → o (B,S',96,128) + state (B,96,128,128)；g_gate (B,S',96,128)；o_norm → o_proj
  → (B,S',7168)
- MLA 层：q_a (B,S',1536) → RMSNorm → q_b (B,S',96·192=18432) → view (B,S',96,192) →
  transpose (B,96,S',192)，split nope/rot (128,64)；kv_a (B,S',512+64=576) → split →
  kv_a_layernorm → kv_b (B,S',96·(192-64+128)=96·256=24576) → view/transpose → split
  k_pass (B,96,S',128) / v (B,96,S',128)；k_rot (B,1,S',64)→expand→cat key (B,96,S',192)；
  cache update；FA2 等价 attn（FA2 时 v pad 到 192，输出切片回 128）→ (B,S',96,128) →
  reshape (B,S',12288) → g_proj sigmoid 门控 → o_proj (B,S',7168)
- attn_res：self_attention_res_proj/norm（7168→1）；_apply_attn_res 内 v
  (num_tokens, num_blocks+1, 7168)，probs (num_tokens,1,num_blocks+1)，输出 (num_tokens,7168)
- FFN dense（layer0）：w1/w3 (B,S',33792)、cat (B,S',67584)、SituAndMul、w2 (B,S',7168)
- MoE（layer1..92）：gate：hidden (B·S',7168) → logits fp32 (B·S',896) → sigmoid →
  topk16 idx (B·S',16) / weight (B·S',16)（renorm ×scaling）；latent down (B·S',3584)；
  896 专家各 (n_e,3072)；y → routed_expert_norm (3584) → up (B·S',7168)；
  shared_experts：gate/up (B·S',6144)、down (B·S',7168)
- 最终：norm (B,S',7168) → output_attn_res → lm_head (B,S',163840)

decode 场景（q_len=1，缓存续算）：MLA key/value 经 cache concat，attention (B,96,1,past+1)；
KDA mode='fused_recurrent'（:561），conv cache (B,·,3)，recurrent state (B,96,128,128)。

## 九、覆盖清单草稿（contract.coverage）

层型（语言侧 3 + 视觉 1 + 投影 1）：
- kda_dense（layer 0 唯一实例）
- kda_moe（layer 1-2,4-6,...，68 层）
- mla_moe（layer 3,7,...,92，24 层）
- moonvit_layer（视觉 27 层）
- patchmergerv2（projector 唯一）
层外组件：patch_embed、pos_emb、rope_2d、tpool、merge、embed_tokens、lm_head、
final norm、output_attn_res、KimiDynamicCache（非 Module 状态容器）。

## 十、开放问题（open_questions 草稿）

1. fla 原生 chunk_kda/fused_recurrent_kda 的内部精确数学（尤其 use_gate_in_kernel 下
   dt_bias/A_log 的组合式、safe_gate/lower_bound 的截断位置、transpose_state_layout 的
   状态布局）无法在本机核对（fla 未安装且不联网）——shim 按毛坯调用契约 + Kimi Delta
   Attention 论文语义重建，等价性验证只能到"对拍数学定义"级。recurring state 布局
   按 shim 定义 (B,H,K,V) 声明，与 fla 原生布局的对应关系未验证。
2. 视觉注意力真实生产路径为 flash_attn_varlen（kernel 融合、分块 softmax），实测走
   毛坯自带 eager 回退；两者数学定义相同（softmax(QK^T/√d)V），但 eager 的 mask 构造
   （:111-119）逐段 bool 掩码。对 shape/dtype 结论无影响。
3. Learnable2DInterpPosEmbDivided_fixed 中 get_rope_shape 带 @torch.compile(dynamic=True)
   （:162-169），TorchVista 抓图若受 dynamo 干扰，将记录为等价替换（去除 compile 包装，
   数学不变）。(此后如发生，见 capture 侧 replacements 实录)
4. 权重未加载真实 checkpoint（毛坯 model.safetensors.index.json 存在但本任务随机初始化），
   数值结论一律不对外推。
5. KimiMoEGate 中 use_grouped_topk=True 但 num_expert_group=1 → grouped 分支不激活
   （:724-746 else 分支），top-k 直接对 scores+bias 取。

## （追加条目区——融合后若有修正，追加于此，不回改上文）

## （融合裁决追加条目 · 2026-10-10 attempt 3）

以下为 S3 融合裁决阶段对上文推导的修正与补充（原文不回改）：

1. **proj 中间维修正**：PatchMergerMLPV2 中间维 = 4×mm_hidden_size（modeling_kimi_k3.py:789-795）。
   本笔记初稿把缩水后该维写成「4096→128」，×4 因子遗漏——缩水 mm_hidden=128 → 该维=512。
   融合 IR 以符号 Hp 声明。
2. **KDA unpad 展平分支补充**（初稿遗漏）：含图输入时 attention_mask 非 None，KDA 前向走
   get_unpad_data/index_first_axis 展平 (B,S,·)→(1,Tu,·)（modeling_kimi_linear.py:566-574）；
   `_update_linear_attn_mask` 在 cache_position[0]>0 或全 1 掩码时返回 None（:1124-1133），
   故 prefill_text/decode 不展平、prefill_vision 的视觉支路 (1,39) 展平。融合 IR 以
   Bu/Su 符号表达该调用域。
3. **Rope2D 全量频率表硬编码**（初稿遗漏）：MoonViT3dEncoder.__init__ 以硬编码
   Rope2DPosEmbRepeated(dim,512,512) 构造（modeling_kimi_k3.py:579-582），_precompute 频率表
   N=512×512=262144 与输入 grid 无关，运行时仅切片；缩水实例同样 512²。
4. **双层型共享模板**：o_proj/g_proj 在 KDA（输入/输出宽 P=Hk·Dk）与 MLA（Hq·Vd）两型下
   维度语义不同，实测逐条归位确认；b_proj 输出=头数 Hk（beta），f_a 输出=Dk、f_b 输出=P。
5. **实测印证的补充结论**：视觉 norm 为原生 aten rms_norm 融合算子（nn.RMSNorm，与语言侧
   KimiRMSNorm 分解链不同）；PytorchGELUTanh 的 gelu 算子未出现在实测图（抓图引擎折叠/未追踪），
   视觉 mlp.activation 节点保留 llm_only。
6. **融合结论**：机械融合 42 个 mismatch 全部完成三问归因（逻辑侧字面量/符号缺陷或机械全局
   绑定与对齐口径所致，无一处无法裁决），转为 verified；attn_res 由 llm_only 升级 verified
   （记录自层节点回填）；rope_2d/mlp/activation/merge/lm.cache 保留 llm_only 并写明原因。
   详见 fusion/diff_report.md 复核记录。
