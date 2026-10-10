# S2a 逻辑分析推导笔记 · DeepSeek-V3.2

- 记录时间：2026-10-10 13:40（阶段二开始；本笔记为独立原始记录，落盘后不回改，修正以追加条目方式记录）
- 引擎：LLM 逻辑推导（独立引擎）。本阶段未读取任何 capture/ 产物（当时 capture/ 目录为空，仅创建了目录骨架）。
- 输入：`inputs/DeepSeek-V3.2/` 毛坯源码 + `outputs/DeepSeek-V3.2/audit/audit_report.json`（阶段一产物）。

## 1. 毛坯构成（审计 + 通读）

| 文件 | 角色 |
|---|---|
| `inference/model.py`（923 行） | 全部模型结构：ModelArgs + 12 个 nn.Module + RoPE/Hadamard/weight_dequant 辅助函数 + Transformer 入口 |
| `inference/kernel.py`（275 行） | 三个 tilelang FP8 kernel：act_quant / fp8_gemm / fp8_index（模块顶层 `tilelang.set_log_level`，import 即依赖 tilelang） |
| `inference/generate.py` | 生成脚本（argparse 在 `__main__` 守卫内；main 读 config json → ModelArgs(**json) → Transformer；权重用 safetensors load_model） |
| `inference/convert.py` | HF→单卡权重转换脚本（与 forward 无关） |
| `inference/config_671B_v3.2.json` | **671B 真实配置**（generate.py main 加载它构造 ModelArgs） |
| `config.json`（根） | HF 格式同源配置（deepseek_v32，arch DeepseekV32ForCausalLM） |

入口类：`Transformer(args: ModelArgs)`。构造参数来源：generate.py `main()` 读 `config_671B_v3.2.json` 后 `ModelArgs(**json.load(f))`。`ModelArgs` dataclass 默认值是另一套小型冒烟配置（dim=2048/n_layers=27/64 家），**逻辑推导以 671B 配置为准**（它才是 DeepSeek-V3.2 本尊），符号值全部锚定到该配置。

## 2. 配置 → 维度映射（671B）

vocab=129280, dim=7168, inter_dim=18432, moe_inter_dim=2048, n_layers=61, n_dense_layers=3（层 0-2 稠密 MLP，层 3-60 MoE）, n_heads=128, n_routed_experts=256, n_shared_experts=1, n_activated_experts=8, n_expert_groups=8, n_limited_groups=4, route_scale=2.5, score_func=sigmoid, q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, index_n_heads=64, index_head_dim=128, index_topk=2048, dtype="fp8", scale_fmt="ue8m0"；max_seq_len/max_batch_size 取 ModelArgs 默认 16384/8（671B json 未含）。

关键全局（model.py L13-15）：`world_size=1, rank=0` 模块级默认；Transformer.__init__ L873-875 从 dist 读（未初始化则 1/0）。`Linear.dtype` 在 Transformer.__init__ L876 被改为 fp8（args.dtype=="fp8"）→ **所有未显式给 dtype 的 Linear 权重是 float8_e4m3fn**，且带 128×128 块 scale（fp32）。

## 3. 层结构清单（源码依据）

- Transformer：embed(ParallelEmbedding) → 61×Block(ModuleList) → norm(RMSNorm) → head(ColumnParallelLinear, fp32)；buffer freqs_cis（complex64 [max_seq_len, 32]）。
- Block：attn_norm(RMSNorm) → attn(MLA) → ffn_norm(RMSNorm, 带 residual) → ffn（层<3: MLP；层≥3: MoE）。
- MLA：wq_a(Linear) → q_norm(RMSNorm) → wq_b(ColParallel) → wkv_a(Linear) → kv_norm(RMSNorm) → wkv_b(ColParallel) → wo(RowParallel) + indexer(Indexer) + kv_cache/pe_cache buffer。
- Indexer：wq_b(Linear) → wk(Linear) → k_norm(LayerNorm) → weights_proj(Linear, fp32) + k_cache(fp8)/k_scale_cache buffer。
- MLP：w1/w3(ColParallel) + w2(RowParallel)，silu(w1·x)·w3(x)。
- MoE：gate(Gate) + experts(ModuleList[Expert]×256) + shared_experts(MLP)。
- Gate：weight [256,dim]；dim==7168 时有 bias [256]（671B 命中）。

层型分布：dense_block ×3（层 0-2）、moe_block ×58（层 3-60）；两型 Block 的 attn/归一化完全同构。

## 4. dtype 关键事实（逐处核对）

- 默认 dtype：generate.py L120 `torch.set_default_dtype(torch.bfloat16)` → bf16 环境。Gate.weight/expert 无显式 dtype 的 Parameter = bf16。
- Linear 权重 fp8（除 weights_proj、head 显式 fp32）→ `linear()`（L134-163）分支：weight 非 fp8 走 F.linear；fp8 走 `act_quant(x) + fp8_gemm`。
- fp8_gemm 输出 dtype = `torch.get_default_dtype()`（bf16，kernel.py L193）。
- RMSNorm/LayerNorm：fp32 计算，cast 回输入 dtype（L296-306 / L321）。
- MLA 的 fp8 KV 缓存模拟：act_quant 后立即反量化回 bf16 存 kv_cache（L570-572，源码注释明说是精度模拟）。
- RoPE：complex64 中间态（view_as_complex on fp32），输出 cast 回 bf16（L416-425）。
- Indexer：weights_proj 输入 x.float()（fp32）；index_score fp32；mask 相加 fp32+=bf16 广播；topk 输出 int64。
- Gate：x.float() 与 weight.float() → F.linear fp32 → 全程 fp32（含 sigmoid/bias/group/topk/gather/归一/×2.5）。
- head：输入 h[:,-1].float()（fp32）× fp32 权重 → logits fp32。

## 5. 逐模块 forward 形状推导（671B 值代入，world_size=1）

记号：B 批、S 序长（prefill）、T=start_pos+S（KV 长度）、V=min(index_topk, T)、N=B·S。

### Transformer.forward（L889-913）
tokens [B,S] int64 → freqs_cis 切片 [S,32] complex64 → mask=(S,S) bf16 -inf 上三角（S>1；S=1 时 None）→ embed [B,S,7168] bf16 → 61×Block → norm → head(h[:,-1].float()) → logits [B,129280] fp32。

### Block（L831-851）
residual=None（层 0）：x,res = attn_norm(x), x；否则 attn_norm(x, residual) 双输入。attn 输出 [B,S,7168]；ffn_norm 返回 (normed, residual)；ffn 输出 [B,S,7168]。

### MLA prefill（mask≠None，L574-589）
- qr = q_norm(wq_a(x))：[B,S,1536] bf16
- q = wq_b(qr) → view [B,S,128,192]；split q_nope [..,128] / q_pe [..,64]；q_pe 交错 RoPE → [B,S,128,64]
- kv = wkv_a(x) [B,S,576]；split kv [..,512] / k_pe [..,64]；kv_norm → [B,S,512]；k_pe RoPE → [B,S,1,64]→squeeze
- kv act_quant → (fp8 [B,S,512], scale fp32 [B,S,4])；反量化回 bf16 [B,S,512]；写 kv_cache/pe_cache
- q=cat(q_nope,q_pe) [B,S,128,192]；kv=wkv_b(kv) [B,S,32768]→view [B,S,128,256]→split k_nope [..,128] / v [..,128]
- k=cat(k_nope, k_pe.expand 128 头) [B,S,128,192]
- scores=einsum("bshd,bthd->bsht")×softmax_scale → [B,S,128,S] bf16
- indexer → topk_indices [B,S,min(2048,T)] int64
- index_mask=(B,S,S) -inf scatter 0 += mask；scores+=；softmax；x=einsum("bsht,bthd->bshd", v) → [B,S,128,128]
- wo(x.flatten(2)) → [B,S,7168]

### MLA decode（mask=None，S=1，L590-606）
- wkv_b 权重矩阵路径：fp8 时 weight_dequant 一次缓存（L591-593）→ [128,256,512] 视图
- q_nope einsum("bshd,hdc->bshc", wkv_b[:, :128]) → [B,1,128,512]（c=kv_lora）
- scores = einsum("bshc,btc", kv_cache[:B,:T]) + einsum("bshr,btr", pe_cache) → [B,1,128,T] ×scale
- index_mask (B,1,T) scatter；softmax；x=einsum("bsht,btc")→[B,1,128,512]→einsum("bshc,hdc", wkv_b[:,-128:])→[B,1,128,128]→wo

### Indexer.forward（L457-487）
- q=wq_b(qr) [B,S,64×128=8192]→view [B,S,64,128]；split（rope 在前！）q_pe [..,64] / q_nope [..,64]；**非交错** RoPE（interleaved=False）
- k=wk(x) [B,S,128]；k_norm(LayerNorm 128)；split 后 k_pe 非交错 RoPE；cat → [B,S,128]
- rotate_activation：Hadamard（bf16，scale=128^-0.5），q [B,S,64,128]、k [B,S,128]
- act_quant q → (fp8 [B,S,64,128], scale fp32 [B,S,64,1])；k → (fp8 [B,S,128], scale [B,S,1])；写 fp8 k_cache/k_scale_cache
- weights = weights_proj(x.float())×64^-0.5 [B,S,64] fp32；weights = weights.unsqueeze(-1)×q_scale×softmax_scale → [B,S,64,1]
- index_score = fp8_index(q_fp8, weights, k_cache[:B,:T], k_scale_cache) → [B,S,T] fp32（语义见 kernel.py L269-272：relu(q·k) 按 head 加权和 ×k_scale）
- += mask；topk(min(2048,T)) → int64；**dist.broadcast 无 world_size 守卫（L485）——单进程也执行，抓图必须初始化进程组**

### MoE（L780-804）
x.view(-1,7168) [N,7168] → gate → (weights [N,8] fp32, indices [N,8] int64)：
- scores=linear(x.float(), w.float()) fp32 [N,256] → sigmoid（score_func）→ +bias [256] → view [N,8,32] → 组内 top2 求和 [N,8] → topk_groups=4 选组 → mask -inf → topk 8 → gather 原始分 → /sum（sigmoid 归一）→ ×2.5
- counts=bincount(indices)（tolist → CPU 同步）；逐专家 where(indices==i) → y[idx] += expert(x[idx])×weights[idx,top,None]（y fp32 累加）
- expert：w1 [n_j,7168]→[n_j,2048]；silu(w1.float())×w3.float() → bf16；w2 → [n_j,7168]
- y += shared_experts(x)（MLP dim=7168, inter=1×2048, reduce_output=False）；type_as(bf16).view(B,S,7168)

### 稠密 MLP（L633-643）
w1 [B,S,18432]；silu(w1(x).float())×w3(x).float() → type_as bf16；w2 → [B,S,7168]。

### ParallelEmbedding（ws=1 分支）
mask 分支被 world_size>1 守卫跳过；直接 F.embedding → [B,S,7168] bf16。

## 6. kernel.py 数学语义（tilelang kernel 的等价数学定义）

- **act_quant**（L87-111 + kernel L36-84）：末维按 128 分块；s = amax/448（amax 钳到 ≥1e-4）；y = clamp(x/s, ±448) cast fp8 e4m3；scale_fmt 非空（ue8m0）时 s = 2^ceil(log2(amax/448))（fast_round_scale 的位运算等价：exp-127+(man≠0) 即 ceil(log2)）。返回 (fp8, fp32 scale)，形状同 x（末维缩 128 分之一）。
- **fp8_gemm**（L171-196 + kernel L114-168）：C[m,n] = Σ_k a[m,k]·b[n,k]·sa[m,⌊k/128⌋]·sb[⌊n/128⌋,⌊k/128⌋]，fp32 累加，输出 cast 默认 dtype（bf16）。等价于逐块反量化后的 fp32 matmul。
- **fp8_index**（L254-274 + kernel L199-251）：o[b,m,n] = ks[b,n] · Σ_h relu(Σ_d q[b,m,h,d]·k[b,n,d]) · qs[b,m,h]，fp32。q_s 形参实收 [b,s,h,1]（weights，已并入 softmax_scale 与 q_scale）。
- **weight_dequant**（model.py L490-495，纯 torch，非替换件）：fp8 权重按 128×128 块 ×scale 反量化回默认 dtype。

## 7. 覆盖与分支决策

- path 契约：ModuleList 容器段不写（`{i}.attn.wq_a` 而非 `layers.{i}...`）；logical.shape 描述**节点主输入**（融合引擎以 func_info 首个输入张量比对——已核 fuse.py 对齐契约）。
- IR 粒度：模块级节点为主（实测树里模块节点与其内部 aten 共享路径前缀，同场景模块记录优先）；shim 调用点逐一开节点（每个 fp8 Linear 下 act_quant/fp8_gemm 两个子节点，路径 `{i}.attn.wq_a.act_quant` 等）。
- 已知同路径双形状点（提前声明，预期融合期 mismatch，归因=同函数多调用点）：
  1. `{i}.attn.indexer.act_quant`：q 侧 [B,S,64,128] 与 k 侧 [B,S,128] 两次调用同路径，IR 主输入记 q 侧。
  2. `{i}.ffn.{j}` 及其子节点：每专家 token 数 NJ 数据依赖，实例间不同，多实例匹配必然符号冲突。
- 分支差异：dense/MoE 共用 `{i}.ffn` 模板节点（路径冲突不可二开，branch_note 区分）；MLA prefill/decode 两分支同节点不同 scenario；层 0 attn_norm residual=None 单参变体；yarn/mscale 分支在 max_seq_len>original_seq_len 时触发（671B 真：16384>4096 触发）。
- 无 forward 图节点、不参与对齐的模型级胶水：mask 构造、freqs_cis 切片（Transformer.forward 直属 aten，树路径为空串）——记入 coverage.folded。
- 构造期常量：freqs_cis buffer（[16384,32] complex64，yarn 修正 L394-397）→ IR 节点 status=llm_only。

## 8. open_questions（逻辑侧真实不确定项）

1. fast_hadamard_transform 的行序约定：CUDA kernel 实现为自然序（Sylvester）Walsh-Hadamard，本机无该库无法对拍原实现；数学定义（y_i=Σ_j(-1)^{popcount(i&j)}x_j）与正交性可验证，但与原 kernel 的位序一致性只能到"约定声明"级。
2. tilelang ue8m0 round_scale 的位运算在 amax/448 恰为 2 的整数次幂时的边界行为（man==0 → 不进位），已从源码推出等价 ceil(log2)——置信度高但无原实现可对拍。
3. dist.broadcast（L485）在无分布式初始化时必然抛错——这是毛坯对"单进程推理"的一个事实性缺陷（官方代码假定总在分布式下运行），不影响 shape/dtype 推导，但决定了抓图环境必须 init_distributed。

（以上三条在 capture 侧以 shim/环境声明处理，均不构成 shape/dtype 的不确定性。）

## 9. 追加更正（阶段二内自查，抓图前）

- §1 表中"generate.py L120 set_default_dtype"行号有误：核实后为 **L109**（`torch.set_default_dtype(torch.bfloat16)`）；构造参数来源实为 **L112-113**（`args = ModelArgs(**json.load(f))`）与 **L116**（`model = Transformer(args)`）。IR evidence 不引用 generate.py 行号（仅 model.py / kernel.py），不受影响。

## 10. 事后补记（阶段四定稿后追加，非逻辑推导原始记录——独立性以 §1-§9 的时间戳为准）

补记性质：S3 融合定稿后按留痕要求回写的执行结论，不改动任何逻辑推导内容（§1-§9 为抓图前原始记录，mtime 13:44:55）。

- 抓图采用 forced_depth 0-4 五档矩阵：torchvista 的 wrap_module 三分支（守卫不可见 / 成节点但内部隐藏 / 透明帧）决定了单一深度无法同时覆盖全部层级——d0 成顶层+Block 节点，d4 才能见到全部 shim 调用点；首抓 11 作业发现 16 个第 4 层级 shim 调用点缺失（indexer.wq_b/wk、shared_experts、expert 的 w1-w3），增补 d4 两作业后全覆盖（13 作业，0 failed）。
- 同因回补 IR：d0 作业下 Block 成模块节点，而逻辑 IR 原无 Block 复合节点 → 抓图前在 analysis_ir 插入 `{i}` 节点（Block.forward L831-851），定版时间 14:31:31 早于最终抓图窗口 14:35:45。
- S3 定稿结论（详见 fusion/diff_report.md 复核记录）：66 verified / 1 mismatch / 1 llm_only / 0 capture_only。唯一 mismatch 为 Indexer act_quant 双调用点设计性保留（q [B,S,IH,ID] / k [B,S,ID]）。
- 退化路由实测形态：随机初始化 → 门控近似均匀 → 全 token 同路由至 top-8 专家，NJ 跨专家一致（prefill 64 / decode 1），预声明的多实例符号冲突未发生——已归因为随机初始化形态而非模型本征行为（scale_note ③）。
- §8.2 的 ue8m0 位运算等价推导获自检证实：对拍数学定义 17 项全过（act_quant 逐位一致）。
