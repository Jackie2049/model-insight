# 融合差异报告（机械融合产出，供复核裁决）

- IR 节点（含追加）：68；状态分布：{"verified": 66, "llm_only": 1, "mismatch": 1}
- 实测记录：12256 条；已对齐 526 条；未覆盖模块路径 0 个

## 场景符号绑定表
- decode：B=1、D=7168、HDV=1024、I=1024、ID=128、IH=8、KV=512、MI=256、N=1、NJ=1、QL=1536、S=1
- prefill：B=1、D=7168、HDV=1024、I=1024、ID=128、IH=8、KV=512、MI=256、N=64、NJ=64、QL=1536、S=64

## 不一致节点（须复核）（1）
- `{i}.attn.indexer.act_quant`：[{"scenario": "decode", "notes": ["ActQuantShim_343（full 对齐）：维度数不同：逻辑 ['B', 'S', 'IH', 'ID'] vs 实测 [1, 1, 128]", "ActQuantShim_771（full 对齐）：维度数不同：逻辑 ['B', 'S', 'IH', 'ID'] vs 实测 [1, 1, 128]", "ActQuantShim_1199（full 对齐）：维度数不同：逻辑 ['B', 'S', 'IH', 'ID'] vs 实测 [1, 1, 128]", "ActQuantShim_1627（full 对齐）：维度数不同：逻辑 ['B', 'S', 'IH', 'ID'] vs 实测 [1, 1, 128]"]}, {"scenario": "prefill", "notes": ["ActQuantShi

## 仅逻辑节点（1）
- `freqs_cis`：任何场景均未测到（未执行 ≠ 不存在，页面将带标注）

## 仅实测节点（2a 未覆盖）（0）

## 已验证节点（66）
- `embed`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `norm`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `head`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn_norm`：10 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn_norm`：10 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn`：10 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wq_a`：9 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wq_a.act_quant`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wq_a.fp8_gemm`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wq_b`：9 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wq_b.act_quant`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wq_b.fp8_gemm`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.q_norm`：9 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wkv_a`：9 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wkv_a.act_quant`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wkv_a.fp8_gemm`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.kv_norm`：9 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.act_quant`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wkv_b`：5 条实测记录全部一致（场景：['prefill']）
- `{i}.attn.wkv_b.act_quant`：4 条实测记录全部一致（场景：['prefill']）
- `{i}.attn.wkv_b.fp8_gemm`：4 条实测记录全部一致（场景：['prefill']）
- `{i}.attn.wo`：9 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wo.act_quant`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.wo.fp8_gemm`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.indexer`：9 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.indexer.wq_b`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.indexer.wq_b.act_quant`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.indexer.wq_b.fp8_gemm`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.indexer.wk`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.indexer.wk.act_quant`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.indexer.wk.fp8_gemm`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.indexer.k_norm`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.indexer.weights_proj`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.attn.indexer.fp8_index`：8 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn`：10 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.w1`：6 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.w1.act_quant`：6 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.w1.fp8_gemm`：6 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.w2`：6 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.w2.act_quant`：6 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.w2.fp8_gemm`：6 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.w3`：6 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.w3.act_quant`：6 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.w3.fp8_gemm`：6 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.gate`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.shared_experts`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.shared_experts.w1`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.shared_experts.w1.act_quant`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.shared_experts.w1.fp8_gemm`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.shared_experts.w2`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.shared_experts.w2.act_quant`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.shared_experts.w2.fp8_gemm`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.shared_experts.w3`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.shared_experts.w3.act_quant`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.shared_experts.w3.fp8_gemm`：2 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.{j}`：16 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.{j}.w1`：16 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.{j}.w1.act_quant`：16 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.{j}.w1.fp8_gemm`：16 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.{j}.w2`：16 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.{j}.w2.act_quant`：16 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.{j}.w2.fp8_gemm`：16 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.{j}.w3`：16 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.{j}.w3.act_quant`：16 条实测记录全部一致（场景：['decode', 'prefill']）
- `{i}.ffn.{j}.w3.fp8_gemm`：16 条实测记录全部一致（场景：['decode', 'prefill']）

## 复核指引
- mismatch：先核对 evidence 的行号是否真的支持逻辑侧；再看对齐是否正确（路径模板 vs 实测树路径）；最后考虑替换件是否改变了行为。
- 修正逻辑侧时必须同步补/改 evidence；无法裁决的把 status 改为 conflict 并写 open_questions。
- capture_only：属于逻辑分析的覆盖缺口，补分析（或确认辅助路径后在 concerns 说明）。
---

## 复核记录（S3 定稿，融合后人工裁决留痕）

**融合结果**：66 verified / 1 mismatch / 1 llm_only / 0 capture_only / 0 conflict（68 节点，`ir_schema.py` 校验通过）。

1. **mismatch · `{i}.attn.indexer.act_quant` —— 保留 mismatch，设计性归因（非错误）**
   三问复核：① 逻辑侧证据支持成立——wq_b（model.py L460-462）与 wk（L466-467）调用同一全局 `act_quant` 绑定（L10 from-import），IR 以单节点覆盖双调用点，主输入取 q 侧 [B,S,IH,ID]；② 对齐正确——q 侧实测块与逻辑逐维一致，k 侧块 [B,S,ID]（三作业 decode 均为 [1,1,128]）是第二调用点（L475）的真实形状，与证据行推导一致；③ 非替换件行为差异——k 侧形状由毛坯 kernel 语义决定，与 shim 无关。q/k 两侧形状均已被实测确认（k 侧另经 `fp8_index` 的 k 输入链路间接验证）。消除需把同一 shim 路径拆为双 IR 节点（同路径不可二开），故按规约显式保留，处置已写入该节点 open_questions（S3 裁决条目），可视化将带 mismatch 标注。

2. **专家族 `{i}.ffn.{j}`（含 w1/w2/w3 及其 shim 子节点）—— verified，预声明风险未发生，实测形态已声明**
   逻辑侧曾预声明"多实例模板匹配可能符号绑定冲突 → 预期 mismatch"。实测：随机初始化下 MoE 门控分数近似均匀，全部 token 路由到相同 top-8 专家 → 活跃专家 NJ 跨实例一致（prefill N=64、decode N=1），其余 24 个专家 counts=0 被跳过（无实测记录，属 coverage.folded 折叠声明范围）。预声明的冲突没有发生，专家族全部 verified（16 条实测记录一致）。该形态是随机初始化的实测结果而非模型本征路由行为——已写入专家族节点 concerns 与 `model.scale_note` ③，真实权重下 NJ 将随专家实例变化。

3. **llm_only · `freqs_cis` —— 保持 llm_only**
   构造期常量（Transformer.__init__ 预计算注册），无前向记录可抓；其 shape/dtype 已在 macro d0 作业的 embedding 输入侧间接体现。属声明的折叠覆盖，非遗漏。

4. **替换件台账（4 项）与 spec.shims（3 项）的差异说明**
   `contract.replacements` 共 4 项：ActQuantShim / Fp8GemmShim / Fp8IndexShim 三项与 spec.shims 挂树重绑一一对应（shim_class 关联齐全）；第 4 项 hadamard_transform 走 sys_path 首位遮蔽同名模块机制（函数级调用、非挂树实例），故不在 spec.shims[]，机制差异已在条目 equiv/evidence 中披露。全部 4 项等价验证等级为"对拍数学定义"（原实现本机不可运行：tilelang 与 fast_hadamard_transform CUDA 库均未安装），自检 17 项全过（capture/shims/selftest.py）。

5. **抓图缺口修复记录（d4 深度）**
   首轮 11 作业融合出现 16 个 llm_only shim 节点（indexer.wq_b/wk、shared_experts.w1-w3、expert w1-w3 的 shim 调用点在第 4 层级，forced_depth=3 不可见）→ 增补 macro_prefill_d4 / macro_decode_d4 两作业后全部覆盖，最终 0 capture_only。作业清单 13 个（prefill/decode × d0-d4 + 3 micro），见 capture_summary.json。

6. **融合去重机制备注**
   多深度作业（d0-d4）对同一路径产生多条记录，fuse.py 按"模块记录优先于算子记录"规则去重；节点实测 blocks 按（场景，capture_job）保留多深度代表记录，未静默丢弃任何作业数据。
