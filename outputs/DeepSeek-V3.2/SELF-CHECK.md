# SELF-CHECK · DeepSeek-V3.2 全量算子分析（S5 总自检）

- 日期：2026-10-10；代理：算子分析 Agent（双引擎规程 agent-kit_ops-analysis/SKILL.md）
- 范围：`outputs/DeepSeek-V3.2/` 全部交付物对照 SPEC.md 逐项核对
- 总结论：**九条验收标准全部通过；诚实性红线逐条自查无违反；无 kit 修改需披露**

## 一、交付物清单对照（SPEC 交付物表）

| # | 交付物 | 位置 | 状态 | 证据 |
|---|---|---|---|---|
| 1 | 静态审计报告 | `audit/audit_report.json` | ✔ | 覆盖毛坯全部 .py（inference/model.py、inference/kernel.py 等 4 文件登记于 IR source_files_hash）；missing_deps 列明 tilelang 与 fast_hadamard_transform（替换件存在的原因） |
| 2 | 逻辑分析 IR | `llm-analysis/analysis_ir.json` | ✔ | 68 节点全 llm_only 起步；evidence 五要素（file/lines/symbol/conclusion/kind）齐备；校验通过 |
| 3 | 推导笔记 | `llm-analysis/notes.md` | ✔ | 独立原始记录，首写 13:44:55，早于全部抓图产物（最早 measured 14:35:45）；append-only，9 节 |
| 4 | 抓图 spec | `capture/spec.json` | ✔ | capture_driver 校验通过并成功驱动 13 作业（all_ok=true）；缩水实例化在 hooks.py docstring 与 IR scale_note 双处声明 |
| 5 | shim 件 | `capture/shims/` | ✔ | kernel.py（3 个挂树+重绑 shim，含数学定义与源码行号对照）、fast_hadamard_transform.py（sys_path 遮蔽）、selftest.py（17 项对拍自检全过） |
| 6 | 抓图产物 | `capture/*.html` + `*.measured.json` + `capture_summary.json` | ✔ | 13 作业全部成功（prefill/decode × d0-d4 + 3 micro），0 failed nodes |
| 7 | 融合 IR | `fusion/final_ir.json` | ✔ | `ir_schema.py` 校验通过（68 节点 / 4 项替换 / 4 源文件哈希）；66 verified / 1 mismatch / 1 llm_only / 0 capture_only / 0 conflict；measured 与 logical 并列保存 |
| 8 | 差异报告 | `fusion/diff_report.md` | ✔ | 唯一 mismatch 处置在案（复核记录第 1 条）；复核记录 6 条（含专家族、llm_only、替换台账、d4 缺口修复、去重机制） |
| 9 | 可视化页面 | `viz/` | ✔ | 13 份注入后 HTML + README.md（入口导览/场景深度表/标记读法/模板路径对应） |

`outputs/DeepSeek-V3.2/monitor.md` 为观测者流水：本代理全程未读、未写、未删（双引擎独立性约定），其存在可作时序旁证。

## 二、验收标准（通用九条）逐条核对

1. **全中文** — ✔ 全部交付物文案、注释、报告均为中文（自检即本文件；HTML 注入横幅、diff_report、IR dim_meaning 均中文）。
2. **毛坯零修改** — ✔ `find inputs/DeepSeek-V3.2 -name '__pycache__' -o -name '*.pyc'` → 0；`find inputs/DeepSeek-V3.2 -type f -newer audit/audit_report.json`（任务首个产物）→ 0，毛坯无任何文件被触碰。全程 `PYTHONDONTWRITEBYTECODE=1` + kit 内 `sys.dont_write_bytecode=True`。
3. **双引擎独立性可审计** — ✔ 时序链（mtime）：notes.md 13:44:55 → analysis_ir.json 定版 14:31:31 → 抓图窗口 14:35:45-14:36:22（measured/capture_summary）→ 融合定稿 → 注入 14:46。逻辑分析（笔记+IR）完成于抓图开始之前；两侧目录隔离（llm-analysis/ 与 capture/ 无交叉引用）；IR 逻辑结论全部源自码推导（evidence.kind=source_derived），未引用任何实测数值。
4. **覆盖清单逐项有着落** — ✔ `final_ir.json → contract.coverage`：layer_types（稠密层 0-2 / MoE 层 3）、captured（5 条实测代表：稠密块×3 层、MoE 块、MLA 专项、顶层、场景×深度矩阵）、folded（层间重复折叠为 {i}/{j} 模板的推导覆盖声明）。层数/层型分布源码依据（model.py Transformer/Block 构造）。分支保真 + 规模缩水逐项声明于 `model.scale_note`（不冒充全量；dim=7168 等关键分支维度全保真）。
5. **场景齐全** — ✔ prefill（B=1,S=64,start_pos=0）/ decode（B=1,S=1,start_pos=64）各 ≥6 个成功作业；每类层（稠密块/MoE 块/MLA 注意力）各 ≥1 micro 成功。节点 scenarios 字段均为 ["prefill","decode"]，与场景集一致。
6. **kernel 替换合规** — ✔ `contract.replacements` 4 项，每项 what/why/equiv/evidence + shim_class 齐全；验证方法=独立参考实现对拍数学定义（原实现本机不可运行，等级如实标注为"对拍数学定义"，未冒充"对拍原实现"）；结果留痕于 `capture/shims/selftest.py`（17 项：act_quant 逐位一致、fp8_gemm 误差 ≤1.7e-8、fp8_index 误差 5.96e-8、hadamard ≤7.4e-3 bf16 噪声内 + 正交往返）。第 4 项（hadamard）为 sys_path 模块遮蔽机制、无挂树实例，与 spec.shims 3 项挂树重绑的差异已在条目内显式披露。
7. **IR 质量** — ✔ `ir_schema.py` 通过（本轮最后一次运行："校验通过：68 个节点，4 项 kernel 替换，4 个源文件已登记哈希"）；measured/logical 并列保存；唯一 mismatch 显式保留并附三问复核归因 + S3 裁决 open_questions 条目；无静默覆盖（多深度作业记录按模块优先规则去重，声明于 diff_report 复核记录第 6 条）。
8. **可视化质量** — ✔ 13 份 HTML 经 `render_fusion.py` 注入成功（每份：shim 红虚线 6 个挂树目标 + 状态徽标 + 融合面板统计 + 场景标注）；面板计数与 IR 一致（66/1/1/0）；llm_only 节点有"仅逻辑分析、尚未运行验证"提示条（freqs_cis 一项）。读法导览见 `viz/README.md`。
9. **全链可重跑** — ✔ 实际执行链即按 SKILL §3 命令：audit → S2a → capture_driver（spec 驱动，13 作业）→ fuse → 定稿 → render_fusion，全程一次通过。幂等性说明（如实）：`render_fusion.py` 重跑安全（注入去重）；`fuse.py` 重跑会重建**机械融合产物**（该部分确定性一致），随后需重放 S3 定稿编辑——定稿变更共 6 项且全部留痕：① model.config_snapshot/scale_note、② replacements 4 项、③ coverage.captured 5 条、④ mismatch 节点 branch_note+open_questions 追加、⑤ 专家族 4 节点 concerns 追加、⑥（diff_report 复核记录 6 条）。按 diff_report 复核记录与上表可完整重放。

## 三、诚实性红线自查（逐条）

| 红线 | 自查方法 | 结论 |
|---|---|---|
| 编造 evidence 行号 | IR 全部 evidence.file/lines 可回溯：抓图前对 inference/model.py、inference/kernel.py 逐行核对（notes.md §1-§7 原始记录）；source_files_hash 登记可验 | 未违反 |
| 编造实测数据 | 实测块全部来自 capture 作业产物（measured.json→fuse 机械搬运），无人工写入数值 | 未违反 |
| 软化 schema 标准 | ir_schema.py 未做任何修改（见下"kit 修改披露"）；校验以原版脚本运行 | 未违反 |
| "仅逻辑分析"写成"已验证" | freqs_cis 保持 llm_only，viz 面板带"尚未运行验证"提示；IR 内无 llm_only 冒充 verified 的节点 | 未违反 |
| 替换件行为写成模型本征行为 | 4 项替换全部显式标注（红虚线 + replacements 台账 + shim_class）；专家族退化路由明确归因为"随机初始化的实测形态，非模型本征路由行为"（scale_note ③ + concerns） | 未违反 |
| 静默丢弃冲突/未覆盖 | 0 capture_only、0 conflict；唯一 mismatch 显式保留带归因；24 个 counts=0 未激活专家属折叠声明范围（coverage.folded + scale_note ③），非静默丢弃 | 未违反 |

## 四、kit 修改披露（SKILL ①③条款）

**本轮零修改。** 证据：`find agent-kit_ops-analysis -type f -newer audit/audit_report.json` → 0（任务期间 kit 目录无任何文件变动）。SKILL.md / SPEC.md 未触碰。无 3 条件审批事项。

## 五、已知限制与如实声明（汇总）

1. **缩水实例化**：671B → 4 层/8 头/32 专家/vocab 1024 等，逐项对照表见 `model.scale_note`；关键分支维度（dim=7168、n_dense_layers=3、sigmoid 分组路由、qk 头维 128/64/128、fp8+ue8m0）全保真。
2. **未加载真实权重**：量化一致随机初始化（种子 20261010），数值仅用于管线健康性；shape/dtype 结论不受影响。
3. **验证等级上限**：tilelang 与 fast_hadamard_transform 本机均不可运行，4 项替换的等价验证为"对拍数学定义"级（非"对拍原实现"级）；hadamard 行序约定与原 CUDA kernel 的一致性止于约定声明（notes.md §8.1 open_question）。
4. **设计性 mismatch**：`{i}.attn.indexer.act_quant`（同 shim 双调用点 q/k 形状不同，单节点覆盖取 q 侧），保留 mismatch 并全程可追溯（diff_report 复核记录第 1 条）。
5. **decode 作业缓存形态**：d0 作业 rebuild=false 真实串联 prefill 缓存续写；其余 decode 作业重建模型零缓存运行（形状/dtype 等价，声明于 scale_note ②）。
6. **专家族退化路由**：随机初始化 → 门控分数近似均匀 → 全 token 路由相同 top-8 专家，NJ 跨实例一致；24 个未激活专家无实测记录（折叠声明覆盖）。真实权重下 NJ 将随专家变化。
7. **单卡退化**：init_distributed=true 单卡执行，Indexer 的 dist.broadcast（model.py L484-486 无 rank 守卫）在单卡语义下直通；多卡行为超出本抓图范围（IR 对应节点已声明）。
