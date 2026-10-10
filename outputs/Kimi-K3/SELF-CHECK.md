# SELF-CHECK — Kimi-K3 全量算子分析 · 总自检（阶段六）

- 自检时间：2026-10-10（attempt 3）
- 执行代理：算子分析 Agent（attempt 3，断点续作完成阶段四裁决与阶段五/六重做）
- 证据指针均相对 `outputs/Kimi-K3/`（绝对路径根：
  `/Volumes/Agent/workspace/projects/model-insight/model-insight-0928/`）。

## 〇、断点链完整披露（诚实性前提）

| 阶段 | 执行者 | 结果与现状 |
|---|---|---|
| 阶段一 S0 审计 / 阶段二 S2a 逻辑分析 | attempt 1 代理 | 完成；`audit/`、`llm-analysis/`（analysis_ir.json + notes.md 正文）为原始产物，本次未重做、未删除 |
| 阶段三 S1+S2b 抓图（20 作业） | attempt 1 代理 | 完成；`capture/`（20/20 ok）为原始产物，本次未重跑作业、未删除 |
| 阶段四 S3 机械融合 + 复核 | attempt 1 代理 | 机械融合完成，**复核被中断**（final_ir.json 停留在未裁决的机械草稿） |
| 阶段五 / 六（初版） | attempt 2 代理 | 基于上述**未裁决草稿**做过一遍 viz 注入与 SELF-CHECK；其结论（verified 21 / mismatch 42 等）随本次定稿**全部作废重做** |
| 阶段四定稿（三问归因） | attempt 3（本代理） | 按任务指令先重跑 `fuse.py` 再生确定性机械草稿，随后完成 42/42 mismatch 三问归因；`llm-analysis/notes.md` 以**追加条目**方式记录修正（原文未回改） |
| 阶段五 / 六（定稿版） | attempt 3（本代理） | 剥离 attempt 2 留在 capture/*.html 尾部的旧草稿注入块后，用定稿 IR 重新注入并复制 viz/；本文件为重写版 |

对本文件证据效力的影响：attempt 2 代理留下的 SELF-CHECK（被本文件覆盖）曾记录
SPEC §3 指定时间对 `notes.md`(13:53:28) 早于 `capture/capture_summary.json`(14:31:31)
成立；本代理按规程向 notes.md 追加融合修正条目后其 mtime 已更新，**该时间对不再成立**，
独立性判定改以内容级证据为准（见下文第 3 条）。

## 一、交付物清单逐项（SPEC 表）

| # | 交付物 | 位置 | 结论 | 证据 |
|---|---|---|---|---|
| 1 | 静态审计报告 | `audit/audit_report.json` | ✅ | 毛坯 8 个 `.py` 全部入扫；`syntax_errors: []`；`missing_deps` 5 项（PIL/einops/fla/flash_attn/tiktoken）带 import 行号与调用点；入口类与层结构结论见 `llm-analysis/notes.md` §二 |
| 2 | 逻辑分析 IR | `llm-analysis/analysis_ir.json` | ✅ | `ir_schema.py` 校验通过；全节点 `llm_only`、0 个 `measured`；每节点 evidence 五要素；`contract.replacements`/`coverage.captured` 在阶段二为空属本阶段事实（定稿值在 fusion/） |
| 3 | 推导笔记 | `llm-analysis/notes.md` | ✅（有追加） | 正文为 attempt 1 独立原始记录（§一阅读清单声明未读任何 capture 产物——当时 capture/ 尚不存在）；文末「融合裁决追加条目」为本代理按"只追加不回改"规程补记的 6 条修正；开放问题 1-5 保留在正文 |
| 4 | 抓图 spec | `capture/spec.json` | ✅ | 20 traces 经 capture_driver 校验全部跑通（`capture_summary.json`：`all_ok=true`、20/20 `status=ok`）；缩水实例化声明于 `spec.notes.scale_note` 与 `hooks.py` TEXT_CFG/VISION_CFG；`spec.shims=[]` 为有意声明（导入级接管不适用类挂树，见 spec.notes"shims 说明"） |
| 5 | shim 件 | `capture/shims/` | ✅ | R1 einops 遮蔽 / R2 fla 六组件 / R3 FA2 注册表覆盖 + hooks 兼容补丁 P1-P4；数学定义逐文件头注；等价对拍 **21/21 通过**留痕（`shims/equiv_check.py` + `equiv_check_result.txt`，验证等级如实标"对拍数学定义"级——fla 原生核未装且禁联网，无法更高） |
| 6 | 抓图产物 | `capture/*.html`+`*.measured.json`+`capture_summary.json` | ✅ | 20/20 ok（3 宏 + 17 微）；层型全覆盖且每类 ≥1 micro：kda_dense→`micro_layer_kda0`、kda_moe→`micro_layer_kda_moe1`、mla_moe→`micro_layer_mla3`、moonvit_layer→`micro_vit_layer0`、patchmergerv2→`micro_projector`；外围 embed/norm/lm_head/patch_embed/pos_emb/gate/expert/dense 各有 micro |
| 7 | 融合 IR | `fusion/final_ir.json` | ✅（attempt 3 定稿） | `ir_schema.py` 校验通过（69 节点、**4 项 kernel 替换**、8 源文件哈希登记）；五态：**verified 64 / llm_only 5 / mismatch 0 / conflict 0 / capture_only 0**；measured/logical 并列保存；42/42 mismatch 逐一裁决，无静默覆盖 |
| 8 | 差异报告 | `fusion/diff_report.md` | ✅ | 机械差异清单（fuse.py attempt 3 重跑产出）+ 文末「复核记录（S3 人工裁决 · attempt 3）」：42 行逐节点归因表 + 归因分布四类 + llm_only 6 个处置 + 契约层补全说明 + 断点续作说明 |
| 9 | 可视化页面 | `viz/` | ✅ | 20 个注入后 HTML（attempt 3 用定稿 IR 重注入）+ `viz/README.md`（入口/清单/面板/标记读法）；注入后回读断言通过（工具内置）；面板数字取自 final_ir.json（verified 64 / llm_only 5），与 IR 一致 |

## 二、验收标准九条逐条

1. **全中文** ✅ — 各交付物文案/注释/报告全中文（本文件、viz/README、shims/README、
   diff_report 复核记录、节点 branch_note/concerns 均中文）。
2. **毛坯零修改** ✅ — 本代理实测（2026-10-10）：`final_ir.json` 登记的 8 个源文件
   sha256 与 `inputs/Kimi-K3/` 现文件 **8/8 一致**（逐个重算复核，非沿用旧结论）；
   `find inputs/Kimi-K3 -name "__pycache__" -o -name "*.pyc"` 命中 **0**；目录文件数 17
   无新增。抓图与裁决全程 `PYTHONDONTWRITEBYTECODE=1`。
3. **双引擎独立性可审计** ✅（时间对失效，内容级证据为准）—
   - 内容级硬证据一：`notes.md` 正文 §一声明阅读材料不含任何 capture 产物且注明
     "当时 capture/ 尚不存在"，正文 shape 推导全部为 config.json 全量口径；
   - 内容级硬证据二（最强）：`analysis_ir.json` 逻辑 shape 携带全量字面量
     （投影器/proj 4096、H=7168、词表 163840 等），与实测缩水值（512/1000）系统性矛盾，
     并直接构成 42 处机械 mismatch 的一部分——若逻辑侧读过实测数据不可能如此；
   - 目录隔离：capture/ 侧产物无任何对 llm-analysis/ 的引用；llm-analysis/ 69 节点
     0 个 measured 块；
   - mtime 时间对：原成立（attempt 2 的 SELF-CHECK 记录 13:53:28 < 14:31:31），
     因本代理合规追加 notes.md 而失效（见"〇、断点链"）；`analysis_ir.json` 的 mtime
     异常（晚于抓图）经两代代理内容核查均为"文件内容仍是纯净阶段二产物"的
     无害触碰；观测流水 `monitor.md` 可作时序旁证（按边界本代理未读）。
4. **覆盖清单逐项有着落** ✅ — `fusion/final_ir.json` → `contract.coverage`：
   `layer_types` 五类 + 外围逐项定义（含 is_kda_layer +1 语义的层号范围）；
   `captured` 20 个作业逐项列出覆盖面；`folded` 声明代表层覆盖与 93+27 层模板折叠；
   层结构事实（69 KDA + 24 MLA = 93、layer0 dense）源码依据
   `configuration_kimi_k3.py:152-156`、`modeling_kimi_linear.py:893-900`；
   分支维度保真 + 规模缩水逐项声明（`model.scale_note`，不冒充全量——
   全量对照保留于 `model.config_snapshot`「config_json_全量对照」）。
5. **场景齐全** ✅ — prefill_vision / prefill_text / decode 三场景各有 ≥1 宏作业 ok
   （macro_prefill_vision / macro_prefill_text / macro_decode，capture_summary 20/20 ok）；
   节点 `scenarios` 字段与三场景集一致（fuse.py 依 measured 场景回填）。
6. **kernel 替换合规** ✅ — `contract.replacements` 4 项（R1/R2/R3/P），每项含
   what/why/equiv/evidence/shim_class；等价验证方法与结果留痕
   （`shims/equiv_check.py` 21/21，`shims/equiv_check_result.txt`）；验证等级如实标注
   为"对拍数学定义"级（无一虚标"对拍原实现"——fla 原生核与 flash_attn 均未安装且禁联网）；
   替换件行为与毛坯本征行为的区分写入相关节点 branch_note/concerns
   （q/k/v_conv1d、o_norm、self_attn 的 FA2 域记录）。
7. **IR 质量** ✅ — `final_ir.json` 过 `ir_schema.py`（69 节点、4 替换、8 哈希）；
   measured/logical 并列；mismatch/conflict 显式保留口径执行完毕：42 mismatch 全部
   三问归因（无 conflict——无不可裁决项，归因依据逐条落 branch_note/evidence）；
   无静默覆盖（机械草稿由 fuse.py 再生，人工裁决以字段叠加方式呈现，
   diff_report 同时保留机械清单与复核记录）。
8. **可视化质量** ✅ — 20 页可打开（TorchVista 交互/缩放正常；注入仅追加不改动原图）；
   shim 节点红虚线：本模型替换件均为导入级/注册级，图中无挂树 shim 节点，
   红虚线目标为 **0（忠实输出而非遗漏）**，替换提示由每页右上角融合面板的
   "等价替换清单"（4 项）承载，替换件在图中的具象体现（ShortConvolution 标签、
   conv1d/silu 算子、FA2 等价函数域内 fp32 matmul/softmax）在 `viz/README.md` 逐项说明；
   融合面板数字与 IR 一致（同一数据源生成）；llm_only 节点在面板有
   "仅逻辑分析、尚未运行验证"提示（未执行 ≠ 不存在）。
9. **全链可重跑** ✅ — 本 attempt 实际重跑了链路后半程验证幂等与确定性：
   `fuse.py`（重跑再生 69 节点机械草稿）→ 人工裁决脚本 → `ir_schema.py` 通过 →
   `render_fusion.py`（剥离旧注入后 20/20 重注入，exit 0，写后回读断言通过）→
   viz/ 副本更新。capture 侧命令 `capture_driver.py --spec ...` 未重跑（20/20 ok
   产物在案，重跑为纯重复计算），命令与契约见 SKILL §3 / spec.json。

## 三、诚实性红线逐条确认

- **编造 evidence 行号**：无——本 attempt 新增/修订的 evidence 行号均经 grep 逐条
  复核（如 :566-574 unpad、:583-600 conv、:703-746 gate、:841-877 moe_infer、
  :973-1046/:1075-1088 attn_res、:226-236 KimiRMSNorm、:64-82 SituAndMul、
  :783-815 PatchMergerMLPV2、:579-582 硬编码 512×512、:621-646 tpool）。
- **编造实测数据**：无——全部 measured 块原样取自 `capture/*.measured.json`
  （attn_res 回填块注明溯源字段 note）。
- **软化 schema 标准**：无——ir_schema.py 原样执行并通过；未修改 kit 任何脚本
  （本次无 kit 缺陷需要修复）。
- **把"仅逻辑分析"写成"已验证"**：无——5 个 llm_only 节点（mlp、mlp.activation、
  rope_2d、merge、lm.cache）保留并逐个写明不可实测原因与旁证范围；
  attn_res 升级 verified 仅在有实测回填块之后。
- **把替换件行为写成模型本征行为**：无——R1/R2/R3/P 四项在 replacements、
  面板清单、节点 branch_note/concerns 三处声明；conv cache (Bu,P,K-1) 布局、
  FA2 等价函数域内 fp32 算子等均注明"shim/等价函数行为"。
- **静默丢弃冲突节点或未覆盖模块**：无——0 conflict、0 capture_only、
  未覆盖模块路径 0；42 个 mismatch 逐条留痕于 diff_report 复核记录。

## 四、关键修正与如实披露（attempt 3 特有）

1. **逻辑侧修正（经三问归因确认逻辑侧有误，同步 evidence）**：
   - 投影器中间维：初稿「4096→128」×4 因子有误，实际 = 4×mm_hidden（缩水 512），
     符号 Hp（notes.md 追加条目 1 + scale_note）；
   - 视觉 MLP 中间维照抄全量字面量 → 符号 Iv（缩水 256）；
   - KDA unpad 展平分支（:566-574）与 Rope2D 512×512 硬编码（:579-582）初稿遗漏，
     以 Bu/Su/Dr 符号补全（追加条目 2/3）。
2. **机械融合口径现象（非逻辑或实测错误）**：场景级全局符号绑定、双层型共享模板
   （o_proj/g_proj 的 P vs Hq·Vd）、模板 I/O 域 vs 记录内部域、fp32 内部 vs bf16 I/O、
   运行时动态值（Ne/Nv0）——逐节点归因见 diff_report 复核记录。
3. **mtime 证据对失效**：见"〇、断点链"；独立性判定以内容级证据为准。
4. **spec.shims=[] 与红虚线 0 目标**：导入级/注册级替换件不适用类挂树机制
   （driver 的 shims[] 条目必须是可 add_module 的 nn.Module），强造条目会破坏
   全链重跑；面板清单承载替换声明。
5. **attempt 2 遗留处理**：capture/*.html 尾部旧草稿注入块已剥离（剥离后标记残留 0、
   文件尾还原为 TorchVista 原生结尾）再重注入；viz/ 旧副本与旧 README、旧 SELF-CHECK
   全部被定稿版覆盖。
6. **attn_res 回填**：实测块自层节点复制（普通方法共享 tree_path，无独立 scope），
   每块带 note 溯源字段；块边界追加与 output_attn_res 的未逐条展开部分在 concerns
   如实声明为源码推导。

## 五、遗留的开放问题（如实保留，均不阻断交付）

1. fla 原生 chunk_kda/fused_recurrent_kda 内核内部实现与 recurrent state 布局对应关系
   无法本机对拍（fla 未安装且禁联网）——shim 为"对拍数学定义"级（notes.md 开放问题 1）。
2. 视觉注意力生产路径为 flash_attn_varlen，实测走毛坯自带 eager 回退（数学定义相同，
   对 shape/dtype 结论无影响；notes.md 开放问题 2）。
3. 视觉 mlp.activation（PytorchGELUTanh）的 gelu 算子未被 torchvista 登记——激活存在性
   仅有源码与形状链旁证，节点保留 llm_only。
4. merge 图文合并的根域算子无法与 output_attn_res 逐条区分——节点保留 llm_only。
5. 权重为随机初始化，一切数值结论不外推（notes.md 开放问题 4）。

## 事后补充（kit 维护者，非本 run 交付）

kit 在本 run 之后升级了可视化流程：`viz/index.html` 集成面板与 `viz/views.json` 由
`agent-kit_ops-analysis/scripts/render_index.py` 事后生成（数据实时读取 capture_summary
与 final_ir，五态/替换件计数与 IR 同源）。本 run 其余交付物未改动；在升级后的 kit 下，
这两个文件属于阶段五标准产出（SPEC 交付物 9）。
