# 通用交付与验收标准（SPEC）

> 本文件定义"算子分析 Agent 交付件"的验收口径，适用于任意模型。
> 阶段六总自检逐条对照本文件，证据指针写进 `outputs/<模型>/SELF-CHECK.md`。

## 交付物清单

| # | 交付物 | 位置 | 验收要点 |
|---|---|---|---|
| 1 | 静态审计报告 | `audit/audit_report.json` | 覆盖毛坯全部 .py；缺依赖清单与调用点齐全 |
| 2 | 逻辑分析 IR | `llm-analysis/analysis_ir.json` | 过 schema 校验；全节点 llm_only + evidence 五要素 |
| 3 | 推导笔记 | `llm-analysis/notes.md` | 独立原始记录，证据隔离下产生（判据见验收 3；落盘后不可回改） |
| 4 | 抓图 spec | `capture/spec.json` | 过 driver 校验；缩水实例化已声明 |
| 5 | shim 件 | `capture/shims/` | 每个 shim：挂树+接管双动作、数学定义、等价验证留痕 |
| 6 | 抓图产物 | `capture/*.html` + `*.measured.json` + `capture_summary.json` | 每场景 ≥1 作业 ok；每类层 ≥1 micro ok |
| 7 | 融合 IR | `fusion/final_ir.json` | 过 schema 校验；measured/logical 并列；五态使用正确 |
| 8 | 差异报告 | `fusion/diff_report.md` | 每个 mismatch/capture_only 有处置；复核记录在案 |
| 9 | 可视化页面 | `viz/`（注入后 HTML + index.html 集成面板 + views.json + README） | shim 红虚线 + 状态徽标 + 融合面板 + 场景/层级导览 |

> 另：`outputs/<模型>/monitor.md` 是观测者的过程流水——非代理交付物，
> 执行代理不得读写（见 SKILL §0），可作独立性与过程审计的旁证。

## 验收标准（通用九条）

1. **全中文**：交付件内文案、代码注释、报告一律中文。
2. **毛坯零修改**：`inputs/<模型>/` 无任何新增文件（含 `__pycache__`），
   文件哈希与任务开始时一致。
3. **双引擎独立性可审计**：**双向零交叉引用**——`llm-analysis/` 两件产物
   （analysis_ir.json 与 notes.md）全文不引用 capture 产物（spec/hooks/shims/
   html/measured/capture_summary）；`capture/` 的 spec、hooks、shims 与实测 JSON
   不引用 `llm-analysis/` 内容（可 grep 检查，证据指针写进 SELF-CHECK）。
   两侧产物目录隔离。时序仅作旁证而非必要条件：串行形态 mtime 先后、并行形态
   `monitor.md` 分线时间链皆可；并行执行（两线产物时间交错）为合法形态。
4. **覆盖清单逐项有着落**：`contract.coverage` 声明每类层型、实测代表层、
   折叠层范围；层结构事实（层数/层型分布）有源码依据；分支维度保真，
   规模缩水逐项声明（不冒充全量）。
5. **场景齐全**：核心场景（按模型实际形态，典型如 prefill/decode）每个都
   有成功作业；节点 `scenarios` 字段与场景集一致。
6. **kernel 替换合规**：每项替换有 what/why/equiv/evidence/shim_class；
   等价性验证方法与结果留痕；验证等级如实标注（对拍原实现 > 对拍数学定义 >
   仅元数据）。
7. **IR 质量**：`final_ir.json` 过 `ir_schema.py` 校验；实测/逻辑并列保存；
   mismatch/conflict 显式保留（含 open_questions）；无静默覆盖。
8. **可视化质量**：HTML 打开可用（交互、缩放正常）；shim 节点红虚线 +
   等价替换提示；融合面板数字与 IR 一致；llm_only 节点在面板有
   "尚未运行验证"提示；`viz/index.html` 集成面板可打开，场景/视图切换正常，
   其五态与替换件计数与 `final_ir.json` 一致，链接与注入页面一一对应。
9. **全链可重跑**：按 SKILL §3 命令能从 spec 重跑 capture→fuse→render→index
   且结果一致（幂等注入与整体覆盖已处理重复运行）。

## 诚实性红线（违反 = 交付不合格）

- 编造 evidence 行号 / 编造实测数据 / 软化 schema 标准；
- 把"仅逻辑分析"写成"已验证"；把替换件行为写成模型本征行为；
- 静默丢弃冲突节点或未覆盖模块。
