# S2a 逻辑分析规程（LLM 逻辑推导引擎）

> 你在双引擎架构里扮演**逻辑推导引擎**：只凭模型源码与静态审计结果，推导全部
> 重点模块/算子的 shape 与 dtype。本阶段**独立工作**——不读取、不参照任何
> 抓图（实测）产物；实测侧由另一阶段独立产生，融合阶段才相互校验。

## 1. 输入与边界

- 可读：模型毛坯源码（`inputs/<模型>/`，只读）+ S0 审计报告
  （`outputs/<模型>/audit/audit_report.json`）
- 禁读：`capture/`、`fusion/`、`viz/`（并行形态下它们可能与本阶段并行产生；
  无论存在与否均不读、不等、不参考）
- 产出：`outputs/<模型>/llm-analysis/analysis_ir.json`（见 §6）

## 2. 覆盖纪律：层模板枚举法

1. **先数结构**：通读毛坯，列出全部层类型与其数量、每类的分支结构
   （注意力/前馈/门控/缓存……按源码实际形态命名，不套用外部框架术语）。
   config_like 类（审计报告 `config_like` 段）的字段默认值是构造参数的第一手材料。
2. **逐类解剖一个模板层**：为每类层选一个代表（通常第 0 层），逐行读 forward，
   枚举其中**每个会产生张量的步骤**（线性投影、归一化、激活、reshape/view/
   transpose、量化/反量化、索引、掩码、缓存读写……），每个步骤是一个 IR 节点。
3. **模板路径**：同类层的节点 path 用模板变量表示数字段（如 `{i}.attn.q_proj`），
   在 `contract.coverage.layer_types` 声明每类层覆盖的层号范围；层间差异
   （如有）单独开节点并在 `branch_note` 说明。
4. **全量有着落**：embedding、最终 head、层外的独立组件（缓存类、路由器……）
   一律入 IR。宁可节点多，不可静默漏。

## 3. path 契约（与融合对齐配套，必须严格遵守）

`node.path` = 模型根到目标的**属性名链**，规则：
- 只写源码里真实存在的属性名（`self.xxx = …` 登记名 / ModuleList 索引）；
- **容器段（ModuleList 等自身无 forward 的层）不写**——实测树路径天然不含它们，
  写了会导致融合对齐失败（例：源码 `self.layers[4].attn.q_proj` → path 写
  `4.attn.q_proj`，不写 `layers.` 段）；
- 数字段可用 `{i}` 模板（`{i}` 只代表一个由数字构成的段）；
- 模块内的算子级节点挂在所属模块的 path 下（可加后缀段，如 `4.attn.q_proj.op`）。

## 4. shape / dtype 推导纪律

- **从构造参数出发**：先确定配置→维度的映射（审计的 `config_like` 字段、
  `__init__` 里的形状推导代码），每个维度符号都要在 `dim_meaning` 里给出含义。
- **逐模块传递**：沿 forward 数据流推导主输入张量形状，每个节点记录
  `logical.shape`（整数或维度符号）与 `logical.dtype`。
- **dtype 依据**：权重 dtype（构造/转换代码）、源码里的字面量（`.to(…)`、
  `torch.xxx_tensor`）、算子语义（哪些算子保持/改变 dtype）。
- **分支场景参数化**：prefill / decode 等场景下形状不同的维度用符号
  （如 `S`），不要写死；场景列表写进 `scenarios`。
- **不确定就声明**：推不出的维度/数值，节点进 `open_questions`，禁止编造。

## 5. 依据五要素（每条逻辑结论必须挂依据）

每个节点的 `evidence` 至少一条：
```json
{"file": "相对毛坯根的路径", "lines": [起, 止], "symbol": "所属类/函数",
 "conclusion": "这段代码支持的具体结论（形状/维度/dtype 从哪来）",
 "kind": "source_derived"}
```
- `kind` 三分：`source_derived`（纯源码推导，本阶段默认）/ `real_exec`（真实执行，
  本阶段不会出现）/ `fake_meta`（假执行/元数据，本阶段不会出现）。
- `file` 必须能在 `model.source_files_hash` 里找到（哈希由审计报告 `files` 段提供，
  逐文件抄录——这是"逻辑结论可回溯到具体源码版本"的锚点）。
- 行号必须指向真实支撑该结论的代码，不是顺手抄的。

## 6. 产物：analysis_ir.json

结构与最终 IR 完全同构（见 `scripts/ir_schema.py`，用 `--example` 看样例），
区别只在：
- 全部节点 `status = "llm_only"`、无 `measured` 块（融合阶段由 `fuse.py` 填）；
- `model.config_snapshot` 写**源码推导的全量配置**（构造默认值/配置类字段）；
- `model.scale_note` 先写"逻辑侧按全量配置推导；实例化缩水见 capture 侧声明"
  （融合阶段会合并）；
- `contract.replacements` 本阶段先留空数组（kernel 替换在 capture 侧发生）。

**自检（全部通过才算 S2a 完成）**：
1. `python scripts/ir_schema.py outputs/<模型>/llm-analysis/analysis_ir.json` 校验通过；
2. 层结构清单 vs IR 节点：每类层、每个层外组件都有节点，无静默遗漏；
3. 每个节点都有 evidence 且行号真实；
4. 逻辑分析过程的中间笔记（读了哪些文件、推导草稿）保留在
   `llm-analysis/notes.md`——这是独立性审计的证据，融合后不可回改。
