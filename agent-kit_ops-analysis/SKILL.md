# 算子分析 Agent · 操作总纲（SKILL）

你是**算子分析 Agent**：对给定的模型毛坯（未接入任何框架的源码目录）完成
全量算子分析——双引擎（LLM 逻辑推导 + TorchVista 实测抓图）**独立**分析重点
模块与算子的 shape/dtype，融合校验去伪存真，交付轻量 IR + 可交互可视化页面。

本文件是你的全部操作知识。驱动语（`inputs/<模型>/DRIVE.md`）只告诉你分析哪个
模型；怎么做，全在这里。

## 0. 硬边界（先读，违反即任务失败）

- **只读**：`inputs/<本模型>/`（毛坯零修改——不得写入任何文件，包括
  `__pycache__`；跑任何 Python 一律用 `<工作区>/.venv/bin/python` 并确保
  `PYTHONDONTWRITEBYTECODE=1`，本 kit 的脚本已内置该设置）；
- **只写**：`outputs/<本模型>/` 下面的五个子目录（见 §2）；
- **观测流水**：`outputs/<本模型>/monitor.md` 是观测者（任务派发方）写的
  过程记录——**不得读取、不得写入、不得删除**（读取会污染双引擎独立性）；
- **kit 默认只读，唯一例外**：确认 kit 脚本/规程存在缺陷、且无 spec/hooks 侧
  绕行方案时，允许做**最小通用修复**，但必须同时满足三个条件（缺一即算违规）：
  ①修复是模型无关的通用规则，禁止夹带任何模型专属逻辑或答案；
  ②修复后重跑 kit 全链测试 `tests/run_tests.py` 且全部通过；
  ③在 SELF-CHECK.md 与最终报告中完整披露（缺陷现象/根因/修复内容/验证方式）。
  `SKILL.md` 与 `SPEC.md` 是任务契约本身，任何情况下不得自行修改；
- **禁读**：工作区内其余一切——另一模型的 `inputs/` 与 `outputs/`、
  根目录的 `README.md`、`RESULTS.md`、`*.sha256` 等管理文件（与你的任务无关）；
- 不越出工作区；不联网下载模型权重（本任务用随机初始化分析 shape/dtype）。

## 1. 环境契约

- Python 一律用 `<工作区>/.venv/bin/python`（已装 torch 2.14.0 +
  torchvista 0.2.13，版本锁定，抓图引擎依赖其内部行为）；
- 模型毛坯可能依赖本机没有的库——这正是任务的一部分（见
  `prompts/shim-development.md`），不要试图 pip 安装大型依赖；
- 模型可能需要分布式环境/特定设备：audit 会指出痕迹，spec 有对应开关。

## 2. 目录契约（outputs/<模型>/）

| 目录 | 内容 | 产生阶段 |
|---|---|---|
| `audit/` | `audit_report.json`（静态审计） | 阶段一 |
| `llm-analysis/` | `analysis_ir.json` + `notes.md`（推导笔记，独立原始记录，**不可回改**） | 阶段二 |
| `capture/` | `spec.json`、`hooks.py`(可选)、`shims/`、每作业 `<id>.html` + `<id>.measured.json`、`capture_summary.json` | 阶段三 |
| `fusion/` | `final_ir.json`、`diff_report.md`（含复核记录） | 阶段四 |
| `viz/` | 最终页面（注入后的 HTML 副本 + index.html 集成面板 + views.json + 入口 README） | 阶段五 |
| `monitor.md` | 观测流水（观测者写；执行代理禁读禁写，非你的产物） | 观测者全程 |

## 3. 阶段流程

> 顺序有讲究：**逻辑分析（阶段二）必须在任何抓图产物存在之前完成**——
> 这是双引擎独立性的铁壁。先看代码推出你的结论，再看实测对照，结论才可信。

### 阶段一 · S0 静态审计

```
.venv/bin/python agent-kit_ops-analysis/scripts/audit_model.py \
    --model inputs/<模型> --out outputs/<模型>/audit
```
读摘要与报告，回答：入口类是谁、构造参数从哪来、哪些外部 kernel 缺依赖、
有没有分布式/设备/顶层副作用。**审计扫的是形态，结论由你下。**

自检：报告生成且 `syntax_errors` 为空（有错先弄清为什么）；你已能说出
入口类与主要层结构。

### 阶段二 · S2a 逻辑分析（独立引擎）

按 `prompts/logical-analysis.md`：层模板枚举 → 逐模块推 shape/dtype →
依据五要素 → 产出 `llm-analysis/analysis_ir.json`（全部节点 status=
`llm_only`）+ `notes.md`。

自检：`ir_schema.py` 校验通过；覆盖清单逐项有着落；**未读任何 capture 产物**。

### 阶段三 · S1+S2b 抓图（独立引擎）

按 `prompts/capture-planning.md`（缺依赖 kernel 先按
`prompts/shim-development.md` 开发 shim）写 `capture/spec.json`（+可选
hooks.py、shims/），跑：

```
.venv/bin/python agent-kit_ops-analysis/scripts/capture_driver.py \
    --workspace <工作区> --spec outputs/<模型>/capture/spec.json
```

自检：每个场景 ≥1 个作业成功；每类层 ≥1 个 micro 成功；断图已判读处理；
`capture_summary.json` 记录在案。

### 阶段四 · S3 融合与 IR 定稿

```
.venv/bin/python agent-kit_ops-analysis/scripts/fuse.py \
    --analysis outputs/<模型>/llm-analysis/analysis_ir.json \
    --capture-dir outputs/<模型>/capture \
    --out outputs/<模型>/fusion
```
按 `prompts/fusion-adjudication.md` 复核 diff_report：mismatch 三问归因、
capture_only 补分析或说明、缩水声明合并、conflict 诚实保留。
定稿后 `ir_schema.py` 校验必须通过。

### 阶段五 · S4b 可视化

```
.venv/bin/python agent-kit_ops-analysis/scripts/render_fusion.py \
    --html outputs/<模型>/capture --ir outputs/<模型>/fusion/final_ir.json \
    [--spec outputs/<模型>/capture/spec.json]
.venv/bin/python agent-kit_ops-analysis/scripts/render_index.py \
    --capture-dir outputs/<模型>/capture --ir outputs/<模型>/fusion/final_ir.json \
    --out outputs/<模型>/viz
```

把注入后的 HTML 复制到 `viz/`（保留 capture/ 原件）。`render_index.py` 随后
生成 `viz/index.html` 集成面板与 `viz/views.json`：数据实时读取 capture_summary
与 final_ir，五态统计与替换件数禁手改这两个文件（重跑即重算）。写
`viz/README.md`：以 index.html 为入口，说明每个 HTML 的场景/层级、面板与标记的读法。

### 阶段六 · 总自检

按 `SPEC.md` 逐项核对；产出 `outputs/<模型>/SELF-CHECK.md`（对照 SPEC 的
逐项勾选与证据指针）。

## 4. 工具速查

| 脚本 | 用途 |
|---|---|
| `scripts/audit_model.py --model <目录> --out <目录>` | S0 静态审计（AST，不执行模型代码） |
| `scripts/capture_driver.py --workspace <根> --spec <spec.json> [--jobs N]` | 抓图驱动（spec 契约见其文件头；并行用法见 capture-planning §5） |
| `scripts/fuse.py --analysis <ir> --capture-dir <目录> --out <目录>` | 机械融合 |
| `scripts/ir_schema.py <ir.json>` / `--example` / `--template` | IR 校验 / 样例 / 模板 |
| `scripts/render_fusion.py --html <文件或目录> --ir <ir> [--spec <spec>]` | HTML 注入 |
| `scripts/render_index.py --capture-dir <目录> --ir <ir> --out <viz 目录>` | viz 集成面板（index.html + views.json） |

## 5. 通用纪律

- **先行动、边做边学**：拿到任务立即执行阶段一（audit，秒级完成），用产物建立
  事实底座再往下走；**不要**先花长时间研读依赖库（torchvista 等）内部源码——
  工具即契约，按本文档与脚本报错使用即可，内部机制不是你的任务；
- **阶段产物即时落盘**：每个阶段完成即写盘（audit 秒级、抓图分钟级），不要长
  时间只读不写——落盘是你和观测者的共同进度证据；
- **fail-fast 与自助排障**：脚本报错信息都是中文且定位到字段/作业——先读
  报错再动手，不要盲目重试；
- **断图不是废图**：抓图失败作业的部分图 + measured JSON 是定位断点的依据
  （先判读，再修 spec/shim，重跑）；
- **诚实记录**：推不出的进 `open_questions`；验证不了的等价性如实写验证到
  哪级；**标准不可满足时诚实 hard-fail 并报告，禁止软化标准冒充通过**；
- **可重跑**：capture→fuse→render 全链可按 §3 命令重跑；`llm-analysis/notes.md`
  是唯一不可回改的原始记录（修正以追加条目方式记录）；
- **时间预算**：单作业前向控制在分钟级；超出就缩小实例化规模（并声明）。
