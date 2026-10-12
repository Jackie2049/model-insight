# 并行编排规程（派发方专用）

> 本文件写给**任务派发方（观测者）**：把 S0 之后的阶段二与阶段三分派给两个
> 互不通信的空白代理并行执行，汇合后派第三个空白代理完成阶段四五六。
> 执行代理只读 SKILL.md 与各自线别的 prompts，不读本文件；派发方持有它。
> 独立性铁壁的并行形态（双向禁读）见 SKILL §3 引言；串行形态的次序强制不变。

## 1. 时序与目录归属

```
S0 静态审计（派发方完成或派代理完成）→ T0 同时派发两线
  逻辑线 = SKILL §3 阶段二   → 只写 outputs/<模型>/llm-analysis/
  抓图线 = SKILL §3 阶段三   → 只写 outputs/<模型>/capture/
两线完成 → 派发方汇合校验 → 派融合代理 = 阶段四五六
  融合代理 → 只写 outputs/<模型>/fusion/、viz/、SELF-CHECK.md
```

- `audit/` 在 T0 前就位；两线代理**禁重跑审计、禁写 audit/**（防双写竞争）；
- 两线产物落盘时间交错属正常（并行形态），独立性判据按 SPEC 验收 3 的
  双向零交叉引用执行，时序仅作旁证。

## 2. 任务书公共头部（三模板共用）

```text
- 角色与形态：你是空白代理，只执行本任务书界定的范围；本任务书与 SKILL 冲突时
  以任务书为准（SKILL 是方法契约，任务书是执行范围的权威）
- 模型 / 工作区根 / kit 根：inputs/<模型>；<工作区根>；agent-kit_ops-analysis/
- 前置状态：S0 已完成——outputs/<模型>/audit/audit_report.json 已在盘。
  禁止重跑审计，禁止写 audit/ 目录
- 环境契约：Python 一律 <工作区根>/.venv/bin/python（SKILL §1）
- 硬边界：SKILL §0 全文适用（毛坯只读零修改；monitor.md 禁读禁写；kit 禁改，
  三条件例外条款照旧；不越出工作区）
- 汇报格式：完成时返回——产物清单（路径+字节数）、自检结果、各步耗时、
  token 用量、遇到的报错与处置；不粘贴大段产物内容
```

## 3. 任务书模板

### 3.1 逻辑线任务书（阶段二）

```text
<公共头部>
- 身份：你是「逻辑线代理」，双引擎独立引擎之一；另有其他代理在做其他工作，
  你不与之通信、不关心其存在
- 范围：仅 SKILL §3 的阶段二（S2a 逻辑分析）。不执行阶段三及之后任何阶段
- 可读：inputs/<模型>/（只读）、outputs/<模型>/audit/audit_report.json、
  prompts/logical-analysis.md、SKILL §0/§1/§2 与 §3 阶段二小节
- 禁读：outputs/<模型>/ 下除 audit/ 外的一切（capture/、fusion/、viz/ 等）——
  即便已在盘上也不读、不等、不参考
- 只写：outputs/<模型>/llm-analysis/（仅此一目录）
- 产物与完成标志：analysis_ir.json + notes.md 落盘；scripts/ir_schema.py 校验
  通过；prompts/logical-analysis.md 的阶段自检全过；notes.md 记录你读过的文件
  与推导链（独立性审计的原始记录，落盘后不可回改）
- 完成即停：按汇报格式返回后等待，不进入任何后续阶段
```

### 3.2 抓图线任务书（阶段三）

```text
<公共头部>
- 身份：你是「抓图线代理」，双引擎独立引擎之一；不与其他代理通信
- 范围：仅 SKILL §3 的阶段三（S1 抓图准备 + S2b 抓图运行）。
  不执行阶段二、阶段四及之后任何阶段
- 可读：inputs/<模型>/（只读）、outputs/<模型>/audit/audit_report.json、
  prompts/capture-planning.md、prompts/shim-development.md、
  SKILL §0/§1/§2 与 §3 阶段三小节
- 禁读：outputs/<模型>/llm-analysis/、fusion/、viz/——即便已在盘上也不读、
  不等、不参考（双向禁读铁律的抓图侧）
- 只写：outputs/<模型>/capture/（仅此一目录）
- 抓图运行要求：capture_driver.py 带 --jobs N 运行（N 由派发方给定）；
  退出码 1 时按 capture-planning.md §5 断图判读，修 spec/shim 后重跑
- 【可选 pin 段——派发方按需填写】作业清单与缩水覆盖表固定如下
  （<作业 id/场景/forced_depth/rebuild 表>；<缩水表>）；未 pin 的方面
  （shim 方案、hooks 写法）由你按规程自行推导
- 产物与完成标志：spec.json + hooks.py + shims/ + 每作业 <id>.html +
  <id>.measured.json + capture_summary.json；capture-planning.md §6
  阶段自检四条全过
- 完成即停：按汇报格式返回后等待，不进入融合
```

### 3.3 融合代理任务书（阶段四五六）

```text
<公共头部>
- 身份：你是「融合代理」；两线（逻辑、抓图）产物已齐备并经派发方校验。
  你与两线代理同样互不通信
- 范围：SKILL §3 的阶段四、五、六（S3 融合 → S4b 可视化 → 总自检）
- 可读：outputs/<模型>/ 全部产物（两侧证据自本阶段起允许会合）、SKILL 全文、
  prompts/fusion-adjudication.md
- 禁读：monitor.md（观测流水，非交付物）；kit 之外的一切
- 只写：outputs/<模型>/fusion/、viz/、SELF-CHECK.md（llm-analysis/ 与
  capture/ 只读——notes.md 不可回改，capture 产物禁改）
- 硬性动作：fuse.py → 按 prompts/fusion-adjudication.md 逐条复核
  （mismatch 三问、capture_only 处置）→ ir_schema.py 校验 →
  render_fusion.py + render_index.py → 按 SPEC.md 九条逐项自检。
  其中验收 3 按新判据执行：双向零交叉引用（对两侧产物 grep 对方目录特征词，
  证据指针写进 SELF-CHECK）+ monitor.md 分线时间链作旁证
- 完成即停：汇报含五态统计、九条逐项勾选、耗时与 token 用量
```

## 4. monitor.md 分线记录规范（观测者）

- 每条 `[T+mm:ss] [线别] 事件`；线别 ∈ {观测, 逻辑线, 抓图线, 融合}；
  T0 = 两线派发时刻（unix 时间戳写入 T0 文件；S0 审计发生在 T0 前，记为 T- 事件）；
- 必记事件：各线派发/启动、关键产物落盘（notes.md、analysis_ir、spec.json、
  shims、首个作业、capture_summary）、各线完成、汇合校验、融合代理派发、
  SELF-CHECK 落盘、交付；
- 观测者不得在 monitor.md 写任何会把一侧产物内容泄漏给另一侧的摘要
  （执行代理虽禁读 monitor.md，保持记录无交叉内容才是干净的证据链）；
- 观测纪律：终止任何执行代理前，以执行瞬间的进程与产物双确认为准；长静默期
  先发非破坏探针再考虑处置（教训沉淀见 RERUN-R2-REPORT §六）。
