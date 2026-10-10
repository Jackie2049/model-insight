# S3 融合裁决规程（机械融合之后的人工复核）

> `fuse.py` 已完成机械可判的部分：路径对齐、符号绑定、shape/dtype 比对、
> 状态分配（verified / mismatch / llm_only / capture_only）与双方证据并置，
> 产出 `fusion/final_ir.json`（草稿）+ `fusion/diff_report.md`。
> 你的职责是**语义复核**：对每处不一致给出归因与修正，无法裁决的诚实保留。

## 1. mismatch 复核三问（按顺序）

对 diff_report 里每个 mismatch 节点：

1. **逻辑侧证据真的支持吗？** 回到 evidence 的 file:lines 重读源码——
   逻辑推导常见的错：维度映射搞反、忘了 view/transpose、把权重形状当输入形状、
   dtype 想当然。若逻辑侧错了：修正 `logical`，**必须同步修正/补充 evidence**，
   状态转 verified（或保留 mismatch 若实测侧也有疑点）。
2. **对齐对了吗？** 模板 path 是否匹配错了实测节点（同名不同物）？
   看匹配报告的 capture_node/capture_job 与树路径。对齐错了：修 path 后重跑
   `fuse.py`（机械融合可以重跑，逻辑分析原始记录 `llm-analysis/notes.md`
   不可回改——修正记录为新增条目）。
3. **替换件改变行为了吗？** 该节点在 shim 影响范围内吗（mystery kernel 的
   输出 dtype/形状是否本来就应该不同）？若实测反映的是替换件行为：在节点
   `branch_note`/`concerns` 声明"此实测值受等价替换影响"，不得当作模型
   本征行为写进结论。

三问都排除不了差异 → `status: "conflict"`，open_questions 写明两侧证据
（逻辑侧 evidence vs 实测 capture_node + 场景），**显式保留，不得静默取舍**。

## 2. capture_only（实测有、逻辑无）

diff_report 的"仅实测节点"是 S2a 的覆盖缺口：
- 补逻辑分析（读源码、补 evidence、logical 块），状态转 verified/mismatch；
- 或确认属于辅助/外部组件（缓存包装、工具模块）：在 concerns 说明为何
  不纳入全量分析，保留 capture_only 状态。

## 3. 符号绑定表复核

diff_report 的"场景符号绑定表"里，同一符号在同场景绑定冲突会被 fuse 标为
mismatch——这通常意味着逻辑侧把两个不同语义的维度用了同一个符号（如两个
"S"），或模型内部确有分支差异。按实际语义拆符号或补 branch_note。

## 4. 缩水声明合并

把 capture 侧的实例化参数（spec.constructor_args）与 S2a 的全量配置
（analysis_ir.model.config_snapshot）合并：`config_snapshot` 以**实际实例化
参数**为准，`scale_note` 逐项声明缩水（哪些维度缩了、缩成多少、为什么不影响
算子形态结论）。

## 5. 定稿自检（全部通过才算 S3 完成）

1. `python scripts/ir_schema.py fusion/final_ir.json` 校验通过；
2. diff_report 里每个 mismatch / capture_only 都有处置结果
   （修正 / 转冲突 / 说明保留），无"看都没看"的遗留；
3. replacements 数与 spec.shims 一致且 shim_class 关联齐全；
4. 修正逻辑侧时 evidence 同步更新；notes.md 只增不改；
5. 每个 conflict 节点的 open_questions 足以让第三方复现两侧证据。

## 6. 产物与下一步

- `fusion/final_ir.json`（定稿）+ `fusion/diff_report.md`（复核后可在文末
  追加"复核记录"小节：每个 mismatch 的归因一行）；
- 下一步：S4b 可视化（`render_fusion.py` 消费 final_ir + spec 注入 HTML）。
