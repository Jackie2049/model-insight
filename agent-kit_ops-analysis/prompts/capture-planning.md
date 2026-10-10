# S1 抓图准备规程（spec.json 编写法）

> 目标：为 `capture_driver.py` 写出可一次跑通的 `spec.json`（+ 可选 hooks.py），
> 让模型在本机 CPU 上真实前向、TorchVista 抓图。driver 本体零模型知识，
> 一切模型判断都由你写进 spec。

## 1. 从审计报告出发

`outputs/<模型>/audit/audit_report.json` 的用法：
- `missing_deps` + `external_calls`：哪些外部 kernel 被哪些行调用——
  逐个判断是否在实际执行路径上（不在路径上的不必替换，但要记录判断依据）；
  在路径上的 → 按 `prompts/shim-development.md` 开发等价替换件后再写进 spec。
- `classes`（`nn_module_like` + `has_forward`）：入口类的候选；
  `config_like`：构造参数的字段与默认值底稿。
- `dist_refs`：模型是否在导入/构造期就要分布式环境（是 → spec 里
  `init_distributed: true`，driver 会在**导入模型模块之前**完成单进程初始化）。
- `toplevel_calls`：import 时刻副作用，提前预判（如自动下载、建目录、断言设备）。
- `weight_refs`：权重加载痕迹——本任务用随机初始化分析，须在 IR 里声明
  "未加载真实权重"（这不影响 shape/dtype 分析，但影响数值结论的解释）。

## 2. 实例化策略（缩水声明纪律）

- **最小可行**：内存装得下、单步前向秒级~分钟级；
- **分支维度优先保真**：注意力头数/分支结构等影响算子形态的维度不缩；
- **规模维度允许缩水**：层数、词表、专家数、中间维……可缩，但必须：
  - 每类层至少留 2 个实例（层间差异才可见）；
  - 缩水数值写进 spec 的 constructor_args，最终**逐项声明进 IR 的
    `model.scale_note`**（"分支保真、规模缩水"），不得冒充全量。

## 3. spec.json 逐项写法

契约全文见 `scripts/capture_driver.py` 文件头。要点：

- `model.entry.sys_path`：**列表前面的路径优先级最高**。需要用 shim 目录
  遮蔽毛坯内同名模块时，把 shim 目录放前面、模型目录放后面；
- `constructor_args`：配合审计的 config_like 字段推导；构造失败时 driver 的
  报错会指出缺什么参数——回配置类源码补；
- `default_dtype`：模型源码/权重期望的 dtype（构造前设置，参数沿用）；
- `seed`：固定（权重可复现，重建模型不漂移）；
- `shims[]`：每个 shim 登记 `module`（shim 文件路径）/`class`/`mount_name`
  （挂树属性名，建议 `_shim_` 前缀）/`rebind`（把 shim 实例重绑到入口模块的
  原函数名上——挂树提供图上可见性，重绑提供调用接管，两者都要）；
- `hooks`（可选）：三标准钩子任选——
  - `make_model(ctx)`：构造特殊（需要预处理权重/包装类）时完全接管；
  - `make_inputs(scenario, model, ctx)`：输入构造复杂（依赖模型属性、需要
    预填缓存等）时用，返回 tuple 或 dict；
  - `wrap_model(model, ctx)`：接口适配（把 kwargs 调用包成 positional）。

## 4. 追踪作业（traces[]）设计

- **至少一个 macro**（整模型，`forced_depth: 1`）× 每个核心场景；
  场景集按模型实际形态定（典型如 prefill / decode；有分支结构则加分支场景），
  覆盖全部场景是硬要求；
- **每类层一个 micro**（`kind: "micro"`，`target` = `get_submodule` 路径），
  输入尺寸用小规模（目的：看清层内算子链，不是复算全模型）；
- `rebuild: true`（默认）= 每作业前重建模型；需要"上一场景的缓存状态"的
  场景（如 decode 复用 prefill 的 KV）把后续作业设 `rebuild: false` 串联；
- micro 的树路径从子模块起算——融合用后缀对齐，spec 无需特殊处理。

## 5. 跑 driver 与诊断

```
<工作区>/.venv/bin/python agent-kit_ops-analysis/scripts/capture_driver.py \
    --workspace <工作区> --spec outputs/<模型>/capture/spec.json
```

- 看每个作业的中文结果行：`ok`（节点数/失败节点数）或失败原因；
- **断图判读**（作业 failed 时）：先开对应 `<作业id>.html` 看部分图断在哪、
  `<作业id>.measured.json` 的 `run.exception.traceback_tail` 定位——
  常见归因：缺 shim（audit 的 external_calls 漏判断）/ 构造参数不对 /
  输入形状不对 / 需要预置状态（改 hooks）。修 spec 或 shim 后重跑；
- 全部作业 `ok`（或失败节点数=0 且图中已含你需要的链路）才算 S2b 完成；
  `capture_summary.json` 留作记录。

## 6. 阶段自检

1. 每个场景至少一个作业成功；每类层至少一个 micro 成功；
2. spec 里每个 shim 都在 IR `contract.replacements` 有对应条目
   （补 `shim_class` 字段关联），`what/why/equiv/evidence` 四要素齐全；
3. 缩水声明草稿（constructor_args vs 全量配置）已写好，待融合阶段并入 IR；
4. spec.json / hooks.py / shims/ 全部在 `outputs/<模型>/capture/` 内
   （模型专属产物不入 kit）。
