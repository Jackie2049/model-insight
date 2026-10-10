# RESULTS.md — 实测结果报告

两个真实模型的全量算子分析产物。每份产物由 `outputs/<模型名>/SELF-CHECK.md`
按 SPEC 九条标准逐条自检，本报告汇总关键数字与方法边界。

## 总览

| | DeepSeek-V3.2 | Kimi-K3 |
|---|---|---|
| 毛坯源文件 | 4 个 `.py` 入扫（含 inference/kernel.py） | 8 个 `.py` 入扫（语言 + 视觉 + 投影器） |
| 融合节点 | 68 | 69 |
| 五态分布 | 🟢 66 verified / 🟠 1 mismatch / ⚪ 1 llm_only | 🟢 64 verified / ⚪ 5 llm_only |
| 抓图作业 | 13（全成功，0 failed nodes） | 20（全成功，3 宏 + 17 微） |
| kernel 替换件 | 4 | 4 |
| 等价对拍 | 17/17 通过 | 21/21 通过 |
| 层结构 | 稠密层 0-2 + MoE 层 3；MLA 注意力 | KDA 层 93（layer0 为 dense）+ MLA 层 24 + 视觉塔 + 投影器 |
| 场景 | prefill（S=64）/ decode（S=1, start_pos=64） | prefill_text / prefill_vision / decode |

两个模型均为**缩水实例化**（关键分支维度保真，规模缩减，逐项对照见各自
`fusion/final_ir.json → model.scale_note`）+ 随机初始化权重：
**shape/dtype 结论保真，数值结论不可外推**。

## DeepSeek-V3.2

**覆盖**：prefill/decode 两场景 × 深度矩阵（d0-d4）+ 稠密块 ×3、MoE 块、
MLA 注意力、顶层各 ≥1 micro；层间重复按 `{i}/{j}` 模板折叠为推导覆盖。

**五态**：66 verified / 1 mismatch / 1 llm_only / 0 conflict。
唯一 mismatch 为**设计性保留**：`attn.indexer.act_quant` 同一 shim 有
q/k 两个调用点、形状不同，单节点覆盖取 q 侧——差异全程可追溯
（`fusion/diff_report.md` 复核记录第 1 条）。llm_only 为 `freqs_cis`
（旋转位置表，面板带"尚未运行验证"提示）。

**替换件**（`contract.replacements` 4 项，等价验证均为"对拍数学定义"级——
原 kernel 本机不可运行，未冒充"对拍原实现"级）：

| 替换件 | 机制 | 对拍结果 |
|---|---|---|
| act_quant（fp8 量化） | 挂树重绑 | 逐位一致 |
| fp8_gemm | 挂树重绑 | 误差 ≤1.7e-8 |
| fp8_index | 挂树重绑 | 误差 5.96e-8 |
| fast_hadamard_transform | sys_path 模块遮蔽 | bf16 噪声内（≤7.4e-3）+ 正交往返 |

挂树重绑在图中以红虚线渲染；sys_path 遮蔽无挂树实例，差异在台账内显式披露。

## Kimi-K3

**覆盖**：prefill_text / prefill_vision / decode 三场景宏作业 +
五类层型与全部外围模块的 micro 作业（kda_dense / kda_moe / mla_moe /
moonvit_layer / patchmergerv2，及 embed / norm / lm_head / gate / expert 等）；
93 + 24 层按代表层 + 模板折叠声明。

**五态**：64 verified / 5 llm_only / 0 mismatch / 0 conflict。
llm_only 5 项各有结构性不可实测原因：`mlp`（激活算子未被抓图引擎登记，
仅有源码与形状链旁证）、`rope_2d`（硬编码 512×512 表，逻辑侧修正后保留）、
`merge`（根域算子无法与注意力输出逐条区分）、`lm.cache`（非 Module 容器）等。

**融合裁决**：机械比对产生 42 处差异，按三问归因规程逐条裁决
（`fusion/diff_report.md` 复核记录），归因分布四类：

1. **逻辑侧字面量未随缩水实例化**（3 节点）——投影器中间维、视觉 MLP
   中间维照抄全量数值，修正为符号并同步 evidence；
2. **逻辑侧遗漏源码分支**（2 处）——KDA unpad 展平域、Rope2D 硬编码表，
   补全符号；
3. **机械融合口径现象**（多数）——两侧结论均正确、口径不同：场景级全局
   符号绑定冲突、双层型共享模板、模板 I/O 域 vs 记录内部域、fp32 内部算子
   vs bf16 I/O、运行时动态值；
4. **替换件行为声明**——conv1d 域布局与 cache 形状、FA2 等价函数域内
   fp32 算子等，如实标注为 shim 行为、不冒充模型本征。

无一条"为通过而强改"：所有状态变更均附归因与证据同步，机械差异清单
（42 条 raw diffs）与人工裁决记录在 diff_report 内并存可审计。

**替换件**（4 项）：einops 导入遮蔽、fla 六组件数学重实现
（Gated DeltaNet 系）、FA2 注册表覆盖（生产路径 flash_attn_varlen 不可用，
实测走毛坯自带 eager 回退，数学定义相同）、hooks 兼容补丁。
等价对拍 21/21，验证等级如实标注"对拍数学定义"级（fla 原生核未装且禁联网）。

## 双引擎独立性证据

- **时序链**：V3.2 逻辑笔记 `notes.md`（13:44:55）早于最早抓图产物
  （14:35:45）；K3 逻辑笔记 birth（13:53:28）早于抓图 summary（14:28:53）。
  逻辑分析完成于一切抓图开始之前，目录互相隔离、无交叉引用。
- **内容级硬证据**：逻辑侧 IR 携带全量字面量（投影器 4096、H=7168、
  词表 163840 等），与实测缩水值（512/1000）系统性矛盾——若逻辑侧读过
  实测数据不可能如此；这些矛盾恰好构成机械融合需裁决的差异来源。
- 每个 verified 节点的 `measured` 与 `logical` 块并列保存在
  `fusion/final_ir.json`，结论可逐字段复核。

## 已知边界

1. **缩水实例化**：规模缩减以适配单机显存（V3.2 671B→4 层/8 头/32 专家/
   vocab 1024；K3 同理），关键分支维度保真、逐项声明于 `scale_note`。
2. **随机初始化权重**：数值仅用于管线健康性；专家路由等运行时行为
   （如 V3.2 门控分数近似均匀、全 token 路由相同 top-8）是随机权重下的
   实测形态，非模型本征行为，已在 `concerns` 声明。
3. **验证等级上限**：全部替换件的等价验证为"对拍数学定义"级
   （原 kernel 本机不可运行且分析过程禁联网），未虚标更高级。
4. **单卡退化**：分布式模型以单卡执行，多卡行为（如 V3.2 Indexer 的
   dist.broadcast）不在结论内。
5. **抓图驱动依赖** torchvista 0.2.13 内部 API（requirements.txt 双锁版本），
   升级 torch / torchvista 前须先跑全链回归。

## 复现

```bash
# 1) 查看任一模型的结果页面
python3 -m http.server 8899
# 浏览器打开 http://127.0.0.1:8899/outputs/DeepSeek-V3.2/viz/
#            http://127.0.0.1:8899/outputs/Kimi-K3/viz/

# 2) 校验融合 IR schema（kit 自带）
python3 agent-kit_ops-analysis/scripts/ir_schema.py outputs/DeepSeek-V3.2/fusion/final_ir.json
python3 agent-kit_ops-analysis/scripts/ir_schema.py outputs/Kimi-K3/fusion/final_ir.json

# 3) 逐字节核对毛坯零修改
cd inputs && shasum -a 256 -c ../inputs-manifest.sha256
```

完整自检过程见各模型 `outputs/<模型名>/SELF-CHECK.md`；
方法与规程见 [agent-kit_ops-analysis/SKILL.md](agent-kit_ops-analysis/SKILL.md)。
