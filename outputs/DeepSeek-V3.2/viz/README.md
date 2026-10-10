# DeepSeek-V3.2 算子分析 · 可视化导览

本目录是 **S2b 实测抓图（TorchVista）+ S3 融合状态注入** 后的 13 份交互式调用图 HTML。
逻辑侧（S2a）结论与实测的逐节点对照见 `../fusion/final_ir.json`（68 节点：66 verified / 1 mismatch / 1 llm_only），裁决记录见 `../fusion/diff_report.md` 复核记录节。

> 标注提醒：本模型为**缩水实例化 + 随机初始化**（未加载真实权重，分支维度保真、规模维度缩水，逐项声明见 `../fusion/final_ir.json` 的 `model.scale_note`）。图上的 **shape/dtype 结论**不受影响；**数值**仅用于管线健康性判断。

## 从哪份开始看

建议按"整体骨架 → 逐层深入 → 专项细看"三步：

1. `macro_prefill_d0.html` —— 最小骨架（20 节点）：顶层 embed / 4 个 Block / norm / head，看模型总体数据流；
2. 逐深度下钻：`d1`（层内主链 attn/ffn）→ `d2`（MLA 子模块、Gate、专家调度）→ `d3`（线性层与 Indexer 内部、专家分支）→ `d4`（全部 shim 调用点，最完整）；
3. 专项细看：`micro_mla_attn.html`（MLA + DSA Indexer 全链，115 节点）是算子密度最高的一份。

## 文件清单（13 份）

| 文件 | 场景 | forced_depth | 节点数 | 适合看什么 |
|---|---|---|---|---|
| `macro_prefill_d0.html` | prefill (B=1,S=64) | 0 | 20 | 顶层骨架、Block 粒度 |
| `macro_prefill_d1.html` | prefill | 1 | 48 | 层内主链 attn/ffn/attn_norm |
| `macro_prefill_d2.html` | prefill | 2 | 555 | MLA 子模块、Gate 路由、专家调度、kv 量化 |
| `macro_prefill_d3.html` | prefill | 3 | 1623 | Linear 内部（act_quant→fp8_gemm）、Indexer 子模块、专家实例 |
| `macro_prefill_d4.html` | prefill | 4 | 3817 | 全部 shim 调用点（最完整，文件最大） |
| `macro_decode_d0.html` | decode (B=1,S=1,start_pos=64) | 0 | 17 | decode 骨架；**rebuild=false，真实串联 prefill 缓存续写** |
| `macro_decode_d1.html` | decode | 1 | 45 | decode 层内主链 |
| `macro_decode_d2.html` | decode | 2 | 644 | decode 的 MLA/MoE 子模块（注意 KV 追加读路径） |
| `macro_decode_d3.html` | decode | 3 | 1692 | decode 的 Indexer（T=start_pos+1=65）与专家分支 |
| `macro_decode_d4.html` | decode | 4 | 3658 | decode 全部 shim 调用点 |
| `micro_dense_block.html` | prefill（块级输入直喂） | 0 | 11 | 稠密 Block（layers.0）签名与输入构造 |
| `micro_moe_block.html` | prefill | 0 | 11 | MoE Block（layers.3）签名与输入构造 |
| `micro_mla_attn.html` | prefill | 0 | 115 | MLA 注意力 + DSA Indexer 全链专项（含 3 个 shim） |

decode 作业除 d0 外在重建模型上运行（KV 缓存零填充、start_pos=64，形状/dtype 与串联一致，声明见 scale_note ②）。

## 面板与标记怎么读

页面顶部有注入的融合信息横幅，图内节点有两类叠加标记：

- **红虚线描边 = 等价替换件（shim）**。毛坯的 3 个 tilelang kernel（act_quant / fp8_gemm / fp8_index）本机不可运行，由纯 torch 等价实现接管（挂树重绑）；Hadamard 走模块遮蔽不显形。红虚线节点的数值来自替换件，其等价性依据（对拍数学定义级自检 17 项全过）见 `../fusion/final_ir.json` 的 `contract.replacements`（4 项台账）。
- **状态徽标 = 融合裁决**（按显示名聚合）：**绿描边 verified**（逻辑推导与实测逐维一致）；**橙描边 mismatch**（形状/dtype 不一致，本图集仅 1 处：Indexer 的 act_quant 双调用点设计性 mismatch，归因见 diff_report 复核记录第 1 条）。
- 横幅图例与统计条给出本页 verified / mismatch / llm_only / capture_only 计数；横幅中还有替换件清单（4 项）与 mismatch 清单。
- **llm_only 提示**：横幅会提示"另有 N 个节点仅逻辑分析、尚未运行验证"——本模型仅 `freqs_cis` 一项（构造期常量，无前向记录可抓，声明见 diff_report 复核记录第 3 条）。看到 llm_only 字样时，含义是"仅有逻辑推导、未运行验证"，**不是**"已验证通过"。
- TorchVista 原生面板（点选节点）：输入/输出 shape 与 dtype、调用栈、源码位置，可与 `final_ir.json` 中同路径节点的 `logical.shape` / `dim_meaning` 对照。

## 图路径与 IR 模板节点的对应

IR（`final_ir.json`）用模板路径聚合层间重复结构：`{i}` = 层号（本实例化 0-3），`{j}` = 专家号（0-31）；图上看到的具体路径（如 `layers.0.attn.indexer.act_quant`）对应模板 `{i}.attn.indexer.act_quant`。同一路径在多深度作业下重复出现时，IR 按"模块记录优先于算子记录"去重，实测 blocks 保留各（场景，作业）代表记录，无静默丢弃。

## 环境备注

抓图在宿主 Python（`.venv`，torch + torchvista）完成；tilelang 与 fast_hadamard_transform CUDA 库未安装（即替换件存在的原因，见 audit `missing_deps`）。`init_distributed=true`（毛坯 Indexer 有 dist.broadcast，多卡语义下单卡退化执行，声明于 IR 对应节点）。

> 事后补充（kit 维护者）：`index.html` 为聚合面板，由 `render_index.py` 生成
> （场景/视图切换、iframe 预览、分析摘要），浏览各页面从它进入即可。
