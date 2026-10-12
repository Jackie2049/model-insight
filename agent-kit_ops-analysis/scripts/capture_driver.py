#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2b 通用抓图驱动：读 spec.json（+ 可选 hooks.py）→ 构造模型 → 挂 shim →
按场景执行真实前向 → TorchVista 抓图 → 每个追踪作业产出 HTML + 实测 JSON。

本驱动不含任何模型知识：模型入口、构造参数、场景输入、shim 登记全部由
执行代理写进 spec.json（编写规程见 prompts/capture-planning.md）。

spec.json 契约（spec_version 1.0；所有路径相对工作区根目录）：
  {
    "spec_version": "1.0",
    "model": {
      "name": "显示名",
      "dir": "inputs/某模型",
      "entry": {
        "sys_path": ["…shim目录", "…模型目录"],   # 列表前面的路径优先级更高（可遮蔽同名模块）
        "module": "model", "class": "Model",
        "constructor_args": {}
      },
      "default_dtype": "bfloat16",   # 可选：构造前 set_default_dtype
      "seed": 0,                     # 可选：固定随机种子（权重可复现）
      "eval_mode": true,             # 可选：默认 true
      "init_distributed": false      # 可选：true 则在导入模型模块前做单进程初始化
    },
    "hooks": "outputs/…/capture/hooks.py",   # 可选：标准钩子（见下）
    "shims": [                                # 可选：等价替换件登记
      {"module": "…/shims/kernel.py", "class": "MyShim", "args": {},
       "mount_name": "_shim_x",
       "rebind": [{"module": "model", "attr": "my_kernel"}]}
    ],
    "traces": [
      {"id": "macro_prefill",        # 产物文件名前缀（必填、唯一）
       "kind": "macro",              # macro=整模型 | micro=子模块（target 路径）
       "target": "layers.4",         # micro 必填：相对（包裹后的）模型的 get_submodule 路径
       "scenario": "prefill",        # 场景标签（写进实测 JSON，供融合对齐）
       "inputs": {"args": [配方…]}   # 或 {"hook": true} 交给 hooks.make_inputs
       ,"forced_depth": 1,           # 可选：默认 1（调用时模块栈深度≥该值的模块才成节点）
       "collapse_depth": 1,          # 可选：仅影响 HTML 折叠
       "attr_names": true,           # 可选：显示名用属性名
       "height": 800,                # 可选
       "rebuild": true               # 可选：默认 true（该作业前重建模型；串联场景设 false）
      }
    ]
  }

输入配方（inputs.args 数组的元素，按位置顺序组成 tuple）：
  {"type": "float_tensor", "shape": [1, 8], "dtype": "bfloat16"}  # dtype 可省（用默认）
  {"type": "int_tensor", "shape": [1, 8], "low": 0, "high": 1023}
  {"type": "const", "value": 8}                                    # 非张量位置参数

hooks.py 标准钩子（全部可选，签名固定）：
  make_model(ctx) -> nn.Module          # 完全接管构造（ctx 见 DriverContext）
  make_inputs(scenario, model, ctx) -> tuple | dict   # 场景输入（返回 tuple 或 dict）
  wrap_model(model, ctx) -> nn.Module   # 构造+挂 shim 后、抓图前的适配器包裹

产物（写在 spec.json 所在目录）：
  <job_id>.html            TorchVista 交互图（失败作业也渲染部分图）
  <job_id>.measured.json   实测记录（容器数据 + 逐节点派生表 + 运行元信息）
  capture_summary.json     全部作业的状态汇总

用法（必须用工作区 .venv 的 python，见 SKILL.md 环境契约）：
  <工作区>/.venv/bin/python agent-kit_ops-analysis/scripts/capture_driver.py \
      --workspace <工作区根> --spec outputs/<模型>/capture/spec.json [--job 作业id]

退出码：0=全部作业成功；1=存在失败作业（部分图与实测 JSON 仍会落盘，供断图判读）。

⚠ 版本耦合警示：本驱动 import torchvista 的内部 API（process_graph / plot_graph /
  build_immediate_ancestor_map / ExportFormat 等，见下方 import 区）。torchvista
  版本升级前必须先重跑 tests/run_tests.py 全链回归；requirements.txt 已双锁版本。
"""
import argparse
import contextlib
import gc
import importlib
import importlib.util
import io
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------- 出错即停的中文工具
class SpecError(Exception):
    """spec.json 不符合契约（附中文说明）。"""


def _need(d, key, where):
    """取必填字段，缺失给中文报错。"""
    if key not in d or d[key] in (None, "", []):
        raise SpecError(f"{where} 缺少必填字段 {key!r}")
    return d[key]


# ---------------------------------------------------------------- 上下文（给 hooks 用）
class DriverContext:
    """标准钩子可见的驱动上下文：只暴露路径与 spec，不暴露内部状态。"""

    def __init__(self, workspace, model_dir, spec, current_job=None):
        self.workspace = workspace      # 工作区根（Path）
        self.model_dir = model_dir      # 模型毛坯目录（Path，只读）
        self.spec = spec                # 整个 spec 字典
        self.current_job = current_job or {}
        import torch
        self.torch = torch

    def log(self, msg):
        print(f"    [hooks] {msg}")


# ---------------------------------------------------------------- spec 校验
def load_spec(spec_path, workspace):
    """读入并校验 spec.json（结构级校验，中文报错）。"""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if spec.get("spec_version") != "1.0":
        raise SpecError(f"spec_version 必须为 '1.0'，当前 {spec.get('spec_version')!r}")
    m = _need(spec, "model", "spec")
    _need(m, "dir", "spec.model")
    _need(m, "entry", "spec.model")
    entry = m["entry"]
    if "hooks" not in entry and "make_model" not in entry:
        _need(entry, "module", "spec.model.entry")
        _need(entry, "class", "spec.model.entry")
    _need(entry, "sys_path", "spec.model.entry")
    if not isinstance(entry["sys_path"], list) or not entry["sys_path"]:
        raise SpecError("spec.model.entry.sys_path 必须是非空数组（至少含模型目录）")
    traces = _need(spec, "traces", "spec")
    if not isinstance(traces, list) or not traces:
        raise SpecError("spec.traces 必须是非空数组")
    ids = set()
    for i, job in enumerate(traces):
        where = f"spec.traces[{i}]"
        _need(job, "id", where)
        if job["id"] in ids:
            raise SpecError(f"traces 的 id 重复：{job['id']!r}")
        ids.add(job["id"])
        if job.get("kind", "macro") not in ("macro", "micro"):
            raise SpecError(f"{where}.kind 只能是 macro 或 micro")
        if job.get("kind") == "micro" and not job.get("target"):
            raise SpecError(f"{where} 为 micro 时必须提供 target（get_submodule 路径）")
        _need(job, "scenario", where)
        _need(job, "inputs", where)
    return spec


# ---------------------------------------------------------------- 模块加载
def import_file_as_module(path, module_name):
    """按文件路径把 .py 加载成模块（shim 文件 / hooks 文件用）。"""
    p = Path(path)
    if not p.exists():
        raise SpecError(f"文件不存在：{p}")
    spec = importlib.util.spec_from_file_location(module_name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod          # 注册后才能支持模块内相对引用
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 模型构建
def build_model(state):
    """按 spec 构建全新模型：种子 → dtype → (可选)分布式 → 导入 → 实例化 → eval → 挂 shim。

    state 里持有解析好的路径与模块缓存；每次调用产出一个干净实例。
    """
    import torch
    spec, ws = state["spec"], state["workspace"]
    m = spec["model"]

    # 1) 种子：固定后权重可复现（重建模型仍得到同一组随机权重）
    seed = m.get("seed", 0)
    torch.manual_seed(seed)
    random.seed(seed)

    # 2) 默认 dtype：必须在构造之前设置（构造期创建的参数会沿用）
    if m.get("default_dtype"):
        dt = getattr(torch, m["default_dtype"], None)
        if dt is None:
            raise SpecError(f"default_dtype 无法识别：{m['default_dtype']!r}（如 bfloat16/float32）")
        torch.set_default_dtype(dt)

    # 3) 分布式：必须在导入模型模块之前完成（模块导入期可能读取全局状态）
    if m.get("init_distributed") and not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(random.randint(20000, 45000)))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        print("  · 已完成单进程分布式初始化（gloo, world_size=1）")

    # 4) sys.path：列表前面的路径优先级更高（逆序 insert(0) 后，首个条目位于最前）
    #    —— 这是"shim 目录遮蔽毛坯内同名模块"的通用机制，靠声明顺序而非硬编码
    for rel_path in reversed(m["entry"]["sys_path"]):
        abs_path = (ws / rel_path).resolve()
        if not abs_path.exists():
            raise SpecError(f"sys_path 条目不存在：{rel_path}（解析为 {abs_path}）")
        s = str(abs_path)
        if s not in sys.path:
            sys.path.insert(0, s)

    # 5) 实例化：hooks.make_model 完全接管，否则 module.class(**constructor_args)
    if state["hooks"] is not None and hasattr(state["hooks"], "make_model"):
        model = state["hooks"].make_model(state["ctx_for_build"])
        if model is None:
            raise SpecError("hooks.make_model 返回了 None（必须返回 nn.Module）")
    else:
        entry = m["entry"]
        mod = importlib.import_module(entry["module"])
        state["entry_module"] = mod
        cls = getattr(mod, entry["class"], None)
        if cls is None:
            raise SpecError(f"模块 {entry['module']!r} 中找不到类 {entry['class']!r}")
        model = cls(**entry.get("constructor_args", {}))

    # 6) eval 模式（默认开）：避免 Dropout/BN 等随机行为污染测量
    if m.get("eval_mode", True):
        model.eval()
    elif model.training:
        print("  · ⚠️ 模型处于训练模式（eval_mode=false）：测量含随机行为，请在 IR 中声明")

    # 7) 挂 shim：以 nn.Module 进模型树（图上可见），可选重绑命名空间
    for sh in spec.get("shims", []):
        if sh["module"] not in state["shim_modules"]:
            state["shim_modules"][sh["module"]] = import_file_as_module(
                ws / sh["module"], f"_shim_mod_{Path(sh['module']).stem}")
        shim_mod = state["shim_modules"][sh["module"]]
        cls = getattr(shim_mod, sh["class"], None)
        if cls is None:
            raise SpecError(f"shim 模块 {sh['module']} 中找不到类 {sh['class']!r}")
        inst = cls(**sh.get("args", {}))
        model.add_module(sh["mount_name"], inst)
        for rb in sh.get("rebind", []):
            # 重绑目标：优先用已导入的入口模块；hook 构造时按 rb.module 名导入
            target_mod = state["entry_module"]
            if target_mod is None:
                target_mod = sys.modules.get(rb["module"]) or importlib.import_module(rb["module"])
            setattr(target_mod, rb["attr"], inst)
            print(f"  · shim 已挂树并重绑：{sh['mount_name']} → {rb['module']}.{rb['attr']}")

    # 8) 可选适配器包裹（构造+挂载之后）
    if state["hooks"] is not None and hasattr(state["hooks"], "wrap_model"):
        model = state["hooks"].wrap_model(model, state["ctx_for_build"])
    return model


# ---------------------------------------------------------------- 输入构造
def build_inputs(job, model, state):
    """按作业声明构造输入：hook=true 交给 hooks.make_inputs，否则按配方生成。"""
    spec, ws = state["spec"], state["workspace"]
    import torch
    inp = job["inputs"]
    if inp.get("hook"):
        if state["hooks"] is None or not hasattr(state["hooks"], "make_inputs"):
            raise SpecError(f"作业 {job['id']} 声明 inputs.hook=true，但 hooks.py 没有 make_inputs")
        ctx = DriverContext(ws, state["model_dir"], spec, job)
        args = state["hooks"].make_inputs(job["scenario"], model, ctx)
        if not isinstance(args, (tuple, dict)):
            raise SpecError("hooks.make_inputs 必须返回 tuple（位置参数）或 dict（具名输入）")
        return args
    recipes = inp.get("args")
    if not isinstance(recipes, list):
        raise SpecError(f"作业 {job['id']} 的 inputs 必须是 {{'args': [...]}} 或 {{'hook': true}}")
    out = []
    for i, r in enumerate(recipes):
        t = r.get("type")
        if t == "const":
            out.append(r.get("value"))
        elif t == "float_tensor":
            dtype = getattr(torch, r["dtype"]) if r.get("dtype") else torch.get_default_dtype()
            out.append(torch.randn(*r["shape"], dtype=dtype, device=r.get("device", "cpu")))
        elif t == "int_tensor":
            out.append(torch.randint(int(r.get("low", 0)), int(r.get("high", 1024)),
                                     r["shape"], device=r.get("device", "cpu")))
        else:
            raise SpecError(f"inputs.args[{i}] 的 type 无法识别：{t!r}"
                            f"（float_tensor / int_tensor / const，或改用 hook）")
    return tuple(out)


# ---------------------------------------------------------------- 派生表（融合的直接输入）
def derive_nodes(adj_list, module_info, func_info, node_to_ancestors,
                 node_to_attr_name, graph_node_name_to_without_suffix):
    """把容器数据整理成逐节点派生表：树路径、输入张量规格、出边形状、参数形状。

    树路径 = 从外到内的模块属性名链（ModuleList 层级不出现在栈上，天然省略；
    数字层索引保留）。模块节点与其内部 aten 算子共享同一段树路径前缀。
    """
    def attr_of(node):
        return (node_to_attr_name.get(node)
                or graph_node_name_to_without_suffix.get(node) or node)

    def tree_path(node):
        # node_to_ancestors[node] 是"由内到外"的祖先栈，反转成"由外到内"
        chain = list(node_to_ancestors.get(node, []))[::-1]
        parts = [attr_of(a) for a in chain]
        if node in module_info:            # 模块节点自己也占一段路径
            parts.append(attr_of(node))
        return ".".join(p for p in parts if p)

    def tensor_specs(value):
        """从 func_info 的某个参数值里递归抽出 tensor 规格。"""
        if isinstance(value, dict) and value.get("_type") == "tensor":
            return [{"shape": value["shape"], "dtype": value["dtype"]}]
        if isinstance(value, list):
            out = []
            for v in value:
                out.extend(tensor_specs(v))
            return out
        if isinstance(value, dict):
            out = []
            for v in value.values():
                out.extend(tensor_specs(v))
            return out
        return []

    nodes = []
    for name, data in adj_list.items():
        fi = func_info.get(name, {})
        ins = []
        for a in fi.get("positional_args", []):
            ins.extend(tensor_specs(a))
        for v in fi.get("keyword_args", {}).values():
            ins.extend(tensor_specs(v))
        mi = module_info.get(name, {})
        nodes.append({
            "name": name,
            "label": graph_node_name_to_without_suffix.get(name, name),
            "node_type": data.get("node_type"),
            "failed": bool(data.get("failed")),
            "tree_path": tree_path(name),
            "ancestors": list(node_to_ancestors.get(name, []))[::-1],
            "input_tensors": ins,                       # 输入侧实测 shape+dtype（金矿）
            "output_dims": [e.get("dims") for e in data.get("edges", [])],  # 出边形状串
            "module_type": mi.get("type"),
            "params": {k: v["shape"] for k, v in mi.get("parameters", {}).items()},
        })
    return nodes


# ---------------------------------------------------------------- 单个追踪作业
def run_job(job, model, state):
    """执行一个追踪作业：抓图 + 渲染 + 实测 JSON 落盘。返回作业状态记录。"""
    import torch
    from torchvista.engine import process_graph
    from torchvista.render import plot_graph
    from torchvista.graph_transforms import build_immediate_ancestor_map
    from torchvista.enums import ExportFormat

    ws, spec = state["workspace"], state["spec"]
    out_dir = state["out_dir"]
    job_id = job["id"]

    # 目标解析：macro=整模型；micro=子模块（相对包裹后的模型）
    if job.get("kind", "macro") == "macro":
        target = model
    else:
        try:
            target = model.get_submodule(job["target"])
        except AttributeError as e:
            return {"id": job_id, "status": "failed",
                    "error": f"micro target 路径解析失败：{job['target']!r}（{e}）"}

    inputs = build_inputs(job, model, state)

    # TorchVista 容器（与官方 trace_model 相同的一组；每作业全新）
    adj_list = {}
    module_info = {}
    func_info = {}
    parent_module_to_nodes = defaultdict(list)
    parent_module_to_depth = {}
    graph_node_name_to_without_suffix = {}
    graph_node_display_names = {}
    node_to_module_path = {}
    node_to_ancestors = defaultdict(list)
    repeat_containers = set()
    node_to_attr_name = {}
    forced = job.get("forced_depth", 1)
    collapse = job.get("collapse_depth", 1)
    attr_names = job.get("attr_names", True)
    height = job.get("height", 800)

    exception = None
    t0 = time.time()
    try:
        process_graph(
            target, inputs, adj_list, module_info, func_info,
            node_to_module_path, parent_module_to_nodes, parent_module_to_depth,
            graph_node_name_to_without_suffix, graph_node_display_names,
            node_to_ancestors, repeat_containers, node_to_attr_name,
            show_non_gradient_nodes=True,
            forced_module_tracing_depth=forced,
            show_module_attr_names=attr_names,
            show_compressed_view=False)     # 压缩视图会改写数据，测量一律不用
    except Exception as e:
        # 断图语义：process_graph 已尽量恢复环境，容器里保留"成功节点 + 失败节点"
        # ——部分图不是废品，是定位断点的依据（判读规则见 SKILL.md）
        exception = e
    elapsed = round(time.time() - t0, 2)

    # 实测 JSON：容器原样（可序列化部分）+ 派生表 + 运行元信息
    measured = {
        "job": {"id": job_id, "kind": job.get("kind", "macro"),
                "target": job.get("target", ""), "scenario": job["scenario"],
                "options": {"forced_depth": forced, "collapse_depth": collapse,
                            "attr_names": attr_names}},
        "run": {"status": "failed" if exception else "ok",
                "elapsed_sec": elapsed,
                "torch": torch.__version__,
                "exception": None if exception is None else {
                    "type": type(exception).__name__,
                    "message": str(exception)[:2000],
                    "traceback_tail": traceback.format_exc().splitlines()[-30:]},
                "spec_model_name": spec["model"].get("name", "")},
        "graph": {
            "adj_list": adj_list,
            "module_info": module_info,
            "func_info": func_info,
            "node_to_module_path": node_to_module_path,
            "node_to_ancestors": dict(node_to_ancestors),
            "node_to_attr_name": node_to_attr_name,
            "graph_node_display_names": graph_node_display_names,
            "graph_node_name_to_without_suffix": graph_node_name_to_without_suffix,
            "repeat_containers": sorted(repeat_containers),
        },
        "derived": {"nodes": derive_nodes(
            adj_list, module_info, func_info, node_to_ancestors,
            node_to_attr_name, graph_node_name_to_without_suffix)},
    }

    # 渲染 HTML（失败也渲染部分图；抑制 IPython display 的终端噪音）
    html_path = out_dir / f"{job_id}.html"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            plot_graph(
                adj_list, module_info, func_info, node_to_module_path,
                parent_module_to_nodes, parent_module_to_depth,
                graph_node_name_to_without_suffix, graph_node_display_names,
                node_to_attr_name,
                build_immediate_ancestor_map(node_to_ancestors, adj_list),
                collapse, height, job.get("width"),
                ExportFormat.HTML, attr_names,
                repeat_containers, show_modular_view=False, export_path=str(html_path))
    except Exception as e:
        return {"id": job_id, "status": "failed",
                "error": f"HTML 渲染失败：{type(e).__name__}: {e}"}

    (out_dir / f"{job_id}.measured.json").write_text(
        json.dumps(measured, ensure_ascii=False), encoding="utf-8")

    n_total = len(adj_list)
    n_failed = sum(1 for d in adj_list.values() if d.get("failed"))
    status = "failed" if exception else "ok"
    print(f"  · 作业 {job_id}：{status}（{elapsed}s，节点 {n_total}，其中失败 {n_failed}）"
          + (f" → {type(exception).__name__}: {str(exception)[:120]}" if exception else ""))
    return {"id": job_id, "status": status, "nodes": n_total, "failed_nodes": n_failed,
            "elapsed_sec": elapsed,
            "exception": None if exception is None else f"{type(exception).__name__}: "
                        f"{str(exception)[:300]}",
            "html": html_path.name, "measured": f"{job_id}.measured.json"}


# ---------------------------------------------------------------- 主流程
def main(argv=None):
    ap = argparse.ArgumentParser(description="S2b 通用抓图驱动（零模型知识）")
    ap.add_argument("--workspace", required=True, help="工作区根目录（spec 内相对路径的基准）")
    ap.add_argument("--spec", required=True, help="spec.json 路径（产物写在其所在目录）")
    ap.add_argument("--job", default=None, help="只跑指定 id 的作业（调试用）")
    args = ap.parse_args(argv)

    # 不写字节码：毛坯目录只读，绝不能留下 __pycache__
    sys.dont_write_bytecode = True

    ws = Path(args.workspace).resolve()
    spec_path = Path(args.spec).resolve()
    try:
        spec = load_spec(spec_path, ws)
    except SpecError as e:
        print(f"✗ spec.json 不符合契约：{e}")
        return 1
    out_dir = spec_path.parent
    model_dir = (ws / spec["model"]["dir"]).resolve()
    if not model_dir.exists():
        print(f"✗ spec.model.dir 不存在：{spec['model']['dir']}")
        return 1

    # hooks.py（可选）
    hooks = None
    if spec.get("hooks"):
        try:
            hooks = import_file_as_module(ws / spec["hooks"], "_ops_kit_hooks")
        except Exception as e:
            print(f"✗ hooks.py 加载失败：{type(e).__name__}: {e}")
            return 1

    state = {
        "spec": spec, "workspace": ws, "model_dir": model_dir,
        "out_dir": out_dir, "hooks": hooks, "entry_module": None,
        "shim_modules": {},
        "ctx_for_build": DriverContext(ws, model_dir, spec),
    }

    traces = [j for j in spec["traces"] if args.job in (None, j["id"])]
    if not traces:
        print(f"✗ 没有匹配的作业（--job {args.job}）")
        return 1

    print(f"◆ 抓图驱动启动：模型 {spec['model'].get('name', '')}，"
          f"{len(traces)} 个作业，产物目录 {out_dir}")
    results, model = [], None
    for job in traces:
        # rebuild（默认 true）：每个作业前重建干净模型（种子固定 → 权重一致）
        if model is None or job.get("rebuild", True):
            if model is not None:
                del model
                gc.collect()
            model = build_model(state)
        results.append(run_job(job, model, state))

    # 汇总
    summary = {"spec": str(spec_path), "jobs": results,
               "all_ok": all(r["status"] == "ok" for r in results)}
    (out_dir / "capture_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"◆ 完成：{n_ok}/{len(results)} 个作业成功；汇总已写入 capture_summary.json")
    for r in results:
        if r["status"] != "ok":
            print(f"  ✗ 作业 {r['id']} 失败：{r.get('error') or r.get('exception')}")
    if not summary["all_ok"]:
        print("  （失败作业的部分图与实测 JSON 已落盘——先做断图判读，再决定修复方式）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
