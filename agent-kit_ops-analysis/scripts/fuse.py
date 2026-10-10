#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 机械融合引擎：逻辑分析 IR（2a 产物）× 实测 JSON（2b 产物）→ final_ir.json 草稿。

分工原则：本脚本只做"机械可判"的融合（路径对齐、符号绑定、shape/dtype 比对、
状态分配、双方证据并置），不做语义裁决——mismatch 的最终归因、无法对齐的原因
判断，由执行代理按 prompts/fusion-adjudication.md 复核后落进 IR（改逻辑侧须补
依据；确实无法裁决的标 conflict 保留）。

对齐规则（与 2a 的 path 契约配套）：
  - 2a 的 node.path = 模型根到目标的"属性链"，只含带 forward 的模块段
    （ModuleList 等容器段不出现在实测树路径里，故 path 也不写）；数字层段
    可写成模板变量（如 "{i}.attn.wq_a"）
  - 实测侧匹配路径 = derived.nodes[].tree_path（macro 作业从模型根起算；
    micro 作业为相对路径，由 load_measured 拼回完整契约路径，存于
    match_path，见 _strip_container_segments）
  - 匹配 = 模板正则对实测匹配路径整串匹配；允许一个 IR 节点匹配多个实测
    节点（一对多），也允许多场景分别匹配

比对规则：
  - 主输入 = 该节点 func_info 里第一个出现的输入张量（位置参数优先）
  - logical.shape 的维度符号（字符串）在"同一场景内"必须绑定一致；跨场景
    允许不同（如序列长度 prefill=8192 / decode=1）
  - dtype 归一化后比对（bf16≡bfloat16、fp32≡float32、torch. 前缀剥掉）
  - 状态分配：全部已测场景一致 → verified；任一场景不一致 → mismatch；
    从未测到 → llm_only；实测模块节点未被任何 IR 节点覆盖 → 追加 capture_only

产物（写入 --out 目录）：
  final_ir.json   融合草稿（measured.blocks 多场景并列；供代理复核后定稿）
  diff_report.md  差异报告（统计 / 符号绑定表 / mismatch 明细 / 未对齐双向清单）

用法：
  python fuse.py --analysis outputs/<模型>/llm-analysis/analysis_ir.json \
                 --capture-dir outputs/<模型>/capture \
                 --out outputs/<模型>/fusion
退出码：0=融合完成（是否存在 mismatch 由报告呈现，不是错误）。
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

# 不写字节码：毛坯目录只读，绝不能留下 __pycache__
sys.dont_write_bytecode = True

# dtype 归一化别名表（双侧统一到 torch 标准名）
_DTYPE_ALIAS = {
    "bf16": "bfloat16", "fp16": "float16", "half": "float16",
    "fp32": "float32", "float": "float32", "fp64": "float64", "double": "float64",
}


def norm_dtype(s):
    """dtype 归一化：剥 torch. 前缀 + 常见别名映射；归一不了的原文返回。"""
    if not isinstance(s, str):
        return str(s)
    t = s.strip().lower().removeprefix("torch.")
    return _DTYPE_ALIAS.get(t, t)


# ---------------------------------------------------------------- 路径模板
def template_regex(path):
    """把 IR 模板路径编译成命名组正则：{name} → (?P<name>[^.]+)。

    惯例约定 {i}/{j} 为数字段（ModuleList 索引，见本文件头部契约）——
    编成 \\d+ 以免误吸同位置的具名子模块（如 {i}.ffn.{j} 不应匹配
    ffn.gate / ffn.shared_experts）。
    """
    out, i, names = [], 0, []
    for m in re.finditer(r"\{([A-Za-z_]\w*)\}", path):
        out.append(re.escape(path[i:m.start()]))
        pat = r"\d+" if m.group(1) in ("i", "j") else r"[^.]+"
        out.append(f"(?P<{m.group(1)}>{pat})")
        names.append(m.group(1))
        i = m.end()
    out.append(re.escape(path[i:]))
    return re.compile("".join(out)), names


def match_paths(tpl, measured_path):
    """模板匹配实测匹配路径：整串匹配（fullmatch）。

    自 load_measured 引入 match_path 前缀拼回后，micro 作业的相对 tree_path
    已拼成完整契约路径（剥容器段的 target 前缀 + 相对路径），全部实测路径
    与模板同为"模型根起算的属性链"——整串匹配即足。早先的"段边界后缀匹配"
    是为未拼回的相对路径兜底的，拼回后成为误配源（如单段模板 {i} 会后缀
    误吸 3.ffn.5 这类深层路径），故移除；误伤检查：移除当轮已验证无任何
    仅靠 suffix 命中的匹配对。

    返回 (是否匹配, 匹配方式, 模板变量取值)。
    """
    rx, names = template_regex(tpl)
    m = rx.fullmatch(measured_path)
    if m:
        return True, "full", {n: m.group(n) for n in names if m.group(n) is not None}
    return False, "", {}


# ---------------------------------------------------------------- 读取
def load_analysis(path):
    """读 2a 逻辑分析 IR；基础结构缺失给中文报错。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("nodes"), list) or not data["nodes"]:
        sys.exit(f"✗ analysis_ir.json 缺少 nodes（2a 产物不完整）")
    for i, nd in enumerate(data["nodes"]):
        if not nd.get("path"):
            sys.exit(f"✗ analysis_ir.json nodes[{i}] 缺少 path")
        if not nd.get("logical"):
            sys.exit(f"✗ analysis_ir.json nodes[{i}] 缺少 logical（2a 必须先完成逻辑侧）")
    return data


def _strip_container_segments(target: str) -> str:
    """剥掉 get_submodule 路径里的 ModuleList 容器段（用于 micro 拼回匹配路径）。

    树路径契约不含 ModuleList 容器段（torchvista 的树路径天然剥除），但
    job.target 是 get_submodule 路径、含容器段（如 layers.3.ffn.experts.5）。
    规则（heuristic，文档化）：段序列中"纯字母段紧跟纯数字段"时，该字母段
    是容器段（ModuleList 名），剥除——layers.0 / experts.5 均命中；属性链
    （ffn.gate、attn.indexer 等字母-字母）不受影响。
    """
    segs = target.split(".")
    out = []
    for k, s in enumerate(segs):
        if s.isalpha() and k + 1 < len(segs) and segs[k + 1].isdigit():
            continue                      # 容器段（ModuleList 名），剥除
        out.append(s)
    return ".".join(out)


def load_measured(capture_dir):
    """读全部 *.measured.json；按 (tree_path, scenario, job) 建索引。

    micro 作业的 tree_path 从子模块起算（相对路径）——为使其可与全模型模板
    对齐，额外构造 match_path = 剥容器段后的 job.target 前缀拼回（target 是
    get_submodule 路径、含 ModuleList 容器段，而树路径契约不含容器段，须剥除
    才能对上模板；见 _strip_container_segments）。match_path 只用于匹配与
    对齐判空；tree_path 原样保留在记录里。
    """
    entries = []          # 每条 = 一个实测模块节点在某作业中的记录
    failed_jobs = []
    for fp in sorted(glob.glob(str(Path(capture_dir) / "*.measured.json"))):
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        job_id = data["job"]["id"]
        scenario = data["job"]["scenario"]
        job_target = data["job"].get("target", "") or ""
        is_micro = data["job"].get("kind", "macro") == "micro" and bool(job_target)
        if data["run"]["status"] != "ok":
            failed_jobs.append({
                "job": job_id, "scenario": scenario,
                "exception": data["run"]["exception"]["message"] if data["run"].get("exception") else ""})
        for nd in data["derived"]["nodes"]:
            tp = nd["tree_path"]
            if is_micro:
                prefix = _strip_container_segments(job_target)
                match_path = f"{prefix}.{tp}" if tp else prefix
            else:
                match_path = tp
            entries.append({
                "job": job_id, "scenario": scenario,
                "name": nd["name"], "tree_path": tp, "match_path": match_path,
                "label": nd["label"], "module_type": nd["module_type"],
                "node_type": nd["node_type"], "failed": nd["failed"],
                "input_tensors": nd["input_tensors"],
                "output_dims": nd["output_dims"],
            })
    return entries, failed_jobs


# ---------------------------------------------------------------- 比对
def primary_input(entry):
    """主输入 = func_info 里第一个输入张量（位置参数优先）。"""
    return entry["input_tensors"][0] if entry["input_tensors"] else None


def compare_shapes(logical_shape, measured_shape, bindings):
    """逐维比对；符号写入 bindings 并检查场景内一致性。

    返回 (一致?, 中文差异说明)；符号冲突也视为不一致（维度语义被破坏）。
    """
    if len(logical_shape) != len(measured_shape):
        return False, (f"维度数不同：逻辑 {logical_shape} vs 实测 {measured_shape}")
    for k, (ld, md) in enumerate(zip(logical_shape, measured_shape)):
        if isinstance(ld, str):                       # 维度符号
            if ld in bindings and bindings[ld] != md:
                return False, (f"第 {k} 维符号 {ld} 在本场景内绑定冲突："
                               f"既有 {bindings[ld]} 又有 {md}（维度语义不一致）")
            bindings[ld] = md
        elif ld != md:
            return False, (f"第 {k} 维不同：逻辑 {ld} vs 实测 {md}")
    return True, ""


def alignable(entry):
    """可参与对齐的实测记录：模块节点 + 算子节点（aten）。

    根直属子模块在 forced_depth=1 的 macro 图里不登记为模块节点，但它的
    内部 aten 算子会登记且树路径正是该模块路径——把算子节点纳入池子，
    这些模块的实测才对得上。Input/Output/Constant/Parameter 不参与对齐。
    """
    if not entry.get("match_path"):
        return False
    if entry["module_type"]:
        return True
    return (entry.get("node_type") or "").lower() == "operation"


# ---------------------------------------------------------------- 主流程
def main(argv=None):
    ap = argparse.ArgumentParser(description="S3 机械融合引擎")
    ap.add_argument("--analysis", required=True, help="2a 的 analysis_ir.json")
    ap.add_argument("--capture-dir", required=True, help="2b 抓图产物目录（*.measured.json）")
    ap.add_argument("--out", required=True, help="融合产物目录")
    args = ap.parse_args(argv)

    analysis = load_analysis(args.analysis)
    entries, failed_jobs = load_measured(args.capture_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 对齐池：模块节点 + 算子节点（见 alignable 说明）；idx 一律是 entries 的下标
    pool = [idx for idx, e in enumerate(entries) if alignable(e)]
    print(f"◆ 读入：IR 节点 {len(analysis['nodes'])}，实测记录 {len(entries)} 条"
          f"（对齐池 {len(pool)} 条）"
          + (f"；失败作业 {len(failed_jobs)} 个" if failed_jobs else ""))

    # ---- 逐 IR 节点对齐 + 比对 ----
    report_lines = []
    matched_entry_ids = set()      # 已被 IR 覆盖的实测记录（id = index in entries）
    # 场景级符号绑定表：scenario -> {符号: 值}
    scenario_bindings = {}
    node_reports = []

    for nd in analysis["nodes"]:
        matches = []               # (entries 下标, 匹配方式, 模板变量取值)
        for idx in pool:
            ok, basis, groups = match_paths(nd["path"], entries[idx]["match_path"])
            if ok:
                matches.append((idx, basis, groups))
        nd.setdefault("measured", None)
        if not matches:
            nd["status"] = "llm_only"
            nd["measured"] = None
            node_reports.append({"path": nd["path"], "status": "llm_only",
                                 "detail": "任何场景均未测到（未执行 ≠ 不存在，页面将带标注）"})
            continue

        blocks, issues, ok_all = [], [], True
        # 按场景归并（同场景可能 macro/micro 多条记录，都保留）；
        # 场景内两层取材：有模块节点记录时只用模块记录（模块主输入 = 模块逻辑
        # 形态的同粒度证据），该场景仅有 aten 记录时才用 aten 兜底（forced_depth
        # 下不成模块节点的根直属子模块，其 aten 挂在模块路径上——见 alignable）。
        by_scenario = {}
        for idx, basis, groups in matches:
            e = entries[idx]
            by_scenario.setdefault(e["scenario"], []).append((idx, basis, e))
        for sc, lst in by_scenario.items():
            if any(e["module_type"] for _, _, e in lst):
                lst = [t for t in lst if t[2]["module_type"]]
            bindings = scenario_bindings.setdefault(sc, {})
            sc_ok, sc_notes = True, []
            for idx, basis, e in lst:
                matched_entry_ids.add(idx)
                pi = primary_input(e)
                block = {"scenario": sc, "capture_node": e["name"], "capture_job": e["job"]}
                if pi:
                    block["shape"] = pi["shape"]
                    block["dtype"] = norm_dtype(pi["dtype"])
                else:
                    sc_notes.append(f"{e['name']}：无输入张量可对比（仅记录出边形状）")
                    block["output_dims"] = e["output_dims"][:6]
                    blocks.append(block)
                    continue
                ok, why = compare_shapes(nd["logical"].get("shape", []), pi["shape"], bindings)
                dtype_ok = norm_dtype(nd["logical"].get("dtype", "")) == block["dtype"]
                if not ok:
                    sc_ok = False
                    sc_notes.append(f"{e['name']}（{basis} 对齐）：{why}")
                elif not dtype_ok:
                    sc_ok = False
                    sc_notes.append(f"{e['name']}（{basis} 对齐）：dtype 不同："
                                    f"逻辑 {nd['logical'].get('dtype')} vs 实测 {block['dtype']}")
                block["output_dims"] = e["output_dims"][:6]
                blocks.append(block)
            if not sc_ok:
                ok_all = False
                issues.append({"scenario": sc, "notes": sc_notes})
        nd["measured"] = {
            "kind": "real_exec",
            "blocks": blocks,
        }
        if ok_all:
            nd["status"] = "verified"
            node_reports.append({"path": nd["path"], "status": "verified",
                                 "detail": f"{len(blocks)} 条实测记录全部一致"
                                           f"（场景：{sorted(by_scenario)}）"})
        else:
            nd["status"] = "mismatch"
            nd.setdefault("open_questions", [])
            for iss in issues:
                nd["open_questions"].append(
                    f"[场景 {iss['scenario']}] " + "；".join(iss["notes"])
                    + "（待复核：逻辑侧推导有误？对齐错了？还是替换件改变了行为？）")
            node_reports.append({"path": nd["path"], "status": "mismatch",
                                 "detail": json.dumps(issues, ensure_ascii=False)})

    # ---- 实测有、IR 没有的模块节点 → 追加 capture_only ----
    # key 用 match_path（micro 作业含 job.target 前缀，保留定位上下文）
    uncovered = {}
    for idx, e in enumerate(entries):
        if idx in matched_entry_ids or not e["module_type"] or not e["tree_path"]:
            continue
        key = e["match_path"]
        u = uncovered.setdefault(key, {"tree_path": key, "module_type": e["module_type"],
                                       "blocks": [], "jobs": set()})
        pi = primary_input(e)
        blk = {"scenario": e["scenario"], "capture_node": e["name"],
               "capture_job": e["job"]}
        if pi:
            blk["shape"] = pi["shape"]
            blk["dtype"] = norm_dtype(pi["dtype"])
        blk["output_dims"] = e["output_dims"][:6]
        u["blocks"].append(blk)
        u["jobs"].add(e["job"])
    for key, u in sorted(uncovered.items()):
        analysis["nodes"].append({
            "path": key,
            "stage": f"（实测发现，逻辑分析未覆盖：{u['module_type']}）",
            "ops_flow": [u["module_type"]],
            "measured": {"kind": "real_exec", "blocks": u["blocks"]},
            "logical": None,
            "evidence": [],
            "scenarios": sorted({b["scenario"] for b in u["blocks"]}),
            "branch_note": None,
            "concerns": ["逻辑分析未覆盖此节点——补充分析或确认其属于辅助路径"],
            "status": "capture_only",
            "open_questions": ["为何逻辑分析没有枚举到该模块？（遗漏/辅助组件/命名差异）"],
        })
        node_reports.append({"path": key, "status": "capture_only",
                             "detail": f"{u['module_type']}，实测于 {'、'.join(sorted(u['jobs']))}"})

    # ---- 落盘：final_ir.json 草稿 ----
    analysis["model"]["scale_note"] = analysis["model"].get("scale_note", "")
    final_path = out_dir / "final_ir.json"
    final_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    # ---- 校验草稿（复用 ir_schema；llm_only 节点缺 measured 是合法的）----
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ir_schema import validate
        errs = validate(analysis)
        if errs:
            print(f"◆ ⚠️ 草稿未通过 schema 校验（{len(errs)} 处，多为 2a 侧缺字段，按提示补齐）")
            for e in errs[:10]:
                print(f"    - {e}")
    except ImportError:
        print("◆ ⚠️ 找不到 ir_schema.py，跳过草稿自校验")

    # ---- diff_report.md ----
    n_by = {}
    for r in node_reports:
        n_by[r["status"]] = n_by.get(r["status"], 0) + 1
    lines = ["# 融合差异报告（机械融合产出，供复核裁决）", ""]
    lines.append(f"- IR 节点（含追加）：{len(analysis['nodes'])}；"
                 f"状态分布：{json.dumps(n_by, ensure_ascii=False)}")
    lines.append(f"- 实测记录：{len(entries)} 条；已对齐 {len(matched_entry_ids)} 条；"
                 f"未覆盖模块路径 {len(uncovered)} 个")
    if failed_jobs:
        lines.append(f"- ⚠️ 失败作业 {len(failed_jobs)} 个（断图判读提示）：")
        for fj in failed_jobs:
            lines.append(f"    - {fj['job']}（场景 {fj['scenario']}）：{fj['exception'][:160]}")
    lines.append("")
    lines.append("## 场景符号绑定表")
    for sc, b in sorted(scenario_bindings.items()):
        pairs = "、".join(f"{k}={v}" for k, v in sorted(b.items()))
        lines.append(f"- {sc}：{pairs or '（无符号）'}")
    lines.append("")
    for st, title in (("mismatch", "不一致节点（须复核）"), ("llm_only", "仅逻辑节点"),
                      ("capture_only", "仅实测节点（2a 未覆盖）"), ("verified", "已验证节点")):
        rows = [r for r in node_reports if r["status"] == st]
        lines.append(f"## {title}（{len(rows)}）")
        for r in rows:
            lines.append(f"- `{r['path']}`：{r['detail'][:400]}")
        lines.append("")
    lines.append("## 复核指引")
    lines.append("- mismatch：先核对 evidence 的行号是否真的支持逻辑侧；再看对齐是否正确"
                 "（路径模板 vs 实测树路径）；最后考虑替换件是否改变了行为。")
    lines.append("- 修正逻辑侧时必须同步补/改 evidence；无法裁决的把 status 改为 conflict 并写 open_questions。")
    lines.append("- capture_only：属于逻辑分析的覆盖缺口，补分析（或确认辅助路径后在 concerns 说明）。")
    (out_dir / "diff_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"◆ 融合完成：{n_by.get('verified', 0)} verified / {n_by.get('mismatch', 0)} mismatch / "
          f"{n_by.get('llm_only', 0)} llm_only / {n_by.get('capture_only', 0)} capture_only")
    print(f"◆ 产物：{final_path} 与 {out_dir / 'diff_report.md'}")
    print("◆ 下一步：按 prompts/fusion-adjudication.md 复核 diff_report，定稿后重跑 ir_schema.py 校验")
    return 0


if __name__ == "__main__":
    sys.exit(main())
