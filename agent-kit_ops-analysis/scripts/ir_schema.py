#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量 IR schema：定义与校验（阶段 4a 的契约工具）。

本文件是"算子分析 Agent"的核心契约之一：双引擎（LLM 逻辑分析 / TorchVista 实测抓图）
的融合产物 final_ir.json 必须通过本校验器的校验。

schema 要点（ir_version 1.0）：
  - 三段式：model（模型与缩水声明）/ contract（场景、kernel 替换、覆盖清单）/ nodes（算子节点）
  - 铁律①：measured（实测）与 logical（逻辑）并列保存，冲突不得静默覆盖
  - 铁律②：未执行 ≠ 不存在——仅有逻辑侧的节点，status 必须是 llm_only（页面会带
    "仅逻辑分析，尚未运行验证"标注）
  - status 五态：verified（双方一致）/ mismatch（形状或 dtype 不一致）/ llm_only（仅逻辑）/
    capture_only（仅实测）/ conflict（融合复核后仍无法裁决的冲突）
  - 依据五要素：file（源码相对路径）+ lines（起止行号）+ symbol（所属符号）+
    conclusion（该代码支持的具体结论）+ 文件哈希（在 model.source_files_hash 中统一登记，
    evidence.file 必须能在其中找到对应）
  - evidence.kind 三分：real_exec（真实执行测得）/ fake_meta（假执行/元数据推得）/
    source_derived（纯源码推导）

用法：
  python ir_schema.py <ir.json>            # 校验一个 IR 文件
  python ir_schema.py --example            # 打印一份合法的最小示例
  python ir_schema.py --template out.json  # 生成模板文件供填写

退出码：0=校验通过；1=校验失败（错误清单打印到 stdout，全部为中文说明）。
"""
import json
import sys
from pathlib import Path

# 不写字节码：毛坯目录只读，绝不能留下 __pycache__
sys.dont_write_bytecode = True

IR_VERSION = "1.0"

# status 合法取值（五态）
STATUS_VALUES = {"verified", "mismatch", "llm_only", "capture_only", "conflict"}
# evidence.kind 合法取值（三分）
EVIDENCE_KINDS = {"real_exec", "fake_meta", "source_derived"}
# 实测记录的 kind 合法取值
MEASURED_KINDS = {"real_exec", "fake_meta"}


class ValidationError(Exception):
    """携带中文错误说明的校验异常。"""


def _err(errors, node_id, msg):
    """把一条错误（定位到节点路径或段名）追加进清单。"""
    where = f"[{node_id}] " if node_id else ""
    errors.append(f"{where}{msg}")


def _check_str(errors, obj, key, node_id, required=True, allow_none=False):
    """检查 obj[key] 是字符串（或按允许为 None）。"""
    v = obj.get(key)
    if v is None:
        if allow_none:
            return None
        if required:
            _err(errors, node_id, f"缺少字段 {key!r}")
        return None
    if not isinstance(v, str) or not v.strip():
        _err(errors, node_id, f"字段 {key!r} 必须是非空字符串")
    return v


def _check_lines(errors, obj, node_id):
    """检查 lines 字段：[起始行, 结束行]，两个正整数且起始 ≤ 结束。"""
    v = obj.get("lines")
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        _err(errors, node_id, "lines 必须是 [起始行, 结束行] 两个整数")
        return
    a, b = v
    if not (isinstance(a, int) and isinstance(b, int) and 1 <= a <= b):
        _err(errors, node_id, f"lines 取值非法：{v}（应为 1 起始、起始≤结束的行号）")


def validate(data):
    """校验一个已加载的 IR 字典。返回错误清单（空清单 = 通过）。"""
    errors = []
    if not isinstance(data, dict):
        return ["顶层必须是 JSON 对象"]

    # ---------- ir_version ----------
    if data.get("ir_version") != IR_VERSION:
        _err(errors, "ir_version", f"必须为 {IR_VERSION!r}，当前为 {data.get('ir_version')!r}")

    # ---------- model 段 ----------
    model = data.get("model")
    if not isinstance(model, dict):
        _err(errors, "model", "缺少 model 段（必须是对象）")
        model = {}
    _check_str(errors, model, "name", "model")
    # source_files_hash：文件相对路径 -> sha256；evidence.file 必须能对上
    sfh = model.get("source_files_hash")
    if not isinstance(sfh, dict) or not sfh:
        _err(errors, "model", "source_files_hash 必须是非空对象（文件相对路径 -> 哈希）")
        sfh = sfh if isinstance(sfh, dict) else {}
    for k, v in sfh.items():
        if not isinstance(k, str) or not isinstance(v, str) or len(v) < 8:
            _err(errors, "model", f"source_files_hash[{k!r}] 的键值必须是字符串（哈希至少 8 位）")
    if not isinstance(model.get("config_snapshot"), dict):
        _err(errors, "model", "config_snapshot 必须是对象（实例化参数快照）")
    # scale_note：缩水声明——有缩水就必须写清，没缩水也要显式说明"未缩水"
    _check_str(errors, model, "scale_note", "model")

    # ---------- contract 段 ----------
    contract = data.get("contract")
    if not isinstance(contract, dict):
        _err(errors, "contract", "缺少 contract 段（必须是对象）")
        contract = {}
    scenarios = contract.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        _err(errors, "contract", "scenarios 必须是非空数组")
        scenarios = []
    for i, sc in enumerate(scenarios):
        _check_str(errors, sc, "id", f"contract.scenarios[{i}]")
    # replacements：每项 what/why/equiv/evidence 四要素
    reps = contract.get("replacements", [])
    if not isinstance(reps, list):
        _err(errors, "contract", "replacements 必须是数组")
        reps = []
    for i, r in enumerate(reps):
        rid = f"contract.replacements[{i}]"
        if not isinstance(r, dict):
            _err(errors, rid, "必须是对象")
            continue
        _check_str(errors, r, "what", rid)      # 替换了什么
        _check_str(errors, r, "why", rid)       # 为什么必须替换
        _check_str(errors, r, "equiv", rid)     # 数学等价说明
        _check_str(errors, r, "evidence", rid)  # 原实现的源码依据（行号等）
    # coverage：覆盖清单（全量声明的载体）
    cov = contract.get("coverage")
    if not isinstance(cov, dict):
        _err(errors, "contract", "缺少 coverage 段（层型清单/实测代表/折叠声明）")
        cov = {}
    if not isinstance(cov.get("layer_types"), dict) or not cov.get("layer_types"):
        _err(errors, "contract.coverage", "layer_types 必须是非空对象（层型 -> 层号范围/说明）")
    if not isinstance(cov.get("captured"), list):
        _err(errors, "contract.coverage", "captured 必须是数组（实测覆盖的代表层，逐项列出）")
    _check_str(errors, cov, "folded", "contract.coverage")  # 折叠层的推导覆盖声明

    # ---------- nodes 段 ----------
    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        _err(errors, "nodes", "nodes 必须是非空数组")
        nodes = []
    paths = set()
    for i, nd in enumerate(nodes):
        nid = f"nodes[{i}]"
        if not isinstance(nd, dict):
            _err(errors, nid, "必须是对象")
            continue
        path = _check_str(errors, nd, "path", nid)
        if path:
            if path in paths:
                _err(errors, nid, f"path 重复：{path}")
            paths.add(path)
        _check_str(errors, nd, "stage", nid, allow_none=True)   # 所属层/阶段（人类可读）
        ops_flow = nd.get("ops_flow", [])
        if not isinstance(ops_flow, list) or not all(isinstance(s, str) for s in ops_flow):
            _err(errors, nid, "ops_flow 必须是字符串数组（算子序列）")

        measured = nd.get("measured")
        logical = nd.get("logical")
        has_m = isinstance(measured, dict) and bool(measured)
        has_l = isinstance(logical, dict) and bool(logical)

        # 实测块结构：kind + 多场景 blocks（每场景一条实测记录，可只测部分场景）
        if has_m:
            m = measured
            if m.get("kind") not in MEASURED_KINDS:
                _err(errors, nid, f"measured.kind 必须是 {sorted(MEASURED_KINDS)} 之一")
            blocks = m.get("blocks")
            if not isinstance(blocks, list) or not blocks:
                _err(errors, nid, "measured.blocks 必须是非空数组（每场景一条：scenario/shape/dtype）")
                blocks = []
            for j, b in enumerate(blocks):
                bid = f"{nid}.measured.blocks[{j}]"
                if not isinstance(b, dict):
                    _err(errors, bid, "必须是对象")
                    continue
                sc = _check_str(errors, b, "scenario", bid)
                # 同场景允许多条记录（模板节点匹配到的每个实例各一条；
                # capture_node 字段区分实例）
                shp = b.get("shape")
                if shp is not None and (not isinstance(shp, list)
                                         or not all(isinstance(d, int) for d in shp)):
                    # shape 允许缺省（该节点无输入张量可对比时省略键）；给了就必须是整数数组
                    _err(errors, bid, "shape 必须是整数数组（实测形状不含符号）")
                # dtype：有 shape 时必填；无输入张量的记录块可整体省略
                if b.get("shape") is not None:
                    _check_str(errors, b, "dtype", bid)
        # 逻辑块结构
        if has_l:
            lshp = logical.get("shape")
            if not isinstance(lshp, list) or not all(
                    isinstance(d, (int, str)) for d in lshp):
                _err(errors, nid, "logical.shape 必须是数组（每维为整数或维度符号字符串）")
            if not isinstance(logical.get("dim_meaning"), dict):
                _err(errors, nid, "logical.dim_meaning 必须是对象（维度符号 -> 含义）")
            _check_str(errors, logical, "dtype", nid)
            _check_str(errors, logical, "kind", nid)

        # 铁律①：至少一侧存在
        if not has_m and not has_l:
            _err(errors, nid, "measured 与 logical 至少要有一侧（并列保存，允许只有一侧）")
        # 铁律②：仅有逻辑侧必须显式 llm_only / conflict
        status = nd.get("status")
        if status not in STATUS_VALUES:
            _err(errors, nid, f"status 必须是 {sorted(STATUS_VALUES)} 之一，当前 {status!r}")
        else:
            if not has_m and status not in ("llm_only", "conflict"):
                _err(errors, nid, "无 measured 的节点 status 只能是 llm_only 或 conflict（未执行≠不存在）")
            if not has_l and status not in ("capture_only", "conflict"):
                _err(errors, nid, "无 logical 的节点 status 只能是 capture_only 或 conflict")

        # evidence：依据五要素（file/lines/symbol/conclusion，哈希在 model 段统一登记）
        evs = nd.get("evidence", [])
        if not isinstance(evs, list):
            _err(errors, nid, "evidence 必须是数组")
            evs = []
        if has_l and not evs:
            # 逻辑侧存在则必须有源码依据（实测侧的依据是抓图产物本身，可不重复）
            _err(errors, nid, "存在 logical 的节点必须提供至少一条 evidence（依据五要素）")
        for j, ev in enumerate(evs):
            eid = f"{nid}.evidence[{j}]"
            if not isinstance(ev, dict):
                _err(errors, eid, "必须是对象")
                continue
            f = _check_str(errors, ev, "file", eid)
            _check_lines(errors, ev, eid)
            _check_str(errors, ev, "symbol", eid)
            _check_str(errors, ev, "conclusion", eid)
            if ev.get("kind") is not None and ev.get("kind") not in EVIDENCE_KINDS:
                _err(errors, eid, f"evidence.kind 若填写必须是 {sorted(EVIDENCE_KINDS)} 之一")
            # 依据文件必须在 model.source_files_hash 中登记（可回溯到具体哈希）
            if f and f not in sfh:
                _err(errors, eid, f"依据文件 {f!r} 未在 model.source_files_hash 中登记（无法回溯哈希）")

        # scenarios：该节点适用的场景列表
        sc_list = nd.get("scenarios", [])
        if not isinstance(sc_list, list):
            _err(errors, nid, "scenarios 必须是数组")
        # open_questions：未决问题（conflict/mismatch 节点必须有）
        oq = nd.get("open_questions", [])
        if not isinstance(oq, list):
            _err(errors, nid, "open_questions 必须是数组")
        if status in ("conflict", "mismatch") and not oq:
            _err(errors, nid, f"status={status} 的节点必须写 open_questions（冲突显式保留）")

    return errors


EXAMPLE = {
    "ir_version": "1.0",
    "model": {
        "name": "示例模型",
        "source_files_hash": {"model_dir/model.py": "0123456789abcdef"},
        "config_snapshot": {"dim": 64, "n_layers": 2},
        "scale_note": "分支维度保真；词表与中间维缩水（声明具体数值）",
    },
    "contract": {
        "scenarios": [{"id": "prefill"}, {"id": "decode"}],
        "replacements": [{
            "what": "自定义 kernel my_quant",
            "why": "本机无该库/仅 GPU 可用",
            "equiv": "数学等价说明：块内 amax 缩放 + 低位宽 cast",
            "evidence": "model_dir/kernel.py:10-25",
        }],
        "coverage": {
            "layer_types": {"dense_block": "layers.0", "attn_block": "layers.1"},
            "captured": ["dense_block@layer0", "attn_block@layer1"],
            "folded": "两类型层均已实测代表层，无折叠层",
        },
    },
    "nodes": [{
        "path": "layers.{i}.proj",
        "stage": "第 {i} 层 → 投影",
        "ops_flow": ["linear"],
        "measured": {"kind": "real_exec",
                     "blocks": [{"scenario": "prefill", "shape": [1, 8, 64],
                                 "dtype": "bfloat16", "capture_node": "proj_12"}]},
        "logical": {"shape": ["B", "S", "dim"],
                     "dim_meaning": {"B": "批量", "S": "序列长度", "dim": "模型宽度"},
                     "dtype": "bfloat16",
                     "kind": "source_derived"},
        "evidence": [{"file": "model_dir/model.py", "lines": [31, 33],
                       "symbol": "Block.forward",
                       "conclusion": "proj 输入为归一化后的隐藏态，输出维度=dim",
                       "kind": "source_derived"}],
        "scenarios": ["prefill", "decode"],
        "branch_note": None,
        "concerns": [],
        "status": "verified",
        "open_questions": [],
    }],
}


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    if argv[0] == "--example":
        print(json.dumps(EXAMPLE, ensure_ascii=False, indent=2))
        return 0
    if argv[0] == "--template":
        if len(argv) < 2:
            print("用法：ir_schema.py --template <输出路径>")
            return 1
        Path(argv[1]).write_text(
            json.dumps(EXAMPLE, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"模板已写入：{argv[1]}（填好后用 ir_schema.py <文件> 校验）")
        return 0
    p = Path(argv[0])
    if not p.exists():
        print(f"文件不存在：{p}")
        return 1
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败：{e}")
        return 1
    errors = validate(data)
    if errors:
        print(f"校验失败，共 {len(errors)} 处：")
        for i, msg in enumerate(errors, 1):
            print(f"  {i}. {msg}")
        return 1
    n_nodes = len(data.get("nodes", []))
    n_rep = len(data.get("contract", {}).get("replacements", []))
    print(f"校验通过：{n_nodes} 个节点，{n_rep} 项 kernel 替换，"
          f"{len(data.get('model', {}).get('source_files_hash', {}))} 个源文件已登记哈希")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
