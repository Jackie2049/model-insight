#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kit 全链机制测试：合成小模型（含假外部 kernel）走通
audit → 逻辑 IR 校验 → spec/驱动抓图 → 融合 → HTML 注入。

只验证 kit 的机制（工具链路），不碰任何真实模型。
用法：在任意目录执行 `python agent-kit_ops-analysis/tests/run_tests.py [--python <解释器路径>]`
解释器定位顺序：--python 参数 > 环境变量 OPS_KIT_PYTHON > kit 上层逐级探测 .venv/bin/python > 当前解释器

断言要点：
  1. audit 识别缺依赖 nonexistent_kit、调用点、入口类形态
  2. 夹具 analysis_ir（模拟 2a 产物）过 schema 校验
  3. driver 三作业（macro×2 场景 + micro 代表层）全部 ok；
     shim 挂树且被真实调用（macro 图中出现 MysteryActShim 模块节点）；
     毛坯目录零 __pycache__（只读纪律生效）
  4. fuse 四条状态通道全命中：verified / mismatch（fc1 故意错维度）/
     llm_only（ghost 故意不存在）/ capture_only（shim 未进逻辑 IR）
  5. final_ir 过 schema 校验
  6. render 注入成功 + 幂等（二次运行不重复注入）
"""
import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent          # .../agent-kit_ops-analysis/tests
KIT = TESTS.parent                                # .../agent-kit_ops-analysis
ROOT = KIT.parent                                 # 工作区根（kit 的上一级目录）
RUN = TESTS / "run"                               # 测试产物目录（每次清空）
FIX = TESTS / "fixture"

FAILURES = []


def step(name):
    print(f"\n=== {name} ===")


def check(cond, msg):
    """断言：失败记录、继续跑（一次看全所有问题）。"""
    tag = "✓" if cond else "✗"
    print(f"  {tag} {msg}")
    if not cond:
        FAILURES.append(msg)
    return cond


def run(cmd, **kw):
    """跑子进程（默认用工作区 venv 的 python 语义由调用方给定）。"""
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(f"  [子进程失败 {r.returncode}] {' '.join(map(str, cmd))}")
        print("  --- stdout ---"); print(r.stdout[-2000:])
        print("  --- stderr ---"); print(r.stderr[-2000:])
    return r


def resolve_python(explicit=None):
    """定位子进程用的 python 解释器。

    顺序：--python 参数 > 环境变量 OPS_KIT_PYTHON > kit 上层逐级探测 .venv/bin/python
    > 回退当前解释器（回退前预检 torch/torchvista 可导入，缺失则指向 requirements.txt）。
    """
    if explicit:
        # 注意：不得 resolve()——.venv/bin/python 常是指向 base python 的 symlink，
        # 解析后 venv 语义（site-packages 隔离）即被破坏
        p = Path(explicit).expanduser().absolute()
        if not p.exists():
            print(f"✗ --python 指定的解释器不存在：{p}")
            sys.exit(1)
        return p
    env = os.environ.get("OPS_KIT_PYTHON")
    if env:
        p = Path(env).expanduser().absolute()
        if p.exists():
            print(f"· 使用环境变量 OPS_KIT_PYTHON 指定的解释器：{p}")
            return p
        print(f"⚠ OPS_KIT_PYTHON 指向的解释器不存在（{p}），改用逐级探测")
    d = KIT
    for _ in range(4):                       # kit → 工作区根 → 再向上两级
        cand = d / ".venv" / "bin" / "python"
        if cand.exists():
            return cand
        if d.parent == d:
            break
        d = d.parent
    print(f"⚠ 未找到 .venv/bin/python，回退当前解释器：{sys.executable}")
    missing = []
    for mod in ("torch", "torchvista"):
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        print(f"✗ 当前解释器缺少依赖：{', '.join(missing)}")
        print("  请先创建 venv 并安装：pip install -r requirements.txt（版本锁说明见 README）")
        print("  或用 --python <路径> / 环境变量 OPS_KIT_PYTHON 指定已装好的解释器")
        sys.exit(1)
    return Path(sys.executable)


def main():
    ap = argparse.ArgumentParser(description="kit 全链机制测试（合成夹具，不碰真实模型）")
    ap.add_argument("--python", default=None, metavar="<解释器路径>",
                    help="子进程使用的 python 解释器；缺省按 resolve_python 顺序探测")
    args = ap.parse_args()
    if RUN.exists():
        shutil.rmtree(RUN)
    (RUN / "capture").mkdir(parents=True)
    (RUN / "llm-analysis").mkdir(parents=True)

    py = str(resolve_python(args.python))
    scripts = KIT / "scripts"

    # ---------- 1. S0 静态审计 ----------
    step("1/6 S0 静态审计")
    r = run([py, str(scripts / "audit_model.py"),
             "--model", str(FIX / "model_pkg"), "--out", str(RUN / "audit")])
    if not check(r.returncode == 0, "audit_model.py 退出码 0"):
        return finish()
    audit = json.loads((RUN / "audit" / "audit_report.json").read_text(encoding="utf-8"))
    missing = {m["module"] for m in audit["missing_deps"]}
    check("nonexistent_kit" in missing, f"缺依赖识别出 nonexistent_kit（实际：{sorted(missing)}）")
    check(bool(audit["external_calls"]), "外部 kernel 调用点非空")
    tm = [c for c in audit["classes"] if c["name"] == "TinyModel"]
    check(bool(tm) and tm[0]["nn_module_like"] and tm[0]["has_forward"],
          "TinyModel 识别为疑似 nn.Module 且带 forward")
    check("model.py" in audit["files"], "文件哈希清单含 model.py")

    # ---------- 2. 夹具逻辑 IR（模拟 2a 产物） ----------
    step("2/6 夹具 analysis_ir 组装与校验")
    sha = hashlib.sha256((FIX / "model_pkg" / "model.py").read_bytes()).hexdigest()
    ir_fixture = json.loads((FIX / "analysis_ir.json").read_text(encoding="utf-8"))
    ir_fixture["model"]["source_files_hash"]["model.py"] = sha
    analysis_path = RUN / "llm-analysis" / "analysis_ir.json"
    analysis_path.write_text(json.dumps(ir_fixture, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    r = run([py, str(scripts / "ir_schema.py"), str(analysis_path)])
    check(r.returncode == 0, "analysis_ir 过 ir_schema 校验")

    # ---------- 3. spec + driver 抓图 ----------
    step("3/6 spec 组装 + capture_driver 三作业")
    spec = {
        "spec_version": "1.0",
        "model": {
            "name": "TinyModel（测试夹具）",
            "dir": "fixture/model_pkg",
            "entry": {
                # 列表前面的路径优先级更高：shim 目录遮蔽毛坯内同名模块
                "sys_path": ["fixture/shims", "fixture/model_pkg"],
                "module": "model", "class": "TinyModel",
                "constructor_args": {"dim": 16, "n_layers": 3, "vocab": 64},
            },
            "default_dtype": "bfloat16", "seed": 0,
            "eval_mode": True, "init_distributed": False,
        },
        "shims": [{
            "module": "fixture/shims/nonexistent_kit.py",
            "class": "MysteryActShim", "mount_name": "_shim_act",
            "rebind": [{"module": "model", "attr": "mystery_act"}],
        }],
        "traces": [
            {"id": "macro_prefill", "kind": "macro", "scenario": "prefill",
             "inputs": {"args": [{"type": "int_tensor", "shape": [2, 8],
                                  "low": 0, "high": 64}]}},
            {"id": "macro_decode", "kind": "macro", "scenario": "decode",
             "inputs": {"args": [{"type": "int_tensor", "shape": [2, 1],
                                  "low": 0, "high": 64}]}},
            {"id": "micro_layer1", "kind": "micro", "target": "layers.1",
             "scenario": "prefill",
             "inputs": {"args": [{"type": "float_tensor", "shape": [2, 6, 16]}]}},
        ],
    }
    spec_path = RUN / "capture" / "spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    r = run([py, str(scripts / "capture_driver.py"),
             "--workspace", str(TESTS), "--spec", str(spec_path)],
            cwd=str(TESTS))
    if not check(r.returncode == 0, "capture_driver 退出码 0"):
        return finish()
    summary = json.loads((RUN / "capture" / "capture_summary.json").read_text(encoding="utf-8"))
    check(summary.get("all_ok") is True, "三作业全部 ok")
    for jid in ("macro_prefill", "macro_decode", "micro_layer1"):
        check((RUN / "capture" / f"{jid}.html").exists()
              and (RUN / "capture" / f"{jid}.measured.json").exists(),
              f"{jid} 产出 HTML + measured.json")
    mp = json.loads((RUN / "capture" / "macro_prefill.measured.json").read_text(encoding="utf-8"))
    shim_nodes = [n for n in mp["derived"]["nodes"]
                  if n["module_type"] == "MysteryActShim"]
    check(bool(shim_nodes), "shim 以模块节点出现在 macro 图（挂树+重绑生效）")
    if shim_nodes:
        check(any("_shim_act" in n["tree_path"] for n in shim_nodes),
              "shim 节点树路径含挂载名 _shim_act")
        with_input = [n for n in shim_nodes if n["input_tensors"]]
        check(bool(with_input)
              and with_input[0]["input_tensors"][0]["shape"] == [2, 8, 32]
              and with_input[0]["input_tensors"][0]["dtype"] == "torch.bfloat16",
              "shim 节点 func_info 记录了输入 shape/dtype（(2,8,32) bf16）")
    check(not (FIX / "model_pkg" / "__pycache__").exists(),
          "毛坯目录无 __pycache__（只读纪律 + dont_write_bytecode 生效）")

    # ---------- 4. 融合 ----------
    step("4/6 fuse 机械融合（四条状态通道）")
    r = run([py, str(scripts / "fuse.py"),
             "--analysis", str(analysis_path),
             "--capture-dir", str(RUN / "capture"),
             "--out", str(RUN / "fusion")])
    check(r.returncode == 0, "fuse 退出码 0")
    final = json.loads((RUN / "fusion" / "final_ir.json").read_text(encoding="utf-8"))
    st = {}
    for nd in final["nodes"]:
        st.setdefault(nd["status"], []).append(nd["path"])
    check("verified" in st, f"verified 通道命中（{len(st.get('verified', []))} 个）")
    check(any(p.endswith("fc1") for p in st.get("mismatch", [])),
          f"mismatch 通道命中 fc1（实际：{st.get('mismatch')}）")
    check(any("ghost" in p for p in st.get("llm_only", [])),
          f"llm_only 通道命中 ghost（实际：{st.get('llm_only')}）")
    check(any("_shim_act" in p for p in st.get("capture_only", [])),
          f"capture_only 通道命中 _shim_act（实际：{st.get('capture_only')}）")
    # 双场景并列：norm 节点应有 prefill 与 decode 两个实测块
    norm = [nd for nd in final["nodes"] if nd["path"] == "{i}.norm"][0]
    scen = {b["scenario"] for b in norm["measured"]["blocks"]}
    check(scen == {"prefill", "decode"}, f"norm 节点双场景实测并列（实际：{scen}）")
    diff = (RUN / "fusion" / "diff_report.md").read_text(encoding="utf-8")
    check("场景符号绑定表" in diff and "S=8" in diff, "diff_report 含符号绑定表（S=8）")

    # ---------- 5. 定稿校验 ----------
    step("5/6 final_ir 过 schema 校验")
    r = run([py, str(scripts / "ir_schema.py"), str(RUN / "fusion" / "final_ir.json")])
    check(r.returncode == 0, "final_ir 过 ir_schema 校验")

    # ---------- 6. HTML 注入 ----------
    step("6/6 render_fusion 注入（含幂等）")
    r = run([py, str(scripts / "render_fusion.py"),
             "--html", str(RUN / "capture"),
             "--ir", str(RUN / "fusion" / "final_ir.json"),
             "--spec", str(spec_path)])
    check(r.returncode == 0, "render_fusion 退出码 0")
    mark = "ops-kit-fusion-injection"
    htmls = sorted((RUN / "capture").glob("*.html"))
    check(len(htmls) == 3, f"3 个 HTML 被处理（实际 {len(htmls)}）")
    for h in htmls:
        text = h.read_text(encoding="utf-8")
        check(text.count(mark) >= 1, f"{h.name} 含注入标记")
        check("算子分析融合面板" in text, f"{h.name} 含融合面板")
    r2 = run([py, str(scripts / "render_fusion.py"),
              "--html", str(RUN / "capture"),
              "--ir", str(RUN / "fusion" / "final_ir.json"),
              "--spec", str(spec_path)])
    check(r2.returncode == 0, "二次运行退出码 0（幂等）")
    for h in htmls:
        # 注释锚点唯一（面板 id 与脚本里也会出现标记字符串，不能直接 count(mark)）
        check(h.read_text(encoding="utf-8").count(f"<!-- {mark} -->") == 1,
              f"{h.name} 幂等（注入注释锚点恰好 1 处）")

    return finish()


def finish():
    print("\n" + "=" * 46)
    if FAILURES:
        print(f"✗ 测试失败：{len(FAILURES)} 处未通过")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("✓ 全部测试通过（audit / driver / fuse / schema / render 全链机制验证）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
