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
  7. render_index 集成面板：viz/index.html + views.json 生成、五态与 IR 一致、
     链接页面在盘、views.json 与内联 VIEWS 同源、二次运行字节一致（幂等）
  8. 作业分组并行：--jobs 2 与串行产物一致（inputs_sha256 逐作业相等）；
     rebuild:false 链保序（组内串行）；链内续流 / 组首组局部语义可证明；
     spawn 子进程零 __pycache__；--job 修复为链前缀语义
"""
import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
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
    step("1/8 S0 静态审计")
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
    step("2/8 夹具 analysis_ir 组装与校验")
    sha = hashlib.sha256((FIX / "model_pkg" / "model.py").read_bytes()).hexdigest()
    ir_fixture = json.loads((FIX / "analysis_ir.json").read_text(encoding="utf-8"))
    ir_fixture["model"]["source_files_hash"]["model.py"] = sha
    analysis_path = RUN / "llm-analysis" / "analysis_ir.json"
    analysis_path.write_text(json.dumps(ir_fixture, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    r = run([py, str(scripts / "ir_schema.py"), str(analysis_path)])
    check(r.returncode == 0, "analysis_ir 过 ir_schema 校验")

    # ---------- 3. spec + driver 抓图 ----------
    step("3/8 spec 组装 + capture_driver 三作业")
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

    # ---------- 4. 作业分组并行 ----------
    step("4/8 作业分组并行：--jobs 2 vs 串行（链保序 + 输入指纹一致）")
    spec_p = copy.deepcopy(spec)
    # chain_a 与 chain_b、solo_c 用同一配方：独立重建时三者输入必然相同（组局部语义），
    # 链内顺延 RNG 流的 chain_b 则必然不同——指纹差异恰好可证续流语义
    spec_p["traces"] = [
        {"id": "chain_a", "kind": "macro", "scenario": "prefill",
         "inputs": {"args": [{"type": "int_tensor", "shape": [2, 8], "low": 0, "high": 64}]}},
        {"id": "chain_b", "kind": "macro", "scenario": "decode", "rebuild": False,
         "inputs": {"args": [{"type": "int_tensor", "shape": [2, 8], "low": 0, "high": 64}]}},
        {"id": "solo_c", "kind": "macro", "scenario": "prefill",
         "inputs": {"args": [{"type": "int_tensor", "shape": [2, 8], "low": 0, "high": 64}]}},
        {"id": "chain_d", "kind": "micro", "target": "layers.1", "scenario": "prefill",
         "rebuild": False,
         "inputs": {"args": [{"type": "float_tensor", "shape": [2, 6, 16]}]}},
        {"id": "bad_target", "kind": "micro", "target": "layers.99", "scenario": "prefill",
         "inputs": {"args": [{"type": "float_tensor", "shape": [2, 6, 16]}]}},
    ]
    pdir = RUN / "capture-parallel"
    pdir.mkdir()
    specp_path = pdir / "spec.json"
    specp_path.write_text(json.dumps(spec_p, ensure_ascii=False, indent=2), encoding="utf-8")
    r = run([py, str(scripts / "capture_driver.py"),
             "--workspace", str(TESTS), "--spec", str(specp_path), "--jobs", "2"],
            cwd=str(TESTS))
    check(r.returncode == 1, "并行轮退出码 1（bad_target 必败，故障不外溢）")
    if not (pdir / "capture_summary.json").exists():
        check(False, "并行轮未产出 capture_summary.json（driver 异常退出）")
        return finish()
    psummary = json.loads((pdir / "capture_summary.json").read_text(encoding="utf-8"))
    order = [j["id"] for j in psummary.get("jobs", [])]
    check(order == [j["id"] for j in spec_p["traces"]],
          f"T1 summary 作业顺序 == traces 顺序（实际 {order}）")
    st = {j["id"]: j["status"] for j in psummary.get("jobs", [])}
    check(all(st.get(x) == "ok" for x in ("chain_a", "chain_b", "solo_c", "chain_d")),
          f"T1 四个好作业全 ok（故障隔离，实际 {st}）")
    bad = next((j for j in psummary.get("jobs", []) if j["id"] == "bad_target"), {})
    check(st.get("bad_target") == "failed"
          and "target" in (bad.get("error") or bad.get("exception") or ""),
          "T1 bad_target 记 failed 且中文错误指向 target 解析失败")

    fp = {}
    for jid in ("chain_a", "chain_b", "solo_c", "chain_d"):
        mj = json.loads((pdir / f"{jid}.measured.json").read_text(encoding="utf-8"))
        fp[jid] = mj["run"]["inputs_sha256"]
    sdir = RUN / "capture-serial"
    sdir.mkdir()
    specs_path = sdir / "spec.json"
    specs_path.write_text(json.dumps(spec_p, ensure_ascii=False, indent=2), encoding="utf-8")
    r = run([py, str(scripts / "capture_driver.py"),
             "--workspace", str(TESTS), "--spec", str(specs_path)],
            cwd=str(TESTS))
    check(r.returncode == 1, "串行轮（同 spec）退出码 1")
    fps = {}
    for jid in ("chain_a", "chain_b", "solo_c", "chain_d"):
        mj = json.loads((sdir / f"{jid}.measured.json").read_text(encoding="utf-8"))
        fps[jid] = mj["run"]["inputs_sha256"]
    check(fp == fps,
          "T2 --jobs 2 与串行的四作业 inputs_sha256 逐一相等（并行零漂移）")
    check(fp["chain_b"] != fp["chain_a"],
          "T3 链内作业指纹 ≠ 组首（rebuild:false 续流语义成立）")
    check(fp["solo_c"] == fp["chain_a"],
          "T5 同种子同配方的独立组首指纹相等（组局部语义回归锚）")

    # T4：只含 chain_b 的 spec（traces[0] 即 rebuild:false → 无前驱、独立重建态）
    spec_b = copy.deepcopy(spec_p)
    spec_b["traces"] = [j for j in spec_p["traces"] if j["id"] == "chain_b"]
    bdir = RUN / "capture-chainonly"
    bdir.mkdir()
    (bdir / "spec.json").write_text(json.dumps(spec_b, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    r = run([py, str(scripts / "capture_driver.py"),
             "--workspace", str(TESTS), "--spec", str(bdir / "spec.json")],
            cwd=str(TESTS))
    check(r.returncode == 0, "T4 单链作业 spec（无前驱）退出码 0")
    mjb = json.loads((bdir / "chain_b.measured.json").read_text(encoding="utf-8"))
    check(mjb["run"]["inputs_sha256"] == fp["chain_a"]
          and mjb["run"]["inputs_sha256"] != fp["chain_b"],
          "T4 无前驱的 chain_b 与组首同输入（独立重建态），≠ 链内续流值（链状态依赖真实存在）")

    check(not (FIX / "model_pkg" / "__pycache__").exists()
          and not (FIX / "shims" / "__pycache__").exists(),
          "T6 并行轮后毛坯与 shims 目录零 __pycache__（spawn 子进程不写 .pyc）")

    r = run([py, str(scripts / "capture_driver.py"),
             "--workspace", str(TESTS), "--spec", str(specs_path), "--job", "chain_b"],
            cwd=str(TESTS))
    check(r.returncode == 0, "T7 --job chain_b 退出码 0（组首前缀连带执行）")
    ssum2 = json.loads((sdir / "capture_summary.json").read_text(encoding="utf-8"))
    ids2 = [j["id"] for j in ssum2.get("jobs", [])]
    check(ids2 == ["chain_a", "chain_b"], f"T7 --job 执行组首前缀（实际 {ids2}）")
    mjb2 = json.loads((sdir / "chain_b.measured.json").read_text(encoding="utf-8"))
    check(mjb2["run"]["inputs_sha256"] == fp["chain_b"],
          "T7 --job 前缀执行下 chain_b 指纹与全量轮相等（链状态保真）")

    # ---------- 5. 融合 ----------
    step("5/8 fuse 机械融合（四条状态通道）")
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
    step("6/8 final_ir 过 schema 校验")
    r = run([py, str(scripts / "ir_schema.py"), str(RUN / "fusion" / "final_ir.json")])
    check(r.returncode == 0, "final_ir 过 ir_schema 校验")

    # ---------- 6. HTML 注入 ----------
    step("7/8 render_fusion 注入（含幂等）")
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

    # ---------- 7. 集成面板 ----------
    step("8/8 render_index 集成面板（含幂等）")
    viz = RUN / "viz"
    idx = viz / "index.html"
    render_index_cmd = [py, str(scripts / "render_index.py"),
                        "--capture-dir", str(RUN / "capture"),
                        "--ir", str(RUN / "fusion" / "final_ir.json"),
                        "--out", str(viz)]
    r = run(render_index_cmd)
    check(r.returncode == 0, "render_index 退出码 0")
    check(idx.is_file(), "viz/index.html 生成")
    check((viz / "views.json").is_file(), "viz/views.json 生成")
    idx_text = idx.read_text(encoding="utf-8")
    if idx.is_file():
        for token in ("__VIEWS__", "__DETAILS__", "__SUMMARY__", "__MODEL__"):
            check(token not in idx_text, f"无占位符残留（{token}）")
        m = re.search(r"const VIEWS=(\[.*?\]);\nconst DETAILS=", idx_text, re.S)
        check(m is not None, "index.html 含 VIEWS 数组")
        if m:
            views = json.loads(m.group(1))
            n_jobs = len(json.loads((RUN / "capture" / "capture_summary.json")
                                    .read_text(encoding="utf-8")).get("jobs", []))
            check(len(views) == n_jobs, f"VIEWS 条数 == capture 作业数 {n_jobs}（实际 {len(views)}）")
            check(all((RUN / "capture" / v["file"]).is_file() for v in views),
                  "面板引用的页面全部在盘")
        vj_path = viz / "views.json"
        if vj_path.is_file() and m:
            vj = json.loads(vj_path.read_text(encoding="utf-8"))
            check([v["id"] for v in vj.get("views", [])] == [v["id"] for v in json.loads(m.group(1))],
                  "views.json 与内联 VIEWS 同源一致")
        ms = re.search(r"const SUMMARY=(\{.*?\});\nconst STATUS_META", idx_text, re.S)
        check(ms is not None, "index.html 含 SUMMARY 统计")
        if ms:
            statuses = json.loads(ms.group(1))["statuses"]
            nodes = json.loads((RUN / "fusion" / "final_ir.json")
                               .read_text(encoding="utf-8"))["nodes"]
            counter = Counter(n["status"] for n in nodes)
            expect = {k: counter.get(k, 0)
                      for k in ("verified", "mismatch", "llm_only", "capture_only", "conflict")}
            check(statuses == expect, f"五态统计与 IR 一致（实际 {statuses}）")
    h1 = hashlib.md5(idx.read_bytes()).hexdigest() if idx.is_file() else ""
    r2 = run(render_index_cmd)
    h2 = hashlib.md5(idx.read_bytes()).hexdigest() if idx.is_file() else ""
    check(r2.returncode == 0 and h1 != "" and h1 == h2, "二次运行字节一致（幂等）")

    return finish()


def finish():
    print("\n" + "=" * 46)
    if FAILURES:
        print(f"✗ 测试失败：{len(FAILURES)} 处未通过")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("✓ 全部测试通过（audit / driver / 分组并行 / fuse / schema / render / index 全链机制验证）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
