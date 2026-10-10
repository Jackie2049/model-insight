#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4b HTML 后处理：把"等价替换标记 + 融合状态徽标"注入 TorchVista 导出的 HTML。

为什么用"文件末尾追加"而不是找 </body>：TorchVista 模板导出的 HTML 是无
<html>/<body> 标签的文档片段，不存在 </body> 锚点（历史教训：以 </body> 为
锚点的注入会静默无操作）。本工具的可靠性纪律：
  1) 幂等：目标 HTML 已含注入标记则跳过（不重复注入）；
  2) 写后校验：追加完成后回读文件断言标记确实存在，失败立即报错——绝不静默。

注入内容（一段自包含的 <style> + <div> + <script>，追加在文件末尾）：
  - shim 红虚线标记：图上属"等价替换件"的节点描红虚线（信息来自 spec.shims
    的挂载名/类名 + IR replacements 的说明文字）
  - 融合状态徽标：verified 绿描边 / mismatch·conflict 橙描边（按显示名聚合
    ——模板 IR 的同一属性名标记到全部实例，与"层折叠但差异保留"的口径一致）
  - 右上角融合面板：模型与缩水声明、状态计数、替换清单、不一致/未测清单
  - MutationObserver：图形交互重绘后自动重新打标（折叠/展开会重建节点）

用法：
  python render_fusion.py --html <html文件或目录> --ir <final_ir.json> [--spec <spec.json>]

HTML 目录模式下逐个处理（自动配对同目录 <作业id>.measured.json 取场景）。
退出码：0=全部处理成功；1=任一失败（含注入校验失败——必须修好再交付）。
"""
import argparse
import json
import sys
from pathlib import Path

# 不写字节码：毛坯目录只读，绝不能留下 __pycache__
sys.dont_write_bytecode = True

INJECTION_MARK = "ops-kit-fusion-injection"

# 注入的页面脚本（CFG 由 Python 侧填充为 JSON）。要点：
#  - 页面级 const（adj_list / graph_node_display_names 等）在同一文档的后续
#    <script> 里可直接引用——但本脚本不依赖它们，只用 DOM 文本匹配，兼容性最好
#  - 节点组的基础 class 形如 "node_<uid>"，修饰 class（module_/operation_…）
#    不以 node_ 开头，故按"存在 node_ 前缀 class"过滤节点组
_JS = r"""
(function () {
  var CFG = __OPS_KIT_CFG__;
  function classes(el) { return Array.prototype.slice.call(el.classList || []); }
  function isNodeGroup(el) {
    return classes(el).some(function (c) { return c.indexOf("node_") === 0; });
  }
  function boundaryRx(name) {
    // 显示名按"词边界"匹配：避免短名误配长名（如 attn 误配 attn_norm）
    var esc = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return new RegExp("(^|[^A-Za-z0-9_])" + esc + "([^A-Za-z0-9_]|$)");
  }
  var shimRx = CFG.shimNames.map(boundaryRx);
  var statusEntries = [];
  for (var nm in CFG.statusByDisplay) statusEntries.push([boundaryRx(nm), CFG.statusByDisplay[nm]]);
  function mark() {
    var groups = document.querySelectorAll("g");
    for (var i = 0; i < groups.length; i++) {
      var g = groups[i];
      if (!isNodeGroup(g)) continue;
      if (g.classList.contains("ops-kit-marked")) continue;
      g.classList.add("ops-kit-marked");
      var text = "\n" + (g.textContent || "").trim() + "\n";
      var hitShim = null, hitStatus = null;
      for (var s = 0; s < shimRx.length; s++) {
        if (shimRx[s].test(text)) { hitShim = CFG.shimNames[s]; break; }
      }
      for (var e = 0; e < statusEntries.length; e++) {
        if (statusEntries[e][0].test(text)) {
          var st = statusEntries[e][1];
          if (!hitStatus || CFG.severity[st] > CFG.severity[hitStatus]) hitStatus = st;
        }
      }
      if (hitShim) {
        g.classList.add("ops-kit-shim");
        var t = document.createElementNS("http://www.w3.org/2000/svg", "title");
        t.textContent = "等价替换：" + (CFG.shimNotes[hitShim] || hitShim);
        g.appendChild(t);
      } else if (hitStatus) {
        g.classList.add("ops-kit-" + hitStatus);
      }
    }
  }
  var pending = null;
  function schedule() {
    if (pending) clearTimeout(pending);
    pending = setTimeout(mark, 150);
  }
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", schedule);
  else schedule();
  var host = document.querySelector('[id^="graph-container_"]') || document.body;
  if (host && window.MutationObserver) {
    new MutationObserver(schedule).observe(host, { childList: true, subtree: true });
  }
  var panel = document.getElementById("ops-kit-fusion-injection");
  if (panel) {
    var head = panel.querySelector(".ops-kit-head");
    if (head) head.addEventListener("click", function () {
      panel.classList.toggle("ops-kit-collapsed");
    });
  }
})();
"""

_CSS = """
/* —— 算子分析 Agent 融合注入（render_fusion.py）—— */
g.ops-kit-shim ellipse, g.ops-kit-shim rect, g.ops-kit-shim path {
  stroke: #e03131 !important; stroke-dasharray: 7 5 !important; stroke-width: 3px !important;
}
g.ops-kit-verified ellipse, g.ops-kit-verified rect, g.ops-kit-verified path {
  stroke: #2f9e44 !important; stroke-width: 2.5px !important;
}
g.ops-kit-mismatch ellipse, g.ops-kit-mismatch rect, g.ops-kit-mismatch path,
g.ops-kit-conflict ellipse, g.ops-kit-conflict rect, g.ops-kit-conflict path {
  stroke: #e8590c !important; stroke-width: 3px !important;
}
#ops-kit-fusion-injection {
  position: fixed; top: 12px; right: 12px; z-index: 9999;
  max-width: 420px; max-height: 80vh; overflow: auto;
  background: rgba(255,255,255,.96); color: #212529;
  border: 1px solid #ced4da; border-radius: 8px;
  font: 12px/1.5 -apple-system, "PingFang SC", "Segoe UI", sans-serif;
  box-shadow: 0 2px 12px rgba(0,0,0,.18); padding: 10px 12px;
}
#ops-kit-fusion-injection .ops-kit-head {
  cursor: pointer; font-weight: 700; font-size: 13px; margin-bottom: 6px;
  user-select: none;
}
#ops-kit-fusion-injection.ops-kit-collapsed .ops-kit-body { display: none; }
#ops-kit-fusion-injection .ops-kit-row { margin: 2px 0; }
#ops-kit-fusion-injection .ops-kit-legend { margin: 6px 0; }
#ops-kit-fusion-injection .ops-kit-dot {
  display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 6px; vertical-align: -1px;
}
#ops-kit-fusion-injection .ops-kit-mono { font-family: ui-monospace, Menlo, monospace; }
#ops-kit-fusion-injection .ops-kit-bad { color: #c92a2a; }
#ops-kit-fusion-injection .ops-kit-warn { color: #e8590c; }
"""


def _esc(s):
    """把文本安全放进 HTML。"""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def build_panel_html(cfg):
    """右上角融合面板（服务端拼好，页面脚本只负责折叠与打标）。"""
    counts = cfg["counts"]
    rows = [
        f'<div class="ops-kit-head">算子分析融合面板（点击折叠）</div>',
        f'<div class="ops-kit-body">',
        f'<div class="ops-kit-row"><b>{_esc(cfg["model_name"])}</b>'
        + (f' · 场景 {_esc(cfg["scenario"])}' if cfg.get("scenario") else "") + "</div>",
        f'<div class="ops-kit-row ops-kit-mono" style="color:#495057">'
        f'{_esc(cfg["scale_note"][:180])}</div>',
        '<div class="ops-kit-legend">'
        '<span class="ops-kit-dot" style="background:#e03131"></span>等价替换（红虚线）　'
        '<span class="ops-kit-dot" style="background:#2f9e44"></span>verified　'
        '<span class="ops-kit-dot" style="background:#e8590c"></span>mismatch/conflict'
        "</div>",
        "<div>节点状态："
        f"verified {counts.get('verified', 0)} · mismatch {counts.get('mismatch', 0)} · "
        f"llm_only {counts.get('llm_only', 0)} · capture_only {counts.get('capture_only', 0)} · "
        f"conflict {counts.get('conflict', 0)}</div>",
    ]
    if cfg["replacements"]:
        rows.append('<div style="margin-top:6px"><b>等价替换清单</b>'
                    f'（{len(cfg["replacements"])} 项，图上为红虚线）：</div>')
        for r in cfg["replacements"]:
            rows.append(f'<div class="ops-kit-row">· {_esc(r)}</div>')
    bad = cfg["mismatches"][:12]
    if bad:
        rows.append('<div style="margin-top:6px"><b>待复核节点</b>'
                    f'（详见 diff_report.md）：</div>')
        for b in bad:
            rows.append(f'<div class="ops-kit-row ops-kit-bad ops-kit-mono">{_esc(b)}</div>')
    note = cfg.get("llm_only_note")
    if note:
        rows.append(f'<div class="ops-kit-row ops-kit-warn" style="margin-top:6px">{_esc(note)}</div>')
    rows.append("</div>")
    return "".join(rows)


def build_config(html_path, ir, spec):
    """为单个 HTML 组装注入配置（shim 名单、状态-显示名映射、面板数据）。"""
    # 该 HTML 对应的作业与场景：文件名 = 作业 id，同目录找 measured.json
    job_id = html_path.stem
    measured_json = html_path.with_name(f"{job_id}.measured.json")
    scenario, display_names = None, {}
    if measured_json.exists():
        mj = json.loads(measured_json.read_text(encoding="utf-8"))
        scenario = mj["job"]["scenario"]
        display_names = mj["graph"]["graph_node_display_names"]

    # ---- shim 名单（spec.shims 的挂载名 + 类名；说明取 IR replacements）----
    shim_names, shim_notes = [], {}
    rep_by_class = {}
    for r in ir.get("contract", {}).get("replacements", []):
        if r.get("shim_class"):
            rep_by_class[r["shim_class"]] = f"{r.get('what', '')}：{r.get('equiv', '')[:80]}"
    if spec:
        for sh in spec.get("shims", []):
            note = rep_by_class.get(sh["class"], sh.get("class", ""))
            shim_names.append(sh["mount_name"])
            shim_notes[sh["mount_name"]] = note
            if sh["class"] not in shim_names:
                shim_names.append(sh["class"])
                shim_notes[sh["class"]] = note

    # ---- 状态 → 显示名（只统计本场景有实测记录的节点；按显示名聚合）----
    severity = {"verified": 0, "mismatch": 1, "conflict": 2}
    status_by_display = {}
    counts = {}
    mismatches, llm_only_paths = [], []
    for nd in ir.get("nodes", []):
        st = nd.get("status")
        if st:
            counts[st] = counts.get(st, 0) + 1
        if st in ("mismatch", "conflict"):
            mismatches.append(f"{nd.get('path')} [{st}]")
        if st == "llm_only":
            llm_only_paths.append(nd.get("path"))
        for b in (nd.get("measured") or {}).get("blocks", []):
            if scenario is not None and b.get("scenario") != scenario:
                continue
            disp = display_names.get(b.get("capture_node")) or b.get("capture_node")
            if not disp:
                continue
            if st in severity:
                cur = status_by_display.get(disp)
                if cur is None or severity[st] > severity[cur]:
                    status_by_display[disp] = st
    llm_note = None
    if llm_only_paths:
        llm_note = (f"另有 {len(llm_only_paths)} 个节点仅逻辑分析、尚未运行验证"
                    "（未执行 ≠ 不存在），清单见 IR。")
    cfg = {
        "shimNames": shim_names, "shimNotes": shim_notes,
        "statusByDisplay": status_by_display, "severity": severity,
        "counts": counts, "mismatches": mismatches,
        "model_name": ir.get("model", {}).get("name", ""),
        "scale_note": ir.get("model", {}).get("scale_note", ""),
        "replacements": [f"{r.get('what', '')}（{r.get('equiv', '')[:60]}）"
                         for r in ir.get("contract", {}).get("replacements", [])],
        "scenario": scenario, "llm_only_note": llm_note,
    }
    return cfg


def inject(html_path, ir, spec):
    """对单个 HTML 执行注入：幂等检查 → 追加 → 回读断言。"""
    text = html_path.read_text(encoding="utf-8")
    if INJECTION_MARK in text:
        print(f"  · {html_path.name}：已含注入标记，跳过（幂等）")
        return True
    cfg = build_config(html_path, ir, spec)
    js = _JS.replace("__OPS_KIT_CFG__", json.dumps(cfg, ensure_ascii=False))
    panel = build_panel_html(cfg)
    block = (f"\n<!-- {INJECTION_MARK} -->\n<style>{_CSS}</style>\n"
             f'<div id="{INJECTION_MARK}">{panel}</div>\n'
             f"<script>{js}</script>\n")
    html_path.write_text(text + block, encoding="utf-8")

    # 写后校验（可靠性纪律：杜绝"以为注入了其实没生效"的静默无操作）
    after = html_path.read_text(encoding="utf-8")
    if INJECTION_MARK not in after:
        print(f"  ✗ {html_path.name}：注入后回读未发现标记（写入异常，交付禁止）")
        return False
    print(f"  · {html_path.name}：已注入（shim 标记 {len(cfg['shimNames'])} 个目标，"
          f"状态徽标 {len(cfg['statusByDisplay'])} 个显示名，场景 {cfg['scenario']}）")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="S4b 融合信息注入 TorchVista HTML")
    ap.add_argument("--html", required=True, help="单个 HTML 或目录")
    ap.add_argument("--ir", required=True, help="final_ir.json（融合定稿）")
    ap.add_argument("--spec", default=None, help="spec.json（提供 shim 挂载名；可选）")
    args = ap.parse_args(argv)

    ir = json.loads(Path(args.ir).read_text(encoding="utf-8"))
    spec = None
    if args.spec:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    target = Path(args.html)
    htmls = ([target] if target.is_file()
             else sorted(target.glob("*.html")) if target.is_dir() else [])
    if not htmls:
        print(f"✗ 没有可处理的 HTML：{target}")
        return 1
    print(f"◆ 注入开始：{len(htmls)} 个 HTML，IR={args.ir}")
    ok = all(inject(h, ir, spec) for h in htmls)
    print("◆ 完成" + ("" if ok else "（存在失败，必须修复后再交付）"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
