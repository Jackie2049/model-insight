#!/usr/bin/env python3
"""render_index.py — 生成 viz/index.html 集成面板（模型无关）。

以 DeepSeek-V3.2 手搓面板为模板：CSS 与交互 JS 全部保留并改造为模型无关，
数据全部实时读取、零硬编码：
  - capture/capture_summary.json   作业清单（id/status/nodes/html）
  - capture/<id>.measured.json     每作业元数据（kind/scenario/target）
  - fusion/final_ir.json           模型名、五态统计、替换件、缩水声明

用法：
  python3 render_index.py --capture-dir <capture 目录> --ir <final_ir.json> --out <viz 目录>

只写两个文件：<out>/index.html 与 <out>/views.json，不碰其余页面。
写前校验链接文件在盘，写后回读断言五态数字与链接数；重跑整体覆盖（幂等）。
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- 报错与校验


def fail(msg: str) -> None:
    print(f"[render_index] 错误：{msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- 模板（源自手搓面板，JS 交互全保留）

CSS = (
    ":root{font-family:Inter,-apple-system,BlinkMacSystemFont,\"PingFang SC\",\"Microsoft Yahei\",sans-serif;"
    "color:#292722;background:#f8f7f4;--panel:#f2f0eb;--panel2:#ebe8e1;--line:#e2ded6;--muted:#77736c;"
    "--accent:#c56a45;--accent-soft:#f4e7df;--good:#39745c;--warn:#a66a26}\n"
    "*{box-sizing:border-box}body{margin:0;min-height:100vh}button,a{font:inherit}button{cursor:pointer}a{color:inherit}"
    ".app{display:grid;grid-template-columns:292px minmax(0,1fr);height:100vh;min-height:580px}"
    ".rail{background:var(--panel);border-right:1px solid var(--line);overflow:auto;padding:22px 17px}"
    ".brand{display:flex;align-items:center;gap:9px;font-size:11px;letter-spacing:.09em;font-weight:750;color:#716d66;text-transform:uppercase}"
    ".mark{width:24px;height:24px;display:grid;place-items:center;border-radius:50%;background:#d97850;color:#fff;font-size:14px;letter-spacing:0}"
    ".rail h1{font-size:22px;line-height:1.2;letter-spacing:-.035em;margin:22px 0 8px;font-weight:650}"
    ".subtle{color:var(--muted);font-size:12px;line-height:1.55}"
    ".section{margin-top:27px}"
    ".section-head{display:flex;align-items:center;justify-content:space-between;color:#77736c;font-size:10px;letter-spacing:.08em;font-weight:700;text-transform:uppercase;margin-bottom:9px}"
    ".tabs{display:grid;grid-template-columns:1fr 1fr;gap:7px}"
    ".tab,.item{border:1px solid transparent;background:transparent;color:#383630;border-radius:9px}"
    ".tab{padding:10px 7px;font-size:13px;font-weight:600}"
    ".tab:hover,.item:hover{background:#eae7e0}"
    ".tab.active,.item.active{border-color:#e6c6b6;background:#f5e8e1;color:#7d4029;box-shadow:none}"
    ".tab small{display:block;font-size:10px;color:#8f786b;font-weight:450;margin-top:3px}"
    ".items{display:grid;gap:3px}"
    ".item{text-align:left;padding:9px 10px;display:flex;align-items:center;gap:10px;width:100%}"
    ".n{flex:0 0 27px;height:27px;border-radius:8px;display:grid;place-items:center;background:#e8e5de;color:#706b62;font-size:12px;font-weight:650}"
    ".item.active .n{background:#e6c7b7;color:#87462e}"
    ".item strong{display:block;font-size:12px;font-weight:600}"
    ".item small{display:block;color:var(--muted);font-size:10px;margin-top:3px;line-height:1.25}"
    ".count{margin-left:auto;color:#8a857c;font-size:10px;white-space:nowrap}"
    ".empty{color:var(--muted);font-size:12px;border:1px dashed #d7d1c7;padding:12px;border-radius:9px}"
    ".stats{display:grid;gap:5px}"
    ".stat{display:flex;align-items:center;gap:7px;font-size:11px;color:#4c4842}"
    ".stat b{font-weight:650}"
    ".dot{width:8px;height:8px;border-radius:50%;flex:0 0 8px}"
    ".note{margin-top:9px;font-size:10px;line-height:1.5;color:var(--muted);cursor:help}"
    ".main{min-width:0;display:flex;flex-direction:column}"
    ".top{display:flex;justify-content:space-between;align-items:center;gap:14px;padding:17px 23px;border-bottom:1px solid var(--line);min-height:78px;background:#fbfaf8}"
    ".eyebrow{color:#a85a3b;font-size:10px;letter-spacing:.1em;text-transform:uppercase;font-weight:700}"
    ".heading{display:flex;align-items:center;gap:10px;margin-top:5px;flex-wrap:wrap}"
    ".heading h2{font-size:17px;margin:0;font-weight:620}"
    ".pill{border:1px solid #e2ded6;color:#77736c;border-radius:50px;padding:3px 8px;font-size:10px}"
    ".topactions{display:flex;gap:7px;flex-shrink:0}"
    ".action{background:#c56a45;color:white;border:1px solid #c56a45;border-radius:8px;text-decoration:none;padding:8px 11px;font-size:11px}"
    ".action:hover{background:#ad5938}"
    ".action.secondary{background:transparent;color:#57534d;border-color:#ddd8cf}"
    ".action.secondary:hover{background:#f0eee9}"
    ".context{display:flex;gap:16px;align-items:center;justify-content:space-between;padding:10px 23px;border-bottom:1px solid var(--line);color:#6e6961;font-size:11px;min-height:45px;background:#fbfaf8}"
    ".context .warning{color:var(--warn)}"
    ".context .good{color:var(--good)}"
    ".framewrap{position:relative;flex:1;min-height:0;background:#f0f1f2}"
    ".framewrap iframe{width:100%;height:100%;border:0;background:white}"
    ".loading{position:absolute;z-index:1;inset:0;display:grid;place-items:center;background:#f0f1f2;color:#686d72;font-size:13px}"
    ".loading[hidden]{display:none}"
    ".foot{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:8px 23px;border-top:1px solid var(--line);font-size:10px;color:var(--muted);background:#fbfaf8}"
    ".foot a{color:#9e573b;text-decoration:none}"
    ".focused .rail{display:none}.focused .app{grid-template-columns:1fr}"
    "@media(max-width:850px){.app{grid-template-columns:1fr;height:auto;min-height:100vh}"
    ".rail{border-right:0;border-bottom:1px solid var(--line);max-height:290px;padding:14px}"
    ".rail h1{font-size:18px;margin:10px 0 3px}.rail .section{margin-top:14px}"
    ".items{grid-template-columns:repeat(2,minmax(0,1fr))}.item{padding:8px}"
    ".main{height:calc(100vh - 260px);min-height:450px}.top{padding:10px}.context,.foot{padding:8px 10px}"
    ".action{padding:7px}.topactions{gap:4px}}"
)

TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__MODEL__ · 算子流水</title>
<style>
__CSS__
</style>
</head>
<body>
<div class="app">
<aside class="rail" aria-label="视图导航">
  <div class="brand"><span class="mark">◈</span> Model-Insight</div>
  <h1>__MODEL__<br>算子流水</h1>
  <section class="section"><div class="section-head">01 · 运行场景</div><div class="tabs" id="scenarios"></div></section>
  <section class="section"><div class="section-head">02 · __DEPTH_SECTION__ <span id="depthCount"></span></div><div class="items" id="depths"></div></section>
  <section class="section"><div class="section-head">03 · 专项模块实验</div><div class="items" id="modules"></div></section>
  <section class="section"><div class="section-head">04 · 分析摘要</div><div id="summary"></div></section>
</aside>
<main class="main">
  <header class="top"><div><div class="eyebrow" id="breadcrumb">OPERATOR FLOW</div><div class="heading"><h2 id="viewTitle">加载中</h2><span class="pill" id="viewType"></span></div></div><div class="topactions"><button class="action secondary" id="focusBtn" type="button">专注视图</button><a class="action" id="standalone" href="#" target="_blank" rel="noopener">单独打开 ↗</a></div></header>
  <div class="context"><span id="viewDescription"></span><span id="viewEvidence"></span></div>
  <div class="framewrap"><div class="loading" id="loading">正在加载 TorchVista 交互图…</div><iframe id="graph" title="TorchVista 算子图" loading="eager"></iframe></div>
  <footer class="foot"><span>基于现有 TorchVista 页面 · 轻量融合 IR 独立保存</span><span>__FOOT_LINKS__</span></footer>
</main></div>
<script>
const VIEWS=__VIEWS__;
const DETAILS=__DETAILS__;
const SUMMARY=__SUMMARY__;
const STATUS_META=[["verified","已验证","#39745c"],["mismatch","不一致","#a66a26"],["llm_only","仅推导","#77736c"],["capture_only","仅捕获","#77736c"],["conflict","冲突","#b3452f"]];
const $=id=>document.getElementById(id);
const byId=Object.fromEntries(VIEWS.map(v=>[v.id,v]));
const SCENARIOS=[...new Set(VIEWS.map(v=>v.scenario))].map(s=>({key:s,label:s.charAt(0).toUpperCase()+s.slice(1).replace(/_/g,' '),count:VIEWS.filter(v=>v.scenario===s).length}));
let scene=SCENARIOS[0].key, selected='', focused=false;
function hashState(){const q=new URLSearchParams(location.hash.slice(1));return {scene:q.get('scene'),view:q.get('view')};}
function setHash(v){history.replaceState(null,'','#scene='+encodeURIComponent(v.scenario)+'&view='+encodeURIComponent(v.id));}
function choose(id){const v=byId[id];if(!v)return;scene=v.scenario;selected=id;setHash(v);render();}
function sceneLabel(s){const m=SCENARIOS.find(x=>x.key===s);return m?m.label:s;}
function sceneButton(s){const meta=SCENARIOS.find(x=>x.key===s);const b=document.createElement('button');b.type='button';b.className='tab'+(scene===s?' active':'');b.innerHTML='<span>'+meta.label+'</span><small>'+meta.count+' 个视图</small>';b.onclick=()=>{const old=byId[selected];const preferred=old&&old.kind==='macro'?VIEWS.find(v=>v.kind==='macro'&&v.scenario===s&&v.depth===old.depth):null;const target=preferred||VIEWS.find(v=>v.kind==='macro'&&v.scenario===s)||VIEWS.find(v=>v.scenario===s);choose(target.id)};return b;}
function macroTitle(v){return v.depth==null?v.title:('D'+v.depth+' · '+(DETAILS['depth'+v.depth]?.title||v.title));}
function viewButton(v){const b=document.createElement('button');b.type='button';b.className='item'+(selected===v.id?' active':'');const micro=v.kind==='micro';const title=micro?(DETAILS[v.id]?.title||v.target||v.id):macroTitle(v);const subtitle=micro?(DETAILS[v.id]?.description||v.target):(DETAILS['depth'+v.depth]?.description||'');const badge=micro?'◇':(v.depth==null?'◆':v.depth);b.innerHTML='<span class="n">'+badge+'</span><span><strong>'+title+'</strong><small>'+subtitle+'</small></span><span class="count">'+(v.nodes??'—')+'</span>';b.onclick=()=>choose(v.id);return b;}
function renderSummary(){const box=$('summary');const stats=document.createElement('div');stats.className='stats';for(const [k,zh,color] of STATUS_META){const n=SUMMARY.statuses[k]||0;const row=document.createElement('div');row.className='stat';if(!n)row.style.opacity='.45';row.innerHTML='<span class="dot" style="background:'+color+'"></span><b>'+n+'</b><span>'+zh+'</span>';stats.appendChild(row);}const note=document.createElement('div');note.className='note';note.textContent='替换件 '+SUMMARY.replacementCount+' 个 · '+SUMMARY.scaleNote;note.title=SUMMARY.replacementsFull||'';box.replaceChildren(stats,note);}
function render(){const v=byId[selected];if(!v)return;$('scenarios').replaceChildren(...SCENARIOS.map(x=>sceneButton(x.key)));const macros=VIEWS.filter(x=>x.kind==='macro'&&x.scenario===scene).sort((a,b)=>(a.depth??99)-(b.depth??99));$('depthCount').textContent=macros.length+' 张';$('depths').replaceChildren(...macros.map(viewButton));const micros=VIEWS.filter(x=>x.kind==='micro'&&x.scenario===scene);$('modules').replaceChildren(...(micros.length?micros.map(viewButton):[(()=>{const e=document.createElement('div');e.className='empty';e.textContent='当前场景没有单独的模块实验图；可在整模型视图逐级查看。';return e;})()]));const info=DETAILS[v.id]||DETAILS['depth'+v.depth]||{};const micro=v.kind==='micro';$('breadcrumb').textContent=sceneLabel(scene)+' / '+(micro?'模块实验':'整模型')+' / '+(micro?v.target:(v.depth==null?'整模型':'D'+v.depth));$('viewTitle').textContent=micro?(info.title||v.target):macroTitle(v);$('viewType').textContent=micro?'独立模块输入':'整模型捕获';$('viewDescription').textContent=info.long||info.description||'';$('viewEvidence').textContent=(v.nodes??'—')+' 节点 · '+(v.status==='ok'?'抓图成功':'状态 '+v.status);$('viewEvidence').className=v.status==='ok'?'good':'warning';$('standalone').href=v.file;const frame=$('graph');if(frame.getAttribute('src')!==v.file){$('loading').hidden=false;frame.src=v.file;}}
$('graph').addEventListener('load',()=>{ $('loading').hidden=true; try{const panel=$('graph').contentDocument.getElementById('ops-kit-fusion-injection');if(panel)panel.classList.add('ops-kit-collapsed');}catch(e){} });
$('focusBtn').onclick=()=>{focused=!focused;document.body.classList.toggle('focused',focused);$('focusBtn').textContent=focused?'显示导航':'专注视图';};
window.addEventListener('hashchange',()=>{const h=hashState();if(byId[h.view]){scene=byId[h.view].scenario;selected=h.view;render();}});
const h=hashState();selected=byId[h.view]?h.view:(VIEWS.find(v=>v.kind==='macro'&&v.status==='ok')||VIEWS.find(v=>v.status==='ok')||VIEWS[0]).id;scene=byId[selected].scenario;setHash(byId[selected]);render();renderSummary();
</script>
</body></html>
"""

# 深度系列的中性文案（模型无关）：depth → (title, description, long)
DEPTH_META = {
    0: ("模型骨架", "Embedding → Block → Norm → Head", "先看整个模型的主数据流与层级边界。"),
    1: ("层内主链", "Attention、FFN、Norm", "展开每层的注意力、前馈和归一化模块。"),
    2: ("功能模块", "层内各功能子模块", "查看注意力子模块、前馈、门控与专家等计算构成。"),
    3: ("内部运算", "线性层与逐算子计算点", "查看子模块内部的计算点与张量形状。"),
}
DEPTH_FALLBACK = ("算子细节", "全部可见调用点", "显示最细粒度算子，图较大，初次加载可能需要等待。")

STATUS_KEYS = ("verified", "mismatch", "llm_only", "capture_only", "conflict")


def short_model_name(name: str, fallback: str) -> str:
    """模型短名：按「（」截断；空值兜底输出目录名。"""
    name = (name or "").strip()
    if not name:
        return fallback
    idx = name.find("（")
    return name[:idx].strip() if idx > 0 else name


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_views(capture_dir: Path, summary: dict) -> list:
    """从 capture_summary + measured.json 组装视图清单（含写前校验）。"""
    views, missing_html, missing_meta = [], [], []
    for job in summary.get("jobs", []):
        jid = job.get("id") or ""
        html_name = job.get("html") or f"{jid}.html"
        html_path = capture_dir / html_name
        if not html_path.is_file():
            missing_html.append(html_name)
        measured_path = capture_dir / (job.get("measured") or f"{jid}.measured.json")
        if not measured_path.is_file():
            missing_meta.append(measured_path.name)
            continue
        meta = json.loads(measured_path.read_text())
        mjob = meta.get("job") or {}
        scenario = (mjob.get("scenario") or "").strip()
        if not scenario:
            fail(f"{measured_path.name} 的 job.scenario 为空，无法归入运行场景")
        kind = mjob.get("kind") or "macro"
        m = re.search(r"_d(\d+)$", jid)
        depth = int(m.group(1)) if (kind == "macro" and m) else (0 if kind == "micro" else None)
        target = (mjob.get("target") or "").strip() or None
        if kind == "macro":
            title = (DEPTH_META.get(depth, DEPTH_FALLBACK)[0] if depth is not None else "整模型")
        else:
            title = target or jid
        views.append({
            "id": jid,
            "file": html_name,
            "kind": kind,
            "scenario": scenario,
            "depth": depth,
            "target": target,
            "title": title,
            "nodes": job.get("nodes"),
            "status": job.get("status") or "unknown",
        })
    if missing_html:
        fail("以下 capture 页面缺失（VIEWS 引用了不存在的 html）：\n  " + "\n  ".join(missing_html))
    if missing_meta:
        fail("以下 measured.json 缺失（无法取 job 元数据）：\n  " + "\n  ".join(missing_meta))
    if not views:
        fail("capture_summary.json 的 jobs 为空，没有可展示的视图")
    return views


def build_details(views: list) -> dict:
    """中性文案（模型无关）：depth 键 + micro 逐作业键。"""
    details = {}
    for v in views:
        if v["kind"] == "macro":
            if v["depth"] is None:
                details.setdefault("depthnull", {
                    "title": "整模型", "description": "单次整模型捕获",
                    "long": "该运行无深度系列，整模型一张图完成观察。",
                })
            else:
                t, desc, long = DEPTH_META.get(v["depth"], DEPTH_FALLBACK)
                details.setdefault(f"depth{v['depth']}", {"title": t, "description": desc, "long": long})
        else:
            key = v["target"] or v["id"]
            details.setdefault(v["id"], {
                "title": key, "description": "独立输入捕获",
                "long": f"使用单独构造的输入捕获 {key}；与整模型视图分别产生。",
            })
    return details


def build_summary(ir: dict) -> dict:
    """五态实时统计（禁硬编码）+ 替换件 + 缩水声明。"""
    nodes = ir.get("nodes") or []
    counter = Counter(n.get("status") for n in nodes)
    statuses = {k: int(counter.get(k, 0)) for k in STATUS_KEYS}
    if sum(statuses.values()) != len(nodes):
        fail(f"IR 节点状态计数异常：{statuses} 总和 != 节点数 {len(nodes)}（出现未知状态值）")
    replacements = ir.get("contract", {}).get("replacements") or []
    whats = [str(r.get("what") or "").strip() for r in replacements]
    scale_note = truncate(ir.get("model", {}).get("scale_note"), 140)
    return {
        "statuses": statuses,
        "totalNodes": len(nodes),
        "replacementCount": len(replacements),
        "replacementsFull": "\n".join(f"· {w}" for w in whats if w),
        "scaleNote": scale_note or "（IR 未提供缩水声明）",
    }


def build_foot_links(out_dir: Path) -> str:
    links = ['<a href="../fusion/final_ir.json" target="_blank" rel="noopener">查看分析 JSON ↗</a>']
    if (out_dir / "README.md").is_file():
        links.append('<a href="README.md" target="_blank" rel="noopener">图集说明 ↗</a>')
    return "　·　".join(links)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 viz/index.html 集成面板（数据实时读取，零硬编码）")
    ap.add_argument("--capture-dir", required=True, help="capture 目录（含 capture_summary.json 与各页面）")
    ap.add_argument("--ir", required=True, help="fusion/final_ir.json 路径")
    ap.add_argument("--out", required=True, help="输出目录（通常为 viz/）")
    args = ap.parse_args()

    capture_dir, out_dir = Path(args.capture_dir), Path(args.out)
    summary_path = capture_dir / "capture_summary.json"
    if not summary_path.is_file():
        fail(f"找不到 {summary_path}")
    if not Path(args.ir).is_file():
        fail(f"找不到 {args.ir}")

    summary = json.loads(summary_path.read_text())
    ir = json.loads(Path(args.ir).read_text())

    model = short_model_name(ir.get("model", {}).get("name"), out_dir.resolve().parent.name)
    views = build_views(capture_dir, summary)
    details = build_details(views)
    panel_summary = build_summary(ir)

    depth_views = [v for v in views if v["kind"] == "macro" and v["depth"] is not None]
    depth_section = "整模型观察尺度" if depth_views else "整模型视图"

    dump = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    html = (TEMPLATE
            .replace("__CSS__", CSS)
            .replace("__MODEL__", model)
            .replace("__DEPTH_SECTION__", depth_section)
            .replace("__FOOT_LINKS__", build_foot_links(out_dir))
            .replace("__VIEWS__", dump(views))
            .replace("__DETAILS__", dump(details))
            .replace("__SUMMARY__", dump(panel_summary)))
    for token in ("__CSS__", "__MODEL__", "__DEPTH_SECTION__", "__FOOT_LINKS__", "__VIEWS__", "__DETAILS__", "__SUMMARY__"):
        if token in html:
            fail(f"占位符 {token} 未被替换，模板异常")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html)
    (out_dir / "views.json").write_text(json.dumps({
        "schema_version": "0.2",
        "generated_by": "render_index.py",
        "model": model,
        "summary": panel_summary,
        "views": views,
    }, ensure_ascii=False, indent=2) + "\n")

    # ---- 写后回读断言：占位符残留 / VIEWS 条数 / 五态数字 / 链接数 ----
    back = (out_dir / "index.html").read_text()
    m = re.search(r"const VIEWS=(\[.*?\]);\nconst DETAILS=", back, re.S)
    if not m:
        fail("回读失败：index.html 中找不到 VIEWS 数组")
    back_views = json.loads(m.group(1))
    if len(back_views) != len(views):
        fail(f"回读 VIEWS 条数 {len(back_views)} != 生成条数 {len(views)}")
    linked = {v["file"] for v in back_views}
    absent = [f for f in linked if not (capture_dir / f).is_file()]
    if absent:
        fail("回读发现面板链接指向不存在的页面：" + ", ".join(absent))
    back_summary = json.loads(re.search(r"const SUMMARY=(\{.*?\});\nconst STATUS_META", back, re.S).group(1))
    if back_summary["statuses"] != panel_summary["statuses"]:
        fail("回读五态统计与 IR 统计不一致")
    if not all(f'"{v["file"]}"' in back for v in back_views):
        fail("回读发现面板内缺失页面引用")

    print(f"[render_index] 完成：{out_dir / 'index.html'}")
    print(f"  模型={model} · 视图 {len(back_views)} 个（场景 {len(set(v['scenario'] for v in views))} 组）")
    print(f"  五态=" + "/".join(f"{k}:{n}" for k, n in panel_summary["statuses"].items())
          + f" · 替换件 {panel_summary['replacementCount']} 个")


if __name__ == "__main__":
    main()
