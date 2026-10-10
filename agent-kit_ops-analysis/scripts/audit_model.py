#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S0 静态审计：不执行任何模型代码，仅用 AST 解析毛坯目录。

产出 audit_report.json（落在 --out 指定目录，通常为 outputs/<模型>/audit/），
并在 stdout 打印中文摘要。审计"形态"而不认"具体库"：本工具不内置任何模型
或依赖库的名字级知识，只按语法形态归纳。

审计内容（全部为通用形态扫描）：
  1. files            文件清单 + 每文件 sha256（可直接作为 IR 的 source_files_hash 底稿）
  2. imports          全部 import（模块、名字、file:line）+ 本环境可用性判定
  3. missing_deps     当前环境装不上的顶层模块（潜在"非 torch kernel/外部库"来源）
  4. fallback_branches try/except ImportError 的回退分支（file:line + 回退时做了什么）
  5. external_calls   缺依赖名字的全部调用点（file:line:name：模型依赖外部 kernel 的证据）
  6. classes          类清单：基类、是否疑似 nn.Module 子类、是否有 forward、方法列表
  7. config_like      疑似配置类（dataclass 或大量类属性默认值）：字段与默认值
  8. dist_refs        分布式痕迹（init_process_group / world_size / rank / barrier ...）
  9. device_refs      设备痕迹（.cuda / device= / mps / np / cuda 字符串）
 10. weight_refs      权重加载痕迹（safetensors / torch.load / load_state_dict / from_pretrained）
 11. toplevel_calls   模块顶层可执行语句（import 时刻副作用：建目录/打印/初始化……）
 12. init_outline     模型类 __init__ 里的子模块登记轮廓（self.xxx = ... 的静态归纳）

用法：
  python audit_model.py --model <毛坯目录> --out <审计产物目录>

退出码：0=审计完成（即使发现缺依赖也完成——那是发现，不是故障）。
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

# 不写字节码：毛坯目录只读，绝不能留下 __pycache__
sys.dont_write_bytecode = True

# 标准 Python 自带模块的顶层名（用于区分"外部第三方依赖"与"标准库"）
_STDLIB_TOP = set(getattr(sys, "stdlib_module_names", ())) or set(sys.builtin_module_names)
# 审计时跳过的目录名
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".ipynb_checkpoints"}

# 分布式痕迹的属性/方法名（按语法形态匹配，不针对具体框架）
_DIST_MARKERS = {
    "init_process_group", "destroy_process_group", "get_rank", "get_world_size",
    "barrier", "all_reduce", "all_gather", "broadcast", "is_initialized",
    "world_size", "local_rank", "tensor_model_parallel", "expert_model_parallel",
}
# 设备痕迹的字符串
_DEVICE_MARKERS = {"cuda", "mps", "npu", "xpu", "hpu", "cpu"}


def sha256_of(path):
    """计算文件 sha256（流式读取）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path, model_dir):
    """转为相对毛坯根目录的 POSIX 路径（供 IR evidence 使用）。"""
    return Path(path).resolve().relative_to(Path(model_dir).resolve()).as_posix()


def module_available(top_name):
    """判断某顶层模块在当前环境是否可用（不真正导入，只查 finder）。"""
    try:
        spec = importlib.util.find_spec(top_name)
        return spec is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


class FileAudit(ast.NodeVisitor):
    """对单个 .py 文件做 AST 归纳。"""

    def __init__(self, relpath, source):
        self.rel = relpath
        self.src_lines = source.splitlines()
        self.imports = []          # {module, name, asname, line}
        self.fallbacks = []        # {line, handler_line, names, branch_src}
        self.classes = []          # 类记录
        self.toplevel_calls = []   # {line, code}
        self.dist_refs = []        # {line, what}
        self.device_refs = []      # {line, code}
        self.weight_refs = []      # {line, what}
        self.external_calls = []   # {line, func}
        self._imported_names = {}  # asname/名字 -> 模块名（本文件可见绑定）

    # ---------- 工具 ----------
    def _src(self, lineno, end=None):
        end = end or min(lineno + 2, len(self.src_lines))
        return "；".join(self.src_lines[lineno - 1:end]).strip()[:200]

    # ---------- import ----------
    def visit_Import(self, node):
        for a in node.names:
            self.imports.append({"module": a.name, "name": a.name,
                                 "asname": a.asname, "line": node.lineno})
            self._imported_names[a.asname or a.name] = a.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        mod = "." * node.level + (node.module or "")
        for a in node.names:
            if a.name == "*":
                continue
            self.imports.append({"module": mod, "name": a.name,
                                 "asname": a.asname, "line": node.lineno})
            self._imported_names[a.asname or a.name] = f"{mod}.{a.name}"
        self.generic_visit(node)

    # ---------- try/except 回退 ----------
    def visit_Try(self, node):
        for h in node.handlers:
            # 只关心 ImportError/ModuleNotFoundError 回退
            names = []
            for t in (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
                      if h.type else []):
                if isinstance(t, ast.Name):
                    names.append(t.id)
            if any("Import" in n or "Module" in n for n in names):
                self.fallbacks.append({
                    "line": node.lineno,
                    "handler_line": h.lineno,
                    "handler_type": "/".join(names),
                    "branch_src": self._src(h.lineno, h.lineno + 3),
                })
        self.generic_visit(node)

    # ---------- 类 ----------
    def visit_ClassDef(self, node):
        bases = [self._name_of(b) for b in node.bases]
        methods, attrs, init_outline = [], {}, []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
                if item.name == "__init__":
                    init_outline = self._scan_init(item)
            elif isinstance(item, ast.Assign):
                for t in item.targets:
                    if isinstance(t, ast.Name):
                        attrs[t.id] = self._const_of(item.value)
        # 是否疑似 nn.Module 子类：基类名里出现 Module（含别名，如 nn.Module / NNModule）
        is_module_like = any("Module" in (b or "") for b in bases)
        self.classes.append({
            "name": node.name, "line": node.lineno, "bases": [b for b in bases if b],
            "nn_module_like": is_module_like,
            "has_forward": "forward" in methods,
            "methods": methods,
            "class_attrs": attrs,
            "init_outline": init_outline,
        })
        self.generic_visit(node)

    def _scan_init(self, fn):
        """静态归纳 __init__ 中的 self.xxx = ... 登记（子模块树轮廓，尽力而为）。"""
        out = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                            and t.value.id == "self" and node.lineno):
                        out.append({"attr": t.attr, "line": node.lineno,
                                    "rhs": self._src(node.lineno, node.lineno)[:120]})
        return out

    # ---------- 调用点 / 引用 ----------
    def visit_Call(self, node):
        f = self._name_of(node.func)
        if f:
            # 分布式 / 权重加载 / 外部依赖调用点
            if any(m in f for m in _DIST_MARKERS):
                self.dist_refs.append({"line": node.lineno, "what": f})
            if any(m in f.lower() for m in ("load_state_dict", "torch.load",
                                            "safetensors", "from_pretrained")):
                self.weight_refs.append({"line": node.lineno, "what": f})
            # 调用的顶层名（剥掉属性链）是否来自某个 import
            top = f.split(".")[0]
            src = self._imported_names.get(top)
            if src:
                self.external_calls.append({"line": node.lineno, "func": f, "from": src})
        self.generic_visit(node)

    def visit_Attribute(self, node):
        f = self._name_of(node)
        if f and any(m in f for m in _DIST_MARKERS):
            self.dist_refs.append({"line": node.lineno, "what": f})
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in _DEVICE_MARKERS:
            self.device_refs.append({"line": node.lineno, "code": f"设备字符串 {node.value!r}"})
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in _DEVICE_MARKERS:
            self.device_refs.append({"line": node.lineno, "code": f"设备名 {node.id}"})
        self.generic_visit(node)

    # ---------- 顶层可执行 ----------
    def visit_Module(self, node):
        for item in node.body:
            # 装饰器下的 def/class 是声明，不算副作用；赋值本身多数无害但调用要报
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(item, ast.Assign):
                # 赋值右侧若有调用，记下（如 MODEL_PATH = Path(__file__).mkdir()）
                for n in ast.walk(item.value):
                    if isinstance(n, ast.Call):
                        self.toplevel_calls.append(
                            {"line": item.lineno, "code": self._src(item.lineno)[:120]})
                        break
                continue
            if isinstance(item, (ast.Import, ast.ImportFrom, ast.Pass, ast.If)):
                # if 顶层可能是 __main__ 守卫或常量推导，扫其内部调用
                for n in ast.walk(item):
                    if isinstance(n, ast.Call):
                        self.toplevel_calls.append(
                            {"line": n.lineno, "code": self._src(n.lineno)[:120]})
                        break
                continue
            self.toplevel_calls.append({"line": getattr(item, "lineno", 0),
                                        "code": self._src(getattr(item, "lineno", 1))[:120]})
        self.generic_visit(node)

    # ---------- 常量/名字的简版取值 ----------
    @staticmethod
    def _name_of(node):
        """把表达式还原成点链名字（nn.Module → 'nn.Module'），还原不了返回 None。"""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
        return None

    def _const_of(self, node):
        """类属性默认值的简版字面化（够读即可，不求全）。"""
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            s = self._src(node.lineno, node.lineno) if hasattr(node, "lineno") else "?"
            return s[:80]


def audit(model_dir):
    """主审计流程：返回 audit_report 字典。"""
    model_dir = Path(model_dir).resolve()
    py_files = sorted(p for p in model_dir.rglob("*.py")
                      if not any(part in _SKIP_DIRS for part in p.parts))
    other_files = sorted(p for p in model_dir.rglob("*")
                         if p.is_file() and p.suffix != ".py"
                         and not any(part in _SKIP_DIRS for part in p.parts))

    report = {
        "tool": "audit_model.py（S0 静态审计）",
        "model_dir": str(model_dir),
        "env": {"python": sys.version.split()[0],
                "torch": None, "torchvista": None},
        "files": {}, "other_files": [],
        "imports": [], "missing_deps": [], "fallbacks": [],
        "external_calls": [], "classes": [], "config_like": [],
        "dist_refs": [], "device_refs": [], "weight_refs": [],
        "toplevel_calls": [], "syntax_errors": [],
    }
    # 环境版本（供 spec 编写参考；失败不阻断审计）
    try:
        import torch
        report["env"]["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import torchvista
        report["env"]["torchvista"] = getattr(torchvista, "__version__", "未知")
    except Exception:
        pass

    seen_top_modules = {}
    for p in py_files:
        rp = rel(p, model_dir)
        report["files"][rp] = sha256_of(p)
        try:
            source = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            report["syntax_errors"].append(
                {"file": rp, "error": "非 UTF-8 文本（可能为二进制/其他编码）"})
            continue
        try:
            tree = ast.parse(source, filename=str(p))
        except SyntaxError as e:
            report["syntax_errors"].append({"file": rp, "error": f"语法错误：{e}"})
            continue
        fa = FileAudit(rp, source)
        fa.visit(tree)
        # 归并进总报告（每条带 file）
        report["imports"] += [{**im, "file": rp} for im in fa.imports]
        report["fallbacks"] += [{**fb, "file": rp} for fb in fa.fallbacks]
        report["classes"] += [{**c, "file": rp} for c in fa.classes]
        report["dist_refs"] += [{**d, "file": rp} for d in fa.dist_refs]
        report["device_refs"] += [{**d, "file": rp} for d in fa.device_refs]
        report["weight_refs"] += [{**w, "file": rp} for w in fa.weight_refs]
        report["toplevel_calls"] += [{**t, "file": rp} for t in fa.toplevel_calls]
        # 外部调用点单独按文件存（后续按缺依赖过滤）
        report["external_calls"] += [{**c, "file": rp} for c in fa.external_calls]

    for p in other_files:
        report["other_files"].append({"file": rel(p, model_dir), "size": p.stat().st_size})

    # ---- 缺依赖判定：import 的顶层模块不在标准库、不在本环境、也不在毛坯内 ----
    local_top = {p.stem for p in py_files}
    local_top |= {p.parent.name for p in py_files if p.name == "__init__.py"}
    for im in report["imports"]:
        top = im["module"].lstrip(".").split(".")[0]
        if not top:
            continue
        seen_top_modules.setdefault(top, {"files": set(), "lines": []})
        seen_top_modules[top]["files"].add(im["file"])
        seen_top_modules[top]["lines"].append(f'{im["file"]}:{im["line"]}')
    for top, info in sorted(seen_top_modules.items()):
        if top in _STDLIB_TOP or top in local_top or top == "":
            continue
        if not module_available(top):
            report["missing_deps"].append({
                "module": top,
                "used_in": sorted(info["files"]),
                "import_sites": info["lines"][:10],
            })
    # ---- external_calls 过滤成"真值得看的"：来源是缺依赖模块 ----
    missing_tops = {m["module"] for m in report["missing_deps"]}
    report["external_calls"] = [
        c for c in report["external_calls"]
        if c["from"].lstrip(".").split(".")[0] in missing_tops
    ]
    # ---- 配置类判定：dataclass 装饰或 ≥2 个带默认值的类属性 ----
    for c in report["classes"]:
        defaults = {k: v for k, v in c["class_attrs"].items() if v is not None}
        if len(defaults) >= 2 or any("dataclass" in (b or "") for b in c["bases"]):
            report["config_like"].append({
                "file": c["file"], "line": c["line"], "name": c["name"],
                "fields": defaults,
            })
    # 去重 dist_refs（属性与调用会重复计数同一行）
    _dd, seen = [], set()
    for d in report["dist_refs"]:
        k = (d["file"], d["line"], d["what"])
        if k not in seen:
            seen.add(k)
            _dd.append(d)
    report["dist_refs"] = _dd
    return report


def summarize(report):
    """打印中文摘要（面向执行代理的排障入口）。"""
    n_py = len(report["files"])
    print(f"◆ 审计完成：{n_py} 个 .py 文件，{len(report['other_files'])} 个其他文件")
    if report["env"]["torch"]:
        print(f"◆ 运行环境：torch {report['env']['torch']}，torchvista {report['env']['torchvista']}")
    if report["syntax_errors"]:
        print(f"◆ ⚠️ {len(report['syntax_errors'])} 个文件解析失败（详见 syntax_errors）")
    if report["missing_deps"]:
        print(f"◆ ⚠️ 缺依赖（本环境不可用）：")
        for m in report["missing_deps"]:
            sites = "、".join(m["import_sites"][:4])
            print(f"    - {m['module']}（用于：{sites}）")
        print(f"  → 共 {len(report['external_calls'])} 个调用点引用这些模块（external_calls 清单）")
    else:
        print("◆ 所有 import 的顶层依赖在本环境均可用")
    if report["fallbacks"]:
        print(f"◆ 回退分支（try/except ImportError）：{len(report['fallbacks'])} 处——"
              f"构造模型前必须先确定每个回退走哪条路径")
    model_classes = [c for c in report["classes"] if c["nn_module_like"]]
    fwd = [c for c in model_classes if c["has_forward"]]
    print(f"◆ 类：{len(report['classes'])} 个（疑似 nn.Module 子类 {len(model_classes)}，"
          f"其中带 forward {len(fwd)}）")
    for c in fwd[:12]:
        print(f"    - {c['file']}:{c['line']} {c['name']}（基类 {','.join(c['bases'])}）")
    if len(fwd) > 12:
        print(f"    …… 另 {len(fwd) - 12} 个见报告")
    if report["config_like"]:
        print(f"◆ 疑似配置类：{len(report['config_like'])} 个（字段默认值可用于推导构造参数）")
    if report["dist_refs"]:
        files = sorted({d["file"] for d in report["dist_refs"]})
        print(f"◆ ⚠️ 分布式痕迹：{len(report['dist_refs'])} 处引用（{ '、'.join(files) }）"
              f"——单进程构造前必须处理（初始化时序）")
    if report["weight_refs"]:
        print(f"◆ 权重加载引用：{len(report['weight_refs'])} 处（随机初始化分析可绕开，但须声明）")
    if report["toplevel_calls"]:
        print(f"◆ 模块顶层副作用调用：{len(report['toplevel_calls'])} 处（import 时刻会执行，先看清）")


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0 静态审计（AST，不执行模型代码）")
    ap.add_argument("--model", required=True, help="模型毛坯目录（只读）")
    ap.add_argument("--out", required=True, help="审计产物目录（写 audit_report.json）")
    args = ap.parse_args(argv)

    report = audit(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    summarize(report)
    print(f"◆ 报告已写入：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
