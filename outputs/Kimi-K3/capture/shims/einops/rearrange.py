"""einops.rearrange 极小子集：支持 单轴/括号分组/省略号 的拆分-置换-合并。"""
import torch


def _tokenize(side):
    toks, buf, dots = [], "", False
    for ch in side:
        if ch == " ":
            if buf:
                toks.append(("axes", buf.split())); buf = ""
            continue
        if ch == ".":
            dots = True
            if buf:
                toks.append(("axes", buf.split())); buf = ""
            continue
        if dots:
            toks.append(("dots", "...")); dots = False
        if ch == "(" or ch == ")":
            if buf:
                toks.append(("axes", buf.split())); buf = ""
            toks.append(("paren", ch))
        else:
            buf += ch
    if buf:
        toks.append(("axes", buf.split()))
    if dots:
        toks.append(("dots", "..."))
    return toks


def _parse(side, rank):
    """返回 (tokens, dots_pos)：tokens 为 ('single', name) | ('group', names) | ('dots',)。"""
    raw = _tokenize(side)
    out, dots = [], None
    depth = 0
    for kind, val in raw:
        if kind == "paren":
            if val == "(":
                depth += 1
                if depth > 1:
                    raise ValueError("不支持嵌套括号: " + side)
                cur = []
                out.append(["group_open", cur])
            else:
                depth -= 1
        elif kind == "axes":
            if depth:
                out[-1][1].extend(val)
            else:
                for n in val:
                    out.append(["single", n])
        elif kind == "dots":
            if dots is not None:
                raise ValueError("省略号最多出现一次: " + side)
            dots = len(out)
            out.append(["dots", None])
    tokens = []
    for item in out:
        if item[0] == "single":
            tokens.append(("single", item[1]))
        elif item[0] == "group_open":
            tokens.append(("group", item[1]))
        else:
            tokens.append(("dots", None))
    n_named = sum(1 for t in tokens if t[0] != "dots")
    if dots is not None and rank >= 0 and rank < n_named:
        raise ValueError(f"省略号维度不足: {side} 需要 {n_named} 显式维, 实际 rank={rank}")
    return tokens, dots


def rearrange(tensor, pattern, **axes_lengths):
    lhs, rhs = [s.strip() for s in pattern.split("->")]
    t = tensor
    rank = t.dim()
    l_toks, l_dots = _parse(lhs, rank)
    r_toks, r_dots = _parse(rhs, -1)

    # ---- 语义轴与尺寸 ----
    sizes = {}
    phys_dims = list(t.shape)
    sem_order = []          # 输入侧语义轴顺序（拆开括号、展开省略号后）
    dots_n = rank - (len(l_toks) - 1) if l_dots is not None else 0

    pi = 0
    di = 0
    for idx, (kind, val) in enumerate(l_toks):
        if kind == "dots":
            for _ in range(dots_n):
                name = f"...{di}"
                sizes[name] = phys_dims[pi]; pi += 1; di += 1
                sem_order.append(name)
        elif kind == "single":
            sizes[val] = phys_dims[pi]; pi += 1
            sem_order.append(val)
        else:  # group：拆分
            D = phys_dims[pi]; pi += 1
            known = {n: axes_lengths[n] for n in val if n in axes_lengths}
            unknown = [n for n in val if n not in axes_lengths]
            if len(unknown) > 1:
                raise ValueError(f"分组 {val} 有多个未知维度，需提供 axes_lengths")
            if unknown:
                prod_known = 1
                for v in known.values():
                    prod_known *= v
                if D % prod_known != 0:
                    raise ValueError(f"分组 {val}: {D} 不能整除 {prod_known}")
                known[unknown[0]] = D // prod_known
            for n in val:
                sizes[n] = known[n]
                sem_order.append(n)

    # 输出侧语义轴顺序；省略号轴直接复用输入侧展开结果（顺序保持）
    dot_axes = [a for a in sem_order if a.startswith("...")]
    sem_out = []
    for kind, val in r_toks:
        if kind == "dots":
            if l_dots is None:
                raise ValueError("输出侧省略号无输入侧对应")
            sem_out.extend(dot_axes)
        elif kind == "single":
            if val not in sizes:
                raise ValueError(f"输出轴 {val} 未在输入侧出现")
            sem_out.append(val)
        else:
            for n in val:
                if n not in sizes:
                    raise ValueError(f"输出分组轴 {n} 未在输入侧出现")
                sem_out.append(n)

    if sorted(sem_order) != sorted(sem_out):
        raise ValueError(f"输入输出轴集合不一致: {sorted(sem_order)} vs {sorted(sem_out)}")

    # ---- 1) 先把输入物理维拆成语义轴（拆括号）----
    new_shape = []
    pi = 0
    dots_consumed = 0
    for idx, (kind, val) in enumerate(l_toks):
        if kind == "dots":
            for _ in range(dots_n):
                new_shape.append(sizes[f"...{dots_consumed}"]); dots_consumed += 1
        elif kind == "single":
            new_shape.append(sizes[val])
        else:
            new_shape.extend(sizes[n] for n in val)
    t = t.reshape(new_shape)

    # ---- 2) 置换到输出语义顺序 ----
    perm = [sem_order.index(a) for a in sem_out]
    t = t.permute(*perm)

    # ---- 3) 合并输出分组 ----
    merged = []
    i = 0
    # 重建输出 token 序列（dots 展开为实际轴名）
    exp_out = []
    for kind, val in r_toks:
        if kind == "dots":
            exp_out.append(("dots", None))
        else:
            exp_out.append((kind, val))
    si = 0
    merged_shape = []
    for kind, val in exp_out:
        if kind == "dots":
            for a in dot_axes:
                merged_shape.append(sizes[a]); si += 1
        elif kind == "single":
            merged_shape.append(sizes[val]); si += 1
        else:
            prod = 1
            for n in val:
                prod *= sizes[n]
            merged_shape.append(prod); si += len(val)
    return t.reshape(merged_shape)
