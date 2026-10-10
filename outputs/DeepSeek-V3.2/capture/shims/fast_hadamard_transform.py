# -*- coding: utf-8 -*-
"""fast_hadamard_transform 遮蔽模块（shim）· DeepSeek-V3.2

毛坯 model.py rotate_activation（L428-432）在函数体内延迟
`from fast_hadamard_transform import hadamard_transform`，本机未安装该 CUDA 库
（audit_report.missing_deps）。本文件经 spec.sys_path 首位遮蔽同名模块，
提供函数 `hadamard_transform(x, scale)` 的纯 torch 数学等价实现。

数学定义（自然序 / Sylvester 构造的 Walsh-Hadamard 变换）：
    H_1 = [[1]]；H_{2n} = [[H_n, H_n], [H_n, -H_n]]（元素 ±1）
    y = scale · (x @ H)，与 fast_hadamard_transform 库一致（库内 H 为未归一化
    Sylvester 矩阵、scale 由调用方传入；rotate_activation 传 N^-0.5，整体正交）。
行序约定说明见 llm-analysis/notes.md §8.1（open_question：与原 CUDA kernel
位序一致只能到"约定声明"级——库未装，无法对拍原实现）。
"""
import torch


def _hadamard_matrix(n: int, device, dtype: torch.dtype) -> torch.Tensor:
    """递归构造 n 阶 Sylvester Walsh-Hadamard 矩阵（元素 ±1）。n 必须是 2 的幂。"""
    assert n >= 1 and (n & (n - 1)) == 0, f"维度必须是 2 的幂：{n}"
    h = torch.ones(1, 1, device=device, dtype=dtype)
    size = 1
    while size < n:
        h = torch.cat([
            torch.cat([h, h], dim=1),
            torch.cat([h, -h], dim=1),
        ], dim=0)
        size *= 2
    return h


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """y = scale · (x @ H_n)，H_n 为未归一化 Walsh-Hadamard 矩阵。

    与库签名一致：支持任意前置维（末维为变换维）；内部 fp32 计算，输出转回
    输入 dtype（调用处 rotate_activation 断言输入 bf16、期望输出 bf16）。
    """
    n = x.size(-1)
    h = _hadamard_matrix(n, x.device, torch.float32)
    y = torch.matmul(x.float(), h.to(x.device)) * float(scale)
    return y.to(x.dtype)
