# -*- coding: utf-8 -*-
"""测试夹具模型：合成小模型（仅用于 kit 机制验证，与任何真实模型无关）。

刻意包含一个"本环境不存在的依赖" nonexistent_kit——用于验证 kit 的完整链路：
静态审计识别缺依赖 → shim 等价替换 → 遮蔽导入 + 挂树 + 重绑 → 图上红虚线标记。
"""
import torch
import torch.nn as nn

from nonexistent_kit import mystery_act  # 假外部 kernel（本环境不可用）


class TinyBlock(nn.Module):
    """普通块：归一化 → 升维投影 → 外部激活 kernel → 降维投影。"""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = mystery_act(h)        # 外部 kernel 调用点（audit 应报出）
        return self.fc2(h)


class TinyModel(nn.Module):
    """入口模型：embedding → N 个 TinyBlock → 输出头。"""

    def __init__(self, dim=32, n_layers=3, vocab=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([TinyBlock(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
