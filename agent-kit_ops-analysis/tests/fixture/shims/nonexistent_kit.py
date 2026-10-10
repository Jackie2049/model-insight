# -*- coding: utf-8 -*-
"""shim 模块：对假外部依赖 nonexistent_kit 的等价替换件。

两层用途：
  1. 同名遮蔽——本文件所在目录排进 sys_path 最前面时，模型毛坯的
     `from nonexistent_kit import mystery_act` 解析到本模块；
  2. 被 capture_driver 的 spec.shims 引用——按 class 实例化后挂树
     （mount_name）并重绑（rebind）到入口模块的 mystery_act 名字上。

数学定义（对拍基准）：mystery_act ≡ GELU（按原 kernel 语义假设），
数值纪律：内部计算升 float32，输出转回输入 dtype。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MysteryActShim(nn.Module):
    """mystery_act 的纯 torch 等价实现（fp32 计算、dtype 回转）。"""

    def forward(self, x):
        in_dtype = x.dtype
        return F.gelu(x.to(torch.float32)).to(in_dtype)


# 模块级实例：遮蔽导入路径时，模型 import 到的即是它
mystery_act = MysteryActShim()
