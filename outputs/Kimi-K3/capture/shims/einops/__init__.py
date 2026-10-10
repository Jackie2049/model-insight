# einops 兼容 shim：仅实现 rearrange（本毛坯唯一用到的接口）
from .rearrange import rearrange

__all__ = ["rearrange"]
