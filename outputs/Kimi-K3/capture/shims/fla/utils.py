# fla.utils 兼容 shim：tensor_cache 装饰器（本环境仅作恒等直通）
import functools


def tensor_cache(fn):
    """fla 的 @tensor_cache：按张量形状缓存中间结果。
    本 shim 不做缓存（分析场景输入不重复），仅保持装饰器契约。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper
