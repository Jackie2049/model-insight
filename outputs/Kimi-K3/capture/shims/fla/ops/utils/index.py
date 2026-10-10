# fla.ops.utils.index 兼容 shim：mask → cu_seqlens 工具
import torch


def prepare_lens_from_mask(mask):
    return mask.sum(dim=-1).to(torch.long)


def prepare_cu_seqlens_from_mask(mask):
    lens = prepare_lens_from_mask(mask)
    zeros = lens.new_zeros(1)
    return torch.cat([zeros, lens.cumsum(dim=0)]).to(torch.int32)


def prepare_position_ids_from_cu_seqlens(cu_seqlens):
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    out = torch.cat([torch.arange(n, device=cu_seqlens.device) for n in lens])
    return out
