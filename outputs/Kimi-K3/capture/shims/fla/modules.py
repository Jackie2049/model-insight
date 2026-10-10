# fla.modules 兼容 shim：FusedRMSNormGated / ShortConvolution（纯 torch 数学等价）
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedRMSNormGated(nn.Module):
    """数学定义（与 fla.modules.FusedRMSNormGated 对齐）：
        y = rmsnorm(x) * act(o)
        rmsnorm(x) = x / sqrt(mean(x^2) + eps) * weight
        act='sigmoid' 时 act(o) = sigmoid(o)
    内部升 float32 计算，输出转回 x.dtype。
    """

    def __init__(self, hidden_size, eps=1e-6, activation="sigmoid"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, x, o):
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        xf = xf * self.weight.float()
        if self.activation == "sigmoid":
            g = torch.sigmoid(o.float())
        elif self.activation in (None, "silu", "swish"):
            g = F.silu(o.float())
        else:
            raise NotImplementedError(self.activation)
        return (xf * g).to(dt)


class ShortConvolution(nn.Module):
    """数学定义（与 fla.modules.ShortConvolution 对齐）：
        因果深度可分离一维卷积（每通道独立, kernel=K）+ 激活。
        y[:, t] = act(sum_{k=0..K-1} w[:, k] * x_pad[:, t + k])   (x_pad 左侧补 cache/零)
        cache: (B, D, K-1) —— 上一步的原始输入（未激活）尾部状态
    """
    def __init__(self, hidden_size, kernel_size=4, activation="silu", bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size,
            kernel_size=kernel_size, groups=hidden_size, bias=bias,
        )

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, activation=None):
        B, L, D = x.shape
        act = activation or self.activation
        if cu_seqlens is not None:
            # 变长：逐段因果卷积（x 已是 (1, Ttot, D) 打包形态）
            xs = x.squeeze(0)
            outs, states = [], []
            start = 0
            new_cache = []
            for i in range(1, cu_seqlens.numel()):
                end = int(cu_seqlens[i])
                seg = xs[start:end]
                segs = seg.transpose(0, 1).unsqueeze(0)  # (1, D, Tseg)
                if cache is not None:
                    pad = cache[:, :, : self.kernel_size - 1]
                else:
                    pad = segs.new_zeros(B, D, self.kernel_size - 1)
                y = F.conv1d(torch.cat([pad, segs], dim=-1),
                             self.conv1d.weight, self.conv1d.bias,
                             groups=self.hidden_size)
                outs.append(y[0].transpose(0, 1))
                st = torch.cat([pad, segs], dim=-1)[:, :, -(self.kernel_size - 1):]
                states.append(st[0])   # (D, K-1)，各段堆叠为 (num_seq, D, K-1)
                start = end
            y = torch.cat(outs, dim=0).unsqueeze(0)
            final = torch.stack(states, dim=0) if output_final_state else None
        else:
            xu = x.transpose(1, 2)  # (B, D, L)
            if cache is not None:
                pad = cache
            else:
                pad = xu.new_zeros(B, D, self.kernel_size - 1)
            cat = torch.cat([pad, xu], dim=-1)  # (B, D, K-1+L)
            y = F.conv1d(cat, self.conv1d.weight, self.conv1d.bias, groups=D)
            y = y.transpose(1, 2)  # (B, L, D)
            final = cat[:, :, -(self.kernel_size - 1):].contiguous() if output_final_state else None
        if act == "silu" or act == "swish":
            y = F.silu(y)
        elif act is not None:
            raise NotImplementedError(act)
        return y, final
