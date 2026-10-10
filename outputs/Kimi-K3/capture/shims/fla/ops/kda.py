# fla.ops.kda 兼容 shim：chunk_kda / fused_recurrent_kda
# 数学定义（Kimi Delta Attention · 门控 DeltaNet，细粒度逐通道门控）：
#   对每个 head h、每个 token t（S ∈ R^{K×V} 为该 head 状态）：
#     q_t = l2norm(q_t), k_t = l2norm(k_t)                     (use_qk_l2norm_in_kernel)
#     beta_t = sigmoid(beta_raw_t)                             (use_beta_sigmoid_in_kernel)
#     g_t = -exp(A_log_h) * softplus(g_raw_t + dt_bias)        (use_gate_in_kernel)
#     g_t = clamp(g_t, min=lower_bound)                        (safe_gate=True 且给 lower_bound)
#     D_t = exp(g_t)                                           (K 维逐元素衰减, 形状 (K,))
#     S'    = D_t[:, None] * S_{t-1}
#     u_t   = k_t · S'                                          (检索用衰减后状态)
#     S_t   = S' + beta_t * k_t ⊗ (v_t - u_t)
#     o_t   = S_t^T q_t
#   数值纪律：q/k 归一化与门控在 float32 计算；状态 float32 累积；输出/状态转回输入 dtype。
# 与 fla 原生 Triton 核的差异（诚实声明）：实现为逐步循环而非分块并行；
# 状态布局固定 (B, H, K, V)（transpose_state_layout 参数记录但不改变本 shim 行为）。
import torch
import torch.nn.functional as F


def _l2norm(x):
    xf = x.float()
    return xf / xf.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def _kda_recurrence(q, k, v, g, beta, A_log, dt_bias, initial_state,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True, safe_gate=False,
                    lower_bound=None):
    """q,k: (B,T,H,K)  v: (B,T,H,V)  g_raw: (B,T,H,K)  beta_raw: (B,T,H)
    A_log: (H,)  dt_bias: (H*K,)  initial_state: (B,H,K,V) | None"""
    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta.float())
    else:
        beta = beta.float()
    if use_gate_in_kernel:
        gf = g.float() + dt_bias.float().view(H, K)
        gf = -torch.exp(A_log.float()).view(H, 1) * F.softplus(gf)
        if safe_gate and lower_bound is not None:
            gf = torch.clamp(gf, min=float(lower_bound))
    else:
        gf = g.float()
    decay = torch.exp(gf)  # (B,T,H,K)

    state = initial_state.float() if initial_state is not None else \
        q.new_zeros(B, H, K, V, dtype=torch.float32)
    outs = []
    for t in range(T):
        S = state * decay[:, t].unsqueeze(-1)              # (B,H,K,V)
        u = torch.einsum("bhk,bhkv->bhv", k[:, t].float(), S)   # (B,H,V)
        delta = beta[:, t].unsqueeze(-1) * (v[:, t].float() - u)
        state = S + k[:, t].float().unsqueeze(-1) * delta.unsqueeze(-2)
        o_t = torch.einsum("bhk,bhkv->bhv", q[:, t].float(), state)  # (B,H,V)
        outs.append(o_t)
    o = torch.stack(outs, dim=1).to(dtype)   # (B,T,H,V)
    return o, state.to(dtype)


def _kda_dispatch(q, k, v, g, beta, A_log, dt_bias, initial_state,
                  use_qk_l2norm_in_kernel, use_gate_in_kernel,
                  use_beta_sigmoid_in_kernel, safe_gate, lower_bound,
                  output_final_state, cu_seqlens):
    if cu_seqlens is not None:
        # 打包变长：逐序列递归，最终状态按 (num_seq, H, K, V) 堆叠
        outs, states = [], []
        n = cu_seqlens.numel() - 1
        for i in range(n):
            s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            init = None
            if initial_state is not None:
                init = initial_state[i].unsqueeze(0)
            o_i, st_i = _kda_recurrence(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e],
                                        beta[:, s:e], A_log, dt_bias, init,
                                        use_qk_l2norm_in_kernel, use_gate_in_kernel,
                                        use_beta_sigmoid_in_kernel, safe_gate, lower_bound)
            outs.append(o_i)          # (1, seg_len, H, V)
            states.append(st_i[0])    # (H, K, V)
        o = torch.cat(outs, dim=1)
        return o, (torch.stack(states, dim=0) if output_final_state else None)
    o, state = _kda_recurrence(q, k, v, g, beta, A_log, dt_bias, initial_state,
                               use_qk_l2norm_in_kernel, use_gate_in_kernel,
                               use_beta_sigmoid_in_kernel, safe_gate, lower_bound)
    return o, (state if output_final_state else None)


def chunk_kda(q=None, k=None, v=None, g=None, beta=None, A_log=None, dt_bias=None,
              initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
              use_beta_sigmoid_in_kernel=False, safe_gate=False, lower_bound=None,
              transpose_state_layout=False, cu_seqlens=None, **kwargs):
    return _kda_dispatch(q, k, v, g, beta, A_log, dt_bias, initial_state,
                         use_qk_l2norm_in_kernel, use_gate_in_kernel,
                         use_beta_sigmoid_in_kernel, safe_gate, lower_bound,
                         output_final_state, cu_seqlens)


def fused_recurrent_kda(q=None, k=None, v=None, g=None, beta=None, A_log=None, dt_bias=None,
                        initial_state=None, output_final_state=False,
                        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                        use_beta_sigmoid_in_kernel=False, safe_gate=False, lower_bound=None,
                        transpose_state_layout=False, cu_seqlens=None, **kwargs):
    return _kda_dispatch(q, k, v, g, beta, A_log, dt_bias, initial_state,
                         use_qk_l2norm_in_kernel, use_gate_in_kernel,
                         use_beta_sigmoid_in_kernel, safe_gate, lower_bound,
                         output_final_state, cu_seqlens)
