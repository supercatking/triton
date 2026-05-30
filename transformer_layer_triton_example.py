import math

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x, weight, out, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    x_row = tl.load(x + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x_row * x_row, axis=0) / n_cols
    y = x_row * tl.rsqrt(var + eps) * w
    tl.store(out + row * n_cols + offsets, y, mask=mask)


@triton.jit
def residual_add_kernel(a, b, out, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    av = tl.load(a + offsets, mask=mask)
    bv = tl.load(b + offsets, mask=mask)
    tl.store(out + offsets, av + bv, mask=mask)


@triton.jit
def causal_attention_kernel(
    q,
    k,
    v,
    out,
    stride_bh,
    stride_t,
    n_ctx: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    bh = tl.program_id(0)
    q_block = tl.program_id(1)
    q_offsets = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, head_dim)

    q_ptrs = q + bh * stride_bh + q_offsets[:, None] * stride_t + d_offsets[None, :]
    qv = tl.load(q_ptrs, mask=q_offsets[:, None] < n_ctx, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, head_dim), tl.float32)
    scale = 1.4426950408889634 / tl.sqrt(head_dim + 0.0)

    for start_n in range(0, n_ctx, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k + bh * stride_bh + n_offsets[None, :] * stride_t + d_offsets[:, None]
        kv = tl.load(k_ptrs, mask=n_offsets[None, :] < n_ctx, other=0.0)
        scores = tl.dot(qv, kv) * scale
        causal_mask = n_offsets[None, :] <= q_offsets[:, None]
        scores = tl.where(causal_mask, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp2(scores - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v + bh * stride_bh + n_offsets[:, None] * stride_t + d_offsets[None, :]
        vv = tl.load(v_ptrs, mask=n_offsets[:, None] < n_ctx, other=0.0)
        acc += tl.dot(p.to(vv.dtype), vv)
        m_i = m_ij

    acc = acc / l_i[:, None]
    out_ptrs = out + bh * stride_bh + q_offsets[:, None] * stride_t + d_offsets[None, :]
    tl.store(out_ptrs, acc, mask=q_offsets[:, None] < n_ctx)


@triton.jit
def swiglu_kernel(gate, up, out, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(gate + offsets, mask=mask).to(tl.float32)
    u = tl.load(up + offsets, mask=mask).to(tl.float32)
    silu = g / (1.0 + tl.exp(-g))
    tl.store(out + offsets, silu * u, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    out = torch.empty_like(x)
    rows, cols = x.numel() // x.shape[-1], x.shape[-1]
    block = triton.next_power_of_2(cols)
    rmsnorm_kernel[(rows,)](x, weight, out, cols, eps, BLOCK=block)
    return out


def residual_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    n = a.numel()
    residual_add_kernel[(triton.cdiv(n, 256),)](a, b, out, n, BLOCK=256)
    return out


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, heads, n_ctx, head_dim = q.shape
    qf = q.reshape(batch * heads, n_ctx, head_dim).contiguous()
    kf = k.reshape(batch * heads, n_ctx, head_dim).contiguous()
    vf = v.reshape(batch * heads, n_ctx, head_dim).contiguous()
    out = torch.empty_like(qf)
    causal_attention_kernel[(batch * heads, triton.cdiv(n_ctx, 16))](
        qf,
        kf,
        vf,
        out,
        n_ctx * head_dim,
        head_dim,
        n_ctx,
        head_dim,
        BLOCK_M=16,
        BLOCK_N=32,
    )
    return out.reshape(batch, heads, n_ctx, head_dim)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(gate)
    n = gate.numel()
    swiglu_kernel[(triton.cdiv(n, 256),)](gate, up, out, n, BLOCK=256)
    return out


def triton_transformer_layer(x, params, n_heads):
    batch, n_ctx, d_model = x.shape
    head_dim = d_model // n_heads

    h = rmsnorm(x, params["attn_norm"])
    qkv = h.reshape(batch * n_ctx, d_model) @ params["w_qkv"]
    q, k, v = qkv.reshape(batch, n_ctx, 3, n_heads, head_dim).unbind(dim=2)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    attn = causal_attention(q, k, v)
    attn = attn.transpose(1, 2).reshape(batch * n_ctx, d_model)
    attn_out = (attn @ params["w_o"]).reshape(batch, n_ctx, d_model)
    x = residual_add(x, attn_out)

    h = rmsnorm(x, params["mlp_norm"])
    mlp_in = h.reshape(batch * n_ctx, d_model)
    gate = mlp_in @ params["w_gate"]
    up = mlp_in @ params["w_up"]
    hidden = swiglu(gate, up)
    mlp_out = (hidden @ params["w_down"]).reshape(batch, n_ctx, d_model)
    return residual_add(x, mlp_out)


def torch_reference_layer(x, params, n_heads):
    batch, n_ctx, d_model = x.shape
    head_dim = d_model // n_heads

    h = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * params["attn_norm"]
    qkv = h.reshape(batch * n_ctx, d_model) @ params["w_qkv"]
    q, k, v = qkv.reshape(batch, n_ctx, 3, n_heads, head_dim).unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    causal = torch.tril(torch.ones(n_ctx, n_ctx, device=x.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    attn = torch.softmax(scores, dim=-1) @ v
    attn = attn.transpose(1, 2).reshape(batch * n_ctx, d_model)
    x = x + (attn @ params["w_o"]).reshape(batch, n_ctx, d_model)

    h = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * params["mlp_norm"]
    mlp_in = h.reshape(batch * n_ctx, d_model)
    hidden = torch.nn.functional.silu(mlp_in @ params["w_gate"]) * (mlp_in @ params["w_up"])
    return x + (hidden @ params["w_down"]).reshape(batch, n_ctx, d_model)


def main() -> None:
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    batch, n_ctx, d_model, n_heads, d_ff = 2, 64, 128, 4, 256

    scale = 0.02
    params = {
        "attn_norm": torch.randn(d_model, device=device, dtype=dtype),
        "mlp_norm": torch.randn(d_model, device=device, dtype=dtype),
        "w_qkv": torch.randn(d_model, 3 * d_model, device=device, dtype=dtype) * scale,
        "w_o": torch.randn(d_model, d_model, device=device, dtype=dtype) * scale,
        "w_gate": torch.randn(d_model, d_ff, device=device, dtype=dtype) * scale,
        "w_up": torch.randn(d_model, d_ff, device=device, dtype=dtype) * scale,
        "w_down": torch.randn(d_ff, d_model, device=device, dtype=dtype) * scale,
    }
    x = torch.randn(batch, n_ctx, d_model, device=device, dtype=dtype)

    y_triton = triton_transformer_layer(x, params, n_heads)
    y_ref = torch_reference_layer(x, params, n_heads)
    torch.cuda.synchronize()

    max_diff = (y_triton - y_ref).abs().max().item()
    torch.testing.assert_close(y_triton, y_ref, rtol=2e-2, atol=2e-2)
    print("One-layer transformer Triton example passed")
    print(f"shape: batch={batch}, seq={n_ctx}, d_model={d_model}, heads={n_heads}, d_ff={d_ff}")
    print("kernels: RMSNorm, causal attention, residual add, SwiGLU")
    print(f"max_abs_diff={max_diff:.6f}")


if __name__ == "__main__":
    main()
