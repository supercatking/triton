import argparse
import math

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x, weight, out, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(values * values, axis=0) / n_cols
    y = values * tl.rsqrt(var + eps) * w
    tl.store(out + row * n_cols + offsets, y, mask=mask)


@triton.jit
def row_softmax_kernel(x, out, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(out + row * n_cols + offsets, numerator / denominator, mask=mask)


@triton.jit
def causal_softmax_kernel(x, out, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    causal = offsets <= row
    mask = (offsets < n_cols) & causal
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(out + row * n_cols + offsets, numerator / denominator, mask=mask)
    tl.store(out + row * n_cols + offsets, 0.0, mask=(offsets < n_cols) & ~causal)


@triton.jit
def causal_attention_kernel(
    q,
    k,
    v,
    out,
    stride_bh: tl.constexpr,
    stride_t: tl.constexpr,
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
        scores = tl.where(n_offsets[None, :] <= q_offsets[:, None], scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp2(scores - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v + bh * stride_bh + n_offsets[:, None] * stride_t + d_offsets[None, :]
        vv = tl.load(v_ptrs, mask=n_offsets[:, None] < n_ctx, other=0.0)
        acc += tl.dot(p.to(vv.dtype), vv)
        m_i = m_ij

    out_ptrs = out + bh * stride_bh + q_offsets[:, None] * stride_t + d_offsets[None, :]
    tl.store(out_ptrs, acc / l_i[:, None], mask=q_offsets[:, None] < n_ctx)


@triton.jit
def swiglu_bias_kernel(gate, up, bias_gate, bias_up, out, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(gate + offsets, mask=mask).to(tl.float32)
    u = tl.load(up + offsets, mask=mask).to(tl.float32)
    bg = tl.load(bias_gate + offsets, mask=mask).to(tl.float32)
    bu = tl.load(bias_up + offsets, mask=mask).to(tl.float32)
    gb = g + bg
    silu = gb / (1.0 + tl.exp(-gb))
    tl.store(out + offsets, silu * (u + bu), mask=mask)


def bench_ms(fn) -> float:
    return triton.testing.do_bench(fn, warmup=20, rep=80)


def bandwidth_gbps(byte_count: int, ms: float) -> float:
    return byte_count / (ms * 1.0e-3) / 1.0e9


def run_rmsnorm(rows: int, cols: int, dtype: torch.dtype) -> None:
    x = torch.randn((rows, cols), device="cuda", dtype=dtype)
    weight = torch.randn((cols,), device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    call = lambda: rmsnorm_kernel[(rows,)](x, weight, out, cols, 1.0e-5, BLOCK=block)
    ref = lambda: x.float() * torch.rsqrt(x.float().pow(2).mean(dim=1, keepdim=True) + 1.0e-5) * weight.float()
    call()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref(), rtol=1.0e-2, atol=1.0e-2)
    ms = bench_ms(call)
    bytes_moved = rows * cols * x.element_size() * 2 + cols * weight.element_size()
    print(f"rmsnorm rows={rows:<4d} cols={cols:<5d} dtype={str(dtype):<13s} {ms:8.4f} ms {bandwidth_gbps(bytes_moved, ms):8.1f} GB/s")


def run_softmax(rows: int, cols: int, dtype: torch.dtype) -> None:
    x = torch.randn((rows, cols), device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    call = lambda: row_softmax_kernel[(rows,)](x, out, cols, BLOCK=block)
    call()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), torch.softmax(x.float(), dim=1), rtol=1.0e-2, atol=1.0e-2)
    ms = bench_ms(call)
    bytes_moved = rows * cols * x.element_size() * 2
    print(f"softmax rows={rows:<4d} cols={cols:<5d} dtype={str(dtype):<13s} {ms:8.4f} ms {bandwidth_gbps(bytes_moved, ms):8.1f} GB/s")


def run_causal_softmax(n_ctx: int, dtype: torch.dtype) -> None:
    x = torch.randn((n_ctx, n_ctx), device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_ctx)
    call = lambda: causal_softmax_kernel[(n_ctx,)](x, out, n_ctx, BLOCK=block)
    mask = torch.tril(torch.ones((n_ctx, n_ctx), device="cuda", dtype=torch.bool))
    ref = torch.softmax(x.float().masked_fill(~mask, float("-inf")), dim=1).masked_fill(~mask, 0)
    call()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref, rtol=1.0e-2, atol=1.0e-2)
    ms = bench_ms(call)
    bytes_moved = n_ctx * n_ctx * x.element_size() * 2
    print(f"causal_softmax n_ctx={n_ctx:<4d} dtype={str(dtype):<13s} {ms:8.4f} ms {bandwidth_gbps(bytes_moved, ms):8.1f} GB/s")


def run_attention(batch: int, heads: int, n_ctx: int, head_dim: int, dtype: torch.dtype) -> None:
    q = torch.randn((batch, heads, n_ctx, head_dim), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    qf = q.reshape(batch * heads, n_ctx, head_dim).contiguous()
    kf = k.reshape(batch * heads, n_ctx, head_dim).contiguous()
    vf = v.reshape(batch * heads, n_ctx, head_dim).contiguous()
    out = torch.empty_like(qf)
    grid = (batch * heads, triton.cdiv(n_ctx, 16))
    call = lambda: causal_attention_kernel[grid](
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
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    mask = torch.tril(torch.ones((n_ctx, n_ctx), device="cuda", dtype=torch.bool))
    ref = torch.softmax(scores.float().masked_fill(~mask, float("-inf")), dim=-1) @ v.float()
    call()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.reshape(batch, heads, n_ctx, head_dim).float(), ref, rtol=2.0e-2, atol=2.0e-2)
    ms = bench_ms(call)
    flops = 4.0 * batch * heads * n_ctx * n_ctx * head_dim / (ms * 1.0e-3) / 1.0e12
    print(f"attention b={batch} h={heads} n={n_ctx:<4d} d={head_dim:<3d} dtype={str(dtype):<13s} {ms:8.4f} ms {flops:8.2f} TFLOPS")


def run_swiglu(size: int, dtype: torch.dtype) -> None:
    gate = torch.randn((size,), device="cuda", dtype=dtype)
    up = torch.randn_like(gate)
    bias_gate = torch.randn_like(gate)
    bias_up = torch.randn_like(gate)
    out = torch.empty_like(gate)
    block = 256
    grid = (triton.cdiv(size, block),)
    call = lambda: swiglu_bias_kernel[grid](gate, up, bias_gate, bias_up, out, size, BLOCK=block)
    ref = lambda: torch.nn.functional.silu((gate + bias_gate).float()) * (up + bias_up).float()
    call()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref(), rtol=1.0e-2, atol=1.0e-2)
    ms = bench_ms(call)
    bytes_moved = size * gate.element_size() * 5
    print(f"swiglu size={size:<9d} dtype={str(dtype):<13s} {ms:8.4f} ms {bandwidth_gbps(bytes_moved, ms):8.1f} GB/s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run larger benchmark shapes")
    args = parser.parse_args()

    dtypes = (torch.float16, torch.bfloat16)
    hidden_sizes = [1024] if not args.full else [128, 1024, 4096, 8192]
    for dtype in dtypes:
        for hidden in hidden_sizes:
            run_rmsnorm(rows=512, cols=hidden, dtype=dtype)
        run_softmax(rows=512, cols=1024 if not args.full else 2048, dtype=dtype)
        run_causal_softmax(n_ctx=128 if not args.full else 512, dtype=dtype)
        run_attention(batch=2, heads=4, n_ctx=64 if not args.full else 128, head_dim=64, dtype=dtype)
        run_swiglu(size=1 << (20 if not args.full else 24), dtype=dtype)


if __name__ == "__main__":
    main()
