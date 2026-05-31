import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def fused_rmsnorm_swiglu_residual_kernel(
    x,
    residual,
    weight,
    gate,
    up,
    bias_gate,
    bias_up,
    out,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    base = row * n_cols + offsets

    xv = tl.load(x + base, mask=mask, other=0.0).to(tl.float32)
    rv = tl.load(residual + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    gv = tl.load(gate + base, mask=mask, other=0.0).to(tl.float32)
    uv = tl.load(up + base, mask=mask, other=0.0).to(tl.float32)
    bg = tl.load(bias_gate + offsets, mask=mask, other=0.0).to(tl.float32)
    bu = tl.load(bias_up + offsets, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(xv * xv, axis=0) / n_cols
    norm = xv * tl.rsqrt(var + eps) * w
    gate_value = norm * gv + bg
    up_value = norm * uv + bu
    silu = gate_value / (1.0 + tl.exp(-gate_value))
    tl.store(out + base, rv + silu * up_value, mask=mask)


def torch_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    bias_gate: torch.Tensor,
    bias_up: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    norm = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=1, keepdim=True) + eps) * weight.float()
    gate_value = norm * gate.float() + bias_gate.float()
    up_value = norm * up.float() + bias_up.float()
    return residual.float() + torch.nn.functional.silu(gate_value) * up_value


def triton_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    bias_gate: torch.Tensor,
    bias_up: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    fused_rmsnorm_swiglu_residual_kernel[(rows,)](
        x,
        residual,
        weight,
        gate,
        up,
        bias_gate,
        bias_up,
        out,
        cols,
        eps,
        BLOCK=block,
    )
    return out


def bench_ms(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def bandwidth_gbps(byte_count: int, ms: float) -> float:
    return byte_count / (ms * 1.0e-3) / 1.0e9


def run_case(rows: int, cols: int, dtype: torch.dtype) -> None:
    x = torch.randn((rows, cols), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    gate = torch.randn_like(x)
    up = torch.randn_like(x)
    weight = torch.randn((cols,), device="cuda", dtype=dtype)
    bias_gate = torch.randn((cols,), device="cuda", dtype=dtype)
    bias_up = torch.randn((cols,), device="cuda", dtype=dtype)
    eps = 1.0e-5

    out = triton_fused(x, residual, weight, gate, up, bias_gate, bias_up, eps)
    reference = torch_reference(x, residual, weight, gate, up, bias_gate, bias_up, eps)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), reference, rtol=2.0e-2, atol=2.0e-2)

    triton_ms = bench_ms(lambda: triton_fused(x, residual, weight, gate, up, bias_gate, bias_up, eps))
    torch_ms = bench_ms(lambda: torch_reference(x, residual, weight, gate, up, bias_gate, bias_up, eps))
    bytes_moved = rows * cols * x.element_size() * 6 + cols * x.element_size() * 3
    print(
        f"rows={rows:<5d} cols={cols:<5d} dtype={str(dtype):<13s} "
        f"triton={triton_ms:8.4f} ms torch={torch_ms:8.4f} ms "
        f"triton_bw={bandwidth_gbps(bytes_moved, triton_ms):8.1f} GB/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run larger hidden sizes")
    args = parser.parse_args()

    shapes = [(512, 1024)]
    if args.full:
        shapes = [(1024, 1024), (1024, 4096), (512, 8192)]

    for dtype in (torch.float16, torch.bfloat16):
        for rows, cols in shapes:
            run_case(rows, cols, dtype)


if __name__ == "__main__":
    main()
