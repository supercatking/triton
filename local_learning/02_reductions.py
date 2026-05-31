import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(x, out, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + row, tl.sum(values, axis=0))


@triton.jit
def row_max_kernel(x, out, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    tl.store(out + row, tl.max(values, axis=0))


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


def bench_ms(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def bandwidth_gbps(byte_count: int, ms: float) -> float:
    return byte_count / (ms * 1.0e-3) / 1.0e9


def run_case(rows: int, cols: int, dtype: torch.dtype) -> None:
    x = torch.randn((rows, cols), device="cuda", dtype=dtype)
    weight = torch.randn((cols,), device="cuda", dtype=dtype)
    block = triton.next_power_of_2(cols)
    row_out = torch.empty((rows,), device="cuda", dtype=torch.float32)
    norm_out = torch.empty_like(x)

    cases = [
        (
            "row_sum",
            lambda: row_sum_kernel[(rows,)](x, row_out, cols, BLOCK=block),
            lambda: x.float().sum(dim=1),
            rows * cols * x.element_size() + rows * row_out.element_size(),
            row_out,
        ),
        (
            "row_max",
            lambda: row_max_kernel[(rows,)](x, row_out, cols, BLOCK=block),
            lambda: x.float().max(dim=1).values,
            rows * cols * x.element_size() + rows * row_out.element_size(),
            row_out,
        ),
        (
            "rmsnorm",
            lambda: rmsnorm_kernel[(rows,)](x, weight, norm_out, cols, 1.0e-5, BLOCK=block),
            lambda: x.float() * torch.rsqrt(x.float().pow(2).mean(dim=1, keepdim=True) + 1.0e-5) * weight.float(),
            rows * cols * x.element_size() * 2 + cols * weight.element_size(),
            norm_out,
        ),
    ]

    print(f"\nshape=({rows}, {cols}) dtype={dtype} block={block}")
    print(f"{'kernel':<10} {'triton_ms':>10} {'torch_ms':>10} {'GB/s':>10}")
    for name, triton_call, torch_ref, bytes_moved, result in cases:
        triton_call()
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), torch_ref().float(), rtol=1.0e-2, atol=1.0e-2)
        triton_ms = bench_ms(triton_call)
        torch_ms = bench_ms(lambda ref=torch_ref: ref())
        print(f"{name:<10} {triton_ms:10.4f} {torch_ms:10.4f} {bandwidth_gbps(bytes_moved, triton_ms):10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run more hidden sizes")
    args = parser.parse_args()

    shapes = [(512, 1024)]
    if args.full:
        shapes = [(1024, 128), (1024, 1024), (1024, 4096), (512, 8192)]
    for rows, cols in shapes:
        for dtype in (torch.float32, torch.float16):
            run_case(rows, cols, dtype)


if __name__ == "__main__":
    main()
