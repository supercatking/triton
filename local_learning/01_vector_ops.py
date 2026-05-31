import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(x, out, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(out + offsets, tl.load(x + offsets, mask=mask), mask=mask)


@triton.jit
def scale_kernel(x, out, scale: tl.constexpr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask)
    tl.store(out + offsets, values * scale, mask=mask)


@triton.jit
def axpy_kernel(x, y, out, scale: tl.constexpr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    xv = tl.load(x + offsets, mask=mask)
    yv = tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, xv * scale + yv, mask=mask)


@triton.jit
def relu_kernel(x, out, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask)
    tl.store(out + offsets, tl.maximum(values, 0.0), mask=mask)


@triton.jit
def square_sum_kernel(x, y, out, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    xv = tl.load(x + offsets, mask=mask)
    yv = tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, xv * xv + yv * yv, mask=mask)


def bench_ms(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def bandwidth_gbps(byte_count: int, ms: float) -> float:
    return byte_count / (ms * 1.0e-3) / 1.0e9


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2.0e-2, atol=2.0e-2)
    else:
        torch.testing.assert_close(actual, expected)


def run_case(n_elements: int, dtype: torch.dtype, block: int) -> None:
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    y = torch.randn(n_elements, device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, block),)
    scale = 1.25

    cases = [
        ("copy", lambda: copy_kernel[grid](x, out, n_elements, BLOCK=block), lambda: x, 2),
        ("scale", lambda: scale_kernel[grid](x, out, scale, n_elements, BLOCK=block), lambda: x * scale, 2),
        ("axpy", lambda: axpy_kernel[grid](x, y, out, scale, n_elements, BLOCK=block), lambda: x * scale + y, 3),
        ("relu", lambda: relu_kernel[grid](x, out, n_elements, BLOCK=block), lambda: torch.relu(x), 2),
        ("square_sum", lambda: square_sum_kernel[grid](x, y, out, n_elements, BLOCK=block), lambda: x * x + y * y, 3),
    ]

    print(f"\nshape=({n_elements},) dtype={dtype} block={block}")
    print(f"{'kernel':<12} {'triton_ms':>10} {'torch_ms':>10} {'GB/s':>10}")
    for name, triton_call, torch_ref, tensors_touched in cases:
        triton_call()
        torch.cuda.synchronize()
        assert_close(out, torch_ref())
        triton_ms = bench_ms(triton_call)
        torch_ms = bench_ms(lambda ref=torch_ref: ref())
        bytes_moved = n_elements * x.element_size() * tensors_touched
        print(f"{name:<12} {triton_ms:10.4f} {torch_ms:10.4f} {bandwidth_gbps(bytes_moved, triton_ms):10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run larger benchmark shapes")
    args = parser.parse_args()

    sizes = [1 << 20, 16 * (1 << 20)] if args.full else [1 << 20]
    for n_elements in sizes:
        for dtype in (torch.float32, torch.float16):
            run_case(n_elements, dtype, block=256)


if __name__ == "__main__":
    main()
