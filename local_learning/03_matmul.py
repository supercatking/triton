import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a,
    b,
    c,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, k, BLOCK_K):
        k_mask = k_start + offs_k
        av = tl.load(a_ptrs, mask=(offs_m[:, None] < m) & (k_mask[None, :] < k), other=0.0)
        bv = tl.load(b_ptrs, mask=(k_mask[:, None] < k) & (offs_n[None, :] < n), other=0.0)
        acc += tl.dot(av, bv, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


def triton_matmul(a: torch.Tensor, b: torch.Tensor, block_m: int, block_n: int, block_k: int, warps: int, stages: int):
    m, k = a.shape
    kb, n = b.shape
    assert k == kb
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    return c


def bench_ms(fn) -> float:
    return triton.testing.do_bench(fn, warmup=10, rep=50)


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return 2.0 * m * n * k / (ms * 1.0e-3) / 1.0e12


def run_shape(m: int, n: int, k: int, configs: list[tuple[int, int, int, int, int]]) -> None:
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    reference = a @ b

    print(f"\nmatmul: ({m}, {k}) x ({k}, {n}) -> ({m}, {n})")
    print(f"{'BM':>4} {'BN':>4} {'BK':>4} {'warps':>5} {'stages':>6} {'ms':>10} {'TFLOPS':>10}")
    for block_m, block_n, block_k, warps, stages in configs:
        result = triton_matmul(a, b, block_m, block_n, block_k, warps, stages)
        torch.cuda.synchronize()
        torch.testing.assert_close(result, reference, rtol=1.0e-2, atol=1.0e-2)
        ms = bench_ms(lambda: triton_matmul(a, b, block_m, block_n, block_k, warps, stages))
        print(f"{block_m:4d} {block_n:4d} {block_k:4d} {warps:5d} {stages:6d} {ms:10.4f} {tflops(m, n, k, ms):10.1f}")

    torch_ms = bench_ms(lambda: a @ b)
    print(f"{'torch':>27} {torch_ms:10.4f} {tflops(m, n, k, torch_ms):10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="include 4096 and rectangular shapes")
    args = parser.parse_args()

    configs = [
        (16, 32, 32, 4, 4),
        (32, 32, 32, 4, 4),
        (32, 64, 32, 4, 4),
    ]
    shapes = [(512, 512, 512)]
    if args.full:
        shapes = [(1024, 1024, 1024), (4096, 4096, 4096), (2048, 4096, 1024), (4096, 1024, 2048)]
        configs = [
            (32, 64, 64, 4, 4),
            (64, 64, 64, 4, 4),
            (64, 128, 64, 8, 4),
        ]

    for shape in shapes:
        run_shape(*shape, configs=configs)


if __name__ == "__main__":
    main()
