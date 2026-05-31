import os

import torch
import triton
import triton.language as tl


@triton.jit
def axpy_kernel(
    x,
    y,
    out,
    scale: tl.constexpr,
    n_elements,
    BLOCK: tl.constexpr,
    DEBUG_KERNEL: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    # Use the "Triton: axpy kernel - interpreter pdb" launch config before
    # enabling this. In normal GPU mode, Triton kernels cannot be single-stepped
    # by the Python debugger because they are compiled and executed on the GPU.
    if DEBUG_KERNEL:
        import pdb
        pdb.set_trace()

    xv = tl.load(x + offsets, mask=mask, other=0.0)
    yv = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, xv * scale + yv, mask=mask)


def main() -> None:
    torch.manual_seed(0)
    n_elements = 20
    block = 8
    scale = 1.25

    x = torch.arange(n_elements, device="cuda", dtype=torch.float32)
    y = torch.full((n_elements,), 2.0, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, block),)
    debug_kernel = os.environ.get("TRITON_KERNEL_PDB", "0") == "1"

    # Put a VSCode breakpoint here first. Step into this line to inspect Triton
    # runtime internals, or step over it to inspect out after synchronization.
    axpy_kernel[grid](x, y, out, scale, n_elements, BLOCK=block, DEBUG_KERNEL=debug_kernel)
    torch.cuda.synchronize()

    expected = x * scale + y
    torch.testing.assert_close(out, expected)
    print("axpy debug passed")
    print("grid     =", grid)
    print("x        =", x.cpu().tolist())
    print("y        =", y.cpu().tolist())
    print("out      =", out.cpu().tolist())
    print("expected =", expected.cpu().tolist())


if __name__ == "__main__":
    main()
