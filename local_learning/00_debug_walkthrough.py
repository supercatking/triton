import os

import torch
import triton
import triton.language as tl


@triton.jit
def debug_add_kernel(x, y, out, n_elements, BLOCK: tl.constexpr, DEBUG_KERNEL: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    # Use the "interpreter kernel debug" launch config before enabling this.
    if DEBUG_KERNEL:
        import pdb
        pdb.set_trace()

    xv = tl.load(x + offsets, mask=mask, other=0.0)
    yv = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, xv + yv, mask=mask)


def main() -> None:
    torch.manual_seed(0)
    n_elements = 16
    block = 8
    x = torch.arange(n_elements, device="cuda", dtype=torch.float32)
    y = torch.full((n_elements,), 2.0, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, block),)
    debug_kernel = os.environ.get("TRITON_KERNEL_PDB", "0") == "1"

    # Put a VSCode breakpoint on the next line first. Inspect x, y, out, grid,
    # block, and debug_kernel before stepping over the Triton launch.
    debug_add_kernel[grid](x, y, out, n_elements, BLOCK=block, DEBUG_KERNEL=debug_kernel)
    torch.cuda.synchronize()

    expected = x + y
    torch.testing.assert_close(out, expected)
    print("debug walkthrough passed")
    print("x        =", x.cpu().tolist())
    print("y        =", y.cpu().tolist())
    print("out      =", out.cpu().tolist())
    print("expected =", expected.cpu().tolist())


if __name__ == "__main__":
    main()
