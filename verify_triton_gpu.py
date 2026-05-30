import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main() -> None:
    n_elements = 1024 * 1024
    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=256)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, x + y)
    print("Triton GPU vector-add verification passed")
    print(f"torch={torch.__version__}")
    print(f"triton={triton.__version__}")
    print(f"device={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
