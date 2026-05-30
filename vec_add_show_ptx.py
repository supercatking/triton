import torch
import triton
import triton.language as tl


@triton.jit
def vec_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main():
    n_elements = 1024
    block_size = 256
    x = torch.arange(n_elements, device="cuda", dtype=torch.float32)
    y = torch.full((n_elements,), 2.0, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, block_size),)
    vec_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, x + y)
    print("vec_add OK")
    print("first 8:", out[:8].cpu().tolist())

    print("\n=== PTX ===")
    cache_tuple = vec_add_kernel.device_caches[0]
    compiled_kernel = next(iter(cache_tuple[0].values()))
    print(compiled_kernel.asm["ptx"])


if __name__ == "__main__":
    main()
