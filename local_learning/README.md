# Local Triton Learning Track

This directory turns the two-week learning plan into runnable exercises on this
checkout. Run everything from the repository root after activating the existing
virtual environment:

```bash
cd /home/zyz/triton
source .venv/bin/activate
git switch zdev
```

The scripts are intentionally standalone. Each one contains a PyTorch reference,
Triton kernels, correctness checks, and a small benchmark table.

## Suggested Order

1. `python local_learning/01_vector_ops.py`
   - Elementwise kernels: copy, scale, AXPY, ReLU, square-sum.
2. `python local_learning/02_reductions.py`
   - Row reductions and a simple RMSNorm.
3. `python local_learning/03_matmul.py`
   - Minimal fp16 matmul plus first tuning sweep.
4. `python local_learning/04_dl_kernels.py`
   - RMSNorm, softmax, causal softmax, causal attention, SwiGLU fusion.
5. `python local_learning/05_fused_project.py`
   - Mini final project: fused RMSNorm + SwiGLU + residual.

Use `--full` for larger benchmark shapes once the quick runs pass:

```bash
python local_learning/03_matmul.py --full
python local_learning/04_dl_kernels.py --full
```

## Reading Checklist

For each kernel, identify:

- Input shape, dtype, and contiguity assumptions.
- Grid mapping from program ids to data tiles.
- Masking for boundary handling.
- Accumulator dtype.
- Meta-parameters such as block sizes, `num_warps`, and `num_stages`.
- Benchmark baseline and the reported bandwidth or TFLOPS.
