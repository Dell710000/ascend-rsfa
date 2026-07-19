# Contributing

ASCEND-RSFA welcomes reproducibility fixes, new validated shapes, kernel
optimizations, documentation improvements, and Ascend platform reports.

## Before opening a pull request

1. Keep the optimized kernel API compatible with
   `attention(q, k, v, causal, sm_scale, BLOCK_M, BLOCK_N)`.
2. Add or update correctness coverage for every new shape or data type.
3. Report the exact NPU, CANN, torch_npu, TritonAscend, warmup, and iteration
   configuration for performance claims.
4. Include both raw timings and a comparison against the repository baseline.
5. Do not commit generated `msprof` directories or machine-local settings.

Run the repository checks before submitting:

```bash
python .github/scripts/validate_repo.py
python benchmarks/run_correctness_test.py
```

The second command requires a supported Ascend NPU environment. If it cannot
be run, state that clearly in the pull request.

## Benchmark changes

Performance results are sensitive to tiling, compiler versions, clock state,
and synchronization. Use the existing benchmark scripts, keep warmup and timed
iterations visible, and avoid comparing numbers collected with different
software stacks without labeling them.
