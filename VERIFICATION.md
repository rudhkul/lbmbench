# Verification records

`./run_all.sh verify` compares every propagation pattern and layout with the
two-lattice pull (AB pull, SoA) after 51 steps on a 24^3 lattice, for every
number format and collision operator (34 configurations per pattern), and
compares the moment representation with the regularised population kernel.

## Results in this repository

| Pattern | RTX 5080 | A100 SXM4 40GB |
|---|---|---|
| AA | bitwise identical in 34/34 | bitwise identical in 34/34 |
| AB push | bitwise in 30/34; 4 differ (≤1.2e-7 FP32, ≤8.6e-6 16-bit) | bitwise identical in 34/34 |
| Esoteric Pull | bitwise in 15/34; 19 differ (≤2.2e-16 FP64/mixed, ≤1.2e-7 FP32, ≤7.3e-6 16-bit) | identical differences to the RTX 5080 |
| Moment representation vs. regularised | 2.2e-16 (FP64) | 2.2e-16 (FP64) |

The differences are at the level of one rounding of the storage format.
The Esoteric Pull differences are identical on both GPUs, so they are not
caused by architecture-specific code generation; they are consistent with a
different order of floating-point operations in that kernel.

## Note on `results/a100-sxm4-40gb/verify.log`

The A100 run (Google Colab) used an earlier revision of `lbmbench.cu` whose
verification step applied a single threshold of 1e-12 to all formats. With
that threshold, the 11 FP32 and 16-bit Esoteric Pull differences listed above
were printed as `CHECK`. The released revision uses format-dependent
thresholds (1e-12 FP64/mixed, 1e-6 FP32, 1e-4 16-bit), which classify them as
round-off; the RTX 5080 log was produced with the released revision.

The two revisions differ **only** in the verification threshold and its label
(`lbmbench.cu`, function running `--mode verify`); all computational kernels
used for the A100 performance, accuracy and energy measurements are identical
to the released source. The evidence is in `records/a100_colab_run/`:

- `lbmbench_colab_A100_run.ipynb`: the executed Colab notebook of the A100 run
  (18 Oct 2026, CUDA 13.0, `sm_80`), with its outputs. Its file listing shows
  the source sizes used (`lbmbench.cu` 39 424 bytes), and its printed results
  match `results/a100-sxm4-40gb/` (e.g. 8993 MLUPS for D3Q19 FP32 AB pull).
- `lbmbench_a100_run.cu`: the exact source of that run.
- `source_diff_vs_release.patch`: its complete difference from the released
  `lbmbench.cu` — the verification threshold and label only (8 changed lines).

To regenerate the A100 log with the released code:

```bash
make ARCH=sm_80
./run_all.sh verify      # expect: number of CHECK lines (should be 0): 0
```
