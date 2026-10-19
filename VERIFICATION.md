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

## A100: original run and complete rerun with the released source

`results/a100-sxm4-40gb/` is the dataset used in the review (Google Colab,
18 Oct 2026). That run used an earlier revision of `lbmbench.cu` whose
verification step applied a single threshold of 1e-12 to all formats, so its
`verify.log` labels the 11 FP32 and 16-bit Esoteric Pull differences listed
above as `CHECK`. The released revision uses format-dependent thresholds
(1e-12 FP64/mixed, 1e-6 FP32, 1e-4 16-bit). The two revisions differ **only**
in this threshold and its label; see `records/a100_colab_run/` (executed
notebook, exact source, and `source_diff_vs_release.patch`, 8 changed lines).

`results/a100-sxm4-40gb-rerun/` is a complete rerun of all experiments with
the released source (repository clone, Colab A100 SXM4 40GB, 19 Oct 2026,
ECC on). Compared with the original run:

| Quantity | Result |
|---|---|
| verification (`verify.log`) | 0 `CHECK`; all 140 differences identical to the original run |
| accuracy (E6, 120 cases) | L2 errors bitwise identical |
| throughput (E1-E5, 381 cases) | median ratio rerun/original 1.000; all within -1.2% / +2.9% |
| energy per update (E7, 13 cases) | within 1.6% |
| copy bandwidth | 1357 vs. 1358 GB/s |
