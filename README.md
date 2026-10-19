# lbmbench: a controlled GPU benchmark for the core lattice Boltzmann kernel

`lbmbench` is a single CUDA source file that changes one implementation factor
at a time while keeping everything else fixed:

| Factor | Options |
|---|---|
| Velocity set | D3Q19, D3Q27 |
| Propagation | AB pull, AB push, AA, Esoteric Pull, moment representation (MR) |
| Layout | SoA, AoS |
| Storage format | `fp64`, `mixed` (FP64 arithmetic / FP32 storage), `fp32`, `fp16s` (IEEE half, 2^15 prescale), `fp16c` (custom 1-4-11) |
| Collision | BGK, TRT, regularised |
| DDF shifting | on / off |

It accompanies the review article *GPU implementations of the lattice Boltzmann
method: a review of algorithms, software and performance* (A. Kulkarni,
manuscript in preparation). Raw measurements on an NVIDIA RTX 5080 and an
NVIDIA A100 SXM4 40 GB are included under `results/`.

## Requirements
- NVIDIA GPU and CUDA toolkit (CUDA ≥ 12.8 for Blackwell, `sm_120`)
- Python 3 with `matplotlib` for the analysis script
- `nvidia-smi` for the energy experiment

## Build
```bash
make                    # default ARCH=sm_120 (RTX 50xx)
make ARCH=sm_80         # A100
make ARCH=native        # current GPU
make FAST=1             # D3Q19 only, faster compile
```

## 1. Correctness check (about 1 min) — run first
```bash
./run_all.sh verify
```
Every line should report `bitwise identical` or `OK`, and the script ends with
`number of CHECK lines (should be 0): 0`. Small differences (round-off level)
between patterns are expected for push and Esoteric Pull because of FMA
contraction; the AA pattern is always bitwise identical to AB pull.

## 2. Full campaign (about 20–40 min per GPU)
```bash
./run_all.sh            # verify + perf + acc + energy
./run_all.sh perf       # device info, bandwidth, E1–E5
./run_all.sh acc        # E6 Taylor–Green accuracy
./run_all.sh energy     # E7 energy per update (nvidia-smi power logging)
```
Results go to `results/<host>_<gpu>/`. Keep the GPU otherwise idle and record
driver, compiler, clock and power settings (`system_info.txt` is written
automatically).

## 3. Figures and tables
```bash
python3 analyze.py results/rtx5080 results/a100-sxm4-40gb --fig figures --tab tables
```
This reproduces the benchmark figures and tables of the review from the
included CSV files.

## Single runs
```bash
./lbmbench --mode info
./lbmbench --mode bw
./lbmbench --mode perf --q 19 --stream ep --prec fp16s --coll bgk --n 256 --steps 200
./lbmbench --mode tgv  --q 19 --stream ep --prec fp32 --shift 0 --nx 128 --ny 128 --nz 1 --tau 0.8 --U 0.01 --steps 2075
./lbmbench --mode verify --q 27 --prec fp64 --coll reg --n 24
```
Each run appends one CSV line (`--csv`, default `results.csv`) with the GPU,
the factors, MLUPS, bytes per cell per step, effective bandwidth, memory per
cell, timestamps (for energy), L2 error, mass drift and verification
difference.

## Notes
- Memory: the largest default case (D3Q27 fp64 AB at 256³) needs 7.2 GB; E4
  goes up to 384³ D3Q19 fp32 AB (8.6 GB).
- In `tgv` mode, the output of the last step corresponds to t = steps.
- `fp16c` is a custom 1-4-11 format, similar in spirit to FluidX3D's FP16C but
  not identical. `fp16s` is IEEE binary16 with 2^15 prescaling.
- AA and Esoteric Pull compile one kernel per time-step parity, which keeps
  register use low for D3Q27.
- The benchmark measures the bulk kernel on a periodic, dense lattice only; it
  does not include boundaries, sparse geometries or multi-GPU communication.

## Licence
MIT — see [LICENSE](LICENSE). If you use the benchmark or data, please cite
the review article (see `CITATION.cff`).
