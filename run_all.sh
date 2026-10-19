#!/usr/bin/env bash
# =============================================================================
# Full benchmark campaign for the GPU-LBM review.  Run once on each machine
# (RTX 5080 at work, RTX 5060 Ti at home).  Results go to results/<host>_<gpu>/.
#
#   ./run_all.sh            # everything (~20-40 min depending on GPU)
#   ./run_all.sh verify     # only the correctness cross-check (~1 min)
#   ./run_all.sh perf       # info + bw + E1..E5
#   ./run_all.sh acc        # TGV accuracy (E6)
#   ./run_all.sh energy     # E7 (needs nvidia-smi)
#
# Close other GPU applications (browser, desktop compositor load is OK but keep
# it idle) and let the GPU warm up; results are medians of repeats.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
WHAT=${1:-all}
BIN=./lbmbench
[ -x "$BIN" ] || { echo "build first: make"; exit 1; }
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' )
OUT=results/$(hostname -s)_${GPU}
mkdir -p "$OUT"
REP=3
N=256          # lattice for E1-E3,E5 (D3Q27 fp64 AB at 256^3 needs 7.2 GB)

run() { echo "+ $BIN $*"; $BIN "$@"; }

do_verify() {
  for q in 19 27; do for p in fp64 fp32 mixed fp16s fp16c; do for c in bgk trt reg; do
    run --mode verify --q $q --prec $p --coll $c --n 24 --vsteps 51 --U 0.05 --csv "$OUT/verify.csv"
  done; done; done 2>&1 | tee "$OUT/verify.log"
  echo; echo "number of CHECK lines (should be 0): $(grep -c "CHECK" "$OUT/verify.log" || true)"
}

do_perf() {
  run --mode info > "$OUT/device_info.csv"
  { nvidia-smi; nvidia-smi -q -d ECC,CLOCK,POWER; nvcc --version; } > "$OUT/system_info.txt" 2>&1 || true
  for r in 1 2 3; do run --mode bw --csv "$OUT/bw.csv"; done
  # E1: memory layout (SoA vs AoS)
  for rep in $(seq $REP); do for q in 19 27; do for s in abpull aa; do for l in soa aos; do for p in fp32 fp64; do
    run --mode perf --tag E1 --q $q --stream $s --layout $l --prec $p --coll bgk --n $N --steps 200 --csv "$OUT/perf.csv"
  done; done; done; done; done
  # E2: propagation pattern x number format
  for rep in $(seq $REP); do for q in 19 27; do for s in abpull abpush aa ep; do for p in fp64 mixed fp32 fp16s fp16c; do
    run --mode perf --tag E2 --q $q --stream $s --prec $p --coll bgk --n $N --steps 200 --csv "$OUT/perf.csv"
  done; done; done; done
  # E3: collision operator cost (+ moment representation)
  for rep in $(seq $REP); do for q in 19 27; do for p in fp32 fp64; do
    for s in abpull ep; do for c in bgk trt reg; do
      run --mode perf --tag E3 --q $q --stream $s --prec $p --coll $c --n $N --steps 200 --csv "$OUT/perf.csv"
    done; done
    run --mode perf --tag E3 --q $q --stream mr --prec $p --coll reg --n $N --steps 200 --csv "$OUT/perf.csv"
  done; done; done
  # E4: problem-size sweep (cache / occupancy / tail effects)
  for rep in $(seq $REP); do for n in 32 48 64 96 128 160 192 224 256 320 384; do
    st=$(python3 -c "print(max(50, int(4e9/($n**3))))")
    for cfg in "abpull fp32" "ep fp32" "ep fp16s"; do set -- $cfg
      run --mode perf --tag E4 --q 19 --stream $1 --prec $2 --coll bgk --n $n --steps $st --csv "$OUT/perf.csv"
    done
  done; done
  # E5: thread-block size
  for rep in $(seq $REP); do for b in 32 64 128 192 256; do for s in abpull ep; do
    run --mode perf --tag E5 --q 19 --stream $s --prec fp32 --coll bgk --n $N --steps 200 --block $b --csv "$OUT/perf.csv"
  done; done; done
}

do_acc() {
  # E6a: precision x Mach number, 2D Taylor-Green vortex, L=128, tau=0.8, t = one decay time
  L=128; TAU=0.8
  T=$(python3 -c "import math; nu=($TAU-0.5)/3; k=2*math.pi/$L; print(int(round(1/(2*nu*k*k))))")
  for U in 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005; do for p in fp64 mixed fp32 fp16s fp16c; do for sh in 1 0; do
    run --mode tgv --tag E6a --q 19 --stream ep --prec $p --coll bgk --shift $sh --nx $L --ny $L --nz 1 \
        --tau $TAU --U $U --steps $T --csv "$OUT/acc.csv"
  done; done; done
  # E6b: grid convergence at fixed Re (diffusive scaling U ~ 1/L), all formats, shifted
  for L in 32 64 128 256 512; do
    U=$(python3 -c "print(0.08*32/$L)")
    T=$(python3 -c "import math; nu=($TAU-0.5)/3; k=2*math.pi/$L; print(int(round(1/(2*nu*k*k))))")
    for p in fp64 fp32 fp16s fp16c; do for c in bgk reg; do
      run --mode tgv --tag E6b --q 19 --stream ep --prec $p --coll $c --nx $L --ny $L --nz 1 \
          --tau $TAU --U $U --steps $T --csv "$OUT/acc.csv"
    done; done
  done
}

do_energy() {
  # E7: energy per lattice update. Power sampled every 50 ms by nvidia-smi.
  nvidia-smi --query-gpu=timestamp,power.draw --format=csv,noheader,nounits -lms 50 > "$OUT/power.csv" &
  PID=$!; sleep 2
  for q in 19 27; do for s in abpull ep; do for p in fp64 fp32 fp16s; do
    run --mode perf --tag E7 --q $q --stream $s --prec $p --coll bgk --n $N --steps 1500 --warm 100 --csv "$OUT/energy.csv"
    sleep 2
  done; done; done
  run --mode perf --tag E7 --q 19 --stream mr --prec fp32 --coll reg --n $N --steps 1500 --warm 100 --csv "$OUT/energy.csv"
  sleep 2; kill $PID
}

case $WHAT in
  verify) do_verify ;;
  perf)   do_perf ;;
  acc)    do_acc ;;
  energy) do_energy ;;
  all)    do_verify; do_perf; do_acc; do_energy ;;
  *) echo "unknown target $WHAT"; exit 1 ;;
esac
echo "done -> $OUT"
