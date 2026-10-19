#!/usr/bin/env python3
"""Count FP64 instructions per lattice update in the compiled AB-pull SoA kernels.

Compile only (no GPU needed):
    nvcc -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo -arch=sm_120 -cubin -o lb.cubin lbmbench.cu
    cuobjdump -sass lb.cubin > lb.sass
    python3 tools/count_fp64_sass.py lb.sass
Each thread updates one lattice node, so the static count of the kernel is the
count per update. DFMA, DADD and DMUL each occupy one FP64 issue slot.
The counts reported in the review were obtained with CUDA 13.3.73 (the compiler
used for the RTX 5080 measurements); CUDA 13.0.88 gives identical counts.
"""
import collections, re, sys

PREC = {"fp64": "3P64", "mixed": "3PMX"}
COLL = {"bgk": 0, "trt": 1, "reg": 2}
txt = open(sys.argv[1]).read()
funcs = {}
for f in re.split(r"\n\s*Function : ", txt)[1:]:
    funcs[f.split()[0]] = re.findall(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)", f)
print("q   coll prec   FP64_instr  F2F_64  FP64_flop")
for q in (19, 27):
    for coll, ci in COLL.items():
        for prec, pt in PREC.items():
            name = f"_Z6k_stepI5D3Q{q}{pt}Li{ci}ELi0ELi0ELi0EEvPNT0_1SES4_3PrmPd"
            if name not in funcs:
                continue
            ins = funcs[name]
            c = collections.Counter(i.split(".")[0] for i in ins)
            n64 = c["DFMA"] + c["DADD"] + c["DMUL"]
            f2f = sum(1 for i in ins if i.startswith("F2F") and "64" in i)
            print(f"{q:<3} {coll:<4} {prec:<6} {n64:>10} {f2f:>7} {2*c['DFMA']+c['DADD']+c['DMUL']:>10}")
