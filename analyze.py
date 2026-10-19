#!/usr/bin/env python3
"""
Turn the CSV output of run_all.sh into the figures and LaTeX tables used in
the review manuscript.

    python3 analyze.py results/<host1>_<gpu1> [results/<host2>_<gpu2> ...] \
            --fig ../paper/figures --tab ../paper/tables

Every perf number is the median over repeats. Percent-of-bandwidth uses the
median measured float4-copy bandwidth (mode bw) of the same GPU as 100 %.
"""
import argparse, csv, glob, os, statistics as st
from collections import defaultdict
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (fixed order, never cycled); text in ink tones.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "savefig.bbox": "tight", "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
PREC_ORDER = ["fp64", "mixed", "fp32", "fp16s", "fp16c"]
PREC_LABEL = {"fp64": "FP64", "mixed": "FP64/FP32 (mixed)", "fp32": "FP32",
              "fp16s": "FP32/FP16 (IEEE)", "fp16c": "FP32/FP16 (custom)"}
STREAM_LABEL = {"abpull": "AB pull", "abpush": "AB push", "aa": "AA", "ep": "Esoteric pull", "mr": "Moments (MR)"}


def read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def gpu_label(rows, d):
    if not rows:
        return os.path.basename(d)
    n = rows[0]["gpu"].replace("_", " ").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
    return n.replace("A100-SXM4-40GB", "A100 SXM4 40GB")


def med(xs):
    return st.median(xs) if xs else float("nan")


def load(d):
    perf = read(os.path.join(d, "perf.csv"))
    bw = [float(r["eff_bw_gbs"]) for r in read(os.path.join(d, "bw.csv"))]
    return dict(dir=d, perf=perf, bw=med(bw), acc=read(os.path.join(d, "acc.csv")),
                energy=read(os.path.join(d, "energy.csv")), power=os.path.join(d, "power.csv"),
                name=gpu_label(perf or read(os.path.join(d, "bw.csv")), d))


def agg(rows, tag, keys):
    g = defaultdict(list)
    for r in rows:
        if r["tag"] == tag:
            g[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = dict(mlups=med([float(r["mlups"]) for r in rs]),
                      bw=med([float(r["eff_bw_gbs"]) for r in rs]),
                      bcs=float(rs[0]["bytes_cell_step"]), mem=float(rs[0]["mem_bytes_cell"]), n=len(rs))
    return out


def bars(ax, groups, series, val, colors, labels, ylabel):
    w = 0.8 / len(series)
    for j, s in enumerate(series):
        xs, ys = [], []
        for i, g in enumerate(groups):
            v = val(g, s)
            if v is not None:
                xs.append(i - 0.4 + w * (j + 0.5)); ys.append(v)
        ax.bar(xs, ys, w * 0.9, color=colors[j], label=labels[j], edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_ylabel(ylabel)


def fig_patterns(G, figdir):
    """E2: propagation pattern x number format, % of measured copy bandwidth and MLUPS."""
    fig, axes = plt.subplots(len(G), 2, figsize=(7.0, 2.3 * len(G)), squeeze=False)
    for gi, g in enumerate(G):
        a = agg(g["perf"], "E2", ["q", "stream", "prec"])
        for qi, q in enumerate(["19", "27"]):
            ax = axes[gi][qi]
            groups = ["abpull", "abpush", "aa", "ep"]
            bars(ax, groups, PREC_ORDER, lambda s, p: a.get((q, s, p), {}).get("mlups"),
                 SERIES, [PREC_LABEL[p] for p in PREC_ORDER], "MLUPS")
            ax.set_xticklabels([STREAM_LABEL[s].replace('Esoteric pull', 'EP') for s in groups])
            ax.set_title(f"{g['name']} - D3Q{q}", color=INK)
            # bandwidth ceilings for 8/4/2-byte storage
            if g["bw"] == g["bw"]:
                for bytes_, ls in [(8, ":"), (4, "--"), (2, "-.")]:
                    ax.axhline(g["bw"] * 1e3 / (2 * int(q) * bytes_), color=INK2, lw=0.7, ls=ls)
            if gi == 0 and qi == 1:
                ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.text(0.5, -0.02, "Horizontal lines: copy-bandwidth ceiling for 8-, 4- and 2-byte storage (dotted, dashed, dash-dot).",
             ha="center", color=INK2, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "bench_patterns.pdf")); plt.close(fig)


def fig_layout_collision(G, figdir):
    fig, axes = plt.subplots(len(G), 3, figsize=(10.0, 2.4 * len(G)), squeeze=False)
    for gi, g in enumerate(G):
        a1 = agg(g["perf"], "E1", ["q", "stream", "layout", "prec"])
        ax = axes[gi][0]
        groups = [("19", "abpull"), ("19", "aa"), ("27", "abpull"), ("27", "aa")]
        ser = [("soa", "fp32"), ("aos", "fp32"), ("soa", "fp64"), ("aos", "fp64")]
        bars(ax, groups, ser, lambda gg, s: a1.get((gg[0], gg[1], s[0], s[1]), {}).get("mlups"),
             SERIES, [f"{l.upper()} {p.upper()}" for l, p in ser], "MLUPS")
        ax.set_xticklabels([f"D3Q{q}\n{STREAM_LABEL[s]}" for q, s in groups])
        ax.set_title(f"{g['name']} - memory layout (BGK)", color=INK); ax.legend(fontsize=6.5, ncol=2)
        a3 = agg(g["perf"], "E3", ["q", "stream", "prec", "coll"])
        groups = [("19", "abpull"), ("19", "ep"), ("19", "mr"), ("27", "abpull"), ("27", "ep"), ("27", "mr")]
        ser = ["bgk", "trt", "reg"]
        for col, prec in [(1, "fp32"), (2, "fp64")]:
            ax = axes[gi][col]
            def v(gg, c, prec=prec):
                return a3.get((gg[0], gg[1], prec, c), {}).get("mlups")
            bars(ax, groups, ser, v, SERIES, ["BGK", "TRT", "Regularised"], f"MLUPS ({prec.upper()})")
            ax.set_xticklabels([f"Q{q}\n{STREAM_LABEL[s].replace('Esoteric pull','EP').replace('Moments (MR)','MR')}"
                                for q, s in groups])
            ax.set_title(f"{g['name']} - collision, {prec.upper()}", color=INK); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "bench_layout_collision.pdf")); plt.close(fig)


def fig_size(G, figdir):
    fig, axes = plt.subplots(1, len(G), figsize=(3.5 * len(G), 2.4), squeeze=False)
    for gi, g in enumerate(G):
        a = agg(g["perf"], "E4", ["stream", "prec", "nx"])
        ax = axes[0][gi]
        for j, (s, p) in enumerate([("abpull", "fp32"), ("ep", "fp32"), ("ep", "fp16s")]):
            pts = sorted((int(k[2]) ** 3, v["mlups"]) for k, v in a.items() if k[0] == s and k[1] == p)
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], "-o", color=SERIES[j],
                        label=f"{STREAM_LABEL[s]}, {PREC_LABEL[p]}")
        ax.set_xscale("log"); ax.set_xlabel("lattice nodes"); ax.set_ylabel("MLUPS (D3Q19, BGK)")
        ax.set_title(g["name"], color=INK); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "bench_size.pdf")); plt.close(fig)


def fig_accuracy(G, figdir):
    g = G[0]
    rows = g["acc"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    for j, p in enumerate(PREC_ORDER):
        for sh, ls, mk in [("1", "-", "o"), ("0", "--", "s")]:
            pts = sorted((float(r["U"]), float(r["l2err"])) for r in rows
                         if r["tag"] == "E6a" and r["prec"] == p and r["shift"] == sh)
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], ls, marker=mk, color=SERIES[j],
                        mfc="white" if sh == "0" else SERIES[j],
                        label=PREC_LABEL[p] + ("" if sh == "1" else " (no shift)"))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("velocity amplitude $U$ (lattice units)")
    ax.set_ylabel("relative $L_2$ velocity error"); ax.set_title("TGV, $L=128$, $\\tau=0.8$", color=INK)
    ax.legend(fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax = axes[1]
    k = 0
    for p in ["fp64", "fp32", "fp16s", "fp16c"]:
        for c, ls in [("bgk", "-"), ("reg", "--")]:
            pts = sorted((int(r["nx"]), float(r["l2err"])) for r in rows
                         if r["tag"] == "E6b" and r["prec"] == p and r["coll"] == c)
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], ls, marker="o", color=SERIES[PREC_ORDER.index(p)],
                        label=f"{PREC_LABEL[p]}, {c.upper()}")
            k += 1
    xs = [32, 512]
    ax.plot(xs, [2e-3 * (32 / x) ** 2 for x in xs], color=INK2, lw=0.8, ls=":", label="2nd order")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("$L$ (diffusive scaling, fixed Re)")
    ax.set_ylabel("relative $L_2$ velocity error"); ax.set_title("grid convergence", color=INK)
    ax.legend(fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "bench_accuracy.pdf")); plt.close(fig)


def energy_rows(g):
    """Integrate nvidia-smi power samples over each timed window."""
    if not g["energy"] or not os.path.exists(g["power"]):
        return []
    samples = []
    with open(g["power"]) as f:
        for line in f:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                t = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp() * 1e3
                samples.append((t, float(parts[1])))
            except ValueError:
                pass
    # nvidia-smi prints local time without a zone; find the time-zone offset
    # (multiple of 15 min) that places the power samples inside the timed windows.
    best, off = -1, 0.0
    for k in range(-14 * 4, 14 * 4 + 1):
        o = k * 900e3
        hits = sum(1 for r in g["energy"]
                   if any(float(r["t_start_ms"]) <= t + o <= float(r["t_end_ms"]) for t, _ in samples[::10]))
        if hits > best:
            best, off = hits, o
    samples = [(t + off, p) for t, p in samples]
    out = []
    for r in g["energy"]:
        t0, t1 = float(r["t_start_ms"]), float(r["t_end_ms"])
        ps = [p for t, p in samples if t0 <= t <= t1]
        if len(ps) < 3:
            continue
        P = st.mean(ps); E = P * float(r["time_s"])
        lups = float(r["mlups"]) * 1e6 * float(r["time_s"])
        out.append(dict(q=r["q"], stream=r["stream"], prec=r["prec"], coll=r["coll"], mlups=float(r["mlups"]),
                        P=P, nJ=E / lups * 1e9))
    return out


def tables(G, tabdir):
    # Table: E2 summary per GPU
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"GPU & Pattern & \multicolumn{5}{c}{D3Q19 BGK, MLUPS (\% of copy bandwidth)} \\",
             r" & & FP64 & mixed & FP32 & FP16S & FP16C \\", r"\midrule"]
    for g in G:
        a = agg(g["perf"], "E2", ["q", "stream", "prec"])
        for s in ["abpull", "abpush", "aa", "ep"]:
            cells = []
            for p in PREC_ORDER:
                v = a.get(("19", s, p))
                cells.append("--" if not v else f"{v['mlups']:.0f} ({100 * v['bw'] / g['bw']:.0f})")
            lines.append(f"{g['name']} & {STREAM_LABEL[s]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    open(os.path.join(tabdir, "bench_e2.tex"), "w").write("\n".join(lines) + "\n")
    # Table: energy
    lines = [r"\begin{tabular}{lllrrr}", r"\toprule",
             r"GPU & Lattice & Scheme & MLUPS & mean power (W) & energy (nJ/LUP) \\", r"\midrule"]
    for g in G:
        for e in energy_rows(g):
            lines.append(f"{g['name']} & D3Q{e['q']} & {STREAM_LABEL[e['stream']]}, {PREC_LABEL[e['prec']]} & "
                         f"{e['mlups']:.0f} & {e['P']:.0f} & {e['nJ']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(tabdir, "bench_energy.tex"), "w").write("\n".join(lines) + "\n")
    # Device table
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"GPU & measured copy bandwidth (GB/s) & D3Q19 FP32 AB-pull roofline (MLUPS) \\",
             r"\midrule"]
    for g in G:
        lines.append(f"{g['name']} & {g['bw']:.0f} & {g['bw'] * 1e3 / 152:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(tabdir, "bench_devices.tex"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--fig", default="figures")
    ap.add_argument("--tab", default="tables")
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True); os.makedirs(a.tab, exist_ok=True)
    G = [load(d) for d in a.dirs]
    fig_patterns(G, a.fig); fig_layout_collision(G, a.fig); fig_size(G, a.fig); fig_accuracy(G, a.fig)
    tables(G, a.tab)
    for g in G:
        print(f"{g['name']}: copy BW = {g['bw']:.1f} GB/s, {len(g['perf'])} perf rows, {len(g['acc'])} acc rows")
    print("figures ->", a.fig, " tables ->", a.tab)


if __name__ == "__main__":
    main()
