"""
fig:time-ratios -- where the time goes, against sequence length.  (eval, single column)

L40S includes the trace job's repeats (644212: x=48-80, and a third node at
x=64).  A100 uses the T block only (H moves the collectives off NVLink, G
carries ballast, P is profiled); its points come from the EXCLUSIVE job
8051891, with x=150 from the shared-node 8039176, the only run that measured
it; the two runs agree within their repeats at x<=100.

  (a) STOCK OFFLOAD'S TAX over static whole-layer checkpointing, steps 1+,
      paired on the same node, on V100, A30 and L40S.  V100 follows the
      post-hoc model tax = kappa*x / T(x) (7758), drawn solid over the
      fitted range and dotted out to the x=356 frontier.  A30 and L40S are NOT
      linear in x, so they get markers only -- never a fitted kappa.  A30's tax
      turns negative beyond x~90 because static's resident set slows the
      attention backward kernel; the A30 matched-footprint cost (offload +
      idle ballast vs static), the honest counterpart of the negative points,
      is echoed and left to the prose, not drawn.
  (b) THE REPLAYED ATTENTION'S SHARE of the step: what remains at long context
      is recomputation, not placement.

Data: V100 from 7758 (tax) and 7756 (attention share), quoted, not re-read.
A30 / L40S taxes are RE-READ here from the job outputs (mean of steps 1+ of each
paired repeat; tax = mean delta / mean static over every repeat at that x,
whisker = range of the per-repeat taxes).  Attention shares from the profiled
arms of 630188, 633912, 642029 and 8039176, quoted.

Regime: L=64 tier-A, chunked CE 256, fp16 memory-efficient kernel, fused-input
norm, use_cache off, b=1, world 4, W=0, CPU offload off.
SIZING: FIG_W = 0.85 * ps.COL_W (2.8305 in -> 1698 px at 600 dpi), included at
width=0.85\columnwidth, so the scale factor is still 1; no tight_layout.
At 85% width the label margins are full size in inches; the V100 legend
label breaks after "PCIe," so the key clears the x=48 points; the 2% tag
sits under its rule; the V100 frontier note is three lines right of A30's
x=150 point (two lines would cross it at this width), so (a)'s y floor is
-2.4; (b)'s letter sits in the panel gap.
"""

import glob
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "..", "profile_results"))

# --- V100, quoted -------------------------------------------------------------
# 7758: x -> (tax %, +- %).  x < 16 omitted: run-to-run spread is as large as
# the effect there, and the paper makes no claim below it.
V100_TAX = {16: (4.3, 3.5), 24: (4.9, 2.2), 32: (3.3, 1.6), 48: (3.0, 0.5),
            64: (2.2, 0.5), 100: (1.7, 0.1), 104: (1.5, 0.1)}
KAPPA = 0.032                                   # s per x, V100 (7758)
T_COEF = (0.02123, 0.0397, 2.066)               # T(x) = a x^2 + b x + c, fitted x = 8..104
X_FIT = (24, 104)                               # kappa is linear in x from 24 (7758)
X_FRONTIER = 356                                # V100 offload frontier (7747)
V100_ATTN = {48: 15.6, 100: 16.7, 200: 17.1}    # 7756

# --- A30 / L40S, re-read from the job outputs ---------------------------------
TAX_ROOTS = {
    "A30": ["a30_633912/a30_out/633912", "a30f_642029/a30f_out"],
    "L40S": ["l40s4/630188", "l40s4/634078", "l40s4/644212"],
    # A100: the exclusive job (8051891) for x<=100; x=150 only from the
    # shared-node run (8039176). (root, x filter) entries.
    "A100": ["a100_8051891/a100_out/8051891", ("a100_8039176/a100_out/8039176", {150})],
}
# Offload + idle ballast against static, A30 (642029 block G): the matched-footprint cost.
A30_MATCHED = {100: 0.99, 150: 0.58}
# Attention-replay share of the step: x -> (lo, hi) %.
ATTN = {"A30": {64: (15.3, 15.4), 150: (16.2, 16.2)},   # 633912, 642029
        "L40S": {64: (13.0, 13.3)},                     # 630188
        "A100": {64: (15.8, 15.8), 150: (16.3, 16.3)}}  # 8039176
ARM = re.compile(r"^(T|S)_(static|offload)_fp16_mem_L64_x(\d+)_n\d+(?:_r(\d+))?$")

# --- style --------------------------------------------------------------------
GPU_STYLE = {   # colour, marker, legend label -- marker carries GPU without colour
    "V100": (ps.PALETTE["blue_main"], "o", "V100 (PCIe, no GPUDirect)"),
    "A30": (ps.PALETTE["teal"], "s", "A30 (NVLink pairs)"),
    "L40S": (ps.PALETTE["violet"], "^", "L40S (PCIe only)"),
    "A100": (ps.PALETTE["red_strong"], "v", "A100 (NVSwitch)"),
}
C_NOTE = ps.EXTRA["neutral_text"]
C_RULE = ps.EXTRA["neutral_line"]
MS = 4.2
FS_ANNOT = 7.0

SCALE = 0.85                                  # printed at 0.85\columnwidth
FIG_W, FIG_H = SCALE * ps.COL_W, SCALE * 2.72  # 2.8305 x 2.312 in
PX_W = int(round(FIG_W * ps.DPI))             # 1698
AX_L, AX_R = 0.46, 0.08                       # inches: tick + axis label margins kept
B_BOTTOM, B_H = 0.325, 0.50                   # inches
A_BOTTOM = B_BOTTOM + B_H + 0.14              # gap holds (a)'s tick marks only
A_H = FIG_H - A_BOTTOM - 0.09
PANEL_X = -0.42                               # inches left of the axes, as at full width
X_LIM = (13.0, 420.0)
X_TICKS = [16, 32, 64, 128, 256]
A_LIM = (-2.4, 7.0)        # -2.4: room under zero for the frontier note
B_LIM = (11.5, 18.5)

OUT_LOCAL = "fig12.png"
OUT_IN_USE = os.path.join("..", "rendered", "fig12.png")


def remote_taxes(gpu):
    """x -> (tax %, lo %, hi %, n repeats, mean static s), steps 1+, paired."""
    per = {}
    for rel in TAX_ROOTS[gpu]:
        rel, keep = rel if isinstance(rel, tuple) else (rel, None)
        arms = {}
        for d in glob.glob(os.path.join(RES, rel, "*")):
            m = ARM.match(os.path.basename(d))
            f = os.path.join(d, "steps_rank0.json")
            if not m or not os.path.exists(f) or (keep and int(m[3]) not in keep):
                continue
            steps = json.load(open(f))["steps"][1:]
            arms[(m[2], int(m[3]), m[4] or "1")] = st.mean(s["step_ms"] for s in steps) / 1e3
        for (mode, x, rep), t in arms.items():
            if mode == "static" and ("offload", x, rep) in arms:
                per.setdefault(x, []).append((t, arms[("offload", x, rep)] - t))
    out = {}
    for x, pairs in sorted(per.items()):
        ts = st.mean(p[0] for p in pairs)
        each = [100 * d / t for t, d in pairs]
        out[x] = (100 * st.mean(p[1] for p in pairs) / ts, min(each), max(each), len(pairs), ts)
    return out


def v100_model(x):
    a, b, c = T_COEF
    return 100 * KAPPA * x / (a * x * x + b * x + c)


def style_x(ax, labels):
    ax.set_xscale("log", base=2)
    ax.set_xlim(*X_LIM)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([str(t) for t in X_TICKS] if labels else [])
    ax.minorticks_off()
    ax.tick_params(pad=2)


def main():
    ps.apply_publication_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    w = (FIG_W - AX_L - AX_R) / FIG_W
    ax = fig.add_axes([AX_L / FIG_W, A_BOTTOM / FIG_H, w, A_H / FIG_H])
    bx = fig.add_axes([AX_L / FIG_W, B_BOTTOM / FIG_H, w, B_H / FIG_H])

    # ---- (a) offload's tax ---------------------------------------------------
    ax.axhline(0.0, color="black", linewidth=0.6, zorder=1)
    ax.axhline(2.0, color=C_RULE, linewidth=0.6, linestyle=(0, (1.5, 1.5)), zorder=1)
    # Under its rule at 85% width: the key reaches down to just above it.
    ax.text(X_LIM[1] / 1.04, 1.82, "2%", fontsize=FS_ANNOT, color=C_RULE,
            ha="right", va="top")

    c, mk, _ = GPU_STYLE["V100"]
    xs = np.geomspace(*X_FIT, 120)
    ax.plot(xs, v100_model(xs), color=c, linewidth=1.0, zorder=2)
    xs = np.geomspace(X_FIT[1], X_FRONTIER, 80)
    ax.plot(xs, v100_model(xs), color=c, linewidth=1.0, linestyle=(0, (1.2, 1.6)), zorder=2)
    ax.plot([X_FRONTIER], [v100_model(X_FRONTIER)], marker=mk, markersize=MS,
            markerfacecolor="white", markeredgecolor=c, markeredgewidth=0.9,
            linestyle="none", zorder=4)
    # Three lines right of A30's x=150 point; two would run over it at 85%.
    ax.text(X_LIM[1] / 1.03, -0.22,
            f"V100 model:\n{v100_model(X_FRONTIER):.1f}% at its\nfrontier",
            fontsize=FS_ANNOT, color=c, ha="right", va="top", linespacing=0.95)
    vx = sorted(V100_TAX)
    ax.errorbar(vx, [V100_TAX[x][0] for x in vx], yerr=[V100_TAX[x][1] for x in vx],
                fmt=mk, markersize=MS, color=c, markeredgecolor="black",
                markeredgewidth=0.6, elinewidth=0.6, capsize=0, zorder=5)

    taxes = {}
    for gpu in ("A30", "L40S", "A100"):
        c, mk, _ = GPU_STYLE[gpu]
        t = taxes[gpu] = remote_taxes(gpu)
        xs = sorted(t)
        y = np.array([t[x][0] for x in xs])
        lo = y - np.array([t[x][1] for x in xs])
        hi = np.array([t[x][2] for x in xs]) - y
        ax.plot(xs, y, color=c, linewidth=0.7, alpha=0.6, zorder=3)
        ax.errorbar(xs, y, yerr=[lo, hi], fmt=mk, markersize=MS, color=c,
                    markeredgecolor="black", markeredgewidth=0.6,
                    elinewidth=0.6, capsize=0, zorder=5)

    # The matched-footprint cost (A30_MATCHED) is prose, not a marker: at
    # x=100 it lands on the L40S point (0.99 vs 1.01) and would hide under it.
    c = GPU_STYLE["A30"][0]
    ax.text(74, -1.15, "A30: static's footprint\nslows attention backward",
            fontsize=FS_ANNOT, color=c, ha="right", va="center", linespacing=1.05)

    ax.set_ylim(*A_LIM)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_ylabel("offload's tax (%)", labelpad=2)
    style_x(ax, labels=False)
    ps.panel_label(ax, "(a)", x=PANEL_X / (w * FIG_W), y=1.0 - 0.055 / A_H)

    # One boxed key in (a)'s empty upper right, four rows; (b) uses the same
    # markers and colours.  At 85% width the
    # V100 label breaks after "PCIe," so the key clears the x=48 points.
    handles = [Line2D([], [], marker=GPU_STYLE[g][1], color=GPU_STYLE[g][0],
                      markeredgecolor="black", markeredgewidth=0.6,
                      markersize=MS, linewidth=0.7) for g in GPU_STYLE]
    labels = [GPU_STYLE[g][2].replace("PCIe, ", "PCIe,\n") for g in GPU_STYLE]
    leg = ax.legend(handles, labels, loc="upper right",
                    bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=FS_ANNOT,
                    handlelength=1.6, handletextpad=0.4, borderaxespad=0.3,
                    borderpad=0.3, labelspacing=0.2)
    for t in leg.get_texts():
        t.set_linespacing(1.0)

    # ---- (b) replayed attention ----------------------------------------------
    for gpu, pts in (("V100", {x: (v, v) for x, v in V100_ATTN.items()}),) + tuple(ATTN.items()):
        c, mk, _ = GPU_STYLE[gpu]
        xs = sorted(pts)
        mid = np.array([(pts[x][0] + pts[x][1]) / 2 for x in xs])
        half = np.array([(pts[x][1] - pts[x][0]) / 2 for x in xs])
        bx.plot(xs, mid, color=c, linewidth=0.7, alpha=0.6, zorder=3)
        bx.errorbar(xs, mid, yerr=half, fmt=mk, markersize=MS, color=c,
                    markeredgecolor="black", markeredgewidth=0.6,
                    elinewidth=0.6, capsize=0, zorder=5)
    bx.set_ylim(*B_LIM)
    bx.set_yticks([12, 15, 18])
    bx.set_ylabel("replay (%)", labelpad=2)
    style_x(bx, labels=True)
    bx.set_xlabel(r"tokens per step ($\times 1024$), batch 1", labelpad=2)
    # (b) sits in the gap, clear of "replay (%)", which is as tall as the panel.
    ps.panel_label(bx, "(b)", x=PANEL_X / (w * FIG_W), y=1.0 + 0.03 / B_H)

    ps.save(fig, OUT_LOCAL, OUT_IN_USE, script_dir=HERE, expect_px=PX_W)
    plt.close(fig)

    # ---- echo every plotted quantity -----------------------------------------
    print("(a) offload's tax over static, steps 1+")
    for x in vx:
        print(f"  V100  x={x:<4d} {V100_TAX[x][0]:+.2f}% +- {V100_TAX[x][1]}   model {v100_model(x):.2f}%")
    print(f"  V100  model at the x={X_FRONTIER} frontier {v100_model(X_FRONTIER):.2f}%")
    for gpu, t in taxes.items():
        for x, (tax, lo, hi, n, ts) in t.items():
            print(f"  {gpu:<5} x={x:<4d} {tax:+.2f}%  repeats {n}  range [{lo:+.2f}, {hi:+.2f}]  static {ts:.2f} s")
    for x, v in A30_MATCHED.items():
        print(f"  A30   x={x:<4d} matched footprint {v:+.2f}%")
    print("(b) attention-replay share of the step")
    for x, v in V100_ATTN.items():
        print(f"  V100  x={x:<4d} {v:.1f}%")
    for gpu, pts in ATTN.items():
        for x, (lo, hi) in pts.items():
            print(f"  {gpu:<5} x={x:<4d} {lo:.1f}-{hi:.1f}%")


if __name__ == "__main__":
    main()
