"""
fig:modern -- V100-fitted predictions, measured on Ampere and Ada.  (eval, single column)

Drawn from the A30 (633912, 642029, 642032), L40S (630188, 634078, 644212)
and A100 (8039176, shared DGX node -- bytes do not depend on the neighbour;
the A100 job ran no capacity arms, so it has no span in (b)) jobs, plus a
V100 row in (a), so the four platforms read as parallel tests.  Every marker
is MEASURED; open markers are the predictions registered before each run.

  (a) BYTES CARRY.  Residual (measured peak - harness line) of every byte arm,
      the line built from V100 calibration only (workspace 16.2559 MiB carried,
      not refitted).  Re-read here from the job outputs into remote_bytes.json
      by the same rule the job readers use (max peak over steps 1+).
      Points whose line needed a term added AFTER the run are drawn grey and
      labelled with what was registered: the flash plane (+1.5000 MiB/x,
      registered 0) and the 3B deep-replay instant at x=16
      (+0.148 MiB as registered).  Excluded and echoed: the A30
      follow-up's E block (expandable segments OFF, an allocator control the
      harness is not calibrated for) and G block (an idle ballast tensor is
      part of the peak by design).
  (b) SPANS CARRY, ABSOLUTE LENGTHS DO NOT.  Largest fit of stock offload over
      largest fit of static, measured (filled, whisker = the true ratio's
      bracket) against the ratio registered before the run (open).  V100 and
      A30 tier-A L=64; L40S the Llama-3.2-3B shape, with and without FSDP CPU
      offload on both arms.  L40S 3B registered 194/103 on a card-total input
      error; the span is insensitive to it.

Regime: chunked CE 256, fp16 memory-efficient kernel (bf16 / flash arms
named), fused-input norm, use_cache off, b=1, world 4, W=0.
SIZING: FIG_W = 0.85 * ps.COL_W (2.8305 in), placed at 0.85\columnwidth, so
scale 1; no tight_layout.

The V100 row in (a): its arms are fig15's V100 base-shape points (world 4,
registered in each job header) restricted to the mechanisms the remote rows
carry: static, offload, forced all-moved, +/- CPU offload.  The gated
instrument's arms (below the cap; fig13) are excluded and echoed.  Drawn where
they landed: the V100 registered miss 7756 offload+cpu x=100 (+18.00; two
outcomes registered, the alternative held), and static at x=100/104 and
static+cpu x=112/116 (+0.017 to +0.023), hits under the paper's 0.05 MiB
tolerance.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BYTES = os.path.join(HERE, "remote_bytes.json")
EXCLUDE_BLK = {"E": "expandable segments off (allocator control)",
               "G": "idle ballast held in the peak by design"}

# --- spans: (label, measured (off, st), fail rungs (off, st), registered (off, st), source)
SPANS = [
    ("V100", (356, 104), (360, 108), (356, 104), "1.14/1.15"),
    ("A30", (564, 164), (568, 166), (566, 164), "1.31/1.33"),
    ("L40S 3B", (189, 101), (190, 102), (194, 103), "1.29"),
    ("L40S 3B\n+ CPU off.", (258, 134), (259, 135), (256, 133), "1.32"),
]

ROWS = [("V100", "tierA", "V100"), ("A30", "tierA", "A30"), ("L40S", "tierA", "L40S"), ("L40S", "3B", "L40S 3B"),
        ("A100", "tierA", "A100")]
GPU_C = {"V100": ps.PALETTE["blue_main"], "A30": ps.PALETTE["teal"],
         "L40S": ps.PALETTE["violet"], "A100": ps.PALETTE["red_strong"]}
MECH = {"static": "s", "offload": "o", "evict": "D"}
C_POST = ps.EXTRA["neutral_line"]
C_NOTE = ps.EXTRA["neutral_text"]
FS_ANNOT = 7.0
MS = 3.6

# Printed at 0.85 of the column in both directions,
# \includegraphics[width=0.85\columnwidth], so the canvas IS that width and the
# type stays at scale 1.  Label margins (text that does not shrink) are full
# size; the data areas are what is smaller.
SCALE = 0.85
FIG_W, FIG_H = SCALE * ps.COL_W, 2.77     # 2.8305 in -> 1698 px at 600 dpi
AX_L, AX_R = 0.70, 0.10
B_BOTTOM, B_H = 0.34, 0.68
A_BOTTOM = B_BOTTOM + B_H + 0.33          # 0.33 in holds (a)'s ticks + xlabel
A_H = FIG_H - 0.14 - A_BOTTOM             # 0.14 in holds the key above (a)
PANEL_DX = -0.595               # in; (a)/(b) letters, left of the axes
LINTHRESH = 0.01                # MiB; same symlog threshold as fig15
A_LIM = (-0.01, 160.0)
A_YLIM = (-0.40, 4.50)          # row units; 5 rows over A_H
B_LIM = (1.6, 3.75)
B_YLIM = (-0.90, 3.66)          # row units; 4 rows over B_H; top pad clears (a)'s xlabel
CPU_LAB_DY = 3.5                # pt; lifts the hanging label's first line onto its row
B_REG_DY = 0.38                 # row units the registered marker sits above

OUT_LOCAL = "fig9.png"
# In the band the arms share a few exact values, so one marker per
# mechanism, stacked, with a count per row.
OUT_IN_USE = os.path.join("..", "rendered", "fig9.png")
STACK_PITCH = 0.175              # row units between stacked mechanism markers (~0.046 in)
MECH_ORDER = ["static", "offload", "evict", "staticcpu", "offloadcpu"]


V100_MODE = {"static": "static", "offload": "offload", "static+cpu": "staticcpu",
             "offload+cpu": "offloadcpu"}


def load_v100():
    """fig15's V100 base-shape parity points, as fig9 arms; returns (arms, gated)."""
    import importlib.util
    path = os.path.join(HERE, "..", "f15", "plot_fig15.py")
    spec = importlib.util.spec_from_file_location("plot_fig15", path)
    f15 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f15)
    arms, gated = [], []
    for r in f15.points():
        if r["row"] != "V100":
            continue
        if r["mechanism"].startswith("engine"):
            if "force-evict" not in r["notes"].lower():
                gated.append(r)
                continue
            mode = "evict"
        else:
            mode = V100_MODE[r["mechanism"]]
        arms.append(dict(gpu="V100", job=r["job"], arm=f"{r['mechanism']}_x{r['x']}",
                         mode=mode, shape="tierA", x=r["x"], meas=r["measured_mib"],
                         res=r["res"], res_reg=r["res"], posthoc=False))
    return arms, gated


def main():
    arms = json.load(open(BYTES))
    v100, gated = load_v100()
    arms = v100 + arms
    excluded = [a for a in arms if a["arm"].split("_")[0] in EXCLUDE_BLK]
    arms = [a for a in arms if a["arm"].split("_")[0] not in EXCLUDE_BLK]

    ps.apply_publication_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    w = (FIG_W - AX_L - AX_R) / FIG_W
    ax = fig.add_axes([AX_L / FIG_W, A_BOTTOM / FIG_H, w, A_H / FIG_H])
    bx = fig.add_axes([AX_L / FIG_W, B_BOTTOM / FIG_H, w, B_H / FIG_H])

    # ---- (a) residuals against the REGISTERED line -----------------------------
    ax.axvspan(-LINTHRESH, LINTHRESH, color=ps.PALETTE["neutral"], alpha=0.35,
               linewidth=0, zorder=0)
    ax.axvline(0.0, color="black", linewidth=0.6, zorder=1)

    def marker(x, y, mode, gpu):
        base = mode.replace("cpu", "")
        ax.plot(x, y, marker=MECH[base], markersize=MS, color=GPU_C[gpu],
                markeredgecolor="black", markeredgewidth=0.4, linestyle="none",
                zorder=3)
        if mode.endswith("cpu"):
            ax.plot(x, y, marker="o", markersize=1.2, color="white",
                    linestyle="none", zorder=4)

    stats = []
    print("(a) in-band stacks: row, value, mechanism, count")
    for i, (gpu, shape, _) in enumerate(ROWS):
        y0 = len(ROWS) - 1 - i
        sel = [a for a in arms if a["gpu"] == gpu and a["shape"] == shape]
        band = [a for a in sel if abs(a["res_reg"]) < LINTHRESH]
        # IN THE BAND: arms share a handful of exact values, so one marker per
        # mechanism present, stacked, at its true x; one count per row.
        modes = [m for m in MECH_ORDER if any(a["mode"] == m for a in band)]
        k = len(modes)
        for j, m in enumerate(modes):
            ms = [a for a in band if a["mode"] == m]
            xv = sum(a["res_reg"] for a in ms) / len(ms)
            y = y0 + ((k - 1) / 2 - j) * STACK_PITCH
            marker(xv, y, m, gpu)
            print(f"   {gpu:<5}{shape:<6} {xv:+.4f}  {m:<11} {len(ms)}"
                  f"  (spread {max(a['res_reg'] for a in ms) - min(a['res_reg'] for a in ms):.4f})")
        ax.text(0.0035, y0, f"$n{{=}}{len(band)}$", fontsize=FS_ANNOT,
                color=C_NOTE, ha="left", va="center")
        # OUTSIDE: the registered misses, one marker each (they differ).
        out = [a for a in sel if abs(a["res_reg"]) >= LINTHRESH]
        for j, a in enumerate(sorted(out, key=lambda a: a["res_reg"])):
            y = y0 + ((j % 3) - 1) * STACK_PITCH
            marker(a["res_reg"], y, a["mode"], gpu)
        hits = [a for a in sel if not a["posthoc"]]
        stats.append((gpu, shape, len(sel), len(hits),
                      max(abs(a["res_reg"]) for a in hits),
                      max(abs(a["res"]) for a in sel)))

    # The V100 note is three lines (two would run into the +0.02 MiB misses
    # at this width); the flash note sits a little lower to clear it, and the
    # "every other arm" note sits right, off the n = 129 count.
    ax.text(9.0, 2.88, "flash plane,\n+1.5 MiB/x", fontsize=FS_ANNOT, color=C_NOTE,
            ha="right", va="center", linespacing=1.0)
    ax.text(0.24, 1.0, "x=16: a deeper\nreplay instant", fontsize=FS_ANNOT,
            color=C_NOTE, ha="left", va="center", linespacing=1.0)
    ax.text(12.0, 3.93, "two\noutcomes\nregistered", fontsize=FS_ANNOT, color=C_NOTE,
            ha="right", va="center", linespacing=1.0)
    # The paper's one tolerance, the +/-0.05 MiB written into the job scripts
    # before each run.
    ax.text(0.03, 2.0, "every other arm\nwithin 0.05 MiB", fontsize=FS_ANNOT,
            color=C_NOTE, ha="left", va="center", linespacing=1.0)
    # key for the stacks ("one marker per mechanism; n = arms") is in the caption

    ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=2.2)
    ax.set_xlim(*A_LIM)
    ax.set_xticks([-0.01, -0.005, 0, 0.005, 0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(["$-$.01", "", "0", "", ".01", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_ylim(*A_YLIM)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[2] for r in ROWS][::-1])
    # Row labels carry the platform colour, so no colour key is needed.
    for t, r in zip(ax.get_yticklabels(), ROWS[::-1]):
        t.set_color(GPU_C[r[0]])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", pad=2)
    ax.set_xlabel("measured $-$ registered line (MiB)", labelpad=2)
    ps.panel_label(ax, "(a)", x=PANEL_DX / (w * FIG_W), y=1.0)

    handles = [Line2D([], [], marker=MECH[m], color="white", markeredgecolor="black",
                      markeredgewidth=0.6, markersize=MS + 0.4, linestyle="none")
               for m in ("static", "offload", "evict")]
    handles.append((Line2D([], [], marker="s", color="white", markeredgecolor="black",
                           markeredgewidth=0.6, markersize=MS + 0.4, linestyle="none"),
                    Line2D([], [], marker="o", color="black", markersize=1.4,
                           linestyle="none")))
    ax.legend(handles, ["static", "offload", "all moved", "+ CPU off."],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=4,
              fontsize=FS_ANNOT, handlelength=0.8, handletextpad=0.3, columnspacing=0.7,
              borderaxespad=0.1, frameon=False, borderpad=0.0)   # single row above: no box

    # ---- (b) spans --------------------------------------------------------------
    for i, (label, (fo, fs), (xo, xs), (ro, rs), _) in enumerate(SPANS):
        y = len(SPANS) - 1 - i
        gpu = label.split(" ")[0]
        c = GPU_C[gpu]
        meas, lo, hi = fo / fs, fo / xs, xo / fs
        reg = ro / rs
        bx.plot([lo, hi], [y, y], color=c, linewidth=1.0, solid_capstyle="butt", zorder=2)
        bx.plot(meas, y, marker="o", markersize=MS + 0.6, color=c,
                markeredgecolor="black", markeredgewidth=0.6, zorder=4)
        bx.plot(reg, y + B_REG_DY, marker="v", markersize=MS + 0.4, color="white",
                markeredgecolor=c, markeredgewidth=0.9, zorder=4)
        bx.text(hi + 0.05, y, f"{meas:.3f}", fontsize=FS_ANNOT, color=c,
                ha="left", va="center")
    bx.set_xlim(*B_LIM)
    bx.set_ylim(*B_YLIM)
    bx.set_yticks(range(len(SPANS)))
    bx.set_yticklabels([s[0] for s in SPANS][::-1])
    for t, sp in zip(bx.get_yticklabels(), SPANS[::-1]):
        t.set_color(GPU_C[sp[0].split(" ")[0]])
    bx.tick_params(axis="y", length=0, pad=3)
    # The two-line label hangs from its row (first line on the row, the CPU
    # qualifier below), so it reads with its own row, not the one above.
    # After tick_params, which resets label transforms.
    cpu_lab = bx.get_yticklabels()[0]
    cpu_lab.set_linespacing(1.0)
    cpu_lab.set_va("top")
    cpu_lab.set_transform(cpu_lab.get_transform() + mtransforms.ScaledTranslation(
        0, CPU_LAB_DY / 72, fig.dpi_scale_trans))
    bx.tick_params(axis="x", pad=2)
    bx.set_xlabel(r"span: offload over static (largest fits, $\times$)", labelpad=2)
    # The row labels name the platform by colour, so the key shows the two
    # marker kinds only, in neutral grey.
    grey = ps.EXTRA["neutral_text"]
    meas_h = Line2D([], [], marker="o", color=grey, markeredgecolor="black",
                    markeredgewidth=0.6, markersize=MS + 0.6, linestyle="none")
    reg_h = Line2D([], [], marker="v", color="white", markeredgecolor=grey,
                   markeredgewidth=0.9, markersize=MS + 0.4, linestyle="none")
    bx.legend([meas_h, reg_h], ["measured", "registered"], loc="center",
              bbox_to_anchor=(0.62, 0.30), ncol=1, fontsize=FS_ANNOT,
              handlelength=1.0, handletextpad=0.4, labelspacing=0.3,
              borderaxespad=0.0)
    ps.panel_label(bx, "(b)", x=PANEL_DX / (w * FIG_W), y=1.0)

    ps.save(fig, OUT_LOCAL, OUT_IN_USE, script_dir=HERE,
            expect_px=int(round(FIG_W * ps.DPI)))
    plt.close(fig)

    # ---- echo -----------------------------------------------------------------
    print("(a) byte residuals: measured - line registered before the run (current harness in brackets)")
    for gpu, shape, n, nh, worst_reg, worst_now in stats:
        print(f"  {gpu:<5} {shape:<6} {n:3d} arms, {nh} hit as registered (max |r| {worst_reg:.4f} MiB);"
              f" against today's harness max |r| {worst_now:.4f} MiB")
    for a in arms:
        if a["posthoc"]:
            print(f"    MISS as registered: {a['job']} {a['arm']}  {a['res_reg']:+.4f}"
                  f"  [{a['res']:+.4f} with the post-hoc term]")
    print(f"  EXCLUDED {len(gated)} gated V100 instrument arms (below the cap; fig13)")
    v_out = [a for a in v100 if abs(a["res_reg"]) >= LINTHRESH]
    for a in v_out:
        print(f"    V100 outside band: {a['job']} {a['arm']}  {a['res_reg']:+.4f}")
    rest = [a for a in arms if not a["posthoc"] and abs(a["res_reg"]) < 1.0]
    print(f"  every arm but the labelled misses: max |r| {max(abs(a['res_reg']) for a in rest):.4f} MiB")
    for a in excluded:
        print(f"  EXCLUDED {a['job']} {a['arm']}  {a['res']:+.4f}  "
              f"{EXCLUDE_BLK[a['arm'].split('_')[0]]}")
    print("(b) spans")
    for label, (fo, fs), (xo, xs), (ro, rs), src in SPANS:
        print(f"  {label:<16} measured {fo}/{fs} = {fo / fs:.3f}  true ratio in"
              f" [{fo / xs:.3f}, {xo / fs:.3f})  registered {ro}/{rs} = {ro / rs:.3f}")


if __name__ == "__main__":
    main()
