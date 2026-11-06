"""
fig:frontier -- the headline measurement.  (eval, single column)

WORLD 4 (headline: 3.423 = 356/104, jobs 7743/7744).

One panel:
  THE HEADLINE, full token range: static whole-layer checkpointing (its
      measured line points, x=8 excluded), the engine's W=0 floor with every
      W=0 rung (x=108/140/280 below the pick cap, 328/356 on the floor), the
      capacity bracket, the span arrow between the two largest fits, and the
      two failures that end the span.

Encoding: static in house neutral; the engine in blue; every rung filled
(all measured); red is capacity only.

WHAT EACH ELEMENT IS (world 4, L=64, c=256, regime B, v=32000, b=1; MiB,
x = tokens/1024; inputs in ../world4.py):
  static          1476.80 + 132.320279x; slope world-invariant (7744 S, max
                  residual 0.006 over x=16/24/48/104).  Fits x=104, OOMs 108.
  floor W=0       1866.89 + 37.820x, 63 picks.  Slope derived; the intercept
                  is the one measured on its own frontier rungs (two x, zero
                  residual), 12.00 MiB below the intercept derived from
                  parameter counts -- the step-1 gradient pile grows by 3
                  shards at world 4, not 5.  THE FLOOR IS NOT A FIT.
  bracket         W=0's own: largest fit 15330.92 (x=356) to ~15438 (x=360:
                  13.67 GiB live + a 1440.00 MiB request, refused).  Never an
                  intersection of the depths' brackets.
  stock offload   measured peaks (7747, world 4: x=100/200/356) on the floor
                  line, so the text's "both claims measured here" holds; the
                  forced all-64 arm at x=100 equals offload's peak to the byte
                  (same marker).

HONESTY CONSTRAINTS: an OOM does not bound capacity from above (upper edge
dashed, "~" in the caption); only the two failures that end the headline span
are drawn; 3.6432 is an allowance and stays in the text, not here; the
peak drawn per rung is the MAXIMUM over the run's steps.

SIZING: FIG_W = SCALE * ps.COL_W = 2.8305 in, included at width=0.85\\columnwidth,
so the scale factor is still 1 (type, lines and markers are the printed sizes);
emitted 1698 px at 600 dpi.  No tight_layout / bbox tight.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MIB_PER_GIB = 1024.0

import world4 as w4

# --- lines, MiB, x = tokens/1024, b=1, L=64, WORLD 4 ---------------------------
C_L3, M_L3 = w4.C_STATIC, w4.M_STATIC       # chunked CE ON (7744 block S)
C_FLOOR_W0, M_FLOOR_W0, _ = w4.FLOORS[0]

X_MEASURED_L3 = [16, 24, 48, 104]

# --- capacity bracket: the HEADLINE config's own, never an intersection -------
CAP_LO_MIB = w4.PEAK[(0, 356)]   # largest fit, engine x=356 at W=0 -- achieved
CAP_HI_MIB = 15438.1             # 13.67 GiB live + a 1440.00 MiB request, refused
CAP_LO = CAP_LO_MIB / MIB_PER_GIB
CAP_HI = CAP_HI_MIB / MIB_PER_GIB

# --- engine rungs: (x, max-over-steps peak MiB, picks at that step) -----------
ENGINE_W0 = [(108, 14957.39, 5),
             (140, 14751.64, 25),
             (280, 14166.48, 58),
             (328, 14271.95313, 63),
             (356, 15330.92188, 63)]
# stock offload, world 4 (7747): x -> peak MiB, max over steps
OFFLOAD = [(100, 5648.9219), (200, 9430.9531), (356, 15330.9219)]

X_STATIC_FIT, X_STATIC_FAIL = w4.X_STATIC_FIT, w4.X_STATIC_FAIL
X_ENGINE_FIT, X_ENGINE_FAIL = w4.X_ENGINE_FIT[0], w4.X_ENGINE_FAIL[0]
SPAN_MEASURED = X_ENGINE_FIT / X_STATIC_FIT         # 3.4231

X_MEET_L3 = (CAP_LO_MIB - C_L3) / M_L3
X_MEET_W0 = (CAP_LO_MIB - C_FLOOR_W0) / M_FLOOR_W0  # 356.0, exact

# --- canvas: inches -----------------------------------------------------------
# 85% of the column in both directions: LaTeX includes it at
# width=0.85\columnwidth, so the canvas is authored at that width and the type
# stays at scale 1.  Label margins are full size; only the data area is smaller.
SCALE = 0.85
FIG_W, FIG_H = SCALE * ps.COL_W, 1.46     # 2.8305 in
PX_W = int(round(FIG_W * ps.DPI))         # 1698
AX_L, AX_R = 0.46, 0.06
A_BOTTOM, A_H = 0.36, 1.01
X_LIM = (0.0, 412.0)
Y_LIM = (0.0, 18.0)
Y_SPAN_BAR = 16.55   # above the band: the floor crosses every horizontal below
                     # it inside [104, 328], which would read as an intersection

# --- semantic colours ----------------------------------------------------------
C_STATIC = ps.EXTRA["neutral_text"]     # the baseline: house neutral
C_ENGINE = ps.ROLE["engine"]
C_CEIL = ps.EXTRA["neutral_line"]
C_CAP = ps.ROLE["capacity"]

OUT_LOCAL = "fig10.png"
OUT_IN_USE = os.path.join("..", "rendered", "fig10.png")


def line_gib(x, c, m):
    return (c + m * np.asarray(x, dtype=float)) / MIB_PER_GIB


def clipped(c, m, xlim=X_LIM, ytop=Y_LIM[1]):
    x_top = (ytop * MIB_PER_GIB - c) / m
    x = np.linspace(xlim[0], min(xlim[1], x_top), 400)
    return x, line_gib(x, c, m)


def _rect(bottom, height):
    return [AX_L / FIG_W, bottom / FIG_H,
            (FIG_W - AX_L - AX_R) / FIG_W, height / FIG_H]


def band(ax, label_x=None):
    ax.axhspan(CAP_LO, CAP_HI, facecolor=C_CAP, alpha=0.30, edgecolor="none",
               zorder=1)
    ax.axhline(CAP_LO, color=C_CAP, linewidth=0.8, zorder=2)
    ax.axhline(CAP_HI, color=C_CAP, linewidth=0.8, linestyle=(0, (2.0, 1.6)),
               zorder=2)


def rung(ax, x, y, size=4.2):
    ax.plot(x, y, marker="D", markersize=size, linestyle="none",
            color=C_ENGINE, markeredgecolor="black",
            markeredgewidth=0.6, zorder=7)


def failure(ax, x, y):
    ax.plot([x], [y], marker="x", markersize=6.2, markeredgewidth=2.6,
            linestyle="none", color="white", zorder=8)
    ax.plot([x], [y], marker="x", markersize=4.8, markeredgewidth=1.3,
            linestyle="none", color=C_CAP, zorder=9)


def panel_headline(ax):
    band(ax)
    # Upper-left corner, two lines: at 1.28 in the band-to-arrow gap is too
    # short for the label.
    ax.text(8.0, CAP_HI + 0.30, "capacity\nbracket", ha="left",
            va="bottom", fontsize=ps.FS_ANNOT, color=C_CAP, linespacing=0.95)

    ax.plot(*clipped(C_L3, M_L3), color=C_STATIC, linewidth=1.6, zorder=5)
    ax.plot(X_MEASURED_L3, line_gib(X_MEASURED_L3, C_L3, M_L3), marker="s",
            markersize=3.3, linestyle="none", color=C_STATIC,
            markeredgecolor="black", markeredgewidth=0.5, zorder=6)
    ax.plot(*clipped(C_FLOOR_W0, M_FLOOR_W0), color=C_ENGINE, linewidth=1.6,
            zorder=5)
    rung(ax, [e[0] for e in ENGINE_W0], [e[1] / MIB_PER_GIB for e in ENGINE_W0])
    ax.plot([o[0] for o in OFFLOAD], [o[1] / MIB_PER_GIB for o in OFFLOAD],
            marker="o", markersize=2.4, linestyle="none", color="black",
            zorder=8)

    for x_fail, c, m in ((X_STATIC_FAIL, C_L3, M_L3),
                         (X_ENGINE_FAIL, C_FLOOR_W0, M_FLOOR_W0)):
        failure(ax, x_fail, float(line_gib([x_fail], c, m)[0]))

    for x_end in (X_STATIC_FIT, X_ENGINE_FIT):
        ax.plot([x_end, x_end], [CAP_HI, Y_SPAN_BAR], color=C_CEIL,
                linewidth=0.7, linestyle=(0, (1.6, 1.6)), zorder=3)
    ax.annotate("", xy=(X_ENGINE_FIT, Y_SPAN_BAR),
                xytext=(X_STATIC_FIT, Y_SPAN_BAR),
                arrowprops=dict(arrowstyle="<|-|>", color=C_ENGINE,
                                linewidth=1.1, mutation_scale=7,
                                shrinkA=0, shrinkB=0))  # tips at 104 / 356
                                # exactly; the default 2 pt inset put the
                                # left head on the static line
    # Centred right of the arrow's midpoint (230): at a 2.31 in data width the
    # label's left end otherwise meets the static line above the band.
    ax.text(250.0, Y_SPAN_BAR + 0.20,
            f"$x{{=}}{X_STATIC_FIT} \\rightarrow {X_ENGINE_FIT}$:  "
            f"$\\mathbf{{{SPAN_MEASURED:.3f}}}\\mathbf{{\\times}}$",
            ha="center", va="bottom", fontsize=ps.FS_ANNOT, color=C_ENGINE)

    # Frameless, right-aligned (handles on the right), short labels on top: the
    # key's ink is a staircase that fits the triangle under the floor line; a
    # boxed key at this width crosses the floor near x=150-200.
    handles = [
        Line2D([], [], color=C_STATIC, linewidth=1.6, marker="s",
               markersize=3.3, markeredgecolor="black", markeredgewidth=0.5,
               label="static baseline"),
        Line2D([], [], color="black", linewidth=0, marker="o",
               markersize=2.4, label="stock offload"),
        Line2D([], [], color=C_ENGINE, linewidth=1.6, marker="D",
               markersize=4.0, markeredgecolor="black", markeredgewidth=0.6,
               label="instrument: floor and rungs"),
        Line2D([], [], color=C_CAP, linewidth=0, marker="x", markersize=4.8,
               markeredgewidth=1.3, label="smallest shape that OOMs"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0),
              frameon=False, markerfirst=False, handlelength=1.5,
              handletextpad=0.5, labelspacing=0.15, borderpad=0.2,
              borderaxespad=0.2)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel("$x$ (1024 tokens per rank)", labelpad=2)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_ylabel("peak HBM (GiB)", labelpad=3)


def main():
    ps.apply_publication_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    a = fig.add_axes(_rect(A_BOTTOM, A_H))
    panel_headline(a)

    here = os.path.dirname(os.path.abspath(__file__))
    ps.save(fig, OUT_LOCAL, OUT_IN_USE, script_dir=here, expect_px=PX_W)
    plt.close(fig)

    print(f"  bracket            [{CAP_LO_MIB:.3f}, ~{CAP_HI_MIB:.1f}] MiB"
          f"  = [{CAP_LO:.4f}, {CAP_HI:.4f}] GiB, {CAP_HI_MIB - CAP_LO_MIB:.1f}"
          " MiB wide")
    print(f"  meets bracket at   +chunked CE x={X_MEET_L3:.2f}"
          f"   W=0 x={X_MEET_W0:.2f} (exact)")
    print(f"  span, measured     {X_ENGINE_FIT}/{X_STATIC_FIT}"
          f" = {SPAN_MEASURED:.4f}   (crossings give"
          f" {X_MEET_W0 / X_MEET_L3:.4f})")
    print(f"  ratio of ceilings  [{X_ENGINE_FIT / X_STATIC_FAIL:.3f},"
          f" {X_ENGINE_FAIL / X_STATIC_FIT:.3f})")
    for x, peak in OFFLOAD:
        print(f"  offload x={x:<4d} peak {peak:9.4f}  floor"
              f" {C_FLOOR_W0 + M_FLOOR_W0 * x:9.4f}"
              f"  resid {peak - C_FLOOR_W0 - M_FLOOR_W0 * x:+.4f}")
    for x, peak, picks in ENGINE_W0:
        floor = C_FLOOR_W0 + M_FLOOR_W0 * x
        print(f"  W0 x={x:<4d} peak {peak:9.4f}  picks {picks:2d}"
              f"  static line {C_L3 + M_L3 * x:9.2f}"
              f"  floor {floor:9.2f}  resid {peak - floor:+8.4f}")
    for tag, x_fail, c, m in (("static", X_STATIC_FAIL, C_L3, M_L3),
                              ("W0", X_ENGINE_FAIL, C_FLOOR_W0, M_FLOOR_W0)):
        print(f"  failure {tag:<6s} x={x_fail} at"
              f" {c + m * x_fail:9.2f} MiB"
              f"  ({c + m * x_fail - CAP_HI_MIB:+.2f} vs band top)")


if __name__ == "__main__":
    main()
