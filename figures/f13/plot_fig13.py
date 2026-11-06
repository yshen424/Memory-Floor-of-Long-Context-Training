"""
fig:dose -- the dose-response: relief against boundaries moved.  (single column)

EVERY FILLED MARKER IS MEASURED, world 4, L=64, step 1 (the binding step):
  y = (static line at x - measured peak) / (1.50 x MiB),
i.e. the relief in boundaries' worth at that x.  Static line 1476.80 +
132.3201x (7744 S); with CPU offload 183.89 + 132.3201x (7756).
  engine W=0, below the cap: x=100 / 108 / 140 / 280 -> 0 / 5 / 25 / 58 picks
      (7747 M4, 7744 E)
  engine at the cap: W=0 x=328, 356 (63 picks); W=1 x=328, 340, 344 (62);
      W=2 x=328 (61) (7743, 7744); W=0 + CPU offload x=388 (63, 7756)
  force-evict 64/64: x=100 (7747 M3), and + CPU offload x=100 (7756 M3)
OPEN MARKERS are the model: picks below the cap; at the cap
  (L-W-1) - (c_floor,W - c_static)/(1.50x), the relocated instant's extra
  intercept (two gathered layers + accumulated gradient shards) from
  parameter counts with the W-1 shard count (harness/terms.py).
  The registered E-P3 floor sat 12.00 MiB higher (the step-1 floor miss):
  0.02 boundaries at x=356, below what the panel can show; the caption says so.
The grey cross is what the 64th boundary would give if it counted.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
import world4 as w4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

L = 64
B = 1.50                                  # MiB per boundary per x
C_STATIC, M = w4.C_STATIC, w4.M_STATIC
C_STATIC_CPU = 183.89                     # 7756 static+cpu intercept
C_FLOOR = {W: c for W, (c, _m, _n) in w4.FLOORS.items()}
C_FLOOR_CPU = 573.98                      # 7756 offload+cpu = engine+cpu floor

# (label, x, boundaries moved, measured peak MiB, depth W or None, cpu)
POINTS = [
    ("below", 100, 0, 14708.8267, 0, False),           # 7747 M4, gate moves 0
    ("below", 108, 5, w4.PEAK[(0, 108)], 0, False),
    ("below", 140, 25, w4.PEAK[(0, 140)], 0, False),
    ("below", 280, 58, w4.PEAK[(0, 280)], 0, False),
    ("cap", 328, 63, w4.PEAK[(0, 328)], 0, False),
    ("cap", 356, 63, w4.PEAK[(0, 356)], 0, False),
    ("cap", 328, 62, w4.PEAK[(1, 328)], 1, False),
    ("cap", 340, 62, w4.PEAK[(1, 340)], 1, False),
    ("cap", 344, 62, w4.PEAK[(1, 344)], 1, False),
    ("cap", 328, 61, w4.PEAK[(2, 328)], 2, False),
    ("cap", 388, 63, 15248.26, 0, True),               # 7756 engine+cpu
    ("force", 100, 64, 5648.9219, 0, False),           # 7747 M3
    ("force", 100, 64, 4356.01, 0, True),              # 7756 M3
]

C_ENG = ps.ROLE["engine"]
C_DEPTH = {0: ps.PALETTE["blue_main"], 1: ps.PALETTE["blue_secondary"],
           2: ps.PALETTE["blue_secondary"]}
C_FORCE = ps.EXTRA["neutral_text"]
C_CAP = ps.ROLE["capacity"]
C_NOTE = ps.EXTRA["neutral_text"]
MS = 4.0

FIG_W, FIG_H = ps.COL_W, 2.08
A_BOX = (0.42, 0.38, 1.50, 1.45)          # left, bottom, width, height (in)
B_BOX = (2.20, 0.38, 1.08, 1.45)
OUT_LOCAL = "fig13.png"
# Cap groups are dodged sideways; the zoom is a shared tint.
OUT_IN_USE = os.path.join("..", "rendered", "fig13.png")
DODGE = 0.30                               # boundaries; within-group spread
# (b) one horizontal x label per cap row, values in the order of the points:
# moved -> (W, anchor in boundaries, offset from the row's lowest relief, ha,
# "W = k" stacked over the values?, print "x =?").  W=0 keeps its label left
# of the row; W=1 has no room there (it crossed the spine) and W=2's point
# sits at the left edge, so their labels head the stack.
ROW_LABEL = {63: (0, 63.0, -0.13, "center", False, True),
             62: (1, 62.0, -0.12, "center", True, False),
             61: (2, 61.25, +0.14, "left", True, True)}


def relief(x, peak, cpu):
    c = C_STATIC_CPU if cpu else C_STATIC
    return (c + M * x - peak) / (B * x)


def model(kind, x, n, W, cpu):
    if kind == "below":
        return float(n)
    usable = L - max(W, 0) - 1             # the 64th never counts
    c_floor = C_FLOOR_CPU if cpu else C_FLOOR[W]
    c_static = C_STATIC_CPU if cpu else C_STATIC
    return usable - (c_floor - c_static) / (B * x)


def draw(ax, p, zoom):
    kind, x, n, peak, W, cpu = p
    y, yp = relief(x, peak, cpu), model(kind, x, n, W, cpu)
    c = C_FORCE if kind == "force" else C_DEPTH[W]
    ax.plot(n, yp, marker="D", markersize=MS + 2.2, markerfacecolor="none",
            markeredgecolor=c, markeredgewidth=0.7, linestyle="none", zorder=3)
    ax.plot(n, y, marker="D", markersize=MS, color=c, markeredgecolor="black",
            markeredgewidth=0.4, linestyle="none", zorder=4)
    if cpu:
        ax.plot(n, y, marker="o", markersize=1.3, color="white",
                linestyle="none", zorder=5)
    return y, yp


def main():
    ps.apply_publication_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([A_BOX[0] / FIG_W, A_BOX[1] / FIG_H, A_BOX[2] / FIG_W,
                       A_BOX[3] / FIG_H])
    bx = fig.add_axes([B_BOX[0] / FIG_W, B_BOX[1] / FIG_H, B_BOX[2] / FIG_W,
                       B_BOX[3] / FIG_H])

    print("kind   x    moved  W cpu   peak(MiB)   relief  model   meas-model(MiB)")
    for p in POINTS:
        y, yp = draw(ax, p, False)
        kind, x, n, peak, W, cpu = p
        print(f"{kind:6s} {x:4d} {n:5d}  {W} {int(cpu)}   {peak:10.2f}  "
              f"{y:7.3f} {yp:7.3f}  {(yp - y) * B * x:+8.3f}")

    # (a) the diagonal: one boundary's worth per boundary moved, to the cap
    ax.plot([0, 63], [0, 63], color=C_ENG, linewidth=ps.LW_RULE,
            linestyle=(0, (3, 2)), zorder=2)
    ax.axhline(L - 1, color=C_CAP, linewidth=ps.LW_RULE, linestyle=":",
               zorder=1)
    ax.axvline(L - 1, color=C_CAP, linewidth=ps.LW_RULE, linestyle=":",
               zorder=1)
    ax.text(61, L - 1 + 1.5, "cap: $L{-}1 = 63$", ha="right", va="bottom",
            fontsize=ps.FS_ANNOT, color=C_CAP)
    for x, n, dx, ha in ((100, 0, 3.5, "left"), (108, 5, 3.5, "left"),
                         (140, 25, 3.5, "left")):
        ax.text(n + dx, n, f"$x{{=}}{x}$" + (" (none)" if n == 0 else ""),
                ha=ha, va="center", fontsize=ps.FS_ANNOT, color=C_NOTE)
    # x=280 sits under the legend's corner: labelled through a leader instead
    ax.annotate("$x{=}280$", xy=(57.5, 56.5), xytext=(61.5, 40.0), ha="right",
                va="center", fontsize=ps.FS_ANNOT, color=C_NOTE,
                arrowprops=dict(arrowstyle="-", color=C_NOTE, linewidth=0.5,
                                shrinkA=1, shrinkB=3))
    ax.set_xlim(-3, 69)
    ax.set_ylim(-3, 72)
    ax.set_xticks([0, 32, 64])
    ax.set_yticks([0, 16, 32, 48, 64])
    ax.set_xlabel("boundaries moved", labelpad=2)
    ax.set_ylabel("relief (boundaries' worth)", labelpad=2)
    # The zoom is marked by a shared tint, not a frame (a frame is too
    # faint): the region of (a) that (b) enlarges and the
    # whole of (b) carry the same light fill, keyed in (a)'s legend.
    Z = (60.5, 64.7, 59.15, 63.3)
    C_ZOOM = ps.PALETTE["highlight"]
    A_ZOOM = 0.18
    ax.add_patch(plt.Rectangle((Z[0] - 1.5, Z[2] - 1.5), Z[1] - Z[0] + 3.0,
                               Z[3] - Z[2] + 3.0, facecolor=C_ZOOM,
                               alpha=A_ZOOM, edgecolor="none", zorder=0))
    bx.set_facecolor((*matplotlib.colors.to_rgb(C_ZOOM), A_ZOOM))
    # the tint is keyed in the legend ("enlarged in (b)")

    # (b) the cap, magnified.  Points within one count are SPREAD SIDEWAYS
    # in order of x (their heights differ by 390 MiB / 1.5x, which is real);
    # the model markers are left to (a) -- each sits on its measured point to
    # 0.012 MiB (0.00002 boundaries).  The two forced-64 runs are identical
    # (with and without CPU offload) and are drawn as one marker.
    bx.axhline(L - 1, color=C_CAP, linewidth=ps.LW_RULE, linestyle=":")
    groups = {}
    for p in POINTS:
        if p[0] != "below":
            groups.setdefault(p[2], []).append(p)
    print("(b) dodged groups:")
    rows = {}                                # moved -> [(dodged pos, relief, x)]
    for n, ps_ in sorted(groups.items()):
        if n == 64:                          # identical pair -> one marker
            ps_ = ps_[:1]
        ps_ = sorted(ps_, key=lambda q: (q[1], q[5]))
        k = len(ps_)
        for i, (kind, x, _n, peak, W, cpu) in enumerate(ps_):
            xx = n + (i - (k - 1) / 2) * DODGE
            y = relief(x, peak, cpu)
            c = C_FORCE if kind == "force" else C_DEPTH[W]
            bx.plot(xx, y, marker="D", markersize=MS, color=c,
                    markeredgecolor="black", markeredgewidth=0.4,
                    linestyle="none", zorder=4)
            if cpu:
                bx.plot(xx, y, marker="o", markersize=1.3, color="white",
                        linestyle="none", zorder=5)
            if n != 64:
                rows.setdefault(n, []).append((xx, y, x))
            print(f"   moved {n}  x={x}{' +cpu' if cpu else ''}  at {xx:.2f}, relief {y:.3f}")
    y64 = 64 - (C_FLOOR[0] - C_STATIC) / (B * 100)
    bx.plot(64, y64, marker="x", markersize=MS + 0.5,
            color=ps.PALETTE["neutral"], markeredgewidth=0.9,
            linestyle="none", zorder=3)
    print(f"if the 64th counted at x=100: {y64:.3f}")
    bx.text(64.12, y64 - 0.14, "64th, if\ncounted", ha="center", va="top",
            fontsize=7.0, color=ps.EXTRA["neutral_line"], linespacing=0.95)
    lab = dict(fontsize=ps.FS_ANNOT, color=C_NOTE, va="center")
    bx.text(62.5, 62.27, "$W{=}0$", ha="right", **lab)
    bx.text(64.0, 60.40 - 0.14, "all 64,\n$x{=}100$\n$\\pm$CPU", ha="center",
            va="top", fontsize=7.0, color=C_NOTE, linespacing=0.95)
    # One horizontal label per row ("W = k" over the x values, left to right
    # like the points), under the row or beside it: rotated per-point x labels
    # are unreadable at print size.
    for n, (W, lx, dy, ha, stack, prefix) in ROW_LABEL.items():
        r = sorted(rows[n])
        xs = ("$x{=}$" if prefix else "") + ", ".join(str(q[2]) for q in r)
        ly = min(q[1] for q in r) + dy
        bx.text(lx, ly, (f"$W{{=}}{W}$\n" if stack else "") + xs, ha=ha,
                linespacing=0.95, **{**lab, "va": "top"})
        print(f"   row label, moved {n}: W={W}, '{xs}' at ({lx}, {ly:.2f})")
    bx.set_xlim(60.5, 64.7)
    bx.set_ylim(59.15, 63.3)
    bx.set_xticks([61, 62, 63, 64])
    bx.set_yticks([60, 61, 62, 63])
    bx.set_xlabel("moved (tint in (a))", labelpad=2)

    top = (A_BOX[1] + A_BOX[3] + 0.06) / FIG_H
    fig.text(A_BOX[0] / FIG_W - 0.10, top, r"$\bf{(a)}$  every placement",
             ha="left", va="bottom", fontsize=ps.FS_ANNOT)
    fig.text(B_BOX[0] / FIG_W - 0.07, top, r"$\bf{(b)}$  at the cap",
             ha="left", va="bottom", fontsize=ps.FS_ANNOT)
    handles = [Line2D([], [], marker="D", color=C_DEPTH[0], markeredgecolor="black",
                      markeredgewidth=0.4, markersize=MS, linestyle="none"),
               Line2D([], [], marker="D", markerfacecolor="none",
                      markeredgecolor=C_DEPTH[0], markeredgewidth=0.7,
                      markersize=MS + 2.2, linestyle="none"),
               Line2D([], [], marker="D", color=C_FORCE, markeredgecolor="black",
                      markeredgewidth=0.4, markersize=MS, linestyle="none")]
    handles.append(Patch(facecolor=ps.PALETTE["highlight"], alpha=0.35,
                         edgecolor="none"))
    ax.legend(handles, ["measured", "model", "forced: all 64", "enlarged in (b)"],
              loc="upper left", bbox_to_anchor=(0.02, 0.875), ncol=1, fontsize=7.0,
              handlelength=1.0, handletextpad=0.3, labelspacing=0.25)

    here = os.path.dirname(os.path.abspath(__file__))
    ps.save(fig, OUT_LOCAL, OUT_IN_USE, script_dir=here, expect_px=ps.PX_COL)
    plt.close(fig)


if __name__ == "__main__":
    main()
