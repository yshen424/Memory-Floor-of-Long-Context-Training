"""
fig:unit -- the recompute unit sets the functional form.  (design, single col)

One 2-panel float.  One question -- "the recompute unit sets the functional
form" -- answered as cause (a) beside consequence (b).  The two panels share
their category set DRAWN, not asserted: per-module is the same grey in both,
whole-layer the same blue in both.

PANEL (a) -- what one unit of recompute OWNS, per layer per 1024 tokens.

  per-module   22.04688 MiB/layer/x  (attention 6.04688 + MLP 16.00000),
               of which 3.00 is migratable -- the inbound residual only.
  whole-layer   1.5000 MiB/layer/x, and ALL of it is migratable: the only
               save the seam is offered is the residual-stream boundary.
               1.5000 = 2 B x d_model x 1024 -- arithmetic, not a fit.

  Per-module: read off the bundle registry of a retaining per-module arm
  (7941, x = 2 and 4: all 128 bundles trusted, one 1.5x MiB inbound storage
  each), and its peak gap to the recompute twin agrees to 0.0001 MiB/layer/x.
  Whole-layer: 2 B x 768 per token, arithmetic.

PANEL (b) -- achievable span at perfect relief against depth.

    span(L) = slope(L) / (slope(L) - migratable(L))
    whole-layer:  1 + 1.50 L / n              LINEAR IN DEPTH, UNBOUNDED
    per-module:   (440.77 + 6.0078 L) / (440.77 + 3.0078 L)  ->  1.9974

  n = 440.77 MiB/x with chunked CE off (7744 block U), 36.3203 with chunked CE
  (harness arithmetic; the four-rank lines at L=32/64/128 agree to 0.0003).
  L=128 on the chunked-CE curve is a MEASURED (filled) point (job 7761: D1
  static slope 228.3201 -> span 6.2863).

Every point is a world-4 line.  Per-module is drawn WITH the chunked loss,
n = 421.81 - 6.0078*64 = 37.31 (7747 per-module line 1485.80 + 421.81x,
x=32/33; per-module at L=32, 7938, gives the per-layer 6.0078 from two
depths); whole-layer without chunking is measured at L=64 only (7744,
1414.92 + 536.77x).  The per-layer constants (6.0078 at the peak, 3.0000
migratable) are byte terms.

-----------------------------------------------------------------------------
THE BINDING HAZARD.  Panel (a)'s bars are OWNED mass; panel (b)'s spans come
from the PEAK-SLOPE per-layer terms (per-module 6.0078 at the peak against a
3.0078 non-migratable remainder).  Under whole-layer the two coincide at
1.5000; under per-module they do not.  Adjacent panels invite a reader to read
(b) off (a)'s bar heights, so the CAPTION states the relation once,
positively.  Nothing in this drawing may imply the spans are read off the
bars -- which is why panel (a) carries no span number at all: the L=64 spans
sit on their own curves in (b), where they belong.

THE TRAP.  Per-module's span is LARGER at every measured depth while its
absolute capability is SMALLER at every depth: whole-layer with no migration
beats per-module with perfect migration by 96.50 MiB/x at L=64.  Per-module
is grey, dashed, "contrast", never a baseline; the absolute verdict is a
caption obligation.

THE CEILING prints to THREE decimals: 1.997.  At two it renders "2.00" and
destroys the only thing the line says.

AXIS OF (b): t = log2(y - Y_SHIFT).  A plain log2 axis gives 1-2x a third of
the height and the two lower curves (1.1-1.5x) merge; the shifted origin
stretches the low band smoothly and fades to plain log2 by ~4x, with no kink.
The caption must name it: whole-layer growth is linear in L and this axis
draws it concave; only per-module saturates.

SIZING: FIG_W = SCALE * ps.COL_W, included at width=SCALE\\columnwidth, so the
scale factor is still 1 and the FS_* sizes are the printed sizes:
SCALE = 0.85 -> 2.8305 x 2.64 in.
No tight_layout(), no bbox_inches="tight"; axes placed in inches.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# --- (a) measured owned mass, MiB per layer per 1024 tokens ------------------
PM_OWNED = 22.04688          # attention 6.04688 + MLP 16.00000 (7941 registry)
PM_MIGRATABLE = 3.00000      # inbound residual only, 1.5000 x 2 bundles/layer (7941)
WL_OWNED = 1.50000           # one fp16 residual-stream boundary
WL_MIGRATABLE = 1.50000      # all of it
BYTES_PER_ELEM, D_MODEL, TOKENS = 2, 768, 1024

# --- (b) measured slope terms, MiB per 1024 tokens ---------------------------
N_LEVER2 = 440.77            # non-layer, chunked CE OFF (7744 block U: 536.77 - 1.50*64)
N_LEVER3 = 36.3203           # chunked CE ON: harness arithmetic, depth-independent
MIG_WHOLE_LAYER = 1.5000
PM_PER_LAYER = 6.0078        # per-module, at the PEAK -- not owned: (421.8125 - 229.5625)/32 from the L=64 (7747, 7938 R) and L=32 (7938 P) lines
PM_MIG_PER_LAYER = 3.0000    # two unit inputs x 1.5000, arithmetic
PM_CEILING = PM_PER_LAYER / (PM_PER_LAYER - PM_MIG_PER_LAYER)   # 1.9974

L_MEASURED_LEVER2 = [64]           # world 4: 7744 unchunked line
L_MEASURED_LEVER3 = [32, 64, 128]   # 128: 7761, span 6.2863
L_MEASURED_PM = [64]               # world 4: 7747, chunked loss
PM_SLOPE_L64 = 421.8125            # 7747 per-module line, x=32/33
N_PM = PM_SLOPE_L64 - PM_PER_LAYER * 64   # 37.31, chunked loss
L_REF = 64                   # where the chunked-CE-off contrast is annotated

# --- canvas: every length in inches, converted once --------------------------
SCALE = 0.85                             # printed at width=0.85\columnwidth
FIG_W, FIG_H = SCALE * ps.COL_W, 2.64    # 2.8305 x 2.64 in
PX_W = int(round(FIG_W * ps.DPI))        # 1698 px, verified on save
AX_L, AX_R = 0.45, 0.06                  # left/right margins, shared by both panels
A_BOTTOM, A_H = 1.82, 0.68               # panel (a)
B_BOTTOM, B_H = 0.40, 0.96               # panel (b)
KEY_R = 0.03                             # (b)'s key hugs the right edge (in)
KEY_GAP = 0.10                           # gap between (b) and its key (in); (b)'s width follows the key

A_XLIM = (0.0, 25.2)
A_ROW_Y = {"per_module": 1.0, "whole_layer": 0.0}
A_BAR_H = 0.40
A_YLIM = (-0.30, 1.72)

X_LIM = (0.0, 140.0)
Y_SHIFT = 0.9
Y_LIM = (0.97, 8.6)
Y_MAJOR = [1, 1.25, 1.5, 2, 4, 8]
Y_MINOR = [1.1, 1.75, 3, 6]

# --- semantic colours: one per UNIT, the same in both panels -----------------
C_WHOLE = ps.ROLE["engine"]
C_PM = ps.ROLE["per_module"]
C_NONMIG = ps.PALETTE["neutral"]         # what the replay re-creates at the peak
C_CEILING = ps.ROLE["capacity"]
C_NOTE = ps.EXTRA["neutral_text"]
A_LEVER3, A_LEVER2 = ps.alpha_ramp(2, lo=0.55)

OUT_LOCAL = "fig5.png"
OUT_IN_USE = os.path.join("..", "rendered", "fig5.png")


def span_whole_layer(L, n):
    return (n + MIG_WHOLE_LAYER * L) / n


def span_per_module(L):
    return ((N_PM + PM_PER_LAYER * L)
            / (N_PM + (PM_PER_LAYER - PM_MIG_PER_LAYER) * L))


def _yfwd(y):
    return np.log2(np.maximum(np.asarray(y, dtype=float) - Y_SHIFT, 1e-9))


def _yinv(t):
    return np.power(2.0, np.asarray(t, dtype=float)) + Y_SHIFT


def _rect(bottom, height):
    return [AX_L / FIG_W, bottom / FIG_H,
            (FIG_W - AX_L - AX_R) / FIG_W, height / FIG_H]


def panel_owned(ax):
    """(a): horizontal stacks, one row per unit, unit name above its bar."""
    pm_nonmig = PM_OWNED - PM_MIGRATABLE
    y_pm, y_wl = A_ROW_Y["per_module"], A_ROW_Y["whole_layer"]
    edge = dict(edgecolor="black", linewidth=ps.LW_EDGE, zorder=3)

    ax.barh(y_pm, PM_MIGRATABLE, height=A_BAR_H, color=C_PM, **edge)
    ax.barh(y_pm, pm_nonmig, left=PM_MIGRATABLE, height=A_BAR_H,
            color=C_NONMIG, **edge)
    ax.barh(y_wl, WL_MIGRATABLE, height=A_BAR_H, color=C_WHOLE, **edge)

    # In-place values -- the house pattern; no legend for two segments.
    ax.text(PM_MIGRATABLE / 2, y_pm, f"{PM_MIGRATABLE:.2f}", ha="center",
            va="center", fontsize=ps.FS_ANNOT, color="white", zorder=5)
    ax.text(PM_MIGRATABLE + pm_nonmig / 2, y_pm,
            f"non-migratable {pm_nonmig:.2f}", ha="center", va="center",
            fontsize=ps.FS_ANNOT, color="black", zorder=5)
    ax.text(PM_OWNED + 0.35, y_pm, f"{PM_OWNED:.2f}", ha="left", va="center",
            fontsize=ps.FS_ANNOT, color="black")
    # The punch: the non-migratable block has zero height, not a small one.
    ax.text(WL_OWNED + 0.45, y_wl,
            f"{WL_OWNED:.2f}, all migratable;  non-migratable: exactly 0",
            ha="left", va="center", fontsize=ps.FS_ANNOT, color="black")

    # Unit names above their rows (categorical abstracting: no y ticks), so
    # both panels keep one shared left edge.
    ax.text(0.0, y_pm + A_BAR_H / 2 + 0.07,
            "per-module: only the inbound residual is migratable",
            ha="left", va="bottom", fontsize=ps.FS_ANNOT, color=C_PM)
    ax.text(0.0, y_wl + A_BAR_H / 2 + 0.07,
            "whole-layer: one fp16 residual boundary",
            ha="left", va="bottom", fontsize=ps.FS_ANNOT, color=C_WHOLE)

    ax.set_xlim(*A_XLIM)
    ax.set_ylim(*A_YLIM)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(axis="x", pad=2)
    ax.set_xlabel("owned per layer (MiB per 1024 tokens)", labelpad=2)


def panel_span(ax):
    """(b): achievable span at perfect relief against depth."""
    L = np.linspace(0.0, X_LIM[1], 800)

    ax.axhspan(Y_LIM[0], PM_CEILING, color=C_NONMIG, alpha=0.30,
               linewidth=0, zorder=0)
    ax.axhline(PM_CEILING, color=C_CEILING, linewidth=ps.LW_RULE,
               linestyle=(0, (3.2, 1.8)), zorder=2)
    # The limit's label sits in the key (on the plot it collides with the
    # L=32 point).

    ax.plot(L, span_whole_layer(L, N_LEVER3), color=C_WHOLE, alpha=A_LEVER3,
            linewidth=1.7, solid_capstyle="round", zorder=5,
            label="whole-layer + chunked CE")
    ax.plot(L, span_whole_layer(L, N_LEVER2), color=C_WHOLE, alpha=A_LEVER2,
            linewidth=1.7, solid_capstyle="round", zorder=4,
            label="whole-layer")
    ax.plot(L, span_per_module(L), color=C_PM, linewidth=1.3,
            linestyle=(0, (4.0, 2.0)), zorder=3, label="per-module (contrast)")

    mk = dict(markeredgewidth=1.1, linestyle="none", zorder=6)
    ax.plot(L_MEASURED_LEVER3,
            [span_whole_layer(x, N_LEVER3) for x in L_MEASURED_LEVER3],
            marker="o", markersize=4.4, color=C_WHOLE, alpha=A_LEVER3,
            markerfacecolor=C_WHOLE, **mk)
    ax.plot(L_MEASURED_LEVER2,
            [span_whole_layer(x, N_LEVER2) for x in L_MEASURED_LEVER2],
            marker="s", markersize=3.8, color=C_WHOLE, alpha=A_LEVER2,
            markerfacecolor=C_WHOLE, **mk)
    ax.plot(L_MEASURED_PM, [span_per_module(x) for x in L_MEASURED_PM],
            marker="^", markersize=4.0, color=C_PM, markerfacecolor=C_PM, **mk)

    # 2.32x sits right-aligned ON the L=32 point (offset left it hits the y
    # spine); 6.29x has no leader and sits under its own point (as one row it
    # runs into 3.64x).
    for depth, off, ha, va in ((32, (0, 4.5), "right", "bottom"),
                               (64, (5, -2), "left", "top"),
                               (128, (-3, -5.5), "center", "top")):
        ax.annotate(f"{span_whole_layer(depth, N_LEVER3):.2f}$\\times$",
                    xy=(depth, span_whole_layer(depth, N_LEVER3)),
                    xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=ps.FS_ANNOT, color=C_WHOLE)

    # The trap, drawn at one depth with the lever state held: the taller bar
    # in (a) has the BETTER ratio here.  Chunked CE off on both.  1.835x sits
    # below-LEFT of its point: below-right it hits the whole-layer curve.
    ax.annotate(f"{span_per_module(L_REF):.3f}$\\times$",
                xy=(L_REF, span_per_module(L_REF)), xytext=(-4, -6.5),
                textcoords="offset points", ha="right", va="top",
                fontsize=ps.FS_ANNOT, color=C_PM)
    ax.annotate(f"{span_whole_layer(L_REF, N_LEVER2):.3f}$\\times$",
                xy=(L_REF, span_whole_layer(L_REF, N_LEVER2)), xytext=(4, -4),
                textcoords="offset points", ha="left", va="top",
                fontsize=ps.FS_ANNOT, color=C_WHOLE, alpha=A_LEVER2 + 0.2)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks([0, 32, 64, 96, 128])
    ax.set_yscale("function", functions=(_yfwd, _yinv))
    ax.yaxis.set_major_locator(mticker.FixedLocator(Y_MAJOR))
    ax.yaxis.set_minor_locator(mticker.FixedLocator(Y_MINOR))
    ax.yaxis.set_major_formatter(
        mticker.FixedFormatter([f"{v:g}" for v in Y_MAJOR]))
    ax.tick_params(pad=2)
    ax.set_xlabel("depth $L$ (layers)", labelpad=2)
    # centred below mid-height so its top clears the (b) label
    ax.set_ylabel(r"span at $\varphi = 1$  ($\times$)", labelpad=3, y=0.42)
    # The key: a boxed legend OUTSIDE the plot, to the
    # right of (b), for the three curves and the measured points.
    from matplotlib.legend_handler import HandlerTuple
    lines = [Line2D([], [], color=C_WHOLE, alpha=A_LEVER3, linewidth=1.7),
             Line2D([], [], color=C_WHOLE, alpha=A_LEVER2, linewidth=1.7),
             Line2D([], [], color=C_PM, linewidth=1.3, linestyle=(0, (4.0, 2.0))),
             Line2D([], [], color=C_CEILING, linewidth=ps.LW_RULE,
                    linestyle=(0, (3.2, 1.8)))]
    pts = tuple(Line2D([], [], marker=m, markersize=sz, color=c, alpha=a,
                       linestyle="none", markeredgewidth=0.9)
                for m, sz, c, a in (("o", 3.6, C_WHOLE, A_LEVER3),
                                    ("s", 3.1, C_WHOLE, A_LEVER2),
                                    ("^", 3.4, C_PM, 1.0)))
    fig = ax.figure
    key = fig.legend(lines + [pts],
                     ["whole-layer\n+ chunked CE", "whole-layer",
                      "per-module\n+ chunked CE\n(contrast)",
                      f"per-module\nlimit {PM_CEILING:.3f}$\\times$", "measured"],
                     loc="center right",
                     bbox_to_anchor=(1.0 - KEY_R / FIG_W, (B_BOTTOM + B_H / 2) / FIG_H),
                     handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
                     handlelength=1.9, handletextpad=0.35, labelspacing=0.3,
                     borderpad=0.3, borderaxespad=0.0, fontsize=ps.TYPE_FLOOR)
    # (b) takes every inch the narrow key leaves.
    fig.canvas.draw()
    key_left = key.get_window_extent().x0 / fig.dpi
    ax.set_position([AX_L / FIG_W, B_BOTTOM / FIG_H,
                     (key_left - KEY_GAP - AX_L) / FIG_W, B_H / FIG_H])


def main():
    ps.apply_publication_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_a = fig.add_axes(_rect(A_BOTTOM, A_H))
    ax_b = fig.add_axes(_rect(B_BOTTOM, B_H))   # narrowed to fit the key in panel_span

    panel_owned(ax_a)
    panel_span(ax_b)
    ps.panel_label(ax_a, "(a)", x=-AX_L / (FIG_W - AX_L - AX_R) + 0.005, y=1.0)
    ps.panel_label(ax_b, "(b)", x=-AX_L / (ax_b.get_position().width * FIG_W) + 0.005,
                   y=1.0)

    here = os.path.dirname(os.path.abspath(__file__))
    ps.save(fig, OUT_LOCAL, OUT_IN_USE, script_dir=here, expect_px=PX_W)
    plt.close(fig)

    pm_nonmig = PM_OWNED - PM_MIGRATABLE
    print(f"  (a) per-module owned {PM_OWNED:.5f}  migratable {PM_MIGRATABLE:.5f}"
          f"  non-migratable {pm_nonmig:.5f}")
    print(f"  (a) whole-layer owned {WL_OWNED:.5f}  migratable {WL_MIGRATABLE:.5f}"
          f"  non-migratable {WL_OWNED - WL_MIGRATABLE:.5f}")
    print(f"  (a) owned-mass ratio {PM_OWNED / WL_OWNED:.3f}x;  1.5000 check: "
          f"{BYTES_PER_ELEM * D_MODEL * TOKENS / 1048576:.4f} MiB")
    print(f"  (b) per-module ceiling (L -> inf)   {PM_CEILING:.4f}")
    for depth in (32, 64, 128):
        print(f"  (b) L={depth:<4d} whole loss {span_whole_layer(depth, N_LEVER2):.4f}"
              f"   chunked loss {span_whole_layer(depth, N_LEVER3):.4f}"
              f"   per-module {span_per_module(depth):.4f}")


if __name__ == "__main__":
    main()
