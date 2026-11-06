"""
Fig. 1 (fig:hbm-split) -- peak HBM splits into fixed, migratable and
non-migratable mass, and each lever shrinks the third one.

The intro's one figure. It has to carry three things at once: the
decomposition, the design rule (every non-migratable term removed multiplies
what migration can reach), and the payoff (the span).

Every bar is built from a measured, allocator-observed peak-HBM line (DATA
below); none is derived analytically.

============================================================================
WHY THIS x AND WHY THREE BARS
============================================================================
The token count is the figure's only real design constraint, because the
three configurations have different SLOPES: pick x too high and one bar
dwarfs the others, which no linear axis can show. At X_TOKENS the levers are
still visibly separated, and exactly one bar crosses the ceiling -- which is
what makes the OOM band do any work at all. The two bars below the ceiling
are MEASURED at this x. The bar above it is the only projection, and it is
projected precisely because that configuration CANNOT run here -- which is
the annotation, not a defect.

THREE BARS, NOT FOUR. There is no "engine live" bar: at this x the
projection sits far below budget, so the gate picks nothing and an engine
bar would be a counterfactual drawn as a measurement. The engine's payoff is
the span annotation instead, which is legitimate because the span is
READABLE OFF THE BAR: it is (migratable + non-migratable) / non-migratable,
i.e. how much taller the stack could grow if the middle segment were made
absent. That is exactly what the engine does at phi = 1.0.

============================================================================
SIZING CONTRACT -- do not break this
============================================================================
FIG_W is the printed width (acmart sigconf \\columnwidth = 241.15 pt =
3.33 in, ps.COL_W), so the LaTeX scale factor is 1 and every FS_* is the point size on
the page. Embed with width=\\columnwidth and NO second scale. Margins are
set with subplots_adjust, so do not add tight_layout() or
bbox_inches="tight".
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================================
# DATA -- all measured. Edit here, never in the plotting code.
# ============================================================================
# Fitted peak-HBM lines, MiB against x = tokens/1024. Regime B
# (sdpa_mem_efficient, fp16), fused-input RMSNorm, --use_cache off,
# vocab 32000, WORLD 4, b = 1, L = 64.
#
#   per-module     7747: 1485.80 + 421.81x, chunked CE (see below)
#   whole-layer    7744 block U: 1414.92 + 536.77x, zero residual at x=2,4,8
#   + chunked CE   7744 block S: 1476.80 + 132.320279x, x in [16,104]
#
# MIGRATABLE is the unit-boundary mass and is measured separately, not
# backed out of the slope: 3.00 MiB/layer/x under the per-module unit (the
# inbound residual only) and 1.50 MiB/layer/x under the whole-layer unit
# (2 B x hidden x 1024). Chunked CE does not touch the boundary term, so
# bars 2 and 3 share it.
L_DEPTH = 64
# Every bar is a world-4 line.  The per-module bar carries the chunked loss
# (7747: 1485.80 + 421.81x, measured at x=32/33), so each of the top two bars
# undoes ONE design choice of the bottom bar: the unit (top) or the loss
# (middle).  At x=32 both fitting bars are MEASURED (14983.80, 5711.05) and
# exactly one bar (whole-layer, whole loss: 1414.92 + 536.77x, 7744) crosses
# the ceiling and is the projection.
X_TOKENS = 32  # tune: x = tokens/1024; see the docstring before changing

BARS = [
    # (tick label, intercept MiB, total slope MiB/x, migratable MiB/layer/x)
    ("per-module\n+ chunked CE", 1485.80, 421.8125, 3.00),
    ("whole-layer unit", 1414.92, 536.77, 1.50),
    ("whole-layer\n+ chunked CE", 1476.7989, 132.320279, 1.50),
]

# Usable HBM on one V100 (16 GB card). This is the CARD limit and
# is config-independent, which is why a single line across three different
# configurations is legitimate here. Effective capacity is lower and must be
# bracketed per config; the caption says so.
CEILING_GIB = 15.77

MIB_PER_GIB = 1024.0

# ============================================================================
# LAYOUT KNOBS -- edit these first
# ============================================================================

# HORIZONTAL bars, to keep the figure short.  A log axis is wrong for a
# stacked bar: on a log axis a segment's drawn length depends on where it sits
# in the stack, so a large non-migratable segment would draw barely longer
# than a much smaller migratable one -- the exact comparison this figure
# exists for.  Rows keep a linear axis and cost height per row, not per GiB.
FIG_W, FIG_H = ps.COL_W, 1.64  # tune: FIG_W must stay at \columnwidth
BAR_H = 0.56  # tune: bar thickness in row units (0..1)
X_MAX = 27.5  # tune: right edge, GiB; must clear the span labels
ROW_PAD = 0.52  # tune: padding above the first and below the last row

# --- Margins, in inches (converted once below) -------------------------------
IN_LEFT, IN_RIGHT, IN_BOTTOM, IN_TOP = 0.74, 0.06, 0.33, 0.20
M_LEFT, M_RIGHT = IN_LEFT / FIG_W, 1 - IN_RIGHT / FIG_W
M_BOTTOM, M_TOP = IN_BOTTOM / FIG_H, 1 - IN_TOP / FIG_H

# --- Type: the shared scale (paperstyle) -------------------------------------
FS_SPAN = ps.FS_AXIS  # "span 1.30x" above each bar -- the payoff, one step up
FS_CEILING = ps.FS_ANNOT
FS_OOM = ps.FS_AXIS

# --- Colors: house semantics ---------------------------------------------------
# neutral = background (the fixed term), blue = what the engine moves, red =
# the contrast (what the replay re-creates and no policy can evict).  The OOM
# band is the lightest red so the non-migratable segment stays readable on it.
C_FIXED = ps.PALETTE["neutral"]
C_MIGRATABLE = ps.PALETTE["blue_main"]
C_NONMIG = ps.PALETTE["red_2"]
C_OOM_BAND = ps.PALETTE["red_1"]
OOM_BAND_ALPHA = 0.45
C_CEILING = ps.ROLE["capacity"]
C_CONTRAST_TEXT = ps.ROLE["per_module"]
BAR_EDGE = "black"
BAR_LW = ps.LW_EDGE
# The bar over the ceiling is the one PROJECTION (that configuration cannot
# run at this shape): drawn at reduced alpha with dashed edges -- the bar analogue of
# the paper-wide open-marker = not-measured convention.
PROJECTED_ALPHA = 0.55
# GiB; at 3.33 in "projected" fills the over-ceiling segment, so centre it
# slightly right of the segment midpoint to clear the dashed ceiling line.
OOM_NUDGE = 0.18

# --- Output -----------------------------------------------------------------
OUT_LOCAL = "fig1_peak_hbm_stack.png"
OUT_IN_USE = os.path.join("..", "rendered", "fig1.png")

# ============================================================================
# DERIVED -- computed from DATA so an annotation can never drift from a bar
# ============================================================================

labels, fixed, migratable, nonmig, spans = [], [], [], [], []
for name, intercept, slope, mig_per_layer in BARS:
    mig_slope = mig_per_layer * L_DEPTH
    labels.append(name)
    fixed.append(intercept / MIB_PER_GIB)
    migratable.append(mig_slope * X_TOKENS / MIB_PER_GIB)
    nonmig.append((slope - mig_slope) * X_TOKENS / MIB_PER_GIB)
    # The span is a ratio of SLOPES, so it carries no intercept and no
    # capacity -- which is why it is the quotable number and the bar height
    # is not. Geometrically it is the ratio of the two upper segments' sum
    # to the top segment alone.
    spans.append(slope / (slope - mig_slope))

totals = [f + m + n for f, m, n in zip(fixed, migratable, nonmig)]

# ============================================================================
# PLOT
# ============================================================================

ps.apply_publication_style()
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=M_LEFT, right=M_RIGHT, bottom=M_BOTTOM, top=M_TOP)

ys = list(range(len(BARS)))              # row 0 on top (y axis inverted)
projected = [t > CEILING_GIB for t in totals]   # cannot run -> not measured

ax.axvspan(CEILING_GIB, X_MAX, facecolor=C_OOM_BAND, alpha=OOM_BAND_ALPHA,
           linewidth=0, zorder=0)

lefts = [0.0] * len(ys)
for seg, color in ((fixed, C_FIXED), (migratable, C_MIGRATABLE),
                   (nonmig, C_NONMIG)):
    for y, w, l, proj in zip(ys, seg, lefts, projected):
        ax.barh(y, w, BAR_H, left=l, color=color,
                alpha=PROJECTED_ALPHA if proj else 1.0,
                edgecolor=BAR_EDGE, linewidth=BAR_LW,
                linestyle=(0, (2.5, 1.5)) if proj else "-", zorder=2)
    lefts = [l + w for l, w in zip(lefts, seg)]

ax.axvline(CEILING_GIB, color=C_CEILING, linewidth=ps.LW_RULE,
           linestyle=(0, (4, 2)), zorder=3)
# The last row ends at 4 GiB, so the space left of the line is free there.
ax.text(CEILING_GIB - 0.4, ys[-1], f"{CEILING_GIB} GiB usable",
        ha="right", va="center", fontsize=FS_CEILING, color=C_CEILING)

for y, total, span, proj, name in zip(ys, totals, spans, projected, labels):
    color = C_CONTRAST_TEXT if name.startswith("per-module") else "black"
    # Spans in one right-aligned column: a label trailing each bar would cross
    # the ceiling line on row 2, and a column reads as a column of results.
    ax.text(X_MAX - 0.2, y, f"span\n{span:.2f}$\\times$", ha="right",
            va="center", fontsize=FS_SPAN, color=color, linespacing=1.0)
    if proj:
        ax.text(total + 0.3, y, "OOM\nprojected", ha="left",
                va="center", fontsize=7.0, color="black",
                linespacing=1.1, zorder=4)

ax.set_ylim(len(ys) - 1 + ROW_PAD, -ROW_PAD)
ax.set_yticks(ys)
ax.set_yticklabels([l.replace(" unit", "") for l in labels], linespacing=1.1)
ax.tick_params(axis="y", length=0, pad=3)
ax.set_xlim(0, X_MAX)
ax.set_xticks([0, 5, 10, 15, 20, 25])
ax.set_xlabel(f"peak HBM (GiB) at $L=64$, {X_TOKENS * 1024:,} tokens",
              labelpad=2)

# One row above the plot, stack order left to right.
handles = [
    mpatches.Patch(facecolor=c, edgecolor=BAR_EDGE, linewidth=BAR_LW, label=l)
    for c, l in ((C_FIXED, "fixed"), (C_MIGRATABLE, "migratable"),
                 (C_NONMIG, "non-migratable"))
]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02),
          ncol=3, handlelength=1.2, handleheight=0.9, columnspacing=1.2,
          frameon=False, borderpad=0.0)  # a single row above the plot: no box

ps.save(fig, OUT_LOCAL, OUT_IN_USE, script_dir=os.path.dirname(os.path.abspath(__file__)))

# Echo the plotted quantities so a reader of the log can check them against
# the measured lines without running a debugger.
for name, f, m, n, t, s in zip(labels, fixed, migratable, nonmig, totals, spans):
    flat = name.replace("\n", " ")
    print(f"  {flat:28s} fixed {f * MIB_PER_GIB:9.2f} + mig {m * MIB_PER_GIB:9.2f}"
          f" + nonmig {n * MIB_PER_GIB:10.2f} = {t * MIB_PER_GIB:10.2f} MiB"
          f"  span {s:.4f}")
