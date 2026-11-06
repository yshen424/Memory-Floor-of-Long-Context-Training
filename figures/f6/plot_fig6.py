"""
fig:peak-instant -- where the peak is, and whether an absence covers it.
(design, single column)

The figure is the evidence for the structural cap -- at most L-W-1
boundaries can be absent at the peak, for ANY boundary mechanism -- so L63's
verdict reads 'unusable' (structural).  Panel (a) is a COMPOSITION TREND on an
UNEVEN, CONTINUOUS event axis, which panel (b) shares.

PANEL (a) -- MEASURED, two strips (one per run), job 7744 (world 4,
W=0).  Every event
of the step is drawn (325 per step, from profile_rank0.json).  Each event i closes an interval (i-1, i] and
carries two numbers, and the strip is filled by what they mean:

  resident   allocated_mb at the interval's close          (blue fill)
  transient  interval_peak_mb - allocated_mb: what was     (red fill)
             allocated inside the interval and gone by its close

Both are drawn as steps -- no interpolation, no smoothing.  INSIDE the
magnified turn every event is its own step.  OUTSIDE it each LAYER is one step
(its highest interval peak, and its resident at the layer's close): drawn
per-event there, the alternating high/low intervals of each layer render as a
comb of hairlines at 0.006 in per event, which reads as noise.  The caption
says so.  The transient is NOT labelled
"non-migratable": it also holds backward's own temporaries.  But it tracks the
non-migratable slope closely (values echoed on stdout), which is why the fill shows the decomposition.

  x=140, BELOW the cap: resident climbs through forward (boundaries kept),
    the peak is the first backward replay, post_bwd_layer_63.
  x=356, AT the cap: resident stays ~2.6-3.1 GiB (63 of 64 boundaries staged out
    at creation); the transient is the whole peak, and it relocates to
    post_bwd_layer_1.  L0 is 6 MiB lower (one world-4 gradient shard; floor at max(W,1)) -- 6 MiB on a
    15 GiB axis is invisible at any x-stretch, so the caption carries it.

THE AXIS.  Event ordinal, piecewise linear and continuous (not broken):
forward L0..L59 squeezed, the TURN (forward L60 .. loss .. backward L60)
magnified, the rest of backward at moderate density.  The turn is shaded in
all three axes so the magnification is declared, and the caption says so.
The same scale applies to panel (b), so the one peak line runs straight down.

PANEL (b) -- SCHEMATIC lanes at W=0, placed at REAL event positions:
creation inside the layer's forward interval, restore at its pre_bwd event,
consumed at its post_bwd event.  L63's restore lands at pre_bwd_layer_63, one
event before the replay interval that holds the peak -- unusable.  Copy
durations are nominal (not measured here).

SIZING: FIG_W = ps.COL_W = 3.33 in = acmart \\columnwidth, scale 1.
Axes placed in inches; no tight_layout(), no bbox_inches="tight".
"""

import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

# --- measured source: job 7744 (world 4), W=0, rank 0, last step -------------
import world4 as w4
RUNS_ROOT = os.path.join("..", "..", "profile_results")
ARMS = {140: w4.RUN_W0[140], 356: w4.RUN_W0[356]}
N_LAYERS = 64
W = 0
GIB = 1024.0
TURN_LAYERS = 4                     # magnified: the last 4 fwd and first 4 bwd
END_LAYERS = 16                     # moderately magnified: first/last 16 layers

# --- canvas: inches -----------------------------------------------------------
FIG_W, FIG_H = ps.COL_W, 2.87
AX_L, AX_R = 0.44, 0.06
A1_BOTTOM, A2_BOTTOM, A_H = 1.97, 1.26, 0.65
B_BOTTOM, B_H = 0.38, 0.62
# fwd ends L0-15 / fwd L16-59 / turn / bwd L59-16 / bwd ends L15-0, axis fractions.
# The two ends are widened: they hold L12's copy and
# restore in panel (b) and the relocated x=356 peak in panel (a).
X_SHARE = (0.15, 0.08, 0.32, 0.12, 0.33)   # 3.33 in: fwd middle +0.01 so L15/L60 ticks clear
MIN_LANE_IN = 0.07                  # min drawn width of a lane piece OUTSIDE the turn
A_END = 0.14                        # lighter shade: the ends are magnified less
A_YLIM = (0.0, 21.0)
B_YLIM = (-0.36, 1.36)
LANE_H = 0.44

# --- semantic colours ----------------------------------------------------------
C_RESIDENT = ps.PALETTE["blue_secondary"]
A_RESIDENT = 0.55
C_RESIDENT_EDGE = ps.PALETTE["blue_main"]
C_TRANSIENT = ps.PALETTE["red_2"]
C_PEAK = "black"                    # the instant; red is taken by the transient
C_REFUSE = ps.PALETTE["red_strong"]
C_LANE = ps.PALETTE["blue_secondary"]
C_NOTE = ps.EXTRA["neutral_text"]
C_TURN = ps.PALETTE["neutral"]
A_TURN = 0.30

OUT_LOCAL = "fig6.png"
OUT_IN_USE = os.path.join("..", "rendered", "fig6.png")


def load_arm(directory):
    """All events of the last step (step 0 carries allocator warm-up)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                RUNS_ROOT, directory, "profile_rank0.json.gz")
    with gzip.open(os.path.normpath(path), "rt") as f:
        snaps = json.load(f)["snapshots"]
    start = max(i for i, s in enumerate(snaps) if s["phase"] == "step_start")
    seg = snaps[start:]
    return ([s["phase"] for s in seg],
            np.array([s["allocated_mb"] for s in seg]) / GIB,
            np.array([s["interval_peak_mb"] for s in seg]) / GIB)


class EventIndex:
    """Where each named event sits on the axis.  Phases repeat (the replay
    re-enters pre_fwd_layer_k inside backward), so the forward and backward
    occurrences are looked up separately."""

    def __init__(self, phases):
        self.phases = phases
        self.fwd_end = phases.index("forward_end")

    def fwd(self, name):
        return self.phases.index(name)

    def bwd(self, name):
        return self.phases.index(name, self.fwd_end)


def make_scale(ix, n_events):
    """Piecewise-linear forward/inverse maps: event index -> axis fraction."""
    t0 = ix.fwd(f"pre_fwd_layer_{N_LAYERS - TURN_LAYERS}") - 1
    t1 = ix.bwd(f"post_bwd_layer_{N_LAYERS - TURN_LAYERS}")
    e0 = ix.fwd(f"post_fwd_layer_{END_LAYERS - 1}")
    e1 = ix.bwd(f"pre_bwd_layer_{END_LAYERS - 1}") - 1
    knots_x = np.array([0.0, e0, t0, t1, e1, n_events - 1.0])
    knots_t = np.cumsum([0.0, *X_SHARE])

    def fwd(x):
        return np.interp(np.asarray(x, dtype=float), knots_x, knots_t)

    def inv(t):
        return np.interp(np.asarray(t, dtype=float), knots_t, knots_x)

    return (fwd, inv), (t0, t1), (e0, e1)


def _rect(bottom, height):
    return [AX_L / FIG_W, bottom / FIG_H,
            (FIG_W - AX_L - AX_R) / FIG_W, height / FIG_H]


def setup_x(ax, scale, turn, ends, n_events, ix, labels=False):
    ax.set_xscale("function", functions=scale)
    ax.set_xlim(0, n_events - 1)
    ax.axvspan(*turn, color=C_TURN, alpha=A_TURN, linewidth=0, zorder=0)
    for x0, x1 in ((0, ends[0]), (ends[1], n_events - 1)):
        ax.axvspan(x0, x1, color=C_TURN, alpha=A_END, linewidth=0, zorder=0)
    ticks = [ix.fwd("post_fwd_layer_0"), ends[0], turn[0], ix.fwd_end,
             turn[1], ends[1], ix.bwd("post_bwd_layer_0")]
    ax.set_xticks(ticks)
    if not labels:
        ax.tick_params(axis="x", labelbottom=False)
        return
    lt, le = N_LAYERS - TURN_LAYERS, END_LAYERS - 1
    ax.set_xticklabels(["$L_0$", f"$L_{{{le}}}$", f"$L_{{{lt}}}$", "loss",
                        f"$L_{{{lt}}}$", f"$L_{{{le}}}$", "$L_0$"])
    tf = ax.get_xaxis_transform()
    for x, text in ((ends[0], "forward"), (ends[1] + 20, "backward")):
        # 0.23 in below the axis, whatever the panel height
        ax.text(x, -0.23 / B_H, text, transform=tf, ha="center", va="top",
                fontsize=ps.FS_AXIS)


def blocks(phases, turn):
    """Step segments (x0, x1, event indices).  Inside the turn: one per event.
    Outside: one per layer (forward: pre+post; backward: pre_bwd + replay +
    post_bwd); the bookkeeping events stand alone."""
    segs, i, n = [], 1, len(phases)
    while i < n:
        p = phases[i]
        if turn[0] < i <= turn[1] or not (p.startswith("pre_fwd")
                                         or p.startswith("pre_bwd")):
            segs.append((i - 1, i, [i]))
            i += 1
            continue
        j = i
        while not phases[j].startswith("post_"):
            j += 1
        segs.append((i - 1, j, list(range(i, j + 1))))
        i = j + 1
    return segs


def strip(ax, alloc, peak, segs, label):
    xs, res, top = [], [], []
    for x0, x1, idx in segs:
        r, t = alloc[idx[-1]], max(peak[k] for k in idx)
        xs += [x0, x1]
        res += [r, r]
        top += [t, t]
    ax.fill_between(xs, 0, res, color=C_RESIDENT, alpha=A_RESIDENT,
                    linewidth=0, zorder=2)
    ax.fill_between(xs, res, top, color=C_TRANSIENT, linewidth=0, zorder=2)
    ax.plot(xs, res, color=C_RESIDENT_EDGE, linewidth=0.6, zorder=3)
    ax.set_ylim(*A_YLIM)
    ax.set_yticks([0, 5, 10, 15])
    ax.text(0.012, 0.96, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=ps.FS_ANNOT, linespacing=1.1)


def draw_lane(ax, y, layer, create, restore, consumed, verdict, scale, turn):
    """Pieces sit at their true events.  OUTSIDE the magnified turn a resident
    piece is 1-2 events long, a sliver at any honest scale, so it is drawn at
    least MIN_LANE_IN wide, growing AWAY from the peak line (creation grows
    right, restore grows left) so the covers-the-peak reading cannot change.
    Panel (b) is labelled schematic; the caption says durations are nominal."""
    edge = dict(edgecolor="black", linewidth=ps.LW_EDGE)
    y0 = y - LANE_H / 2
    fwd, inv = scale
    min_frac = MIN_LANE_IN / (FIG_W - AX_L - AX_R)
    outside = not (turn[0] <= create <= turn[1])
    copy_end = create + 0.8                      # nominal copy duration
    if outside:
        copy_end = max(copy_end, float(inv(fwd(create) + min_frac)))
        restore = min(restore, float(inv(fwd(consumed) - min_frac)))
    ax.add_patch(Rectangle((create, y0), copy_end - create, LANE_H,
                           facecolor=C_LANE, zorder=4, **edge))
    ax.add_patch(Rectangle((copy_end, y0), restore - copy_end, LANE_H,
                           facecolor="white", edgecolor="black",
                           linewidth=ps.LW_EDGE * 0.9,
                           linestyle=(0, (2.2, 1.6)), zorder=3))
    ax.add_patch(Rectangle((restore, y0), consumed - restore, LANE_H,
                           facecolor=C_LANE, zorder=4, **edge))
    color = C_REFUSE if verdict == "unusable" else C_RESIDENT_EDGE
    lx = -AX_L / (FIG_W - AX_L - AX_R)
    tf = ax.get_yaxis_transform()
    ax.text(lx + 0.01, y + 0.03, f"$L_{{{layer}}}$", transform=tf, ha="left",
            va="bottom", fontsize=ps.FS_AXIS, clip_on=False)
    ax.text(lx + 0.01, y - 0.03, verdict, transform=tf, ha="left", va="top",
            fontsize=ps.FS_ANNOT, color=color, style="italic", clip_on=False)


def main():
    ps.apply_publication_style()
    data = {x: load_arm(d) for x, d in ARMS.items()}
    phases = data[140][0]
    assert all(d[0] == phases for d in data.values()), "event sequences differ"
    ix = EventIndex(phases)
    n = len(phases)
    scale, turn, ends = make_scale(ix, n)
    segs = blocks(phases, turn)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    a1 = fig.add_axes(_rect(A1_BOTTOM, A_H))
    a2 = fig.add_axes(_rect(A2_BOTTOM, A_H))
    b = fig.add_axes(_rect(B_BOTTOM, B_H))

    # --- (a) measured strips ---------------------------------------------------
    peaks = {}
    for ax, tokens, label in ((a1, 140, "$x{=}140$,\nbelow the cap"),
                              (a2, 356, "$x{=}356$, at the cap")):
        _, alloc, peak = data[tokens]
        setup_x(ax, scale, turn, ends, n, ix)
        strip(ax, alloc, peak, segs, label)
        i = int(np.argmax(peak))
        peaks[tokens] = (i, phases[i], peak[i])
        ax.plot([i - 0.5], [peak[i]], marker="o", markersize=4.0,
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=1.1, zorder=9)
    fig.text(0.02 / FIG_W, (A2_BOTTOM + A_H + 0.02) / FIG_H, "allocated (GiB)",
             rotation=90, ha="left", va="center", fontsize=ps.FS_AXIS)

    i356 = peaks[356][0]
    a2.text(i356 - 1.5, peaks[356][2] + 0.9, "$\\mathtt{post\\_bwd\\_layer\\_1}$",
            ha="right", va="bottom", fontsize=ps.FS_ANNOT)

    # --- (b) schematic lanes at real event positions -----------------------------
    setup_x(b, scale, turn, ends, n, ix, labels=True)
    for y, layer, verdict in ((1.0, 12, "usable"), (0.0, 63, "unusable")):
        create = ix.fwd(f"pre_fwd_layer_{layer}") + 0.1
        restore = ix.bwd(f"pre_bwd_layer_{layer + W}") + 0.5
        consumed = ix.bwd(f"post_bwd_layer_{layer}")
        draw_lane(b, y, layer, create, restore, consumed, verdict, scale, turn)
    b.set_ylim(*B_YLIM)
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    # Centred in the clear stretch between the peak line and L12's restore.
    b.text(float(scale[1](0.555)), 1.0, "absent from HBM", ha="center",
           va="center", fontsize=ps.FS_ANNOT, color=C_RESIDENT_EDGE,
           style="italic", zorder=6)
    # the filled blocks are the boundary resident in HBM
    b.text(ix.bwd("post_bwd_layer_12") + 2.0, 1.0, "in HBM", ha="left",
           va="center", fontsize=ps.FS_ANNOT, color=C_RESIDENT_EDGE, zorder=6)
    b.text(ix.fwd("pre_fwd_layer_61"), -0.02,
           "restored one event\nbefore the peak", ha="right", va="center",
           fontsize=ps.FS_ANNOT, color=C_REFUSE, linespacing=1.15)

    # --- the one peak line: the measured x=140 instant ---------------------------
    px = peaks[140][0] - 0.5
    style = dict(color=C_PEAK, linewidth=0.9, linestyle=(0, (3.0, 1.8)))
    for ax in (a1, a2, b):
        ax.axvline(px, zorder=8, **style)
    for top, bot, ybot in ((a1, a2, A_YLIM[1]), (a2, b, B_YLIM[1])):
        fig.add_artist(ConnectionPatch((px, A_YLIM[0]), (px, ybot),
                                       coordsA=top.transData,
                                       coordsB=bot.transData, zorder=8,
                                       **style))
    a1.text(px + 1.0, 20.6, "peak instant, $x{=}140$:\n"
            "$\\mathtt{post\\_bwd\\_layer\\_63}$", ha="left", va="top",
            fontsize=ps.FS_ANNOT, linespacing=1.2)

    # --- titles and the one key -------------------------------------------------
    lx = -AX_L / (FIG_W - AX_L - AX_R) + 0.005
    a1.set_title(r"$\bf{(a)}$  measured, $W{=}0$", loc="left",
                 fontsize=ps.FS_ANNOT, pad=3, x=lx)
    b.set_title(r"$\bf{(b)}$  schematic, same axis", loc="left",
                fontsize=ps.FS_ANNOT, pad=3, x=lx)
    b.set_title(f"usable: $L{{-}}W{{-}}1 = {N_LAYERS - W - 1}$ of {N_LAYERS}",
                loc="right", fontsize=ps.FS_ANNOT, pad=3)
    handles = [Patch(facecolor=C_RESIDENT, alpha=A_RESIDENT,
                     edgecolor=C_RESIDENT_EDGE, linewidth=0.6),
               Patch(facecolor=C_TRANSIENT, edgecolor="none"),
               Patch(facecolor=C_TURN, alpha=A_TURN, edgecolor="none")]
    # The key sits on the title line of (a): the strips are too short to hold
    # it.
    a1.legend(handles, ["resident", "transient", "magnified"],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3,
              fontsize=7.0, handlelength=0.9, handleheight=0.75,
              handletextpad=0.3, columnspacing=0.8, borderaxespad=0.15,
              frameon=False, borderpad=0.0)   # single row above the plot: no box

    here = os.path.dirname(os.path.abspath(__file__))
    ps.save(fig, OUT_LOCAL, OUT_IN_USE, script_dir=here)
    plt.close(fig)

    # --- stdout: every plotted quantity near the turn, and the summaries ------
    for tokens in (140, 356):
        _, alloc, peak = data[tokens]
        i, name, v = peaks[tokens]
        tr = peak - alloc
        print(f"  (a) x={tokens}: {n} events; peak {name} = {v * GIB:.2f} MiB;"
              f" max transient {tr.max() * GIB:.2f} MiB;"
              f" resident at peak {alloc[i] * GIB:.2f} MiB")
        for j in range(int(turn[0]), int(turn[1]) + 1):
            print(f"      {phases[j]:<20s} peak {peak[j] * GIB:9.2f}"
                  f"  resident {alloc[j] * GIB:9.2f}"
                  f"  transient {tr[j] * GIB:9.2f}")
    _, _, pk = data[356]
    b1, b0 = pk[ix.bwd("post_bwd_layer_1")], pk[ix.bwd("post_bwd_layer_0")]
    p48, p16 = pk[ix.bwd("post_bwd_layer_48")], pk[ix.bwd("post_bwd_layer_16")]
    print(f"  (a) x=356 bwd L1 - L0 = {(b1 - b0) * GIB:+.2f} MiB;"
          f"  gradient pile L48->L16 {(p16 - p48) * GIB / 32:.3f} MiB/layer")
    print(f"  axis: ends = events 0..{ends[0]} and {ends[1]}..{n - 1};"
          f" turn = events {turn[0]}..{turn[1]}; shares {X_SHARE}")
    print(f"  (b) peak line at event {px}; lanes at W={W}")


if __name__ == "__main__":
    main()
