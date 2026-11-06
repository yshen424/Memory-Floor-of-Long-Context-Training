"""
fig:mechanisms -- every mechanism's frontier, predicted before it ran and
measured.  (evaluation, figure*, FIG_W = ps.FULL_W = 7.0 in)

ALL V100, WORLD 4, L=64, chunked CE c=256, regime B fp16, fused-input,
use_cache off, v=32000, b=1, W=0.  Every filled marker and every bar is a
measured peak at a shape that ran; every open marker is a number written into
the job header before submission.
  vanilla        7747 V   fit 7 (14158.13), fail 8 in FORWARD; V-P1: 7/8,
                          14158.1 on 1555.79 + 1800.33x
  per-module     7747 P   fit 33 (15405.61), fail 34; P-P1: 33 a coin flip
                          (32 or 33), slope then unmeasured -> no registered
                          line, drawn as "guide"
  static         7744 S   fit 104 (15238.11), fail 108; S-P2: 104/108, 15238.11
  static + CPU   7756 S   fit 116 (15533.04), fail 120; S-P1: 112 or 116,
                          15533.02 at 116
  stock offload  7747 O   fit 356 (15330.92), fail 360; O-P2: 356 fits, 360
                          the decider; O-P1 on the engine's line, 15330.92
  offload + CPU  7756 O   fit 388 (15248.26), fail 392; O-P2: 388 or 392,
                          TWO-SIDED 15230.26 | 15248.26 (the step-1 gradient
                          pile goes or stays); the second side held
  instrument     7744 E   fit 356 (15330.92), fail 360; E-P4: 352 or 356,
                          E-P3 15342.93 on the registered floor 1878.90 --
                          THE STEP-1 FLOOR MISS (-12.01 MiB)
  instr. + CPU   7756 E   fit 388 (15248.26), fail 392; E-P1: on offload+cpu's
                          line, closes at its rung (388 or 392)
Every closing failure is step-1 backward on 4x MiB (the replayed feed-forward
activation) except vanilla's, which dies in forward.
Composition at the largest fit: fixed = the mechanism's line intercept;
migratable resident = boundaries on the device at the peak (1.50x each;
per-module: its 3.00/layer/x inbound residual); non-migratable = the rest.
The bytes ABSENT at the peak (63 boundaries) are printed, not drawn: stock
offload carries no telemetry, so for it they are 63 x 1.50x by construction.
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
from matplotlib.legend_handler import HandlerTuple

GIB = 1024.0
L = 64
B = 1.50
NONMIG = w4.M_ALL64                       # 36.3203 MiB/x
CARD_GIB = 15.77
X_STATIC = w4.X_STATIC_FIT                # 104, the span's denominator
X_STATIC_CPU = 116                        # like-for-like with CPU offload on

PM_SLOPE = 15405.6123 - 14983.7998       # 7747 P x=33 minus x=32 = 421.81
PM_C = 14983.7998 - PM_SLOPE * 32

# key: label, marker, colour, cpu, fit, fail, registered fits,
#      measured peak, registered peak(s) at the fit (None = no line),
#      fixed, resident migratable, absent boundaries
C_ST, C_OFF, C_ENG = (ps.ROLE["static"], ps.PALETTE["blue_secondary"],
                      ps.ROLE["engine"])
C_GREY = ps.EXTRA["neutral_line"]
C_NOTE = ps.EXTRA["neutral_text"]
ROWS = [
    dict(label="no checkpointing", m="X", c=C_NOTE, cpu=False,
         fit=7, fail=8, reg=[7], peak=14158.1265, pred=[14158.1],
         fixed=1555.79, res=0.0, absent=0, vanilla=True),
    dict(label="per-module (contrast)", m="^", c=C_GREY, cpu=False,
         fit=33, fail=34, reg=[32, 33], peak=15405.6123, pred=None,
         fixed=PM_C, res=3.00 * L * 33, absent=0, contrast=True),
    dict(label="static whole-layer", m="s", c=C_ST, cpu=False,
         fit=104, fail=108, reg=[104], peak=15238.1079, pred=[15238.11],
         fixed=w4.C_STATIC, res=B * L * 104, absent=0),
    dict(label="+ CPU offload", m="s", c=C_ST, cpu=True,
         fit=116, fail=120, reg=[112, 116], peak=15533.04, pred=[15533.02],
         fixed=183.89, res=B * L * 116, absent=0),
    dict(label="stock offload", m="o", c=C_OFF, cpu=False,
         fit=356, fail=360, reg=[356, 360], peak=15330.9219, pred=[15330.92],
         fixed=w4.FLOORS[0][0], res=B * 356, absent=63),
    dict(label="+ CPU offload", m="o", c=C_OFF, cpu=True,
         fit=388, fail=392, reg=[388, 392], peak=15248.26,
         pred=[15230.26, 15248.26], fixed=573.98, res=B * 388, absent=63),
    dict(label="instrument", m="D", c=C_ENG, cpu=False,
         fit=356, fail=360, reg=[352, 356], peak=15330.9219, pred=[15342.93],
         fixed=w4.FLOORS[0][0], res=B * 356, absent=63),
    dict(label="+ CPU offload", m="D", c=C_ENG, cpu=True,
         fit=388, fail=392, reg=[388, 392], peak=15248.26,
         pred=[15230.26, 15248.26], fixed=573.98, res=B * 388, absent=63),
]

C_FIXED = ps.PALETTE["neutral"]
C_MIG = ps.PALETTE["blue_main"]
C_NONMIG = ps.PALETTE["red_2"]
C_CAP = ps.ROLE["capacity"]
C_NOTE = ps.EXTRA["neutral_text"]
FS = ps.FS_ANNOT
MS = 4.2
LINTHRESH = 0.01

FIG_W, FIG_H = ps.FULL_W, 2.44
PITCH = 0.18
AX_B, AX_H = 0.72, PITCH * len(ROWS)
PANELS = {"a": (1.22, 2.02), "b": (3.50, 1.95), "c": (5.68, 1.10)}
OUT_LOCAL = "fig14.png"
OUT_IN_USE = os.path.join("..", "rendered", "fig14.png")


class Overlay(tuple):
    """Legend handle: markers drawn on top of each other (a white centre)."""


def axes(fig, k):
    left, w = PANELS[k]
    ax = fig.add_axes([left / FIG_W, AX_B / FIG_H, w / FIG_W, AX_H / FIG_H])
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_yticks(range(len(ROWS)))
    ax.tick_params(axis="y", length=0)
    for i in (2, 4, 6):                    # mechanism groups
        ax.axhline(i - 0.5, color=ps.PALETTE["neutral"], linewidth=0.4,
                   zorder=0)
    return ax


def mark(ax, x, y, r, size=MS):
    ax.plot(x, y, marker=r["m"], markersize=size, color=r["c"],
            markeredgecolor="black", markeredgewidth=0.4, linestyle="none",
            zorder=5)
    if r["cpu"]:
        ax.plot(x, y, marker="o", markersize=1.3, color="white",
                linestyle="none", zorder=6)


def panel_reach(ax):
    ax.set_xscale("log", base=2)
    ax.set_yticklabels([r["label"] for r in ROWS])
    for t, r in zip(ax.get_yticklabels(), ROWS):
        if r["label"].startswith("+"):
            t.set_color(C_NOTE)
    print("\n(a) reach   fit  fail  registered fits   span vs static (104; 116 with CPU offload)")
    for i, r in enumerate(ROWS):
        # No [fit, fail) bar: one rung is invisible on log x; the fit is
        # printed and the failure is the next rung.
        mark(ax, r["fit"], i, r)
        for xr in r["reg"]:
            ax.plot(xr, i - 0.36, marker="v", markersize=MS - 0.6,
                    markerfacecolor="white", markeredgecolor=r["c"],
                    markeredgewidth=0.8, linestyle="none", zorder=5)
        span = r["fit"] / (X_STATIC_CPU if r["cpu"] else X_STATIC)
        txt = f"{r['fit']}, {span:.2f}$\\times$"
        hit = r["fit"] in r["reg"]
        if r["fit"] > 60:
            ax.text(r["fit"] / 1.22, i + 0.04, txt, ha="right", va="center",
                    fontsize=FS)
        else:
            ax.text(r["fail"] * 1.22, i + 0.04, txt, ha="left", va="center",
                    fontsize=FS)
        print(f"  {r['label']:22s} {r['fit']:4d} {r['fail']:4d}  "
              f"{str(r['reg']):12s}  {span:.3f}  fit registered: {hit}")
    ax.set_xlim(4, 640)
    ax.set_xticks([4, 16, 64, 256])
    ax.set_xticklabels(["4", "16", "64", "256"])
    ax.set_xlabel("largest fit $x$ (1024 tokens), log", labelpad=2)


def panel_comp(ax):
    ax.set_yticklabels([])
    ax.axvline(CARD_GIB, color=C_CAP, linewidth=ps.LW_RULE,
               linestyle=(0, (4, 2)), zorder=4)
    ax.text(CARD_GIB + 0.4, 0.5, f"{CARD_GIB} GiB\nusable", ha="left",
            va="center", fontsize=FS, color=C_CAP, linespacing=1.0)
    h = 0.62
    print("\n(b) composition at the fit (MiB): fixed  non-mig  resident  "
          "absent(GiB)")
    for i, r in enumerate(ROWS):
        nonmig = r["peak"] - r["fixed"] - r["res"]
        style = dict(edgecolor="black", linewidth=ps.LW_EDGE, zorder=3,
                     linestyle=(0, (2, 1.2)) if r.get("contrast") else "-")
        left = 0.0
        segs = [(r["fixed"], C_FIXED)]
        if r.get("vanilla"):
            segs.append((nonmig, "white"))
        else:
            segs += [(nonmig, C_NONMIG), (r["res"], C_MIG)]
        for v, col in segs:
            ax.barh(i, v / GIB, h, left=left, color=col, **style)
            left += v / GIB
        if r.get("vanilla"):
            ax.text(left / 2 + r["fixed"] / GIB / 2, i, "saved activations",
                    ha="center", va="center", fontsize=FS, color=C_NOTE)
        absent = r["absent"] * B * r["fit"] / GIB
        if absent:
            ax.text(CARD_GIB + 0.4, i, f"+{absent:.1f}", ha="left",
                    va="center", fontsize=FS, color=C_MIG)
        print(f"  {r['label']:22s} {r['fixed']:8.2f} {nonmig:8.2f} "
              f"{r['res']:8.2f}  {absent:6.2f}   sum {r['fixed'] + nonmig + r['res']:.2f}")
    ax.set_xlim(0, 19.8)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xlabel("peak at the fit (GiB)", labelpad=2)


def panel_resid(ax):
    ax.set_yticklabels([])
    ax.axvspan(-LINTHRESH, LINTHRESH, color=ps.PALETTE["neutral"], alpha=0.35,
               linewidth=0, zorder=0)
    print("\n(c) measured - registered peak at the fit (MiB)")
    for i, r in enumerate(ROWS):
        if r["pred"] is None:
            ax.text(0, i, "no line registered", ha="center", va="center",
                    fontsize=FS, color=C_NOTE, style="italic")
            print(f"  {r['label']:22s} none")
            continue
        res = [r["peak"] - p for p in r["pred"]]
        for j, v in enumerate(res):
            if j == 0 and len(res) > 1:      # the side that did not hold
                ax.plot(v, i, marker=r["m"], markersize=MS,
                        markerfacecolor="white", markeredgecolor=r["c"],
                        markeredgewidth=0.8, linestyle="none", zorder=5)
            else:
                mark(ax, v, i, r)
        print(f"  {r['label']:22s} " + "  ".join(f"{v:+.3f}" for v in res))
    ax.text(-30.0, 5.62, "floor miss", ha="left", va="bottom", fontsize=FS,
            color=C_NOTE)
    ax.text(30.0, 4.62, "two-sided", ha="right", va="bottom", fontsize=FS,
            color=C_NOTE)
    ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=0.6)
    ax.set_xlim(-40, 60)
    ax.set_xticks([-10, 0, 10])
    ax.set_xticklabels(["$-10$", "0", "10"])
    ax.set_xlabel("meas. $-$ registered (MiB)", labelpad=2)


def main():
    ps.apply_publication_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    a, b, c = (axes(fig, k) for k in "abc")
    panel_reach(a)
    panel_comp(b)
    panel_resid(c)

    top = (AX_B + AX_H + 0.08) / FIG_H
    for k, text in (("a", r"$\bf{(a)}$  frontier: largest fit"),
                    ("b", r"$\bf{(b)}$  composition at the fit; +GiB absent"),
                    ("c", r"$\bf{(c)}$  peak vs line")):
        fig.text(PANELS[k][0] / FIG_W, top, text, ha="left", va="bottom",
                 fontsize=FS)
    # One boxed key per panel, centred under it (a single shared key barely
    # covers (c)).  Shape and colour name the
    # mechanism through the row labels; a white dot is FSDP CPU offload.
    def key(k, handles, labels, ncol, handlelength=1.2, handler_map=None):
        left, w = PANELS[k]
        loc, x = {"a": ("lower left", 0.02),
                  "b": ("lower left", left - 0.22),
                  "c": ("lower right", FIG_W - 0.02)}[k]
        fig.legend(handles, labels, loc=loc,
                   bbox_to_anchor=(x / FIG_W, 0.02), ncol=ncol,
                   fontsize=FS, handlelength=handlelength, handleheight=0.7,
                   handler_map=handler_map,
                   handletextpad=0.4, columnspacing=0.9, labelspacing=0.3)
    def filled(m, c, cpu=False, size=MS):
        h = Line2D([], [], marker=m, color=c, markeredgecolor="black",
                   markeredgewidth=0.4, markersize=size, linestyle="none")
        if not cpu:
            return h
        return Overlay((h, Line2D([], [], marker="o", markersize=1.3,
                                  color="white", linestyle="none")))

    def hollow(m, c, size=MS):
        return Line2D([], [], marker=m, markerfacecolor="white",
                      markeredgecolor=c, markeredgewidth=0.8, markersize=size,
                      linestyle="none")

    mech = [("X", C_NOTE), ("^", C_GREY), ("s", C_ST), ("o", C_OFF), ("D", C_ENG)]
    tup = {tuple: HandlerTuple(ndivide=None, pad=0.15),
           Overlay: HandlerTuple(ndivide=1)}
    # (a): every filled shape is a measured fit, every open triangle a
    # registered one; a white centre is FSDP CPU offload (shape = row).
    key("a", [tuple(filled(m, c, size=MS - 0.6) for m, c in mech),
              tuple(filled(m, c, cpu=True) for m, c in mech[2:]),
              hollow("v", C_NOTE, MS - 0.6)],
        ["measured largest fit", "white centre: + CPU offload",
         "registered largest fit"], 2, handlelength=3.0, handler_map=tup)
    key("b", [Patch(facecolor=C_FIXED, edgecolor="black", linewidth=0.5),
              Patch(facecolor=C_NONMIG, edgecolor="black", linewidth=0.5),
              Patch(facecolor=C_MIG, edgecolor="black", linewidth=0.5)],
        ["fixed", "non-migratable", "migratable, resident"], 2)
    # (c): the same shapes and colours as (a); open = the registered side
    # that did not hold.
    key("c", [tuple(filled(m, c) for m, c in mech[2:]),
              tuple(hollow(m, c) for m, c in (("o", C_OFF), ("D", C_ENG)))],
        ["measured $-$ registered", "side not held"], 1,
        handlelength=2.6, handler_map=tup)

    here = os.path.dirname(os.path.abspath(__file__))
    ps.save(fig, OUT_LOCAL, OUT_IN_USE, script_dir=here, expect_px=ps.PX_FULL)
    plt.close(fig)


if __name__ == "__main__":
    main()
