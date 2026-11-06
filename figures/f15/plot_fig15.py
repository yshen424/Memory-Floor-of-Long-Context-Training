"""
fig:parity -- every peak predicted before its run, against the measured peak.

Data: parity_points.csv beside this script; remote rows from
../f9/remote_bytes.json; predicted = the number written before the run.

  (a) PARITY, short: measured against predicted (GiB, log2), only tall enough
      to show every arm on y = x.  Shape = mechanism, colour = platform row.
  (b) RESIDUAL BY PLATFORM: five flat strips, one per row (V100, V100 shape
      sweep, A30, L40S, A100), each with its own x-axis.  x = measured -
      predicted on an uneven axis: linear and wide inside +-0.01 MiB, log
      beyond; a histogram (0.002 MiB bins inside, quarter decades outside),
      bar height ~ log(1 + arms), counts on the large bins.  Row labels carry
      the platform colour, which (a) and (c) reuse without a key.
  (c) SHARE WITHIN AN ERROR: per row, the fraction of arms whose |residual|
      is at most x; the 0.05 MiB hit tolerance drawn; misses are tail steps.

SIZING: placed at width=0.90\columnwidth, so FIG_W = SCALE * ps.COL_W =
2.997 in and the scale factor is still 1 (fonts, lines and markers at their
printed sizes); axes placed in inches.
"""

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "parity_points.csv")
REMOTE = os.path.join(HERE, "..", "f9", "remote_bytes.json")
REMOTE_EXCLUDE = {"E": "A30 allocator control (expandable segments off)",
                  "G": "idle ballast in the peak (A30, A100)"}


def _f(v):
    return float(v) if v not in ("", None) else None


def load():
    with open(DATA) as f:
        rows = [r for r in csv.DictReader(f)]
    for r in rows:
        r["predicted_mib"] = _f(r["predicted_mib"])
        r["measured_mib"] = _f(r["measured_mib"])
        r["synthetic"] = False
    return rows


SCOPE_MECH = {"static", "engine", "offload", "static+cpu", "offload+cpu",
              "engine+cpu"}


def out_of_scope(r):
    """Scope: the configuration the pre-registered predictions cover -- world 4,
    the four mechanisms (+/- FSDP CPU offload), chunked CE on -- plus every
    shape-sweep row (7761, kept WITH its misses) and every remote row.
    Returns the exclusion reason, or None if the row is in scope."""
    if r["world"] != "4":
        return f"world {r['world']}"
    if r["mechanism"] not in SCOPE_MECH:
        return f"mechanism {r['mechanism']}"
    if r["job"] == "7752":
        return "micro-batching study (not in the paper)"
    n = r["notes"].lower()
    if "unchunked" in n or "c=1024" in n:
        return "chunked loss off or c != 256"
    return None


def is_reproduction(r):
    """A header that pre-registers a previously MEASURED value (a seeded
    control, a bit-identical twin) predicts nothing the model derives, so it
    is not a parity point.  Echoed, not silently dropped."""
    n = r["notes"].lower()
    return "reproduc" in n or "bit-identically" in n


def load_remote():
    """Every A30 / L40S / A100 byte arm as a parity row; returns (rows, excluded)."""
    import json
    mech = {"static": "static", "offload": "offload", "evict": "engine",
            "staticcpu": "static+cpu", "offloadcpu": "offload+cpu"}
    rows, excluded = [], []
    for a in json.load(open(REMOTE)):
        blk = a["arm"].split("_")[0]
        if blk in REMOTE_EXCLUDE:
            excluded.append((a, REMOTE_EXCLUDE[blk]))
            continue
        rows.append(dict(job=a["job"], world="4", gpu=a["gpu"], mechanism=mech[a["mode"]],
                         x=str(a["x"]), predicted_mib=a["meas"] - a["res_reg"],
                         measured_mib=a["meas"], synthetic=False, shape=a["shape"],
                         notes=a["arm"] + ("  [term added after the run]" if a["posthoc"] else "")))
    return rows, excluded



EXACT = 0.05                      # |residual| <= this counts as a hit (MiB)
LIN = 0.01                        # linear half-width of the residual axis (MiB)
LIN_W = 1.6                       # axis units given to [0, LIN]; one decade = 1
X_LIM = (-15.0, 400.0)
V4_JOB = "7761"

# The visible name of the 7761 row is "V100, shape sweep"; the key joins the
# strip label's lines, and the dict key "V100 V4" is internal.
ROWS = [("V100", "V100"), ("V100 V4", "V100,\nshape\nsweep"), ("A30", "A30"),
        ("L40S", "L40S"), ("A100", "A100")]
COL = {"V100": ps.PALETTE["blue_main"], "V100 V4": ps.PALETTE["blue_secondary"],
       "A30": ps.PALETTE["teal"], "L40S": ps.PALETTE["violet"],
       "A100": ps.PALETTE["red_strong"]}
C_NOTE = ps.EXTRA["neutral_text"]
FS = 7.0

# Top-down layout in inches: a short parity scatter, then one flat strip per
# platform with its own x-axis, then the share panel.  The page sets this
# figure at 0.90 \columnwidth.  The label margins (AX_L, AX_R, the *_GAP
# rows) hold text that does not shrink, so the data areas are what is smaller.
SCALE = 0.90
FIG_W = SCALE * ps.COL_W          # 2.997 in -> 1798 px at 600 dpi
AX_L, AX_R = 0.62, 0.08
KEY_H = 0.38                      # the platform key above (a)
A_H = 0.56                        # (a) parity: tall enough to show the line
A_GAP = 0.37                      # (a)'s tick labels + x label
STRIP_H, STRIP_GAP = 0.25, 0.17   # (b) one strip + its tick labels
B_GAP = 0.37                      # (b)'s x label + (c)'s panel tag
C_H = 0.72
BOTTOM = 0.36
FIG_H = (KEY_H + A_H + A_GAP + len(ROWS) * STRIP_H
         + (len(ROWS) - 1) * STRIP_GAP + B_GAP + C_H + BOTTOM)
OUT_LOCAL = "fig15.png"
OUT_IN_USE = os.path.join("..", "rendered", "fig15.png")


def fwd(r):
    r = np.asarray(r, dtype=float)
    a = np.abs(r)
    t = np.where(a <= LIN, LIN_W * a / LIN, LIN_W + np.log10(np.maximum(a, LIN) / LIN))
    return np.sign(r) * t


def inv(t):
    t = np.asarray(t, dtype=float)
    a = np.abs(t)
    r = np.where(a <= LIN_W, a * LIN / LIN_W, LIN * 10 ** (a - LIN_W))
    return np.sign(t) * r


def row_of(r):
    if r["gpu"] == "V100":
        return "V100 V4" if r["job"] == V4_JOB else "V100"
    return r["gpu"]


def points():
    rows = [r for r in load() if r["predicted_mib"] is not None]
    rows = [r for r in rows if not is_reproduction(r) and out_of_scope(r) is None]
    remote, _ = load_remote()
    pts = [r for r in rows + remote if r["measured_mib"] is not None]
    for r in pts:
        r["res"] = r["measured_mib"] - r["predicted_mib"]
        r["row"] = row_of(r)
    return pts



A_LIM = (0.75, 64.0)              # GiB, log2
GIB_TICKS = [1, 2, 4, 8, 16, 32, 64]
MECH = {"static": ("s", "static"), "engine": ("D", "instrument"),
        "offload": ("o", "stock offload")}
MS = 3.2


def base_mech(m):
    return m.replace("+cpu", ""), m.endswith("+cpu")


def panel_parity(ax, pts):
    ax.plot(A_LIM, A_LIM, color=ps.EXTRA["neutral_line"], linewidth=0.8,
            linestyle=(0, (3.0, 2.0)), zorder=2)
    ax.text(36.0, 18.0, "$y = x$", ha="left", va="top", fontsize=FS,
            color=ps.EXTRA["neutral_line"])
    for r in pts:
        mech, cpu = base_mech(r["mechanism"])
        x, y = r["predicted_mib"] / 1024.0, r["measured_mib"] / 1024.0
        ax.plot(x, y, marker=MECH[mech][0], markersize=MS, color=COL[r["row"]],
                markeredgecolor="black", markeredgewidth=0.4, linestyle="none",
                zorder=4)
        if cpu:
            ax.plot(x, y, marker="o", markersize=1.1, color="white",
                    linestyle="none", zorder=5)
    for setscale, setlim, axis in ((ax.set_xscale, ax.set_xlim, ax.xaxis),
                                   (ax.set_yscale, ax.set_ylim, ax.yaxis)):
        setscale("log", base=2)
        setlim(*A_LIM)
        axis.set_major_locator(mticker.FixedLocator(GIB_TICKS))
        axis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in GIB_TICKS]))
        axis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks([1, 8, 64])
    ax.set_yticklabels(["1", "8", "64"])
    ax.set_xlabel("predicted peak (GiB), written before the run", labelpad=2)
    ax.set_ylabel("measured\n(GiB)", labelpad=2, linespacing=1.0)
    h = [Line2D([], [], marker=MECH[m][0], markersize=MS + 0.4, linestyle="none",
                color="black") for m in MECH]
    h.append((Line2D([], [], marker="s", markersize=MS + 0.4, linestyle="none",
                     color="black"),
              Line2D([], [], marker="o", markersize=1.2, linestyle="none",
                     color="white")))
    # The mechanism key lives in (a), boxed, in the empty lower-right corner,
    # below the diagonal, where no point can fall (an upper-left box covers
    # the low-x points); the
    # platform-colour key is figure-wide (see main), since (a)-(c) use it.
    short = {"stock offload": "offload"}       # narrower, as in fig9's key
    labels = [short.get(MECH[m][1], MECH[m][1]) for m in MECH] + ["+ CPU off."]
    # Four entries, filled column-major as static/offload | instrument/+ CPU
    # off., so the two wide labels share a column and the box clears the
    # points on the diagonal of (a).
    order = [0, 2, 1, 3]
    ax.legend([h[i] for i in order], [labels[i] for i in order],
              loc="lower right",
              bbox_to_anchor=(1.0, 0.0), ncol=2, fontsize=FS,
              handlelength=0.7, handletextpad=0.2, columnspacing=0.5,
              labelspacing=0.1, borderaxespad=0.15, borderpad=0.25)


def platform_key(fig):
    """One row above everything: colour = platform, for (a), (b) and (c).
    A single row above the plots, so no box."""
    from matplotlib.patches import Patch
    hp = [Patch(facecolor=COL[k], edgecolor="black", linewidth=0.4)
          for k, _ in ROWS]
    lp = [lab.replace("\n", " ") for _, lab in ROWS]
    fig.legend(hp, lp, loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.04 / FIG_H), ncol=5, fontsize=FS,
               title="colour = platform, in (a)$-$(c)", title_fontsize=FS,
               handlelength=1.0, handleheight=0.7, handletextpad=0.3,
               columnspacing=0.7, frameon=False, borderpad=0.0,
               borderaxespad=0.0)


def edges():
    """Bin edges in MiB: 0.002 MiB wide inside the linear band, a quarter
    decade wide outside it, symmetric about zero."""
    inner = np.arange(-LIN, LIN + 1e-12, 0.002)
    outer = LIN * 10 ** np.arange(0.25, 4.76, 0.25)
    return np.concatenate([-outer[::-1], inner, outer])


NOTES = {"V100": (-5.0, 0.45, "step-1 floor", "left"),
         "A30": (12.0, 0.45, "flash plane", "right"),
         "L40S": (0.3, 0.30, "deeper replay", "left")}


def strip(ax, pts, key, label, last):
    ax.set_xscale("function", functions=(fwd, inv))
    ax.axvspan(-LIN, LIN, color=ps.PALETTE["neutral"], alpha=0.35, linewidth=0,
               zorder=0)
    e = edges()
    nmax = 130.0
    res = np.array([r["res"] for r in pts if r["row"] == key])
    n, _ = np.histogram(res, bins=e)
    assert n.sum() == len(res), (key, n.sum(), len(res))
    for i, (lo, hi, k) in enumerate(zip(e[:-1], e[1:], n)):
        if not k:
            continue
        h = 0.08 + 0.72 * math.log1p(k) / math.log1p(nmax)
        ax.bar(lo, h, width=hi - lo, align="edge", color=COL[key],
               edgecolor="black", linewidth=0.3, zorder=3)
        if k >= 20 or (k >= 10 and k == n.max()):
            # A count sits beside its bar, on the side away from zero when
            # that neighbouring bin is empty: on these strips a count above the
            # bar meets its neighbour's count (L40S 129 / 26) and the "0" tick
            # label of the strip above.  A
            # bar flanked on both sides (V100's 33) keeps its count on top.
            side = -1 if hi <= 0 else 1
            j = i + side
            if 0 <= j < len(n) and n[j] == 0:
                ax.annotate(f"{k}", xy=(lo if side < 0 else hi, h),
                            xytext=(1.2 * side, 0), textcoords="offset points",
                            ha="right" if side < 0 else "left", va="top",
                            fontsize=FS, color=C_NOTE)
            else:
                ax.text(float(inv((fwd(lo) + fwd(hi)) / 2)), h + 0.03, f"{k}",
                        ha="center", va="bottom", fontsize=FS, color=C_NOTE)
        print(f"   {key:<8} [{lo:+.4f}, {hi:+.4f})  {k}")
    if key in NOTES:
        x, y, text, ha = NOTES[key]
        ax.text(x, y, text, ha=ha, va="center", fontsize=FS, color=C_NOTE)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    # a three-line label ("V100, shape sweep") sits a little
    # higher, to clear its own strip's -10^1 tick label below it
    ax.text(-0.03, 0.45 if label.count("\n") == 2 else 0.4, label,
            transform=ax.transAxes, ha="right", va="center",
            fontsize=FS, color=COL[key], linespacing=1.0)
    ax.set_xlim(*X_LIM)
    ax.set_xticks([-10, -0.01, 0, 0.01, 1, 100])
    ax.set_xticklabels(["$-10^{1}$", "$-10^{-2}$", "0", "$10^{-2}$", "$10^{0}$",
                        "$10^{2}$"])
    ax.minorticks_off()
    ax.tick_params(axis="x", pad=1, labelsize=FS)
    if last:
        ax.set_xlabel("measured $-$ predicted (MiB)", labelpad=2)


def panel_ecdf(ax, pts):
    ax.set_xscale("log")
    ax.axvline(EXACT, color=C_NOTE, linewidth=0.6, linestyle=(0, (1.5, 1.5)))
    print("(b) share within the hit tolerance")
    for key, label in ROWS:
        a = np.sort(np.maximum(np.abs([r["res"] for r in pts if r["row"] == key]),
                               1.1e-4))
        y = np.arange(1, len(a) + 1) / len(a)
        xs = np.concatenate([[1e-4], a, [X_LIM[1]]])
        ys = np.concatenate([[0], y, [1]])
        ax.step(xs, ys, where="post", color=COL[key], linewidth=1.1)
        print(f"   {key:<8} {np.mean(a <= EXACT):.3f} of {len(a)}")
    n = len(pts)
    h = sum(abs(r["res"]) <= EXACT for r in pts)
    # x = 2.0: at 1.5 the note's corner meets the V100 shape-sweep step at
    # ~1.2 MiB on (c).
    ax.text(2.0, 0.03, f"{h} of {n} arms\nwithin {EXACT} MiB",
            ha="left", va="bottom", fontsize=FS, color=C_NOTE, linespacing=1.0)
    ax.set_xlim(1e-4, X_LIM[1])
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", ".5", "1"])
    ax.set_xticks([1e-4, 1e-2, 1, 100])
    ax.set_xticklabels(["$10^{-4}$", "$10^{-2}$", "$10^{0}$", "$10^{2}$"])
    ax.minorticks_off()
    ax.set_xlabel("|measured $-$ predicted| (MiB)", labelpad=2)
    ax.set_ylabel("share of arms", labelpad=2)


def main():
    ps.apply_publication_style()
    pts = points()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    w = (FIG_W - AX_L - AX_R) / FIG_W

    def axes(top, h):
        return fig.add_axes([AX_L / FIG_W, (FIG_H - top - h) / FIG_H, w, h / FIG_H])

    top = KEY_H
    a = axes(top, A_H)
    panel_parity(a, pts)
    platform_key(fig)
    top += A_H + A_GAP
    print("(b) histogram strips: row, bin (MiB), arms")
    strips = []
    for i, (key, label) in enumerate(ROWS):
        ax = axes(top, STRIP_H)
        strip(ax, pts, key, label, last=(i == len(ROWS) - 1))
        strips.append(ax)
        top += STRIP_H + STRIP_GAP
    top += B_GAP - STRIP_GAP
    c = axes(top, C_H)
    panel_ecdf(c, pts)
    lx = -AX_L / (FIG_W - AX_L - AX_R) + 0.005
    for ax, tag in ((a, "(a)"), (strips[0], "(b)"), (c, "(c)")):
        ax.text(lx, 1.02, tag, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=ps.FS_PANEL, fontweight="bold")
    ps.save(fig, OUT_LOCAL, OUT_IN_USE, script_dir=HERE,
            expect_px=int(round(FIG_W * ps.DPI)))
    plt.close(fig)
    print(f"  canvas {FIG_W:.3f} x {FIG_H:.2f} in; arms drawn {len(pts)}: " + ", ".join(
        f"{k} {sum(r['row'] == k for r in pts)}" for k, _ in ROWS))


if __name__ == "__main__":
    main()
