"""House style for this paper's figures: the figures4papers look at column width.

WHAT THIS IS. The figures4papers house style specifies an API rather than
shipping one (the style package is not part of this repository). This module is
that API, implemented once and scaled to this paper's canvas, so a figure
script imports a look instead of re-declaring twelve constants.

THE ONE TRANSLATION THAT MATTERS. The house style is authored for slide-sized
canvases: ``font.size=24`` with ``axes.linewidth=3``, a ratio of 8:1. Our type
floor is 7 pt and our body size is 8.5, so copying the absolute 3 would put a
3 pt spine around 8 pt text. Every weight here is carried across as that RATIO,
not as the number -- 8.5/8 gives ~1.05 for spines, and bar edges land near 0.7.

WHERE THE PROJECT OVERRIDES THE HOUSE, AND WHY. The authors' figure rules
win on conflict, in three places: the house calls ``tight_layout``/``bbox_inches='tight'`` (both crop to
ink, which silently reintroduces a LaTeX scale factor and breaks the 1:1 rule),
it defaults to 300 dpi (our floor is 400 at printed width, our target 600), and
it reaches for hatching for print safety (at 3.33 in the strokes are spaced in
figure inches, so a small box gets about two of them, drawn through its label).
Hatching is replaced here by an alpha ramp plus explicit edges, which is the
same house pattern read one level up: one hue means one mechanism.

FONTS. Helvetica and Arial are not installed on this machine. Liberation Sans
is, and is metrically Arial-compatible, so it is the faithful stand-in and is
what the house 'helvetica' resolves to here.

Run ``python3 paperstyle.py`` for a self-check of fonts, widths and the palette.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Canvas: authored at printed width so LaTeX's scale factor is exactly 1 ──
COL_W = 3.33          # acmart sigconf \columnwidth = 241.15 pt
FULL_W = 7.0          # acmart sigconf \textwidth  = 506.30 pt, for figure*
DPI = 600
DPI_FLOOR = 400
TYPE_FLOOR = 7.0      # pt; nothing printed may fall below this

# Expected pixel widths at DPI. Verified after every save.
PX_COL = int(round(COL_W * DPI))    # 1998
PX_FULL = int(round(FULL_W * DPI))  # 4200

# ── Palette: verbatim from references/api.md; do not edit, it is the style ──
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

# Project extension, kept separate so the house palette stays diffable against
# upstream. api.md's PALETTE carries one neutral (#CFCECE) because it is a FILL
# tone; a 1 pt stroke of it is invisible in print and this paper draws its
# contrast arm as a line. These two are the darker neutrals design-theory.md
# section 3 names alongside it, so the family is unchanged.
EXTRA = {
    "neutral_line": "#767676",
    "neutral_text": "#4D4D4D",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
    PALETTE["teal"], PALETTE["violet"], PALETTE["neutral"],
]

# ── Semantic roles for THIS paper ────────────────────────────────────────────
# The house rule is blue=proposed, green=improvement, red=baseline/contrast,
# neutral=background. Bound here to the paper's arms once, so no script picks a
# hex by hand and no two figures disagree about what the engine's colour is.
ROLE = {
    "engine": PALETTE["blue_main"],        # the proposed policy
    "static": PALETTE["red_strong"],       # static whole-layer -- the baseline
    "lever2": PALETTE["blue_secondary"],   # whole-layer without chunked CE
    "per_module": EXTRA["neutral_line"],   # CONTRAST arm, never a baseline
    "vanilla": PALETTE["neutral"],
    "capacity": PALETTE["red_strong"],     # the capacity rule / bracket
    "improve": PALETTE["green_3"],
    "callout": PALETTE["highlight"],       # single callout per figure, at most
    "alt": PALETTE["teal"],
}


def alpha_ramp(n, lo=0.35, hi=1.0):
    """One hue at n alphas -- the house ablation ramp.

    Says "same mechanism, n settings" where n separate hues would say "n
    systems". Used for the restore-depth family and for lever ablations. Drawn
    dark-to-light in the order given, so pass the arms in the order the reader
    should read them.
    """
    if n == 1:
        return [hi]
    return list(np.linspace(hi, lo, n))


# ── Type scale: house ratios, our floor ──────────────────────────────────────
FS_AXIS = 8.5
FS_TICK = 7.5
FS_LEGEND = 7.5
FS_ANNOT = 7.5
FS_PANEL = 8.5   # panel letters (a), (b)

# Line and edge weights, carried across as house ratios at our type size.
LW_SPINE = 1.05  # house 3 at font 24
LW_CURVE = 1.4
LW_EDGE = 0.7    # house 1.5-3 bar edges
LW_RULE = 0.9


@dataclass(frozen=True)
class FigureStyle:
    """api.md's FigureStyle, defaulted to this paper instead of to slides."""

    font_size: float = FS_AXIS
    axes_linewidth: float = LW_SPINE
    use_tex: bool = False
    font_family: tuple = ("Liberation Sans", "DejaVu Sans", "sans-serif")
    grid: bool = False


def apply_publication_style(style: FigureStyle | None = None):
    """Configure rcParams once, before any figure is created."""
    s = style or FigureStyle()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(s.font_family),
        "font.size": s.font_size,
        "text.usetex": s.use_tex,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": s.axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": FS_AXIS,
        "axes.titlesize": FS_AXIS,
        "axes.grid": s.grid,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.major.width": s.axes_linewidth,
        "ytick.major.width": s.axes_linewidth,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.width": s.axes_linewidth * 0.7,
        "ytick.minor.width": s.axes_linewidth * 0.7,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": FS_LEGEND,
        # One legend style for every figure: a white
        # box with a dark-grey frame (patch.linewidth below sets its weight).
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.facecolor": "white",
        "legend.edgecolor": EXTRA["neutral_text"],
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.0,
        "legend.borderpad": 0.4,
        "legend.columnspacing": 1.0,
        "lines.linewidth": LW_CURVE,
        "lines.markersize": 3.4,
        "lines.markeredgewidth": 0.9,
        "patch.linewidth": LW_EDGE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.facecolor": "white",
        # Deliberately NOT 'tight': cropping to ink changes the emitted width
        # and reintroduces the LaTeX scale factor this paper authors away.
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return s


def create_subplots(nrows=1, ncols=1, figsize=None, full_width=False, **kwargs):
    """api.md's create_subplots. Returns (fig, flat 1-D array of Axes).

    Width defaults to the printed width, never to matplotlib's 6.4 in -- the
    whole sizing rule is that the figure is authored at the size it prints.
    """
    if figsize is None:
        figsize = ((FULL_W if full_width else COL_W), 2.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(np.asarray(axes)).ravel()


def legend_panel(ax, handles, labels, ncol=1, title=None, loc="center", **kw):
    """The house dedicated-legend-panel pattern: an axis that holds only a key.

    Use when the legend would otherwise sit on data. The panel costs height but
    buys back the space a large in-axes legend steals, and it is what keeps a
    multi-panel figure to one key instead of one per panel.
    """
    ax.set_axis_off()
    leg = ax.legend(handles, labels, loc=loc, ncol=ncol,
                    fontsize=FS_LEGEND, title=title, **kw)
    if title:
        leg.get_title().set_fontsize(FS_LEGEND)
    return leg


def panel_label(ax, text, x=-0.16, y=1.04, **kw):
    """(a)/(b) in axes fraction, placed identically across panels."""
    return ax.text(x, y, text, transform=ax.transAxes, fontsize=FS_PANEL,
                   fontweight="bold", va="bottom", ha="left", **kw)


def tighten_ylim(ax, values, margin=0.08, floor_zero=False):
    """House dynamic y-scaling: fit the range, not 0-100.

    ``floor_zero`` keeps 0 in view, which is mandatory whenever a length is
    being read as a quantity (bars), and wrong whenever it squashes the effect
    (our slopes, which live far from the origin).
    """
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], dtype=float)
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * margin if hi > lo else (abs(hi) * margin or 1.0)
    ax.set_ylim(0.0 if floor_zero else lo - pad, hi + pad)
    return ax.get_ylim()


def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=FS_ANNOT, padding=1.5,
                  color="black", horizontal=False):
    """api.md's annotate_bars: the value printed in place, not read off a grid."""
    out = []
    for b in bars:
        if horizontal:
            v = b.get_width()
            out.append(ax.annotate(fmt.format(v),
                                   xy=(v, b.get_y() + b.get_height() / 2),
                                   xytext=(padding, 0), textcoords="offset points",
                                   ha="left", va="center", fontsize=fontsize,
                                   color=color))
        else:
            v = b.get_height()
            out.append(ax.annotate(fmt.format(v),
                                   xy=(b.get_x() + b.get_width() / 2, v),
                                   xytext=(0, padding), textcoords="offset points",
                                   ha="center", va="bottom", fontsize=fontsize,
                                   color=color))
    return out


def make_grouped_bar(ax, categories, series, labels, ylabel="Value", colors=None,
                     annotate=False, fmt="{:.2f}", alphas=None, width=0.8):
    """api.md's make_grouped_bar, with black edges as the house prescribes."""
    colors = colors or DEFAULT_COLORS
    n = len(series)
    idx = np.arange(len(categories))
    w = width / n
    last = None
    for i, (vals, lab) in enumerate(zip(series, labels)):
        off = (i - (n - 1) / 2) * w
        last = ax.bar(idx + off, vals, w, label=lab,
                      color=colors[i % len(colors)],
                      alpha=(alphas[i] if alphas else 1.0),
                      edgecolor="black", linewidth=LW_EDGE)
        if annotate:
            annotate_bars(ax, last, fmt=fmt)
    ax.set_xticks(idx)
    ax.set_xticklabels(categories)
    if ylabel:
        ax.set_ylabel(ylabel)
    return last


def make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None,
               bands=None, alphas=None, markers=None, linestyles=None):
    """api.md's make_trend. ``bands`` is (lo, hi) per series for fill_between.

    HARD LIMIT, from design-theory.md section 6: at most 4 primary curves on one
    axis. More than that is a split into panels, not a bigger legend.
    """
    if len(y_series) > 4:
        raise ValueError(
            f"{len(y_series)} curves on one axis; the house limit is 4. "
            "Split into panels.")
    colors = colors or DEFAULT_COLORS
    lines = []
    for i, (y, lab) in enumerate(zip(y_series, labels)):
        c = colors[i % len(colors)]
        ln, = ax.plot(x, y, label=lab, color=c,
                      alpha=(alphas[i] if alphas else 1.0),
                      marker=(markers[i] if markers else None),
                      linestyle=(linestyles[i] if linestyles else "-"))
        lines.append(ln)
        if bands and bands[i] is not None:
            lo, hi = bands[i]
            ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    return lines


def make_scatter(ax, x, y, label=None, color=None, size=14, alpha=0.8,
                 measured=True):
    """api.md's make_scatter, plus this paper's measured/projected convention.

    Filled marker = measured; open marker = extrapolated or projected. The
    convention is paper-wide and is never carried by colour alone.
    """
    color = color or PALETTE["blue_main"]
    kw = dict(s=size, label=label, alpha=alpha, linewidths=0.9)
    if measured:
        return ax.scatter(x, y, color=color, edgecolors=color, **kw)
    return ax.scatter(x, y, facecolors="white", edgecolors=color, **kw)


def make_heatmap(ax, matrix, x_labels=None, y_labels=None, cmap="magma",
                 cbar_label=None, annotate=False, fmt="{:.2f}"):
    """api.md's make_heatmap."""
    m = np.asarray(matrix, dtype=float)
    im = ax.imshow(m, cmap=cmap, aspect="auto")
    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
    if annotate:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                        fontsize=FS_ANNOT)
    if cbar_label:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label(cbar_label, fontsize=FS_AXIS)
        cb.ax.tick_params(labelsize=FS_TICK)
        cb.outline.set_linewidth(LW_SPINE)
    return im


# ── Provisional-content marks. Each is ONE call; delete the line to remove. ──

def watermark(fig, text="NOT UPDATED"):
    """Diagonal stamp for a figure held back from the current pass.

    Removal is deleting the single call site. Nothing else in the script
    depends on it, by construction.
    """
    # Sized to the canvas, not fixed: a fixed 26 pt runs off a 2.5 in figure.
    size = min(26.0, 4.2 * fig.get_figwidth())
    fig.text(0.5, 0.5, text, fontsize=size, color=EXTRA["neutral_text"],
             alpha=0.20, ha="center", va="center", rotation=22,
             zorder=1000, fontweight="bold")


def placeholder_banner(fig, text="PLACEHOLDER - SYNTHETIC DATA, NOT A RESULT"):
    """Stamp for a figure whose NUMBERS are invented pending a real run.

    Says what the watermark cannot: the layout is final and the values are not,
    so the future edit is replacing constants. Removal is deleting this one
    call. Any script that calls this must also refuse to write into
    figures_in_use/ -- see ``save``'s ``placeholder`` argument, which enforces
    it rather than trusting the author to remember.
    """
    fig.text(0.5, 0.5, text, fontsize=11, color=PALETTE["red_strong"],
             alpha=0.30, ha="center", va="center", rotation=18,
             zorder=1000, fontweight="bold")


# ── Verification: the rules that only a check can keep ───────────────────────

def check_type_floor(fig, floor=TYPE_FLOOR):
    """Every printed string at or above the floor. Returns a list of offenders."""
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        try:
            s = t.get_text()
        except Exception:
            continue
        if not s or not t.get_visible():
            continue
        sz = t.get_fontsize()
        if sz < floor - 1e-6:
            bad.append((round(sz, 2), s[:40]))
    return sorted(set(bad))


def check_width(path, expected_px):
    """The 1:1 rule, verified on the emitted file rather than assumed."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return w, h, (w == expected_px)


def save(fig, *rel_paths, script_dir, dpi=DPI, expect_px=PX_COL,
         placeholder=False):
    """Write, then verify width and type floor, and say so on stdout.

    ``placeholder=True`` drops any path under ``figures_in_use/``: a figure
    built on invented numbers must not be reachable by the LaTeX build, and
    making that a mechanism rather than a habit is the point.
    """
    offenders = check_type_floor(fig)
    if offenders:
        raise ValueError(f"type below the {TYPE_FLOOR} pt floor: {offenders}")
    if dpi < DPI_FLOOR:
        raise ValueError(f"dpi {dpi} is under the {DPI_FLOOR} floor")

    written = []
    for rel in rel_paths:
        path = os.path.normpath(os.path.join(script_dir, rel))
        if placeholder and "figures_in_use" in path.replace(os.sep, "/"):
            print(f"SKIPPED {path}  (placeholder: synthetic numbers stay out "
                  f"of the build)")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor="white")
        w, h, ok = check_width(path, expect_px)
        flag = "OK" if ok else f"WIDTH MISMATCH, expected {expect_px}"
        print(f"wrote {path}  {w} x {h} px  {flag}")
        if not ok:
            raise ValueError(f"{path}: emitted {w} px, expected {expect_px}; "
                             f"the LaTeX scale factor is no longer 1")
        written.append(path)
    return written


def _selfcheck():
    import matplotlib.font_manager as fm
    print("paperstyle self-check")
    have = {f.name for f in fm.fontManager.ttflist}
    s = apply_publication_style()
    chosen = next((f for f in s.font_family if f in have), "sans-serif")
    print(f"  font        : {chosen}  (requested {s.font_family[0]})")
    print(f"  column      : {COL_W} in at {DPI} dpi -> {PX_COL} px")
    print(f"  full width  : {FULL_W} in at {DPI} dpi -> {PX_FULL} px")
    print(f"  spine lw    : {s.axes_linewidth}  (house 3 at font 24, ratio 8:1)")
    print(f"  type floor  : {TYPE_FLOOR} pt; body {FS_AXIS}, ticks {FS_TICK}")
    print(f"  palette     : {len(PALETTE)} house + {len(EXTRA)} project")
    print(f"  roles       : {', '.join(sorted(ROLE))}")
    print(f"  alpha ramp 3: {[round(a, 3) for a in alpha_ramp(3)]}")

    fig, axes = create_subplots(1, 1, figsize=(COL_W, 2.2))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.96)
    x = np.linspace(0, 10, 40)
    make_trend(axes[0], x, [x * 1.5, x * 1.0], ["a", "b"],
               colors=[ROLE["engine"], ROLE["static"]],
               xlabel="x", ylabel="y")
    axes[0].legend(loc="upper left")
    out = save(fig, "_selfcheck.png", script_dir=os.path.dirname(
        os.path.abspath(__file__)))
    plt.close(fig)
    for p in out:
        os.remove(p)
        print(f"  removed {p}")
    print("  OK")


if __name__ == "__main__":
    _selfcheck()
