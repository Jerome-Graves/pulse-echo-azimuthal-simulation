r"""House matplotlib style for the pulse-echo COF manuscript.

Every figure script in sim/figures imports this module. It is the single
thing that makes the fourteen figures of the paper look like one paper:
one page geometry, one set of type sizes, one series encoding, one save
path, one set of checks on the way out.

Reads nothing. Writes nothing on import. save() writes
    <GitHub>/pulse-echo-cof-paper/figures/<name>.pdf
and prints the finished page size in millimetres, so a figure that has
been rescaled or has outgrown the page is caught the moment it is made.

Page geometry was measured from an Elsevier-typeset article in this
literature (Simonetti, NDT and E International 137 (2023) 102835) and
confirmed by compiling elsarticle: \columnwidth 252 pt (88.90 mm),
\textwidth 522 pt (184.15 mm), \textheight 682 pt (240.6 mm). Elsevier's
published artwork slots are 90 / 140 / 190 mm; the three widths below sit
just inside them and match elsarticle exactly.

Figures are authored at final printed size and must be included with an
explicit width,

    \includegraphics[width=88.9mm]{figures/regime.pdf}

NOT width=\columnwidth. The review build sets \columnwidth to 165.1 mm,
which would magnify every label by 1.86x and put the 7 pt tick labels at
13 pt. save() prints the correct line for each figure. Wiring the figures
into the manuscript is the caller's job, not this module's.

Dependencies: matplotlib and numpy. PyMuPDF, if it happens to be
installed, is used by save() to verify the finished page (width, fonts,
extractable text); without it that check is reported as skipped.
"""
from pathlib import Path

import matplotlib as mpl

# Figures are written, never shown. Agg keeps every script runnable on a
# machine with no display and stops windows appearing during a batch.
mpl.use("Agg")

import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
from matplotlib.ticker import (FixedFormatter,         # noqa: E402
                               FixedLocator,
                               NullFormatter)

# ------------------------------------------------------------------ page

MM = 1 / 25.4                       # millimetres to inches

W_SINGLE = 88.90 * MM               # = \columnwidth, Elsevier 90 mm slot
W_ONEHALF = 140.00 * MM             # Elsevier 1.5-column slot
W_DOUBLE = 184.15 * MM              # = \textwidth, Elsevier 190 mm slot
H_MAX = 230.0 * MM                  # ceiling: 240.6 mm text height, less
                                    # room for the caption

WIDTH_MM = {"single": 88.90, "onehalf": 140.00, "double": 184.15}

# Assigned size of every figure in the manuscript, (width mm, height mm).
# Heights include the axis labels but not the caption. Keeping the table
# here rather than in each script means one place decides whether the
# paper's figures line up.
FIGURES = {
    "regime":         (88.90, 62.0),    # 1 panel
    "geometry":       (140.00, 62.0),   # main schematic plus inset
    "microstructure": (184.15, 64.0),   # 4 panels in a row
    "trace":          (88.90, 55.0),    # 1 panel, log y
    "tiltmethod":     (140.00, 60.0),   # 1 schematic
    "staircase":      (140.00, 48.0),   # 3 panels in a row, no axes
    "tilterror":      (184.15, 62.0),   # 1 x 3 panels, shared y
    "dispersion":     (88.90, 78.0),    # 2 panels stacked, shared x
    "ensemble":       (88.90, 58.0),    # 1 panel
    "tofaz":          (88.90, 62.0),    # 1 panel, 3 series
    "codafield":      (184.15, 96.0),   # 3 images above, 1 wide panel
    "identify":       (88.90, 58.0),    # 1 panel
    "scales":         (88.90, 48.0),    # 1 panel, no y axis
    "concept":        (88.90, 55.0),    # 1 schematic
}

DEFAULT_HEIGHT_MM = 62.0            # the modal assigned height above

# Margins are carried in millimetres, not fractions, because the same
# 8 pt axis label needs the same physical space whether the panel is
# 88.90 or 184.15 mm wide. The defaults hold a 7 pt tick label, an 8 pt
# axis label at labelpad 2.5, and a panel letter above the axes.
MARGIN_MM = dict(left=11.0, right=2.5, bottom=8.5, top=4.5)

# Any row carrying a colourbar needs this much horizontal space. At 0.42
# the colourbar label collided with the next panel's y label.
WSPACE_COLOURBAR = 0.55

PAPER_FIGURES = (Path(__file__).resolve().parents[3]
                 / "pulse-echo-cof-paper" / "figures")

# ---------------------------------------------------------------- colour

# Okabe-Ito, ordered so that the worst-case separation as series are
# added is maximised under deuteranopia, protanopia, tritanopia, full
# colour and greyscale. Use series in this order, always. Beyond four
# series colour alone fails, which is why the pairing below is not
# optional. Okabe-Ito yellow is absent on purpose: at a relative
# luminance of 0.744 it is invisible as a line on white.
SERIES = [
    ("#000000", (0, ()),                              "o"),
    ("#0072B2", (0, (4.0, 1.6)),                      "s"),
    ("#E69F00", (0, (1.0, 1.4)),                      "^"),
    ("#CC79A7", (0, (5.5, 1.6, 1.0, 1.6)),            "D"),
    ("#D55E00", (0, (2.0, 1.4, 1.0, 1.4, 1.0, 1.4)),  "v"),
    ("#009E73", (0, (7.0, 2.0)),                      "P"),
]
PALETTE = [c for c, _, _ in SERIES]
BLACK, BLUE, ORANGE, PURPLE, VERMILION, BLUEGREEN = PALETTE

GREY = "#4D4D4D"        # annotation, reference slopes, secondary text
GREY_LIGHT = "#BFBFBF"  # individual realisations, gate shading, cell grid
GRID = "#CCCCCC"
TINT_A = "#B9CFE4"      # schematic medium A
TINT_B = "#F7E7CE"      # schematic medium B, 0.208 of greyscale gap from A
HIGHLIGHT = "#F0E442"   # Okabe-Ito yellow: fills and highlights only

MAX_MARKERS = 12        # markers per curve, above which the line is lost

# Luminance monotonicity checked numerically over 65 samples. Anything
# not monotonic (RdBu_r, coolwarm, seismic, turbo, jet) cannot carry
# magnitude, because the print run may be greyscale.
CMAP_SEQ = "viridis"        # magnitude fields, always with a colourbar
CMAP_SEQ_ALT = "cividis"    # second sequential scale, if two are needed
CMAP_SEQ_DARK = "magma"     # widest luminance span of the four
CMAP_MONO = "Greys"         # strictly mono figures
CMAP_SIGNED = "RdBu_r"      # signed wavefields ONLY, and only with the
                            # zero level drawn as a contour or a clipping
                            # level stated in the caption

# Line weights, in points, gathered so that a reader can see the whole
# hierarchy at once rather than hunting through fourteen scripts.
LW = {
    "data": 0.9, "emphasis": 1.1, "realisation": 0.3, "spine": 0.5,
    "minor_tick": 0.4, "reference": 0.6, "annotation": 0.4,
    "schematic": 0.5, "interface": 0.8, "cell_grid": 0.25,
    "scale_bar": 1.2, "colourbar": 0.5, "errorbar": 0.5,
}

# Type sizes, in points, at the final printed width. Nothing below 6.5 pt
# except mathtext super and subscripts, which land at 0.7x of their base.
FS = {
    "label": 8.0, "tick": 7.0, "legend": 7.0, "panel": 8.0,
    "condition": 7.0, "annotation": 7.0, "cbar_label": 7.0,
    "cbar_tick": 6.5, "min": 6.5,
}

# ------------------------------------------------------------- rcParams

RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 7.0,
    "axes.labelsize": 8.0, "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.5, "axes.edgecolor": "#000000",
    "axes.labelcolor": "#000000",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": False,
    "grid.color": "#CCCCCC", "grid.linewidth": 0.3,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "axes.labelpad": 2.5,
    "lines.linewidth": 0.9, "lines.markersize": 3.0,
    "lines.markeredgewidth": 0.5,
    "legend.frameon": True, "legend.framealpha": 1.0,
    "legend.edgecolor": "#000000", "legend.fancybox": False,
    "legend.borderpad": 0.35, "legend.labelspacing": 0.28,
    "legend.handlelength": 2.2, "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.4, "legend.columnspacing": 1.0,
    "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.unicode_minus": True,
}


def apply():
    """Install the house style.

    fonttype 42 because matplotlib defaults to Type 3, which Elsevier
    discourages and which some referees cannot search or copy from.
    savefig.bbox 'standard' because 'tight' silently changes the page
    width, and the whole point of authoring at final size is that the
    width is the one asked for. unicode_minus so a minus sign is U+2212
    and not a hyphen.
    """
    mpl.rcParams.update(RC)


# ------------------------------------------------------------ colormaps

def cmap(name=CMAP_SEQ):
    """Fetch a colormap.

    matplotlib 3.11 removed matplotlib.cm.get_cmap, so the registry is
    the only supported way in. Wrapped here so no figure script has to
    care again.
    """
    return mpl.colormaps[name]


def luminance(colour):
    """Rec. 709 relative luminance of a colour, 0 (black) to 1 (white).

    The paper prints in mono unless the caption says otherwise, so this
    is the number that decides whether two colours are still two things
    on the page. Use it to check a palette rather than trusting the eye
    on a bright screen.
    """
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
           for c in mpl.colors.to_rgb(colour)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def greyscale(colour):
    """The grey a colour becomes when the page is printed in mono."""
    y = luminance(colour)
    s = y * 12.92 if y <= 0.0031308 else 1.055 * y ** (1 / 2.4) - 0.055
    return mpl.colors.to_hex((s, s, s))


# --------------------------------------------------------------- series

def series(i, n=None, reference=False, emphasis=False, dense=False):
    """Keyword set for series i: colour paired with dash and marker.

    Colour alone fails for a colourblind reader and fails again in
    greyscale, so every series carries all three channels and any one of
    them is enough to tell it apart. i wraps around the palette, but past
    four series the colours are no longer separable in mono and the dash
    and marker are doing all of the work.

    reference draws open markers, the convention in this literature for
    measured points against a theory line (Huang, Runborg). emphasis is
    for the one line the panel is about (an ensemble mean, a truth
    curve). n is the number of points in the curve and is used to thin
    the markers to at most MAX_MARKERS, below which they annotate the
    line and above which they bury it.
    """
    colour, dash, marker = SERIES[i % len(SERIES)]
    kw = dict(color=colour, linestyle=dash, marker=marker,
              markersize=2.4 if dense else 3.0,
              markeredgecolor=colour, markeredgewidth=0.5,
              markerfacecolor="white" if reference else colour,
              linewidth=LW["emphasis"] if emphasis else LW["data"])
    if n is not None:
        kw["markevery"] = max(1, int(np.ceil(n / MAX_MARKERS)))
    return kw


# --------------------------------------------------------------- canvas

def _size_mm(width, height_mm):
    """Resolve a size given as a figure name, a slot name, or millimetres."""
    if isinstance(width, str):
        if width in FIGURES:
            w, h = FIGURES[width]
            return w, h if height_mm is None else height_mm
        if width not in WIDTH_MM:
            raise KeyError("unknown width %r: use a figure name from "
                           "FIGURES, one of %s, or a number in mm"
                           % (width, sorted(WIDTH_MM)))
        w = WIDTH_MM[width]
    else:
        w = float(width)
    return w, DEFAULT_HEIGHT_MM if height_mm is None else height_mm


def canvas(width="single", height_mm=None):
    """Empty figure at final printed size, for irregular layouts.

    width is a figure name from FIGURES (which also supplies its assigned
    height), a slot name ('single', 'onehalf', 'double'), or a number in
    millimetres. Pair with mm_margins() and fig.add_gridspec() when the
    panels are not a plain grid.
    """
    w, h = _size_mm(width, height_mm)
    if h > H_MAX / MM:
        raise ValueError("height %.1f mm exceeds the %.1f mm ceiling; the "
                         "caption still has to fit under it" % (h, H_MAX / MM))
    return plt.figure(figsize=(w * MM, h * MM))


def mm_margins(fig, **margins_mm):
    """Margins in millimetres as the fractions gridspec wants.

    Keys are left, right, bottom, top, each the clear space at that edge
    in millimetres. Defaults are MARGIN_MM.
    """
    m = dict(MARGIN_MM)
    unknown = set(margins_mm) - set(m)
    if unknown:
        raise KeyError("not a margin: %s" % sorted(unknown))
    m.update(margins_mm)
    w, h = fig.get_size_inches()
    out = dict(left=m["left"] * MM / w, right=1 - m["right"] * MM / w,
               bottom=m["bottom"] * MM / h, top=1 - m["top"] * MM / h)
    if out["left"] >= out["right"] or out["bottom"] >= out["top"]:
        raise ValueError("margins %s do not fit a %.1f x %.1f mm figure"
                         % (m, w / MM, h / MM))
    return out


def figure(width="single", height_mm=None, nrows=1, ncols=1,
           sharex=False, sharey=False, wspace=0.30, hspace=0.30,
           **margins_mm):
    """Figure and axes at the correct Elsevier width, laid out explicitly.

    Returns (fig, ax) for a single panel and (fig, axes) otherwise.

    The layout is set here and never touched again: no tight_layout and
    no bbox_inches='tight' anywhere, because both change the final width
    and break the contract that a point in this script is a point on the
    page. Pass wspace=WSPACE_COLOURBAR in any row carrying a colourbar.
    """
    fig = canvas(width, height_mm)
    gs = dict(wspace=wspace, hspace=hspace, **mm_margins(fig, **margins_mm))
    return fig, fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                             squeeze=True, gridspec_kw=gs)


# ------------------------------------------------------- panel labelling

DX_NO_YLABEL = -0.022   # panels with no y tick labels sit this much closer


def panel_letter(fig, ax, s, dx=-0.055, dy=0.012):
    """Place a panel letter in FIGURE coordinates, above and left of ax.

    Figure coordinates, not axes coordinates, so that the letters line up
    in a column down the whole figure even when the panels have different
    heights. Call after the layout is final (after any colourbar), since
    it reads ax.get_position().
    """
    bb = ax.get_position()
    return fig.text(bb.x0 + dx, bb.y1 + dy, s, fontsize=FS["panel"],
                    fontweight="bold", va="bottom", ha="left")


def panel(ax, letter, inside=False, dx=-0.055, dy=0.012):
    """Elsevier panel letter, bold and bracketed, '(a)'.

    Outside the axes by default. inside=True puts it top-left within the
    axes on a white box, which is the only option on an image panel that
    fills its cell (Simonetti Fig. 6). Not Nature's bare bold 'a'.
    """
    s = letter if letter.startswith("(") else "(%s)" % letter
    if inside:
        return ax.text(0.03, 0.955, s, transform=ax.transAxes,
                       fontsize=FS["panel"], fontweight="bold",
                       va="top", ha="left", color="black", zorder=6,
                       bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    return panel_letter(ax.get_figure(), ax, s, dx, dy)


def condition(ax, text, x=0.035, y=0.94, box=False, colour="black"):
    """Condition label inside the axes, top-left.

    A condition ('girdle', n_lambda = 6) is not a title and never sits
    above the axes as one. Journals put that text in the caption; this is
    the one place it is allowed on the plot, where it labels the panel it
    belongs to and costs no height.

    This corner is also where an inside panel letter goes, so on an image
    panel that carries one, move the condition right (x around 0.17).
    """
    kw = dict(fc="white", ec="none", alpha=0.85, pad=1.2) if box else None
    return ax.text(x, y, text, transform=ax.transAxes,
                   fontsize=FS["condition"], va="top", ha="left",
                   color=colour, zorder=6, bbox=kw)


def direct_label(ax, x, y, text, colour=BLACK, ha="left", va="bottom", **kw):
    """Name a curve on the plot, in its own colour.

    Direct labelling is the default and a legend is the fallback: the
    label sits where the reader's eye already is, costs no plotting area,
    and survives greyscale as long as the dash pattern does.
    """
    return ax.text(x, y, text, color=colour, fontsize=FS["annotation"],
                   ha=ha, va=va, **kw)


def legend(ax, *args, **kw):
    """Boxed legend inside the axes, in the emptiest corner.

    The frame width is not an rcParam (it follows patch.linewidth), so it
    has to be set after the legend exists, otherwise the box is 1.0 pt
    against 0.5 pt spines and looks like a mistake. In a multi-panel
    figure put the legend in one panel only.
    """
    leg = ax.legend(*args, **kw)
    leg.get_frame().set_linewidth(LW["spine"])
    return leg


# ----------------------------------------------------- images and rules

def colourbar(mappable, ax, label=None, fraction=0.055, pad=0.03, **kw):
    """Vertical colourbar, one per panel group rather than per panel.

    Pass a list of axes for a shared scale (Simonetti Fig. 6): a second
    identical colourbar is width the figure cannot spare.
    """
    fig = np.ravel(ax)[0].get_figure()
    cb = fig.colorbar(mappable, ax=ax, fraction=fraction, pad=pad, **kw)
    cb.outline.set_linewidth(LW["colourbar"])
    cb.ax.tick_params(width=LW["spine"], size=2.0,
                      labelsize=FS["cbar_tick"])
    if label is not None:
        cb.set_label(label, fontsize=FS["cbar_label"])
    return cb


def scale_bar(ax, x, y, length, label, colour="black", zorder=6):
    """Plain bar in data coordinates with its length written above it.

    No ticks and no box. Required wherever the reader needs a physical
    size to judge the image by, which includes drawing the wavelength to
    scale on any figure that talks about resolution.
    """
    ax.plot([x, x + length], [y, y], color=colour, linewidth=LW["scale_bar"],
            solid_capstyle="butt", zorder=zorder)
    return ax.annotate(label, xy=(x + 0.5 * length, y), xytext=(0, 2.0),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=FS["annotation"], color=colour, zorder=zorder)


def arrow_label(ax, text, xy, xytext, colour="black", **kw):
    """Name a feature and point at it. Black over light, white over dark.

    The caption should not be the only place a feature is identified: a
    reader scanning the figures alone has to be able to find it.
    """
    return ax.annotate(
        text, xy=xy, xytext=xytext, fontsize=FS["annotation"], color=colour,
        arrowprops=dict(arrowstyle="->", linewidth=LW["annotation"],
                        color=colour, shrinkA=1.0, shrinkB=1.0),
        zorder=6, **kw)


def axes_triad(ax, x, y, length, labels=("x", "z"), colour="black",
               zorder=6):
    """Two small labelled arrows fixing the orientation of a schematic.

    A schematic with no axes still has to say which way is which, and a
    triad does it in less space than a pair of axis labels.
    """
    for (dx, dy), name in zip(((length, 0.0), (0.0, length)), labels):
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=colour,
                                    linewidth=LW["schematic"]), zorder=zorder)
        ax.text(x + 1.25 * dx, y + 1.25 * dy, "$%s$" % name, color=colour,
                fontsize=FS["annotation"], ha="center", va="center",
                zorder=zorder)


def schematic(ax):
    """A schematic panel carries no axes at all (Koene Fig. 3).

    Set zorder explicitly on every element you then draw: the default
    ordering will hide the cell grid under the medium patches and the
    true interface under both.

    The anchor matters. An equal-aspect axes shrinks inside the cell it
    was given, and by default it shrinks about the centre, while
    panel_letter() reads the cell. Pinning it to the top left keeps the
    drawing and its letter together.
    """
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_anchor("NW")


# --------------------------------------------- convergence and log axes

def reference_slope(ax, x0, x1, y0, slope, label, offset=2.0):
    """Thin asymptote drawn beside the data and labelled where it lies.

    The convention in this literature (Huang Fig. 3, Runborg Fig. 6.1) is
    a labelled straight line, not a filled slope triangle. The line is
    y = y0 (x/x0)**slope, so slope=-1 falls a decade per decade. The
    label angle comes from the data-to-display transform, because on a
    log axis or at any other aspect a hard-coded angle is wrong. Call it
    after the limits are set.
    """
    y1 = y0 * (x1 / x0) ** slope
    line, = ax.plot([x0, x1], [y0, y1], color=GREY,
                    linestyle=(0, (3.0, 1.5)), linewidth=LW["reference"],
                    zorder=2)
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    xm, ym = ax.transData.inverted().transform(0.5 * (p0 + p1))
    text = ax.annotate(label, xy=(xm, ym), xytext=(0, offset),
                       textcoords="offset points", fontsize=FS["annotation"],
                       color=GREY, ha="center", va="bottom", rotation=angle,
                       rotation_mode="anchor", zorder=2)
    return line, text


def log_minor_off(ax):
    """Stop matplotlib labelling log minor ticks.

    Left alone it writes 2x10^-1, 3x10^-1 and so on between the decades,
    which is clutter, and it renders the exponent at 0.7 x 7 = 4.9 pt.
    The minor ticks themselves stay: they are what makes a log axis
    readable.
    """
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def plain_log_ticks(ax, which, values, labels=None):
    """Label a short log axis with plain numbers, no superscript at all.

    Use where the range spans three decades or fewer: '6 8 10 14 20'
    reads instantly, 10^0.78 does not.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(FixedLocator(list(values)))
    axis.set_major_formatter(FixedFormatter(
        list(labels) if labels is not None else ["%g" % v for v in values]))
    axis.set_minor_formatter(NullFormatter())


def convergence_grid(ax):
    """The one licensed grid: dotted minors on a log-log convergence panel.

    Runborg Fig. 6.1 uses it to let the reader read an order off the
    page. Nowhere else in the paper carries a grid.
    """
    ax.set_axisbelow(True)
    ax.grid(True, which="both", color=GRID, linewidth=0.3,
            linestyle=(0, (1, 2)))


def errorbar(ax, x, y, yerr, i, label=None, reference=True, **kw):
    """Series i with vertical error bars, sized so they read at 7 pt.

    State the interval in the caption (Ray Fig. 5 says '99.73% confidence
    interval (3 sigma rule)'). If the bars come out smaller than the
    markers, drop them and say so in the caption instead, which is what
    Huang Fig. 3 does.
    """
    s = series(i, reference=reference)
    s.update(kw)
    return ax.errorbar(x, y, yerr=yerr, label=label, elinewidth=LW["errorbar"],
                       capsize=1.6, capthick=LW["errorbar"], ecolor=s["color"],
                       zorder=3, **s)


def confidence_band(ax, x, lo, hi, i):
    """Shaded interval under its mean line, in the series colour."""
    return ax.fill_between(x, lo, hi, color=PALETTE[i % len(PALETTE)],
                           alpha=0.18, linewidth=0, zorder=1)


# ----------------------------------------------------------------- save

def _verify(path, w_mm, h_mm, expect):
    """Check the finished PDF. Returns a one-line report.

    Three things go wrong silently and only this catches them: a bbox
    setting that resizes the page, Type 3 fonts that Elsevier will bounce,
    and text that has been drawn as paths and so cannot be searched.
    """
    try:
        import fitz                                    # PyMuPDF, optional
    except ImportError:
        return "page not verified (PyMuPDF absent)"
    with fitz.open(path) as doc:
        page = doc[0]
        pw = page.rect.width / 72.0 * 25.4
        ph = page.rect.height / 72.0 * 25.4
        if abs(pw - w_mm) > 0.1 or abs(ph - h_mm) > 0.1:
            raise AssertionError(
                "%s is %.2f x %.2f mm, asked for %.2f x %.2f mm: something "
                "rescaled it" % (path.name, pw, ph, w_mm, h_mm))
        type3 = sorted({f[3] for f in page.get_fonts(full=True)
                        if f[2] == "Type3"})
        if type3:
            raise AssertionError("%s embeds Type 3 fonts %s: set "
                                 "pdf.fonttype 42" % (path.name, type3))
        text = page.get_text()
        if not text.strip():
            raise AssertionError("%s has no extractable text: the labels "
                                 "have been drawn as paths" % path.name)
        missing = [s for s in expect if s not in text]
        if missing:
            raise AssertionError("%s does not contain %s"
                                 % (path.name, missing))
        nfonts = len({f[3] for f in page.get_fonts(full=True)})
    return ("page %.2f x %.2f mm, no Type 3, %d font(s), %d characters of "
            "searchable text" % (pw, ph, nfonts, len(text)))


def save(fig, name, expect=(), outdir=None):
    """Write <name>.pdf to the paper repository and check it on the way out.

    expect is a list of plain strings (axis labels, units) that must come
    back out of the finished PDF as text. Prints the final size in
    millimetres and the inclusion line to use, so a figure that has
    outgrown its slot, or that would be magnified by a bare
    width=\\columnwidth, is caught here rather than at proof stage.
    """
    out = Path(outdir) if outdir is not None else PAPER_FIGURES
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("%s.pdf" % name)
    fig.savefig(path)                    # PDF only, and never rasterised text

    w_mm, h_mm = (v / MM for v in fig.get_size_inches())
    report = _verify(path, w_mm, h_mm, expect)
    slot = [k for k, v in WIDTH_MM.items() if abs(v - w_mm) < 0.1]
    print("%-16s %7.2f x %6.2f mm  %s" % (path.name, w_mm, h_mm, report))
    print("    \\includegraphics[width=%gmm]{figures/%s.pdf}"
          % (round(w_mm, 2), name))
    if not slot:
        print("    WARNING: %.2f mm is not one of the %s widths"
              % (w_mm, sorted(WIDTH_MM)))
    if h_mm > H_MAX / MM:
        print("    WARNING: %.1f mm tall leaves no room for a caption"
              % h_mm)
    return path


# The style is installed on import as well as on demand: a script that
# forgets to call apply() must not quietly emit a matplotlib-default
# figure into a paper where the other thirteen agree.
apply()
