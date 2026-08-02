r"""Figure 'ensemble': where the ensemble mean stops moving.

Supports Section 4, "Ensemble statistics". The question the panel answers
is how many independent microstructures the headline coda level needs
before its mean is settled, and the answer is read off the width of the
band: with eight girdle tessellations the mean is -84.90 dB re source and
the 95 per cent interval on it is +-0.50 dB, four realisations already
land inside that interval, and the interval is a quarter of the
1.71 dB separation between the two fabrics drawn on the same axes.

Reads (paths resolved from this file, nothing absolute):
    ../results/ensemble.npz     written by analysis/ensemble_stats.py.
                                Keys used: girdle_levels, single_levels,
                                girdle_seeds, conf.

Writes ../../../pulse-echo-cof-paper/figures/ensemble.pdf and prints the
numbers the section quotes.

WHY THE BAND IS ENUMERATED AND NOT BOOTSTRAPPED. The shaded band is the
exact distribution of the mean of N realisations drawn from the eight,
over all 254 non-trivial subsets, not a bootstrap of them. With eight
values the distribution is a finite object and can be written down; a
bootstrap would resample the same eight with replacement and add noise to
an answer already in closed form. The hairlines are its full range.

WHY THE BAND CLOSES AT N = 8 AND THE INTERVAL DOES NOT. The band answers
"how far would my answer have moved had I stopped at N", so at N = 8 it
is a point: there is only one way to average all eight. The residual
uncertainty on the population mean is a different quantity and is the
horizontal dotted pair, which does not close. The two are drawn together
because reading only the first would say that eight realisations settle
the level exactly, which they do not.

REFERENCE. Levels are referred to the source amplitude and never to the
backwall echo. Section 4 establishes that the backwall is inadmissible
across grid resolutions because refining the grid raises it while
lowering the coda; the same reference is kept here so that an ensemble
spread and a resolution step are the same kind of decibel.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import figstyle as fs                                   # noqa: E402
from ensemble_stats import mean_interval, subsample_band  # noqa: E402

ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "ensemble.npz"

# Percentiles of the subsample distribution that the shaded core spans.
# A 90 per cent core rather than 95, because at N = 2 there are only 28
# subsets and a 95 per cent core is indistinguishable from the full
# range, which is drawn separately as the hairlines.
CORE_PC = (5.0, 95.0)

Y_LIMITS_DB = (-86.3, -82.15)


# ------------------------------------------------------------------ draw

def draw_family(ax, levels, index, label, label_xy, hairlines=False):
    """One fabric: subsample band, ensemble mean, and the realisations.

    Returns the mean and the half width of its interval, so that the
    caller can draw the interval of the family the panel is about without
    recomputing it.
    """
    counts, lo, hi, vlo, vhi = subsample_band(levels, *CORE_PC)
    mean, _sd, half = mean_interval(levels)
    colour, dash, marker = fs.SERIES[index]

    fs.confidence_band(ax, counts, lo, hi, index)
    if hairlines:
        for edge in (vlo, vhi):
            ax.plot(counts, edge, color=fs.GREY,
                    linewidth=fs.LW["annotation"], zorder=2)
    # The mean is flat by construction: every subsample distribution is
    # centred on it. It is drawn only so the band has a spine, and it
    # stops where that family's realisations run out.
    ax.plot([0.55, counts[-1] + 0.45], [mean, mean], color=colour,
            dashes=dash[1] or (), linewidth=fs.LW["emphasis"], zorder=4)
    # Every realisation at N = 1, which is the one place they are
    # individually visible, drawn open in the convention this paper uses
    # for measured points.
    ax.plot(np.ones_like(levels), levels, linestyle="none", marker=marker,
            markersize=3.0, markeredgewidth=0.5, markeredgecolor=colour,
            markerfacecolor="white", zorder=5)
    fs.direct_label(ax, label_xy[0], label_xy[1], label, colour=colour,
                    ha=label_xy[2], va=label_xy[3])
    return mean, half


def draw_interval(ax, mean, half, x_right):
    """The interval on the population mean, as a pair of horizontal rules.

    Horizontal rather than an error bar at the last point, because the
    reader has to be able to see which N the shaded band first fits
    inside, and that is a comparison along the whole axis.
    """
    for edge in (mean - half, mean + half):
        ax.plot([0.55, x_right], [edge, edge], color=fs.GREY,
                linestyle=(0, (1.0, 1.6)), linewidth=fs.LW["reference"],
                zorder=3)


def draw(ax, girdle, single):
    """The panel."""
    x_right = len(girdle) + 0.45
    g_mean, g_half = draw_family(
        ax, girdle, 0, "girdle, eight tessellations",
        (x_right - 0.1, -85.50, "right", "top"), hairlines=True)
    draw_family(ax, single, 1, "single maximum, four",
                (len(single) + 0.55, -83.10, "left", "bottom"))
    draw_interval(ax, g_mean, g_half, x_right)

    # Named at the right-hand end of its own rule, where the panel is
    # empty, rather than beside the band it is being compared with.
    fs.direct_label(ax, x_right - 0.1, g_mean + g_half + 0.09,
                    "95 %% interval on the mean, $\\pm$%.2f dB" % g_half,
                    colour=fs.GREY, ha="right", va="bottom")

    ax.set_xlim(0.55, x_right)
    ax.set_ylim(*Y_LIMITS_DB)
    ax.set_xticks(range(1, len(girdle) + 1))
    ax.set_yticks([-86, -85, -84, -83, -82])
    ax.set_xlabel("number of independent tessellations")
    ax.set_ylabel("coda level (dB re source)")


# ------------------------------------------------------------------ main

def main():
    fs.apply()
    if not ARCHIVE.exists():
        raise SystemExit("%s is missing: run analysis/ensemble_stats.py "
                         "first" % ARCHIVE)
    with np.load(ARCHIVE) as z:
        girdle = z["girdle_levels"]
        single = z["single_levels"]
        seeds = z["girdle_seeds"]
        conf = float(z["conf"])

    fig, ax = fs.figure("ensemble", left=12.5, right=2.5, bottom=8.5,
                        top=2.5)
    draw(ax, girdle, single)

    g_mean, g_sd, g_half = mean_interval(girdle)
    s_mean, s_sd, s_half = mean_interval(single)
    counts, _lo, _hi, vlo, vhi = subsample_band(girdle, *CORE_PC)
    inside = counts[(vlo >= g_mean - g_half) & (vhi <= g_mean + g_half)]
    print("girdle, %d tessellations (seeds %s)"
          % (len(girdle), ", ".join(str(int(s)) for s in seeds)))
    print("  ensemble mean %.2f dB re source, sd %.2f, %.0f%% interval "
          "+-%.2f dB" % (g_mean, g_sd, 100 * conf, g_half))
    print("  realisations span %.2f to %.2f dB" % (girdle.min(),
                                                   girdle.max()))
    print("  every N-realisation mean lies inside that interval from "
          "N = %d onward" % inside[0])
    print("single maximum, %d tessellations" % len(single))
    print("  ensemble mean %.2f dB re source, sd %.2f, %.0f%% interval "
          "+-%.2f dB" % (s_mean, s_sd, 100 * conf, s_half))
    print("fabric separation %.2f dB, which is %.1f times the girdle "
          "interval" % (s_mean - g_mean, (s_mean - g_mean) / g_half))

    fs.save(fig, "ensemble",
            expect=["number of independent tessellations",
                    "coda level (dB re source)", "girdle",
                    "single maximum", "interval on the mean"])


if __name__ == "__main__":
    main()
