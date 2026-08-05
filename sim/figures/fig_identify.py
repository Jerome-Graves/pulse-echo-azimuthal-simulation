r"""Figure 'identify': the specimen picked out of a line-up of 48.

Supports the claim of Sec. 5.2 that the backscattered coda identifies the
microstructure that produced it. For each specimen the facet model is
evaluated for its own tessellation and for 47 others drawn from the same
construction rules, all through identical code, and the whole set is
scored against the one measured field. If the coda were stochastic, the
specimen's own tessellation would land anywhere in the pack.

The unit of replication is the TESSELLATION and not the sweep, because
several sweeps can carry the same microstructure. An earlier version of
this figure plotted seven sweeps against 44 candidates from a cache that
no longer matches the production programme, which has since supplied
eight independent tessellations and 48 candidates.

The uniform-orientation control is plotted last. It has the geometry of
the seed-11 specimen but no acoustic contrast across its boundaries, so
it cannot scatter, and its own tessellation should score no better than
any other. It does not, which is what excludes a purely geometric
artefact of the staircase rendering.

Reads (no simulation, no GPU):
    ../analysis/tessellation_replication.npz
        names      the 17 sweep names, the eight girdle rows first
        cands      the 48 candidate seeds
        ident      (17, 48) score of every candidate against every sweep
        ident_own  (17, 6) own score, rank, p, z, runner-up, its score

Writes <paper>/figures/identify.pdf and prints the counts and the exact
binomial level quoted in the caption.

A note on the scores plotted. Range is mapped to time with the two-way
speed measured from each specimen's own backwall, corrected for the delay
of the source wavelet. Uncorrected, that delay enters the speed and is
then added back by the event binner, stretching every predicted arrival
late by 1.15 per cent of its own time. The correction is worth one
identification net, three of eight uncorrected against four of eight
corrected, with only seeds 7 and 89 first in both; its clearer signature
is seed 41, which the uncorrected mapping drives to r = -0.168 and rank
47 of 48, and which reads +0.027 and rank 2 once the mapping is right.
See coda_field in analysis/tessellation_replication.py.
"""
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs                                       # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "analysis" \
    / "tessellation_replication.npz"

# Rows 0 to 7 of 'names' are the eight independent girdle tessellations in
# the order the production programme ran them; row 12 is the zero-contrast
# control on the seed-11 geometry.
GIRDLE = tuple(range(8))
CONTROL = 12

# Column indices within ident_own, as written by
# analysis/tessellation_replication.py.
OWN, RANK = 0, 1


# ────────────────────────────────── load ──────────────────────────────
def load():
    with np.load(DATA, allow_pickle=True) as z:
        return (list(z["names"]), np.asarray(z["cands"]),
                np.asarray(z["ident"]), np.asarray(z["ident_own"]))


# ───────────────────────────────── compute ────────────────────────────
def binom_at_least(k, m, p):
    """Pr(X >= k) for X ~ Bin(m, p), exact.

    Exact rather than asymptotic: m is eight and the tail IS the claim,
    so a normal approximation here would be indefensible.
    """
    return float(sum(comb(m, i) * p ** i * (1 - p) ** (m - i)
                     for i in range(k, m + 1)))


def ranksum_p(ranks, n_cand):
    """Exact Pr(sum of ranks <= observed) under uniform ranks.

    The statistic the figure now leads with. Each rank is uniform on
    1..n_cand and independent across tessellations, so the sum has an
    exactly enumerable null; while the sum does not exceed n_cand it
    collapses to the hockey-stick identity C(s, m) / n_cand^m. Counting
    first places instead discards every near miss, which on this data
    costs four orders of magnitude.
    """
    s, m = int(sum(ranks)), len(ranks)
    if s <= n_cand:
        return float(comb(s, m)) / float(n_cand) ** m
    # general case: integer convolution of m uniform ranks
    dist = np.zeros(m * n_cand + 1)
    dist[1:n_cand + 1] = 1.0 / n_cand
    tot = dist.copy()
    for _ in range(m - 1):
        tot = np.convolve(tot, dist)[:m * n_cand + 1]
    return float(tot[:s + 1].sum())


def row_label(name):
    """The development specimen is named as such, because whether the
    result depends on it is the first thing a reader should check."""
    if name == "girdle_seed11_ppw8_dev":
        return "seed 11 (development)"
    if name.startswith("mx_girdle_s"):
        return "seed " + name.split("_s")[1].split("_")[0]
    if name.startswith("zerocontrast"):
        return "uniform orientation"
    return name


# ─────────────────────────────────── draw ─────────────────────────────
def draw(ax, names, ident, own, rows):
    for y, i in enumerate(rows):
        mine = own[i, OWN]
        rank = int(own[i, RANK])
        others = np.array([s for s in ident[i] if s != mine])
        ax.plot([others.min(), others.max()], [y, y], color=fs.GREY_LIGHT,
                lw=fs.LW["annotation"], zorder=1)
        ax.plot(others, np.full(others.size, y), "o", ms=2.0, mfc="none",
                mec=fs.GREY_LIGHT, mew=0.45, zorder=2)
        first = rank == 1
        ax.plot([mine], [y], "D", ms=4.0, zorder=4, mec=fs.BLACK, mew=0.9,
                mfc=fs.BLACK if first else "white")
        ax.text(1.02, y, str(rank), transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=fs.FS["min"],
                color=fs.BLACK if first else fs.GREY)
    ax.axvline(0.0, color=fs.GREY, lw=fs.LW["annotation"], ls=(0, (1, 2)),
               zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row_label(names[i]) for i in rows])
    ax.invert_yaxis()
    ax.set_xlabel("correlation with the predicted echo train")
    ax.margins(x=0.10, y=0.07)


def main():
    names, cands, ident, own = load()
    rows = list(GIRDLE) + [CONTROL]
    n_cand = len(cands)
    k = sum(1 for i in GIRDLE if int(own[i, RANK]) == 1)
    p = binom_at_least(k, len(GIRDLE), 1.0 / n_cand)
    ranks = [int(own[i, RANK]) for i in GIRDLE]
    pr = ranksum_p(ranks, n_cand)
    expo = int(np.floor(np.log10(pr)))

    fig, ax = fs.figure(width="single", height_mm=74.0,
                        left=30.0, right=7.0, bottom=9.0, top=4.5)
    draw(ax, names, ident, own, rows)
    ax.text(0.02, 0.99, "%d alternatives" % (n_cand - 1),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=fs.FS["annotation"])
    # The rank sum leads, not the first-place count: the count discards
    # the near misses and is weaker on this data by a factor of 2.7e4.
    # Bottom right. Top right collides with the alternatives count, and
    # the middle rows carry the foreign clouds; the corner below the
    # seed-89 marker is the only clear space on the canvas.
    ax.text(0.98, 0.02,
            "mean rank %.1f of %d, chance %.1f\n$p = %.1f\\times 10^{%d}$"
            % (float(np.mean(ranks)), n_cand, (n_cand + 1) / 2.0,
               pr / 10 ** expo, expo),
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=fs.FS["annotation"], color=fs.GREY)
    ax.text(1.02, 1.02, "rank", transform=ax.transAxes,
            fontsize=fs.FS["min"], style="italic", color=fs.GREY, ha="left")

    print("identification against %d candidates, unit = tessellation"
          % n_cand)
    for i in rows:
        print("  %-24s own %+.4f  rank %2d"
              % (row_label(names[i]), own[i, OWN], int(own[i, RANK])))
    print("  %d of %d rank first, exact binomial p = %.3g"
          % (k, len(GIRDLE), p))
    kf = sum(1 for i in list(GIRDLE)[1:] if int(own[i, RANK]) == 1)
    print("  excluding the development specimen: %d of %d, p = %.3g"
          % (kf, len(GIRDLE) - 1,
             binom_at_least(kf, len(GIRDLE) - 1, 1.0 / n_cand)))

    fs.save(fig, "identify",
            expect=["correlation with the predicted echo train",
                    "alternatives", "mean rank", "uniform orientation"])


if __name__ == "__main__":
    main()
