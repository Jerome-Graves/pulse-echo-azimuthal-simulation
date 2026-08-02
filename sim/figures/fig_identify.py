r"""Figure 'identify': the specimen picked out of a line-up of 44.

Supports the claim of Sec. 5.2 that the backscattered coda identifies the
tessellation that produced it. For each sweep the facet model is evaluated
for the specimen's own microstructure and for 43 unrelated tessellations
drawn from the same construction rules, all through identical code, and
the whole set is scored against the one measured field. If the coda were
stochastic the own tessellation would land anywhere in the pack.

Reads:
    <cache>/t4_range.npy
written by analysis/facet_model/t4_range.py. It is an object array with
one row per (sweep, gate, predictor) cell; the columns used here are the
sweep name, the own seed, the gate label, the specular-predictor flag,
the own score, the rank, and the vector of scores over all 44 candidates.
The cells plotted are the range-domain test on the wide gate with the
specular predictor, which is Test A of results/identification.log.

That log is a later re-run of the same test which did not keep the
per-candidate vectors. The two agree on every rank that matters (six of
seven sweeps first) and on the four leading scores to three decimals;
three of the weaker sweeps differ in the second decimal. The figure is
drawn from the run whose candidate vectors survive, so that the strip a
reader sees is the population the quoted rank was taken against.

Writes:
    <paper>/figures/identify.pdf

Prints the own score, rank and the identification probability quoted in
Sec. 5.2.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs                                   # noqa: E402

# The cell of the test matrix the manuscript reports: the range-domain
# test (the axis the instrument actually resolves) over the wide gate,
# with the specular directivity of each facet normal included.
GATE = "wide"
SPECULAR = True

# Candidate order used by t4_range.py: the true seed first, then the
# alternatives. Needed only to find which column of the score vector
# belongs to the sweep's own tessellation.
CANDIDATE_SEEDS = [11, 17, 23, 41] + list(range(100, 140))

# Rows in plotting order, top to bottom: grouped by tessellation seed so
# that the four independent microstructures are visible as blocks, and
# within a block by descending score. The descriptions are the physical
# condition of each sweep, taken from its config.json.
ROWS = [
    ("girdle_perp_ppw8", "girdle, in plane"),
    ("singlemax_ppw8", "single maximum, ppw 8"),
    ("girdle_par", "girdle, vertical"),
    ("rigid_seed11", "single maximum, ppw 6"),
    ("kappa8_seed17", "single maximum"),
    ("oos_seed23", "out of sample"),
    ("iso_gcal", "uniform control"),
]

JITTER_ROWS = 0.20          # vertical spread given to the 43 alternatives
JITTER_SEED = 0             # fixed, so the figure is reproducible


def cache_dir():
    """Directory holding t4_range.npy.

    The identification results are analysis products rather than
    repository content, so they sit beside the two repositories. Both
    locations that have ever held them are tried, relative to this file,
    so nothing here depends on where the tree is checked out.
    """
    here = Path(__file__).resolve()
    candidates = (here.parents[4] / "pulse-echo-analysis-scratch",
                  here.parents[1] / "analysis" / "facet_model")
    for d in candidates:
        if (d / "t4_range.npy").exists():
            return d
    raise FileNotFoundError(
        "t4_range.npy is in none of %s; regenerate it with "
        "analysis/facet_model/t4_range.py" % [str(d) for d in candidates])


# ----------------------------------------------------------------- load

def load_rows():
    """One record per sweep: own score, alternatives, rank, seed."""
    table = np.load(cache_dir() / "t4_range.npy", allow_pickle=True)
    cells = {row[0]: row for row in table
             if row[2] == GATE and bool(row[3]) == SPECULAR}
    out = []
    for name, label in ROWS:
        sweep, own_seed, _, _, r_own, rank, _, _, _, scores = cells[name]
        scores = np.asarray(scores, float)
        own_col = CANDIDATE_SEEDS.index(int(own_seed))
        # The own candidate has to come out of the null population, not
        # merely be plotted on top of it, or the strip would contain the
        # very point it is the reference for.
        alternatives = np.delete(scores, own_col)
        assert np.isclose(scores[own_col], r_own)
        out.append(dict(sweep=sweep, label=label, seed=int(own_seed),
                        r_own=float(r_own), rank=int(rank),
                        alternatives=alternatives))
    return out


# -------------------------------------------------------------- compute

def identification_probability(records):
    """Probability of the observed result if the score carried nothing.

    The seven sweeps are not seven independent trials: four of them are
    different acquisitions of the same seed-11 microstructure, and a
    tessellation that is identifiable once is identifiable four times.
    The unit of replication is therefore the tessellation. A tessellation
    counts as identified when every sweep carrying it ranks its own
    microstructure first out of the 44 candidates, which under the null
    has probability 1/44 per tessellation.
    """
    n_candidates = len(records[0]["alternatives"]) + 1
    by_seed = {}
    for rec in records:
        by_seed.setdefault(rec["seed"], []).append(rec["rank"] == 1)
    first = [all(v) for v in by_seed.values()]
    p_single = 1.0 / n_candidates
    p_joint = float(stats.binom.sf(sum(first) - 1, len(first), p_single))
    return sum(first), len(first), n_candidates, p_joint


# ----------------------------------------------------------------- draw

def mantissa_exponent(p):
    """p as mathtext, 4.6 x 10^-5 rather than 4.6e-05."""
    exponent = int(np.floor(np.log10(p)))
    return r"%.1f \times 10^{%d}" % (p / 10.0 ** exponent, exponent)


def draw(records, n_first, n_tess, p_joint):
    fig, ax = fs.figure("identify", left=33.0, right=2.5, bottom=8.5,
                        top=5.5)
    rng = np.random.default_rng(JITTER_SEED)
    # x in axes fractions, y in row numbers: the label columns are placed
    # relative to the axes and the rows are data.
    edge = ax.get_yaxis_transform()

    ax.axvline(0.0, color=fs.GREY, linewidth=fs.LW["annotation"],
               linestyle=(0, (1, 2)), zorder=1)
    for row, rec in enumerate(records):
        alt = rec["alternatives"]
        ax.plot([alt.min(), alt.max()], [row, row], color=fs.GREY_LIGHT,
                linewidth=fs.LW["annotation"], solid_capstyle="butt",
                zorder=2)
        ax.plot(alt, row + rng.uniform(-JITTER_ROWS, JITTER_ROWS, alt.size),
                linestyle="none", marker="o", markersize=1.9,
                markerfacecolor="none", markeredgecolor=fs.GREY,
                markeredgewidth=0.4, zorder=3)
        # Filled when the own tessellation wins, open when it does not.
        # The one sweep that fails has to be distinguishable in mono and
        # to a colourblind reader, so it cannot be marked by colour.
        identified = rec["rank"] == 1
        ax.plot([rec["r_own"]], [row], linestyle="none", marker="D",
                markersize=3.4, markeredgewidth=0.6,
                markeredgecolor=fs.BLACK,
                markerfacecolor=fs.BLACK if identified else "white",
                zorder=4)
        ax.text(0.985, row, "%d" % rec["rank"], transform=edge,
                fontsize=fs.FS["annotation"], ha="right", va="center")

    ax.set_ylim(len(records) - 0.4, -1.35)
    ax.set_yticks(range(len(records)))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.055, 0.30)
    ax.set_xticks([0.0, 0.1, 0.2])
    ax.set_xlabel("correlation with the predicted echo train")

    for row, rec in enumerate(records):
        ax.text(-0.118, row, rec["label"], transform=edge,
                fontsize=fs.FS["condition"], ha="right", va="center")
        ax.text(-0.022, row, "%d" % rec["seed"], transform=edge,
                fontsize=fs.FS["condition"], ha="right", va="center")
    # The descriptor column needs no heading; the other two do, and only
    # the seed column would otherwise be a bare set of integers.
    for x_frac, head in ((-0.022, "tessellation"), (0.985, "rank")):
        ax.text(x_frac, -1.50, head, transform=edge,
                fontsize=fs.FS["annotation"], ha="right", va="bottom",
                style="italic", color=fs.GREY)

    # One arrow apiece, on the top row only, on the two heights that fit
    # between the frame and the first row.
    top = records[0]
    fs.arrow_label(ax, "%d alternatives" % len(top["alternatives"]),
                   xy=(float(np.median(top["alternatives"])), -0.30),
                   xytext=(-0.052, -1.10), ha="left", va="center")
    fs.arrow_label(ax, "own tessellation", xy=(top["r_own"], -0.16),
                   xytext=(top["r_own"] - 0.010, -0.64), ha="right",
                   va="center")

    # The empty quarter of the panel, under the three weakest sweeps.
    ax.text(0.93, 5.5, "%d of %d tessellations\n$p = %s$"
            % (n_first, n_tess, mantissa_exponent(p_joint)),
            transform=edge, fontsize=fs.FS["annotation"], color=fs.GREY,
            ha="right", va="center", linespacing=1.5)
    return fig


# ----------------------------------------------------------------- main

def main():
    records = load_rows()
    n_first, n_tess, n_candidates, p_joint = identification_probability(
        records)

    print("range-domain identification, %s gate, specular predictor" % GATE)
    print("  %-22s %5s %8s %6s %10s" % ("sweep", "tess.", "r_own", "rank",
                                        "alt. max"))
    for rec in records:
        print("  %-22s %5d %+8.4f %5d/%-3d %+9.4f"
              % (rec["sweep"], rec["seed"], rec["r_own"], rec["rank"],
                 n_candidates, rec["alternatives"].max()))
    print("  %d of %d sweeps rank first"
          % (sum(r["rank"] == 1 for r in records), len(records)))
    print("  %d of %d independent tessellations rank first, p = %.3g"
          % (n_first, n_tess, p_joint))

    fig = draw(records, n_first, n_tess, p_joint)
    fs.save(fig, "identify",
            expect=("correlation with the predicted echo train",
                    "tessellation", "rank", "own tessellation",
                    "uniform control", "alternatives"))


if __name__ == "__main__":
    main()
