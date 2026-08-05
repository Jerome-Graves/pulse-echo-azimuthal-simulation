r"""Figure 'ladder': identification switches on with acoustic contrast.

The strongest control in the paper, and the one that answers the obvious
objection to Sec. 5.1. A reader who is told that a facet model predicts a
recorded echo train, and that the correct microstructure can be picked
out of a line-up, will ask whether the score is reading the scattering or
reading something about the way the specimen is rendered onto the grid.

The contrast ladder answers it by construction. Five sweeps share ONE
tessellation, one rasterisation, one staircased rim and one source. Each
grain's stiffness is interpolated a fraction f of the way from the
orientation-mean tensor toward its own, so f = 0 is a homogeneous
specimen carrying the girdle's mean anisotropy and f = 1 is the physical
specimen. Nothing but the acoustic contrast across the boundaries
changes. Every geometric artefact a candidate could match is present in
full at f = 0.

At f = 0 the specimen's own tessellation ranks 7 of 48, inside the body
of the null. It reaches rank 1 by f = 0.25 and stays first or second
through f = 1. A sixth point is plotted apart from the ladder: the
uniform-orientation specimen, in which every grain carries the SAME
c-axis, so there is no contrast anywhere and the disc keeps only its
single-crystal anisotropy. It ranks 26 of 48, which is the middle of the
null.

Reads (no simulation, no GPU):
    ../analysis/tessellation_replication.npz
        names      the 17 sweep names
        cands      the 48 candidate seeds
        ident_own  (17, 6) own score, rank, p, z, runner-up, its score

Writes <paper>/figures/ladder.pdf and prints the rungs.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs                                       # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "analysis" \
    / "tessellation_replication.npz"

# (sweep name, contrast fraction). The f = 1 rung is the production
# girdle sweep itself; the ladder was run only for f < 1 because that
# sweep already exists and rerunning it would differ only in azimuth
# count, which is decimated to the common grid here.
LADDER = [("girdle_seed11_ppw8_contrast_f000", 0.00), ("girdle_seed11_ppw8_contrast_f025", 0.25),
          ("girdle_seed11_ppw8_contrast_f050", 0.50), ("girdle_seed11_ppw8_contrast_f075", 0.75),
          ("girdle_seed11_ppw8_dev", 1.00)]
CONTROL = "girdle_seed11_ppw8_uniform_axis"
OWN, RANK = 0, 1


# ────────────────────────────────── load ──────────────────────────────
def load():
    with np.load(DATA, allow_pickle=True) as z:
        names = [str(x) for x in z["names"]]
        return names, len(z["cands"]), np.asarray(z["ident_own"])


# ───────────────────────────────── compute ────────────────────────────
def rungs(names, own):
    out = []
    for nm, f in LADDER:
        i = names.index(nm)
        out.append((f, int(own[i, RANK]), float(own[i, OWN])))
    i = names.index(CONTROL)
    return out, (int(own[i, RANK]), float(own[i, OWN]))


# ─────────────────────────────────── draw ─────────────────────────────
def draw(ax, data, ctrl, n_cand):
    f = np.array([d[0] for d in data])
    rk = np.array([d[1] for d in data], float)

    # The null band: under a uniform rank the middle half of the
    # distribution runs from the lower to the upper quartile.
    ax.axhspan(n_cand * 0.25, n_cand * 0.75, color=fs.GREY_LIGHT,
               alpha=0.45, lw=0, zorder=1)
    ax.axhline((n_cand + 1) / 2.0, color=fs.GREY, lw=fs.LW["annotation"],
               ls=(0, (1, 2)), zorder=2)

    ax.plot(f, rk, color=fs.BLACK, lw=fs.LW["data"], zorder=4)
    ax.plot(f, rk, "o", ms=4.2, mfc=fs.BLACK, mec=fs.BLACK, zorder=5)

    # The uniform-orientation control is not a rung: it has no contrast
    # anywhere rather than a reduced contrast, so it is drawn off the
    # ladder at the left, open, to keep the two controls distinct.
    ax.plot([-0.13], [ctrl[0]], "s", ms=4.2, mfc="white", mec=fs.BLACK,
            mew=0.9, zorder=5, clip_on=False)

    ax.set_yscale("log")
    ax.set_ylim(n_cand * 1.35, 0.72)
    ax.set_yticks([1, 2, 5, 10, 24, 48])
    ax.set_yticklabels(["1", "2", "5", "10", "24", "48"])
    ax.set_xlim(-0.28, 1.08)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("acoustic contrast across the grain boundaries, $f$")
    ax.set_ylabel("rank of the correct microstructure\namong %d candidates"
                  % n_cand)


def main():
    names, n_cand, own = load()
    data, ctrl = rungs(names, own)

    fig, ax = fs.figure(width="single", height_mm=62.0,
                        left=22.0, right=4.0, bottom=11.0, top=4.0)
    draw(ax, data, ctrl, n_cand)
    fs.direct_label(ax, 0.62, 1.28, "identified", colour=fs.BLACK,
                    ha="center")
    # "chance" goes right, so the left margin is free for the control
    # label; the control sits low on the axis and its label needs the
    # space above the marker rather than below it.
    fs.direct_label(ax, 1.04, 14.0, "chance", colour=fs.GREY, ha="right")
    fs.direct_label(ax, -0.14, ctrl[0] / 1.5,
                    "uniform\norientation", colour=fs.GREY, ha="center",
                    va="bottom")

    print("contrast ladder, one tessellation, %d candidates" % n_cand)
    for f, rk, r in data:
        print("  f = %.2f   rank %2d of %d   r = %+.4f" % (f, rk, n_cand, r))
    print("  uniform orientation (no contrast anywhere): rank %d, r = %+.4f"
          % ctrl)

    fs.save(fig, "ladder",
            expect=["acoustic contrast", "rank of the correct",
                    "identified", "chance"])


if __name__ == "__main__":
    main()
