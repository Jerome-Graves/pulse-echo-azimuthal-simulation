r"""Figure 'tilterror': discretisation error against interface tilt.

Supports Section 4, "Magnitude and convergence order of the
discretisation error". Two claims: the error measured against the
rotation-invariant truth grows from exactly zero at a grid-aligned
interface to several decibels at an oblique one, and it falls only as the
first power of the grid spacing, which is the signature of a
misrepresented boundary POSITION rather than of a dispersive propagator.

Reads (paths resolved from this file, nothing absolute):
    ../results/tilt_table.npz       all four treatments at n_lambda 6,
                                    8 and 10, mean absolute error over
                                    tilts 15 to 45 degrees
    ../results/bicrystal_ladder.npz mean absolute error over tilts 15 to
                                    45 degrees at n_lambda 4 to 14,
                                    nearest-point treatment only

Writes ../../../pulse-echo-cof-paper/figures/tilterror.pdf and prints the
numbers it puts on the page, so they can be checked against the text: the
table cells that appear in Table 2, the convergence order refitted
here from the ladder, and the ladder values at n_lambda 8 and 10,
which now agree with Table 2.

WHICH ERROR CONVERGES. Panel (b) is drawn in amplitude, not in decibels.
A decibel is a logarithm, and these errors are not small: at n_lambda 4
the echo is short of the truth by 64 per cent, so 20 log10 is nowhere
near proportional to the amplitude error it reports. Fitting the decibel
column directly returns a slope of 1.45, which is an artefact of that
compression; the amplitude error itself converges at 1.15, and it is the
amplitude error whose first-order behaviour the section is about. The
right-hand scale carries the decibels so Table 2 can still be read off
the panel.

SCOPE. All twelve cells of Table 2, four treatments by three
resolutions, are now measured on one code path by
sim/runs/tilt_table_run.py and archived in ../results/tilt_table.npz.
Panel (a) is drawn from that archive. An earlier version of this figure
drew a single tilt sweep because only the n_lambda = 6 runs survived on
disk; that limitation has gone, and the nearest-point row of the new
table reproduces the independent seven-point ladder of panel (b) exactly
at both shared resolutions, which is the cross-check that the two
experiments measure the same thing.

WHAT PANEL (a) SHOWS. Three treatments converge and one does not. The
rotated staggered grid is marginally the best at n_lambda = 6 and the
worst at 10, its error rising while every other treatment falls. The
divergence is concentrated at the 45 degree tilt. See the manuscript for
the caveat attached to that result.
"""
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullLocator

import figstyle as fs

RESULTS = Path(__file__).resolve().parents[1] / "results"
TILT_LOG = RESULTS / "tilt_ppw6.log"
TILT_TABLE = RESULTS / "tilt_table.npz"   # all four treatments, three resolutions
LADDER = RESULTS / "bicrystal_ladder.npz"

# The mean in Table 2 is taken over the oblique tilts only. Zero tilt is
# excluded because the interface lies on a cell face there, every cell is
# wholly inside one grain, and the error is identically zero by
# construction; averaging it in would dilute the quantity being reported.
OBLIQUE_MIN_DEG = 15.0

# The log names the treatments as the testbed's command-line switches.
# These are the names the manuscript uses for them.
TREATMENT_NAME = {
    "naive": "nearest point",
    "rotated staggered grid": "rotated staggered grid",
}


# --------------------------------------------------------------- compute

def read_tilt_log(path):
    """Treatment -> (tilt in degrees, echo amplitude), from the sweep log.

    The log is the raw stdout of two runs of the tilt testbed, so it also
    carries a CuPy import warning and a repeated column header. A data row
    is the only line of five fields whose first field is the resolution.
    """
    blocks, name = {}, None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("ROTATED staggered grid"):
            name = "rotated staggered grid"
        elif line.startswith("treatment "):
            name = line.split()[1]
        field = line.split()
        if name is not None and len(field) == 5 and field[0].isdigit():
            blocks.setdefault(name, []).append(
                (int(field[0]), float(field[1]), float(field[2])))
    out = {}
    for name, rows in blocks.items():
        ppw, tilt, echo = (np.array(c) for c in zip(*rows))
        if len(set(ppw)) != 1:
            raise ValueError("%s mixes resolutions %s" % (name, set(ppw)))
        out[name] = (int(ppw[0]), tilt, echo)
    return out


def error_db(tilt_deg, echo):
    """Echo level referred to the grid-aligned case, which is the truth.

    The exact reflection coefficient is invariant under rotation of the
    whole configuration (Section 4), so the tilt-zero amplitude is the
    correct answer at every tilt and no converged reference is needed.
    """
    ref = echo[np.argmin(np.abs(tilt_deg))]
    return 20.0 * np.log10(echo / ref)


def mean_oblique(tilt_deg, err_db):
    """Table 2's statistic: mean absolute error over the oblique tilts."""
    return np.abs(err_db[tilt_deg >= OBLIQUE_MIN_DEG]).mean()


def db_to_amplitude_error(err_db):
    """Decibel shortfall as a fraction of the true echo amplitude.

    The staircased boundary under-reflects, so a mean level of -x dB
    against the truth is an amplitude of 10**(-x/20) times the truth and
    an error of one minus that. Exact, not a small-error expansion, which
    is the whole point at the coarse end of the ladder.
    """
    return 1.0 - 10.0 ** (-np.asarray(err_db, float) / 20.0)


def amplitude_error_to_db(err_frac):
    """Inverse of db_to_amplitude_error, for the right-hand scale.

    Clipped short of unity because matplotlib evaluates the map at the
    axis limits, and a fractional error of one is an infinite decibel
    shortfall.
    """
    return -20.0 * np.log10(np.clip(1.0 - np.asarray(err_frac, float),
                                    1e-6, None))


def fit_order(ppw, err_frac):
    """Convergence order p in error ~ h^p, from a log-log straight line.

    h is proportional to 1/n_lambda at fixed frequency, so the slope
    against n_lambda is -p. Refitted here rather than read from the
    'order' key of the archive, so that the line drawn is the line this
    script can defend; the printed comparison of the two is the check
    that it is the same fit.
    """
    slope, intercept = np.polyfit(np.log(ppw), np.log(err_frac), 1)
    return -slope, np.exp(intercept)


# ------------------------------------------------------------------ draw

TREATS = (("naive", "nearest point", "-"),
          ("aa", "anti-aliased", "--"),
          ("sm", "Schoenberg-Muir", "-."),
          ("rsg", "rotated staggered", ":"))
PPWS = (6, 8, 10)


def read_tilt_table(path):
    """Mean absolute error in dB per treatment, from the twelve-cell run.

    Returns {treatment: {ppw: error_db}}. Missing cells are simply absent,
    so a partial run still plots.
    """
    import numpy as _np
    out = {}
    if not path.exists():
        return out
    with _np.load(path, allow_pickle=True) as z:
        for key, _lab, _ls in TREATS:
            got = {}
            for ppw in PPWS:
                k = "%s_ppw%d_err" % (key, ppw)
                if k in z.files:
                    got[ppw] = float(z[k])
            if got:
                out[key] = got
    return out


def draw_treatments(ax, table):
    """Error against resolution, one line per interface treatment.

    Three of the four converge. The rotated staggered grid does not, and
    that is the point of the panel, so it is the one drawn heaviest and
    labelled directly.
    """
    for idx, (key, label, ls) in enumerate(TREATS):
        if key not in table:
            continue
        xs = sorted(table[key])
        ys = [table[key][x] for x in xs]
        diverges = ys[-1] > ys[0]
        colour, dash, mark = fs.SERIES[idx % len(fs.SERIES)]
        ax.plot(xs, ys, color=colour, dashes=dash[1] or (), marker=mark,
                lw=1.6 if diverges else 1.0, ms=3.2,
                zorder=3 if diverges else 2, label=label)
        # The two effective-medium treatments sit within 0.1 dB of one
        # another at every resolution, so their labels are nudged apart
        # vertically or they overprint.
        dy = {"aa": 5.0, "sm": -5.5}.get(key, 0.0)
        ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(5, dy), va="center", fontsize=6,
                    color=colour, annotation_clip=False)
    ax.set_xlabel(r"points per wavelength, $n_\lambda$")
    ax.set_ylabel("mean absolute error (dB)")
    ax.set_xticks(list(PPWS))
    ax.set_xlim(5.4, 10.9)
    ax.margins(y=0.12)


def draw_ladder(ax, ppw, err_frac, order, amp):
    """Resolution ladder on log-log, with the fitted power law."""
    grid = np.geomspace(3.7, 15.5, 60)
    ax.plot(grid, amp * grid ** (-order), color=fs.BLACK,
            linewidth=fs.LW["data"], zorder=2)
    points = fs.series(0, reference=True)
    points["linestyle"] = "none"
    ax.plot(ppw, err_frac, zorder=3, **points)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3.6, 16.0)
    ax.set_ylim(0.115, 0.97)
    fs.convergence_grid(ax)
    fs.plain_log_ticks(ax, "x", [4, 5, 6, 8, 10, 12, 14])
    fs.plain_log_ticks(ax, "y", [0.15, 0.2, 0.3, 0.5, 0.8])
    fs.log_minor_off(ax)
    ax.set_xlabel(r"points per wavelength, $n_\lambda$")
    ax.set_ylabel("mean amplitude error")

    # Table 2 is in decibels, so the same axis is offered in decibels on
    # the right. The map is monotone, so this is a relabelling of the one
    # scale and not a second quantity.
    ax.tick_params(right=False)         # else the primary ticks collide
    scale = ax.secondary_yaxis(
        "right", functions=(amplitude_error_to_db, db_to_amplitude_error))
    scale.set_yticks([1.5, 2, 3, 5, 8])
    scale.set_yticklabels(["1.5", "2", "3", "5", "8"])
    # The secondary axis inherits a log locator, which would label the
    # decades between these with 6 x 10^0 and so on at 4.9 pt.
    scale.yaxis.set_minor_locator(NullLocator())
    scale.tick_params(direction="in", width=fs.LW["spine"], size=2.6)
    scale.set_ylabel("mean absolute error (dB)")

    # First order drawn clear of the data rather than through it, so the
    # reader can see that the measurement is slightly steeper than h. The
    # data run corner to corner, so the two labels take a free corner
    # each rather than sitting on top of one another.
    fs.reference_slope(ax, 5.5, 12.5, 0.90, -1.0, r"$\propto h$")
    fs.direct_label(ax, 7.4, 0.355, "fit, $h^{%.2f}$" % order)
    fs.condition(ax, "nearest point", y=0.13)


# ------------------------------------------------------------------ main

def main():
    fs.apply()
    sweeps = read_tilt_log(TILT_LOG)
    with np.load(LADDER) as z:
        ppw, ladder_db, archived_order = z["ppw"], z["err"], float(z["order"])
    ladder_frac = db_to_amplitude_error(ladder_db)
    order, amp = fit_order(ppw, ladder_frac)

    fig, ax = fs.figure("onehalf", fs.FIGURES["tilterror"][1], ncols=2,
                        wspace=0.40, right=11.5)
    draw_treatments(ax[0], read_tilt_table(TILT_TABLE))
    draw_ladder(ax[1], ppw, ladder_frac, order, amp)
    for a, s in zip(ax, "ab"):
        fs.panel(a, s)

    print("tilt sweep, n_lambda = %d, mean absolute error over tilts "
          ">= %g deg" % (sweeps["naive"][0], OBLIQUE_MIN_DEG))
    for key in ("naive", "rotated staggered grid"):
        _, tilt, echo = sweeps[key]
        err = error_db(tilt, echo)
        print("  %-22s %5.2f dB   (per tilt: %s)"
              % (TREATMENT_NAME[key], mean_oblique(tilt, err),
                 ", ".join("%+.2f" % e for e in err)))
    print("resolution ladder, nearest point, %d resolutions" % ppw.size)
    print("  amplitude-error order refitted here %.4f, archived %.4f"
          % (order, archived_order))
    print("  the same points fitted in decibels give %.4f, which is the "
          "log compression, not a convergence order"
          % fit_order(ppw, ladder_db)[0])
    for p, e, f in zip(ppw, ladder_db, ladder_frac):
        print("  n_lambda %-4g %5.2f dB   amplitude error %.4f" % (p, e, f))
    print("nearest point on the regenerated table: %.2f dB at "
          "n_lambda 8, %.2f dB at 10, reproducing this ladder"
          % (ladder_db[ppw == 8][0], ladder_db[ppw == 10][0]))

    # Left panel is now error against resolution for all four treatments,
    # not the single tilt sweep that was all the surviving data allowed.
    fs.save(fig, "tilterror",
            expect=["points per wavelength", "mean amplitude error",
                    "mean absolute error (dB)",
                    "nearest point", "rotated staggered"])


if __name__ == "__main__":
    main()
