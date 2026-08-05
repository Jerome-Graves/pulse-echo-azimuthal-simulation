r"""Figure 'tofaz': time of flight against azimuth, measured and fitted.

Supports the first result of Section 5.1, that the fabric axis is
recovered from the time of flight alone, without reference to the
backscattered field. Each specimen contributes one row: the measured
diametral time of flight against azimuth, the fitted 180 degree periodic
template, and the recovered axis against the true one. The residual panel
below is there so that the reader can see how much of the measurement the
template accounts for, which for these specimens is well under half of
it.

The time of flight is extracted here, from the recorded traces, by the
criterion Section 3.2 states: the first crossing of 25 per cent of the
backwall envelope peak, interpolated between samples. The extraction is
checked against the cached observables the fit itself consumed, and the
check is printed, so nothing in the figure depends on trusting the cache.

Reads (paths resolved from this file, never absolute):
    <repo>/out/sweeps/<sweep>/az*.npz        keys 'trace', 'dt'
    <repo>/out/sweeps/<sweep>/config.json    key 'diameter_mm'
    <repo>/out/sweeps/<sweep>/obs_cache.npz  keys 'az', 'tof_us', for the
        cross-check only
    <repo>/out/sweeps/<sweep>/fit_result.json
        keys 'rots', 'meas_tof_us', 'model_tof_us', 'alpha_probe_deg',
        'truth_alpha_deg', 'alpha_err_deg', 'truth_kappa', for the fitted
        template and the two axes
for <sweep> in rigid_seed11, kappa8_seed17 and oos_seed23.

Writes:
    <GitHub>/pulse-echo-cof-paper/figures/tofaz.pdf

Prints, for each specimen, the recovered and true axis and their
difference, the measured and fitted peak-to-peak amplitude, the residual
rms, and the two consistency checks (extraction against cache, template
against exact 180 degree periodicity).
"""
import json
from pathlib import Path

import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator
from scipy.signal import hilbert

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
SWEEPS = REPO / "out" / "sweeps"

# Specimens shown, in order of increasing axis error, top row first. All
# three are single-maximum fabrics fitted by sim/model/fit_sweep.py; the
# uniform-ODF control iso_gcal is deliberately not shown here, because it
# has no axis to recover and belongs with the null tests.
SPECIMENS = ["rigid_seed11", "kappa8_seed17", "oos_seed23"]

# Uniform-ODF sweep, fitted by the same code. Reported but not drawn: it
# has no axis to recover, so a row for it would need a true-axis mark that
# does not exist. Its fitted amplitude is the number that matters, because
# it says what the template returns when there is no fabric to find.
CONTROL = "iso_gcal"

C_REF = 3850.0                  # mean qP speed (m/s), _t2_common.C_REF

# Leading-edge criterion, Section 3.2. The threshold is a fraction of the
# backwall envelope peak and the peak is sought in a window about the
# nominal two-way time 2D/c. The envelope peak itself is not used: forward
# scattering broadens and skews the arrival by an amount that varies with
# the angle between beam and fabric axis, which would read as extra
# anisotropy.
EDGE_FRACTION = 0.25
BACKWALL_WINDOW_S = 2e-6

# Half-width of the azimuthal average applied before fitting, as in
# sim/model/fit_sweep.py (AZ_SMOOTH_DEG). Azimuths are folded modulo 180 first.
# The fabric pattern is exactly 180 degree periodic but the measurement is
# not: re-rasterising the specimen at each azimuth jitters the leading
# edge between an azimuth and its partner 180 degrees away, and averaging
# the partners removes a jitter the pattern does not contain.
AZ_SMOOTH_DEG = 1.5

# Vertical room reserved above the topmost row, in units of the row step,
# for the row caption and the key to the axis lines.
HEAD_ROOM = 0.95

# The two axis lines are drawn with fixed patterns rather than each row's
# series dash, so that one key describes all three rows. The series dash
# of the third row is itself dotted, which would otherwise collide with
# the true-axis line.
TRUE_AXIS_DASH = (0, (1.0, 1.5))
RECOVERED_AXIS_DASH = (0, (3.0, 1.5))

# Inset from the left and right spines for the labels written inside the
# axes, in degrees of azimuth. Half a degree is not enough: a 7 pt glyph
# then sits on the spine.
LABEL_INSET_DEG = 6.0


# --------------------------------------------------------------- compute

def leading_edge_tof_us(trace, dt, diameter_m):
    """Time of flight to the backwall by the 25 per cent leading edge.

    Sub-sample by linear interpolation across the threshold crossing,
    which matters because one sample is 10 ns and the whole azimuthal
    signal is a few hundred nanoseconds. The envelope is taken from the
    unfiltered record: the band-pass of Section 3.2 exists to define the
    coda, and applying it here would shift the edge by the filter's own
    rise time.
    """
    fs_hz = 1.0 / dt
    envelope = np.abs(hilbert(trace))
    k_nominal = int(2.0 * diameter_m / C_REF * fs_hz)
    half = int(BACKWALL_WINDOW_S * fs_hz)
    start = max(k_nominal - half, 0)
    segment = envelope[start:k_nominal + half]

    i_peak = int(np.argmax(segment))
    threshold = EDGE_FRACTION * segment[i_peak]
    above = np.nonzero(segment[:i_peak + 1] >= threshold)[0]
    j = int(above[0]) if above.size else i_peak
    if j > 0 and segment[j] > segment[j - 1]:
        frac = (threshold - segment[j - 1]) / (segment[j] - segment[j - 1])
    else:
        frac = 0.0
    return (start + j - 1 + frac) / fs_hz * 1e6


def extract_sweep_tof(sweep_dir, azimuths_deg):
    """Leading-edge time of flight at every azimuth of one sweep."""
    diameter_m = json.loads((sweep_dir / "config.json").read_text()
                            )["diameter_mm"] * 1e-3
    tof_us = np.empty(len(azimuths_deg))
    for i, az in enumerate(azimuths_deg):
        with np.load(sweep_dir / ("az%03d.npz" % az)) as z:
            trace = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        tof_us[i] = leading_edge_tof_us(trace, dt, diameter_m)
    return tof_us


def fold_average(azimuths_deg, tof_us, half_width_deg=AZ_SMOOTH_DEG):
    """Average each azimuth with its neighbours and its 180 degree partner."""
    folded = np.asarray(azimuths_deg, float) % 180.0
    out = np.empty_like(tof_us)
    for i, a in enumerate(folded):
        gap = np.abs(folded - a)
        out[i] = tof_us[np.minimum(gap, 180.0 - gap) <= half_width_deg].mean()
    return out


def load_specimen(name):
    """Everything one row of the figure needs, plus its own checks."""
    sweep_dir = SWEEPS / name
    fit = json.loads((sweep_dir / "fit_result.json").read_text())
    azimuths_deg = np.asarray(fit["rots"], int)

    raw_tof_us = extract_sweep_tof(sweep_dir, azimuths_deg)
    tof_us = fold_average(azimuths_deg, raw_tof_us)
    model_us = np.asarray(fit["model_tof_us"], float)

    with np.load(sweep_dir / "obs_cache.npz") as z:
        cached = dict(zip(z["az"].tolist(), z["tof_us"].tolist()))
    cache_gap = max(abs(cached[a] - t) for a, t in zip(azimuths_deg,
                                                      raw_tof_us))
    fit_gap = float(np.abs(tof_us - np.asarray(fit["meas_tof_us"])).max())

    # The template is 180 degree periodic by construction; this measures
    # it on the sampled azimuths rather than asserting it.
    period_gap = float(np.abs(
        model_us - np.interp((azimuths_deg + 180) % 360, azimuths_deg,
                             model_us, period=360.0)).max())

    return dict(name=name, azimuths_deg=azimuths_deg.astype(float),
                tof_us=tof_us, model_us=model_us,
                mean_us=float(tof_us.mean()),
                residual_us=tof_us - model_us,
                alpha_deg=float(fit["alpha_probe_deg"]),
                truth_deg=float(fit["truth_alpha_deg"]),
                error_deg=float(fit["alpha_err_deg"]),
                truth_kappa=float(fit["truth_kappa"]),
                seed=int(json.loads((sweep_dir / "config.json").read_text()
                                    )["seed"]),
                cache_gap_us=cache_gap, fit_gap_us=fit_gap,
                period_gap_us=period_gap)


ROW_CLEARANCE_US = 1.0          # clear space between one row and the next


def row_step_us(specimens):
    """Vertical spacing of the rows, from the data rather than by eye.

    The widest excursion of any specimen plus ROW_CLEARANCE_US, rounded up
    to a fifth of a microsecond. Setting it from the data means the rows
    stay clear of one another, and of each other's captions, whichever
    specimens are listed in SPECIMENS.
    """
    span = max(np.ptp(s["tof_us"] - s["mean_us"]) for s in specimens)
    return float(np.ceil((span + ROW_CLEARANCE_US) * 5.0) / 5.0)


# ------------------------------------------------------------------ draw

def draw_axis_marks(ax, spec, baseline_us, half_us, colour):
    """The true and the recovered fabric axis, over this row only.

    Both are drawn at the axis and at the axis plus 180 degrees, because
    an axis is a direction without a sign and the pattern repeats.
    """
    for repeat in (0.0, 180.0):
        ax.plot([spec["truth_deg"] % 180.0 + repeat] * 2,
                [baseline_us - half_us, baseline_us + half_us],
                color=fs.GREY, linestyle=TRUE_AXIS_DASH,
                linewidth=fs.LW["annotation"], zorder=4)
        ax.plot([spec["alpha_deg"] % 180.0 + repeat] * 2,
                [baseline_us - half_us, baseline_us + half_us],
                color=colour, linestyle=RECOVERED_AXIS_DASH,
                linewidth=fs.LW["reference"], zorder=6)


def draw_rows(ax, specimens, step_us):
    """Measured and fitted time of flight, one offset row per specimen."""
    # Half-height of one row's band, that is, the step less the clearance
    # that row_step_us() built into it. The axis marks span exactly this,
    # and the row caption sits just above it.
    half_us = 0.5 * (step_us - ROW_CLEARANCE_US)
    for i, spec in enumerate(specimens):
        baseline_us = -i * step_us
        colour = fs.SERIES[i][0]

        # Measured: a thin continuous line so that every azimuth is on the
        # page, carrying open markers at the house density so the rows stay
        # separable if the plate is printed in mono.
        measured = fs.series(i, n=spec["azimuths_deg"].size, reference=True,
                             dense=True)
        measured["linewidth"] = fs.LW["spine"]
        measured["linestyle"] = (0, ())
        ax.plot(spec["azimuths_deg"],
                spec["tof_us"] - spec["mean_us"] + baseline_us,
                zorder=3, **measured)

        fitted = fs.series(i, emphasis=True)
        fitted["marker"] = "none"
        ax.plot(spec["azimuths_deg"],
                spec["model_us"] - spec["mean_us"] + baseline_us,
                zorder=5, **fitted)

        draw_axis_marks(ax, spec, baseline_us, half_us, colour)
        label_us = baseline_us + half_us + 0.05
        fs.direct_label(ax, LABEL_INSET_DEG, label_us,
                        "seed %d, $\\kappa$ = %.3g"
                        % (spec["seed"], spec["truth_kappa"]), colour=colour)
        fs.direct_label(ax, 360.0 - LABEL_INSET_DEG, label_us,
                        "axis error %.1f°" % spec["error_deg"],
                        colour=colour, ha="right")

    top_us = HEAD_ROOM * step_us
    ax.set_ylim(-(len(specimens) - 1) * step_us - 0.5 * step_us, top_us)
    ax.yaxis.set_major_locator(FixedLocator(
        [-i * step_us + d for i in range(len(specimens)) for d in (-1, 0, 1)]))
    ax.yaxis.set_major_formatter(FixedFormatter(
        ["−1", "0", "1"] * len(specimens)))
    ax.set_ylabel("time of flight anomaly (µs)")
    fs.direct_label(ax, 360.0 - LABEL_INSET_DEG, top_us - 0.1,
                    "true axis dotted, recovered dashed", colour=fs.GREY,
                    ha="right", va="top")


def draw_residuals(ax, specimens):
    """What the 180 degree periodic template leaves behind.

    Drawn without markers: three curves of unstructured residual are
    already dense, and here the dash patterns alone carry the identity
    that the marker would otherwise have to.
    """
    ax.axhline(0.0, color=fs.GREY, linewidth=fs.LW["annotation"],
               linestyle=TRUE_AXIS_DASH, zorder=1)
    limit = 0.0
    for i, spec in enumerate(specimens):
        kw = fs.series(i)
        kw["marker"] = "none"
        kw["linewidth"] = fs.LW["spine"]
        ax.plot(spec["azimuths_deg"], spec["residual_us"], zorder=2 + i, **kw)
        limit = max(limit, np.abs(spec["residual_us"]).max())
    ax.set_ylim(-1.15 * limit, 1.15 * limit)
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_ylabel("residual (µs)")
    ax.set_xlabel("azimuth (degrees)")


# --------------------------------------------------------------- reports

def report(specimens):
    """The recovery numbers of Section 5.1, and the two checks."""
    print("time of flight, leading edge at %.0f per cent of the backwall "
          "envelope peak" % (100 * EDGE_FRACTION))
    for spec in specimens:
        print("%-15s n_az %3d  mean ToF %.3f us" % (
            spec["name"], spec["azimuths_deg"].size, spec["mean_us"]))
        print("    axis recovered %6.2f deg, true %6.2f deg, error %5.2f deg"
              % (spec["alpha_deg"], spec["truth_deg"], spec["error_deg"]))
        print("    peak to peak: measured %.3f us, fitted %.3f us"
              % (np.ptp(spec["tof_us"]), np.ptp(spec["model_us"])))
        print("    residual rms %.3f us, %.0f per cent of the measured "
              "standard deviation"
              % (spec["residual_us"].std(),
                 100 * spec["residual_us"].std() / spec["tof_us"].std()))
        print("    checks: extraction vs cache %.1e us, folded vs fitted "
              "input %.1e us, template 180 deg periodicity %.1e us"
              % (spec["cache_gap_us"], spec["fit_gap_us"],
                 spec["period_gap_us"]))


def report_control():
    """What the same template returns on a specimen with no fabric."""
    fit = json.loads((SWEEPS / CONTROL / "fit_result.json").read_text())
    smallest = min(np.ptp(np.asarray(
        json.loads((SWEEPS / name / "fit_result.json").read_text()
                   )["model_tof_us"])) for name in SPECIMENS)
    print("%-15s uniform ODF control, not drawn" % CONTROL)
    print("    fitted peak to peak %.3f us, against %.3f us for the "
          "weakest fabric shown"
          % (np.ptp(np.asarray(fit["model_tof_us"])), smallest))
    print("    the axis it reports, %.1f deg, is therefore meaningless"
          % fit["alpha_probe_deg"])


# ------------------------------------------------------------------ main

def main():
    fs.apply()
    specimens = [load_specimen(name) for name in SPECIMENS]
    step_us = row_step_us(specimens)

    fig = fs.canvas("tofaz", 78.0)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.12,
                            **fs.mm_margins(fig, left=11.0, right=2.5,
                                            bottom=8.5, top=4.5))
    ax_tof = fig.add_subplot(grid[0])
    ax_res = fig.add_subplot(grid[1], sharex=ax_tof)
    ax_tof.tick_params(labelbottom=False)

    draw_rows(ax_tof, specimens, step_us)
    draw_residuals(ax_res, specimens)
    ax_res.set_xlim(0.0, 360.0)
    ax_res.set_xticks([0, 90, 180, 270, 360])

    fs.panel(ax_tof, "a")
    fs.panel(ax_res, "b")

    report(specimens)
    report_control()
    fs.save(fig, "tofaz",
            expect=["azimuth (degrees)", "time of flight anomaly (µs)",
                    "residual (µs)", "axis error"])


if __name__ == "__main__":
    main()
