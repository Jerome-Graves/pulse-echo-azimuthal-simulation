r"""Figure 'codafield': the backscattered coda predicted from geometry.

Supports the claim of Sec. 5.2 (Prediction of the backscattered field from
grain-boundary geometry) that the coda is deterministic rather than
stochastic. The measured azimuth-by-time backscatter is reproduced by a
facet model that knows only where the grain boundaries lie and which way
they face, with no parameter fitted to the measurement.

Panels (a) to (c) share one colour scale and one colourbar, so the three
fields are directly comparable; a separate scale per panel would make the
comparison meaningless. Panel (d) is the depth-registration test: the
correlation as the prediction is slid in time. A model that predicted only
how much backscatter there is, and not when it arrives, would give a flat
curve or a peak in an arbitrary place.

Reads (cached azimuth-by-time fields; no simulation, no specimen rebuild,
no GPU):
    <cache>/coda_field_girdle_seed11_ppw8_dev__gp8.npz          measured + true model
    <cache>/coda_field_girdle_seed11_ppw8_dev__wrongseed8.npz   unrelated tessellation
    <cache>/coda_field_girdle_seed11_ppw8_dev__sm8.npz          same tessellation,
                                                   different fabric
Each file carries 'az', 'tgrid', 'meas' (envelope), 'e1' (backwall
envelope peak) and the four predictor variants written by t2_predict.py.
'p_full' is Eq. (eq:facetmodel) with both the reflection weighting and the
facet directivity, which is the predictor the paper states. The measured
field is common to all three files; that is asserted, not assumed.

Writes:
    <paper>/figures/codafield.pdf

Prints the correlations and the registration peaks, which are the numbers
quoted in Sec. 5.2, so they can be checked against the text.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs                                   # noqa: E402

# Clean-band coda gate, Sec. 3. Early enough to precede the backwall
# arrival at 52 us and late enough to be clear of the source ring-down.
CODA_GATE_S = (24e-6, 36e-6)

# Longitudinal reference speed, Sec. 2. Only used to convert a time
# registration error into a range error for the upper axis of panel (d).
C_REF = 3850.0

# The prediction is slid over +-2 us, about four band-limited pulse
# lengths. The slide is cyclic within the 12 us gate, so at the extremes
# a sixth of the field has wrapped; that is harmless here because the
# curve is read near its peak, but it is why the window stops at 2 us.
LAG_HALF_US = 2.0

# The predicted field spans many decades, so it is compared in dB. This
# floor is the one used by the scoring pipeline (shift_null_2d.EPS) and
# is far below any populated bin.
EPS = 1e-30

# Fraction of the measured field left outside the colour scale. The three
# panels must share limits, and the predicted fields have longer tails
# than the measurement, so the limits are set from the measurement and
# the predictions are clipped to them. 95 rather than 100 because the
# measured field is speckled: its extreme samples are single bright
# scatterers that would otherwise set the scale for everything.
CLIP_PERCENTILE = 95.0

SWEEP = "girdle_seed11_ppw8_dev"          # girdle fabric, seed 11, ppw 8
TAG_TRUE = "gp8"                    # the specimen that was simulated
TAG_WRONG = "wrongseed8"            # seed 23, same construction rules
TAG_FABRIC = "sm8"                  # seed 11 again, single-maximum fabric


def cache_dir():
    """Directory holding the cached azimuth-by-time fields.

    The caches are analysis products rather than repository content, so
    they sit beside the two repositories rather than inside either. Both
    locations that have ever held them are tried, relative to this file,
    so nothing here depends on where the tree is checked out.
    """
    here = Path(__file__).resolve()
    candidates = (here.parents[4] / "pulse-echo-analysis-scratch",
                  here.parents[1] / "analysis" / "facet_model")
    for d in candidates:
        if (d / ("coda_field_%s__%s.npz" % (SWEEP, TAG_TRUE))).exists():
            return d
    raise FileNotFoundError(
        "no t2p_*.npz cache in any of %s; regenerate with "
        "analysis/facet_model t2_build.py then t2_predict.py"
        % [str(d) for d in candidates])


# ----------------------------------------------------------------- load

def load_field(tag):
    """Measured and predicted azimuth-by-time power over the coda gate.

    The measurement is a band-passed envelope and the model predicts
    power, so the envelope is squared. Dividing by the backwall peak e1
    first makes the level independent of the source amplitude.
    """
    with np.load(cache_dir() / ("coda_field_%s__%s.npz" % (SWEEP, tag))) as z:
        t_s = z["tgrid"]
        gate = (t_s >= CODA_GATE_S[0]) & (t_s < CODA_GATE_S[1])
        meas = (z["meas"][:, gate] / float(z["e1"])) ** 2
        pred = z["p_full"][:, gate]
        az_deg = z["az"].astype(float)
    return az_deg, t_s[gate], meas, pred


def load_all():
    """The measured field once, and one prediction per tessellation."""
    az_deg, t_s, meas, pred_true = load_field(TAG_TRUE)
    preds = {TAG_TRUE: pred_true}
    for tag in (TAG_WRONG, TAG_FABRIC):
        az_b, t_b, meas_b, pred = load_field(tag)
        # Every prediction is scored against the SAME measurement. If
        # that ever stopped being true the comparison would be between
        # two different experiments, not two different models.
        assert np.array_equal(az_b, az_deg) and np.allclose(t_b, t_s)
        assert np.array_equal(meas_b, meas)
        preds[tag] = pred
    return az_deg, t_s, meas, preds


# -------------------------------------------------------------- compute

def to_db(power):
    return 10.0 * np.log10(power + EPS)


# Number of azimuthal harmonics projected out. The fabric is 180 degree
# periodic, so k = 2 carries most of it, but the whole low-order family is
# removed because any model of the bulk fabric could generate them.
N_HARMONIC = 4


def harmonic_centre(field):
    """Strip the mean, the low-order azimuthal harmonics and the level.

    Section 5.2 claims the coda is predicted by grain-boundary GEOMETRY.
    Anything a model of the bulk fabric could produce must therefore be
    removed before correlating, or the claim is not tested. Removing the
    column means strips the mean decay with depth, which any model with a
    beam and a gate gets right for free. Removing harmonics k = 1 to
    N_HARMONIC at every time sample strips the fabric's own periodicity.
    Removing the row means strips each azimuth's overall loudness, which
    is the scalar result reported separately.

    This returns a smaller correlation than a plainer centring and a more
    significant one, because the null it is scored against is roughly half
    as wide. See analysis/centring_convention.py.
    """
    n_az = field.shape[0]
    j = np.arange(n_az)
    basis = [np.ones(n_az)]
    for k in range(1, N_HARMONIC + 1):
        basis.append(np.cos(2 * np.pi * k * j / n_az))
        basis.append(np.sin(2 * np.pi * k * j / n_az))
    design = np.column_stack(basis)
    fitted = design @ np.linalg.lstsq(design, field, rcond=None)[0]
    without_harmonics = field - fitted
    return without_harmonics - without_harmonics.mean(axis=1, keepdims=True)


def correlation(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def lag_curve(meas_db_centred, pred, lags):
    """Correlation against an imposed depth-registration error.

    The prediction is rolled in time and re-centred at every lag, because
    the centring is part of the statistic and not a display choice.
    """
    return np.array([correlation(
        meas_db_centred, harmonic_centre(to_db(np.roll(pred, k, axis=1))))
        for k in lags])


def column_centre(field):
    """Remove the depth profile only.

    Kept alongside double_centre because the manuscript quotes the
    column-centred correlation. On this sweep the two agree to 0.002, so
    the choice is not carrying the result; printing both says so.
    """
    return field - field.mean(axis=0, keepdims=True)


def compute(meas, preds, dt_s):
    """Centred fields, correlations and registration curves."""
    meas_c = harmonic_centre(to_db(meas))
    meas_col = column_centre(to_db(meas))
    fields = {k: harmonic_centre(to_db(v)) for k, v in preds.items()}
    r = {k: correlation(meas_c, v) for k, v in fields.items()}
    r_col = {k: correlation(meas_col, column_centre(to_db(v)))
             for k, v in preds.items()}
    n_lag = int(round(LAG_HALF_US * 1e-6 / dt_s))
    lags = np.arange(-n_lag, n_lag + 1)
    curves = {k: lag_curve(meas_c, v, lags) for k, v in preds.items()}
    return meas_c, fields, r, r_col, lags * dt_s * 1e6, curves


# ----------------------------------------------------------------- draw

def draw(az_deg, t_s, meas_c, fields, r, lag_us, curves):
    fig = fs.canvas("codafield")
    grid = fig.add_gridspec(
        2, 4, width_ratios=[1.0, 1.0, 1.0, 0.05], height_ratios=[1.18, 1.0],
        wspace=0.21, hspace=0.56,
        **fs.mm_margins(fig, left=11.0, right=13.5, bottom=8.5, top=3.0))

    image_axes = [fig.add_subplot(grid[0, j]) for j in range(3)]
    cbar_ax = fig.add_subplot(grid[0, 3])
    lag_ax = fig.add_subplot(grid[1, 0:3])

    limit = float(np.percentile(np.abs(meas_c), CLIP_PERCENTILE))
    extent = [t_s[0] * 1e6, t_s[-1] * 1e6, az_deg[0], az_deg[-1]]
    panels = [(meas_c, "measured"),
              (fields[TAG_TRUE], "true tessellation"),
              (fields[TAG_WRONG], "wrong tessellation")]
    for ax, (field, name) in zip(image_axes, panels):
        # 'nearest' because the azimuth axis really is sampled at 6 deg;
        # interpolating it would invent structure in the direction the
        # figure is asking the reader to compare.
        im = ax.imshow(field, aspect="auto", origin="lower", extent=extent,
                       cmap=fs.cmap(fs.CMAP_SEQ), vmin=-limit, vmax=limit,
                       interpolation="nearest")
        ax.set_xlabel("time (µs)")
        ax.set_xticks([24, 28, 32, 36])
        ax.set_yticks([0, 90, 180, 270, 360])
        # x=0.17 clears the inside panel letter, which the house style
        # puts in this same corner.
        fs.condition(ax, name, x=0.17, box=True)
    for ax in image_axes[1:]:
        ax.set_yticklabels([])
    image_axes[0].set_ylabel("azimuth (degrees)")

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.outline.set_linewidth(fs.LW["colourbar"])
    cbar.ax.tick_params(width=fs.LW["spine"], size=2.0,
                        labelsize=fs.FS["cbar_tick"])
    cbar.set_label("centred backscattered power (dB)",
                   fontsize=fs.FS["cbar_label"])

    # Series order is also the order of importance: the two predictions
    # that register sit above the one that does not, and the null curve
    # takes the lightest of the three in greyscale.
    curve_style = [
        (TAG_TRUE, 0, "true tessellation", True),
        (TAG_FABRIC, 1, "same grains, other fabric", False),
        (TAG_WRONG, 2, "wrong tessellation", False)]
    lag_ax.axhline(0.0, color=fs.GREY, linewidth=fs.LW["annotation"],
                   zorder=1)
    lag_ax.axvline(0.0, color=fs.GREY, linewidth=fs.LW["annotation"],
                   linestyle=(0, (1, 2)), zorder=1)
    for tag, i, name, emph in curve_style:
        lag_ax.plot(lag_us, curves[tag], label=name, zorder=3,
                    **fs.series(i, n=len(lag_us), emphasis=emph))
    lag_ax.set_xlim(-LAG_HALF_US, LAG_HALF_US)
    lag_ax.set_ylim(-0.13, 0.36)
    lag_ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    lag_ax.set_xlabel("depth-registration error (µs)")
    lag_ax.set_ylabel("correlation")

    # A legend, not direct labels: all three curves converge at both ends
    # of the abscissa and separate only over the central microsecond,
    # where there is no room to write three names.
    fs.legend(lag_ax, loc="upper left", borderaxespad=0.5)

    peak_us = lag_us[int(np.argmax(curves[TAG_TRUE]))]
    fs.arrow_label(lag_ax, "$r = %+.3f$" % r[TAG_TRUE],
                   xy=(peak_us, max(curves[TAG_TRUE])),
                   xytext=(0.62, 0.30), va="center")

    # The upper axis is the same abscissa read as a range error. One
    # microsecond of two-way error is c/2 = 1.9 mm, which is one
    # wavelength at 2 MHz, so the peak width is stated in the units the
    # reader thinks about resolution in.
    range_ax = lag_ax.secondary_xaxis(
        "top", functions=(lambda t: t * C_REF * 1e-3 / 2.0,
                          lambda s: s * 2.0e3 / C_REF))
    range_ax.set_xlabel("range error (mm)")
    range_ax.tick_params(width=fs.LW["spine"])

    for ax, letter in zip(image_axes, "abc"):
        fs.panel(ax, letter, inside=True)
    fs.panel(lag_ax, "d", dx=-0.038)
    return fig


# ----------------------------------------------------------------- main

def main():
    az_deg, t_s, meas, preds = load_all()
    dt_s = float(t_s[1] - t_s[0])
    meas_c, fields, r, r_col, lag_us, curves = compute(meas, preds, dt_s)

    print("%s, %d azimuths, gate %.0f-%.0f us, %d samples at %.0f ns"
          % (SWEEP, len(az_deg), CODA_GATE_S[0] * 1e6, CODA_GATE_S[1] * 1e6,
             len(t_s), dt_s * 1e9))
    print("  %-26s %8s %8s %22s" % ("predictor", "r", "r (col)",
                                    "registration peak"))
    for tag, name in ((TAG_TRUE, "true tessellation"),
                      (TAG_FABRIC, "same grains, other fabric"),
                      (TAG_WRONG, "wrong tessellation")):
        peak = int(np.argmax(curves[tag]))
        print("  %-26s %+8.4f %+8.4f       %+.4f at %+.2f us"
              % (name, r[tag], r_col[tag], curves[tag][peak], lag_us[peak]))
    print("  colour scale +-%.1f dB (%.0fth percentile of the measured "
          "field)" % (float(np.percentile(np.abs(meas_c), CLIP_PERCENTILE)),
                      CLIP_PERCENTILE))

    fig = draw(az_deg, t_s, meas_c, fields, r, lag_us, curves)
    fs.save(fig, "codafield",
            expect=("azimuth (degrees)", "correlation", "range error (mm)",
                    "measured", "true tessellation", "wrong tessellation"))


if __name__ == "__main__":
    main()
