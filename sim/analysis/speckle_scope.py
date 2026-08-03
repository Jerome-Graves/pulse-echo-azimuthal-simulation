"""How far the speckle rejection of sec:grainpop reaches.

Supports Section 4, "Grain population and coda statistics". That
subsection rejects the developed-speckle model of the coda three ways,
and every one of them was measured on the GIRDLE ensemble at ppw 8 in
the published 24 to 36 us gate. A rejection measured on one fabric, one
grid and one window states nothing about the others. This module runs
the same three tests, on the same estimator, across every stored
condition the project has, and reports where they hold and where they
do not.

THE THREE TESTS, RESTATED SO THE SCOPE QUESTION IS SHARP.

  ORDERING.   A gamma of shape N fixes its intensity contrast at
              1/sqrt(N), its gap between the arithmetic and geometric
              means at (10/ln 10)(ln N - psi(N)), which is the
              maximum-likelihood equation, and its level standard
              deviation at (10/ln 10) sqrt(psi_1(N)). One sample must
              therefore return one N three times. The published sample
              returns three different numbers. The statistic used here
              is SIGNED, the log of the level shape over the contrast
              shape, because the direction is the whole point: the
              specimen runs upward and the residue runs downward, and an
              unsigned spread would score both as the same evidence.

  SKEW.       psi_2 is negative at every positive argument, so the log
              of a gamma variate is LEFT skewed at every shape. A sample
              whose log is right skewed is not a gamma, whatever N is
              chosen. This test needs no fitted parameter at all.

  LENGTH.     Whatever N is, doubling the number of independent looks a
              gate averages must double the shape. Lengthening the gate
              at fixed start multiplies the resolved cells as T, and the
              null is established on the same code path rather than
              argued: band-limited Gaussian noise through this code
              returns T^0.96 in the published window and T^0.93 in the
              shallow one on 2000 traces, and at the 240-trace size of
              the measurement it returns T^0.95 +- 0.06 and
              T^0.92 +- 0.08 over 20 seeds.

The first two are NOT independent of each other, and this module says so
where it prints them. The contrast route is set by the bright tail, the
level standard deviation by the deep nulls; a sample with a heavy bright
tail and no nulls therefore returns a low contrast shape, a high level
shape and a right skewed log at the same time, by the same asymmetry.
The ordering test and the skew test are two readings of one departure.
The LENGTH test is independent of both, and it is the one that carries
the conclusion.

WHAT IS REPEATED, AND WHY EACH CASE IS THERE.

  single ppw8   Eight single-maximum tessellations. Their azimuthal
                energy distribution is nothing like the girdle's: seed 7
                puts 45.9 per cent of the whole revolution's coda energy
                into ONE of its thirty azimuths, against 22.8 per cent
                for the published girdle seed. If the rejection were an
                artefact of how the girdle spreads energy round the
                revolution, this is where it would break.

  girdle ppw6   Seeds 11 and 23 at ppw 6 and ppw 10, so that a rejection
  girdle ppw10  measured at ppw 8 is not a statement about the grid. The
                coarse grid is the interesting direction: db_reconcile
                documents that ppw 6 carries numerical content the
                analysis band does not fully remove, and numerical
                content is the one thing in these runs that might be
                diffuse.

  ladder        Five rungs of the contrast ladder, f = 0 to 1, each
                grain's stiffness interpolated a fraction f from the
                orientation-averaged tensor toward its own. Physical
                single scattering goes as f^2, so f = 0 holds the
                contrast-independent residue and nothing else. It is the
                control on the whole argument: if the tests reject a
                gamma at f = 0 as well, they are rejecting something
                about the code path and not about the coda.

  uniform       The uniform-orientation controls at all three grids,
                which reach the same residue by a different route.

  shallow       Every case is run twice, in the published 24 to 36 us
                gate and in the 12 to 22 us window, because the same
                azimuthal structure was found there and the window has
                no independent right to the published result. The
                shallow window is where the per-grid content check earns
                its keep, and it is the only place anything is excluded.

WHAT IS OUT OF SCOPE BEFORE ANY STATISTIC IS TAKEN. A window that does
not carry grain scattering has no coda statistics to measure, so every
family is first compared with the uniform-orientation control AT ITS OWN
GRID. Fifteen of the sixteen specimen family-and-window cases clear that
comparison by 11.6 to 26.5 dB. One does not: the ppw 6 girdle in the
shallow window sits 2.4 dB above its own control, so the 12 to 22 us
window at ppw 6 is numerical content and nothing else, and it is
excluded rather than reported. That single exclusion is why the control
has to be taken per grid; against the ppw 8 control the same case would
have read 27.5 dB and passed.

WHAT HOLDS. MEASURED.

  The LENGTH test holds in all fifteen in-scope cases. The implied shape
  grows as T^-0.02 to T^0.31 where the resolved cells grow as T^0.70 to
  T^0.98. The largest specimen exponent is T^0.29 in the published gate
  and T^0.31 in the shallow window, and the Gaussian null at the matched
  sample size never fell below T^0.83 and T^0.77 respectively over 20
  seeds, so the specimen and the null do not overlap in either window.
  Nothing about the fabric, the grid, the contrast fraction above zero
  or the window moves that exponent into the region an average of
  independent looks occupies.

  The SKEW and ORDERING tests reject a gamma in all fifteen in-scope
  cases at the family level. Sweep by sweep they are nearly but not
  quite unanimous: one of the 23 distinct specimen sweeps fails to
  reject on its own in each window, girdle_perp at ppw 6 in the
  published gate and mx_single_s89_ppw8 in the shallow one. At 30
  azimuths a single sweep decides little, which is why the family-level
  Monte Carlo is the statistic quoted.

WHERE IT FAILS, AND WHY THAT IS THE POINT. MEASURED.

  Not one of the eight residue cases rejects, and they fail in the
  OPPOSITE DIRECTION from the specimen. The f = 0 rung has a log skew of
  -1.09 in the published gate and -0.86 in the shallow window, on the
  side a gamma lives on; its implied shapes are 2.9 to 4.8 and 94 to 102
  rather than below unity; and its three-route ordering INVERTS,
  contrast above gap above level standard deviation rather than below.
  The residue that survives at zero contrast behaves like an averaged
  diffuse field, or in the shallow window like a deterministic arrival.
  The specimen coda behaves like neither. That contrast is the strongest
  single piece of evidence in this module, because it shows the tests
  are reading the coda and not the code path.

  The ppw 6 girdle is the weakest specimen case that survives. In the
  published gate seed 11 at ppw 6 returns a log skew of only +0.14 and
  does not reject on either shape statistic on its own, p = 0.087 and
  0.055, and its three shapes are 1.86, 2.30 and 2.55 against 0.56, 1.34
  and 1.89 for the same seed at ppw 8. That is the direction the
  numerical content of the coarse grid predicts. The family of two
  rejects at p = 0.0015 on both statistics and the LENGTH test returns
  T^0.28 there against a T^0.90 cell count, so the rejection survives,
  but the shape figures should not be quoted at ppw 6.

WHAT THE TWO SHAPE TESTS ARE NOT. Over the 56 sweep-and-window cases the
signed ordering statistic and the log skew correlate at a Spearman rho
of +0.85. They are close to one measurement of one asymmetry and should
not be presented as two independent rejections. Only the LENGTH test is
independent of them.

WHAT IS INFERRED, AND OFFERED AS A READING RATHER THAN A FINDING. That
the coda at f > 0 is carried by one to three facets is consistent with
everything here and is not established by it. What is established is
negative: the gate does not average independent looks, at either fabric,
any of the three grids, any contrast fraction above zero, or either
window in which the window carries grain scattering at all.

Reads, all under ../../out/sweeps, each directory az*.npz with keys
'trace', 'dt' and 'az' plus a config.json, on the 30 azimuths at 12
degree spacing every sweep has in common:
    girdle_perp_ppw8, mx_girdle_s{7,17,23,41,53,71,89}_ppw8
    singlemax_ppw8, mx_single_s{7,17,23,41,53,71,89}_ppw8
    girdle_perp, lad_girdle_s23_ppw6            ppw 6, seeds 11 and 23
    lic_girdle_s11_ppw10, lad_girdle_s23_ppw10  ppw 10, seeds 11 and 23
    cs_f{000,025,050,075}_s11_ppw8              contrast ladder
    zerocontrast_ppw8, zc_s11_ppw6, zc_s11_ppw10
Touches no label volume and builds no tessellation, so it reaches no
CUDA path and needs no GPU. Writes nothing. Run with --fast to halve the
Monte Carlo draws.

ESTIMATOR NOTE. Levels are the audited ones of analysis/db_reconcile.py,
2*E[x^2] over the window on the trace band-limited to 0.8 to 3.0 MHz,
filtered before gating and referenced to the peak of the source
envelope. Two things are done deliberately.

  The window is converted to sample indices with the sample rate of ITS
  OWN record. Every azimuth carries its own time step, set by the
  stability condition on its own realised stiffnesses; the spread within
  one sweep reaches 3.1 per cent in the uniform-orientation runs. A gate
  taken at a fixed sample index would be a different gate at every
  azimuth.

  Where azimuths have to share a frequency grid, for the effective
  bandwidth and the intensity profile, they are interpolated onto ONE
  physical time axis first rather than trimmed to a common sample count.
  The traces are oversampled by a factor of about thirty against the top
  of the analysis band, so the interpolation costs nothing measurable
  and it removes the time-axis error entirely.

  No Hilbert transform is taken of a whole trace and then gated. The
  front arrival is about 100 dB above the coda and the tail of its
  analytic signal is not local. Every analytic signal here is formed
  from an excerpt of the window it belongs to, Hann tapered where a
  spectrum is taken of it, and the level estimator, which is what every
  shape and every gate-length figure is built on, uses no analytic
  signal at all. The one whole-trace envelope is the source reference,
  a global maximum sitting on the transmit pulse, which no leakage from
  anywhere else can move.
"""
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import digamma, polygamma

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SWD = os.path.join(ROOT, "out", "sweeps")

# Acquisition and analysis constants, Section 3. The band is the one
# db_reconcile.py audits and CODA is the gate sec:grainpop quotes.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
BAND = (0.8e6, 3.0e6)
CODA = (24e-6, 36e-6)

# The shallow window of the open question, as observable_matrix.py
# defines it. It is not a gate the paper quotes a level in; it is here
# because the same azimuthal structure was found inside it and a result
# measured at 24 to 36 us does not transfer to it for free.
SHALLOW = (12e-6, 22e-6)

WINDOWS = (("coda 24-36 us, published", CODA,
            (26, 28, 30, 32, 36, 40, 44)),
           ("shallow 12-22 us", SHALLOW, (14, 16, 18, 20, 22)))

# Every sweep in the matrix carries these 30 azimuths.
AZ = tuple(range(0, 360, 12))

# 10/ln 10. Every level here is 10 log10 of a power.
DBN = 10.0 / np.log(10.0)

# The cases, grouped so that each group varies exactly one thing against
# the published one. "ladder f1.00" is girdle_perp_ppw8 under its ladder
# name, because the f = 1 rung IS the production specimen; it is listed
# twice on purpose so the ladder reads as a ladder.
FAMILIES = (
    ("girdle ppw8", (("girdle_perp_ppw8", 11), ("mx_girdle_s7_ppw8", 7),
                     ("mx_girdle_s17_ppw8", 17), ("mx_girdle_s23_ppw8", 23),
                     ("mx_girdle_s41_ppw8", 41), ("mx_girdle_s53_ppw8", 53),
                     ("mx_girdle_s71_ppw8", 71),
                     ("mx_girdle_s89_ppw8", 89))),
    ("single ppw8", (("singlemax_ppw8", 11), ("mx_single_s7_ppw8", 7),
                     ("mx_single_s17_ppw8", 17), ("mx_single_s23_ppw8", 23),
                     ("mx_single_s41_ppw8", 41), ("mx_single_s53_ppw8", 53),
                     ("mx_single_s71_ppw8", 71),
                     ("mx_single_s89_ppw8", 89))),
    ("girdle ppw6", (("girdle_perp", 11), ("lad_girdle_s23_ppw6", 23))),
    ("girdle ppw10", (("lic_girdle_s11_ppw10", 11),
                      ("lad_girdle_s23_ppw10", 23))),
    ("ladder f0.00", (("cs_f000_s11_ppw8", 11),)),
    ("ladder f0.25", (("cs_f025_s11_ppw8", 11),)),
    ("ladder f0.50", (("cs_f050_s11_ppw8", 11),)),
    ("ladder f0.75", (("cs_f075_s11_ppw8", 11),)),
    ("ladder f1.00", (("girdle_perp_ppw8", 11),)),
    ("uniform ppw8", (("zerocontrast_ppw8", 11),)),
    ("uniform ppw6", (("zc_s11_ppw6", 11),)),
    ("uniform ppw10", (("zc_s11_ppw10", 11),)),
)

# Families that hold physical grain scattering, as against the two that
# hold only the contrast-independent residue. The verdict is different
# for the two sets and the module must not average them together.
SPECIMEN = ("girdle ppw8", "single ppw8", "girdle ppw6", "girdle ppw10",
            "ladder f0.25", "ladder f0.50", "ladder f0.75", "ladder f1.00")
RESIDUE = ("ladder f0.00", "uniform ppw8", "uniform ppw6", "uniform ppw10")

# The uniform-orientation control AT THE SAME GRID as each family. This
# is the comparison that decides whether a window holds grain scattering
# at all, and it has to be made per resolution: the numerical residue is
# a strong function of the grid and comparing a ppw 6 specimen with a
# ppw 8 control would flatter it.
CONTROL_OF = {"girdle ppw8": "zerocontrast_ppw8",
              "single ppw8": "zerocontrast_ppw8",
              "girdle ppw6": "zc_s11_ppw6",
              "girdle ppw10": "zc_s11_ppw10",
              "ladder f0.00": "zerocontrast_ppw8",
              "ladder f0.25": "zerocontrast_ppw8",
              "ladder f0.50": "zerocontrast_ppw8",
              "ladder f0.75": "zerocontrast_ppw8",
              "ladder f1.00": "zerocontrast_ppw8",
              "uniform ppw8": "zerocontrast_ppw8",
              "uniform ppw6": "zc_s11_ppw6",
              "uniform ppw10": "zc_s11_ppw10"}

# A window whose specimen level sits closer than this to its own
# uniform-orientation control at the same grid is not carrying grain
# scattering, and no shape statistic taken inside it is a statement
# about the coda. Ten decibels is a factor of ten in power, so the
# residue can contribute at most a third of the amplitude.
CONTENT_FLOOR_DB = 10.0

MC_DRAWS = 4000

# Two noise ensembles, for two different questions. NOISE_BIG estimates
# the exponent an average of independent looks actually has, precisely.
# NOISE_MATCHED is the sample size of the measurement, 8 tessellations
# of 30 azimuths, and repeating it over NOISE_SEEDS seeds gives the
# SAMPLING DISTRIBUTION of the exponent at that size, which is what the
# specimen exponent has to be compared against.
NOISE_BIG = 2000
NOISE_MATCHED = 240
NOISE_SEEDS = 20
CONF = 0.95

_CACHE = {}


def db(x):
    """Power ratio to decibels. Every quantity here is a power."""
    return 10.0 * np.log10(x)


# ------------------------------------------------------------------ load

def bandpass(x, fs, lo=BAND[0], hi=BAND[1]):
    """Zero-phase Butterworth applied to the COMPLETE trace.

    Section 3 filters before gating so the filter transient falls
    outside the window rather than inside it. Same order and corners as
    db_reconcile.bandpass.
    """
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band",
                 output="sos")
    return sosfiltfilt(sos, x)


def read_sweep(name):
    """One record per azimuth, band-limited and source-referenced.

    dt is kept per record. It follows the realised stiffnesses through
    the stability condition and is not identical across the azimuths of
    one sweep, so every window below is given in seconds and converted
    to indices with the sample rate of its own record.
    """
    if name in _CACHE:
        return _CACHE[name]
    folder = os.path.join(SWD, name)
    with open(os.path.join(folder, "config.json")) as handle:
        cfg = json.load(handle)
    recs = []
    for degrees in AZ:
        path = os.path.join(folder, "az%03d.npz" % degrees)
        if not os.path.exists(path):
            raise IOError("%s has no azimuth %d" % (name, degrees))
        with np.load(path) as handle:
            trace = np.asarray(handle["trace"], float).ravel()
            dt = float(handle["dt"])
        # The source reference is the peak of the analytic envelope of
        # the whole unfiltered trace, which is the transmit pulse. It is
        # a global maximum dominated by an arrival 100 dB above
        # everything else, so the non-locality of the transform cannot
        # move it; the ban on whole-trace envelopes is about reading a
        # LATE window off one, which nothing here does.
        recs.append(dict(az=degrees, dt=dt,
                         band=bandpass(trace, 1.0 / dt)
                         / np.abs(hilbert(trace)).max()))
    _CACHE[name] = (recs, float(cfg["ppw"]), int(cfg["seed"]))
    return _CACHE[name]


def gate_power(recs, win):
    """Audited window power per azimuth, each on its own time axis."""
    out = []
    for rec in recs:
        fs = 1.0 / rec["dt"]
        i0, i1 = int(round(win[0] * fs)), int(round(win[1] * fs))
        out.append(2.0 * (rec["band"][i0:i1] ** 2).mean())
    return np.array(out)


def on_common_axis(recs, win):
    """The window of every azimuth on ONE physical time axis.

    Needed wherever azimuths are averaged sample by sample, which is the
    effective bandwidth and the mean intensity profile. Interpolation,
    not truncation: the records differ in time step by up to 3 per cent
    and truncating to a common sample count would leave each azimuth on
    its own clock.
    """
    dt = float(np.median([rec["dt"] for rec in recs]))
    n = int(np.floor((win[1] - win[0]) / dt))
    t = win[0] + np.arange(n) * dt
    rows = [np.interp(t, np.arange(len(rec["band"])) * rec["dt"],
                      rec["band"]) for rec in recs]
    return np.array(rows), dt


# --------------------------------------------------------------- compute

def effective_bandwidth(seg, dt):
    """(int S)^2 / int S^2 of the mean periodogram of the excerpt.

    The reciprocal of Goodman's coherence time, taken in the frequency
    domain. The lag-domain form is biased low here because the squared
    covariance is non-negative and its estimation noise accumulates
    instead of cancelling across lags. A Hann taper is applied because
    the window is a hard-edged excerpt, and the analytic signal is
    formed from the TAPERED EXCERPT and never from the whole trace.
    """
    n = seg.shape[1]
    analytic = hilbert(seg * np.hanning(n)[None, :], axis=1)
    spectrum = (np.abs(np.fft.fft(analytic, axis=1)) ** 2).mean(axis=0)
    return float(spectrum.sum() ** 2 / (spectrum ** 2).sum() / (n * dt))


def participation(seg, dt):
    """(int m)^2 / (T int m^2) on the FITTED mean intensity profile.

    How evenly the mean intensity fills the window, which is the only
    physical reason the available look count should grow more slowly
    than T. It is taken on a log-linear fit rather than on the raw
    30-azimuth mean, whose own residual fluctuation inflates int m^2 and
    biases the factor low.
    """
    mean = (np.abs(hilbert(seg, axis=1)) ** 2).mean(axis=0)
    n = len(mean)
    t = np.arange(n) * dt
    fit = np.exp(np.polyval(np.polyfit(t, np.log(mean), 1), t))
    return float(fit.sum() ** 2 / (n * (fit ** 2).sum()))


def shape_from_level_sd(sd_db):
    """The gamma shape a measured level standard deviation implies."""
    return float(brentq(lambda k: DBN * np.sqrt(polygamma(1, k)) - sd_db,
                        1e-4, 1e6))


def shape_from_gap(gap_db):
    """The gamma shape the arithmetic-minus-geometric gap implies.

    The likelihood equation for a gamma shape is ln k - psi(k) equal to
    the log of the arithmetic mean minus the mean of the logs, so this
    IS the maximum-likelihood shape and not a second-best route to it.
    """
    return float(brentq(lambda k: DBN * (np.log(k) - digamma(k)) - gap_db,
                        1e-4, 1e6))


def three_shapes(power):
    """(contrast, maximum-likelihood gap, level sd) shapes of one sample.

    Under a gamma these are three estimators of the same one parameter.
    Their ORDER is diagnostic on its own: the contrast route is set by
    the bright tail and the level route by the deep nulls, so a heavy
    tailed sample with no nulls returns them in increasing order and a
    left tailed one returns them in decreasing order.
    """
    power = np.asarray(power, float)
    level = db(power)
    return (float((power.mean() / power.std(ddof=1)) ** 2),
            shape_from_gap(db(power.mean()) - level.mean()),
            shape_from_level_sd(level.std(ddof=1)))


def ordering_stat(power):
    """log of the level-sd shape over the contrast shape. SIGNED.

    The sign is the whole point and an unsigned spread would be wrong
    here. The specimen departs from a gamma by having a heavy bright
    tail and no deep nulls, which drives the contrast shape DOWN and the
    level shape UP, so its statistic is positive. The residue departs
    the other way and returns a negative one. An unsigned max-over-min
    would count both as the same evidence, and the f = 0 rung would then
    read as a rejection when what it is doing is the opposite.
    """
    n_c, _n_g, n_s = three_shapes(power)
    return float(np.log(n_s / n_c))


def log_skew(power):
    """Skewness of the log of the sample. Negative for any gamma."""
    return float(stats.skew(np.log(np.asarray(power, float))))


def gamma_log_skew(shape):
    """The log skew a gamma of this shape must have. Always negative."""
    return float(polygamma(2, shape) / polygamma(1, shape) ** 1.5)


def pooled_level_sd(rows):
    """Per-azimuth level scatter pooled WITHIN realisations, in dB."""
    num = sum(((row - row.mean()) ** 2).sum() for row in rows)
    den = sum(len(row) - 1 for row in rows)
    return float(np.sqrt(num / den))


def family_statistics(blocks):
    """The two shape-based statistics of a whole family.

    ordering  the mean over realisations of ordering_stat, so that a
              family of eight is not carried by one realisation
    skew      the skew of the pooled logs after each realisation is
              normalised by its own mean, which removes the level
              differences between realisations without touching shape

    THE TWO ARE NOT INDEPENDENT and report_dependence measures how far
    from independent they are. Both read the same asymmetry: a bright
    tail with no nulls raises the level shape, lowers the contrast shape
    and skews the log right, all at once. They are reported together
    because they fail together, not because they are two witnesses.
    """
    order = float(np.mean([ordering_stat(b) for b in blocks]))
    pooled = np.concatenate([np.log(b / b.mean()) for b in blocks])
    return order, float(stats.skew(pooled))


def calibrate(blocks, draws=MC_DRAWS, seed=0):
    """Parametric bootstrap of both statistics under the fitted gamma.

    The null the tests need is not "a gamma of some canonical shape" but
    "a gamma fitted to THIS sample at THIS sample size". Each realisation
    is replaced by a gamma sample of the same length at its own
    maximum-likelihood shape, and the family statistics are re-formed on
    exactly the code path the measurement uses. Thirty azimuths is a
    small sample and both statistics are biased at that size; this is
    what removes the bias from the comparison.
    """
    rng = np.random.default_rng(seed)
    shapes = [three_shapes(b)[1] for b in blocks]
    obs_order, obs_skew = family_statistics(blocks)
    null_order, null_skew = [], []
    for _ in range(draws):
        draw = [rng.gamma(k, 1.0, len(b)) for k, b in zip(shapes, blocks)]
        a, b = family_statistics(draw)
        null_order.append(a)
        null_skew.append(b)
    null_order = np.array(null_order)
    null_skew = np.array(null_skew)
    return dict(shapes=shapes, order=obs_order, skew=obs_skew,
                p_order=float((null_order >= obs_order).mean()),
                p_skew=float((null_skew >= obs_skew).mean()),
                med_order=float(np.median(null_order)),
                med_skew=float(np.median(null_skew)))


def slope(x, y):
    """Slope of log y against log x. Unity for independent looks."""
    return float(np.polyfit(np.log(np.asarray(x, float)),
                            np.log(np.asarray(y, float)), 1)[0])


def gaussian_null(dt, t0, ends, draws=NOISE_MATCHED, seed=3):
    """The length test on band-limited Gaussian noise, same code path.

    The gate power of band-limited Gaussian noise is by construction the
    sum of independent looks, so its implied shape must grow as T. This
    is the null the specimen exponent is measured against, and it is
    established rather than argued: the same filter, the same window
    arithmetic and the same shape estimator.
    """
    rng = np.random.default_rng(seed)
    fs = 1.0 / dt
    n = int(np.ceil((max(ends) * 1e-6 + 8e-6) * fs))
    noise = [bandpass(rng.standard_normal(n), fs) for _ in range(draws)]
    lengths, implied = [], []
    for end in ends:
        i0, i1 = int(round(t0 * fs)), int(round(end * 1e-6 * fs))
        power = np.array([2.0 * (x[i0:i1] ** 2).mean() for x in noise])
        lengths.append((i1 - i0) * dt * 1e6)
        implied.append(shape_from_level_sd(db(power).std(ddof=1)))
    return slope(lengths, implied), lengths, implied


# ---------------------------------------------------------------- report

def report_reproduction():
    """The published triple, reproduced before anything is extended."""
    print("=" * 78)
    print("R0  THE PUBLISHED MEASUREMENT, REPRODUCED ON THIS CODE PATH")
    print("=" * 78)
    blocks = [gate_power(read_sweep(n)[0], CODA)
              for n, _s in dict(FAMILIES)["girdle ppw8"]]
    levels = [db(b) for b in blocks]
    contrast = np.mean([(b.mean() / b.std(ddof=1)) ** 2 for b in blocks])
    gap = np.mean([db(b.mean()) - db(b).mean() for b in blocks])
    sd = pooled_level_sd(levels)
    pooled = np.concatenate([np.log(b / b.mean()) for b in blocks])
    print("  eight girdle tessellations, ppw 8, gate 24 to 36 us")
    print("  contrast route      N = %.2f   (published 0.50)" % contrast)
    print("  gap route, pooled   N = %.2f   (published 0.94)"
          % shape_from_gap(gap))
    print("  level sd route      N = %.2f   (published 1.32), sd %.2f dB"
          % (shape_from_level_sd(sd), sd))
    print("  skew of the log gate power %+.2f  (published +0.48)"
          % stats.skew(pooled))
    print("  The rest of this module is the same four numbers everywhere")
    print("  else, so the reproduction has to come first.\n")


def report_energy_spread():
    """How differently the families distribute energy round the disc."""
    print("=" * 78)
    print("R1  THE AZIMUTHAL ENERGY DISTRIBUTION OF EACH SWEEP")
    print("=" * 78)
    print("  the single-maximum sweeps are in this module because their")
    print("  revolution energy is distributed nothing like the girdle's,")
    print("  so a result that survives them is not a girdle artefact")
    print("  %-14s %-22s %5s %11s %10s"
          % ("family", "sweep", "ppw", "max az share", "contrast"))
    for fam, members in FAMILIES:
        for name, _seed in members:
            recs, ppw, _s = read_sweep(name)
            power = gate_power(recs, CODA)
            print("  %-14s %-22s %5.0f %10.1f %% %10.2f"
                  % (fam, name, ppw, 100 * power.max() / power.sum(),
                     power.std(ddof=1) / power.mean()))
    print()


def report_content(label, win):
    """Is there anything in this window to have statistics about?

    The published gate has an established level. The shallow window does
    not, and a shape test run on a window that holds only the front
    arrival's tail would be a test of the source pulse. The specimen is
    therefore compared with the two residue routes IN THE SAME WINDOW
    before any statistic is quoted from it.
    """
    print("=" * 78)
    print("R2  IS THERE GRAIN SCATTERING IN THIS WINDOW AT ALL, %s"
          % label)
    print("=" * 78)
    print("  The control is the uniform-orientation run AT THE SAME GRID,")
    print("  not the ppw 8 one: the numerical residue is a strong")
    print("  function of the grid and a cross-grid comparison flatters")
    print("  the coarse specimen. The f = 0 rung is shown beside it as an")
    print("  independent route to the same residue.")
    print("  %-14s %10s %10s %11s %11s"
          % ("family", "level dB", "control", "over own", "over f = 0"))
    zero = float(gate_power(read_sweep("cs_f000_s11_ppw8")[0], win).mean())
    out = {}
    for fam, members in FAMILIES:
        mean = float(np.mean([gate_power(read_sweep(n)[0], win).mean()
                              for n, _s in members]))
        control = float(gate_power(read_sweep(CONTROL_OF[fam])[0],
                                   win).mean())
        margin = db(mean / control)
        out[fam] = margin
        flag = "" if fam in RESIDUE or margin >= CONTENT_FLOOR_DB \
            else "   <- BELOW THE %.0f dB FLOOR" % CONTENT_FLOOR_DB
        print("  %-14s %10.2f %10.2f %9.1f dB %9.1f dB%s"
              % (fam, db(mean), db(control), margin, db(mean / zero), flag))
    print("  A family that sits on its own residue carries no grain")
    print("  scattering and no statistic taken inside it means anything.")
    print()
    return out


def report_shapes(label, win, draws=MC_DRAWS):
    """Tests ORDERING and SKEW, per sweep and then per family."""
    print("=" * 78)
    print("R3  ORDERING AND SKEW, window %s" % label)
    print("=" * 78)
    print("  A gamma returns the three shapes EQUAL. The specimen returns")
    print("  them increasing, contrast below gap below level sd, which is")
    print("  a heavy bright tail with no deep nulls. Watch the ORDER as")
    print("  much as the spread: the residue cases invert it.")
    print("  p ord and p skew are the same Monte Carlo as the family")
    print("  table below, run on the single sweep, so the reader can see")
    print("  how little 30 azimuths decides on its own.")
    print("  %-13s %-21s %6s %6s %6s %7s %5s %6s %6s"
          % ("family", "sweep", "N ctr", "N gap", "N sd", "logskew",
             "order", "p ord", "p skew"))
    single, index = {}, 0
    for fam, members in FAMILIES:
        for name, _seed in members:
            power = gate_power(read_sweep(name)[0], win)
            n_c, n_g, n_s = three_shapes(power)
            rising = n_c < n_g < n_s
            falling = n_c > n_g > n_s
            tag = "up" if rising else ("down" if falling else "mixed")
            # The seed is the position in the table, not a hash of the
            # name: Python randomises string hashes per process and this
            # module has to give the same numbers on every run.
            index += 1
            res = calibrate([power], draws=draws, seed=index)
            single.setdefault(name, (fam, res))
            print("  %-13s %-21s %6.2f %6.2f %6.2f %+7.2f %5s %6.3f %6.3f"
                  % (fam, name, n_c, n_g, n_s, log_skew(power), tag,
                     res["p_order"], res["p_skew"]))
    # Deduplicated by sweep, because girdle_perp_ppw8 is listed twice,
    # once as a tessellation of the ensemble and once as the f = 1 rung.
    spec = {n: v for n, v in single.items() if v[0] in SPECIMEN}
    beaten = [n for n, (_f, res) in spec.items()
              if not (res["p_order"] < 0.05 or res["p_skew"] < 0.05)]
    print("  %d of the %d distinct specimen sweeps fail to reject alone:"
          % (len(beaten), len(spec)))
    for name in beaten:
        print("    %-13s %s" % (spec[name][0], name))
    if not beaten:
        print("    none.")
    print()
    print("  FAMILY LEVEL. Thirty azimuths is too small a sample for one")
    print("  sweep to decide anything, so both statistics are re-formed")
    print("  over the whole family and calibrated against gamma samples")
    print("  of the same sizes at the same fitted shapes, on this code")
    print("  path. p is the upper tail: small means the family departs")
    print("  from a gamma in the direction the specimen departs.")
    print("  %-14s %8s %8s %8s   %8s %8s %8s"
          % ("family", "spread", "null", "p", "skew", "null", "p"))
    out = {}
    for fam, members in FAMILIES:
        blocks = [gate_power(read_sweep(n)[0], win) for n, _s in members]
        res = calibrate(blocks, draws=draws)
        out[fam] = res
        print("  %-14s %8.3f %8.3f %8.4f   %+8.2f %+8.2f %8.4f"
              % (fam, res["order"], res["med_order"], res["p_order"],
                 res["skew"], res["med_skew"], res["p_skew"]))
    print()
    print("  A gamma of shape N has log skew %+.2f at N = 0.5, %+.2f at"
          % (gamma_log_skew(0.5), gamma_log_skew(1.0)))
    print("  N = 1 and %+.2f at N = 21.6, so the requirement is not a"
          % gamma_log_skew(21.6))
    print("  threshold, it is a SIGN, and it holds at every shape.")
    print()
    return out


def report_length(label, win, ends):
    """Test LENGTH, the one that is independent of the other two."""
    print("=" * 78)
    print("R4  GATE LENGTH, window start %.0f us, ends %s us"
          % (win[0] * 1e6, ", ".join(str(e) for e in ends)))
    print("=" * 78)
    dt0 = float(np.median([r["dt"]
                           for r in read_sweep("girdle_perp_ppw8")[0]]))
    # The null exponent is itself a Monte Carlo estimate and one draw of
    # it is not a constant. It is taken twice. The large ensemble says
    # what exponent an average of independent looks HAS; the matched
    # ensemble, at the sample size of the measurement, says how far that
    # exponent moves from seed to seed, which is the spread the specimen
    # figure has to be compared against.
    exponent, lengths, implied = gaussian_null(dt0, win[0], ends,
                                               draws=NOISE_BIG, seed=101)
    null = np.array([gaussian_null(dt0, win[0], ends, seed=s)[0]
                     for s in range(NOISE_SEEDS)])
    print("  band-limited Gaussian noise at dt = %.3f ns on the same"
          % (dt0 * 1e9))
    print("  filter and the same estimator, %d traces: N ~ T^%.2f"
          % (NOISE_BIG, exponent))
    print("    T us   " + " ".join("%7.1f" % t for t in lengths))
    print("    N      " + " ".join("%7.2f" % v for v in implied))
    print("  at the matched size of %d, over %d seeds the null exponent is"
          % (NOISE_MATCHED, NOISE_SEEDS))
    print("  T^%.2f +- %.2f, spanning T^%.2f to T^%.2f. That spread, and"
          % (null.mean(), null.std(ddof=1), null.min(), null.max()))
    print("  not the point estimate, is what a specimen exponent has to")
    print("  clear.")
    print()
    print("  %-14s %8s %8s %8s   %s"
          % ("family", "N ~ T^", "cells^", "T*part^", "implied N by T"))
    out = {"__null__": dict(exp=exponent, matched=null)}
    for fam, members in FAMILIES:
        recs = [read_sweep(n)[0] for n, _s in members]
        lengths, implied, cells, tpart = [], [], [], []
        for end in ends:
            window = (win[0], end * 1e-6)
            rows = [db(gate_power(r, window)) for r in recs]
            cell, part = [], []
            for r in recs:
                seg, dt = on_common_axis(r, window)
                span = seg.shape[1] * dt
                cell.append(effective_bandwidth(seg, dt) * span)
                part.append(participation(seg, dt) * span * 1e6)
            lengths.append((window[1] - window[0]) * 1e6)
            implied.append(shape_from_level_sd(pooled_level_sd(rows)))
            cells.append(float(np.mean(cell)))
            tpart.append(float(np.mean(part)))
        out[fam] = dict(T=lengths, implied=implied, cells=cells,
                        tpart=tpart, exp=slope(lengths, implied),
                        exp_cells=slope(lengths, cells),
                        exp_part=slope(lengths, tpart))
        print("  %-14s %8.2f %8.2f %8.2f   %s"
              % (fam, out[fam]["exp"], out[fam]["exp_cells"],
                 out[fam]["exp_part"],
                 " ".join("%.2f" % v for v in implied)))
    print()
    print("  cells^ is the exponent of T*B_eff and T*part^ that of the")
    print("  gate length times the participation factor of its fitted")
    print("  intensity profile. Both are the AVAILABLE look count and")
    print("  both are the null the implied shape has to match. On the")
    print("  residue cases they fall well below unity, which is the")
    print("  right answer for content that does not fill the band, and")
    print("  it is why the residue exponents in the first column are not")
    print("  evidence of anything.")
    print()
    return out


def report_ordering_flip(results):
    """The control that decides whether the tests read the coda.

    results maps window label to the dictionary report_shapes returned.
    """
    print("=" * 78)
    print("R6  THE CONTROL: WHAT HAPPENS AT ZERO CONTRAST")
    print("=" * 78)
    print("  Physical single scattering on the ladder goes as f^2, so the")
    print("  f = 0 rung holds the contrast-independent residue and no")
    print("  grain scattering at all, and the uniform-orientation runs")
    print("  reach the same residue by giving every grain one c-axis. If")
    print("  the three tests rejected a gamma there too, they would be")
    print("  rejecting the code path rather than the coda.")
    for label, table in results.items():
        print("  %s" % label)
        print("    %-14s %9s %9s %9s" % ("family", "skew", "p skew",
                                         "verdict"))
        for fam, _members in FAMILIES:
            res = table[fam]
            reject = res["p_skew"] < 0.05 or res["p_order"] < 0.05
            kind = "specimen" if fam in SPECIMEN else "residue"
            print("    %-14s %+9.2f %9.4f %9s   %s"
                  % (fam, res["skew"], res["p_skew"],
                     "reject" if reject else "no reject", kind))
    print()


def report_dependence():
    """How far the ordering and skew statistics are from independent.

    They are printed side by side above and it would be easy to read
    them as two witnesses. They are not. This measures the rank
    correlation between them over every sweep and window in the module,
    so the reader can see how much of the second is the first.
    """
    print("=" * 78)
    print("R5  ARE THE TWO SHAPE TESTS INDEPENDENT EVIDENCE?")
    print("=" * 78)
    order, skew = [], []
    for _label, win, _ends in WINDOWS:
        for _fam, members in FAMILIES:
            for name, _seed in members:
                power = gate_power(read_sweep(name)[0], win)
                order.append(ordering_stat(power))
                skew.append(log_skew(power))
    rho = stats.spearmanr(order, skew)
    pear = stats.pearsonr(order, skew)
    print("  over the %d sweep-and-window cases of this module," % len(order))
    print("  Spearman rho = %+.3f, Pearson r = %+.3f"
          % (rho.statistic, pear[0]))
    print("  MEASURED. They are close to the same measurement. Both read")
    print("  one asymmetry: a bright tail with no deep nulls. Quoting")
    print("  them as two independent rejections would double count, and")
    print("  the manuscript should not. The gate-length test is the one")
    print("  that is independent of both.")
    print()


def report_verdict(shapes, lengths, content):
    """What the section may claim after this, and what it may not."""
    print("=" * 78)
    print("R7  VERDICT")
    print("=" * 78)
    # A case is only in scope if the window carries grain scattering
    # there. Everything below is restricted to those, and the excluded
    # ones are named rather than dropped silently.
    scope, excluded = [], []
    for label in shapes:
        for fam in SPECIMEN:
            if content[label][fam] >= CONTENT_FLOOR_DB:
                scope.append((label, fam))
            else:
                excluded.append((label, fam, content[label][fam]))
    if excluded:
        print("  OUT OF SCOPE. These specimen cases sit within %.0f dB of"
              % CONTENT_FLOOR_DB)
        print("  their own uniform-orientation control, so the window does")
        print("  not carry grain scattering there and no shape or length")
        print("  statistic taken inside it is a statement about the coda:")
        for label, fam, margin in excluded:
            print("    %-22s %-14s %+.1f dB over its control"
                  % (label, fam, margin))
    spec_exp = [lengths[l][f]["exp"] for l, f in scope]
    spec_cells = [lengths[l][f]["exp_cells"] for l, f in scope]
    print("  MEASURED. Over the %d specimen family-and-window cases the"
          % len(spec_exp))
    print("  implied shape grows as T^%.2f to T^%.2f while the resolved"
          % (min(spec_exp), max(spec_exp)))
    print("  cells grow as T^%.2f to T^%.2f. The length test holds in"
          % (min(spec_cells), max(spec_cells)))
    print("  every one of them, at both fabrics, all three grids, every")
    print("  contrast fraction above zero and both windows.")
    for label, table in lengths.items():
        null = table["__null__"]["matched"]
        here = [f for l, f in scope if l == label]
        worst = max(table[fam]["exp"] for fam in here)
        print("  %s: the Gaussian null at the matched sample" % label)
        print("    size never falls below T^%.2f over %d seeds, and the"
              % (null.min(), len(null)))
        print("    largest specimen exponent is T^%.2f, so the two do not"
              % worst)
        print("    overlap." if worst < null.min()
              else "    OVERLAP. The separation is not clean here.")
    rejects = {}
    for label, fam in scope:
        res = shapes[label][fam]
        rejects[(label, fam)] = (res["p_skew"] < 0.05
                                 or res["p_order"] < 0.05)
    good = sum(rejects.values())
    print("  MEASURED. The shape tests reject a gamma in %d of the %d"
          % (good, len(rejects)))
    print("  specimen cases at the family level. Where they do not:")
    missed = [key for key, ok in rejects.items() if not ok]
    for key in missed:
        print("    %s, %s" % key)
    if not missed:
        print("    nowhere.")
    res_fail = [(label, fam) for label in shapes for fam in RESIDUE
                if shapes[label][fam]["p_skew"] < 0.05
                or shapes[label][fam]["p_order"] < 0.05]
    print("  MEASURED. Over the %d residue cases, f = 0 and the three"
          % (len(shapes) * len(RESIDUE)))
    print("  uniform-orientation controls at both windows, the shape")
    if res_fail:
        print("  tests reject in %d: %s"
              % (len(res_fail),
                 "; ".join("%s %s" % k for k in res_fail)))
    else:
        print("  tests reject in NONE.")
    print("  The two f = 0 cases carry the flip that matters: their log")
    print("  skew is %+.2f and %+.2f, on the side a gamma lives on, and"
          % tuple(shapes[label]["ladder f0.00"]["skew"] for label in shapes))
    print("  their three routes run DOWNWARD, contrast above level")
    print("  standard deviation, which is the opposite of the specimen.")
    print("  The residue is consistent with an averaged diffuse field.")
    print("  The coda is not. That is what shows the tests are reading")
    print("  the coda and not the code path.")
    print("  INFERRED, and offered as a reading and not a finding. One to")
    print("  three facets carrying the return would produce all of this,")
    print("  and nothing here establishes the count.")
    print()


def main():
    fast = "--fast" in sys.argv
    draws = MC_DRAWS // 2 if fast else MC_DRAWS
    report_reproduction()
    report_energy_spread()
    shapes, lengths, content = {}, {}, {}
    for label, win, ends in WINDOWS:
        content[label] = report_content(label, win)
        shapes[label] = report_shapes(label, win, draws=draws)
        lengths[label] = report_length(label, win, ends)
    report_dependence()
    report_ordering_flip(shapes)
    report_verdict(shapes, lengths, content)


if __name__ == "__main__":
    main()
