"""Ensemble statistics over independent tessellations at ppw 8.

Supports Section 4, "Ensemble statistics", and Table 'ensemble', and
supplies the archive that Fig. 'ensemble' is drawn from. Four questions,
in the order the section asks them:

  1. the mean coda level over independent tessellations and its spread,
  2. how many realisations are needed before that mean stabilises,
  3. whether the spread comes from the tessellation geometry or from the
     orientation draw laid on it,
  4. whether it comes from the microstructure at all, or merely from
     sampling the revolution at 30 azimuths.

THE ESTIMATOR. Levels are the AUDITED ones of analysis/db_reconcile.py:
2*E[x^2] over the coda gate on the trace band-limited to 0.8 to 3.0 MHz,
filtered before gating. The published figures for this subsection used
the global unfiltered Hilbert envelope instead, which counts grid noise
above two thirds of the grid Nyquist and adds a pedestal deposited across
the gate by the front arrival. Both are measured here, because the
difference between them is part of what this module has to report.

THE AVERAGING CONVENTION, which is the decision this rebuild turned on.
A revolution mean is a quadrature over a closed rotation of a
BACKSCATTERED POWER, so the average is taken over powers and converted to
decibels once. Averaging decibels instead returns the geometric mean of
the powers, which is biased low by an amount set by the contrast of the
field: exactly gamma*10*log10(e) = 2.507 dB for fully developed speckle,
and zero for a deterministic field. That bias is not a constant offset
here. Measured over the eight girdle tessellations it ranges from 1.77 to
3.07 dB, and over the four single maxima from 1.29 to 6.59 dB, so a mean
of decibels mixes the speckle contrast of each realisation into its
level. Powers are also the only additive quantity: every control in
Section 4 is a statement about shares of the gate power, and the mean of
the decibels of a sum is not a function of the means of the decibels of
its parts. report_convention prints the whole two-by-two.

Reads, all under ../../out/sweeps, each directory az*.npz with keys
'trace' and 'dt' plus a config.json:
    girdle_perp_ppw8, mx_girdle_s{7,17,23,41,53,71,89}_ppw8
        eight independent tessellations, girdle fabric, kappa = -8
    singlemax_ppw8, mx_single_s{7,17,23,41,53,71,89}_ppw8
        four single-maximum sweeps, kappa = +3.93, each sharing a
        bit-identical tessellation with the girdle sweep of the same seed
Question 3 also rebuilds the eight tessellations on the CPU at a coarse
grid, through sim/core/specimen.py, to measure their geometry directly.

Writes ../results/ensemble.npz for figures/fig_ensemble.py, and prints
every number the section quotes.

WHY A FABRIC EXCHANGE SEPARATES THE TWO SOURCES. DiskSpecimen.build draws
the Laguerre seed points and their weights before it draws any c-axis, so
two specimens with the same seed and different fabric parameters have a
bit-identical tessellation and unrelated orientations. Verified here by
comparing the label volumes. A seed change therefore moves geometry and
orientations together, a fabric change at fixed seed moves the
orientations alone, and the four shared seeds give a balanced 4 by 2
design whose interaction term is the orientation contribution at frozen
facet geometry.

WHY THE SUBSAMPLE BAND IS ENUMERATED AND NOT BOOTSTRAPPED. With eight
realisations there are only 254 non-trivial subsets, so the distribution
of an N-realisation mean is written down exactly. A bootstrap would
resample the same eight values with replacement and add its own noise to
an answer that can be had in closed form.
"""
import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import butter, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = ROOT / "out" / "sweeps"
RESULTS = ROOT / "sim" / "results"

# Acquisition and analysis constants, all as given in Section 3: 2 MHz
# centre frequency, 100 mm disc, reference speed 3850 m/s. The coda gate
# and the analysis band are the ones used throughout the paper, and the
# band is the one db_reconcile.py audits.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
CODA_GATE = (24e-6, 36e-6)
BAND = (0.8e6, 3.0e6)

# Every sweep in the matrix carries these 30 azimuths. The two seed-11
# sweeps were run at 6 degrees and hold 60; they are decimated to the
# common grid so that no realisation is averaged over a different set of
# beam positions from the others.
AZ_STEP_DEG = 12
AZ_COMMON = np.arange(0, 360, AZ_STEP_DEG)

GIRDLE = [("girdle_perp_ppw8", 11), ("mx_girdle_s7_ppw8", 7),
          ("mx_girdle_s17_ppw8", 17), ("mx_girdle_s23_ppw8", 23),
          ("mx_girdle_s41_ppw8", 41), ("mx_girdle_s53_ppw8", 53),
          ("mx_girdle_s71_ppw8", 71), ("mx_girdle_s89_ppw8", 89)]
SINGLE = [("singlemax_ppw8", 11), ("mx_single_s7_ppw8", 7),
          ("mx_single_s17_ppw8", 17), ("mx_single_s23_ppw8", 23),
          ("mx_single_s41_ppw8", 41), ("mx_single_s53_ppw8", 53),
          ("mx_single_s71_ppw8", 71), ("mx_single_s89_ppw8", 89)]

# The audited estimator and the published one, in that order. Keys into
# every power dictionary this module passes about.
AUDITED, LEGACY = "coda_band", "coda_env"

# Tolerances the ensemble mean is asked to hold, in decibels. The
# coarsest is half the ppw 8 to 10 step of Table 'reconcile', which is
# the smallest quantity Section 4 asks the ensemble to resolve; the finer
# two are there to show how fast the requirement grows.
TOL_DB = (1.0, 0.5, 0.25)

# Reported as the standard interval throughout, and stated in the
# caption of Fig. 'ensemble'.
CONF = 0.95

# Arithmetic minus geometric mean, in decibels, for an exponentially
# distributed intensity: gamma * 10 * log10(e). The reference the
# measured speckle contrast is judged against.
RAYLEIGH_EXCESS_DB = 10 * np.log10(np.e) * 0.5772156649015329


# ------------------------------------------------------------------ load

def measure_trace(path):
    """Gate power under both estimators, referenced to the source.

    Powers, not amplitudes, because everything downstream partitions
    power. The source reference is the peak of the analytic envelope of
    the whole unfiltered trace, which is the transmit pulse, and it is
    taken unfiltered under both estimators so that the two differ only in
    how they read the gate.

      coda_env   mean square of the global unfiltered Hilbert envelope
                 inside the gate. The published estimator.
      coda_band  2*E[x^2] inside the gate on the band-limited trace,
                 filtered before gating. The audited estimator.
    """
    with np.load(path) as handle:
        trace = np.asarray(handle["trace"], float).ravel()
        dt = float(handle["dt"])
    fs = 1.0 / dt
    i0, i1 = int(CODA_GATE[0] * fs), int(CODA_GATE[1] * fs)
    envelope = np.abs(hilbert(trace))
    src2 = envelope.max() ** 2
    sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype="band",
                 output="sos")
    filtered = sosfiltfilt(sos, trace)
    return {LEGACY: (envelope[i0:i1] ** 2).mean() / src2,
            AUDITED: 2.0 * (filtered[i0:i1] ** 2).mean() / src2}


def sweep_powers(name, step_deg=AZ_STEP_DEG):
    """Per-azimuth gate powers for one sweep. Returns (az, powers, cfg).

    step_deg selects the azimuth grid. The default keeps only the common
    12 degree grid, so that no realisation is averaged over a different
    set of beam positions from the others; step_deg of 1 keeps every
    record on disk, which only the two seed-11 sweeps have more of.
    """
    folder = SWEEPS / name
    with open(folder / "config.json") as handle:
        cfg = json.load(handle)
    az, rows = [], []
    for entry in sorted(os.listdir(folder)):
        if not (entry.startswith("az") and entry.endswith(".npz")):
            continue
        degrees = int(entry[2:5])
        if degrees % step_deg:
            continue
        az.append(degrees)
        rows.append(measure_trace(folder / entry))
    powers = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    return np.array(az), powers, cfg


def load_matrix():
    """Every sweep of the ensemble, read once. Keyed by sweep name.

    Read once because the two-by-two of estimator and convention needs
    the same 420 traces four times over, and re-reading them per cell
    would quadruple the only expensive step in the module.
    """
    matrix = {}
    for name, seed in GIRDLE + SINGLE:
        step = 1 if name in ("girdle_perp_ppw8", "singlemax_ppw8") else \
            AZ_STEP_DEG
        az, powers, cfg = sweep_powers(name, step_deg=step)
        if cfg["seed"] != seed:
            raise ValueError("%s carries seed %s, expected %d"
                             % (name, cfg["seed"], seed))
        keep = az % AZ_STEP_DEG == 0
        if not np.array_equal(az[keep], AZ_COMMON):
            raise ValueError("%s does not cover the common azimuth grid"
                             % name)
        matrix[name] = dict(az=az, keep=keep, **powers)
    return matrix


# --------------------------------------------------------------- compute

def level(power, convention="energy"):
    """Revolution level in dB under one averaging convention.

    'energy' averages the powers and converts once, which is the
    convention this paper uses and the one the module docstring argues
    for. 'db' averages the per-azimuth decibels, which is the geometric
    mean of the same powers and is kept only so that the published
    convention can be printed beside it.
    """
    power = np.asarray(power, float)
    if convention == "energy":
        return float(10.0 * np.log10(power.mean()))
    return float((10.0 * np.log10(power)).mean())


def family_levels(matrix, family, key=AUDITED, convention="energy"):
    """(seeds, revolution levels, per-azimuth level matrix) for a fabric."""
    seeds, levels, rows = [], [], []
    for name, seed in family:
        entry = matrix[name]
        power = entry[key][entry["keep"]]
        seeds.append(seed)
        levels.append(level(power, convention))
        rows.append(10.0 * np.log10(power))
    return np.array(seeds), np.array(levels), np.array(rows)


def speckle_excess(matrix, family, key=AUDITED):
    """Arithmetic minus geometric mean of the gate power, in decibels.

    The gap between the two averaging conventions, per realisation. It is
    the contrast of the azimuthal intensity distribution expressed as a
    level, so RAYLEIGH_EXCESS_DB is the value a fully developed speckle
    field takes and zero is the value a deterministic field takes.
    """
    out = []
    for name, _seed in family:
        entry = matrix[name]
        power = entry[key][entry["keep"]]
        out.append(level(power, "energy") - level(power, "db"))
    return np.array(out)


def convention_grid(matrix):
    """Ensemble mean and spread in all four cells of the two-by-two.

    Estimator crossed with averaging convention, so that the two changes
    the published subsection needs are separated on the record rather
    than applied together and attributed to one of them.
    """
    grid = {}
    for key in (LEGACY, AUDITED):
        for convention in ("db", "energy"):
            _s, girdle, _m = family_levels(matrix, GIRDLE, key, convention)
            _s, single, _m = family_levels(matrix, SINGLE, key, convention)
            grid[(key, convention)] = dict(girdle=girdle, single=single)
    return grid


def quadrature_error(matrix, key=AUDITED, convention="energy"):
    """Spread of a 30-azimuth revolution mean, measured not assumed.

    A revolution mean is not a sample from a population: the sweep covers
    the whole rotation. Treating the azimuths as correlated samples and
    dividing the scatter by the square root of an effective count
    therefore overstates the error on the mean. The two 60-azimuth sweeps
    settle it: split each into its two interleaved 30-azimuth halves,
    which are two independent 12-degree grids over the same specimen, and
    the half difference of their means is the quantity wanted.

    Reported per sweep and never pooled, because under the energy
    convention the two fabrics do not give the same answer: a
    single-maximum revolution carries much of its energy in a few bright
    azimuths and a 30-azimuth grid resolves that far less well than it
    resolves a girdle.
    """
    out = {}
    for name in ("girdle_perp_ppw8", "singlemax_ppw8"):
        power = matrix[name][key]
        halves = np.array([level(power[0::2], convention),
                           level(power[1::2], convention)])
        out[name] = dict(n=len(power), full=level(power, convention),
                         halves=halves,
                         err=0.5 * abs(halves[0] - halves[1]))
    return out


def effective_sample_size_error(rows):
    """The estimator Section 4 rejects, evaluated so it can be rejected.

    Divides the per-azimuth scatter by the square root of an effective
    sample size taken from the integral scale of the circular
    autocorrelation. Returned as (per-azimuth sd, mean N_eff, implied
    error), pooled over the realisations passed in.
    """
    variances, counts = [], []
    for row in rows:
        variances.append(row.var(ddof=1))
        centred = row - row.mean()
        n = len(centred)
        spectrum = np.fft.rfft(centred, n=2 * n)
        acf = np.fft.irfft(spectrum * np.conj(spectrum), n=2 * n)[:n]
        acf /= (n - np.arange(n))
        acf /= acf[0]
        negative = np.where(acf < 0)[0]
        cut = negative[0] if len(negative) else n // 2
        tau = (1.0 + 2.0 * acf[1:cut].sum()) * AZ_STEP_DEG
        counts.append(360.0 / max(tau, float(AZ_STEP_DEG)))
    sd = float(np.sqrt(np.mean(variances)))
    neff = float(np.mean(counts))
    return sd, neff, sd / np.sqrt(neff)


def subsample_band(levels, lo=5.0, hi=95.0):
    """Exact distribution of the mean of N of the realisations, N = 1..n.

    Returns (N, low percentile, high percentile, min, max). This is the
    honest answer to "what would I have concluded had I stopped at N",
    and it necessarily closes to a point at N = n, where only one subset
    exists. The residual uncertainty at N = n is the confidence interval
    of the mean, which is a different quantity and is reported alongside.
    """
    n = len(levels)
    counts = np.arange(1, n + 1)
    p_lo, p_hi, v_lo, v_hi = [], [], [], []
    for k in counts:
        means = np.array([np.mean(c) for c in
                          itertools.combinations(levels, int(k))])
        p_lo.append(np.percentile(means, lo))
        p_hi.append(np.percentile(means, hi))
        v_lo.append(means.min())
        v_hi.append(means.max())
    return (counts, np.array(p_lo), np.array(p_hi), np.array(v_lo),
            np.array(v_hi))


def subsample_agreement(levels, tol):
    """Fraction of N-realisation means landing within tol of the full mean.

    The stabilisation criterion of the section: the smallest N at which a
    reader who stopped there would have quoted a level within tol of the
    one quoted here, with probability CONF.
    """
    n = len(levels)
    full = levels.mean()
    frac = []
    for k in range(1, n + 1):
        means = np.array([np.mean(c) for c in
                          itertools.combinations(levels, k)])
        frac.append(float(np.mean(np.abs(means - full) <= tol)))
    return np.arange(1, n + 1), np.array(frac)


def mean_interval(levels, conf=CONF):
    """Mean, sample standard deviation, and Student-t interval half width."""
    n = len(levels)
    sd = levels.std(ddof=1)
    half = stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n)
    return levels.mean(), sd, half


def realisations_needed(sd, tol, conf=CONF):
    """Smallest N whose t interval half width falls below tol.

    Solved by search rather than with the normal quantile because at
    these counts the t multiplier is what dominates: 4.30 at N = 3
    against 1.96 asymptotically.
    """
    for n in range(2, 400):
        if stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n) <= tol:
            return n
    raise ValueError("no count below 400 gives a %.2f dB interval at a "
                     "spread of %.2f dB" % (tol, sd))


def variance_ratio_paired(a, b):
    """Pitman-Morgan test that two levels of the same eight differ in spread.

    The four cells of the two-by-two are four measurements of the SAME
    eight tessellations, so an F test, which assumes independent samples,
    is the wrong comparison and is far too conservative here. The paired
    test uses the correlation between the sum and the difference of the
    two series, which is zero exactly when their variances are equal.
    """
    n = len(a)
    r = float(np.corrcoef(a - b, a + b)[0, 1])
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return r, float(t), float(2 * stats.t.sf(abs(t), n - 2))


def variance_partition(girdle, single):
    """Split the spread into tessellation geometry and orientation draw.

    A balanced 4 by 2 layout, one sweep per cell, seeds crossed with
    fabric. Writing g and s for the two levels at a shared tessellation,

        d = g - s     removes the tessellation entirely, so its variance
                      is twice the orientation component
        m = (g+s)/2   keeps the tessellation and averages the two
                      orientation draws, so its variance is the geometry
                      component plus half the orientation component

    which inverts to the two components below. The mean of d is the
    fabric main effect and is not a source of spread: it is the physical
    difference between a girdle and a single maximum, and it is reported
    separately as the yardstick the ensemble uncertainty is judged
    against.

    A negative geometry variance is a legitimate outcome of this
    estimator at three degrees of freedom and means the geometry
    component is not resolved. It is returned unclipped, because
    clipping it to zero would hide exactly that.
    """
    d = girdle - single
    m = 0.5 * (girdle + single)
    var_orient = d.var(ddof=1) / 2.0
    var_geom = m.var(ddof=1) - var_orient / 2.0
    r = float(np.corrcoef(girdle, single)[0, 1])
    return dict(fabric_effect=float(d.mean()), var_orient=float(var_orient),
                var_geom=float(var_geom), icc=r, n_pairs=len(d))


def fisher_interval(r, n, conf=CONF):
    """Interval on a correlation, through the variance-stabilising map."""
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    s = stats.norm.ppf(0.5 + conf / 2) / np.sqrt(n - 3)
    return float(np.tanh(z - s)), float(np.tanh(z + s))


def shared_azimuth_correlation(matrix, key=AUDITED):
    """Does a shared tessellation imprint a common pattern in azimuth?

    An independent check on the paired design that does not go through
    the revolution mean at all. Correlate the per-azimuth level of a
    girdle sweep against that of a single-maximum sweep, once for the
    four pairs that share a tessellation and once for the twelve that do
    not. Both groups cross the fabric, so whatever azimuthal pattern the
    fabric itself imposes is common to them and cancels in the
    comparison; only the tessellation differs.
    """
    girdle = {seed: name for name, seed in GIRDLE}
    same, cross, detail = [], [], []
    for name_s, seed_s in SINGLE:
        for name_g, seed_g in GIRDLE:
            if seed_g not in [s for _n, s in SINGLE]:
                continue
            a = np.log10(matrix[girdle[seed_g]][key]
                         [matrix[girdle[seed_g]]["keep"]])
            b = np.log10(matrix[name_s][key][matrix[name_s]["keep"]])
            r = float(np.corrcoef(a, b)[0, 1])
            (same if seed_g == seed_s else cross).append(r)
            detail.append((seed_g, seed_s, r))
    t, p = stats.ttest_ind(same, cross, equal_var=False)
    return dict(same=np.array(same), cross=np.array(cross), detail=detail,
                t=float(t), p=float(p))


def tessellation_geometry(seeds, coarse_ppw=4.0):
    """Purely geometric descriptors of each tessellation.

    An independent check on the paired design, which has only three
    degrees of freedom. If the spread were driven by the geometry then
    some geometric descriptor of the tessellation would have to track the
    measured level across all eight. Four are computed: the realised
    grain count, the mean equivalent grain diameter, the total interior
    grain-boundary area of the disc, and the number of grain-boundary
    crossings inside the beam column over the coda gate, the last through
    the facet machinery of facet_predictors so that the beam geometry is
    the same one the facet model of Section 5 uses.

    The specular-weighted facet sum of that module is deliberately not
    among them. Its directivity lobe is a few degrees wide at this grain
    size and frequency, so the sum is carried by a handful of near-normal
    facets and is heavy tailed; eight tessellations cannot estimate it,
    and it belongs to the facet-model analyses, which build at production
    resolution.

    Built at a quarter wavelength rather than at the production spacing.
    The Laguerre seed points and weights do not depend on the grid at
    all, so the tessellation is the production one; only its rasterised
    boundary area is coarser, and that enters as a relative comparison
    across seeds. Face counting on a cubic grid overestimates a smooth
    area by a fixed factor, which cancels in that comparison.
    """
    import sys
    sys.path.insert(0, str(ROOT / "sim"))
    sys.path[:0] = [os.path.join(sys.path[0], _d)
                    for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
    sys.path.insert(0, str(ROOT / "sim" / "analysis"))
    sys.path[:0] = [os.path.join(sys.path[0], _d)
                    for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
    import facet_predictors as fp
    from specimen import DiskSpecimen

    # Labelled on the CPU. This module is analysis and has to run on the
    # release machine whether or not it has a CUDA device; the two paths
    # were measured to differ in three cells of 10.7 million, which is
    # far below anything quantified here.
    DiskSpecimen._label_grid_gpu = staticmethod(lambda *a, **k: None)

    h = C_REF / F0 / coarse_ppw
    rows = []
    for seed in seeds:
        built = DiskSpecimen(
            diameter_m=DIA, thickness_m=0.035, n_grains=100, size_cv=0.35,
            concentration=-8.0, spatial_corr=0.0, fabric_axis=(1.0, 0.0, 0.0),
            seed=int(seed)).build(h)
        labels = np.asarray(built["labels"])
        centres = np.asarray(built["seeds"], float)
        cells = np.bincount(labels[labels >= 0].ravel(),
                            minlength=len(centres))
        live = cells[cells > 0]
        d_eq = 2 * (3 * live * h ** 3 / (4 * np.pi)) ** (1 / 3)
        faces = 0
        for axis in range(3):
            lo = np.take(labels, np.arange(labels.shape[axis] - 1), axis=axis)
            hi = np.take(labels, np.arange(1, labels.shape[axis]), axis=axis)
            faces += int(((lo != hi) & (lo >= 0) & (hi >= 0)).sum())
        pred = fp.preds(labels, np.asarray(built["axes"], float), centres, h,
                        AZ_COMMON)
        rows.append(dict(seed=int(seed), n_grains=int((cells > 0).sum()),
                         d_mean_mm=float(d_eq.mean() * 1e3),
                         area_cm2=float(faces * h ** 2 * 1e4),
                         crossings=float(pred["n_cross"].mean())))
    return rows


def proportional_channel_cap(values):
    """Level spread a channel proportional to a descriptor can produce.

    Backscattered power from a population of facets is proportional to
    the illuminated boundary area, and to the number of boundaries the
    beam column crosses, to first order. A relative spread of x in either
    therefore becomes 10 log10(1+x) in level. This is the cap on the
    geometry contribution through that channel, and it needs no fit.
    """
    values = np.asarray(values, float)
    return float(10 * np.log10(1 + values.std(ddof=1) / values.mean()))


def tessellations_are_shared(seed, coarse_ppw=4.0):
    """True when a girdle and a single maximum of one seed share labels.

    The premise of the whole paired design, checked rather than asserted.
    """
    import sys
    sys.path.insert(0, str(ROOT / "sim"))
    sys.path[:0] = [os.path.join(sys.path[0], _d)
                    for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
    from specimen import DiskSpecimen
    DiskSpecimen._label_grid_gpu = staticmethod(lambda *a, **k: None)
    h = C_REF / F0 / coarse_ppw
    common = dict(diameter_m=DIA, thickness_m=0.035, n_grains=100,
                  size_cv=0.35, spatial_corr=0.0, seed=int(seed))
    a = DiskSpecimen(concentration=-8.0, fabric_axis=(1.0, 0.0, 0.0),
                     **common).build(h)
    b = DiskSpecimen(concentration=3.93, fabric_axis=(0.866, 0.5, 0.0),
                     **common).build(h)
    return (np.array_equal(a["labels"], b["labels"])
            and not np.array_equal(a["axes"], b["axes"]))


# ---------------------------------------------------------------- report

def report_convention(matrix):
    """The two-by-two, which is what settles the estimator and the average.

    Printed first because every other number in the module depends on
    which cell it is read from, and because the published subsection sat
    in the one cell of the four that disagrees with the other three.
    """
    grid = convention_grid(matrix)
    print("ESTIMATOR AND AVERAGING CONVENTION, ppw 8, %d azimuths"
          % len(AZ_COMMON))
    print("  %-24s %-9s %9s %7s %9s %7s"
          % ("estimator", "average", "girdle", "sd", "single", "sd"))
    names = {LEGACY: "published global env",
             AUDITED: "audited local in band"}
    labels = {"db": "mean dB", "energy": "mean E"}
    for key in (LEGACY, AUDITED):
        for convention in ("db", "energy"):
            cell = grid[(key, convention)]
            print("  %-24s %-9s %9.2f %7.2f %9.2f %7.2f"
                  % (names[key], labels[convention],
                     cell["girdle"].mean(), cell["girdle"].std(ddof=1),
                     cell["single"].mean(), cell["single"].std(ddof=1)))
    print()
    base = grid[(LEGACY, "db")]["girdle"]
    print("  the two changes are NOT additive in the girdle spread")
    print("  %-34s %8s %8s" % ("change from the published cell", "sd dB",
                               "d sd"))
    for key, convention, tag in (
            (AUDITED, "db", "estimator only"),
            (LEGACY, "energy", "convention only"),
            (AUDITED, "energy", "both, which is what is used")):
        sd = grid[(key, convention)]["girdle"].std(ddof=1)
        print("  %-34s %8.2f %+8.2f" % (tag, sd, sd - base.std(ddof=1)))
    predicted = (grid[(AUDITED, "db")]["girdle"].std(ddof=1)
                 + grid[(LEGACY, "energy")]["girdle"].std(ddof=1)
                 - base.std(ddof=1))
    both = grid[(AUDITED, "energy")]["girdle"].std(ddof=1)
    print("  additive prediction %.2f dB against %.2f observed, "
          "interaction %+.2f dB" % (predicted, both, both - predicted))
    print("  Three of the four cells agree at about 1.0 dB. Only the")
    print("  published corner is low, and it needs BOTH the pedestal and")
    print("  the geometric mean in place at once to be low: undo either")
    print("  and the spread returns to 1.0 dB.")
    print()
    for key, convention in ((AUDITED, "db"), (LEGACY, "energy"),
                            (AUDITED, "energy")):
        r, t, p = variance_ratio_paired(base, grid[(key, convention)]
                                        ["girdle"])
        print("  paired variance test, published against %-16s "
              "t = %+5.2f  p = %.3f" % ("%s/%s" % (names[key].split()[0],
                                                   labels[convention]), t, p))
    print()

    print("SPECKLE CONTRAST OF THE AZIMUTHAL GATE POWER")
    print("  arithmetic minus geometric mean, in dB. Fully developed")
    print("  speckle gives %.3f dB exactly, a deterministic field zero."
          % RAYLEIGH_EXCESS_DB)
    print("  %-24s %10s %10s" % ("family", "published", "audited"))
    for family, tag in ((GIRDLE, "girdle, eight"),
                        (SINGLE, "single maximum, eight")):
        old = speckle_excess(matrix, family, LEGACY)
        new = speckle_excess(matrix, family, AUDITED)
        print("  %-24s %4.2f to %4.2f %4.2f to %4.2f"
              % (tag, old.min(), old.max(), new.min(), new.max()))
    print("  The audited gate power is fully developed speckle in")
    print("  azimuth. The published estimator reads a contrast well")
    print("  below Rayleigh because the out-of-band grid noise and the")
    print("  pedestal of the front arrival fill in its nulls, and the")
    print("  amount they fill in varies from tessellation to")
    print("  tessellation. That variation is why a mean of decibels and")
    print("  a mean of energies do not differ by a constant here.")
    print()


def report_ensemble(matrix):
    """The headline levels, the spread, and the sample-size requirement."""
    g_seeds, g_lev, g_mat = family_levels(matrix, GIRDLE)
    s_seeds, s_lev, s_mat = family_levels(matrix, SINGLE)

    print("CODA LEVEL PER REALISATION, audited, energies, dB re source")
    print("  %-6s %-16s %10s %10s" % ("seed", "fabric", "level", "az sd"))
    for seed, lev, row in zip(g_seeds, g_lev, g_mat):
        print("  %-6d %-16s %10.2f %10.2f"
              % (seed, "girdle", lev, row.std(ddof=1)))
    for seed, lev, row in zip(s_seeds, s_lev, s_mat):
        print("  %-6d %-16s %10.2f %10.2f"
              % (seed, "single maximum", lev, row.std(ddof=1)))

    g_mean, g_sd, g_half = mean_interval(g_lev)
    s_mean, s_sd, s_half = mean_interval(s_lev)
    print("\nensemble mean over independent tessellations")
    print("  girdle          N = %d   %.2f +- %.2f dB (sd %.2f, "
          "%.0f%% interval)" % (len(g_lev), g_mean, g_half, g_sd, 100 * CONF))
    print("  single maximum  N = %d   %.2f +- %.2f dB (sd %.2f)"
          % (len(s_lev), s_mean, s_half, s_sd))
    print("  girdle realisations span %.2f to %.2f dB, range %.2f dB"
          % (g_lev.min(), g_lev.max(), g_lev.max() - g_lev.min()))
    print("  single realisations span %.2f to %.2f dB; dropping seed 17 "
          "leaves\n  a mean of %.2f dB and a spread of %.2f dB, so this "
          "family is one\n  realisation wide and its interval should not "
          "be read as a spread"
          % (s_lev.min(), s_lev.max(),
             np.delete(s_lev, list(s_seeds).index(17)).mean(),
             np.delete(s_lev, list(s_seeds).index(17)).std(ddof=1)))
    separation = s_mean - g_mean
    t, p = stats.ttest_ind(s_lev, g_lev, equal_var=False)
    print("  fabric separation %+.2f dB, %.1f times the girdle interval, "
          "Welch t = %.2f p = %.3f" % (separation, separation / g_half, t, p))
    print("  per-azimuth scatter pooled over the eight girdle sweeps: "
          "%.2f dB" % np.sqrt((g_mat.var(axis=1, ddof=1)).mean()))
    return (g_seeds, g_lev, g_mat, s_seeds, s_lev, g_mean, g_sd, g_half)


def report_sampling(matrix, g_mat, g_sd):
    """Is the spread microstructural, or is it 30 azimuths being 30?"""
    quad = quadrature_error(matrix)
    print("\nAZIMUTHAL SAMPLING ERROR ON ONE REVOLUTION MEAN")
    for name, q in quad.items():
        print("  %-18s %d azimuths, full %.3f, interleaved halves %.3f and "
              "%.3f,\n  %-18s half difference %.3f dB"
              % (name, q["n"], q["full"], q["halves"][0], q["halves"][1],
                 "", q["err"]))
    girdle_err = quad["girdle_perp_ppw8"]["err"]
    print("  A girdle revolution mean is repeatable to %.2f dB, which is "
          "%.0f%% of\n  the %.2f dB ensemble spread and cannot explain it."
          % (girdle_err, 100 * girdle_err / g_sd, g_sd))
    print("  The single maximum is repeatable only to %.2f dB, because "
          "under the\n  energy convention its revolution energy is carried "
          "by a few bright\n  azimuths that a 30-azimuth grid samples "
          "coarsely."
          % quad["singlemax_ppw8"]["err"])
    sd, neff, err = effective_sample_size_error(g_mat)
    print("  The estimator this replaces, per-azimuth scatter %.2f dB over "
          "an\n  effective sample size of %.1f, returns %.2f dB, which is "
          "still larger\n  than the whole ensemble spread and is still the "
          "wrong estimator." % (sd, neff, err))
    return quad, girdle_err


def report_subsamples(g_lev, g_sd):
    """How many realisations the mean needs, two ways."""
    counts, p_lo, p_hi, v_lo, v_hi = subsample_band(g_lev)
    agree = {tol: subsample_agreement(g_lev, tol)[1] for tol in TOL_DB}
    print("\nHOW FAR AN N-REALISATION MEAN CAN BE FROM THE EIGHT-"
          "REALISATION MEAN")
    print("  %-3s %13s %15s %s" % ("N", "5 to 95 pc", "worst case",
                                   "".join("  within %.2f" % t
                                           for t in TOL_DB)))
    for i, k in enumerate(counts):
        print("  %-3d %6.2f %6.2f %7.2f %7.2f %s"
              % (k, p_lo[i], p_hi[i], v_lo[i], v_hi[i],
                 "".join("%11.0f%%" % (100 * agree[t][i]) for t in TOL_DB)))
    print("  smallest N holding a tolerance for %.0f%% of subsets, and the "
          "N whose\n  %.0f%% interval on the population mean is that wide"
          % (100 * CONF, 100 * CONF))
    for tol in TOL_DB:
        settled = counts[agree[tol] >= CONF]
        print("    %.2f dB   subsets N = %-8s interval N = %s"
              % (tol, settled[0] if len(settled) else "never",
                 realisations_needed(g_sd, tol)))
    print("  the worst single realisation is %.2f dB from the mean, so "
          "N = 1 no\n  longer holds even the 1.00 dB tolerance"
          % np.abs(g_lev - g_lev.mean()).max())


def report_partition(matrix, g_seeds, g_lev, s_seeds, s_lev):
    """Geometry against orientation, on the paired design and on all eight."""
    shared = [int(seed) for seed in s_seeds if seed in set(g_seeds)]
    idx_g = [int(np.where(g_seeds == seed)[0][0]) for seed in shared]
    idx_s = [int(np.where(s_seeds == seed)[0][0]) for seed in shared]
    part = variance_partition(g_lev[idx_g], s_lev[idx_s])
    lo, hi = fisher_interval(part["icc"], part["n_pairs"])
    print("\nGEOMETRY AGAINST ORIENTATION, %d shared tessellations"
          % part["n_pairs"])
    print("  seeds %s carry a bit-identical tessellation under both "
          "fabrics: %s" % (list(shared), tessellations_are_shared(shared[0])))
    print("  %-6s %12s %16s %12s" % ("seed", "girdle", "single maximum",
                                     "difference"))
    for seed, a, b in zip(shared, g_lev[idx_g], s_lev[idx_s]):
        print("  %-6d %12.2f %16.2f %12.2f" % (seed, a, b, a - b))
    print("  fabric main effect, girdle minus single maximum  %+.2f dB"
          % part["fabric_effect"])
    d = g_lev[idx_g] - s_lev[idx_s]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print("  paired t = %.2f on %d degrees of freedom, p = %.3f, so the "
          "fabric\n  effect is not established by these four pairs alone"
          % (t, len(d) - 1, 2 * stats.t.sf(abs(t), len(d) - 1)))
    print("  orientation draw at frozen geometry   sd %.2f dB"
          % np.sqrt(max(part["var_orient"], 0.0)))
    print("  tessellation geometry                 var %+.3f dB^2%s"
          % (part["var_geom"],
             ", not resolved" if part["var_geom"] < 0 else ""))
    print("  correlation of the two fabrics across shared tessellations "
          "r = %+.2f, %.0f%% interval [%+.2f, %+.2f]"
          % (part["icc"], 100 * CONF, lo, hi))

    corr = shared_azimuth_correlation(matrix)
    print("\n  per-azimuth pattern, shared against unshared tessellations")
    for seed_g, seed_s, r in corr["detail"]:
        print("    girdle s%-3d vs single s%-3d  r = %+.2f%s"
              % (seed_g, seed_s, r,
                 "   shared tessellation" if seed_g == seed_s else ""))
    print("    shared mean r = %+.2f (n %d), unshared mean r = %+.2f "
          "(n %d), Welch\n    t = %.2f p = %.3f. Both groups cross the "
          "fabric, so the fabric's own\n    azimuthal pattern cancels and "
          "only the tessellation differs."
          % (corr["same"].mean(), len(corr["same"]), corr["cross"].mean(),
             len(corr["cross"]), corr["t"], corr["p"]))
    return part


def report_geometry(g_seeds, g_lev, g_sd):
    """Do any purely geometric descriptors track the measured level?"""
    print("\nGEOMETRIC DESCRIPTORS OF THE EIGHT TESSELLATIONS")
    rows = tessellation_geometry(g_seeds)
    print("  %-6s %8s %10s %12s %11s"
          % ("seed", "grains", "d (mm)", "area (cm2)", "crossings"))
    for row in rows:
        print("  %-6d %8d %10.2f %12.1f %11.1f"
              % (row["seed"], row["n_grains"], row["d_mean_mm"],
                 row["area_cm2"], row["crossings"]))
    print("  correlation with the measured level, n = %d; four descriptors "
          "are\n  tested, so a Bonferroni threshold is p < %.4f"
          % (len(g_lev), 0.05 / 4))
    for key in ("n_grains", "d_mean_mm", "area_cm2", "crossings"):
        x = np.array([row[key] for row in rows], float)
        r = float(np.corrcoef(x, g_lev)[0, 1])
        p = 2 * stats.t.sf(abs(r) * np.sqrt((len(x) - 2) / (1 - r ** 2)),
                           len(x) - 2)
        print("    %-10s r = %+.2f  p = %.3f  (spread %.1f%% of the mean, "
              "cap %.2f dB)"
              % (key, r, p, 100 * x.std(ddof=1) / x.mean(),
                 proportional_channel_cap(x)))
    area = [row["area_cm2"] for row in rows]
    cross = [row["crossings"] for row in rows]
    print("  On the audited estimator the beam-column crossing count is "
          "the one\n  descriptor that tracks the level, r = %+.2f. The "
          "published estimator\n  gave r = +0.55 and nothing significant, "
          "so this is a change of\n  conclusion and not a restatement."
          % float(np.corrcoef(cross, g_lev)[0, 1]))
    print("  boundary area varies by %.2f dB in level, %.0f%% of the "
          "%.2f dB observed\n  spread; the crossing count varies by "
          "%.2f dB, %.0f%% of it"
          % (proportional_channel_cap(area),
             100 * proportional_channel_cap(area) / g_sd, g_sd,
             proportional_channel_cap(cross),
             100 * proportional_channel_cap(cross) / g_sd))
    return rows


# ------------------------------------------------------------------ main

def main():
    matrix = load_matrix()
    report_convention(matrix)
    (g_seeds, g_lev, g_mat, s_seeds, s_lev,
     g_mean, g_sd, g_half) = report_ensemble(matrix)
    quad, quad_err = report_sampling(matrix, g_mat, g_sd)
    report_subsamples(g_lev, g_sd)
    part = report_partition(matrix, g_seeds, g_lev, s_seeds, s_lev)
    rows = report_geometry(g_seeds, g_lev, g_sd)

    _s, s_lev_e, s_mat = family_levels(matrix, SINGLE)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS / "ensemble.npz",
             azimuths=AZ_COMMON,
             girdle_seeds=g_seeds, girdle_levels=g_lev, girdle_azimuth=g_mat,
             single_seeds=s_seeds, single_levels=s_lev, single_azimuth=s_mat,
             quadrature_err=quad_err, tol_db=np.array(TOL_DB), conf=CONF,
             var_orient=part["var_orient"], var_geom=part["var_geom"],
             icc=part["icc"], fabric_effect=part["fabric_effect"],
             area_cm2=np.array([row["area_cm2"] for row in rows], float),
             crossings=np.array([row["crossings"] for row in rows], float))
    print("\nwrote %s" % (RESULTS / "ensemble.npz"))


if __name__ == "__main__":
    main()
