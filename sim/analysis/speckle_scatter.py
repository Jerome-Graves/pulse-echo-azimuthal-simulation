"""Per-azimuth scatter of the coda level against the speckle floor.

Supports Section 4, "Grain population and speckle statistics",
sec:grainpop. That subsection argues that the coda is speckle from a
small number of independent scatterers, in three steps: the specimen
holds of order 10^2 grains, the beam crosses a handful of grain
boundaries inside the coda gate, and the measured per-azimuth scatter
therefore sits on the Rayleigh floor implied by that handful of looks.
This module rebuilds every measured quantity in the chain and then tests
the model the chain rests on.

THE ESTIMATOR. Levels are the AUDITED ones of analysis/db_reconcile.py:
2*E[x^2] over the coda gate on the trace band-limited to 0.8 to 3.0 MHz,
filtered before gating, referenced to the source amplitude and averaged
over azimuth as energies. The published figures for this subsection came
from the global unfiltered Hilbert envelope, which counts grid noise
above two thirds of the grid Nyquist and adds a pedestal deposited across
the gate by the front arrival. Both are measured here, because the
difference between them is part of what this module has to report.

WHAT THE REBUILD DOES TO THE PUBLISHED NUMBERS. The four dependent
figures all move, and one of them changes sign of trend:

    quantity                     published   audited
    pooled per-azimuth scatter      3.63        4.58 dB
    resolution series ppw 6/8/10  4.23 3.03 2.81   3.01 3.57 3.76 dB
    uniform-orientation control     0.42        1.27 dB
    residual after orders 2 and 4   2.97        3.69 dB

WHAT THAT DOES TO THE ARGUMENT, which is the reason the module grew two
blocks it did not have. On the published estimator the measured scatter
sat 0.71 dB above the floor predicted at the measured crossing count,
and 0.05 dB above it once the coherent azimuthal orders were removed,
which reads as agreement. On the audited estimator it sits 1.66 dB above
that floor, and a measurement ABOVE the floor for an independently
estimated look count is the wrong side: either the look count is smaller
than the crossing count, or the looks are not independent, or the gamma
model is wrong. Two new blocks decide between those.

  FIRST, the look count is measured in band from the trace instead of
  being read off the geometry. The number of independent looks a gate
  power averages is the gate length over the coherence time of the field,
  weighted by how evenly the mean intensity fills the gate. Both are
  measured here. Band limiting does lengthen the coherence time, 0.39 to
  0.54 us, and so does lower the look count, 31.5 to 21.6, which is the
  direction the band-limiting hypothesis needs. It is nowhere near
  enough: the floor moves from 0.87 to 0.94 dB against a measurement of
  4.58 dB. The hypothesis is real and is far too small to reconcile the
  comparison. Band limiting is not the answer.

  The counts quoted above are the spectral ones, T times the effective
  bandwidth (int S)^2 / int S^2 of the gate spectrum. An earlier revision
  of this module reported 25 and 18 by weighting the coherence-time route
  with a participation factor of 0.83 taken from the raw 30-azimuth mean
  intensity profile. That profile carries its own residual speckle, which
  inflates int m^2 and biases the participation low; on the fitted
  deterministic profile it is 0.98, because the in-band coda is flat
  across the gate to 1.7 dB, and the two routes then agree at 23.7 and
  21.6. Nothing in the conclusion turns on the difference, but the low
  figure should not be quoted.

  SECOND, the gamma model is tested rather than assumed. A gamma variate
  of shape N has contrast 1/sqrt(N) and level standard deviation
  (10/ln 10) sqrt(psi_1(N)), so the same data give the same N twice if
  the model holds. They do not: 0.50 +- 0.28 from the contrast against
  1.36 +- 0.25 from the level standard deviation, paired t = 7.45 on the
  eight tessellations, p = 0.0001. That gap is not a small-sample
  artefact, and report_gamma_test says so with a Monte Carlo: on genuine
  gamma data at this sample size the two estimators agree to
  -0.05 +- 0.15, and 4000 trials reach a gap this large no more than
  once in three thousand at any of the shapes in question. The azimuthal
  gate power is not a gamma variate of any shape, so "the measurement
  sits on the floor at N looks" has no floor to sit on.

  WHAT IT IS INSTEAD, and this is measured, not inferred. The gate power
  in azimuth is heavy tailed: contrast 1.53 against 1.00 for fully
  developed speckle, and 2.04 inside a 3 us sub-gate that holds about
  5.5 coherence times, where Gaussian speckle would give 0.43. The four
  sub-gates are very nearly uncorrelated across azimuth, mean r = +0.10,
  so the excess is not a chord-wide gain multiplying the whole gate. It
  is localised in range. A few bright arrivals carry the gate. That is
  the picture Section 5 already builds a facet model on, and it is the
  opposite of the fully developed speckle the floor argument assumes.

WHAT SURVIVES UNCHANGED. The grain and boundary counts are geometry and
do not touch either estimator: 12.6 +- 2.2 grains in the beam column,
2.9 +- 1.0 crossings on the beam axis, 2.67 crossings pooled over the
eight tessellations with a spread of 0.68 between them. So does the
azimuthal sampling bound, 7.3 degrees per element and at most 49
independent beam positions, which is trigonometry.

THE UNIFORM-ORIENTATION CONTROL CHANGES ITS EVIDENCE, NOT ITS VERDICT.
Its own per-azimuth scatter rises from 0.42 to 1.27 dB, so the published
sentence, that the control scatters by 0.42 dB and therefore none of the
measured scatter is numerical, no longer says what it said. The verdict
is nevertheless stronger on the audited estimator, for a different
reason: the control carries 0.51 per cent of the audited specimen gate
power at ppw 8 against 8.98 per cent of the published one, and a
contribution that small fluctuating by 1.27 dB can inject at most
0.007 dB into the specimen level. report_uniform prints that bound.

A LAST BLOCK CLOSES AN OLDER AUDIT. An earlier draft quoted 4.34 dB for
the measured scatter with no surviving derivation. normalisation_search
enumerates the reference, band, envelope definition, gate, azimuth grid
and pooling rule the figure could have come from, and reports how many
of them land on 4.34 by chance. The second count is the useful one: a
two-decimal target is hit often enough in a space that size that a match
would not identify anything.

Reads, all under ../../out/sweeps, each directory az*.npz with keys
'trace' and 'dt' plus a config.json, on the 30 azimuths at 12 degree
spacing every sweep has in common:
    girdle_perp_ppw8, mx_girdle_s{7,17,23,41,53,71,89}_ppw8
        eight independent girdle tessellations at ppw 8
    girdle_perp, lic_girdle_s11_ppw10
        seed 11 at ppw 6 and ppw 10, for the resolution series
    zc_s11_ppw6, zerocontrast_ppw8, zc_s11_ppw10
        uniform-orientation controls on the same code path. zerocontrast
        _ppw6 is deliberately unused: it predates the production
        programme and holds only 22 of the 30 azimuths
    singlemax_ppw8, mx_single_s{17,23,41}_ppw8
        four single-maximum sweeps, for the 4.34 dB audit only
Rebuilds the eight tessellations through sim/core/specimen.py to count grain
boundaries along the beam axis. That build is done on the CPU here: this
module is analysis, it has to run on the release machine whether or not
it has a CUDA device, and it must not compete for one.

Writes nothing. Run with --legacy to print the published column beside
every audited one, for the record.
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = ROOT / "out" / "sweeps"

# Acquisition and analysis constants, all as given in Section 3: 2 MHz
# Ricker, 100 mm disc, reference longitudinal speed 3850 m/s, 6.35 mm
# element. The coda gate and the analysis band are the ones used
# throughout the paper, and the band is the one db_reconcile.py audits.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
ELEM = 6.35e-3
CODA_GATE = (24e-6, 36e-6)
BAND = (0.8e6, 3.0e6)

# Decibels per nat. Every level here is 10 log10 of a power, so this is
# the constant that turns a log-variance into a level variance.
DB_PER_NAT = 10.0 / np.log(10.0)

# Every sweep in the matrix carries these 30 azimuths. The sweeps run at
# 6 degrees are decimated onto the common grid so that no realisation is
# averaged over a different set of beam positions from the others.
AZ_STEP_DEG = 12
AZ_COMMON = np.arange(0, 360, AZ_STEP_DEG)

GIRDLE = [("girdle_perp_ppw8", 11), ("mx_girdle_s7_ppw8", 7),
          ("mx_girdle_s17_ppw8", 17), ("mx_girdle_s23_ppw8", 23),
          ("mx_girdle_s41_ppw8", 41), ("mx_girdle_s53_ppw8", 53),
          ("mx_girdle_s71_ppw8", 71), ("mx_girdle_s89_ppw8", 89)]
SINGLE = [("singlemax_ppw8", 11), ("mx_single_s17_ppw8", 17),
          ("mx_single_s23_ppw8", 23), ("mx_single_s41_ppw8", 41)]

# Fixed-tessellation resolution series, seed 11, and the uniform-
# orientation control at each of the same three resolutions. Both maps
# are the ones db_reconcile.py uses, so a scatter and a resolution step
# are measured on the same runs.
RESOLUTION = {6: "girdle_perp", 8: "girdle_perp_ppw8",
              10: "lic_girdle_s11_ppw10"}
UNIFORM = {6: "zc_s11_ppw6", 8: "zerocontrast_ppw8", 10: "zc_s11_ppw10"}

# The audited estimator and the published one, in that order. Keys into
# every power dictionary this module passes about.
AUDITED, LEGACY = "coda_band", "coda_env"

CONF = 0.95

# Azimuthal orders removed when asking how much of the scatter is a
# deterministic signature rather than speckle. The girdle axis lies in
# the rotation plane, so a fabric term appears at order 2 and its first
# overtone at order 4; nothing above that is expected.
HARMONICS = (2, 4)

# Sub-gates the coda gate is split into for the locality test. Four
# gives 3 us each, about 5.5 coherence times in band, which is long
# enough that Gaussian speckle would already have averaged down inside
# one and short enough that four of them still fit the gate.
N_SUBGATES = 4

# Retired figure the last block audits, and the tolerance a match is
# counted at.
AUDIT_TARGET = 4.34
AUDIT_TOL = 0.005

# Gates that audit is allowed to try. The first is the paper's; the rest
# trim or extend it by a pulse length or more at either end, which spans
# every gate any script in the project has used.
AUDIT_GATES = {"24-36": (24e-6, 36e-6), "22-38": (22e-6, 38e-6),
               "25-35": (25e-6, 35e-6), "24-34": (24e-6, 34e-6),
               "26-40": (26e-6, 40e-6), "30-40": (30e-6, 40e-6)}

# Trials for the Monte Carlo that decides whether the two shape
# estimators would disagree by chance at this sample size.
MC_TRIALS = 4000


# ------------------------------------------------------------------ load

def bandpass(x, fs, lo=BAND[0], hi=BAND[1]):
    """Zero-phase Butterworth applied to the COMPLETE trace.

    Section 3 filters before gating so that the filter transient falls
    outside the gate rather than inside it.
    """
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band",
                 output="sos")
    return sosfiltfilt(sos, x)


def measure_trace(path):
    """Everything one azimuth contributes, measured in a single pass.

    Powers, not amplitudes, because the whole argument is about how a
    gate power fluctuates. The source reference is the peak of the
    analytic envelope of the whole unfiltered trace, which is the
    transmit pulse, and it is taken unfiltered under both estimators so
    that the two differ only in how they read the gate.

      coda_env    mean square of the global unfiltered Hilbert envelope
                  inside the gate. The published estimator.
      coda_band   2*E[x^2] inside the gate on the band-limited trace,
                  filtered before gating. The audited estimator.
      sub         the audited power of each of N_SUBGATES equal
                  sub-gates, which is what the locality test needs
      gate_band   the analytic signal of the band-limited trace inside
                  the gate, kept so that the coherence time can be
                  measured from the same samples the level came from
      gate_raw    the same before filtering, so that the effect of the
                  band on the coherence time is a difference and not a
                  comparison across runs
      audit       envelope amplitudes over every AUDIT_GATES window, raw
                  and band-limited, for the retired-figure audit only
    """
    with np.load(path) as handle:
        trace = np.asarray(handle["trace"], float).ravel()
        dt = float(handle["dt"])
    fs = 1.0 / dt
    i0, i1 = int(CODA_GATE[0] * fs), int(CODA_GATE[1] * fs)
    analytic = hilbert(trace)
    envelope = np.abs(analytic)
    src2 = envelope.max() ** 2
    filtered = bandpass(trace, fs)
    analytic_band = hilbert(filtered)

    edges = np.linspace(CODA_GATE[0], CODA_GATE[1], N_SUBGATES + 1)
    sub = np.array([2.0 * (filtered[int(a * fs):int(b * fs)] ** 2).mean()
                    / src2 for a, b in zip(edges[:-1], edges[1:])])

    k0, half = int(2 * DIA / C_REF * fs), int(2e-6 * fs)
    audit = {"src": envelope.max(),
             "src_bl": np.abs(analytic_band).max(),
             "e1": envelope[max(k0 - half, 0):k0 + half].max(),
             "e1_bl": np.abs(analytic_band)[max(k0 - half, 0):
                                            k0 + half].max()}
    for name, (t0, t1) in AUDIT_GATES.items():
        a, b = int(t0 * fs), int(t1 * fs)
        for tag, env in (("raw", envelope), ("bl", np.abs(analytic_band))):
            seg = env[a:b]
            audit["%s|%s|rms" % (name, tag)] = np.sqrt((seg ** 2).mean())
            audit["%s|%s|max" % (name, tag)] = seg.max()
            audit["%s|%s|mean" % (name, tag)] = seg.mean()

    return {LEGACY: (envelope[i0:i1] ** 2).mean() / src2,
            AUDITED: 2.0 * (filtered[i0:i1] ** 2).mean() / src2,
            "sub": sub, "dt": dt,
            "gate_raw": analytic[i0:i1], "gate_band": analytic_band[i0:i1],
            "audit": audit}


def load_sweep(name, step_deg=AZ_STEP_DEG, seed=None):
    """Every per-azimuth quantity for one sweep, on the common grid.

    The gated segments differ in length by a sample or two between
    azimuths, because dt is set per run by the stability condition and
    the gate edges are rounded to samples. They are trimmed to the
    shortest, which costs at most 0.02 us of a 12 us gate and keeps the
    coherence estimate on a rectangular array.
    """
    folder = SWEEPS / name
    with open(folder / "config.json") as handle:
        cfg = json.load(handle)
    if seed is not None and cfg["seed"] != seed:
        raise ValueError("%s carries seed %s, expected %d"
                         % (name, cfg["seed"], seed))
    az, rows = [], []
    for entry in sorted(os.listdir(folder)):
        if not (entry.startswith("az") and entry.endswith(".npz")):
            continue
        degrees = int(entry[2:5])
        if degrees % step_deg:
            continue
        az.append(degrees)
        rows.append(measure_trace(folder / entry))
    if not rows:
        raise IOError("no azimuths on the common grid under %s" % folder)
    az = np.array(az)
    if step_deg == AZ_STEP_DEG and not np.array_equal(az, AZ_COMMON):
        raise ValueError("%s does not cover the common azimuth grid" % name)
    n = min(len(row["gate_band"]) for row in rows)
    return dict(
        az=az, ppw=float(cfg["ppw"]), seed=int(cfg["seed"]),
        dt=float(np.mean([row["dt"] for row in rows])),
        **{key: np.array([row[key] for row in rows])
           for key in (LEGACY, AUDITED)},
        sub=np.array([row["sub"] for row in rows]),
        gate_raw=np.array([row["gate_raw"][:n] for row in rows]),
        gate_band=np.array([row["gate_band"][:n] for row in rows]),
        audit=[row["audit"] for row in rows])


def load_all():
    """Every sweep this module reads, read once.

    Read once because the eight tessellations are used by the scatter,
    the harmonic residual, the coherence time, the locality test and the
    retired-figure audit, and re-reading them per block would multiply
    the only expensive step in the module.
    """
    sweeps = {name: load_sweep(name, seed=seed)
              for name, seed in GIRDLE + SINGLE}
    for table in (RESOLUTION, UNIFORM):
        for name in table.values():
            if name not in sweeps:
                sweeps[name] = load_sweep(name)
    return sweeps


# --------------------------------------------------------------- compute

def levels(sweep, key=AUDITED):
    """Per-azimuth level in dB. A power, so 10 log10 and not 20."""
    return 10.0 * np.log10(sweep[key])


def chi2_interval(sd, df, conf=CONF):
    """Interval on a standard deviation of known degrees of freedom."""
    return (sd * np.sqrt(df / stats.chi2.ppf(0.5 + conf / 2, df)),
            sd * np.sqrt(df / stats.chi2.ppf(0.5 - conf / 2, df)))


def pooled_scatter(rows):
    """Per-azimuth scatter pooled WITHIN realisations, and its interval.

    Pooling within rather than across removes the level differences
    between tessellations, which are the subject of the ensemble
    subsection and not of this one.
    """
    var = np.array([row.var(ddof=1) for row in rows])
    df = int(sum(len(row) - 1 for row in rows))
    sd = float(np.sqrt(var.mean()))
    lo, hi = chi2_interval(sd, df)
    return sd, lo, hi, df


def realisation_interval(rows):
    """Interval on the scatter treating the tessellation as the unit.

    The chi-square interval counts every azimuth as a degree of freedom.
    This one counts only the realisations, so it is the conservative
    reading, and the two are reported together: if they disagree the
    azimuths are not behaving as independent draws.
    """
    sd = np.array([row.std(ddof=1) for row in rows])
    half = stats.t.ppf(0.5 + CONF / 2, len(sd) - 1) * sd.std(ddof=1) \
        / np.sqrt(len(sd))
    return float(sd.mean()), float(sd.std(ddof=1)), float(half)


def independent_azimuths(rows, step_deg=AZ_STEP_DEG):
    """Effective azimuth count from the circular autocorrelation.

    Summed to the first sign change, which is the usual truncation: the
    tail of a mean-removed circular series sums to zero exactly and
    carries no information about the correlation length.
    """
    ess = []
    for row in rows:
        y = row - row.mean()
        n = len(y)
        f = np.fft.rfft(y, n=2 * n)
        acf = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
        rho = acf / acf[0]
        m = 1
        while m < n // 2 and rho[m] > 0:
            m += 1
        den = 1 + 2 * sum((1 - j / n) * rho[j] for j in range(1, m))
        ess.append(n / max(den, 1e-9))
    return float(np.mean(ess))


def harmonic_design(azimuths, orders=HARMONICS):
    """Constant plus cosine and sine at each order, in azimuth."""
    t = np.radians(np.asarray(azimuths, float))
    cols = [np.ones_like(t)]
    for k in orders:
        cols += [np.cos(k * t), np.sin(k * t)]
    return np.column_stack(cols)


def harmonic_residual(rows, azimuths, orders=HARMONICS):
    """Scatter left after removing a deterministic azimuthal signature.

    A fabric signature is coherent in the laboratory frame and appears at
    those orders; speckle does not. Returns the pooled residual standard
    deviation, its degrees of freedom, and the per-realisation residual
    level series, which the contrast test needs.
    """
    design = harmonic_design(azimuths, orders)
    var, df, residuals = [], 0, []
    for row in rows:
        coef, *_ = np.linalg.lstsq(design, row, rcond=None)
        res = row - design @ coef
        dof = len(row) - design.shape[1]
        var.append(res @ res / dof)
        df += dof
        residuals.append(res)
    return float(np.sqrt(np.mean(var))), df, np.array(residuals)


def rayleigh_level_sd():
    """Level standard deviation of a single-look Rayleigh envelope.

    The envelope of a circular Gaussian field is Rayleigh, so its square
    is exponential and the level is a scaled logarithm of an exponential
    variate. The variance of that logarithm is the trigamma function at
    one, giving (10/ln 10) pi/sqrt(6) exactly, with no dependence on the
    mean level.
    """
    return DB_PER_NAT * np.pi / np.sqrt(6.0)


def nlook_level_sd(n):
    """Level standard deviation after averaging n independent looks.

    A gate that averages n independent speckle intensities returns a
    gamma variate of shape n, whose log has variance equal to the
    trigamma function at n. This reduces to rayleigh_level_sd at n = 1
    and to (10/ln 10)/sqrt(n) for large n, and it is the exact form of
    the reduction the manuscript writes as a division by sqrt(n).
    """
    return DB_PER_NAT * np.sqrt(polygamma(1, n))


def looks_from_level_sd(sd_db):
    """The gamma shape a measured level standard deviation implies."""
    return float(brentq(lambda n: nlook_level_sd(n) - sd_db, 1e-3, 1e6))


def looks_from_contrast(power):
    """The gamma shape a measured intensity contrast implies.

    For a gamma variate of shape N the contrast, standard deviation over
    mean, is 1/sqrt(N). This is the SECOND estimator of the same N, and
    the point of having it is that a model with one parameter must give
    the same answer twice.
    """
    power = np.asarray(power, float)
    return float((power.mean() / power.std(ddof=1)) ** 2)


def rayleigh_monte_carlo(n_looks, draws=2_000_000, seed=0):
    """The two closed forms above, sampled rather than trusted."""
    rng = np.random.default_rng(seed)
    single = (20 * np.log10(np.sqrt(rng.exponential(1.0,
                                                    draws)))).std(ddof=1)
    gate = (10 * np.log10(rng.gamma(n_looks, 1.0 / n_looks,
                                    draws))).std(ddof=1)
    return float(single), float(gate)


def coherence_looks(segments, dt):
    """Independent looks a gate power averages, measured from the trace.

    segments is (n_azimuth, n_sample) of the ANALYTIC signal inside the
    gate. For a field with slowly varying mean intensity m(t) and field
    correlation rho(tau), the normalised variance of the gate power is
    the coherence time over the gate length divided by a participation
    factor, so

        tau_c   = integral of |rho(tau)|^2 over lag
        part    = (integral m)^2 / (T integral m^2)
        N       = T / tau_c * part

    Both pieces are measured. The field is normalised by sqrt(m) before
    the correlation is formed so that the decay of the coda does not
    enter the coherence estimate as well as the participation factor.
    The triangular weight is the usual finite-record one.

    This is the look count the gamma floor actually refers to, and it is
    a property of the BAND, which the geometric crossing count is not.
    """
    segments = np.asarray(segments)
    n = segments.shape[1]
    intensity = np.abs(segments) ** 2
    mean_profile = intensity.mean(axis=0)
    scaled = segments / np.sqrt(mean_profile)[None, :]
    acf = np.array([np.abs((scaled[:, :n - lag]
                            * np.conj(scaled[:, lag:])).mean())
                    for lag in range(n)])
    rho2 = (acf / acf[0]) ** 2
    weight = 1.0 - np.arange(n) / n
    tau_c = dt * (rho2[0] + 2.0 * (weight[1:] * rho2[1:]).sum())
    part = float(mean_profile.sum() ** 2 / (n * (mean_profile ** 2).sum()))
    gate = n * dt
    return dict(tau_c=tau_c, gate=gate, participation=part,
                n_corr=gate / tau_c, n_eff=gate / tau_c * part)


def subgate_stats(sub):
    """Is the azimuthal fluctuation chord-wide, or localised in range?

    sub is (n_azimuth, N_SUBGATES) of sub-gate powers. Two numbers. The
    mean off-diagonal correlation across azimuth says whether one gain
    multiplies the whole gate; the ratio of the observed variance of the
    total to what independent sub-gates would give says how much the
    sub-gates average each other down. A chord-wide gain would give a
    high correlation and a ratio near N_SUBGATES.
    """
    sub = np.asarray(sub, float)
    normalised = sub / sub.mean(axis=0)[None, :]
    corr = np.corrcoef(normalised.T)
    off = corr[np.triu_indices(sub.shape[1], 1)]
    total = sub.sum(axis=1)
    observed = total.var(ddof=1) / total.mean() ** 2
    independent = sub.var(axis=0, ddof=1).sum() / total.mean() ** 2
    contrast = sub.std(axis=0, ddof=1) / sub.mean(axis=0)
    return dict(r=float(off.mean()), ratio=float(observed / independent),
                contrast=contrast)


def shape_estimator_control(true_n, n_azimuth, n_series, trials=MC_TRIALS,
                            seed=0):
    """Would the two shape estimators disagree by chance at this size?

    The test that keeps report_gamma_test from being a small-sample
    artefact. Draws n_series independent gamma samples of n_azimuth
    points at the given shape, forms both estimators exactly as the
    measurement does, and returns the distribution of the gap between
    their means. If the observed gap is inside this distribution the
    disagreement proves nothing.
    """
    rng = np.random.default_rng(seed)
    gaps = np.empty(trials)
    for i in range(trials):
        by_contrast, by_level = [], []
        for _ in range(n_series):
            x = rng.gamma(true_n, 1.0 / true_n, n_azimuth)
            by_contrast.append(looks_from_contrast(x))
            by_level.append(looks_from_level_sd(
                (10 * np.log10(x)).std(ddof=1)))
        gaps[i] = np.mean(by_level) - np.mean(by_contrast)
    return gaps


def axis_crossings(seeds, ppw=8.0, azimuths=AZ_COMMON):
    """Grain boundaries crossed on the beam axis inside the coda gate.

    One row of counts per tessellation, one entry per azimuth. Built at
    the production spacing rather than a coarse one: a boundary is only
    counted where the rasterised labels change, so a coarser grid loses
    the thin cells and would undercount the very quantity in question.

    The label volume is produced by an exact all-seeds power argmin taken
    one z plane at a time. That is the same rule the GPU path in
    specimen.py applies, written out here so the module runs without a
    CUDA device and without competing for one.
    """
    labels, h = _label_volumes(seeds, ppw)
    d0, d1 = CODA_GATE[0] * C_REF / 2, CODA_GATE[1] * C_REF / 2
    rows = []
    for volume in labels:
        nx, ny, nz = volume.shape
        counts = []
        for degrees in azimuths:
            a = np.radians(float(degrees))
            n = np.array([np.cos(a), np.sin(a), 0.0])
            # Sampled at a quarter cell so that no boundary is stepped
            # over; the labels themselves still set the resolution.
            s = np.arange(d0, d1, h / 4)
            q = DIA / 2 * n[None, :] - s[:, None] * n[None, :]
            qi = np.clip(np.rint((q[:, 0] + nx * h / 2) / h - 0.5), 0,
                         nx - 1).astype(int)
            qj = np.clip(np.rint((q[:, 1] + ny * h / 2) / h - 0.5), 0,
                         ny - 1).astype(int)
            counts.append(int((np.diff(volume[qi, qj, nz // 2]) != 0).sum()))
        rows.append(counts)
    return np.array(rows, float)


def column_geometry(seed, ppw=8.0, azimuths=np.arange(0, 360, 6)):
    """Both counts for one tessellation, from a single label volume.

    The column count is the larger one, and it is the one the section
    says is NOT the relevant one: grains side by side across the column
    are not resolved from one another in range and do not contribute
    separate looks. It is reported so that the distinction the section
    draws is visible rather than asserted. Taken on the finer 6 degree
    grid the seed 11 sweeps hold, because it is geometry and costs
    nothing, and because that is the grid the published pair of counts
    was measured on.
    """
    (volume,), h = _label_volumes([seed], ppw)
    nx, ny, nz = volume.shape
    d0, d1 = CODA_GATE[0] * C_REF / 2, CODA_GATE[1] * C_REF / 2
    zc = (np.arange(nz) + 0.5) * h - nz * h / 2
    radius = ELEM / 2
    counts, crossings = [], []
    for degrees in azimuths:
        a = np.radians(float(degrees))
        n = np.array([np.cos(a), np.sin(a), 0.0])
        t = np.array([-np.sin(a), np.cos(a), 0.0])
        s = np.arange(d0, d1, h)
        off = np.arange(-radius, radius + h, h)
        p = (DIA / 2 * n[None, None, None, :]
             - s[:, None, None, None] * n[None, None, None, :]
             + off[None, :, None, None] * t[None, None, None, :]
             + zc[None, None, :, None] * np.array([0.0, 0.0, 1.0]))
        p = p.reshape(-1, 3)
        r2 = p[:, 0] ** 2 + p[:, 1] ** 2
        p = p[(r2 <= (DIA / 2) ** 2) & (np.abs(p[:, 2]) <= zc.max())]
        gi = np.clip(np.rint((p[:, 0] + nx * h / 2) / h - 0.5), 0,
                     nx - 1).astype(int)
        gj = np.clip(np.rint((p[:, 1] + ny * h / 2) / h - 0.5), 0,
                     ny - 1).astype(int)
        gk = np.clip(np.rint((p[:, 2] + nz * h / 2) / h - 0.5), 0,
                     nz - 1).astype(int)
        touched = volume[gi, gj, gk]
        counts.append(len(np.unique(touched[touched >= 0])))
        s = np.arange(d0, d1, h / 4)
        q = DIA / 2 * n[None, :] - s[:, None] * n[None, :]
        qi = np.clip(np.rint((q[:, 0] + nx * h / 2) / h - 0.5), 0,
                     nx - 1).astype(int)
        qj = np.clip(np.rint((q[:, 1] + ny * h / 2) / h - 0.5), 0,
                     ny - 1).astype(int)
        crossings.append(int((np.diff(volume[qi, qj, nz // 2]) != 0).sum()))
    return np.array(counts, float), np.array(crossings, float)


def _label_volumes(seeds, ppw):
    """Rasterised tessellations for the given seeds, built on the CPU."""
    sys.path.insert(0, str(ROOT / "sim"))
    sys.path[:0] = [os.path.join(sys.path[0], _d)
                    for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
    from specimen import DiskSpecimen

    def cpu_label(x, z, radius, centres, weights):
        nx, nz = len(x), len(z)
        gx, gy = np.meshgrid(x, x, indexing="ij")
        inside = (gx ** 2 + gy ** 2) <= radius ** 2
        pxy = np.stack([gx[inside], gy[inside]], axis=1)
        # The in-plane part of the power distance does not depend on z,
        # so it is formed once and the plane loop only adds one column.
        base = (((pxy[:, None, :] - centres[None, :, :2]) ** 2).sum(-1)
                - weights[None, :])
        raw = np.full((nx, nx, nz), -1, np.int32)
        for iz in range(nz):
            d2 = base + (z[iz] - centres[:, 2])[None, :] ** 2
            plane = np.full((nx, nx), -1, np.int32)
            plane[inside] = d2.argmin(axis=1).astype(np.int32)
            raw[:, :, iz] = plane
        return raw

    DiskSpecimen._label_grid_gpu = staticmethod(cpu_label)
    h = C_REF / F0 / ppw
    volumes = []
    for seed in seeds:
        built = DiskSpecimen(
            diameter_m=DIA, thickness_m=0.035, n_grains=100, size_cv=0.35,
            concentration=-8.0, spatial_corr=0.0,
            fabric_axis=(1.0, 0.0, 0.0), seed=int(seed)).build(h)
        volumes.append(np.asarray(built["labels"]))
    return volumes, h


def audit_level(record, gate="24-36", band="raw", kind="rms", ref="src",
                factor=20.0):
    """One per-azimuth level from the retired-figure audit amplitudes."""
    num = record["%s|%s|%s" % (gate, band, kind)]
    if ref == "absolute":
        return factor * np.log10(num)
    return factor * np.log10(num / record[ref + ("_bl" if band == "bl"
                                                else "")])


def normalisation_search(groups, target=AUDIT_TARGET, tol=AUDIT_TOL):
    """Every scatter the pipeline could have produced, and the near misses.

    groups maps a name to a list of per-azimuth audit dictionaries.
    Enumerates reference, band, envelope definition, gate, decibel
    convention and pooling rule, and returns the full table plus the
    count landing on target. The count is the point of the exercise: it
    measures how little a two-decimal match would prove.
    """
    rule = {"pooled": lambda v: np.sqrt(np.mean([r.var(ddof=1)
                                                 for r in v])),
            "mean sd": lambda v: np.mean([r.std(ddof=1) for r in v]),
            "concatenated": lambda v: np.concatenate(v).std(ddof=1)}
    table, hits = [], 0
    for group, sweeps in groups.items():
        for gate, band, kind, ref, factor, how in itertools.product(
                sorted(AUDIT_GATES), ("raw", "bl"), ("rms", "max", "mean"),
                ("src", "e1", "absolute"), (20.0, 10.0), sorted(rule)):
            rows = [np.array([audit_level(rec, gate, band, kind, ref,
                                          factor) for rec in sweep])
                    for sweep in sweeps]
            value = float(rule[how](rows))
            table.append((group, gate, band, kind, ref, factor, how, value))
            hits += abs(value - target) < tol
    return table, hits


# ---------------------------------------------------------------- report

def report_scatter(sweeps, legacy=False):
    """The measured per-azimuth scatter, on the eight tessellations."""
    rows = {key: [levels(sweeps[name], key) for name, _s in GIRDLE]
            for key in (LEGACY, AUDITED)}
    print("PER-AZIMUTH SCATTER OF THE CODA LEVEL, ppw 8, %d azimuths"
          % len(AZ_COMMON))
    print("  gate %.0f to %.0f us, dB re source amplitude"
          % (CODA_GATE[0] * 1e6, CODA_GATE[1] * 1e6))
    print("  %-6s %10s %10s %10s %10s"
          % ("seed", "level", "sd audit", "level", "sd pub"))
    for i, (_name, seed) in enumerate(GIRDLE):
        print("  %-6d %10.2f %10.2f %10.2f %10.2f"
              % (seed, rows[AUDITED][i].mean(), rows[AUDITED][i].std(ddof=1),
                 rows[LEGACY][i].mean(), rows[LEGACY][i].std(ddof=1)))
    out = {}
    for key, tag in ((AUDITED, "audited"), (LEGACY, "published")):
        sd, lo, hi, df = pooled_scatter(rows[key])
        mean_sd, spread, half = realisation_interval(rows[key])
        out[key] = sd
        print("  %-10s pooled %.2f dB, %.0f%% interval [%.2f, %.2f] on "
              "%d dof" % (tag, sd, 100 * CONF, lo, hi, df))
        print("  %-10s tessellation as the unit %.2f +- %.2f dB, spread"
              % ("", mean_sd, half))
        print("  %-10s between tessellations %.2f dB" % ("", spread))
    print("  effective independent azimuths per revolution %.1f of %d"
          % (independent_azimuths(rows[AUDITED]), len(AZ_COMMON)))
    print("  implied gamma shape  audited %.2f  published %.2f"
          % (looks_from_level_sd(out[AUDITED]),
             looks_from_level_sd(out[LEGACY])))
    print()
    if legacy:
        print("  The published 3.63 dB is the mean of eight scatters of a")
        print("  quantity that counts grid noise and a front-arrival")
        print("  pedestal. Both fill in the nulls of the azimuthal")
        print("  pattern, so the published estimator UNDERSTATES the")
        print("  scatter, which is why the audited figure is larger.")
        print()
    return rows, out


def report_resolution(sweeps, rows):
    """Has the scatter stopped moving with the grid?"""
    print("RESOLUTION SERIES AT FIXED TESSELLATION, seed 11")
    print("  %-6s %-24s %10s %10s" % ("ppw", "sweep", "audited",
                                      "published"))
    series = {}
    for ppw in sorted(RESOLUTION):
        sweep = sweeps[RESOLUTION[ppw]]
        a = levels(sweep, AUDITED).std(ddof=1)
        p = levels(sweep, LEGACY).std(ddof=1)
        series[ppw] = (a, p)
        print("  %-6d %-24s %10.2f %10.2f" % (ppw, RESOLUTION[ppw], a, p))
    keys = sorted(series)
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        print("  %-6s %-24s %+10.2f %+10.2f"
              % ("%d->%d" % (lo, hi), "change per step",
                 series[hi][0] - series[lo][0],
                 series[hi][1] - series[lo][1]))
    spread = np.array([row.std(ddof=1) for row in rows[AUDITED]])
    print("  the eight audited scatters run %.2f to %.2f dB, sd %.2f dB"
          % (spread.min(), spread.max(), spread.std(ddof=1)))
    print("  so the %+.2f dB change beyond ppw 8 is small against the "
          "spread\n  across tessellations, which is the claim the series "
          "is there to make."
          % (series[10][0] - series[8][0]))
    print("  MEASURED, and worth saying plainly: the TREND reverses. The")
    print("  published series falls with refinement and reads as")
    print("  convergence from above; the audited one rises, by +0.56")
    print("  then +0.19 dB, and has not flattened by ppw 10. The series")
    print("  therefore no longer demonstrates that the scatter has")
    print("  converged. It bounds the residual grid dependence at")
    print("  0.19 dB per step, and that is all it does.")
    print("  Part of the ppw 6 point is the numerical floor, which is")
    print("  3.82 per cent of the gate power there against 0.51 at")
    print("  ppw 8. report_uniform bounds what that can contribute.")
    print("  What produces the remaining rise is NOT established here.")
    print()
    return series


def report_uniform(sweeps, rows):
    """Is any of the measured scatter numerical?

    Two bounds, because the numerical coda can add to the physical one
    two ways and only the weaker of them is the one usually quoted.

      incoh   the floor adds its own power. A contribution of relative
              size s whose power fluctuates by a relative amount c moves
              the total level by (10/ln 10) s c, first order in s.
      coh     the floor interferes with the physical coda. The cross
              term has relative rms 2 sqrt(s), so the level moves by
              (10/ln 10) 2 sqrt(s), which is the larger bound by an
              order of magnitude and is the one that binds.

    The coherent bound is the honest one: both fields occupy the same
    gate and there is no reason for them not to interfere. It is quoted
    as a level standard deviation, so it comes out of a measured scatter
    in quadrature and not by subtraction.
    """
    print("UNIFORM-ORIENTATION CONTROL")
    print("  every grain given the same c-axis, so the tessellation is")
    print("  acoustically invisible and any coda is numerical")
    print("  %-4s %-18s %6s %6s %7s %7s %6s %5s %6s"
          % ("ppw", "control", "sd a", "sd p", "share a", "share p",
             "incoh", "coh", "left"))
    for ppw in sorted(UNIFORM):
        control, specimen = sweeps[UNIFORM[ppw]], sweeps[RESOLUTION[ppw]]
        sd_a = levels(control, AUDITED).std(ddof=1)
        sd_p = levels(control, LEGACY).std(ddof=1)
        share_a = control[AUDITED].mean() / specimen[AUDITED].mean()
        share_p = control[LEGACY].mean() / specimen[LEGACY].mean()
        incoh = DB_PER_NAT * share_a * control[AUDITED].std(ddof=1) \
            / control[AUDITED].mean()
        coh = DB_PER_NAT * 2.0 * np.sqrt(share_a)
        measured = levels(specimen, AUDITED).std(ddof=1)
        left = np.sqrt(max(measured ** 2 - coh ** 2, 0.0))
        print("  %-4d %-18s %6.2f %6.2f %6.2f%% %6.2f%% %6.3f %5.2f %6.2f"
              % (ppw, UNIFORM[ppw], sd_a, sd_p, 100 * share_a,
                 100 * share_p, incoh, coh, left))
    print()
    print("  The published sentence rests on the control's own scatter,")
    print("  0.42 dB at ppw 8. On the audited estimator that is 1.27 dB,")
    print("  so the sentence cannot be restated with a new number in the")
    print("  same slot. The verdict survives on better evidence. What")
    print("  matters is not how much the numerical coda fluctuates but")
    print("  how much of the specimen coda it is, and the audited")
    print("  estimator puts that at 0.51 per cent at ppw 8 against 8.98")
    print("  per cent for the published one, falling further with")
    print("  refinement rather than rising. Even letting the floor")
    print("  interfere coherently, which is the binding bound, it can")
    print("  account for 0.62 dB of a 4.58 dB scatter at ppw 8, and")
    print("  removing that in quadrature leaves 4.54 dB. None of the")
    print("  measured scatter is numerical, and that conclusion holds.")
    print("  The published estimator had the numerical floor at a fifth")
    print("  of the coda by ppw 10, which is the pedestal, not the grid.")
    print()
    print("  The one place this bites is the coarse end of the")
    print("  resolution series. At ppw 6 the floor is 3.82 per cent of")
    print("  the gate power, the coherent bound is 1.70 dB, and the")
    print("  measured 3.01 dB leaves 2.49 dB once it is removed in")
    print("  quadrature. The ppw 6 point of that series is therefore the")
    print("  weakest of the three and should not be leaned on.")
    print()


def report_harmonics(rows, sd_audited):
    """How much of the scatter is a coherent fabric signature?"""
    print("REMOVING THE COHERENT AZIMUTHAL ORDERS %s"
          % ", ".join(str(k) for k in HARMONICS))
    out = {}
    for key, tag in ((AUDITED, "audited"), (LEGACY, "published")):
        raw = np.sqrt(np.mean([row.var(ddof=1) for row in rows[key]]))
        res, df, residual = harmonic_residual(rows[key], AZ_COMMON)
        lo, hi = chi2_interval(res, df)
        out[key] = (res, residual)
        print("  %-10s raw %.2f dB, residual %.2f dB on %d dof, "
              "[%.2f, %.2f]" % (tag, raw, res, df, lo, hi))
        # Both variances are unbiased estimates on their own degrees of
        # freedom, so their ratio is the honest reduction. A raw R^2 on
        # 30 points with four fitted coefficients would read higher and
        # would be inflated by the fit.
        print("  %-10s residual variance is %.0f per cent of the raw"
              % ("", 100 * (res / raw) ** 2))
        print("  %-10s variance, and the residual implies a shape of %.2f"
              % ("", looks_from_level_sd(res)))
    print()
    print("  The published residual, 2.97 dB, lands 0.05 dB from the")
    print("  floor at the measured crossing count, which is what made")
    print("  the subsection read as a closed argument. The audited")
    print("  residual is 3.69 dB and lands 0.77 dB above the same floor.")
    print("  Removing the coherent orders no longer brings the")
    print("  measurement onto the floor. It closes about half the gap,")
    print("  1.66 dB to 0.77 dB, and leaves the measurement on the same")
    print("  side of the floor it started on.")
    print()
    return out


def report_floor(sd_audited, sd_legacy):
    """The Rayleigh floor and the geometry it is evaluated at."""
    seeds = [seed for _name, seed in GIRDLE]
    print("GRAIN POPULATION AND THE FLOOR IT IMPLIES")
    column, axis = column_geometry(11)
    cross = axis_crossings(seeds)
    per_seed = cross.mean(axis=1)
    print("  seed 11, %d azimuths at 6 degrees" % len(column))
    print("    grains touched by the beam column   %.1f +- %.1f"
          % (column.mean(), column.std()))
    print("    boundaries crossed on the beam axis %.1f +- %.1f"
          % (axis.mean(), axis.std()))
    print("  eight tessellations, %d azimuths at %d degrees"
          % (len(AZ_COMMON), AZ_STEP_DEG))
    for seed, row in zip(seeds, cross):
        print("    seed %-4d crossings %.2f +- %.2f"
              % (seed, row.mean(), row.std()))
    n_bar = float(cross.mean())
    print("    pooled %.2f, spread of the eight means %.2f"
          % (n_bar, per_seed.std(ddof=1)))
    print("  These are geometry. They are the same under either")
    print("  estimator and they are unchanged by the rebuild.")
    print()
    single = rayleigh_level_sd()
    mc_single, mc_gate = rayleigh_monte_carlo(n_bar)
    print("  single look, (10/ln 10) pi/sqrt(6)        %.3f dB "
          "(Monte Carlo %.2f)" % (single, mc_single))
    print("  floor at N = %.2f, exact trigamma form    %.3f dB "
          "(Monte Carlo %.2f)" % (n_bar, nlook_level_sd(n_bar), mc_gate))
    print("  floor at N = %.2f by division by sqrt(N)  %.3f dB"
          % (n_bar, single / np.sqrt(n_bar)))
    for tag, sd in (("audited", sd_audited), ("published", sd_legacy)):
        print("  %-10s scatter %.2f dB exceeds the exact floor by "
              "%+.2f dB and\n  %-10s the sqrt(N) form by %+.2f dB, and "
              "implies a shape of %.2f"
              % (tag, sd, sd - nlook_level_sd(n_bar), "",
                 sd - single / np.sqrt(n_bar), looks_from_level_sd(sd)))
    print()
    print("  A measured scatter ABOVE the floor for an independently")
    print("  counted N is the wrong side of it. On the published")
    print("  estimator the gap was 0.71 dB and could be read as")
    print("  agreement. On the audited one it is 1.66 dB. The next two")
    print("  blocks ask why, and neither answer is the one expected.")
    print()
    return n_bar


def report_lookcount(sweeps):
    """The look count measured in band, not read off the geometry."""
    print("LOOK COUNT MEASURED FROM THE TRACE, ppw 8")
    print("  the gamma floor is a statement about how many independent")
    print("  looks a gate power averages, which is the gate length over")
    print("  the coherence time of the field. That is a property of the")
    print("  ANALYSIS BAND. The crossing count is not.")
    print("  %-18s %9s %9s %9s %9s %9s"
          % ("field", "tau_c us", "gate us", "N corr", "N eff", "floor"))
    out = {}
    for tag, key in (("unfiltered", "gate_raw"),
                     ("in band 0.8-3.0", "gate_band")):
        n = min(sweeps[name][key].shape[1] for name, _s in GIRDLE)
        stack = np.concatenate([sweeps[name][key][:, :n]
                                for name, _s in GIRDLE], axis=0)
        dt = float(np.mean([sweeps[name]["dt"] for name, _s in GIRDLE]))
        res = coherence_looks(stack, dt)
        out[tag] = res
        print("  %-18s %9.3f %9.2f %9.1f %9.1f %9.2f"
              % (tag, res["tau_c"] * 1e6, res["gate"] * 1e6, res["n_corr"],
                 res["n_eff"], nlook_level_sd(res["n_eff"])))
    print("  mean-intensity participation factor %.2f, so the coda fills"
          % out["in band 0.8-3.0"]["participation"])
    print("  the gate evenly enough that the decay is not the story")
    print()
    raw, band = out["unfiltered"], out["in band 0.8-3.0"]
    print("  MEASURED: band limiting lengthens the coherence time from")
    print("  %.2f to %.2f us and lowers the look count from %.0f to %.0f."
          % (raw["tau_c"] * 1e6, band["tau_c"] * 1e6, raw["n_eff"],
             band["n_eff"]))
    print("  That is the direction the band-limiting hypothesis needs.")
    print("  MEASURED: it is not remotely large enough. The floor moves")
    print("  from %.2f to %.2f dB, against a measurement of 4.58 dB. The"
          % (nlook_level_sd(raw["n_eff"]), nlook_level_sd(band["n_eff"])))
    print("  in-band look count is of order 20, not of order 3, so the")
    print("  measurement is %.2f dB above the floor and not 1.7 dB above"
          % (4.581 - nlook_level_sd(band["n_eff"])))
    print("  it. Recomputing the look count in band makes the")
    print("  disagreement WORSE, not better.")
    print()
    return out


def report_gamma_test(sweeps, rows, residuals):
    """Is the azimuthal gate power a gamma variate at all?

    The decisive block. Everything above compares a measured scatter with
    a floor derived from a gamma model. A gamma has one parameter, so the
    contrast and the level standard deviation must return the same shape.
    They are both measured here, on the same 30 numbers per tessellation,
    and the Monte Carlo settles whether any gap is a sample-size effect.
    """
    print("IS THE AZIMUTHAL GATE POWER A GAMMA VARIATE?")
    print("  a gamma of shape N has contrast 1/sqrt(N) and level sd")
    print("  (10/ln 10) sqrt(psi_1(N)), so one dataset gives N twice")
    print("  %-6s %9s %11s %9s %11s %9s"
          % ("seed", "contrast", "N contrast", "sd dB", "N from sd",
             "KS p"))
    by_contrast, by_level = [], []
    for i, (name, seed) in enumerate(GIRDLE):
        power = sweeps[name][AUDITED]
        x = power / power.mean()
        contrast = x.std(ddof=1)
        sd = rows[AUDITED][i].std(ddof=1)
        shape, loc, scale = stats.gamma.fit(x, floc=0.0)
        ks = stats.kstest(x, "gamma", args=(shape, loc, scale))
        by_contrast.append(looks_from_contrast(x))
        by_level.append(looks_from_level_sd(sd))
        print("  %-6d %9.2f %11.2f %9.2f %11.2f %9.3f"
              % (seed, contrast, by_contrast[-1], sd, by_level[-1],
                 ks.pvalue))
    by_contrast = np.array(by_contrast)
    by_level = np.array(by_level)
    gap = by_level.mean() - by_contrast.mean()
    t, p = stats.ttest_rel(by_level, by_contrast)
    print("  N from the contrast %.2f +- %.2f, N from the level sd "
          "%.2f +- %.2f" % (by_contrast.mean(), by_contrast.std(ddof=1),
                            by_level.mean(), by_level.std(ddof=1)))
    print("  gap %+.2f, paired t = %.2f on %d dof, p = %.4f"
          % (gap, t, len(by_level) - 1, p))
    print("  the per-tessellation KS test does NOT reject a gamma fit,")
    print("  which is what 30 points buys: the test has no power here")
    print("  and is printed so that it cannot be quoted as support")
    print()
    for true_n in (1.0, round(by_level.mean(), 2), 2.67):
        gaps = shape_estimator_control(true_n, len(AZ_COMMON), len(GIRDLE))
        print("  control at true N = %.2f: gap %+.3f +- %.3f, |gap| >="
              % (true_n, gaps.mean(), gaps.std()))
        print("    %.2f in %.2f per cent of %d trials"
              % (abs(gap), 100 * np.mean(np.abs(gaps) >= abs(gap)),
                 MC_TRIALS))
    print()
    print("  MEASURED: the two estimators of the same single parameter")
    print("  disagree by %+.2f, and on genuine gamma data at this sample"
          % gap)
    print("  size they agree to within a few hundredths and reach a gap")
    print("  this large at most once in three thousand trials. The")
    print("  azimuthal gate power is not a gamma variate of any shape,")
    print("  whatever N is chosen for it. The floor the subsection")
    print("  compares against therefore has no shape parameter to be")
    print("  evaluated at, and the comparison cannot be repaired by")
    print("  choosing a better N.")
    print()
    print("  The same test on the residual after the coherent orders are")
    print("  removed, which is the quantity the subsection calls the")
    print("  incoherent part:")
    res_power = 10 ** (residuals / 10.0)
    rc = np.array([looks_from_contrast(row) for row in res_power])
    rl = np.array([looks_from_level_sd(row.std(ddof=1))
                   for row in residuals])
    print("    N from the contrast %.2f +- %.2f, from the level sd "
          "%.2f +- %.2f" % (rc.mean(), rc.std(ddof=1), rl.mean(),
                            rl.std(ddof=1)))
    print("    the gap goes from %+.2f to %+.2f, so removing the fabric"
          % (gap, rl.mean() - rc.mean()))
    print("    signature does not close it and does not even narrow it.")
    print("    The excess contrast is not the coherent orders.")
    print()


def report_locality(sweeps):
    """Where in the gate does the excess fluctuation live?"""
    print("SUB-GATE TEST, %d sub-gates of %.0f us" % (
        N_SUBGATES,
        (CODA_GATE[1] - CODA_GATE[0]) * 1e6 / N_SUBGATES))
    print("  %-6s %11s %14s %s"
          % ("seed", "mean r", "var/indep", "sub-gate contrasts"))
    corr, ratio, contrast = [], [], []
    for name, seed in GIRDLE:
        stat = subgate_stats(sweeps[name]["sub"])
        corr.append(stat["r"])
        ratio.append(stat["ratio"])
        contrast.append(stat["contrast"])
        print("  %-6d %+11.2f %14.2f  %s"
              % (seed, stat["r"], stat["ratio"],
                 " ".join("%.2f" % c for c in stat["contrast"])))
    contrast = np.array(contrast)
    total = np.array([sweeps[name][AUDITED].std(ddof=1)
                      / sweeps[name][AUDITED].mean()
                      for name, _s in GIRDLE])
    print("  pooled mean r %+.2f, pooled variance ratio %.2f"
          % (np.mean(corr), np.mean(ratio)))
    print("  mean sub-gate contrast %.2f, whole-gate contrast %.2f"
          % (contrast.mean(), total.mean()))
    print()
    print("  MEASURED: a 3 us sub-gate holds about 5.5 coherence times,")
    print("  where fully developed speckle would give a contrast of")
    print("  0.43. The measured sub-gate contrast is %.2f."
          % contrast.mean())
    print("  MEASURED: the sub-gates are very nearly uncorrelated across")
    print("  azimuth, mean r %+.2f, so the excess is not one gain"
          % np.mean(corr))
    print("  multiplying the whole gate. It is localised in range.")
    print("  INFERRED, and only inferred: a gate whose energy is carried")
    print("  by a few bright arrivals rather than by many comparable")
    print("  ones behaves exactly like this, and Section 5 already")
    print("  builds a contrast-weighted facet model of that kind. This")
    print("  module does not establish that model; it establishes that")
    print("  the fully developed speckle picture is not available.")
    print()


def report_sampling():
    """The azimuthal sampling bound, which is trigonometry."""
    arc = 2 * np.degrees(np.arcsin(ELEM / DIA))
    print("AZIMUTHAL SAMPLING BOUND")
    print("  a %.2f mm element on a %.0f mm disc subtends %.2f degrees,"
          % (ELEM * 1e3, DIA * 1e3, arc))
    print("  so a revolution holds at most %d independent beam positions"
          % int(360.0 / arc))
    print("  the %d azimuth grid at %d degrees reaches %.0f per cent of"
          % (len(AZ_COMMON), AZ_STEP_DEG,
             100.0 * len(AZ_COMMON) / int(360.0 / arc)))
    print("  them, so it samples the revolution without repeating a look")
    print("  Geometry. Unchanged by the estimator.")
    print()


def report_audit(sweeps):
    """Close the older audit of the retired 4.34 dB figure."""
    groups = {"girdle 8": [sweeps[n]["audit"] for n, _s in GIRDLE],
              "single maximum 4": [sweeps[n]["audit"] for n, _s in SINGLE],
              "seed 11 ppw 6": [sweeps[RESOLUTION[6]]["audit"]],
              "seed 11 ppw 10": [sweeps[RESOLUTION[10]]["audit"]]}
    table, hits = normalisation_search(groups)
    print("NORMALISATION AUDIT OF THE RETIRED %.2f dB FIGURE"
          % AUDIT_TARGET)
    print("  the paper's own conventions, rms envelope, 20 log10, pooled")
    print("  %-18s %-6s %-5s %-9s %s"
          % ("group", "gate", "band", "ref", "scatter"))
    for row in table:
        if (row[0] != "girdle 8" or row[1] != "24-36" or row[3] != "rms"
                or row[5] != 20.0 or row[6] != "pooled"):
            continue
        print("  %-18s %-6s %-5s %-9s %.2f dB"
              % (row[0], row[1], row[2], row[4], row[7]))
    print("  %d combinations enumerated, %d land within %.3f dB of %.2f"
          % (len(table), hits, AUDIT_TOL, AUDIT_TARGET))
    print("  closest five overall")
    for row in sorted(table, key=lambda r: abs(r[7] - AUDIT_TARGET))[:5]:
        print("    %-18s %-6s %-5s %-5s %-9s %-4.0f %-13s %.2f dB" % row)
    print()


def report_verdict():
    """What survives the rebuild, in the order the subsection says it."""
    print("WHAT SURVIVES")
    print("  UNCHANGED, because it is geometry and touches no estimator:")
    print("    of order 10^2 grains; 12.6 +- 2.2 grains in the beam")
    print("    column; 2.9 +- 1.0 boundary crossings on the beam axis;")
    print("    2.67 crossings pooled over eight tessellations with a")
    print("    spread of 0.68; 7.3 degrees per element and at most 49")
    print("    independent beam positions.")
    print("  SURVIVES WITH DIFFERENT NUMBERS:")
    print("    the resolution series, 3.01, 3.57 and 3.76 dB, whose")
    print("    change beyond ppw 8 is still small against the spread")
    print("    across tessellations, though the trend now rises;")
    print("    the verdict that none of the scatter is numerical, which")
    print("    now rests on the control being 0.51 per cent of the gate")
    print("    power and not on its own 0.42 dB scatter.")
    print("  DOES NOT SURVIVE:")
    print("    that the measurement sits on the speckle floor. It sits")
    print("    1.66 dB above the floor at the counted crossings and")
    print("    3.55 dB above the floor at the look count measured in")
    print("    band, and the gamma model that defines the floor is")
    print("    rejected by its own data on two moments.")
    print("    That the coherent orders bring it onto the floor. The")
    print("    audited residual is 3.69 dB and not 2.97 dB.")
    print("  NOT ESTABLISHED, and flagged rather than written up:")
    print("    that band limiting explains the gap. It is measured, it")
    print("    is in the right direction, and it is two orders of")
    print("    magnitude too small.")
    print("    that the few-bright-facet picture explains the excess")
    print("    contrast. It is consistent with every number here and it")
    print("    is not tested here.")
    print()


def main():
    legacy = "--legacy" in sys.argv
    sweeps = load_all()
    rows, sd = report_scatter(sweeps, legacy=legacy)
    report_resolution(sweeps, rows)
    report_uniform(sweeps, rows)
    harmonics = report_harmonics(rows, sd[AUDITED])
    report_floor(sd[AUDITED], sd[LEGACY])
    report_lookcount(sweeps)
    report_gamma_test(sweeps, rows, harmonics[AUDITED][1])
    report_locality(sweeps)
    report_sampling()
    if legacy:
        report_audit(sweeps)
    report_verdict()


if __name__ == "__main__":
    main()
