"""Ensemble statistics over independent girdle tessellations at ppw 8.

Supports Section 4, "Ensemble statistics", and supplies the archive that
Fig. 'ensemble' is drawn from. Three questions, in the order the section
asks them:

  1. the mean coda level over independent tessellations and its spread,
  2. how many realisations are needed before that mean stabilises,
  3. whether the spread comes from the tessellation geometry or from the
     orientation draw laid on it.

The level is referenced to the SOURCE amplitude and never to the backwall
echo. Section 4, "Choice of reference", establishes that the backwall is
inadmissible across grid resolutions because refinement raises it while
lowering the coda, so their ratio moves by the sum of the two. The same
reference is used here so that an ensemble spread and a resolution step
are the same kind of number and can be compared.

Reads, all under ../../out/sweeps, each directory az*.npz with keys
'trace' and 'dt' plus a config.json:
    girdle_perp_ppw8, mx_girdle_s{7,17,23,41,53,71,89}_ppw8
        eight independent tessellations, girdle fabric, kappa = -8
    singlemax_ppw8, mx_single_s{17,23,41}_ppw8
        four single-maximum sweeps, kappa = +3.93, each sharing a
        bit-identical tessellation with the girdle sweep of the same seed
Question 3 also rebuilds the eight tessellations on the CPU at a coarse
grid, through sim/specimen.py, to measure their geometry directly.

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
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = ROOT / "out" / "sweeps"
RESULTS = ROOT / "sim" / "results"

# Acquisition and analysis constants, all as given in Section 3: 2 MHz
# centre frequency, 100 mm disc, reference speed 3850 m/s. The coda gate
# is the one used throughout the paper. Levels are taken from the
# unfiltered envelope, as in db_reconcile.py, so that they reproduce
# Table 'reconcile' cell for cell rather than nearly.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
CODA_GATE = (24e-6, 36e-6)

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
SINGLE = [("singlemax_ppw8", 11), ("mx_single_s17_ppw8", 17),
          ("mx_single_s23_ppw8", 23), ("mx_single_s41_ppw8", 41)]

# Tolerances the ensemble mean is asked to hold, in decibels. The
# coarsest is half the ppw 8 to 10 step of Table 'reconcile' (-2.21 dB
# on the source reference), which is the smallest quantity Section 4
# asks the ensemble to resolve; the finer two are there because the
# coarsest turns out to be met by a single realisation.
TOL_DB = (1.0, 0.5, 0.25)

# Reported as the standard interval throughout, and stated in the
# caption of Fig. 'ensemble'.
CONF = 0.95


# ------------------------------------------------------------------ load

def sweep_levels(name, step_deg=AZ_STEP_DEG):
    """Per-azimuth coda level in dB re source. Returns (az, levels, cfg).

    The source amplitude is the peak of the analytic envelope of the
    whole trace, which is the transmit pulse; the coda is the
    root-mean-square envelope in the gate.

    step_deg selects the azimuth grid. The default keeps only the common
    12 degree grid, so that no realisation is averaged over a different
    set of beam positions from the others; step_deg of 1 keeps every
    record on disk, which only the two seed-11 sweeps have more of.
    """
    folder = SWEEPS / name
    with open(folder / "config.json") as fh:
        cfg = json.load(fh)
    az, lev = [], []
    for entry in sorted(os.listdir(folder)):
        if not (entry.startswith("az") and entry.endswith(".npz")):
            continue
        degrees = int(entry[2:5])
        if degrees % step_deg:
            continue
        with np.load(folder / entry) as z:
            trace = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        env = np.abs(hilbert(trace))
        a, b = int(CODA_GATE[0] / dt), int(CODA_GATE[1] / dt)
        az.append(degrees)
        lev.append(20 * np.log10(np.sqrt((env[a:b] ** 2).mean()) / env.max()))
    return np.array(az), np.array(lev), cfg


def load_family(family):
    """(seeds, revolution levels, per-azimuth level matrix) for a fabric."""
    seeds, levels, matrix = [], [], []
    for name, seed in family:
        az, lev, cfg = sweep_levels(name)
        if cfg["seed"] != seed:
            raise ValueError("%s carries seed %s, expected %d"
                             % (name, cfg["seed"], seed))
        if not np.array_equal(az, AZ_COMMON):
            raise ValueError("%s does not cover the common azimuth grid"
                             % name)
        seeds.append(seed)
        levels.append(lev.mean())
        matrix.append(lev)
    return np.array(seeds), np.array(levels), np.array(matrix)


# --------------------------------------------------------------- compute

def quadrature_error():
    """Spread of a 30-azimuth revolution mean, measured not assumed.

    A revolution mean is not a sample from a population: the sweep covers
    the whole rotation, and the level varies smoothly enough with azimuth
    that summing 30 of them is closer to a quadrature than to a draw.
    Treating the azimuths as correlated samples and dividing the scatter
    by the square root of an effective count therefore overstates the
    error on the mean badly. The two 60-azimuth sweeps settle it: split
    each into its two interleaved 30-azimuth halves, which are two
    independent 12-degree grids over the same specimen, and the half
    difference of their means is the quantity wanted.
    """
    out = {}
    for name in ("girdle_perp_ppw8", "singlemax_ppw8"):
        _az, lev, _cfg = sweep_levels(name, step_deg=1)
        halves = np.array([lev[0::2].mean(), lev[1::2].mean()])
        out[name] = dict(n=len(lev), halves=halves,
                         err=0.5 * abs(halves[0] - halves[1]))
    return out


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
    for n in range(2, 200):
        if stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n) <= tol:
            return n
    raise ValueError("no count below 200 gives a %.2f dB interval at a "
                     "spread of %.2f dB" % (tol, sd))


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
    sys.path.insert(0, str(ROOT / "sim" / "analysis"))
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


def area_limited_spread(area_cm2):
    """Level spread that the boundary-area variation alone can produce.

    Backscattered power from a population of facets is proportional to
    the illuminated boundary area to first order, so a relative area
    spread of x becomes 10 log10(1+x) in level. This is the cap on the
    geometry contribution through the one channel that has to carry it,
    and it needs no fit.
    """
    area = np.asarray(area_cm2, float)
    return float(10 * np.log10(1 + area.std(ddof=1) / area.mean()))


def tessellations_are_shared(seed, coarse_ppw=4.0):
    """True when a girdle and a single maximum of one seed share labels.

    The premise of the whole paired design, checked rather than asserted.
    """
    import sys
    sys.path.insert(0, str(ROOT / "sim"))
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


# ------------------------------------------------------------------ main

def main():
    g_seeds, g_lev, g_mat = load_family(GIRDLE)
    s_seeds, s_lev, s_mat = load_family(SINGLE)

    print("coda level per realisation, ppw 8, %d azimuths, dB re source"
          % len(AZ_COMMON))
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
    print("  full spread of the girdle realisations %.2f to %.2f dB, "
          "range %.2f dB" % (g_lev.min(), g_lev.max(),
                             g_lev.max() - g_lev.min()))
    print("  per-azimuth scatter pooled over the eight: %.2f dB"
          % np.sqrt((g_mat.var(axis=1, ddof=1)).mean()))

    quad = quadrature_error()
    print("\nazimuthal sampling error on one revolution mean")
    for name, q in quad.items():
        print("  %-18s %d azimuths, interleaved halves %.3f and %.3f, "
              "half difference %.3f dB"
              % (name, q["n"], q["halves"][0], q["halves"][1], q["err"]))
    quad_err = float(np.mean([q["err"] for q in quad.values()]))
    print("  so a revolution mean is repeatable to %.2f dB, which is "
          "%.0f%% of the %.2f dB ensemble spread and cannot explain it"
          % (quad_err, 100 * quad_err / g_sd, g_sd))

    counts, p_lo, p_hi, v_lo, v_hi = subsample_band(g_lev)
    agree = {tol: subsample_agreement(g_lev, tol)[1] for tol in TOL_DB}
    print("\nhow far an N-realisation mean can be from the eight-"
          "realisation mean")
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

    shared = [int(seed) for seed in s_seeds if seed in set(g_seeds)]
    idx_g = [int(np.where(g_seeds == seed)[0][0]) for seed in shared]
    idx_s = [int(np.where(s_seeds == seed)[0][0]) for seed in shared]
    part = variance_partition(g_lev[idx_g], s_lev[idx_s])
    lo, hi = fisher_interval(part["icc"], part["n_pairs"])
    print("\ngeometry against orientation, %d shared tessellations"
          % part["n_pairs"])
    print("  seeds %s carry a bit-identical tessellation under both "
          "fabrics: %s" % (list(shared), tessellations_are_shared(shared[0])))
    print("  fabric main effect, girdle minus single maximum  %+.2f dB"
          % part["fabric_effect"])
    print("  orientation draw at frozen geometry   sd %.2f dB"
          % np.sqrt(max(part["var_orient"], 0.0)))
    print("  tessellation geometry                 var %+.3f dB^2%s"
          % (part["var_geom"],
             ", not resolved" if part["var_geom"] < 0 else ""))
    print("  correlation of the two fabrics across shared tessellations "
          "r = %+.2f, %.0f%% interval [%+.2f, %+.2f]"
          % (part["icc"], 100 * CONF, lo, hi))

    print("\ngeometric descriptors of the eight tessellations")
    rows = tessellation_geometry(g_seeds)
    print("  %-6s %8s %10s %12s %11s"
          % ("seed", "grains", "d (mm)", "area (cm2)", "crossings"))
    for row in rows:
        print("  %-6d %8d %10.2f %12.1f %11.1f"
              % (row["seed"], row["n_grains"], row["d_mean_mm"],
                 row["area_cm2"], row["crossings"]))
    print("  correlation with the measured level, n = %d" % len(g_lev))
    for key in ("n_grains", "d_mean_mm", "area_cm2", "crossings"):
        x = np.array([row[key] for row in rows], float)
        r = float(np.corrcoef(x, g_lev)[0, 1])
        p = 2 * stats.t.sf(abs(r) * np.sqrt((len(x) - 2) / (1 - r ** 2)),
                           len(x) - 2)
        print("    %-10s r = %+.2f  p = %.2f  (spread %.1f%% of the mean)"
              % (key, r, p, 100 * x.std(ddof=1) / x.mean()))
    cap = area_limited_spread([row["area_cm2"] for row in rows])
    print("  boundary area varies by %.2f dB in level, %.0f%% of the "
          "%.2f dB observed spread" % (cap, 100 * cap / g_sd, g_sd))

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS / "ensemble.npz",
             azimuths=AZ_COMMON,
             girdle_seeds=g_seeds, girdle_levels=g_lev, girdle_azimuth=g_mat,
             single_seeds=s_seeds, single_levels=s_lev, single_azimuth=s_mat,
             quadrature_err=quad_err, tol_db=np.array(TOL_DB), conf=CONF,
             var_orient=part["var_orient"], var_geom=part["var_geom"],
             icc=part["icc"], fabric_effect=part["fabric_effect"],
             area_cm2=np.array([row["area_cm2"] for row in rows], float))
    print("\nwrote %s" % (RESULTS / "ensemble.npz"))


if __name__ == "__main__":
    main()
