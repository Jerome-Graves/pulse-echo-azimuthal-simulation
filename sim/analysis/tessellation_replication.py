"""Does the Section 5.2 facet-model result hold on more than one
microstructure?

Section 5.2 claims the backscattered coda is predicted, with no fitted
parameter, by grain-boundary position and orientation. It reports
r = 0.509 on the azimuthal scalar (Fig. codafield) and identification of
the specimen among 44 candidates, three of four tessellations, at
p = 4.6e-5 (Fig. identify). Every converged number behind those claims
came from tessellation seed 11, which was chosen during development and
is therefore the specimen most exposed to selection. Eight independent
girdle tessellations now exist at ppw 8, so the claim can be replicated
rather than asserted.

Four things are computed, all on the 30 azimuths every ppw 8 sweep has in
common so that no sweep is advantaged by finer sampling:

  A  the azimuthal-scalar correlation for each tessellation separately,
     under the circular-shift null of Appendix B and its 1/n floor
  B  the same on the full azimuth-by-time field
  C  the cross-tessellation matrix: a prediction built from specimen A
     scored against specimen B, which must return nothing
  D  identification against the enlarged candidate set the new
     tessellations provide, 48 rather than 44

Significance below the per-sweep 1/n floor comes only from combining
tessellations, and the combination is exact rather than asymptotic: under
the null the true alignment's rank among the n circular shifts is uniform
on 1..n, so ranking first has probability exactly 1/n independently in
each tessellation, and the count over tessellations is binomial. Fisher's
method on the same per-sweep p values is reported beside it as a
cross-check, and is conservative because those p values are discrete
multiples of 1/n.

READS
  out/sweeps/girdle_perp_ppw8/az*.npz                seed 11, 60 azimuths
  out/sweeps/mx_girdle_s{7,17,23,41,53,71,89}_ppw8/az*.npz
  out/sweeps/singlemax_ppw8/az*.npz                  seed 11, other fabric
  out/sweeps/mx_single_s{17,23,41}_ppw8/az*.npz
WRITES
  out/tesscache/tess_s<seed>_p<ppw>_k<kappa>.npz     cached label volumes
  sim/analysis/tessellation_replication.npz          every number printed
"""
import os
import sys
import time

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import binom, chi2, t as student

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
    sys.path[:0] = [os.path.join(sys.path[0], _d)
                    for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

import facet_predictors as FP                          # noqa: E402
import _t2_common as C2                                # noqa: E402
import shift_null_2d as S2                             # noqa: E402
from specimen import DiskSpecimen                      # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                     "out", "tesscache")

# Every ppw 8 sweep holds these 30 azimuths exactly, with no interpolation:
# the multi-tessellation runs step 12 degrees and the two seed-11 runs step
# 6 degrees, so the coarser grid is a strict subset of the finer one.
AZ_COMMON = np.arange(0, 360, 12)
AZ_SEED11 = np.arange(0, 360, 6)
PPW = 8.0

# Fabric parameters as recorded in each sweep's config.json.
KAPPA_GIRDLE, AXIS_GIRDLE = -8.0, (1.0, 0.0, 0.0)

# Far-field divergence half-angle of the 6.35 mm element at 2 MHz (Sec. 3).
# facet_predictors carries the same value, so the scalar predictor here is
# the one whose output Sec. 5.2 quotes.
CONE_DEG = np.degrees(FP.HALF)
N_RAY = 600

# Power envelope of the band-passed source wavelet. Deterministic, and
# reused across every candidate rather than rebuilt per prediction.
# Centring applied to both fields before correlating. "double" removes
# the range decay and the per-azimuth level; "harm" additionally strips
# azimuthal harmonics k = 1 to 4 at every time sample, so nothing a bulk
# fabric model could generate survives. The manuscript quotes harm; see
# centring_convention.py for why the smaller correlation is the more
# significant one. Override from the command line to reproduce either.
CENTRING = sys.argv[1] if len(sys.argv) > 1 else "double"

PULSE_KERNEL = C2.pulse_power_kernel(C2.DT_C)

GIRDLE = [("girdle_perp_ppw8", 11), ("mx_girdle_s7_ppw8", 7),
          ("mx_girdle_s17_ppw8", 17), ("mx_girdle_s23_ppw8", 23),
          ("mx_girdle_s41_ppw8", 41), ("mx_girdle_s53_ppw8", 53),
          ("mx_girdle_s71_ppw8", 71), ("mx_girdle_s89_ppw8", 89)]
SINGLE = [("singlemax_ppw8", 11), ("mx_single_s17_ppw8", 17),
          ("mx_single_s23_ppw8", 23), ("mx_single_s41_ppw8", 41)]

# Controls, all carried on the seed-11 tessellation, so the geometry a
# candidate can match is present and only the acoustic contrast across the
# boundaries changes. zerocontrast gives every grain the same c-axis;
# the cs_f ladder interpolates each grain's stiffness a fraction f from the
# orientation-averaged tensor toward its own, so scattered power must scale
# as f^2 and girdle_perp_ppw8 is the f = 1 rung. If identification survives
# at f = 0 it is matching the rendering of the tessellation, not scattering.
CONTROL = [("zerocontrast_ppw8", 11, "same c-axis in every grain"),
           ("cs_f000_s11_ppw8", 11, "contrast f = 0.00"),
           ("cs_f025_s11_ppw8", 11, "contrast f = 0.25"),
           ("cs_f050_s11_ppw8", 11, "contrast f = 0.50"),
           ("cs_f075_s11_ppw8", 11, "contrast f = 0.75")]

# The 40 wrong-tessellation seeds of the Sec. 5.2 identification test.
# With the eight real girdle tessellations they make 48 candidates,
# against the 44 the manuscript had.
DISTRACTORS = list(range(100, 140))

# Values Sec. 5.2 reports, printed beside the recomputation so that a
# divergence between the manuscript and this script is visible in the
# output rather than having to be looked for. Superseded by this run:
# the earlier three of four against 44 candidates, p = 4.6e-5, was the
# sweep and not the tessellation as the unit of replication, and the
# r_full = 0.509 and r_geom = 0.468 of that era were the development
# specimen alone on a legacy centring.
REF = dict(r_full=0.162, r_geom=0.330, ident_hit=4, ident_of=8,
           n_cand=48, p_ident=1.23e-5)


# ─────────────────────────────── load ─────────────────────────────────
def _label_plane_by_plane(x, z, R, seeds, weights):
    """Exact all-seeds Laguerre argmin, one z-plane at a time, in NumPy.

    specimen.py labels the grid with CuPy and falls back to a KD-tree
    shortlist that materialises a 5.7 GB temporary at ppw 8. This analysis
    reruns no solver and touches no CUDA, so the exact float64 power argmin
    the production builds used is supplied here directly: about 20 s and
    150 MB per specimen.
    """
    nx, nz = len(x), len(z)
    X, Y = np.meshgrid(x, x, indexing="ij")
    inside = (X ** 2 + Y ** 2) <= R ** 2
    p_xy = np.stack([X[inside], Y[inside]], axis=1)
    d_xy = (((p_xy[:, None, :] - seeds[None, :, :2]) ** 2).sum(-1)
            - weights[None, :])
    raw = np.full((nx, nx, nz), -1, np.int32)
    for iz in range(nz):
        d2 = d_xy + (z[iz] - seeds[:, 2])[None, :] ** 2
        plane = np.full((nx, nx), -1, np.int32)
        plane[inside] = d2.argmin(axis=1).astype(np.int32)
        raw[:, :, iz] = plane
    return raw


DiskSpecimen._label_grid_gpu = staticmethod(_label_plane_by_plane)


def tessellation(seed, kappa=KAPPA_GIRDLE, axis=AXIS_GIRDLE, ppw=PPW):
    """Label volume, c-axes and seed points of one specimen, cached.

    Built at the solver's own grid spacing because the set of grains that
    survive with non-zero volume, and hence the random stream that draws
    the c-axes, depends on it. A coarser build is a different specimen from
    the one the solver saw.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"tess_s{seed}_p{ppw:g}_k{kappa:g}.npz")
    if not os.path.exists(path):
        t0 = time.time()
        h = C2.C_REF / C2.F0 / ppw
        b = DiskSpecimen(diameter_m=C2.DIA, thickness_m=0.035, n_grains=100,
                         size_cv=0.35, concentration=kappa, spatial_corr=0.0,
                         fabric_axis=tuple(axis), seed=seed).build(h)
        np.savez_compressed(path, labels=b["labels"], axes=b["axes"],
                            seeds=b["seeds"], h=h)
        print(f"    built seed {seed:>3} kappa {kappa:+.2f}  "
              f"{len(b['seeds'])} grains  {time.time() - t0:.0f} s",
              flush=True)
    with np.load(path) as z:
        return (z["labels"].astype(np.int32), z["axes"], z["seeds"],
                float(z["h"]))


def coda_level(sweep, az_keep):
    """Band-limited coda level in dB re the sweep-mean backwall.

    This is the scalar of Sec. 5.2 taken straight from facet_predictors, so
    that the reproduction of r = 0.509 runs through the same measurement
    code as the original.
    """
    az, lev = FP.measure(os.path.join(FP.OUT, sweep))
    return lev[[int(np.where(az == a)[0][0]) for a in az_keep]]


def loo_median(R):
    """Azimuth-common component of R, leaving each azimuth out.

    Row i is the median over the OTHER azimuths at every time sample, so no
    azimuth is ever compared against a statistic that contains itself. The
    median rather than the mean because the common component is what is
    wanted and a mean is dragged by the few bright azimuths that carry most
    of the coda.
    """
    return np.array([np.median(np.delete(R, i, axis=0), axis=0)
                     for i in range(len(R))])


def azimuth_rf(sweep, tgrid, az_keep=AZ_COMMON):
    """Band-passed RF of every azimuth resampled onto ONE physical axis.

    Each azimuth carries its own dt, set by the CFL limit of its own grid,
    so azimuths cannot be stacked by sample index. Returned sorted by
    azimuth, matching coda_field's row order.
    """
    d = os.path.join(FP.OUT, sweep)
    az, rows = [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        a = int(f[2:5])
        if a not in az_keep:
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        sos = butter(4, [C2.BAND[0] / (fs / 2), C2.BAND[1] / (fs / 2)],
                     btype="band", output="sos")
        rows.append(np.interp(tgrid, np.arange(len(tr)) * dt,
                              sosfiltfilt(sos, tr)))
        az.append(a)
    order = np.argsort(az)
    return np.array(az)[order], np.array(rows)[order]


def raised_cosine(t, lo, hi, pad):
    """1 on [lo, hi], cosine skirts of width pad, 0 outside."""
    w = np.zeros_like(t)
    w[(t >= lo) & (t <= hi)] = 1.0
    a = (t > lo - pad) & (t < lo)
    w[a] = 0.5 * (1 - np.cos(np.pi * (t[a] - (lo - pad)) / pad))
    b = (t > hi) & (t < hi + pad)
    w[b] = 0.5 * (1 + np.cos(np.pi * (t[b] - hi) / pad))
    return w


def coda_field(sweep, tgrid, az_keep=AZ_COMMON, median_removal=False,
               taper=None):
    """Measured envelope on tgrid, and the per-azimuth two-way speed.

    Normalised by the SWEEP-mean backwall, not the per-azimuth one, so that
    the azimuthal variation of the coda is not divided away. The backwall
    time is remeasured from the trace because the multi-tessellation sweeps
    were written without the t1_s field; on girdle_perp_ppw8, where both
    exist, remeasured and recorded agree to 37 ns rms on a 52.6 us arrival
    and track each other across azimuth at r = 0.995.

    The source delay is subtracted before the speed is formed. The trace
    origin is simulation t = 0 and the Ricker peaks at T0_SRC = 1.2/f0, so
    a measured arrival is 2*DIA/c + T0_SRC. Dividing DIA by the raw arrival
    folds that delay into the speed and returns 3782 to 3806 m/s against a
    reference 3850, low by 1.14 to 1.75 per cent; bin_events then adds
    T0_SRC back, so every predicted arrival is stretched late by 1.15 per
    cent of its own time, which is 0.35 us at 30 us (the eight-sweep
    mean is 1.1528 +- 0.0024 per cent; identify_delay_audit.py). Three checks fix the
    sign of this. The corrected estimator returns 3826 to 3850 m/s. The
    residual best rigid offset between prediction and measurement falls
    from -0.356 +- 0.168 us to -0.019 +- 0.146 us. And the account predicts
    with no free parameter that the own score peaks at a speed scale
    t1/(t1 - T0_SRC) = 1.0115, computed from the wavelet alone, against a
    measured 1.0121 +- 0.0040 over the eight girdle sweeps, with unity
    optimal for none of them.

    median_removal selects how the envelope is formed, and defaults to the
    behaviour every published number was computed with:

      False    the published path. Hilbert the band-passed trace at its own
               sample rate, then interpolate the envelope onto tgrid.
      "none"   interpolate the band-passed RF onto tgrid first, then
               Hilbert. Subtracts nothing. This exists so that a removal can
               be compared against a control that shares its resampling.
      "loo"    the same, with the leave-one-out azimuthal median subtracted
               from the RF at every time sample before the Hilbert transform
               (median trace removal, the standard ring-down suppressor in
               GPR and contact NDT).

    The backwall amplitude and time, and hence the sweep-mean normalisation
    and the per-azimuth speed, are ALWAYS measured from the unremoved
    native-rate envelope. A removal that gutted the backwall would otherwise
    change the range-to-time map at the same time as the coda, and the two
    effects could not be told apart.

    taper is (lo, hi, pad) or None, and needs median_removal to be "none" or
    "loo" because it acts on the resampled RF. The RF is multiplied by a
    raised-cosine window before the Hilbert transform, so the envelope
    inside [lo, hi] cannot have imported energy from outside [lo-pad,
    hi+pad]. |hilbert| of a whole trace is not a local operator, and a
    window a few microseconds after an arrival tens of decibels larger will
    otherwise read that arrival's analytic tail as if it were coda.
    """
    d = os.path.join(FP.OUT, sweep)
    az, rows, raw, e1, t1 = [], [], [], [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        a = int(f[2:5])
        if a not in az_keep:
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        sos = butter(4, [C2.BAND[0] / (fs / 2), C2.BAND[1] / (fs / 2)],
                     btype="band", output="sos")
        rf = sosfiltfilt(sos, tr)
        env = np.abs(hilbert(rf))
        k0, w = int(2 * C2.DIA / C2.C_REF * fs), int(3e-6 * fs)
        lo = max(k0 - w, 0)
        seg = env[lo:k0 + w]
        e1.append(seg.max())
        t1.append((lo + int(np.argmax(seg))) * dt)
        t = np.arange(len(tr)) * dt
        rows.append(np.interp(tgrid, t, env))
        if median_removal:
            raw.append(np.interp(tgrid, t, rf))
        az.append(a)
    order = np.argsort(az)
    c_az = 2.0 * C2.DIA / (np.array(t1)[order] - C2.T0_SRC)
    if not median_removal:
        if taper is not None:
            raise ValueError("taper needs median_removal 'none' or 'loo'")
        return np.array(rows)[order] / float(np.mean(e1)), c_az
    R = np.array(raw)[order]
    if median_removal == "loo":
        R = R - loo_median(R)
    elif median_removal != "none":
        raise ValueError(median_removal)
    if taper is not None:
        R = R * raised_cosine(tgrid, *taper)[None, :]
    return (np.abs(hilbert(R, axis=1)) / float(np.mean(e1)), c_az)


# ────────────────────────────── compute ───────────────────────────────
def facet_scalar(tess, az_deg):
    """Eq. (facetmodel) integrated over the coda gate, per azimuth.

    born_spec is the full equation, facet directivity times the squared
    reflection coefficient. geom_only drops the contrast factor and is the
    grain-boundary geometry alone.
    """
    lab, axes, seeds, h = tess
    p = FP.preds(lab, axes, seeds, h, az_deg)
    return {k: p[k] for k in ("geom_only", "born_spec")}


def facet_field(tess, az_deg, tgrid):
    """Eq. (facetmodel) as an azimuth-by-time power field.

    The model's own grain-resolved velocities set both the arrival times
    and the reflection coefficients, so this is the full equation applied
    to a specimen's own tessellation.
    """
    lab, axes, seeds, h = tess
    edges = np.concatenate([tgrid - C2.DT_C / 2, [tgrid[-1] + C2.DT_C / 2]])
    rows = []
    for a in az_deg:
        t, w, _ = C2.facet_events(lab, axes, seeds, h, a, use_dv=True,
                                  use_direc=True, n_ray=N_RAY,
                                  th_max_deg=CONE_DEG)
        rows.append(np.histogram(t, edges, weights=w)[0])
    return np.apply_along_axis(
        lambda r: np.convolve(r, PULSE_KERNEL, mode="same"), 1,
        np.array(rows))


def march_geometry(tess, az_deg):
    """Beam-crossed grain boundaries of a candidate: weight and range.

    The candidate contributes geometry only. Marching does not depend on
    which specimen the candidate will be scored against, so it is done once
    per candidate and reused across all twelve sweeps; recomputing it per
    sweep costs an order of magnitude more for an identical answer.
    """
    lab, axes, seeds, h = tess
    out = []
    for a in az_deg:
        _, w, s = C2.facet_events(lab, axes, seeds, h, a, use_dv=False,
                                  use_direc=True, n_ray=N_RAY,
                                  th_max_deg=CONE_DEG, const_v=True)
        out.append((w, s))
    return out


def bin_events(events, c_az, tgrid):
    """Two-way time field from marched crossings and a measured speed.

    Range is mapped to time with the speed taken from the specimen's own
    backwall arrival rather than from the candidate, so a wrong candidate
    is never additionally penalised for carrying the wrong fabric. The
    source-wavelet delay t0 = 1.2/f0 of Sec. 5.2 is applied here.
    """
    edges = np.concatenate([tgrid - C2.DT_C / 2, [tgrid[-1] + C2.DT_C / 2]])
    rows = [np.histogram(2.0 * s / c_az[q] + C2.T0_SRC, edges, weights=w)[0]
            for q, (w, s) in enumerate(events)]
    return np.apply_along_axis(
        lambda r: np.convolve(r, PULSE_KERNEL, mode="same"), 1,
        np.array(rows))


def shift_rank(pred, meas):
    """Circular-shift null of Appendix B, one-sided on r.

    Returns the correlation at zero offset, its rank among the n rigid
    rotations of the measurement against the prediction (1 is best), and
    p = (1 + #{s != 0 : r_s >= r_0}) / n, which equals rank/n and is
    floored at 1/n.
    """
    n = len(meas)
    r = np.array([np.corrcoef(pred, np.roll(meas, s))[0, 1]
                  for s in range(n)])
    rank = int(np.sum(r >= r[0]))
    return float(r[0]), rank, rank / n


def shift_rank_field(pred, meas):
    """The same null on the azimuth-by-time field, shifted in azimuth."""
    n = meas.shape[0]
    r = np.array([S2.score(np.roll(meas, s, axis=0), pred, CENTRING)
                  for s in range(n)])
    rank = int(np.sum(r >= r[0]))
    return float(r[0]), rank, rank / n


def combine(ranks, n_shift):
    """Combine independent tessellations into one significance level."""
    ranks = np.asarray(ranks, float)
    k, m = int(np.sum(ranks == 1)), len(ranks)
    p_binom = float(binom.sf(k - 1, m, 1.0 / n_shift))
    x = -2.0 * np.sum(np.log(ranks / n_shift))
    return k, m, p_binom, float(chi2.sf(x, 2 * m))


def selection_gap(v):
    """How far the development specimen sits above the fresh ones.

    Seed 11 was chosen while the model was being built, so the honest
    estimate of what the model does on an unseen microstructure is the mean
    over the seven tessellations simulated afterwards. This answers whether
    seed 11 lies inside their spread or above it.
    """
    rest = np.asarray(v[1:], float)
    return (float(rest.mean()), float(rest.std(ddof=1)),
            float((v[0] - rest.mean()) / rest.std(ddof=1)))


def pooled(v):
    """Second-level test with the tessellation as the unit of replication.

    Fisher z is averaged over tessellations and tested against zero using
    the between-tessellation spread as its own error estimate, so this
    statement does not depend on the within-sweep azimuthal autocorrelation
    at all and is not floored at 1/n. Sec. 5.4 already treats the
    tessellation as the unit of replication for the travel-time result.
    One-sided, because the model predicts a positive correlation.
    """
    z = np.arctanh(np.clip(np.asarray(v, float), -0.999, 0.999))
    m = len(z)
    tstat = z.mean() / (z.std(ddof=1) / np.sqrt(m))
    return float(np.tanh(z.mean())), float(student.sf(tstat, m - 1))


def compute_scalar(sweeps, tess, az_deg):
    """Part A for one azimuth set: r, rank and p per sweep per predictor."""
    out = {}
    for name, seed in sweeps:
        meas = FP.strip(coda_level(name, az_deg), az_deg)
        pred = facet_scalar(tess[seed], az_deg)
        out[name] = {k: shift_rank(FP.strip(v, az_deg), meas)
                     for k, v in pred.items()}
    return out


def compute_field(sweeps, tess, tgrid):
    """Part B: full-field r, rank and p per sweep."""
    gate = (tgrid >= C2.CODA_W[0]) & (tgrid < C2.CODA_W[1])
    out = {}
    for name, seed in sweeps:
        E, _ = coda_field(name, tgrid)
        P = facet_field(tess[seed], AZ_COMMON, tgrid)[:, gate]
        out[name] = shift_rank_field(P, E[:, gate] ** 2)
    return out


def compute_cross(sweeps, tess):
    """Part C: prediction-against-specimen and coda-against-coda matrices."""
    meas = {n: FP.strip(coda_level(n, AZ_COMMON), AZ_COMMON)
            for n, _ in sweeps}
    pred = {s: FP.strip(facet_scalar(tess[s], AZ_COMMON)["born_spec"],
                        AZ_COMMON) for _, s in sweeps}
    names = [n for n, _ in sweeps]
    seeds = [s for _, s in sweeps]
    X = np.array([[np.corrcoef(pred[s], meas[n])[0, 1] for n in names]
                  for s in seeds])
    Y = np.array([[np.corrcoef(meas[a], meas[b])[0, 1] for b in names]
                  for a in names])
    return X, Y, seeds


def compute_identification(sweeps, cands, tgrid):
    """Part D: score every sweep against every candidate tessellation."""
    gate = (tgrid >= C2.CODA_W[0]) & (tgrid < C2.CODA_W[1])
    meas = {n: coda_field(n, tgrid) for n, _ in sweeps}
    sc = {n: {} for n, _ in sweeps}
    for c in cands:
        ev = march_geometry(tessellation(c), AZ_COMMON)
        for name, _ in sweeps:
            E, c_az = meas[name]
            P = bin_events(ev, c_az, tgrid)[:, gate]
            sc[name][c] = S2.score(E[:, gate] ** 2, P, CENTRING, log=False)
    hits = {}
    for name, own in sweeps:
        v = np.array([sc[name][c] for c in cands])
        i = cands.index(own)
        other = np.delete(v, i)
        # The strongest wrong candidate is reported alongside the rank
        # because a near-degenerate alternative is the failure mode this
        # test has: two tessellations drawn from the same distribution can
        # present nearly the same facet pattern to the beam.
        j = int(np.argmax(other))
        runner = [c for c in cands if c != own][j]
        hits[name] = (v[i], int(np.sum(v >= v[i])),
                      (1 + int(np.sum(other >= v[i]))) / len(cands),
                      (v[i] - other.mean()) / other.std(ddof=1),
                      runner, other[j])
    return sc, hits


# ──────────────────────────────── draw ────────────────────────────────
def report_scalar(sweeps, rows, ref60):
    print("=" * 79)
    print("A. AZIMUTHAL SCALAR. Eq. (facetmodel) integrated over the coda")
    print("   gate, harmonics k = 0, 2, 4 stripped from both series.")
    print(f"   {len(AZ_COMMON)} common azimuths, shift-null floor "
          f"p = 1/{len(AZ_COMMON)} = {1 / len(AZ_COMMON):.4f}")
    print("=" * 79)
    print(f"{'sweep':<20}{'seed':>5}{'r_geom':>9}{'rank':>6}{'p':>8}"
          f"{'r_full':>9}{'rank':>6}{'p':>8}")
    for name, seed in sweeps:
        g, b = rows[name]["geom_only"], rows[name]["born_spec"]
        print(f"{name:<20}{seed:>5}{g[0]:>9.3f}{g[1]:>6}{g[2]:>8.4f}"
              f"{b[0]:>9.3f}{b[1]:>6}{b[2]:>8.4f}")
    g60 = ref60["girdle_perp_ppw8"]["geom_only"]
    b60 = ref60["girdle_perp_ppw8"]["born_spec"]
    print(f"\n  seed 11 on its native {len(AZ_SEED11)} azimuths, against the"
          f" {REF['r_geom']} and {REF['r_full']} of Sec. 5.2:")
    print(f"    r_geom = {g60[0]:.3f} (rank {g60[1]}/{len(AZ_SEED11)}), "
          f"r_full = {b60[0]:.3f} (rank {b60[1]}/{len(AZ_SEED11)})")
    for key, lab in (("geom_only", "geometry only"), ("born_spec", "full")):
        v = np.array([rows[n][key][0] for n, _ in sweeps])
        ranks = [rows[n][key][1] for n, _ in sweeps]
        k, m, pb, pf = combine(ranks, len(AZ_COMMON))
        mf, sf, z11 = selection_gap(v)
        kf, mf_n, pbf, pff = combine(ranks[1:], len(AZ_COMMON))
        print(f"\n  {lab:<14} over {m} tessellations: mean {v.mean():+.3f} "
              f"sd {v.std(ddof=1):.3f} range "
              f"[{v.min():+.3f},{v.max():+.3f}]")
        print(f"  {'':<14} seed 11 ranks {int(np.sum(v >= v[0]))} of {m} "
              f"by r; {k} of {m} tessellations rank first")
        print(f"  {'':<14} combined: binomial p = {pb:.3g}, "
              f"Fisher p = {pf:.3g}")
        print(f"  {'':<14} the {mf_n} FRESH tessellations alone: mean "
              f"{mf:+.3f} sd {sf:.3f}; seed 11 sits {z11:+.1f} sd above")
        print(f"  {'':<14} fresh only: {kf} of {mf_n} rank first, "
              f"binomial p = {pbf:.3g}, Fisher p = {pff:.3g}")
        ra, pa = pooled(v)
        rr, pr = pooled(v[1:])
        print(f"  {'':<14} tessellation as unit: all {m} r = {ra:+.3f} "
              f"p = {pa:.3g}; fresh {mf_n} r = {rr:+.3f} p = {pr:.3g}")


def report_field(sweeps, rows):
    print("\n" + "=" * 79)
    print("B. FULL AZIMUTH-BY-TIME FIELD, dB, per-time and per-azimuth")
    print("   means removed from both matrices, azimuth circular-shift null")
    print("=" * 79)
    print(f"{'sweep':<20}{'seed':>5}{'r':>9}{'rank':>6}{'p':>8}")
    for name, seed in sweeps:
        r, rank, p = rows[name]
        print(f"{name:<20}{seed:>5}{r:>9.3f}{rank:>6}{p:>8.4f}")
    v = np.array([rows[n][0] for n, _ in sweeps])
    ranks = [rows[n][1] for n, _ in sweeps]
    k, m, pb, pf = combine(ranks, len(AZ_COMMON))
    mf, sf, z11 = selection_gap(v)
    kf, mf_n, pbf, pff = combine(ranks[1:], len(AZ_COMMON))
    print(f"\n  over {m} tessellations: mean {v.mean():+.3f} "
          f"sd {v.std(ddof=1):.3f} range [{v.min():+.3f},{v.max():+.3f}]")
    print(f"  {k} of {m} rank first; binomial p = {pb:.3g}, "
          f"Fisher p = {pf:.3g}")
    print(f"  the {mf_n} FRESH tessellations alone: mean {mf:+.3f} "
          f"sd {sf:.3f}; seed 11 sits {z11:+.1f} sd above")
    print(f"  fresh only: {kf} of {mf_n} rank first, binomial p = {pbf:.3g},"
          f" Fisher p = {pff:.3g}")
    ra, pa = pooled(v)
    rr, pr = pooled(v[1:])
    print(f"  tessellation as unit: all {m} r = {ra:+.3f} p = {pa:.3g}; "
          f"fresh {mf_n} r = {rr:+.3f} p = {pr:.3g}")


def report_cross(X, Y, seeds):
    print("\n" + "=" * 79)
    print("C. OUT OF SAMPLE. Rows are the tessellation the prediction was")
    print("   built from, columns the specimen it is scored against.")
    print("   Harmonics k = 0, 2, 4 stripped, so no shared fabric template")
    print("   survives in either series.")
    print("=" * 79)
    print(f"{'pred / meas':<14}" + "".join(f"{s:>8}" for s in seeds))
    for i, s in enumerate(seeds):
        print(f"{'seed ' + str(s):<14}"
              + "".join(f"{X[i, j]:>8.3f}" for j in range(len(seeds))))
    eye = np.eye(len(seeds), dtype=bool)
    dia, off = np.diag(X), X[~eye]
    print(f"\n  own tessellation   : mean {dia.mean():+.3f} "
          f"sd {dia.std(ddof=1):.3f}")
    print(f"  wrong tessellation : mean {off.mean():+.3f} "
          f"sd {off.std(ddof=1):.3f} range "
          f"[{off.min():+.3f},{off.max():+.3f}]")
    print(f"  separation {(dia.mean() - off.mean()) / off.std(ddof=1):.1f} "
          f"sd of the wrong-tessellation spread")
    win = int(np.sum(np.argmax(X, axis=0) == np.arange(len(seeds))))
    print(f"\n  Read down each column, the specimen's own prediction is the "
          f"strongest\n  of the {len(seeds)} in {win} of {len(seeds)} "
          f"columns, against a chance rate of 1/{len(seeds)} "
          f"(binomial p = "
          f"{binom.sf(win - 1, len(seeds), 1 / len(seeds)):.3g}). The")
    print("  azimuthal scalar on its own therefore does not identify the")
    print("  microstructure, whatever its correlation on the right one.")
    yo = Y[~eye]
    print(f"\n  coda against coda, different tessellations: mean "
          f"{yo.mean():+.3f} sd {yo.std(ddof=1):.3f} range "
          f"[{yo.min():+.3f},{yo.max():+.3f}]")
    print("  Nothing physical is shared between these specimens once the")
    print("  fabric harmonics are gone, so this is the common-mode floor")
    print("  every cross-tessellation score has to be read against.")


def report_identification(sweeps, cands, hits):
    print("\n" + "=" * 79)
    print(f"D. IDENTIFICATION against {len(cands)} candidate tessellations,")
    print("   the 44 of Sec. 5.2 plus the four new girdle seeds. The")
    print("   candidate supplies geometry only: range is mapped to time")
    print("   with the speed measured from the specimen's own backwall.")
    print(f"   Per-sweep floor p = 1/{len(cands)} = {1 / len(cands):.4f}")
    print("=" * 79)
    print(f"{'sweep':<20}{'own':>5}{'r_own':>9}{'rank':>8}{'p_wrong':>9}"
          f"{'z':>7}{'best wrong':>13}")
    for name, own in sweeps:
        r, rank, p, z, run, rv = hits[name]
        print(f"{name:<20}{own:>5}{r:>9.3f}{rank:>5}/{len(cands):<2}"
              f"{p:>9.4f}{z:>7.1f}   s{int(run):<3} {rv:+.3f}")
    girdle = [n for n, _ in GIRDLE]
    k = sum(1 for n in girdle if hits[n][1] == 1)
    m = len(girdle)
    pb = float(binom.sf(k - 1, m, 1.0 / len(cands)))
    kf = sum(1 for n in girdle[1:] if hits[n][1] == 1)
    pbf = float(binom.sf(kf - 1, m - 1, 1.0 / len(cands)))
    print("\n  The unit of replication is the tessellation, and the eight")
    print("  girdle sweeps are eight independent ones.")
    print(f"  correct identifications: {k} of {m} against {len(cands)} "
          f"candidates")
    print(f"  binomial p = P(X >= {k}) for X ~ Bin({m}, 1/{len(cands)}) "
          f"= {pb:.3g}")
    print(f"  excluding the development specimen seed 11: {kf} of {m - 1}, "
          f"p = {pbf:.3g}")
    agree = (k, m, len(cands)) == (REF['ident_hit'], REF['ident_of'],
                                   REF['n_cand'])
    print(f"  Sec. 5.2 states {REF['ident_hit']} of {REF['ident_of']} "
          f"tessellations against {REF['n_cand']} candidates, "
          f"p = {REF['p_ident']:.1e}"
          f"{'' if agree else '   <-- MANUSCRIPT DISAGREES'}")
    print("\n  CONTROLS, all on the seed-11 tessellation. The geometry a")
    print("  candidate could match is present in every one of these; only")
    print("  the acoustic contrast across the boundaries is reduced.")
    print(f"{'sweep':<20}{'r_own':>9}{'rank':>8}{'p_wrong':>9}{'z':>7}"
          f"   note")
    for name, _, note in CONTROL:
        r, rank, p, z, _, _ = hits[name]
        print(f"{name:<20}{r:>9.3f}{rank:>5}/{len(cands):<2}{p:>9.4f}"
              f"{z:>7.1f}   {note}")
    r1, k1 = hits["girdle_perp_ppw8"][0], hits["girdle_perp_ppw8"][1]
    print(f"{'girdle_perp_ppw8':<20}{r1:>9.3f}{k1:>5}/{len(cands):<2}"
          f"{'':>9}{'':>7}   contrast f = 1.00, the f = 1 rung")
    print("  The score is a double-centred correlation and so is blind to")
    print("  level: it is zero at f = 0 and near its final value by")
    print("  f = 0.25. Scattered POWER still scales as f^2; what saturates")
    print("  is the pattern, which is what identification reads.")


def main():
    tgrid = np.arange(20e-6, 42e-6, C2.DT_C)
    tess = {s: tessellation(s) for _, s in GIRDLE}
    cands = sorted(set([s for _, s in GIRDLE] + DISTRACTORS))

    rows_a = compute_scalar(GIRDLE, tess, AZ_COMMON)
    ref60 = compute_scalar(GIRDLE[:1], tess, AZ_SEED11)
    rows_b = compute_field(GIRDLE, tess, tgrid)
    X, Y, seeds = compute_cross(GIRDLE, tess)
    scored = GIRDLE + SINGLE + [(n, s) for n, s, _ in CONTROL]
    sc, hits = compute_identification(scored, cands, tgrid)

    report_scalar(GIRDLE, rows_a, ref60)
    report_field(GIRDLE, rows_b)
    report_cross(X, Y, seeds)
    report_identification(GIRDLE + SINGLE, cands, hits)

    np.savez(os.path.join(HERE, "tessellation_replication.npz"),
             az=AZ_COMMON, cands=np.array(cands),
             names=np.array([n for n, _ in scored]),
             scalar_geom=np.array([rows_a[n]["geom_only"]
                                   for n, _ in GIRDLE]),
             scalar_full=np.array([rows_a[n]["born_spec"]
                                   for n, _ in GIRDLE]),
             field=np.array([rows_b[n] for n, _ in GIRDLE]),
             cross_pred_meas=X, cross_meas_meas=Y,
             ident=np.array([[sc[n][c] for c in cands] for n, _ in scored]),
             ident_own=np.array([hits[n] for n, _ in scored]))


if __name__ == "__main__":
    main()
