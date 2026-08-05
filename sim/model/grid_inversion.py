"""GRIDDED PER-AREA INVERSION (rung 1 of the distributed model).

Jerome's design (2026-07-29): the global two-parameter fabric fit is
saturated at the specimen-realization floor (~+-7 deg axis at 100
grains - per-quarter analysis), because each diameter's specific grain
chain distorts the patterns. This module fits that structure instead
of averaging over it: the disk becomes a grid of cells, each carrying

    d_slowness  - local slowness perturbation (us/mm, vs background)
    d_scatter   - local backscatter strength (dB, vs the global model)

with the two data channels tied to the SAME cells:
    ToF residual(az)        = sum over ray cells of 2*L_cell*d_slowness
    coda residual(az, tbin) = d_scatter at the cell the beam occupies
                              at that time (r = c*t/2 along the ray)

Alignment in Jerome's sense = both residual sets are driven down by
one shared spatial field. First order the two solves are separable
linear least squares with Laplacian smoothness + ridge regularisation
(exactly solvable, no optimiser); the coupling (slowness-corrected
time-to-depth) is available as a second Gauss-Seidel pass.

FRAME (corrected 2026-07-29): rotated_grid shows the solver a specimen
rotated by -rot with the probe fixed on the +x rim, so in the fixed
SPECIMEN frame the beam enters at angle +rot. (The old -rot here was
reasoned from the legacy axes bug and mirrored every observation about
the x-axis relative to the truth maps - a prime suspect for the ~zero
truth correlations of v1.) All cell geometry is specimen-frame so the
recovered fields can be correlated against the per-cell TRUTH (labels
+ axes known for synthetic specimens - the twin's unfair advantage).
NOTE: legacy sweeps (config without axes_convention='rigid2') keep a
geometry-correct attribution under this fix, but their per-grain
c-axis contrasts do not match the truth maps (axes were spun +2*rot
relative to their grains) - regenerate before trusting per-cell work.

================================================================
TRUTH SCORING (rewritten 2026-07-31). READ THIS BEFORE BELIEVING
ANY "no structure recovered" VERDICT FROM THIS MODULE.
================================================================
v1 recorded "no validated structure" (scatter -0.19, slowness +0.01).
That verdict came from SCORING THE WRONG FIELD, not from a broken
inversion. Both v1 truth proxies were physically unrelated to the
quantity each channel actually estimates:

 (1) SCATTER. The coda model is  env_unit(az,t) * sqrt(E[R^2]).
     env_unit is the UNIT-CONTRAST Born envelope, i.e. it already
     contains the exact facet geometry of this realisation (which
     walls, where, how big). Dividing the measurement by it does NOT
     leave "how much scattering material is here" - it leaves a
     per-facet REFLECTION-CONTRAST quantity: how much stronger the
     real facet reflectivities in that cell are than the ODF-average
     sqrt(E[R^2]). The v1 proxy (local CV of the x-beam LOS grain
     speed) measures scattering material, is direction-locked to the
     x axis, and turns out to be an essentially independent field:
     measured corr(v1 proxy, correct truth) = -0.04. Scoring against
     it produced -0.19 for an estimate that scores +0.35 against the
     correct truth. The correct truth needs no new simulation:

         truth_db(az,t) = 20*log10(env_full / (env_unit*sqrt(E[R^2])))

     with env_full the SAME Born sum at full per-facet reflectivity,
     averaged per cell over the observations that hit the cell.

 (2) SLOWNESS. The v1 proxy used a single beam direction (+x) for
     every cell, but every cell is interrogated from ~180 distinct
     chord directions and qP speed in ice is strongly direction
     dependent. The direction-averaged local slowness is the only
     single scalar per cell the operator could recover; the v1
     x-beam version is a different field. (Even the direction-
     averaged truth is an approximation: a single scalar per cell
     cannot represent a direction-dependent anomaly, which is why
     the module also reports the operator-projected truth and the
     representability number corr(A_t @ x_proj, true residual).)

To make a future "no structure" claim separable from a scoring
error, every truth correlation is now reported next to
  * the resolution-matrix diagonal (how much of each cell the
    regularised operator can actually see),
  * a SYNTHETIC-RECOVERY correlation (a known field pushed through
    the same A / regularisation with realistic noise, then
    inverted) - the ceiling this design can reach, and
  * a rotation null (N_ROT_NULL rotations of the truth map) giving
    an empirical p-value.
Truth is also matched-smoothed (R @ truth) before correlating: an
unsmoothed truth vs a Laplacian-regularised estimate correlates
near zero by construction.

TWO OPERATOR BUGS FOUND WHILE DOING THE ABOVE (both fixed here):

 (a) STALE ENVELOPE CACHE. The singlemax_seed11_ppw6_rigid2 grid_env_cache.npz that
     v1 was reading did NOT hold the unit-contrast envelope this code
     computes. Backing it out of the saved residual: its envelope
     equalled the true unit-contrast one times er2(az)^-0.45, a pure
     per-azimuth gain of 0.95 dB rms (A2 = 1.32 dB at 118 deg, no
     time-bin dependence at all). v1 then multiplied by sqrt(er2)
     again, so the effective coda model carried almost NO ODF
     anisotropy and the residual kept the full er2(az) pattern - an
     azimuth-locked signal baked straight into the recovered map.
     That is why the cache is now version-tagged and carries its bin
     edges: any cache without ENV_CACHE_VERSION is discarded.

 (b) RAY QUADRATURE. The ToF operator sampled each chord at 400 equal
     steps. That is not converged: the resulting slowness field only
     reaches corr 0.63 with the exact-operator solution, and the truth
     correlation swings -0.04 / +0.17 / -0.03 / -0.02 as the step
     count goes 200 / 400 / 800 / 3200. Chord lengths are now
     analytic (_chord_lengths).
     Consequence: BOTH of the review's headline numbers were operator
     artefacts. Scatter +0.352 needed the stale cache (a freshly built
     cache gives +0.19 for the same truth on the same estimate), and
     slowness +0.142 needed n_seg=400 exactly (the converged operator
     gives -0.01 raw / +0.07 matched-smoothed at the same 5 mm cell).

CELL SIZE (2026-07-31): the two channels are not resolved at the same
scale, so they get separate grids.
  * coda/scatter: every (az, tbin) observation is attributed to a
    single point, so the sampling is point-like. MEASURED on
    singlemax_seed11_ppw6_rigid2, r(estimate, matched-smoothed Born truth) vs cell
    size: 2.5 mm +0.155, 5 mm +0.196, 6.25 mm +0.189, 10 mm +0.217,
    12.5 mm +0.185, 20 mm +0.186, 25 mm +0.152 - flat inside the
    rotation-null width (sd ~0.12), so there is nothing to tune.
    5 mm is kept: it gives the most scored cells (120) and hence the
    tightest null, and picking the 10 mm peak would be tuning to the
    maximum of a flat curve.
  * ToF/slowness: the mean grain equivalent diameter is 17.2 mm, so a
    5 mm cell is mostly aliasing, and 360 azimuths are only ~180
    independent DIAMETERS (az and az+180 are the same chord, and every
    chord passes through the centre). The numeric rank of A'A is 120
    of 400 cells at 5 mm and 12 of 25 at 20 mm. 20 mm is the default:
    it matches the grain size and doubles the mean resolution diagonal
    (0.25 -> 0.48). It does NOT rescue the channel - see the report.

WHAT THE CORRECTED SCORING GIVES (2026-07-31, 360 az each, r is
against the matched-smoothed truth, p from the rotation null):

  coda / Born facet contrast, 5 mm      r        null mean/sd     p
    singlemax_seed11_ppw6_rigid2                     +0.196     +0.070/0.116   0.167
    singlemax_seed23_ppw6_heldout_axis                       -0.265     -0.124/0.167   0.306
    isotropic_seed41_ppw6_calibration (isotropic control)     -0.102     -0.001/0.152   0.417
  ToF / direction-averaged, 20 mm
    singlemax_seed11_ppw6_rigid2                     +0.044     -0.075/0.121   0.694
    singlemax_seed23_ppw6_heldout_axis                       -0.446     -0.494/0.112   0.667
    isotropic_seed41_ppw6_calibration                         -0.093     -0.041/0.041   0.111

Nothing survives. The sign is not even consistent between specimens,
and singlemax_seed23_ppw6_heldout_axis's eye-catching ToF -0.446 sits exactly on its own
rotation-null mean of -0.494 - which is the whole reason the null is
reported next to every number.

The two channels fail for DIFFERENT and now-separable reasons:

  * coda: the scoring is fine and the design is capable. The
    synthetic-recovery ceiling at the measured signal fraction, using
    the MEASURED residual as the noise, is +0.71 to +0.86 - so if the
    per-facet contrast field were in the data at that level the
    inversion would return it at r > 0.7. It returns 0.2. The reason
    is visible one level up, in the data domain: corr(measured coda
    residual, exact Born truth residual) is +0.054 / -0.032 / +0.030
    over the three sweeps, i.e. ZERO, although the truth residual is
    28-41% of the measured residual variance (it would have to be
    ~0.53 if it were really there). The measured coda residual at
    2 MHz is simply not per-facet reflection contrast.

  * ToF: the scoring cannot be blamed and neither can the model - the
    DATA carry the specimen's true anisotropic ToF anomaly at
    corr +0.40 / +0.46 / +0.53, and at 20 mm cells a single scalar
    per cell can represent that anomaly at corr +0.72 to +0.93. The
    channel dies on SNR and rank: the true anomaly is only 13-22% of
    the measured residual variance, all 360 beams are DIAMETERS so
    there are ~180 independent lines all through the centre, and the
    operator rank is 12 of 25 cells. The synthetic-recovery ceiling
    at the measured signal fraction is +0.04 to +0.23 at EVERY cell
    size tried. No scoring fix can rescue this channel; it needs
    off-centre chords (a second element, or a translating probe).

CLI: python grid_inversion.py <sweep_name> [cell_mm_coda] [cell_mm_tof]
Outputs: <sweep>/grid_inversion.npz + printed residual-reduction and
truth-correlation numbers (the success metrics).
"""
import json
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys
import time

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

from scipy.signal import hilbert               # noqa: E402
from scipy import ndimage                      # noqa: E402
from scipy import sparse                       # noqa: E402
from scipy.sparse.linalg import spsolve        # noqa: E402

import born                                    # noqa: E402
import fit_fabric as FF                        # noqa: E402
import fit_sweep as FS                         # noqa: E402
import ladder                                  # noqa: E402
import sweep_runner as SW                      # noqa: E402
from specimen import DiskSpecimen              # noqa: E402

C_REF = 3850.0
CELL_MM_CODA = 5.0              # point-attributed coda observations
CELL_MM_TOF = 20.0              # ~grain scale; 5 mm is aliasing (see
                                # the cell-size study in the docstring)
CELL_MM = CELL_MM_CODA          # legacy alias (v1 used one grid)
T_BINS = np.arange(20e-6, 44e-6, 2e-6)         # coda time bins
LAM_SMOOTH = 3.0                # Laplacian weight
LAM_RIDGE = 0.3                 # ridge weight

ENV_CACHE_VERSION = 2           # bump whenever the cached envelope
                                # convention changes; v1 caches held
                                # only the unit-contrast envelope and
                                # had no version tag, so they are
                                # rejected and regenerated
SCORE_LEGACY_PROXIES = True     # also score the (wrong) v1 proxies,
                                # for the before/after comparison
N_ROT_NULL = 35                 # rotations of the truth map
SYNTH_TRIALS = 5

_PREP_CACHE = {}                # name -> cell-size-independent prep
_LAP_CACHE = {}


# ── grid geometry ───────────────────────────────────────────────────
def _cells(D_m, cell_mm=None):
    cell_mm = CELL_MM_CODA if cell_mm is None else cell_mm
    n = int(np.ceil(D_m / (cell_mm * 1e-3)))
    centers = (np.arange(n) + 0.5) * (2 * (D_m / 2) / n) - D_m / 2
    return n, centers


def _cell_of(x, y, D_m, n):
    ix = int((x + D_m / 2) / D_m * n)
    iy = int((y + D_m / 2) / D_m * n)
    if 0 <= ix < n and 0 <= iy < n:
        return iy * n + ix
    return -1


def _chord_lengths(D_m, n, th):
    """EXACT length of the az=th diameter inside each grid cell.

    v1 sampled the chord at n_seg=400 equal steps and binned. MEASURED
    2026-07-31: that quadrature has not converged - the recovered
    slowness field only reaches corr 0.63 with the exact-operator
    solution, and refining 200 -> 400 -> 800 -> 3200 steps swings the
    truth correlation over -0.04 / +0.17 / -0.03 / -0.02. The v1
    "slowness map" was therefore dominated by quadrature noise, and any
    correlation computed from it (in either direction) is meaningless.
    Analytic slab crossings remove the error entirely.
    """
    p0 = np.array([D_m / 2 * np.cos(th), D_m / 2 * np.sin(th)])
    dv = -np.array([np.cos(th), np.sin(th)])
    edges = np.linspace(-D_m / 2, D_m / 2, n + 1)
    ts = [0.0, D_m]
    for ax in (0, 1):
        if abs(dv[ax]) > 1e-12:
            t = (edges - p0[ax]) / dv[ax]
            ts += [float(x) for x in t if 0.0 < x < D_m]
    ts = np.unique(np.asarray(ts))
    mid = 0.5 * (ts[:-1] + ts[1:])
    ln = np.diff(ts)
    out = {}
    for m_, L in zip(mid, ln):
        pt = p0 + m_ * dv
        c = _cell_of(pt[0], pt[1], D_m, n)
        if c >= 0:
            out[c] = out.get(c, 0.0) + float(L)
    return out


def _laplacian(n):
    """2D grid Laplacian (n*n cells)."""
    if n in _LAP_CACHE:
        return _LAP_CACHE[n]
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, vals = [], [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = idx[:n - dy or n, :n - dx or n].ravel()
        b = idx[dy:, dx:].ravel()
        rows += [a, b, a, b]
        cols += [a, a, b, b]
        vals += [np.ones_like(a), -np.ones_like(a),
                 -np.ones_like(b), np.ones_like(b)]
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals).astype(float)
    L = sparse.coo_matrix((vals, (rows, cols)),
                          shape=(n * n, n * n)).tocsr()
    _LAP_CACHE[n] = L
    return L


def _solve_field(A, r, n):
    """Regularised linear solve for one field."""
    L = _laplacian(n)
    M = (A.T @ A).tocsr() + LAM_SMOOTH * L + LAM_RIDGE * sparse.identity(n * n)
    return spsolve(M.tocsc(), A.T @ r)


def _resolution(A, n):
    """Dense resolution matrix R = (A'A + reg)^-1 A'A.

    x_hat = R @ x_true for noise-free data, so R @ truth is the ONLY
    field the estimate can be expected to match; diag(R) says how much
    of each cell survives the regularisation.
    """
    AtA = (A.T @ A).toarray()
    M = AtA + LAM_SMOOTH * _laplacian(n).toarray() \
        + LAM_RIDGE * np.eye(n * n)
    return np.linalg.solve(M, AtA)


# ── honesty metrics ─────────────────────────────────────────────────
def _corr(a, b, mask):
    m = mask & np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 5:
        return np.nan, int(m.sum()), np.nan
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    N = int(m.sum())
    z = float(np.arctanh(np.clip(r, -0.999999, 0.999999)) * np.sqrt(N - 3))
    return r, N, z


def _rotation_null(est, truth, mask, n, n_rot=N_ROT_NULL):
    """Rotate the TRUTH MAP about the disk centre and re-correlate.

    A rotated truth has the same spatial spectrum, radial profile and
    histogram as the real one but no longer lines up with the grains,
    so |r| under rotation is the noise floor of this comparison. Far
    more honest than a cell-shuffle null, which destroys the smoothness
    both fields share and therefore always looks significant.
    """
    fin = np.isfinite(truth) & mask
    if int(fin.sum()) < 5:
        return dict(mean=np.nan, sd=np.nan, max_abs=np.nan, p=np.nan)
    t = np.where(np.isfinite(truth), truth, np.nan)
    t = t - np.nanmean(t[fin])
    t = np.nan_to_num(t).reshape(n, n)
    r_obs = _corr(est, truth, mask)[0]
    rs = []
    for a in np.arange(1, n_rot + 1) * (360.0 / (n_rot + 1)):
        tr = ndimage.rotate(t, a, reshape=False, order=1,
                            mode="constant", cval=0.0).ravel()
        rr = _corr(est, tr, mask)[0]
        if np.isfinite(rr):
            rs.append(rr)
    rs = np.asarray(rs)
    hit = int(np.sum(np.abs(rs) >= abs(r_obs))) if np.isfinite(r_obs) else 0
    return dict(mean=float(rs.mean()), sd=float(rs.std(ddof=1)),
                max_abs=float(np.abs(rs).max()),
                p=float((hit + 1) / (len(rs) + 1)), n=len(rs))


def _synth_recovery(A, n, x_true, mask, Res, obs_rms, fracs,
                    ntrial=SYNTH_TRIALS, seed=0, noise_pool=None,
                    roll_block=1):
    """Push a KNOWN cell field through the same operator + noise + reg.

    frac = fraction of the observed residual VARIANCE that is genuinely
    the cell field (the rest is noise). Returns {frac: (r_vs_truth,
    r_vs_smoothed_truth)} - the ceiling this design can reach, so a low
    measured correlation can be attributed to the design rather than to
    the estimate.

    noise_pool: if given, the MEASURED residual is used as the noise
    instead of white Gaussian, circularly shifted by a random whole
    number of azimuths (roll_block rows per azimuth) so it keeps its
    real depth/azimuth correlation structure but no longer lines up
    with the truth. White noise flatters this design badly - the real
    model error is strongly cell-coherent, and a coherent error is
    exactly what a per-cell inversion cannot reject.
    """
    xt = np.where(np.isfinite(x_true), x_true, 0.0).astype(float)
    xt = xt - xt[mask].mean()
    d0 = A @ xt
    out = {}
    if float(d0.std()) <= 0:
        return out
    xsm = Res @ xt
    rng = np.random.default_rng(seed)
    nrows = d0.shape[0]
    for f in fracs:
        scale = obs_rms * np.sqrt(f) / float(d0.std())
        nrms = obs_rms * np.sqrt(max(1.0 - f, 0.0))
        rr, rrs = [], []
        for _ in range(ntrial):
            if noise_pool is None:
                nz = rng.normal(0.0, nrms, nrows)
            else:
                k = int(rng.integers(1, max(nrows // roll_block, 2)))
                nz = np.roll(np.asarray(noise_pool, float), k * roll_block)
                nz = nz - nz.mean()
                nz = nz * (nrms / max(float(nz.std()), 1e-30))
            xh = _solve_field(A, d0 * scale + nz, n)
            rr.append(np.corrcoef(xh[mask], xt[mask])[0, 1])
            rrs.append(np.corrcoef(xh[mask], xsm[mask])[0, 1])
        out[float(f)] = (float(np.mean(rr)), float(np.mean(rrs)))
    return out


# ── specimen truth helpers ──────────────────────────────────────────
def _cell_grain_weights(labels, h_spec, D_m, n):
    """(n*n, n_grains) volume fractions + a has-material mask."""
    nx_s, ny_s, nz_s = labels.shape
    ng = int(labels.max()) + 1
    xi = np.minimum(((np.arange(nx_s) + 0.5) * h_spec
                     / (D_m / n)).astype(int), n - 1)
    w = np.zeros((n * n, ng))
    for iz in range(nz_s):
        sl = labels[:, :, iz]
        ok = sl >= 0
        # labels axes are (x, y): cell index must match _cell_of's
        # iy*n + ix convention
        cells = ((xi[None, :ny_s] * n) + xi[:, None])[ok]
        np.add.at(w, (cells, sl[ok]), 1.0)
    tot = w.sum(axis=1)
    okc = tot > 0
    wn = np.zeros_like(w)
    wn[okc] = w[okc] / tot[okc, None]
    return wn, okc


def _grain_speed(axes_t, dirvec):
    """qP speed of every grain for propagation along dirvec."""
    cs = np.clip(np.abs(axes_t @ dirvec), 0.0, 1.0)
    return np.interp(np.arccos(cs), FF.F._PSI, FF.F._VQP)


def _true_tof(build, rots, D_m, n_s=2000):
    """Exact two-way ToF anomaly of THIS specimen, azimuth by azimuth.

    Integrated at the SPECIMEN voxel resolution (not on the inversion
    grid), so the number is a property of the specimen and the beam
    geometry alone, and the ToF signal fraction it implies does not
    move when the cell size changes. Each chord column is averaged
    over thickness, which is what the through-thickness pencil beam
    integrates.
    """
    labels = np.asarray(build["labels"])
    axes_t = np.asarray(build["axes"], float)
    h = float(build["h"])
    nx, ny, _ = labels.shape
    R = D_m / 2
    seg = D_m / n_s
    s = (np.arange(n_s) + 0.5) * seg
    out = np.zeros(len(rots))
    for i, a in enumerate(rots):
        th = np.radians(float(a))
        dv = np.array([-np.cos(th), -np.sin(th), 0.0])
        x = R * np.cos(th) + s * dv[0]
        y = R * np.sin(th) + s * dv[1]
        ix = np.clip(np.rint(x / h + (nx - 1) / 2).astype(int), 0, nx - 1)
        iy = np.clip(np.rint(y / h + (ny - 1) / 2).astype(int), 0, ny - 1)
        lab = labels[ix, iy, :]
        ok = lab >= 0
        Sg = 1e3 / _grain_speed(axes_t, dv)        # us/mm per grain
        val = np.where(ok, Sg[np.where(ok, lab, 0)], 0.0)
        cnt = ok.sum(axis=1)
        col = np.where(cnt > 0, val.sum(axis=1) / np.maximum(cnt, 1), 0.0)
        out[i] = 2.0 * seg * 1e3 * float(col.sum())
    return out - out.mean()


# ── Born envelope cache (unit-contrast MODEL + full-contrast TRUTH) ──
def _bin_rms(env, fs):
    env = np.asarray(env, float)
    return np.array([np.sqrt((env[int(T_BINS[k] * fs):
                                  int(T_BINS[k + 1] * fs)] ** 2).mean())
                     for k in range(len(T_BINS) - 1)])


def _env_cache(d, build, mcfg, rots, fs_m):
    """Per-azimuth binned Born envelopes, cached in the sweep dir.

    env_unit: unit-contrast (the MODEL denominator, as in fit_fabric).
    env_full: full per-facet reflectivity (the TRUTH numerator). Their
    ratio, divided by sqrt(E[R^2]), is the exact per-facet contrast
    residual the coda channel is trying to map.

    The cache is version-tagged AND carries its bin edges: a v1 cache
    (unit only, no version key) or a cache built with different T_BINS
    is rejected and regenerated. Regeneration is ~95 s per contrast
    per 360 azimuths on CPU.
    """
    path = os.path.join(d, "grid_env_cache.npz")
    have_u, have_f = {}, {}
    if os.path.exists(path):
        try:
            with np.load(path) as z:
                keys = set(z.files)
                ok = ({"cache_version", "rots", "env_unit", "env_full",
                       "t_bins"} <= keys
                      and int(z["cache_version"]) == ENV_CACHE_VERSION
                      and z["t_bins"].shape == T_BINS.shape
                      and np.allclose(z["t_bins"], T_BINS))
                if ok:
                    have_u = {int(r): row for r, row
                              in zip(z["rots"], z["env_unit"])}
                    have_f = {int(r): row for r, row
                              in zip(z["rots"], z["env_full"])}
        except (OSError, ValueError, KeyError):
            have_u, have_f = {}, {}
    missing = [rt for rt in rots
               if int(rt) not in have_u or int(rt) not in have_f]
    if missing:
        print(f"[grid] Born envelope cache: {len(missing)} azimuths x 2 "
              f"contrasts to build", flush=True)
        t0 = time.time()
        for i, rt in enumerate(missing):
            az = np.radians(float(rt))
            _, eu = born.boundary_scatter(dict(build), mcfg, azimuth_rad=az,
                                          unit_contrast=True)
            _, ef = born.boundary_scatter(dict(build), mcfg, azimuth_rad=az,
                                          unit_contrast=False)
            have_u[int(rt)] = _bin_rms(eu, fs_m)
            have_f[int(rt)] = _bin_rms(ef, fs_m)
            if i % 60 == 0:
                print(f"[grid] envelopes {i}/{len(missing)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        ks = sorted(set(have_u) & set(have_f))
        tmp = path[:-4] + f".tmp{os.getpid()}.npz"
        try:
            np.savez(tmp, rots=np.asarray(ks),
                     env_unit=np.asarray([have_u[k] for k in ks]),
                     env_full=np.asarray([have_f[k] for k in ks]),
                     t_bins=T_BINS, cache_version=ENV_CACHE_VERSION)
            SW._atomic_replace(tmp, path, critical=False)
        except OSError:
            pass
    unit = np.asarray([have_u[int(rt)] for rt in rots])
    full = np.asarray([have_f[int(rt)] for rt in rots])
    return unit, full


# ── cell-size-independent preparation ───────────────────────────────
def _prepare(cfg, name, d, reuse=True):
    if reuse and name in _PREP_CACHE:
        return _PREP_CACHE[name]
    with open(os.path.join(d, "fit_result.json"), encoding="utf-8") as f:
        fit = json.load(f)
    alpha_p = float(fit.get("alpha_probe_deg", -fit["alpha_deg"])) % 180
    kappa = float(fit["kappa"])
    D_m = cfg["diameter_mm"] * 1e-3

    # ── data: raw per-azimuth ToF + per-(az, tbin) coda dB ──────────
    FS.AZ_SMOOTH_DEG = 0.0        # the grid fits the structure itself
    data = FS.sweep_data(cfg)
    rots = np.array([r_["rot"] for r_ in data], float)
    tof_m = np.array([r_["tof_us"] for r_ in data])
    print(f"[grid] {len(rots)} azimuths", flush=True)

    # ── background model (the global fit) ───────────────────────────
    geo, _ = FS.sweep_geo(cfg, list(rots))
    _, _, tof_p, _ = FF.predict_basis(alpha_p, kappa, list(rots), geo)
    t_ = tof_p - tof_p.mean()
    b_t = max(float(t_ @ (tof_m - tof_m.mean())) / float(t_ @ t_), 0.2)
    tof_bg = tof_m.mean() + b_t * t_
    r_tof = tof_m - tof_bg                     # us, per azimuth

    f0 = cfg["f0_mhz"] * 1e6
    h = C_REF / f0 / cfg["ppw"]
    sp = DiskSpecimen(diameter_m=D_m,
                      thickness_m=cfg["thickness_mm"] * 1e-3,
                      n_grains=cfg["n_grains"], size_cv=cfg["size_cv"],
                      concentration=cfg["concentration"],
                      spatial_corr=cfg["spatial_corr"],
                      fabric_axis=tuple(cfg["fabric_axis"]),
                      seed=cfg["seed"])
    build = sp.build(h)
    mcfg = ladder.standard_cfg()
    mcfg.probe.f0 = f0
    fs_m = 1.0 / ladder.DT
    env_unit, env_full = _env_cache(d, build, mcfg, rots, fs_m)

    psi = FS._psi_from_axis(rots, alpha_p)
    cal = FS.shape_calibration(psi, cfg["f0_mhz"])
    er2 = np.array([FF.odf_moments(
        np.array([np.cos(np.radians(alpha_p)),
                  np.sin(np.radians(alpha_p)), 0.0]), kappa,
        -np.array([np.cos(np.radians(rt)), np.sin(np.radians(rt)),
                   0.0]))[0] for rt in rots])

    # EXACT per-facet-contrast truth residual, same units + same
    # denominator as the measured coda residual below
    truth_db = 20 * np.log10(env_full
                             / (env_unit * np.sqrt(np.maximum(er2, 1e-30))
                                [:, None]))

    # ── measured coda residual, one row per (az, tbin) ──────────────
    obs_az, obs_k, obs_depth, res_db, obs_truth = [], [], [], [], []
    for i, rt in enumerate(rots):
        with np.load(os.path.join(d, f"az{int(rt):03d}.npz")) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        e = np.abs(hilbert(tr))
        for k in range(len(T_BINS) - 1):
            s0, s1 = int(T_BINS[k] * fs), int(T_BINS[k + 1] * fs)
            meas = np.sqrt((e[s0:s1] ** 2).mean())
            mod = env_unit[i, k] * np.sqrt(max(er2[i], 1e-30))
            if meas <= 0 or mod <= 0 or not np.isfinite(truth_db[i, k]):
                continue
            t_mid = 0.5 * (T_BINS[k] + T_BINS[k + 1])
            # subtract the Ricker envelope-peak source delay (1.2/f0,
            # 0.6 us at 2 MHz) before converting time to depth: without
            # it every cell attribution sat ~1.2 mm too deep
            obs_az.append(i)
            obs_k.append(k)
            obs_depth.append(C_REF * max(t_mid - 1.2 / f0, 0.0) / 2.0)
            res_db.append(20 * np.log10(meas / mod) - cal[i])
            obs_truth.append(truth_db[i, k])
    res_db = np.asarray(res_db)
    obs_truth = np.asarray(obs_truth)
    # remove the free gain from BOTH (absolute level is the global
    # fit's nuisance; the grid fits spatial CONTRAST)
    res_db = res_db - np.median(res_db)
    obs_truth = obs_truth - np.median(obs_truth)

    # the TRUE anisotropic ToF anomaly of this specimen, with the SAME
    # 2-phi background removed that the measured ToF gets (specimen
    # resolution, so it is independent of the inversion cell size)
    tof_true = _true_tof(build, rots, D_m)
    tt_res = tof_true - (float(t_ @ tof_true) / float(t_ @ t_)) * t_

    prep = dict(name=name, d=d, cfg=cfg, D_m=D_m, f0=f0, rots=rots,
                r_tof=r_tof, t_basis=t_, build=build, alpha_p=alpha_p,
                kappa=kappa, res_db=res_db, obs_truth=obs_truth,
                tof_true=tof_true, tt_res=tt_res,
                obs_az=np.asarray(obs_az), obs_k=np.asarray(obs_k),
                obs_depth=np.asarray(obs_depth),
                truth_db=truth_db, er2=er2)
    if reuse:
        _PREP_CACHE[name] = prep
    return prep


# ── the two channels ────────────────────────────────────────────────
def _coda_channel(prep, cell_mm, verbose=True):
    D_m, R = prep["D_m"], prep["D_m"] / 2
    n, _ = _cells(D_m, cell_mm)
    NC = n * n
    rots = prep["rots"]
    th = np.radians(rots)
    probe = np.stack([R * np.cos(th), R * np.sin(th)], axis=1)
    bdir = -np.stack([np.cos(th), np.sin(th)], axis=1)
    pts = probe[prep["obs_az"]] + prep["obs_depth"][:, None] \
        * bdir[prep["obs_az"]]
    cell = np.array([_cell_of(p[0], p[1], D_m, n) for p in pts])
    keep = cell >= 0
    r_coda = prep["res_db"][keep]
    t_obs = prep["obs_truth"][keep]
    cell = cell[keep]
    M = len(r_coda)
    A_c = sparse.coo_matrix((np.ones(M), (np.arange(M), cell)),
                            shape=(M, NC)).tocsr()

    d_scatter = _solve_field(A_c, r_coda, n)
    rc_after = r_coda - A_c @ d_scatter

    # ── truth #1 (CORRECT): exact Born per-facet contrast, per cell ──
    cnt = np.bincount(cell, minlength=NC)
    tr_born = np.full(NC, np.nan)
    tr_born[cnt > 0] = np.bincount(cell, t_obs, NC)[cnt > 0] / cnt[cnt > 0]

    # ── truth #2/#3 (v1 proxies, WRONG - kept for comparison) ───────
    build = prep["build"]
    labels = np.asarray(build["labels"])
    axes_t = np.asarray(build["axes"], float)
    wn, okc = _cell_grain_weights(labels, float(build["h"]), D_m, n)
    tr_xcv = np.full(NC, np.nan)
    tr_dcv = np.full(NC, np.nan)
    if SCORE_LEGACY_PROXIES:
        # v1: local CV of the LOS grain speed for a FIXED +x beam.
        # Measures how much scattering MATERIAL sits in the cell, not
        # the per-facet reflection contrast the residual carries, and
        # is locked to one direction out of ~180 interrogating ones.
        v_x = _grain_speed(axes_t, np.array([1.0, 0.0, 0.0]))
        m_v = wn @ v_x
        s_v = wn @ (v_x ** 2) - m_v ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            tr_xcv[okc] = np.sqrt(np.maximum(s_v[okc], 0)) / m_v[okc]
        # the same proxy, direction-averaged over the real beams (the
        # honest version of the WRONG quantity - still not the residual)
        acc = np.zeros(NC)
        dirs = np.stack([-np.cos(th), -np.sin(th), np.zeros_like(th)], 1)
        for j in range(0, len(dirs), 4):
            vg = _grain_speed(axes_t, dirs[j])
            m_ = wn @ vg
            s_ = wn @ (vg ** 2) - m_ ** 2
            with np.errstate(invalid="ignore", divide="ignore"):
                acc += np.where(okc, np.sqrt(np.maximum(s_, 0))
                                / np.maximum(m_, 1e-9), 0.0)
        tr_dcv[okc] = acc[okc] / len(range(0, len(dirs), 4))

    cover = np.asarray((A_c.T @ np.ones(M)) > 0).ravel()
    mask = cover & okc & np.isfinite(tr_born)
    Res = _resolution(A_c, n)
    frac = float(min((t_obs.std() / r_coda.std()) ** 2, 1.0))
    out = dict(kind="coda", n=n, cell_mm=cell_mm, NC=NC, A=A_c, Res=Res,
               est=d_scatter, mask=mask, n_obs=M,
               rms_before=float(r_coda.std()), rms_after=float(rc_after.std()),
               res_diag_mean=float(np.diag(Res)[mask].mean()),
               res_diag_max=float(np.diag(Res).max()),
               data_corr=float(np.corrcoef(r_coda, t_obs)[0, 1]),
               signal_frac=frac,
               truths={"born_facet_contrast (CORRECT)": tr_born,
                       "v1 x-beam LOS-speed CV (WRONG)": tr_xcv,
                       "dir-averaged LOS-speed CV (WRONG)": tr_dcv},
               headline="born_facet_contrast (CORRECT)",
               obs_rms=float(r_coda.std()), resid=r_coda,
               roll_block=len(T_BINS) - 1)
    if verbose:
        print(f"[grid] coda: {n}x{n} cells of {cell_mm} mm, {M} obs, "
              f"{int(mask.sum())} scored cells", flush=True)
    return out


def _tof_channel(prep, cell_mm, verbose=True):
    D_m = prep["D_m"]
    n, _ = _cells(D_m, cell_mm)
    NC = n * n
    rots = prep["rots"]
    th = np.radians(rots)
    rows, cols, vals = [], [], []
    for i, t_ in enumerate(th):
        for c, L in _chord_lengths(D_m, n, t_).items():
            rows.append(i)
            cols.append(c)
            # two-way path, mm, per (us/mm) slowness perturbation
            vals.append(2.0 * L * 1e3)
    A_t = sparse.coo_matrix((vals, (rows, cols)),
                            shape=(len(rots), NC)).tocsr()
    r_tof = prep["r_tof"]
    d_slow = _solve_field(A_t, r_tof, n)
    rt_after = r_tof - A_t @ d_slow

    build = prep["build"]
    labels = np.asarray(build["labels"])
    axes_t = np.asarray(build["axes"], float)
    wn, okc = _cell_grain_weights(labels, float(build["h"]), D_m, n)

    # per-cell slowness for EVERY interrogating beam direction
    dirs = np.stack([-np.cos(th), -np.sin(th), np.zeros_like(th)], axis=1)
    S_gd = np.stack([1e3 / _grain_speed(axes_t, dv) for dv in dirs], axis=1)
    slow_cd = wn @ S_gd                        # (NC, n_az) us/mm

    # ── truth #1 (CORRECT for a single-scalar-per-cell model):
    #    DIRECTION-AVERAGED local slowness ────────────────────────────
    tr_diravg = np.full(NC, np.nan)
    v = slow_cd[okc].mean(axis=1)
    tr_diravg[okc] = v - v.mean()

    # ── truth #2: the operator-projected true anisotropic residual ──
    # (tt_res is computed at SPECIMEN resolution in _prepare, so this
    # is the best cell field that could explain the real anomaly)
    tt_res = prep["tt_res"]
    tr_proj = _solve_field(A_t, tt_res, n)

    # ── truth #3 (v1 proxy, WRONG - kept for comparison) ────────────
    tr_x = np.full(NC, np.nan)
    if SCORE_LEGACY_PROXIES:
        # v1 evaluated the local slowness for a FIXED +x beam only,
        # although every cell is crossed from ~180 directions and qP
        # speed in ice is strongly direction dependent.
        vx = wn @ _grain_speed(axes_t, np.array([1.0, 0.0, 0.0]))
        s_x = np.where(vx > 0, 1e3 / np.maximum(vx, 1e-9), np.nan)
        tr_x[okc] = s_x[okc] - np.nanmean(s_x[okc])

    cover = np.asarray((A_t.T @ np.ones(len(rots))) > 2).ravel()
    mask = cover & okc
    Res = _resolution(A_t, n)
    frac = float(min((tt_res.std() / r_tof.std()) ** 2, 1.0))
    repr_ = float(np.corrcoef(A_t @ tr_proj, tt_res)[0, 1])
    # every beam is a DIAMETER, so az and az+180 give the identical
    # chord: 360 azimuths are only ~180 independent lines, all through
    # the centre. The numeric rank says how many cell combinations the
    # data can constrain at all.
    ev = np.linalg.eigvalsh((A_t.T @ A_t).toarray())
    rank = int((ev > 1e-6 * ev.max()).sum())
    out = dict(kind="tof", n=n, cell_mm=cell_mm, NC=NC, A=A_t, Res=Res,
               est=d_slow, mask=mask, n_obs=len(rots),
               rms_before=float(r_tof.std()), rms_after=float(rt_after.std()),
               res_diag_mean=float(np.diag(Res)[mask].mean()),
               res_diag_max=float(np.diag(Res).max()),
               data_corr=float(np.corrcoef(r_tof, tt_res)[0, 1]),
               signal_frac=frac, representability=repr_, rank=rank,
               truths={"direction-averaged slowness (CORRECT)": tr_diravg,
                       "operator-projected true ToF": tr_proj,
                       "v1 x-beam slowness (WRONG)": tr_x},
               headline="direction-averaged slowness (CORRECT)",
               obs_rms=float(r_tof.std()), resid=r_tof, roll_block=1)
    if verbose:
        print(f"[grid] tof: {n}x{n} cells of {cell_mm} mm, "
              f"{int(mask.sum())} scored cells, operator rank "
              f"{rank}/{NC}", flush=True)
    return out


def _score(ch, nulls=True, synth=True):
    """Fill in correlations, nulls and the synthetic-recovery ceiling."""
    est, mask, n, Res = ch["est"], ch["mask"], ch["n"], ch["Res"]
    rep = {}
    for label, truth in ch["truths"].items():
        if not np.any(np.isfinite(truth)):
            continue
        t_sm = Res @ np.nan_to_num(truth - np.nanmean(truth[mask]))
        r_raw, N, z_raw = _corr(est, truth, mask)
        r_sm, _, z_sm = _corr(est, t_sm, mask)
        e = dict(r_raw=r_raw, r_smooth=r_sm, N=N, z_raw=z_raw, z_smooth=z_sm)
        if nulls:
            e["null_raw"] = _rotation_null(est, truth, mask, n)
            e["null_smooth"] = _rotation_null(est, t_sm, mask, n)
        if synth:
            fr = sorted({1.0, round(ch["signal_frac"], 3)})
            e["synth"] = _synth_recovery(ch["A"], n, truth, mask, Res,
                                         ch["obs_rms"], fr)
            e["synth_real"] = _synth_recovery(
                ch["A"], n, truth, mask, Res, ch["obs_rms"], fr,
                noise_pool=ch["resid"], roll_block=ch["roll_block"])
        rep[label] = e
    ch["report"] = rep
    return ch


def _print_channel(ch):
    unit = "dB" if ch["kind"] == "coda" else "us"
    print(f"\n=== {ch['kind'].upper()} channel: {ch['n']}x{ch['n']} cells "
          f"of {ch['cell_mm']} mm ===")
    print(f"  residual rms {ch['rms_before']:.3f} -> {ch['rms_after']:.3f} "
          f"{unit}   ({ch['n_obs']} observations)")
    print(f"  resolution diag over scored cells: mean "
          f"{ch['res_diag_mean']:.3f}  max {ch['res_diag_max']:.3f}"
          + (f"   operator rank {ch['rank']}/{ch['NC']}"
             if "rank" in ch else ""))
    print(f"  data domain: corr(measured residual, true residual) "
          f"{ch['data_corr']:+.3f}   signal fraction {ch['signal_frac']:.2f}")
    if "representability" in ch:
        print(f"  representability corr(A @ best cell field, true residual) "
              f"{ch['representability']:+.3f}   <- a single scalar per cell "
              f"cannot carry a direction-dependent anomaly")
    print("  truth field                            r(raw)  r(R@truth)   "
          "rot-null mean/sd/max|r|      p")
    for label, e in ch["report"].items():
        nl = e.get("null_smooth", {})
        print(f"    {label:38s} {e['r_raw']:+.3f}   {e['r_smooth']:+.3f}   "
              f"{nl.get('mean', np.nan):+.3f}/{nl.get('sd', np.nan):.3f}/"
              f"{nl.get('max_abs', np.nan):.3f}  {nl.get('p', np.nan):.3f}")
        for tag, key in (("white noise", "synth"),
                         ("REAL residual as noise", "synth_real")):
            sy = e.get(key, {})
            if sy:
                print(f"      synthetic recovery, {tag:22s} "
                      + "   ".join(f"signal={f:.2f} -> r={v[1]:+.3f}"
                                   for f, v in sorted(sy.items(),
                                                      reverse=True)))
    print(f"  (N = {list(ch['report'].values())[0]['N']} scored cells; "
          f"synthetic recovery is r vs the matched-smoothed truth)")


# ── driver ──────────────────────────────────────────────────────────
def invert(cfg, cell_mm_coda=None, cell_mm_tof=None, reuse=True,
           save=True, nulls=True, synth=True):
    name = cfg["name"]
    d = SW.sweep_dir(name)
    lock = os.path.join(d, "grid.lock")
    with open(lock, "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    try:
        return _invert_locked(cfg, name, d, cell_mm_coda, cell_mm_tof,
                              reuse, save, nulls, synth)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _invert_locked(cfg, name, d, cell_mm_coda, cell_mm_tof, reuse, save,
                   nulls, synth):
    cc = CELL_MM_CODA if cell_mm_coda is None else float(cell_mm_coda)
    ct = CELL_MM_TOF if cell_mm_tof is None else float(cell_mm_tof)
    prep = _prepare(cfg, name, d, reuse=reuse)
    coda = _score(_coda_channel(prep, cc), nulls=nulls, synth=synth)
    tof = _score(_tof_channel(prep, ct), nulls=nulls, synth=synth)
    print(f"\n########## {name} ##########")
    _print_channel(coda)
    _print_channel(tof)

    if save:
        hc = coda["report"][coda["headline"]]
        ht = tof["report"][tof["headline"]]
        tmp = os.path.join(d, "grid_inversion.tmp.npz")
        np.savez(
            tmp,
            # geometry
            n=coda["n"], cell_mm=cc, n_coda=coda["n"], cell_mm_coda=cc,
            n_tof=tof["n"], cell_mm_tof=ct,
            # fields
            d_scatter=coda["est"].reshape(coda["n"], coda["n"]),
            d_slow=tof["est"].reshape(tof["n"], tof["n"]),
            truth_scatter=coda["truths"][coda["headline"]].reshape(
                coda["n"], coda["n"]),
            truth_slow=tof["truths"][tof["headline"]].reshape(
                tof["n"], tof["n"]),
            truth_slow_projected=tof["truths"][
                "operator-projected true ToF"].reshape(tof["n"], tof["n"]),
            mask_coda=coda["mask"].reshape(coda["n"], coda["n"]),
            mask_tof=tof["mask"].reshape(tof["n"], tof["n"]),
            res_diag_coda=np.diag(coda["Res"]).reshape(coda["n"], coda["n"]),
            res_diag_tof=np.diag(tof["Res"]).reshape(tof["n"], tof["n"]),
            # residuals
            coda_rms_before=coda["rms_before"],
            coda_rms_after=coda["rms_after"],
            tof_rms_before=tof["rms_before"], tof_rms_after=tof["rms_after"],
            # headline scores (matched-smoothed is the honest one)
            corr_scatter=hc["r_smooth"], corr_scatter_raw=hc["r_raw"],
            corr_slow=ht["r_smooth"], corr_slow_raw=ht["r_raw"],
            p_scatter=hc.get("null_smooth", {}).get("p", np.nan),
            p_slow=ht.get("null_smooth", {}).get("p", np.nan),
            synth_ceiling_scatter=json.dumps(hc.get("synth", {})),
            synth_ceiling_slow=json.dumps(ht.get("synth", {})),
            signal_frac_coda=coda["signal_frac"],
            signal_frac_tof=tof["signal_frac"],
            representability_tof=tof.get("representability", np.nan),
            report=json.dumps({
                "coda": {k: {kk: vv for kk, vv in v.items()
                             if not kk.startswith("synth")}
                         for k, v in coda["report"].items()},
                "tof": {k: {kk: vv for kk, vv in v.items()
                            if not kk.startswith("synth")}
                        for k, v in tof["report"].items()}}, default=float))
        SW._atomic_replace(tmp, os.path.join(d, "grid_inversion.npz"),
                           critical=True)
        print("[grid] saved grid_inversion.npz", flush=True)
    return dict(coda=coda, tof=tof, prep=prep)


if __name__ == "__main__":
    _cc = float(sys.argv[2]) if len(sys.argv) > 2 else None
    _ct = float(sys.argv[3]) if len(sys.argv) > 3 else None
    invert(SW.load(sys.argv[1]), cell_mm_coda=_cc, cell_mm_tof=_ct)
