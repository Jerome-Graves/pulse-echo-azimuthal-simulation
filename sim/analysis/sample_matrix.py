"""The LEFT side of the observable matrix: every sample property that
could plausibly influence what the probe records, measured from the
cached tessellations and the orientation draws alone.

No wave data is read here and no solver is run. The claim this module
supports is narrow and mechanical: that the specimen side of the
property-by-observable matrix can be enumerated exhaustively and cheaply,
so that the screening stage tests cells rather than intuitions.

TWO REGIMES, and the module writes one file for each.

  BETWEEN specimens.  One row per specimen, n = 8 girdle tessellations,
  8 single maxima on the same eight tessellations, and five controls on
  the seed 11 geometry. A property whose spread across the eight girdle
  rows is small cannot be tested between specimens however interesting
  it is, so report_variation prints that spread for every column and
  names the columns that are effectively constant.

  WITHIN a specimen, across azimuth.  Every property that has a beam
  direction in it is computed again at each of 60 azimuths, 6 degrees
  apart, for every specimen. This is where the statistical power is.
  The 30-azimuth common grid of the multi-tessellation sweeps is every
  second row of it.

180-DEGREE PERIODICITY, which the screening stage has to know per column
and not in general. A quasi-longitudinal speed depends on |c . n| and is
therefore exactly invariant under n -> -n, so a 30-azimuth sweep offers
15 distinct alignments of any speed-derived predictor and the circular
shift null must be run on 15. A marched crossing count is NOT invariant:
the beam enters the disc from the opposite side and the coda gate then
covers a different part of the specimen. report_periodicity measures the
correlation of x(az) with x(az + 180) column by column so the two kinds
are separated by measurement rather than by assumption.

WHERE THE SPECIMENS COME FROM. out/tesscache holds the production
label volumes at ppw 8 for the eight girdle seeds and for four of the
single maxima. DiskSpecimen.build draws the Laguerre seed points and
weights before it draws any c-axis, so a girdle and a single maximum of
the same seed carry a bit-identical tessellation and unrelated
orientations; this module checks that on the four seeds where both are
cached, and then replays the random stream to produce the single-maximum
c-axes of seeds 53, 71 and 89, whose sweeps are still running. The
replay is checked against all eight cached draws and reproduces them
bit-exactly, so those three rows are measured and not modelled.

  MEASURED   everything read from or derived from a cached tessellation,
             the c-axis draws, and the single-crystal elastic constants.
  INFERRED   nothing in the saved arrays. The two derived quantities that
             involve a modelling choice are named as such where they are
             printed: the acoustically dark fraction needs a threshold,
             and the group-velocity skew needs a numerical derivative of
             the qP phase speed.

CUDA IS NEVER TOUCHED. DiskSpecimen._label_grid_gpu is replaced with an
exact all-seeds float64 argmin in NumPy before any build can reach it,
which is the same substitution analysis/tessellation_replication.py
makes and for the same reason.

READS
  out/tesscache/tess_s<seed>_p8_k-8.npz      eight girdle tessellations
  out/tesscache/tess_s<seed>_p8_k3.93.npz    four single maxima
  sim/results/ensemble.npz                   for the crossing-count
                                             reconciliation only
WRITES
  sim/results/sample_matrix.npz              per-specimen matrix
  sim/results/sample_matrix.csv              the same, human readable
  sim/results/sample_matrix_azimuth.npz      specimen by azimuth cube
  sim/results/sample_matrix_azimuth.csv      the same, long form
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")

import _t2_common as C2                                   # noqa: E402
import facet_predictors as FP                             # noqa: E402
import forward as FW                                      # noqa: E402
from ringfwi import anisotropy as AN                      # noqa: E402
from specimen import DiskSpecimen, sample_watson          # noqa: E402

TESS = os.path.join(ROOT, "out", "tesscache")
RESULTS = os.path.join(ROOT, "sim", "results")

C_REF, F0, DIA, THK = 3850.0, 2.0e6, 0.100, 0.035
N_GRAIN_TARGET, SIZE_CV = 100, 0.35
PPW = 8.0
CODA_W = C2.CODA_W                       # (24 us, 36 us) two-way
GATE_S = (CODA_W[0] * C_REF / 2.0, CODA_W[1] * C_REF / 2.0)
HALF_DEG = np.degrees(FP.HALF)           # 8.9 deg far-field half angle
N_RAY_COL = 200                          # cone rays for the beam column
NLAT_PAPER = 7                           # the lattice facet_predictors uses

# 60 azimuths at 6 degrees. The 30-azimuth common grid of the
# multi-tessellation sweeps is every second entry, so both regimes are
# served by one cube.
AZ = np.arange(0, 360, 6)

GIRDLE_KAPPA, GIRDLE_AXIS = -8.0, (1.0, 0.0, 0.0)
SINGLE_KAPPA, SINGLE_AXIS = 3.93, (0.866, 0.5, 0.0)
SEEDS = [11, 7, 17, 23, 41, 53, 71, 89]  # ensemble.npz row order

# Thresholds for the acoustically dark fraction, as normal-incidence
# pressure reflection coefficients. Both are printed; neither is fitted.
DARK_R = (1e-3, 1e-2)

# Permutations for the orientation-clustering null. The label volume is
# held fixed and the c-axes are shuffled among the grains, so the null
# keeps the geometry and destroys only the spatial arrangement.
N_PERM = 500


def _label_plane_by_plane(x, z, R, seeds, weights):
    """Exact all-seeds Laguerre argmin, one z-plane at a time, in NumPy.

    Substituted for the CuPy path of specimen.py so that nothing in this
    module can reach CUDA. Only the iso_gcal control needs it; every
    other specimen comes from the cache.
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


# ------------------------------------------------------------------ load

def cache_path(seed, kappa, ppw=PPW):
    return os.path.join(TESS, "tess_s%d_p%g_k%g.npz" % (seed, ppw, kappa))


def load_cached(seed, kappa, ppw=PPW):
    """(labels int32, axes, seeds, h) of one cached tessellation."""
    with np.load(cache_path(seed, kappa, ppw)) as z:
        return (np.asarray(z["labels"]).astype(np.int32),
                np.asarray(z["axes"], float),
                np.asarray(z["seeds"], float), float(z["h"]))


def replay_axes(seed, n_real, axis, kappa):
    """Reproduce build()'s c-axis draw without labelling the grid.

    build() consumes its random stream in a fixed order: the Monte Carlo
    volume points, then the Laguerre seed points and their lognormal
    radii inside a weight-scale bisection that itself draws nothing, and
    only then the c-axes. None of that depends on the fabric parameters,
    so replaying the cheap part advances the stream to exactly the state
    the production build was in when it drew the orientations. n_real is
    the realised grain count, which the grid labelling sets and which is
    read off the cached tessellation of the same seed.
    """
    sp = DiskSpecimen(diameter_m=DIA, thickness_m=THK,
                      n_grains=N_GRAIN_TARGET, size_cv=SIZE_CV,
                      concentration=kappa, spatial_corr=0.0,
                      fabric_axis=tuple(axis), seed=seed)
    rng = np.random.default_rng(seed)
    R = DIA / 2
    mc = sp._mc_points(R, THK, 400000, rng)
    base = (R / N_GRAIN_TARGET ** (1 / 3)) ** 2
    sigma = np.sqrt(np.log(1 + SIZE_CV ** 2))
    n_seed = N_GRAIN_TARGET
    for outer in range(4):
        pts = sp._seed_points(R, THK, n_seed, rng)
        radii = rng.lognormal(-0.5 * sigma ** 2, sigma, n_seed)
        wshape = base * (radii ** 2 - 1.0)
        lo, hi, scale = 0.0, 8.0, 0.0
        for _ in range(20):
            trial = 0.5 * (lo + hi)
            cnt = np.bincount(sp._assign(mc, pts, trial * wshape),
                              minlength=n_seed).astype(float)
            live = cnt[cnt > 0]
            cv = (live ** (1 / 3)).std() / (live ** (1 / 3)).mean()
            if cv < SIZE_CV:
                lo = trial
            else:
                hi = trial
            scale = trial
        n_live = int((np.bincount(sp._assign(mc, pts, scale * wshape),
                                  minlength=n_seed) > 0).sum())
        if n_live >= N_GRAIN_TARGET or outer == 3:
            break
        n_seed = int(np.ceil(n_seed * N_GRAIN_TARGET / max(n_live, 1)))
    return sample_watson(n_real, axis, kappa, rng)


def check_shared_and_replay():
    """The two premises the single-maximum rows rest on, checked.

    Returns (n_shared_ok, n_shared, n_replay_ok, n_replay).
    """
    shared, shared_ok, rep, rep_ok = 0, 0, 0, 0
    for s in SEEDS:
        pg, ps = cache_path(s, GIRDLE_KAPPA), cache_path(s, SINGLE_KAPPA)
        for path, kappa, axis in ((pg, GIRDLE_KAPPA, GIRDLE_AXIS),
                                  (ps, SINGLE_KAPPA, SINGLE_AXIS)):
            if not os.path.exists(path):
                continue
            with np.load(path) as z:
                cached = np.asarray(z["axes"], float)
            rep += 1
            rep_ok += int(np.array_equal(
                replay_axes(s, len(cached), axis, kappa), cached))
        if os.path.exists(pg) and os.path.exists(ps):
            with np.load(pg) as a, np.load(ps) as b:
                shared += 1
                shared_ok += int(np.array_equal(a["labels"], b["labels"])
                                 and np.array_equal(a["seeds"], b["seeds"])
                                 and not np.array_equal(a["axes"], b["axes"]))
    return shared_ok, shared, rep_ok, rep


def specimen_registry():
    """Every row of the matrix, as a dict of the fields that define it.

    kind is what the row is for: 'girdle' and 'single' are the two
    fabric families, 'control' the seed 11 rows that change the acoustic
    contrast without touching the geometry.
    """
    rows = []
    for s in SEEDS:
        rows.append(dict(name="girdle_s%d" % s, seed=s, kind="girdle",
                         fabric="girdle", kappa=GIRDLE_KAPPA,
                         axis=GIRDLE_AXIS, contrast_f=1.0,
                         sweep="girdle_perp_ppw8" if s == 11
                         else "mx_girdle_s%d_ppw8" % s))
    for s in SEEDS:
        rows.append(dict(name="single_s%d" % s, seed=s, kind="single",
                         fabric="single_max", kappa=SINGLE_KAPPA,
                         axis=SINGLE_AXIS, contrast_f=1.0,
                         sweep="singlemax_ppw8" if s == 11
                         else "mx_single_s%d_ppw8" % s))
    rows.append(dict(name="zerocontrast_s11", seed=11, kind="control",
                     fabric="uniform", kappa=GIRDLE_KAPPA,
                     axis=GIRDLE_AXIS, contrast_f=0.0,
                     sweep="zerocontrast_ppw8", uniform=True))
    for f in (0.00, 0.25, 0.50, 0.75):
        rows.append(dict(name="cs_f%03d_s11" % round(f * 100), seed=11,
                         kind="control", fabric="girdle",
                         kappa=GIRDLE_KAPPA, axis=GIRDLE_AXIS,
                         contrast_f=f,
                         sweep="cs_f%03d_s11_ppw8" % round(f * 100)))
    return rows


def load_specimen(row, store):
    """(labels, axes, seeds, h) for one registry row.

    The label volume is keyed on the seed alone because every fabric and
    every contrast rung of a seed shares it, and only one is held at a
    time: a ppw 8 volume is 416 by 416 by 146 and keeping all eight
    would cost most of a gigabyte to save a few seconds of decompression.
    The uniform control is the girdle draw with one c-axis copied into
    every grain, which is what the solver's zero-contrast treatment does
    to the stiffness table.
    """
    seed = row["seed"]
    if seed not in store:
        store.clear()
        store[seed] = load_cached(seed, GIRDLE_KAPPA)
    lab, axes_g, pts, h = store[seed]
    if row["kind"] == "single":
        path = cache_path(seed, SINGLE_KAPPA)
        if os.path.exists(path):
            with np.load(path) as z:
                axes = np.asarray(z["axes"], float)
        else:
            axes = replay_axes(seed, len(axes_g), SINGLE_AXIS, SINGLE_KAPPA)
    elif row.get("uniform"):
        axes = np.repeat(axes_g[:1], len(axes_g), axis=0)
    else:
        axes = axes_g
    return lab, axes, pts, h


# --------------------------------------------------------------- compute

def fold_deg(v, period=180.0):
    return np.mod(v, period)


def dir_angles(v):
    """(in-plane angle in [0,180), out-of-plane tilt in [0,90]) of an axis."""
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    phi = fold_deg(np.degrees(np.arctan2(v[1], v[0])))
    tilt = np.degrees(np.arcsin(min(abs(v[2]), 1.0)))
    return float(phi), float(tilt)


def grain_volumes(labels, n_grain, h):
    """Voxel counts and volumes of every grain, in index order."""
    live = labels[labels >= 0]
    cells = np.bincount(live.ravel(), minlength=n_grain).astype(float)
    return cells, cells * h ** 3


def size_stats(vol):
    """Volume-equivalent diameters and the shape of their distribution."""
    live = vol[vol > 0]
    d = 2.0 * (3.0 * live / (4.0 * np.pi)) ** (1.0 / 3.0)
    z = (d - d.mean()) / d.std()
    return dict(n_grain=int(live.size),
                d_mean_mm=float(d.mean() * 1e3),
                d_median_mm=float(np.median(d) * 1e3),
                d_cv=float(d.std(ddof=1) / d.mean()),
                d_skew=float((z ** 3).mean()),
                d_max_mm=float(d.max() * 1e3),
                vol_cv=float(live.std(ddof=1) / live.mean()),
                vol_frac_largest=float(live.max() / live.sum()),
                vol_frac_top10=float(np.sort(live)[::-1]
                                     [:max(1, live.size // 10)].sum()
                                     / live.sum()))


def grain_shape(labels, n_grain, h):
    """Per-grain elongation from the voxel covariance, and its texture.

    The covariance is accumulated plane by plane so that no coordinate
    array of the whole volume is ever materialised. Elongation is the
    square root of the ratio of the largest to the smallest eigenvalue,
    which is the aspect ratio of the equivalent ellipsoid. The preferred
    direction is the principal axis of the volume-weighted orientation
    tensor of the long axes, and its own leading eigenvalue says how
    strongly the long axes agree.
    """
    nx, ny, nz = labels.shape
    xs = (np.arange(nx) - (nx - 1) / 2.0) * h
    ys = (np.arange(ny) - (ny - 1) / 2.0) * h
    zs = (np.arange(nz) - (nz - 1) / 2.0) * h
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    xf, yf = X.ravel(), Y.ravel()
    cnt = np.zeros(n_grain)
    s1 = np.zeros((n_grain, 3))
    s2 = np.zeros((n_grain, 6))
    idx6 = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    for k in range(nz):
        lab = labels[:, :, k].ravel()
        m = lab >= 0
        if not m.any():
            continue
        g = lab[m]
        p = np.stack([xf[m], yf[m], np.full(m.sum(), zs[k])], axis=1)
        cnt += np.bincount(g, minlength=n_grain)
        for a in range(3):
            s1[:, a] += np.bincount(g, weights=p[:, a], minlength=n_grain)
        for q, (a, b) in enumerate(idx6):
            s2[:, q] += np.bincount(g, weights=p[:, a] * p[:, b],
                                    minlength=n_grain)
    live = cnt > 0
    mu = np.zeros((n_grain, 3))
    mu[live] = s1[live] / cnt[live, None]
    cov = np.zeros((n_grain, 3, 3))
    for q, (a, b) in enumerate(idx6):
        c = np.zeros(n_grain)
        c[live] = s2[live, q] / cnt[live] - mu[live, a] * mu[live, b]
        cov[:, a, b] = c
        cov[:, b, a] = c
    ev, evec = np.linalg.eigh(cov[live])
    lam = np.clip(ev[:, ::-1], 1e-30, None)
    longax = evec[:, :, -1]
    elong = np.sqrt(lam[:, 0] / lam[:, 2])
    flat = np.sqrt(lam[:, 1] / lam[:, 2])
    w = cnt[live] / cnt[live].sum()
    T = np.einsum("g,gi,gj->ij", w, longax, longax)
    lw, vw = np.linalg.eigh(T)
    order = np.argsort(lw)[::-1]
    phi, tilt = dir_angles(vw[:, order[0]])
    return dict(elong_mean=float(w @ elong),
                elong_median=float(np.median(elong)),
                elong_p90=float(np.percentile(elong, 90)),
                flatten_mean=float(w @ flat),
                elong_axis_a1=float(lw[order[0]]),
                elong_axis_phi_deg=phi,
                elong_axis_tilt_deg=tilt)


def adjacency_faces(labels):
    """Face-adjacent grain pairs and their shared voxel-face counts.

    Every 6-connected face between two different non-negative labels is
    counted once. The face count times h^2 is the rasterised area of
    that facet, which is the natural weight for anything the wave sees
    at a boundary. Face counting on a cubic grid overestimates a smooth
    area by a fixed factor, which cancels in every comparison made here.
    """
    tot = {}
    for ax in range(3):
        lo = np.take(labels, np.arange(labels.shape[ax] - 1), axis=ax)
        hi = np.take(labels, np.arange(1, labels.shape[ax]), axis=ax)
        m = (lo != hi) & (lo >= 0) & (hi >= 0)
        a = lo[m].astype(np.int64)
        b = hi[m].astype(np.int64)
        key = np.minimum(a, b) * 1000000 + np.maximum(a, b)
        u, c = np.unique(key, return_counts=True)
        for k, n in zip(u, c):
            tot[int(k)] = tot.get(int(k), 0) + int(n)
    keys = np.fromiter(tot.keys(), np.int64, len(tot))
    return (keys // 1000000, keys % 1000000,
            np.fromiter(tot.values(), float, len(tot)))


def facet_stats(i, j, faces, pts, h, n_grain, vol_total):
    """Boundary area, facet size distribution, and normal directions.

    For a Laguerre tessellation the plane separating two neighbouring
    cells is perpendicular to the difference of their seed points, in
    closed form and independent of the weights, which only slide the
    plane along that normal. The normals are therefore exact even though
    the areas are rasterised.
    """
    area = faces * h ** 2
    nrm = pts[i] - pts[j]
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    w = area / area.sum()
    T = np.einsum("f,fi,fj->ij", w, nrm, nrm)
    ev = np.sort(np.linalg.eigvalsh(T))[::-1]
    phi = np.arctan2(nrm[:, 1], nrm[:, 0])
    h2 = complex(float(w @ np.cos(2 * phi)), float(w @ np.sin(2 * phi)))
    deg = np.bincount(np.concatenate([i, j]), minlength=n_grain).astype(float)
    srt = np.sort(area)[::-1]
    ntop = max(1, len(area) // 10)
    return dict(n_facet=int(len(area)),
                area_total_cm2=float(area.sum() * 1e4),
                sv_per_mm=float(area.sum() / vol_total / 1e3),
                facet_area_mean_mm2=float(area.mean() * 1e6),
                facet_area_median_mm2=float(np.median(area) * 1e6),
                facet_area_cv=float(area.std(ddof=1) / area.mean()),
                facet_area_top10_frac=float(srt[:ntop].sum() / area.sum()),
                faces_per_grain=float(deg.mean()),
                faces_per_grain_sd=float(deg.std(ddof=1)),
                normal_a1=float(ev[0]), normal_a2=float(ev[1]),
                normal_a3=float(ev[2]),
                normal_absz=float(w @ np.abs(nrm[:, 2])),
                normal_h2_amp=float(abs(h2)),
                normal_h2_phase_deg=float(fold_deg(
                    np.degrees(np.angle(h2)) / 2.0)),
                normals=nrm, area=area)


def nn_spacing(pts):
    """Nearest-neighbour distance between Laguerre seed points."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return dict(nn_seed_mm=float(nn.mean() * 1e3),
                nn_seed_cv=float(nn.std(ddof=1) / nn.mean()))


def orientation_tensor_eigs(axes, weights=None):
    """Eigenvalues and eigenvectors of the second-order orientation tensor."""
    a = np.asarray(axes, float)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    if weights is None:
        w = np.full(len(a), 1.0 / len(a))
    else:
        w = np.asarray(weights, float)
        w = w / w.sum()
    T = np.einsum("g,gi,gj->ij", w, a, a)
    ev, evec = np.linalg.eigh(T)
    o = np.argsort(ev)[::-1]
    return ev[o], evec[:, o]


def contrast_tensors(axes, contrast_f):
    """Fourth-order stiffness of every grain under a contrast rung.

    The solver's contrast treatment interpolates each grain's ROTATED
    6x6 Voigt tensor a fraction f from the mean of all of them toward
    its own, so f = 0 makes every grain identical and f = 1 is the
    untouched polycrystal. The blend is not transversely isotropic, so
    its speeds have to come from the full Christoffel eigenproblem and
    not from the qP-against-psi table. Returned as the 3x3x3x3 form
    because that is what the Christoffel contraction wants, and built
    once per specimen rather than once per azimuth.
    """
    base = AN.ti_stiffness_6(**AN.ICE_MATERIAL)
    c6 = np.array([AN.rotate_voigt_3d(base, AN._rotation_z_to(a))
                   for a in axes])
    mean = c6.mean(axis=0, keepdims=True)
    c6 = mean + contrast_f * (c6 - mean)
    return np.array([AN.voigt6_to_tensor(c) for c in c6])


def qp_speeds(axes, dirs, tensors=None):
    """qP phase speed of every grain along every direction, (n_g, n_dir).

    At full contrast this is the transversely isotropic closed form the
    rest of the project uses, interpolated on the same 901-point table
    that forward.py builds. When a contrast rung has replaced the
    tensors, the Christoffel eigenproblem is solved instead.
    """
    d = np.asarray(dirs, float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    if tensors is None:
        cosang = np.abs(np.asarray(axes, float) @ d.T)
        return np.interp(np.arccos(np.clip(cosang, 0.0, 1.0)),
                         FW._PSI, FW._VQP)
    gam = np.einsum("gijkl,rj,rl->grik", tensors, d, d)
    ev = np.linalg.eigvalsh(gam)[..., -1]
    return np.sqrt(np.clip(ev, 0.0, None) / AN.ICE_RHO)


def skew_table():
    """Group-velocity skew angle of qP against angle from the c-axis.

    INFERRED, in the sense that it is a numerical derivative of the
    tabulated phase speed rather than a closed form: the skew angle
    between the phase and group directions of a transversely isotropic
    medium is arctan((1/v) dv/dpsi). It is tabulated once here and read
    by interpolation, and it is the quantity that predicts the beam
    walk-off azimuths reported in azimuth_sweep.py.
    """
    psi, v = FW._PSI, FW._VQP
    dv = np.gradient(v, psi)
    return psi, np.degrees(np.arctan2(dv, v))


_SKEW_PSI, _SKEW_DEG = skew_table()


def misorientation_stats(axes, i, j, area):
    """Area-weighted misorientation across grain-pair contacts, degrees.

    c-axes are axial, so the angle is folded onto [0, 90].
    """
    cos = np.abs(np.sum(axes[i] * axes[j], axis=1))
    mis = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
    w = area / area.sum()
    order = np.argsort(mis)
    cw = np.cumsum(w[order])
    return dict(misor_mean_deg=float(w @ mis),
                misor_median_deg=float(mis[order][np.searchsorted(cw, 0.5)]),
                misor_sd_deg=float(np.sqrt(w @ (mis - w @ mis) ** 2)),
                misor_p90_deg=float(mis[order][np.searchsorted(cw, 0.9)]),
                misor_frac_lt10=float(w[mis < 10.0].sum()),
                misor=mis)


def clustering(axes, i, j, area, vol, evec1, rng):
    """Is the orientation field spatially clustered, or is it noise?

    Two views, both with the same permutation null: the c-axes are
    shuffled among the grains while the tessellation is held fixed, so
    the null keeps every geometric property and destroys only which
    grain got which orientation.

      ratio    area-weighted adjacent-pair misorientation divided by the
               volume-weighted all-pair misorientation. Below one means
               neighbours are more alike than the disc average.
      moran    Moran's I of cos^2 of the angle to the specimen's own
               principal orientation direction, on the face-adjacency
               graph, weighted by facet area.
    """
    w = area / area.sum()
    cosad = np.abs(np.sum(axes[i] * axes[j], axis=1))
    mis_adj = float(w @ np.degrees(np.arccos(np.clip(cosad, 0.0, 1.0))))
    vw = vol / vol.sum()
    pair = np.outer(vw, vw)
    np.fill_diagonal(pair, 0.0)
    pair /= pair.sum()
    mis_all_m = np.degrees(np.arccos(np.clip(np.abs(axes @ axes.T), 0.0, 1.0)))
    mis_all = float((pair * mis_all_m).sum())

    x = (axes @ evec1) ** 2
    xb = float(vw @ x)
    dev = x - xb
    denom = float(vw @ dev ** 2)
    moran = float((w @ (dev[i] * dev[j])) / denom) if denom > 1e-24 else 0.0

    n = len(axes)
    null_ratio = np.empty(N_PERM)
    null_moran = np.empty(N_PERM)
    for q in range(N_PERM):
        p = rng.permutation(n)
        ap = axes[p]
        c = np.abs(np.sum(ap[i] * ap[j], axis=1))
        null_ratio[q] = w @ np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))
        dp = (ap @ evec1) ** 2 - xb
        null_moran[q] = ((w @ (dp[i] * dp[j])) / denom
                         if denom > 1e-24 else 0.0)

    def zscore(v, null):
        sd = null.std(ddof=1)
        return float((v - null.mean()) / sd) if sd > 1e-24 else 0.0

    return dict(misor_adj_deg=mis_adj, misor_all_deg=mis_all,
                clust_ratio=float(mis_adj / mis_all)
                if mis_all > 1e-12 else 1.0,
                clust_z=zscore(mis_adj, null_ratio),
                moran_i=moran, moran_z=zscore(moran, null_moran))


def misor_vs_distance(axes, pts):
    """Does misorientation grow with separation? Spearman over all pairs.

    A second, graph-free view of orientation clustering: a positive rank
    correlation between the distance separating two grain centres and
    the angle between their c-axes means orientation is organised over a
    length scale rather than grain by grain.
    """
    from scipy import stats
    iu = np.triu_indices(len(axes), 1)
    d = np.linalg.norm(pts[iu[0]] - pts[iu[1]], axis=1)
    mis = np.degrees(np.arccos(np.clip(
        np.abs(np.sum(axes[iu[0]] * axes[iu[1]], axis=1)), 0.0, 1.0)))
    if mis.std() < 1e-12:
        return dict(misor_dist_rho=0.0, misor_dist_p=1.0)
    rho, p = stats.spearmanr(d, mis)
    return dict(misor_dist_rho=float(rho), misor_dist_p=float(p))


def march(labels, h, p0, dirs, s):
    """Nearest-cell labels along a bundle of rays, (n_ray, n_s).

    Delegates to the marcher of _t2_common so that the cell-centre
    rounding convention is the one every other facet analysis uses.
    """
    return C2._march(labels, h, p0, np.atleast_2d(dirs), s)


def crossings(L):
    """(count of label changes per ray, the pair indices at each change)."""
    a, b = L[:, :-1], L[:, 1:]
    cr = (a != b) & (a >= 0) & (b >= 0)
    ri, si = np.nonzero(cr)
    return cr.sum(axis=1), ri, si, a[ri, si], b[ri, si]


def per_azimuth(labels, axes, pts, h, tensors, area_all,
                pair_i, pair_j, normals, vol):
    """Every sample property that has a beam direction in it.

    Two beam geometries are kept apart throughout, because the paper
    calls both of them crossings and they are not the same number.

      AXIS    the single ray down the beam axis, from the entry point on
              the rim to the far wall. This is what a travel-time or a
              backwall observable samples.
      COLUMN  the diverging cone of half-angle 8.9 degrees, marched with
              200 quasi-random rays. This is what a coda observable
              samples, and it crosses far more boundaries because it is
              a volume and not a line.

    Each is reported over the whole traverse and again restricted to the
    range the coda gate covers at the reference speed, 46.2 to 69.3 mm.
    The paper's own crossing count, the 37-ray lattice of
    facet_predictors summed over rays inside the gate, is reproduced
    beside them so the published correlation can be traced.
    """
    ds = h / 2.0
    s_full = (np.arange(int(DIA / ds)) + 0.5) * ds
    gate = (s_full >= GATE_S[0]) & (s_full < GATE_S[1])
    rr, ph, wb = C2.cone_rays(N_RAY_COL, HALF_DEG, 0)
    uu = np.linspace(-1, 1, NLAT_PAPER)
    U, W = np.meshgrid(uu, uu, indexing="ij")
    keep = U ** 2 + W ** 2 <= 1.0
    U, W = U[keep], W[keep]
    s_pap = np.arange(GATE_S[0], GATE_S[1], h)
    vw = vol / vol.sum()
    aw = area_all / area_all.sum()

    out = {}
    for az in AZ:
        n, t1, t2 = C2.beam_frame(float(az))
        p0 = (DIA / 2.0) * n
        d_ax = (-n)[None, :]
        d_col = (-n[None, :] + (rr * np.cos(ph))[:, None] * t1
                 + (rr * np.sin(ph))[:, None] * t2)
        d_col /= np.linalg.norm(d_col, axis=1, keepdims=True)

        lab_ax = march(labels, h, p0, d_ax, s_full)
        n_ax_full, _, _, _, _ = crossings(lab_ax)
        n_ax_gate, _, _, _, _ = crossings(lab_ax[:, gate])
        inside = lab_ax[0] >= 0
        grains_ax = np.unique(lab_ax[0][inside])

        lab_col = march(labels, h, p0, d_col, s_full)
        n_col_full, _, _, _, _ = crossings(lab_col)
        Lg = lab_col[:, gate]
        n_col_gate, cri, _, ca, cb = crossings(Lg)
        gr_col = np.unique(Lg[Lg >= 0])

        v_ax = qp_speeds(axes, d_ax, tensors)[:, 0]
        v_col = qp_speeds(axes, d_col, tensors)
        vbar = float(vw @ v_ax)
        # A travel time averages SLOWNESS, not speed, so the closed-form
        # bulk prediction of an arrival is the volume-weighted harmonic
        # mean. c_vol is the arithmetic one and is kept beside it because
        # the two differ by a term in the variance of the speed, which is
        # itself a fabric descriptor.
        c_slow = 1.0 / float(vw @ (1.0 / v_ax))
        t_one = float(np.sum(ds / v_ax[lab_ax[0][inside]]))
        path = float(inside.sum()) * ds
        c_ax = path / t_one

        dv_all = np.abs(v_ax[pair_i] - v_ax[pair_j]) / (2.0 * vbar)
        dv_cr = np.abs(v_col[ca, cri] - v_col[cb, cri]) / (2.0 * vbar)
        mis_cr = np.degrees(np.arccos(np.clip(
            np.abs(np.sum(axes[ca] * axes[cb], axis=1)), 0.0, 1.0)))
        # Laguerre facet normal of a crossed pair, in closed form from
        # the seed difference, so no lookup into the facet list is
        # needed and the crossed set may repeat pairs freely.
        nm_cr = pts[cb] - pts[ca]
        nm_cr /= np.linalg.norm(nm_cr, axis=1, keepdims=True) + 1e-30
        align = np.abs(np.einsum("fi,fi->f", nm_cr, d_col[cri]))

        face = np.abs(normals @ n)
        cos2 = (axes @ n) ** 2
        psi = np.arccos(np.clip(np.abs(axes @ n), 0.0, 1.0))
        skew = np.interp(psi, _SKEW_PSI, _SKEW_DEG)

        lat_rad = (s_pap * np.tan(FP.HALF))[:, None]
        P = (p0 - s_pap[:, None] * n)[:, None, :] \
            + (lat_rad * U[None, :])[:, :, None] * t1 \
            + (lat_rad * W[None, :])[:, :, None] * t2
        nx, ny, nz = labels.shape
        gi = np.rint((P[..., 0] + nx * h / 2) / h - 0.5).astype(int)
        gj = np.rint((P[..., 1] + ny * h / 2) / h - 0.5).astype(int)
        gk = np.rint((P[..., 2] + nz * h / 2) / h - 0.5).astype(int)
        ok = ((gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
              & (gk >= 0) & (gk < nz))
        Lp = np.where(ok, labels[np.clip(gi, 0, nx - 1),
                                np.clip(gj, 0, ny - 1),
                                np.clip(gk, 0, nz - 1)], -1)
        Ap, Bp = Lp[:-1], Lp[1:]
        n_paper = float(((Ap != Bp) & (Ap >= 0) & (Bp >= 0)).sum())

        gw = vol[gr_col] / vol[gr_col].sum()
        ev_col, _ = orientation_tensor_eigs(axes[gr_col], gw)
        row = dict(
            az_deg=float(az),
            n_cross_axis_full=float(n_ax_full[0]),
            n_cross_axis_gate=float(n_ax_gate[0]),
            n_cross_col_full=float(n_col_full.mean()),
            n_cross_col_gate=float(n_col_gate.mean()),
            n_cross_col_gate_wt=float(np.average(n_col_gate, weights=wb)),
            n_cross_paper=n_paper,
            n_grain_axis=float(grains_ax.size),
            n_grain_col_gate=float(gr_col.size),
            chord_axis_mm=float(path / max(grains_ax.size, 1) * 1e3),
            path_axis_mm=float(path * 1e3),
            t_axis_us=float(t_one * 1e6),
            c_axis_ms=float(c_ax),
            c_vol_ms=vbar,
            c_slow_ms=c_slow,
            v_sd_ms=float(np.sqrt(vw @ (v_ax - vbar) ** 2)),
            cos2_beam=float(vw @ cos2),
            cos2_beam_col=float(gw @ (axes[gr_col] @ n) ** 2),
            a1_col=float(ev_col[0]),
            a3_col=float(ev_col[2]),
            skew_deg=float(vw @ np.abs(skew)),
            skew_p90_deg=float(np.percentile(np.abs(skew), 90)),
            dv_all_rms=float(np.sqrt(aw @ dv_all ** 2)),
            dv_all_mean=float(aw @ dv_all),
            dv_all_p90=float(np.percentile(dv_all, 90)),
            dv_cross_rms=float(np.sqrt((dv_cr ** 2).mean()))
            if dv_cr.size else 0.0,
            dv_cross_mean=float(dv_cr.mean()) if dv_cr.size else 0.0,
            misor_cross_deg=float(mis_cr.mean()) if mis_cr.size else 0.0,
            normal_align=float(align.mean()) if align.size else 0.0,
            facing_frac_10=float(aw[face > np.cos(np.radians(10.0))].sum()),
            facing_frac_20=float(aw[face > np.cos(np.radians(20.0))].sum()),
        )
        for thr in DARK_R:
            row["dark_all_%g" % thr] = float(aw[dv_all < thr].sum())
            row["dark_cross_%g" % thr] = (float((dv_cr < thr).mean())
                                          if dv_cr.size else 1.0)
        # Threshold-free companion to the dark fraction: the smallest
        # share of the boundary AREA that carries half the reflected
        # power. One means every boundary reflects alike, small means a
        # few facets carry the return.
        p = aw * dv_all ** 2
        if p.sum() > 0:
            o = np.argsort(p)[::-1]
            half = np.searchsorted(np.cumsum(p[o]) / p.sum(), 0.5) + 1
            row["bright_half_area"] = float(aw[o][:half].sum())
        else:
            row["bright_half_area"] = 1.0
        for k, v in row.items():
            out.setdefault(k, []).append(v)
    return {k: np.array(v, float) for k, v in out.items()}


def harmonics(az_deg, y, kmax=4):
    """Harmonic content of an azimuthal profile, amplitudes and phases."""
    t = np.radians(np.asarray(az_deg, float))
    cols = [np.ones_like(t)]
    for k in range(1, kmax + 1):
        cols += [np.cos(k * t), np.sin(k * t)]
    A = np.column_stack(cols)
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    out = {"h0": float(c[0])}
    var = float(np.var(y))
    for k in range(1, kmax + 1):
        a, b = c[2 * k - 1], c[2 * k]
        amp = float(np.hypot(a, b))
        out["h%d_amp" % k] = amp
        out["h%d_phase_deg" % k] = float(fold_deg(
            np.degrees(np.arctan2(b, a)) / k, 360.0 / k))
        out["h%d_frac" % k] = float(0.5 * amp ** 2 / var) if var > 0 else 0.0
    return out


def specimen_row(reg, store):
    """One complete row of the left side, plus its azimuth cube.

    The permutation null is seeded from the specimen so the row does not
    depend on where in the registry it was computed.
    """
    rng = np.random.default_rng(1000 + reg["seed"])
    labels, axes, pts, h = load_specimen(reg, store)
    n_g = len(axes)
    cells, vol = grain_volumes(labels, n_g, h)
    vol_total = float(vol.sum())
    row = dict(name=reg["name"], seed=reg["seed"], kind=reg["kind"],
               fabric=reg["fabric"], sweep=reg["sweep"],
               kappa=reg["kappa"], contrast_f=reg["contrast_f"],
               is_single_max=float(reg["fabric"] == "single_max"),
               is_girdle=float(reg["fabric"] == "girdle"),
               ppw=PPW, h_mm=h * 1e3,
               fabric_axis_phi_deg=dir_angles(reg["axis"])[0],
               fabric_axis_tilt_deg=dir_angles(reg["axis"])[1],
               vol_disc_cm3=vol_total * 1e6)
    row.update(size_stats(vol))
    row.update(grain_shape(labels, n_g, h))
    row.update(nn_spacing(pts))

    i, j, faces = adjacency_faces(labels)
    order = np.argsort(i * 1000000 + j)
    i, j, faces = i[order], j[order], faces[order]
    fs = facet_stats(i, j, faces, pts, h, n_g, vol_total)
    normals, area = fs.pop("normals"), fs.pop("area")
    row.update(fs)

    ev_n, vec_n = orientation_tensor_eigs(axes)
    ev_v, vec_v = orientation_tensor_eigs(axes, vol)
    for q in range(3):
        row["a%d_count" % (q + 1)] = float(ev_n[q])
        row["a%d_vol" % (q + 1)] = float(ev_v[q])
    row["a1_vol_minus_count"] = float(ev_v[0] - ev_n[0])
    p1, t1a = dir_angles(vec_v[:, 0])
    p3, t3a = dir_angles(vec_v[:, 2])
    row.update(e1_phi_deg=p1, e1_tilt_deg=t1a, e3_phi_deg=p3,
               e3_tilt_deg=t3a)
    lead = 2 if reg["kappa"] < 0 else 0
    pf, tf = dir_angles(vec_v[:, lead])
    row.update(fabaxis_meas_phi_deg=pf, fabaxis_meas_tilt_deg=tf)

    ms = misorientation_stats(axes, i, j, area)
    ms.pop("misor")
    row.update(ms)
    row.update(clustering(axes, i, j, area, vol, vec_v[:, 0], rng))
    row.update(misor_vs_distance(axes, pts))

    tensors = (contrast_tensors(axes, reg["contrast_f"])
               if reg["contrast_f"] < 1.0 else None)
    cube = per_azimuth(labels, axes, pts, h, tensors, area, i, j,
                       normals, vol)

    for key in ("n_cross_axis_full", "n_cross_axis_gate",
                "n_cross_col_full", "n_cross_col_gate", "n_cross_paper",
                "n_grain_axis", "n_grain_col_gate", "chord_axis_mm",
                "c_axis_ms", "c_vol_ms", "c_slow_ms", "v_sd_ms",
                "dv_all_rms", "dv_cross_rms", "bright_half_area",
                "misor_cross_deg", "normal_align", "facing_frac_10",
                "skew_deg", "cos2_beam", "a1_col",
                "dark_all_0.001", "dark_all_0.01", "dark_cross_0.001"):
        row[key + "_m"] = float(cube[key].mean())
    # Every _m column above is the arithmetic mean over azimuth. The
    # contrast is also summarised quadratically, because that is the
    # convention channel_network/e23_links.py uses for its dv_ray and a
    # column that reconciles with an existing repo number is worth more
    # than a column that nearly does. On the eight girdle tessellations
    # the two summaries correlate at 0.98 and differ by about 6 per cent.
    row["dv_ray_quad"] = float(np.sqrt((cube["dv_all_rms"] ** 2).mean()))
    # The azimuthal speed profile twice over. c_axis is the realisation
    # one ray gives, so it carries the sampling of ten grains as well as
    # the fabric; c_slow is the closed-form bulk prediction with no ray
    # in it. Reporting both separates the fabric signal from the
    # realisation noise that sits on top of it.
    for tag, series in (("c", cube["c_axis_ms"]),
                        ("cslow", cube["c_slow_ms"]),
                        ("cos2", cube["cos2_beam"])):
        s = np.asarray(series, float)
        row[tag + "_min"] = float(s.min())
        row[tag + "_max"] = float(s.max())
        row[tag + "_az_sd"] = float(s.std(ddof=1))
        row[tag + "_aniso_pct"] = float(100.0 * (s.max() - s.min())
                                        / abs(s.mean()))
        for k, v in harmonics(cube["az_deg"], s).items():
            row[tag + "_" + k] = v
    return row, cube


# ------------------------------------------------------------------ draw

def report_premises(shared_ok, shared, rep_ok, rep):
    print("=" * 79)
    print("PREMISES OF THE SINGLE-MAXIMUM ROWS")
    print("=" * 79)
    print("  girdle and single maximum share a label volume at the same")
    print("  seed: %d of %d cached pairs" % (shared_ok, shared))
    print("  the c-axis draw replays bit-exactly from the seed alone:")
    print("  %d of %d cached draws" % (rep_ok, rep))
    print("  seeds 53, 71 and 89 have no cached single maximum, so their")
    print("  orientations are produced by that replay. Their geometry is")
    print("  the girdle geometry of the same seed, which is not an")
    print("  approximation but the same tessellation.")


def report_matrix(rows, cols):
    print("\n" + "=" * 79)
    print("LEFT SIDE OF THE MATRIX, %d specimens by %d properties"
          % (len(rows), len(cols)))
    print("=" * 79)
    groups = [
        ("GRAIN SIZE", ["n_grain", "d_mean_mm", "d_median_mm", "d_cv",
                        "d_skew", "vol_frac_largest", "vol_frac_top10"]),
        ("GRAIN SHAPE", ["elong_mean", "elong_median", "flatten_mean",
                         "elong_axis_a1", "elong_axis_phi_deg",
                         "elong_axis_tilt_deg", "nn_seed_mm", "nn_seed_cv"]),
        ("BOUNDARIES", ["n_facet", "area_total_cm2", "sv_per_mm",
                        "facet_area_mean_mm2", "facet_area_cv",
                        "facet_area_top10_frac", "faces_per_grain"]),
        ("FACET NORMALS", ["normal_a1", "normal_a3", "normal_absz",
                           "normal_h2_amp", "normal_h2_phase_deg"]),
        ("CROSSINGS (azimuth mean)",
         ["n_cross_axis_full_m", "n_cross_axis_gate_m", "n_cross_col_full_m",
          "n_cross_col_gate_m", "n_cross_paper_m", "chord_axis_mm_m",
          "n_grain_axis_m", "n_grain_col_gate_m"]),
        ("FABRIC", ["a1_count", "a3_count", "a1_vol", "a3_vol",
                    "a1_vol_minus_count", "e1_phi_deg", "e1_tilt_deg",
                    "e3_phi_deg", "e3_tilt_deg"]),
        ("CONTACTS", ["misor_mean_deg", "misor_median_deg", "misor_sd_deg",
                      "misor_p90_deg", "misor_frac_lt10", "clust_ratio",
                      "clust_z", "moran_i", "moran_z", "misor_dist_rho"]),
        ("CONTRAST", ["dv_all_rms_m", "dv_ray_quad", "dv_cross_rms_m",
                      "dark_all_0.001_m",
                      "dark_all_0.01_m", "dark_cross_0.001_m",
                      "bright_half_area_m", "normal_align_m",
                      "facing_frac_10_m"]),
        ("BULK SPEED, one ray", ["c_axis_ms_m", "c_min", "c_max",
                                 "c_aniso_pct", "c_az_sd", "c_h2_amp",
                                 "c_h2_phase_deg", "c_h2_frac", "c_h4_amp",
                                 "c_h4_frac", "c_h1_amp", "c_h3_amp"]),
        ("BULK SPEED, closed form", ["c_vol_ms_m", "c_slow_ms_m", "v_sd_ms_m",
                                     "cslow_aniso_pct", "cslow_az_sd",
                                     "cslow_h2_amp", "cslow_h2_phase_deg",
                                     "cslow_h2_frac", "cslow_h4_amp",
                                     "cslow_h4_frac", "skew_deg_m"]),
        ("FABRIC PROJECTION", ["cos2_beam_m", "cos2_h2_amp",
                               "cos2_h2_phase_deg", "cos2_h2_frac",
                               "cos2_aniso_pct", "a1_col_m"]),
    ]
    for title, keys in groups:
        keys = [k for k in keys if k in cols]
        print("\n%s" % title)
        for chunk in [keys[q:q + 4] for q in range(0, len(keys), 4)]:
            print("  %-18s" % "specimen"
                  + "".join("%19s" % k[:19] for k in chunk))
            for r in rows:
                print("  %-18s" % r["name"][:18]
                      + "".join("%19.6g" % r[k] for k in chunk))


def report_variation(rows, cols):
    """Which properties can be tested BETWEEN specimens, and which cannot.

    The eight girdle tessellations are the only family with eight
    independent draws of both geometry and orientation, so they set what
    a between-specimen test can see. A column whose spread across them is
    a small fraction of its own value carries no between-specimen signal
    whatever its physical interest.
    """
    g = [r for r in rows if r["kind"] == "girdle"]
    s = [r for r in rows if r["kind"] == "single"]
    print("\n" + "=" * 79)
    print("BETWEEN-SPECIMEN SPREAD, the eight girdle tessellations")
    print("=" * 79)
    print("  %-26s %11s %10s %8s %11s" % ("property", "mean", "sd", "cv %",
                                          "single mean"))
    flat = []
    for k in cols:
        v = np.array([r[k] for r in g], float)
        if not np.all(np.isfinite(v)):
            continue
        m, sd = float(v.mean()), float(v.std(ddof=1))
        cv = 100.0 * sd / abs(m) if abs(m) > 1e-12 else np.inf
        sv = np.array([r[k] for r in s], float)
        print("  %-26s %11.5g %10.4g %8.2f %11.5g"
              % (k[:26], m, sd, cv, float(sv.mean())))
        if cv < 2.0:
            flat.append((k, cv))
    print("\n  EFFECTIVELY CONSTANT across the eight girdle tessellations")
    print("  (spread below 2 per cent of the mean). These cannot be tested")
    print("  between specimens however interesting they are; they are only")
    print("  usable as within-specimen predictors or as design constants.")
    for k, cv in sorted(flat, key=lambda t: t[1]):
        print("    %-30s %6.3f %%" % (k, cv))
    return [k for k, _ in flat]


def report_regimes(kinds, az, cube, keys):
    """Within against between, and the 180-degree periodicity, per column.

    The ratio that matters for design is the within-specimen spread over
    azimuth against the between-specimen spread of the specimen means. A
    column with a large ratio is a within-specimen predictor and its
    circular-shift null has real power; a column with a small ratio is
    almost a specimen constant and only n = 8 is available for it.
    """
    idx = [q for q, k in enumerate(kinds) if k == "girdle"]
    half = len(az) // 2
    print("\n" + "=" * 79)
    print("PER-AZIMUTH PROPERTIES: where the statistical power is")
    print("=" * 79)
    print("  %-24s %10s %10s %8s %9s %6s"
          % ("property", "within sd", "between sd", "ratio", "r(az+180)",
             "shifts"))
    within_v, between_v, per_v, shift_v = [], [], [], []
    for k in keys:
        M = cube[k][idx]
        within = float(np.mean(M.std(axis=1, ddof=1)))
        between = float(M.mean(axis=1).std(ddof=1))
        rr = [float(np.corrcoef(r, np.roll(r, half))[0, 1])
              for r in M if r.std() > 1e-12]
        per = float(np.mean(rr)) if rr else np.nan
        ratio = within / between if between > 1e-12 else np.inf
        shift = 15 if per > 0.9 else 30
        print("  %-24s %10.4g %10.4g %8.2f %9.3f %6d"
              % (k[:24], within, between, ratio, per, shift))
        within_v.append(within)
        between_v.append(between)
        per_v.append(per)
        shift_v.append(shift)
    print("\n  r(az+180) near +1 means the column is 180 degree periodic by")
    print("  construction, so a 30-azimuth sweep holds 15 distinct")
    print("  alignments and the circular-shift null must be run on 15.")
    print("  Near 0 means the column is not, and all 30 are distinct.")
    print("  The split is not a property of the physics but of the")
    print("  geometry of each column. Everything a c-axis sets is exactly")
    print("  invariant under a beam reversal because qP depends on the")
    print("  magnitude of the direction cosine. Everything the marcher")
    print("  counts inside the coda gate is not, because at az + 180 the")
    print("  gate covers the opposite half of the same chord. The AXIS")
    print("  count over the whole traverse is periodic and the same count")
    print("  restricted to the gate is not, which is the sharpest")
    print("  statement in this table of why the two must be kept apart.")
    return (np.array(within_v), np.array(between_v), np.array(per_v),
            np.array(shift_v))


def report_weighting(rows):
    """Grain-count against volume weighting of the orientation tensor.

    The wave samples volume, not grains, so the volume-weighted tensor
    is the one a scattering or travel-time argument is entitled to use.
    Whether that matters is an empirical question about this ensemble
    and not a general one, so it is measured: if the two weightings
    ranked the specimens the same way, the distinction would be
    bookkeeping.
    """
    from scipy import stats
    print("\n" + "=" * 79)
    print("COUNT AGAINST VOLUME WEIGHTING OF THE ORIENTATION TENSOR")
    print("=" * 79)
    for kind in ("girdle", "single"):
        v = [r for r in rows if r["kind"] == kind]
        if len(v) < 3:
            continue
        a = np.array([r["a1_count"] for r in v])
        b = np.array([r["a1_vol"] for r in v])
        rho, p = stats.spearmanr(a, b)
        r, pp = stats.pearsonr(a, b)
        d = b - a
        print("  %-8s n = %d" % (kind, len(v)))
        print("    a1 count  %.4f +- %.4f     a1 volume %.4f +- %.4f"
              % (a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1)))
        print("    volume minus count %+.4f +- %.4f, positive in %d of %d"
              % (d.mean(), d.std(ddof=1), int((d > 0).sum()), len(d)))
        print("    the two weightings rank the specimens: spearman "
              "%+.3f (p %.3f), pearson %+.3f (p %.3f)"
              % (rho, p, r, pp))
    print("  Volume weighting raises a1 on every girdle tessellation, so")
    print("  it is not a wash: the two are different measurements of the")
    print("  same specimens and the screening stage must carry both.")


def report_reconciliation(rows):
    """Tie the new crossing columns to the published one."""
    from scipy import stats
    path = os.path.join(RESULTS, "ensemble.npz")
    if not os.path.exists(path):
        print("\n  ensemble.npz absent, crossing reconciliation skipped")
        return
    with np.load(path) as z:
        seeds = list(z["girdle_seeds"])
        pub = z["crossings"]
    idx = {r["seed"]: r for r in rows if r["kind"] == "girdle"}
    mine = np.array([idx[int(s)]["n_cross_paper_m"] for s in seeds])
    col = np.array([idx[int(s)]["n_cross_col_gate_m"] for s in seeds])
    axi = np.array([idx[int(s)]["n_cross_axis_gate_m"] for s in seeds])
    print("\n" + "=" * 79)
    print("RECONCILIATION with the published crossing count")
    print("=" * 79)
    print("  ensemble.npz stores the 37-ray lattice sum inside the gate,")
    print("  built at ppw 4. The same quantity at ppw 8 is recomputed here.")
    print("  %-34s %8s %8s" % ("pair", "pearson", "p"))
    for tag, v in (("published ppw4 vs paper ppw8", mine),
                   ("published ppw4 vs column/ray", col),
                   ("published ppw4 vs AXIS gate", axi)):
        r, p = stats.pearsonr(pub, v)
        print("  %-34s %+8.3f %8.4f" % (tag, r, p))
    r, p = stats.pearsonr(axi, col)
    print("  %-34s %+8.3f %8.4f" % ("AXIS gate vs COLUMN gate", r, p))
    print("  The axis count and the column count are different properties")
    print("  and the paper calls both of them crossings. The column is a")
    print("  volume average over a diverging cone; the axis is one line.")


def write_outputs(rows, cols, az, cube, cube_keys, names, kinds, flat,
                  regimes):
    """Both matrices, in npz for the screening stage and csv for a human.

    The screening stage needs three things beyond the numbers: which
    columns are effectively constant between specimens, how many
    distinct circular-shift alignments each per-azimuth column has, and
    the within-against-between ratio that says which regime a column
    belongs in. All three ride along in the archives.
    """
    os.makedirs(RESULTS, exist_ok=True)
    within, between, per, shifts = regimes
    M = np.array([[float(r[k]) for k in cols] for r in rows])
    np.savez(os.path.join(RESULTS, "sample_matrix.npz"),
             names=np.array(names), kinds=np.array(kinds),
             seeds=np.array([r["seed"] for r in rows]),
             sweeps=np.array([r["sweep"] for r in rows]),
             columns=np.array(cols), values=M,
             near_constant=np.array(flat))
    with open(os.path.join(RESULTS, "sample_matrix.csv"), "w") as fh:
        fh.write("name,kind,seed,sweep," + ",".join(cols) + "\n")
        for r, v in zip(rows, M):
            fh.write("%s,%s,%d,%s," % (r["name"], r["kind"], r["seed"],
                                       r["sweep"]))
            fh.write(",".join("%.8g" % x for x in v) + "\n")
    cubeM = np.array([[cube[k][q] for k in cube_keys]
                      for q in range(len(names))])
    np.savez(os.path.join(RESULTS, "sample_matrix_azimuth.npz"),
             names=np.array(names), kinds=np.array(kinds),
             sweeps=np.array([r["sweep"] for r in rows]),
             az_deg=az, columns=np.array(cube_keys), values=cubeM,
             within_sd=within, between_sd=between, r_az180=per,
             shift_alignments_30=shifts)
    with open(os.path.join(RESULTS, "sample_matrix_azimuth.csv"), "w") as fh:
        fh.write("name,kind,az_deg," + ",".join(cube_keys) + "\n")
        for q, r in enumerate(rows):
            for a in range(len(az)):
                fh.write("%s,%s,%d," % (r["name"], r["kind"], az[a]))
                fh.write(",".join("%.8g" % cubeM[q, c, a]
                                  for c in range(len(cube_keys))) + "\n")
    return M, cubeM


# ------------------------------------------------------------------ main

def main():
    import time
    report_premises(*check_shared_and_replay())

    reg = specimen_registry()
    store, rows, cubes = {}, [], []
    print("\nbuilding the matrix, %d specimens by %d azimuths"
          % (len(reg), len(AZ)))
    for r in reg:
        t0 = time.time()
        row, cube = specimen_row(r, store)
        rows.append(row)
        cubes.append(cube)
        print("  %-18s %3d grains  %5.1f s"
              % (r["name"], row["n_grain"], time.time() - t0), flush=True)

    skip = {"name", "kind", "fabric", "sweep"}
    cols = [k for k in rows[0] if k not in skip]
    cube_keys = [k for k in cubes[0] if k != "az_deg"]
    cube = {k: np.array([c[k] for c in cubes]) for k in cube_keys}
    names = [r["name"] for r in rows]
    kinds = [r["kind"] for r in rows]

    report_matrix(rows, cols)
    flat = report_variation(rows, cols)
    regimes = report_regimes(kinds, AZ, cube, cube_keys)
    report_weighting(rows)
    report_reconciliation(rows)
    write_outputs(rows, cols, AZ, cube, cube_keys, names, kinds, flat,
                  regimes)
    print("\nwrote %s" % os.path.join(RESULTS, "sample_matrix.npz"))
    print("wrote %s" % os.path.join(RESULTS, "sample_matrix.csv"))
    print("wrote %s" % os.path.join(RESULTS, "sample_matrix_azimuth.npz"))
    print("wrote %s" % os.path.join(RESULTS, "sample_matrix_azimuth.csv"))
    print("%d specimens, %d per-specimen properties, %d per-azimuth "
          "properties on %d azimuths" % (len(rows), len(cols),
                                         len(cube_keys), len(AZ)))
    print("%d properties are effectively constant across the eight girdle "
          "tessellations" % len(flat))


if __name__ == "__main__":
    main()
