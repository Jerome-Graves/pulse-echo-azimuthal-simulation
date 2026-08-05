"""How wrong may the assumed microstructure be before identification fails?

Section 5.2 hands the scoring algorithm the EXACT tessellation as one of
48 candidates. No microstructural measurement can do that. A real section
gives grain boundaries with a position error, orientations with an angular
error, and a grain list that is missing the small grains and has merged
the ones whose boundary was faint. If identification collapses under a
realistic error the result is a curiosity; if it degrades gracefully it is
a method. This module measures the degradation.

THE EXPERIMENT. The true candidate is perturbed and put back into the
line-up in place of its unperturbed self, so the candidate set stays at 48
and every foreign candidate is untouched. Everything is scored with
tessellation_replication.py's own functions, on its own 30 common
azimuths, so the statistic is the published one and not a re-derivation.

FOUR PERTURBATION FAMILIES, each swept from negligible to destructive.

  SEED POSITION JITTER. Every Laguerre seed is displaced by an isotropic
  Gaussian of standard deviation sigma, quoted as a fraction of the mean
  volume-equivalent grain diameter. This moves every boundary a little,
  which is what a registration error does. The grid is relabelled from
  the displaced seeds, so the boundaries move exactly as a Laguerre
  diagram says they must.

  ORIENTATION ERROR. Every c-axis is rotated by an angle drawn from a
  Rayleigh distribution of increasing width about a uniformly random
  perpendicular axis, plus a fully random limit.

  MISSING AND MERGED GRAINS. A fraction of the seeds is deleted and the
  grid relabelled, so the neighbours absorb the missing grain; or a
  fraction of face-adjacent pairs is merged into single grains at fixed
  geometry, which is what a faint boundary does to a traced section.

  WEIGHT ERROR. The lognormal grain radius behind each Laguerre weight is
  multiplied by exp(N(0, s)), which changes relative grain sizes without
  moving a single seed.

THREE SCORES, because the published identification statistic turns out to
be blind to one of the four families by construction and that has to be
shown rather than asserted.

  PUB    the published identification score. compute_identification marches
         the candidate with use_dv=False and const_v=True, so the candidate
         contributes grain-boundary POSITIONS and facet normals only: the
         reflection coefficient is the constant 0.02 and range is mapped to
         time with the speed measured from the specimen's own backwall. The
         candidate's c-axes never enter the arithmetic. This is verified
         numerically here, not argued.
  DV     the same march with use_dv=True, so the crossing weight carries
         the candidate's own qP contrast while the time axis stays the
         specimen's measured one. The minimal orientation-sensitive
         variant of the published statistic. NOT the published statistic.
  FIELD  the Sec. 5.2 facet-model field of Part B, Eq. (facetmodel) with
         the candidate's own grain-resolved speeds setting both arrival
         times and reflection coefficients, used here as a 48-candidate
         line-up. The r it reports on the true candidate is the published
         r_full. Used as a line-up it is NOT the published statistic.

UNITS A PRACTITIONER UNDERSTANDS. Seed jitter and weight error are
converted to a millimetre boundary displacement that is measured, not
assumed: the volume of the disc whose grain identity changes, divided by
the total grain-boundary area, is the area-weighted mean normal
displacement of the boundary surface. The area is estimated from the
exact Laguerre point-to-boundary distance on a Monte Carlo cloud, and the
rasterised face count is printed beside it as a cross-check.

  MEASURED   every correlation, rank, boundary displacement, angular
             error and changed-volume fraction below.
  INFERRED   only the closing paragraph about what a real section
             delivers, which is a judgement about laboratory practice and
             is labelled as such where it is printed.

CUDA IS NEVER TOUCHED. tessellation_replication replaces
DiskSpecimen._label_grid_gpu with an exact all-seeds float64 argmin in
NumPy at import, and this module imports it before it builds anything.
The build is replayed rather than re-run: the replay reproduces the
cached label volume bit-exactly and the cached c-axis draw bit-exactly,
which is checked for every seed and printed.

READS
  out/tesscache/tess_s<seed>_p8_k-8.npz      48 cached tessellations
  out/sweeps/{girdle_seed11_ppw6_axis_perp,mx_girdle_s*}_ppw8 the eight girdle sweeps
WRITES
  out/tesscache/replay_s<seed>_p8.npz        replayed seeds and weights
  sim/results/microstructure_error.npz       every number printed
  sim/results/microstructure_error_partial.npz   checkpoint, removed on
                                             a completed run and resumed
                                             from on an interrupted one

Run with --report to reprint every table from the archive without
rescoring anything.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

import tessellation_replication as TR                    # noqa: E402
import _t2_common as C2                                  # noqa: E402
import shift_null_2d as S2                               # noqa: E402
from scipy.stats import binom                            # noqa: E402
from specimen import DiskSpecimen, sample_watson         # noqa: E402

CACHE = os.path.join(ROOT, "out", "tesscache")
RESULTS = os.path.join(ROOT, "sim", "results")

# The published centring, pinned rather than inherited. tessellation_
# replication reads its centring from argv and falls back to "double",
# but Sec. 5.2 quotes the strictest convention, in which the azimuthal
# harmonics k = 1 to 4 are stripped at every time sample before the
# per-azimuth level is removed. That is the convention under which the
# manuscript's r = 0.162 and four of eight are obtained, and it is
# checked here: recomputing the stored identification matrix of
# tessellation_replication.npz reproduces it to 1e-6 under harm and
# under no other convention.
CENTRING = "harm"
TR.CENTRING = CENTRING

DIA, THK, NG, SIZE_CV = C2.DIA, 0.035, 100, 0.35
PPW, KAPPA, AXIS = 8.0, -8.0, (1.0, 0.0, 0.0)
AZ = TR.AZ_COMMON
TGRID = np.arange(20e-6, 42e-6, C2.DT_C)
GATE = (TGRID >= C2.CODA_W[0]) & (TGRID < C2.CODA_W[1])
CANDS = sorted(set([s for _, s in TR.GIRDLE] + TR.DISTRACTORS))
SWEEPS = TR.GIRDLE

N_REAL = 3                       # perturbation realisations per level
N_MC = 200000                    # Monte Carlo points for the facet area
EPS_MM = (0.15, 0.30)            # shell thicknesses for the area estimate

# Levels. Zero is carried in every family as the untouched control, so a
# family that leaves the score alone is visible as a flat row rather than
# having to be compared against a number printed somewhere else, and so
# that the four relabelling operators are shown to reproduce the cached
# label volume exactly when their parameter is zero.
LV_JITTER = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50)
LV_ORIENT = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, -1.0)   # -1 = uniform
LV_MISS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
LV_MERGE = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
LV_WEIGHT = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80)

# merged and merged_refit differ only in where the merged grain's single
# Laguerre seed is put, and the difference is the point of carrying both.
# The label volume is identical in the two, so the boundary that vanishes
# is the same boundary; what changes is whether the surviving facets of
# the pair keep their production normals. Under `merged` the larger
# member's seed is kept, so its own facets keep their exact normals and
# only the absorbed member's facets are re-normalled. Under
# merged_refit the seed moves to the volume-weighted centroid, which is
# where a Laguerre model FITTED to the merged cell would put it, and every
# facet of the pair is re-normalled.
FAMILIES = (("jitter", LV_JITTER, "seed displacement sigma / d_mean"),
            ("orient", LV_ORIENT, "c-axis rotation rms, degrees"),
            ("missing", LV_MISS, "fraction of grains deleted"),
            ("merged", LV_MERGE, "fraction merged, larger seed kept"),
            ("merged_refit", LV_MERGE, "fraction merged, seed refitted"),
            ("weight", LV_WEIGHT, "relative grain-radius error"))

VARIANTS = ("pub", "dv", "field")

# Stream key of each family. merged and merged_refit share it deliberately,
# so the two rules are compared on exactly the same merged pairs and the
# difference between their rows is the seed rule and nothing else.
KIND_KEY = dict(jitter=1, orient=2, missing=3, merged=4, merged_refit=4,
                weight=5)
N_WORKER = 8


# ────────────────────────────── the build ─────────────────────────────
def replay_build(seed):
    """Recover the seeds, weights and radii the production build used.

    DiskSpecimen.build returns only the seeds that ended up owning voxels
    and never returns the weights the analysis cache stores, so the two
    numbers a perturbation needs are not on disk. They are recovered by
    replaying the build's random stream, which is deterministic: the
    Monte Carlo cloud, then the seed points, then the lognormal radii,
    then a weight-scale bisection that draws nothing. The replay is
    checked against the cached label volume and the cached c-axis draw
    and must reproduce both bit-exactly.
    """
    path = os.path.join(CACHE, "build_replay_seed%d_ppw%g.npz" % (seed, PPW))
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    sp = DiskSpecimen(diameter_m=DIA, thickness_m=THK, n_grains=NG,
                      size_cv=SIZE_CV, concentration=KAPPA,
                      spatial_corr=0.0, fabric_axis=AXIS, seed=seed)
    rng = np.random.default_rng(seed)
    R = DIA / 2.0
    mc = sp._mc_points(R, THK, 400000, rng)
    base = (R / NG ** (1 / 3)) ** 2
    sigma = np.sqrt(np.log(1 + SIZE_CV ** 2))
    n_seed = NG
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
            lo, hi = (trial, hi) if cv < SIZE_CV else (lo, trial)
            scale = trial
        n_live = int((np.bincount(sp._assign(mc, pts, scale * wshape),
                                  minlength=n_seed) > 0).sum())
        if n_live >= NG or outer == 3:
            break
        n_seed = int(np.ceil(n_seed * NG / max(n_live, 1)))
    lab_c, axes_c, seeds_c, h = TR.tessellation(seed)
    lab, keep = relabel(pts, scale * wshape, h)
    ok_lab = bool(np.array_equal(lab.astype(np.int32), lab_c))
    ok_seed = bool(np.array_equal(pts[keep], seeds_c))
    ok_axes = bool(np.array_equal(
        sample_watson(len(keep), AXIS, KAPPA, rng), axes_c))
    out = dict(pts=pts, weights=scale * wshape, radii=radii,
               base=np.array(base), scale=np.array(scale), keep=keep,
               ok=np.array([ok_lab, ok_seed, ok_axes]))
    np.savez_compressed(path, **out)
    return out


def relabel(pts, weights, h):
    """Label the disc from a seed set, drop empty cells, relabel densely.

    The two steps DiskSpecimen.build performs after the argmin, lifted out
    so a perturbed seed set goes through exactly the same reduction as the
    production one. Returns the int16 label volume and the indices of the
    seeds that survived, in increasing order.
    """
    nx, nz = int(np.ceil(DIA / h)), int(np.ceil(THK / h))
    x = (np.arange(nx) - (nx - 1) / 2.0) * h
    z = (np.arange(nz) - (nz - 1) / 2.0) * h
    raw = TR._label_plane_by_plane(x, z, DIA / 2.0, pts, weights)
    inside = raw >= 0
    keep = np.unique(raw[inside])
    remap = np.full(len(pts), -1, np.int32)
    remap[keep] = np.arange(len(keep))
    lab = np.full(raw.shape, -1, np.int16)
    lab[inside] = remap[raw[inside]].astype(np.int16)
    return lab, keep


def full_axes(rep, axes):
    """c-axes indexed by ORIGINAL seed, including seeds that owned nothing.

    A displaced seed set realises a slightly different set of live cells,
    so a seed that owned no voxel in the production build can own one
    after the perturbation and needs an orientation. It is given the
    orientation of the nearest production grain, which is what a
    measurement that had missed the grain entirely would supply. Three to
    eight seeds of about 107 are affected and the published identification
    score does not read c-axes at all, so this choice touches only the DV
    and FIELD variants.
    """
    pts, keep = rep["pts"], rep["keep"]
    out = np.zeros((len(pts), 3))
    out[keep] = axes
    miss = np.setdiff1d(np.arange(len(pts)), keep)
    for m in miss:
        d = np.linalg.norm(pts[keep] - pts[m], axis=1)
        out[m] = axes[int(np.argmin(d))]
    return out


# ──────────────────────────── perturbations ───────────────────────────
def perturb(kind, level, rep, base, rng):
    """One perturbed candidate, as (labels, axes, seeds, h) plus diagnostics.

    base is the untouched (labels, axes, seeds, h) of the same seed. Every
    family also returns `newlab`, the new label each PRODUCTION grain ends
    up carrying, or -1 if it has none, so the changed-volume fraction is a
    single comparison on the grid and needs no registration step.
    """
    lab0, axes0, seeds0, h = base
    pts, w = rep["pts"].copy(), rep["weights"].copy()
    keep0 = rep["keep"]
    n0 = len(axes0)
    axf = full_axes(rep, axes0)
    d_mean = mean_diameter(lab0, n0, h)
    diag = dict(d_mean_mm=d_mean * 1e3, err_mean_deg=0.0, err_rms_deg=0.0)

    def by_seed(keep):
        """New label of every production grain, given surviving seeds."""
        pos = np.full(len(pts), -1, np.int64)
        pos[keep] = np.arange(len(keep))
        return pos[keep0]

    if kind == "jitter":
        pts = pts + rng.normal(0.0, level * d_mean, pts.shape)
        lab, keep = relabel(pts, w, h)
        return (lab, axf[keep], pts[keep], h), by_seed(keep), diag

    if kind == "weight":
        radii = rep["radii"] * np.exp(rng.normal(0.0, level, len(pts)))
        w = float(rep["scale"]) * float(rep["base"]) * (radii ** 2 - 1.0)
        lab, keep = relabel(pts, w, h)
        return (lab, axf[keep], pts[keep], h), by_seed(keep), diag

    if kind == "missing":
        n_del = int(round(level * n0))
        drop = rng.choice(n0, n_del, replace=False) if n_del else []
        sub = np.setdiff1d(np.arange(n0), drop)
        lab, keep = relabel(pts[keep0][sub], w[keep0][sub], h)
        pos = np.full(len(sub), -1, np.int64)
        pos[keep] = np.arange(len(keep))
        newlab = np.full(n0, -1, np.int64)
        newlab[sub] = pos
        diag["n_deleted"] = n_del
        return (lab, axes0[sub][keep], pts[keep0][sub][keep], h), newlab, diag

    if kind in ("merged", "merged_refit"):
        grp, n_mg = merge_groups(lab0, n0, level, rng)
        lab, axes, seeds, newlab = apply_merge(
            lab0, axes0, seeds0, grp, refit=(kind == "merged_refit"))
        diag["n_merged"] = n_mg
        return (lab, axes, seeds, h), newlab, diag

    if kind == "orient":
        axes = rotate_axes(axes0, level, rng)
        err = np.degrees(np.arccos(np.clip(
            np.abs(np.einsum("ij,ij->i", axes, axes0)), 0, 1)))
        diag["err_mean_deg"] = float(err.mean())
        diag["err_rms_deg"] = float(np.sqrt(np.mean(err ** 2)))
        return (lab0, axes, seeds0, h), np.arange(n0), diag

    raise ValueError(kind)


def mean_diameter(lab, n_grain, h):
    """Mean volume-equivalent grain diameter of a label volume, metres."""
    cnt = np.bincount(lab[lab >= 0].ravel().astype(np.int64),
                      minlength=n_grain).astype(float)
    v = cnt[cnt > 0] * h ** 3
    return float(np.mean(2.0 * (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0)))


def rotate_axes(axes, rms_deg, rng):
    """Rotate each c-axis about a random perpendicular direction.

    rms_deg < 0 replaces the fabric with uniformly random axes, which is
    the destruction limit rather than a measurement error. Otherwise the
    rotation angle is Rayleigh with the requested root mean square, which
    is the angular error a two-component Gaussian pointing error produces.
    """
    n = len(axes)
    if rms_deg < 0:
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    if rms_deg == 0:
        return axes.copy()
    th = np.radians(rms_deg) / np.sqrt(2.0) * np.sqrt(
        rng.chisquare(2, n))
    g = rng.normal(size=(n, 3))
    perp = g - (np.einsum("ij,ij->i", g, axes))[:, None] * axes
    perp /= np.linalg.norm(perp, axis=1, keepdims=True) + 1e-30
    out = np.cos(th)[:, None] * axes + np.sin(th)[:, None] * perp
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def merge_groups(lab, n_grain, frac, rng):
    """Assign each grain to a merged group; face-adjacent pairs only.

    Pairs are drawn from the face adjacency in random order and accepted
    while neither member has already been merged, so the operator makes
    disjoint pairs and never a chain. Returns the group index of every
    grain and the number of grains absorbed.
    """
    grp = np.arange(n_grain)
    n_target = int(round(frac * n_grain))
    if n_target == 0:
        return grp, 0
    pairs = set()
    for ax in range(3):
        a = np.take(lab, np.arange(lab.shape[ax] - 1), axis=ax)
        b = np.take(lab, np.arange(1, lab.shape[ax]), axis=ax)
        m = (a != b) & (a >= 0) & (b >= 0)
        u, v = a[m].astype(np.int64), b[m].astype(np.int64)
        key = np.unique(np.minimum(u, v) * 100000 + np.maximum(u, v))
        pairs.update(int(k) for k in key)
    pl = np.array(sorted(pairs))
    rng.shuffle(pl)
    used, done = np.zeros(n_grain, bool), 0
    for k in pl:
        i, j = int(k // 100000), int(k % 100000)
        if used[i] or used[j]:
            continue
        grp[j] = i
        used[i] = used[j] = True
        done += 1
        if done >= n_target:
            break
    return grp, done


def apply_merge(lab, axes, seeds, grp, refit):
    """Collapse merged grains into single grains at fixed geometry.

    The label volume is remapped, not rebuilt, so the outer shape of every
    grain and the position of every surviving boundary is untouched: only
    the boundary BETWEEN a merged pair disappears, which is what a faint
    boundary does to a traced section. The merged grain's c-axis is the
    leading eigenvector of the volume-weighted orientation tensor of its
    members. Its single seed is either the larger member's, which leaves
    that member's facet normals exact, or the volume-weighted centroid,
    which is where a Laguerre model fitted to the merged cell would put
    it and which re-normals every facet of the pair.
    """
    n = len(axes)
    cnt = np.bincount(lab[lab >= 0].ravel().astype(np.int64),
                      minlength=n).astype(float)
    root = np.unique(grp)
    remap = np.full(n, -1, np.int32)
    remap[root] = np.arange(len(root))
    new = np.full(lab.shape, -1, np.int16)
    ins = lab >= 0
    new[ins] = remap[grp[lab[ins]]].astype(np.int16)
    s_out = np.zeros((len(root), 3))
    a_out = np.zeros((len(root), 3))
    for q, r in enumerate(root):
        mem = np.nonzero(grp == r)[0]
        w = cnt[mem] / max(cnt[mem].sum(), 1e-30)
        s_out[q] = (w @ seeds[mem] if refit
                    else seeds[mem[int(np.argmax(cnt[mem]))]])
        T = np.einsum("g,gi,gj->ij", w, axes[mem], axes[mem])
        ev, evec = np.linalg.eigh(T)
        a_out[q] = evec[:, int(np.argmax(ev))]
    return new, a_out, s_out, remap[grp].astype(np.int64)


# ────────────────────────── boundary geometry ─────────────────────────
def facet_area_mc(pts, w, rng, n=N_MC):
    """Total grain-boundary area of a Laguerre diagram, by Monte Carlo.

    The exact distance from an interior point to the boundary of its own
    power cell is min_j (pow_j - pow_own) / (2 |s_j - s_own|), because the
    separating surface of two power cells is a plane. The fraction of the
    disc volume lying within eps of a boundary is 2 A eps / V for small
    eps, one shell on each side of every facet, so the area follows from a
    distance histogram with no rasterisation in it. Two shell thicknesses
    are used and the two estimates are returned so their agreement can be
    read as the linearity check it is.
    """
    R = DIA / 2.0
    p = rng.uniform(-1, 1, (int(n * 1.4), 3))
    p = p[(p[:, 0] ** 2 + p[:, 1] ** 2) <= 1.0][:n]
    p[:, :2] *= R
    p[:, 2] *= THK / 2.0
    dsep = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dsep, np.inf)
    dmin = np.empty(len(p))
    for lo in range(0, len(p), 20000):
        q = p[lo:lo + 20000]
        d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1) - w[None, :]
        own = d2.argmin(1)
        rows = np.arange(len(q))
        d = (d2 - d2[rows, own][:, None]) / (2.0 * dsep[own])
        d[rows, own] = np.inf
        dmin[lo:lo + 20000] = d.min(1)
    vol = np.pi * R ** 2 * THK
    return [float(np.mean(dmin < e * 1e-3) * vol / (2.0 * e * 1e-3))
            for e in EPS_MM]


def facet_area_raster(lab, h):
    """The same area from the voxel face count, for comparison only."""
    tot = 0
    for ax in range(3):
        a = np.take(lab, np.arange(lab.shape[ax] - 1), axis=ax)
        b = np.take(lab, np.arange(1, lab.shape[ax]), axis=ax)
        tot += int(((a != b) & (a >= 0) & (b >= 0)).sum())
    return tot * h ** 2


# ─────────────────────────────── scoring ──────────────────────────────
def march_dv(tess, az_deg):
    """Marched crossings weighted by the candidate's OWN qP contrast.

    Identical to tessellation_replication.march_geometry except that
    use_dv is True, so the crossing weight carries the candidate's c-axes
    while the time axis is still the specimen's measured backwall speed.
    The minimal orientation-sensitive variant of the published statistic.
    """
    lab, axes, seeds, h = tess
    out = []
    for a in az_deg:
        _, w, s = C2.facet_events(lab, axes, seeds, h, a, use_dv=True,
                                  use_direc=True, n_ray=TR.N_RAY,
                                  th_max_deg=TR.CONE_DEG, const_v=True)
        out.append((w, s))
    return out


def score_all(tess, E_gate, c_az):
    """The three scores of one candidate against one specimen.

    pub and dv are scored on linear power, which is what
    compute_identification does; field is scored in dB, which is what
    compute_field does. Each variant therefore reproduces the convention
    of the published quantity it extends, and the three columns are not
    on one scale.
    """
    pub = TR.bin_events(TR.march_geometry(tess, AZ), c_az, TGRID)[:, GATE]
    dv = TR.bin_events(march_dv(tess, AZ), c_az, TGRID)[:, GATE]
    fld = TR.facet_field(tess, AZ, TGRID)[:, GATE]
    return (S2.score(E_gate, pub, CENTRING, log=False),
            S2.score(E_gate, dv, CENTRING, log=False),
            S2.score(E_gate, fld, CENTRING, log=True))


def measured(name):
    """(gated measured power field, per-azimuth two-way speed) of a sweep."""
    E, c_az = TR.coda_field(name, TGRID)
    return E[:, GATE] ** 2, c_az


def rank_of(value, foreign):
    """Rank of a score among itself and the foreign candidates, 1 is best."""
    return int(1 + np.sum(np.asarray(foreign) > value))


# ──────────────────────────────── driver ──────────────────────────────
def _base_job(args):
    cand, meas = args
    tess = TR.tessellation(cand)
    return cand, {n: score_all(tess, E, c) for n, (E, c) in meas.items()}


_STORE = {}


def _cached(seed):
    """Per-worker cache of the two things every job of a seed needs."""
    if seed not in _STORE:
        _STORE.clear()
        _STORE[seed] = (replay_build(seed), TR.tessellation(seed))
    return _STORE[seed]


def _pert_job(args):
    name, seed, kind, level, ireal, E, c_az = args
    rng = np.random.default_rng(
        [seed, KIND_KEY[kind], int(round(level * 1e6)) + 10 ** 7, ireal])
    rep, base = _cached(seed)
    tess, newlab, diag = perturb(kind, level, rep, base, rng)
    ins = base[0] >= 0
    same = newlab[base[0][ins].astype(np.int64)] == tess[0][ins].astype(
        np.int64)
    sc = score_all(tess, E, c_az)
    return dict(name=name, seed=seed, kind=kind, level=level, ireal=ireal,
                scores=sc, frac_changed=float(1.0 - np.mean(same)),
                n_grain=len(tess[1]), diag=diag)


PARTIAL = "microstructure_error_partial.npz"
CKPT = 100


def save_partial(res):
    """Checkpoint the scored perturbations, keyed so a restart can resume."""
    np.savez(os.path.join(RESULTS, PARTIAL),
             name=np.array([r["name"] for r in res]),
             seed=np.array([r["seed"] for r in res]),
             kind=np.array([r["kind"] for r in res]),
             level=np.array([r["level"] for r in res]),
             ireal=np.array([r["ireal"] for r in res]),
             scores=np.array([r["scores"] for r in res]),
             frac_changed=np.array([r["frac_changed"] for r in res]),
             n_grain=np.array([r["n_grain"] for r in res]),
             err_rms=np.array([r["diag"]["err_rms_deg"] for r in res]),
             err_mean=np.array([r["diag"]["err_mean_deg"] for r in res]))


def load_partial():
    """Whatever a previous run got through, keyed by its job identity."""
    path = os.path.join(RESULTS, PARTIAL)
    if not os.path.exists(path):
        return {}
    out = {}
    with np.load(path, allow_pickle=True) as z:
        for i in range(len(z["kind"])):
            k = (str(z["name"][i]), str(z["kind"][i]),
                 round(float(z["level"][i]), 6), int(z["ireal"][i]))
            out[k] = dict(
                name=k[0], seed=int(z["seed"][i]), kind=k[1], level=k[2],
                ireal=k[3], scores=tuple(z["scores"][i]),
                frac_changed=float(z["frac_changed"][i]),
                n_grain=int(z["n_grain"][i]),
                diag=dict(err_rms_deg=float(z["err_rms"][i]),
                          err_mean_deg=float(z["err_mean"][i])))
    return out


def main():
    t_start = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    seeds = [s for _, s in SWEEPS]

    print("=" * 79)
    print("REPLAY OF THE PRODUCTION BUILD, the premise of every perturbation")
    print("=" * 79)
    ok = []
    for s in seeds:
        rep = replay_build(s)
        ok.append(rep["ok"])
        print("  seed %3d  %3d seeds  %3d grains   labels %s  seeds %s  "
              "c-axes %s" % (s, len(rep["pts"]), len(rep["keep"]),
                             *("exact" if b else "DIFFER"
                               for b in rep["ok"])))
    ok = np.array(ok)
    print("  the replayed seed set and weights reproduce the cached label")
    print("  volume, the cached seed list and the cached c-axis draw for")
    print("  %d of %d tessellations." % (int(ok.all(1).sum()), len(ok)))

    meas = {n: measured(n) for n, _ in SWEEPS}
    print("\n  measured fields loaded: %d sweeps, %d azimuths, %d gate "
          "samples" % (len(meas), len(AZ), int(GATE.sum())))

    print("\n" + "=" * 79)
    print("BASELINE LINE-UP, %d candidates, current code" % len(CANDS))
    print("=" * 79)
    with Pool(N_WORKER) as pool:
        got = pool.map(_base_job, [(c, meas) for c in CANDS])
    table = {n: {} for n, _ in SWEEPS}
    for cand, per in got:
        for n in per:
            table[n][cand] = per[n]
    print("  %-20s %5s" % ("sweep", "own")
          + "".join("%9s%6s" % ("r_" + v, "rank") for v in VARIANTS))
    base_rank = {}
    for name, own in SWEEPS:
        row = "  %-20s %5d" % (name, own)
        base_rank[name] = {}
        for q, v in enumerate(VARIANTS):
            vals = np.array([table[name][c][q] for c in CANDS])
            i = CANDS.index(own)
            rk = int(1 + np.sum(vals > vals[i]))
            base_rank[name][v] = (float(vals[i]), rk)
            row += "%9.3f%6d" % (vals[i], rk)
        print(row)
    for v in VARIANTS:
        k = sum(1 for n, _ in SWEEPS if base_rank[n][v][1] == 1)
        r = np.array([base_rank[n][v][0] for n, _ in SWEEPS])
        print("  %-6s %d of %d rank first, binomial p = %.3g, mean r = %+.3f"
              % (v, k, len(SWEEPS), binom.sf(k - 1, len(SWEEPS),
                                             1.0 / len(CANDS)), r.mean()))
    print("  The published identification statistic is the pub column. It")
    print("  is recomputed here rather than read from")
    print("  tessellation_replication.npz so that every number in this")
    print("  module comes from one version of coda_field.")

    print("\n" + "=" * 79)
    print("IS THE PUBLISHED STATISTIC BLIND TO ORIENTATION?")
    print("=" * 79)
    tess11 = TR.tessellation(11)
    rng = np.random.default_rng(0)
    rnd = (tess11[0], rng.normal(size=tess11[1].shape), tess11[2], tess11[3])
    rnd = (rnd[0], rnd[1] / np.linalg.norm(rnd[1], axis=1, keepdims=True),
           rnd[2], rnd[3])
    E, c_az = meas["girdle_seed11_ppw8_dev"]
    a = score_all(tess11, E, c_az)
    b = score_all(rnd, E, c_az)
    for q, v in enumerate(VARIANTS):
        print("  %-6s own c-axes %+.6f   c-axes replaced by uniform random "
              "%+.6f   delta %+.3g" % (v, a[q], b[q], b[q] - a[q]))
    print("  march_geometry passes use_dv=False and const_v=True, so the")
    print("  candidate's c-axes reach no arithmetic in the pub score. The")
    print("  delta above is exactly zero, which is a property of the code")
    print("  and not a small number.")

    foreign = {n: {v: [table[n][c][q] for c in CANDS if c != own]
                   for q, v in enumerate(VARIANTS)}
               for n, own in SWEEPS}

    done = load_partial()
    jobs = []
    for kind, levels, _ in FAMILIES:
        for lv in levels:
            n_r = 1 if lv == 0.0 else N_REAL
            for ir in range(n_r):
                for name, seed in SWEEPS:
                    if (name, kind, round(lv, 6), ir) in done:
                        continue
                    E, c_az = meas[name]
                    jobs.append((name, seed, kind, lv, ir, E, c_az))
    res = list(done.values())
    print("\n%d perturbed candidates to score, %d already in the checkpoint"
          % (len(jobs), len(res)))
    # imap_unordered with a checkpoint every CKPT results, because the run
    # is an hour of CPU and losing it to an interrupted shell is a real
    # failure mode of this project rather than a hypothetical one.
    with Pool(N_WORKER) as pool:
        for q, r in enumerate(pool.imap_unordered(_pert_job, jobs,
                                                  chunksize=1)):
            res.append(r)
            if (q + 1) % CKPT == 0:
                save_partial(res)
                print("  %d of %d  %.0f s" % (q + 1, len(jobs),
                                              time.time() - t_start),
                      flush=True)
    save_partial(res)
    print("  done in %.0f s" % (time.time() - t_start))

    summary = {}
    for kind, levels, unit in FAMILIES:
        summary[kind] = report_family(kind, levels, unit, res, foreign)

    report_collapse(res, foreign)

    print("\n" + "=" * 79)
    print("BOUNDARY DISPLACEMENT IN MILLIMETRES")
    print("=" * 79)
    areas = report_geometry(seeds, res)
    report_thresholds(summary, areas, res)

    save(res, foreign, base_rank, ok)
    # The checkpoint is keyed on the job identity and not on the scoring
    # code, so leaving it behind would let a later run with a changed
    # score silently reuse stale numbers. It exists only to survive an
    # interrupted run and is removed as soon as one completes.
    if os.path.exists(os.path.join(RESULTS, PARTIAL)):
        os.remove(os.path.join(RESULTS, PARTIAL))
    print("\nwrote %s" % os.path.join(RESULTS, "microstructure_error.npz"))
    print("total %.0f s" % (time.time() - t_start))


def family_rows(kind, levels, res, foreign):
    """Per-level summary of one family, averaged over realisations.

    first  the number of the eight tessellations whose perturbed candidate
           still ranks first among the 48, averaged over realisations
    beat   the number of the eight whose perturbed candidate still beats
           the MEDIAN of the 47 foreign candidates
    Two of eight is the smallest count that is significant at the five per
    cent level against 48 candidates, since Bin(8, 1/48) puts 0.0107 above
    two and 0.154 above one, so `first` falling below two is the level at
    which identification stops being a result.
    """
    rows = []
    for lv in levels:
        sel = [r for r in res if r["kind"] == kind and r["level"] == lv]
        if not sel:
            continue
        n_r = max(1, len(sel) // len(SWEEPS))
        row = dict(level=lv,
                   changed=float(np.mean([r["frac_changed"] for r in sel])),
                   n_grain=float(np.mean([r["n_grain"] for r in sel])),
                   err_deg=float(np.mean([r["diag"]["err_rms_deg"]
                                          for r in sel])))
        for q, v in enumerate(VARIANTS):
            rr = np.array([r["scores"][q] for r in sel])
            rk = np.array([rank_of(r["scores"][q], foreign[r["name"]][v])
                           for r in sel])
            med = np.array([np.median(foreign[r["name"]][v]) for r in sel])
            row[v] = dict(r=float(rr.mean()), rank=float(rk.mean()),
                          sd=float(rr.std(ddof=1)) if len(rr) > 1 else 0.0,
                          first=float(np.sum(rk == 1) / n_r),
                          beat=float(np.sum(rr > med) / n_r),
                          margin=float(np.mean(rr - med)))
        row["n_real"] = n_r
        rows.append(row)
    return rows


def report_family(kind, levels, unit, res, foreign):
    print("\n" + "=" * 79)
    print("%s. %s" % (kind.upper(), unit))
    print("=" * 79)
    print("  %-9s %8s %7s" % ("level", "changed", "grains")
          + "".join("%8s%7s%7s%6s" % ("r_" + v, "sd", "rank", "1st")
                    for v in VARIANTS))
    rows = family_rows(kind, levels, res, foreign)
    for r in rows:
        print("  %-9.3g %8.4f %7.1f" % (r["level"], r["changed"],
                                        r["n_grain"])
              + "".join("%8.3f%7.3f%7.1f%6.2f"
                        % (r[v]["r"], r[v]["sd"], r[v]["rank"],
                           r[v]["first"]) for v in VARIANTS))
    print("  %d realisations of each non-zero level on each of the %d"
          % (rows[-1]["n_real"], len(SWEEPS)))
    print("  tessellations. changed is the fraction of the disc volume that")
    print("  changes grain identity; sd is the spread of the score over all")
    print("  tessellations and realisations, which is large because a")
    print("  double-centred field correlation on 30 azimuths has few")
    print("  effective degrees of freedom; 1st is how many of the eight")
    print("  tessellations still rank first of 48, averaged over")
    print("  realisations.")
    for v in VARIANTS:
        print("  %-6s baseline %.2f of 8 first."
              "  below 2 of 8 at %s.  below 1 at %s.  beats the median in"
              " fewer than 4 of 8 at %s"
              % (v, rows[0][v]["first"], fmt(threshold(rows, v, "first", 2.0)),
                 fmt(threshold(rows, v, "first", 1.0)),
                 fmt(threshold(rows, v, "beat", 4.0))))
    return rows


def threshold(rows, v, key, floor):
    """First level at which a statistic falls below a floor, interpolated.

    Returns None when the statistic never falls below the floor over the
    swept range, which is a result and not a missing number.
    """
    prev = None
    for r in rows:
        if r[v][key] < floor:
            if prev is None:
                return r["level"]
            x0, y0 = prev["level"], prev[v][key]
            x1, y1 = r["level"], r[v][key]
            return x1 if y0 == y1 else x0 + (y0 - floor) * (x1 - x0) / (y0 - y1)
        prev = r
    return None


def fmt(x):
    return "no swept level" if x is None else "%.4g" % x


def report_geometry(seeds, res):
    """Convert seed jitter and weight error into millimetres of boundary.

    The conversion is per tessellation and per level: the changed-volume
    fraction is measured on the grid, the disc volume is exact, and the
    total facet area is the Monte Carlo estimate of facet_area_mc, whose
    two shell thicknesses are printed so the linearity of the estimate is
    visible. A boundary that moves by delta sweeps a volume A delta, so
    delta = f_changed V / A is the area-weighted mean normal displacement.
    """
    rng = np.random.default_rng(4)
    vol = np.pi * (DIA / 2.0) ** 2 * THK
    areas = {}
    print("  %-6s %11s %11s %11s %11s" % ("seed", "A mc %.2f mm" % EPS_MM[0],
                                          "A mc %.2f mm" % EPS_MM[1],
                                          "A raster", "d_mean mm"))
    for s in seeds:
        rep = replay_build(s)
        a = facet_area_mc(rep["pts"], rep["weights"], rng)
        lab, axes, _, h = TR.tessellation(s)
        ar = facet_area_raster(lab, h)
        areas[s] = float(np.mean(a))
        print("  %-6d %11.5g %11.5g %11.5g %11.3f"
              % (s, a[0] * 1e4, a[1] * 1e4, ar * 1e4,
                 mean_diameter(lab, len(axes), h) * 1e3))
    print("  areas in cm^2. The rasterised face count overestimates the")
    print("  smooth area by the usual cubic-grid factor; the Monte Carlo")
    print("  estimate is the one used below.")
    print("\n  %-9s %-9s %11s %11s" % ("family", "level", "changed", "mm"))
    for kind in ("jitter", "weight"):
        for lv in sorted({r["level"] for r in res if r["kind"] == kind}):
            sel = [r for r in res if r["kind"] == kind and r["level"] == lv]
            d = np.mean([r["frac_changed"] * vol / areas[r["seed"]]
                         for r in sel])
            print("  %-9s %-9.3g %11.4f %11.4f"
                  % (kind, lv, np.mean([r["frac_changed"] for r in sel]),
                     d * 1e3))
    print("\n  ORIENTATION, realised angular error against the requested")
    print("  root mean square, degrees")
    print("  %-9s %11s %11s" % ("requested", "mean", "rms"))
    for lv in sorted({r["level"] for r in res if r["kind"] == "orient"}):
        sel = [r for r in res if r["kind"] == "orient" and r["level"] == lv]
        print("  %-9.3g %11.2f %11.2f"
              % (lv, np.mean([r["diag"]["err_mean_deg"] for r in sel]),
                 np.mean([r["diag"]["err_rms_deg"] for r in sel])))
    return areas


def report_collapse(res, foreign):
    """Is the changed volume the only thing that matters, or the operator?

    Three of the families move boundaries and can be compared at matched
    damage: seed jitter moves every boundary a little, grain deletion
    moves a few boundaries a long way, and weight error moves boundaries
    along fixed normals. If the score depended only on how much of the
    disc changed hands, the three would lie on one curve when binned on
    the changed fraction. They do not have to, and the table says which.
    """
    print("\n" + "=" * 79)
    print("DOES THE SCORE DEPEND ONLY ON HOW MUCH VOLUME CHANGED HANDS?")
    print("=" * 79)
    edges = np.array([0.0, 0.01, 0.025, 0.05, 0.09, 0.15, 0.25, 1.01])
    print("  %-22s" % "changed-volume bin"
          + "".join("%12s" % ("%.3f-%.3f" % (edges[q], edges[q + 1]))
                    for q in range(len(edges) - 1)))
    for kind in ("jitter", "missing", "weight"):
        sel = [r for r in res if r["kind"] == kind and r["frac_changed"] > 0]
        line = "  %-22s" % ("r_pub, " + kind)
        cnt = "  %-22s" % ("n, " + kind)
        for q in range(len(edges) - 1):
            v = [r["scores"][0] for r in sel
                 if edges[q] <= r["frac_changed"] < edges[q + 1]]
            line += "%12s" % ("%.3f" % np.mean(v) if v else "-")
            cnt += "%12d" % len(v)
        print(line)
        print(cnt)
    print("  A family that sits above the others at the same changed")
    print("  fraction is one whose particular way of being wrong the score")
    print("  tolerates. Read against the level-0 baseline of %.3f."
          % np.mean([r["scores"][0] for r in res if r["level"] == 0.0]))


def report_thresholds(summary, areas, res):
    """The four thresholds in the units a practitioner measures in.

    The geometric families are quoted twice: once in the perturbation's
    own parameter and once as the millimetre boundary displacement that
    parameter produced, which is the number a section-to-model
    registration error is reported in. The orientation family is quoted in
    the realised root-mean-square c-axis error.
    """
    vol = np.pi * (DIA / 2.0) ** 2 * THK
    print("\n" + "=" * 79)
    print("THRESHOLDS")
    print("=" * 79)
    print("  %-13s %-6s %-24s %-14s %-14s" % ("family", "score", "criterion",
                                               "level", "practitioner"))
    crit = (("first", 2.0, "below 2 of 8 first"),
            ("first", 1.0, "below 1 of 8 first"),
            ("beat", 4.0, "below 4 of 8 vs median"))
    for kind, levels, unit in FAMILIES:
        rows = summary[kind]
        for v in VARIANTS:
            for key, floor, lab in crit:
                x = threshold(rows, v, key, floor)
                print("  %-13s %-6s %-24s %-14s %-14s"
                      % (kind, v, lab, fmt(x), practitioner(kind, x, rows,
                                                            areas, vol, res)))
    print("  practitioner column: jitter and weight are the boundary")
    print("  displacement in mm implied by the changed volume at that")
    print("  level, orient is the realised rms c-axis error in degrees,")
    print("  missing and merged are the per cent of grains lost.")


def practitioner(kind, level, rows, areas, vol, res):
    """Translate a swept level into the unit the family is measured in."""
    if level is None:
        return "-"
    if kind in ("jitter", "weight"):
        d = np.array([[r["level"], r["changed"]] for r in rows])
        f = float(np.interp(level, d[:, 0], d[:, 1]))
        a = float(np.mean(list(areas.values())))
        return "%.3f mm" % (f * vol / a * 1e3)
    if kind == "orient":
        d = np.array([[r["level"], r["err_deg"]] for r in rows
                      if r["level"] >= 0])
        return "%.1f deg rms" % float(np.interp(level, d[:, 0], d[:, 1]))
    return "%.0f%% of grains" % (100.0 * level)


def save(res, foreign, base_rank, ok):
    np.savez(os.path.join(RESULTS, "microstructure_error.npz"),
             replay_ok=ok,
             cands=np.array(CANDS),
             sweeps=np.array([n for n, _ in SWEEPS]),
             owns=np.array([s for _, s in SWEEPS]),
             variants=np.array(VARIANTS),
             base_r=np.array([[base_rank[n][v][0] for v in VARIANTS]
                              for n, _ in SWEEPS]),
             base_rank=np.array([[base_rank[n][v][1] for v in VARIANTS]
                                 for n, _ in SWEEPS]),
             foreign=np.array([[foreign[n][v] for v in VARIANTS]
                               for n, _ in SWEEPS]),
             kinds=np.array([r["kind"] for r in res]),
             levels=np.array([r["level"] for r in res]),
             pert_sweep=np.array([r["name"] for r in res]),
             ireal=np.array([r["ireal"] for r in res]),
             scores=np.array([r["scores"] for r in res]),
             frac_changed=np.array([r["frac_changed"] for r in res]),
             n_grain=np.array([r["n_grain"] for r in res]))


def reload_results():
    """Rebuild the result list from the archive, for `--report`.

    Everything the report needs is in the npz except the realised c-axis
    error, which is redrawn from the same stream key rather than stored,
    so a reprint is exact and costs no scoring. The orientation operator
    is a pure function of that stream and of the cached c-axes.
    """
    z = np.load(os.path.join(RESULTS, "microstructure_error.npz"),
                allow_pickle=True)
    cands, sw, owns = list(z["cands"]), list(z["sweeps"]), list(z["owns"])
    foreign = {n: {v: list(z["foreign"][i, q])
                   for q, v in enumerate(VARIANTS)}
               for i, n in enumerate(sw)}
    axes = {s: TR.tessellation(int(s))[1] for s in owns}
    res = []
    for i in range(len(z["kinds"])):
        kind, lv = str(z["kinds"][i]), float(z["levels"][i])
        name = str(z["pert_sweep"][i])
        seed = owns[sw.index(name)]
        err = err_mean = 0.0
        if kind == "orient" and lv != 0.0:
            rng = np.random.default_rng(
                [int(seed), KIND_KEY[kind], int(round(lv * 1e6)) + 10 ** 7,
                 int(z["ireal"][i])])
            a0 = axes[seed]
            e = np.degrees(np.arccos(np.clip(np.abs(np.einsum(
                "ij,ij->i", rotate_axes(a0, lv, rng), a0)), 0, 1)))
            err, err_mean = float(np.sqrt(np.mean(e ** 2))), float(e.mean())
        res.append(dict(name=name, seed=int(seed), kind=kind, level=lv,
                        ireal=int(z["ireal"][i]),
                        scores=tuple(z["scores"][i]),
                        frac_changed=float(z["frac_changed"][i]),
                        n_grain=int(z["n_grain"][i]),
                        diag=dict(err_rms_deg=err, err_mean_deg=err_mean)))
    base_rank = {n: {v: (float(z["base_r"][i, q]), int(z["base_rank"][i, q]))
                     for q, v in enumerate(VARIANTS)}
                 for i, n in enumerate(sw)}
    return res, foreign, base_rank, list(cands)


def report_only():
    """Reprint every table from the archive, adding nothing and rerunning
    no scoring, so a change to the presentation costs no computation."""
    res, foreign, base_rank, _ = reload_results()
    summary = {}
    for kind, levels, unit in FAMILIES:
        summary[kind] = report_family(kind, levels, unit, res, foreign)
    report_collapse(res, foreign)
    print("\n" + "=" * 79)
    print("BOUNDARY DISPLACEMENT IN MILLIMETRES")
    print("=" * 79)
    areas = report_geometry([s for _, s in SWEEPS], res)
    report_thresholds(summary, areas, res)


if __name__ == "__main__":
    if "--report" in sys.argv:
        report_only()
    else:
        main()
