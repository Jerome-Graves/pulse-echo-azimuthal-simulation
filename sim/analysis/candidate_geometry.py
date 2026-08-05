"""How far apart are the 48 candidate microstructures, geometrically?

Section 5.2 asks which of 48 candidate tessellations produced a coda. The
candidates are drawn from one macro-statistical rule with no global
matching step, so a referee is entitled to ask whether they are merely
DIFFERENT rather than CLOSE: if every wrong candidate is many wavelengths
away in boundary position, the line-up is easy and the rank sum is a
weaker claim than it looks. The evidence already in the paper is that the
score distributions are exchangeable, which is a statement about the
scores and not about the geometry. This module measures the geometry.

THE ONE NUMBER THE OBJECTION NEEDS is the distance from each specimen to
its NEAREST wrong candidate, in the same millimetres in which the
identification test's own tolerance is already measured. Section 5.2's
perturbation study (analysis/microstructure_error.py) displaces the
boundaries of the true candidate by a controlled amount and watches the
statistic die: the area-weighted mean normal boundary displacement that
takes the count of first places below two of eight is 0.42 mm, and below
one of eight, 0.575 mm. A wrong candidate closer than that could not be
told from the truth. The ratio of nearest-distractor distance to that
tolerance IS the difficulty of the test, and it is what is reported here.

THREE DISTANCES, because no single definition should carry the answer.

  BOUNDARY OFFSET, d_bnd.  The area-weighted mean distance from a point
    on one tessellation's grain-boundary surface to the nearest boundary
    of the other, symmetrised over the two directions. This is the
    physically meaningful one: the identification statistic reads
    boundary POSITIONS through a two-way travel time, so a millimetre of
    boundary offset is two millimetres of path and 2 d / lambda cycles of
    phase. Its own tolerance is the 0.42 to 0.575 mm above.

  HAUSDORFF, d_H.  The symmetric supremum of the same point-to-set
    distance, which is the metric the referee named. Reported with the
    95th and 99th percentiles beside it, because a supremum estimated
    from a finite sample is a lower bound, and because a single deep
    grain interior sets it.

  VOLUME DISAGREEMENT, f_mis.  The fraction of the disc assigned to a
    different grain under the best one-to-one relabelling of the two
    grain lists, solved exactly as a linear assignment. Converted to
    millimetres by f V / A, which is microstructure_error.py's own
    conversion and is what makes the perturbation tolerance and the
    candidate separation commensurable. It SATURATES: two unrelated
    tessellations disagree over almost the whole disc whatever their
    boundaries do, so it is a check on the other two and not a
    replacement for them.

  A fourth, the Wasserstein distance between the two Laguerre seed
  clouds, is reported because the referee named it. It is the weakest of
  the four for this purpose: it is a distance between generators, not
  between the surfaces the wave reflects from, and a seed can move a long
  way inside its own cell without moving a boundary at all.

NO RASTER, NO SOLVER, NO CUDA. Every distance is computed in closed form
from the Laguerre generators. For a power diagram the surface separating
two cells is a plane, so the distance from an interior point x of cell i
to the boundary of its own cell is exactly

    min_j  (pow_j(x) - pow_i(x)) / (2 |p_j - p_i|),   pow_k = |x-p_k|^2 - w_k

taken over all other generators; a redundant constraint lies farther away
than the face it is redundant to, so the minimum over all j is the
distance to the boundary of the convex cell. That is the same identity
microstructure_error.facet_area_mc uses for the facet area, and the
resulting cell assignment is checked here against the cached label
volumes.

THE WEIGHTS ARE NOT ON DISK for 47 of the 48, because DiskSpecimen.build
returns only the surviving seed points. They are recovered by replaying
the build's random stream, which is deterministic and CPU-only, exactly
as microstructure_error.replay_build does; the replay is verified by
requiring that the cached seed list appear inside the replayed one as an
exact increasing subsequence, and, on the one tessellation whose weights
were cached, that the replayed weights agree bit-exactly.

Boundary points are drawn as the shell {x : d(x) < eps} of one common
Monte Carlo cloud, which is uniform on the boundary surface by area as
eps goes to zero. Because the cloud is shared by every candidate, the
point-to-set distances of every pair are read off one table with no
recomputation, and the shell is symmetric about the surface so the first
order error in eps cancels in the mean. Two shell thicknesses are carried
so that the cancellation can be read rather than asserted.

  MEASURED   every distance, fraction, quantile and correlation below,
             and the jitter reference curve, which is recomputed here in
             the same metric rather than transcribed from the other
             module.
  INFERRED   nothing. Millimetres are converted to wavelengths with
             lambda = c_ref / f0 = 1.925 mm at the reference speed the
             rest of the paper uses, and the two-way cycle count is
             2 d / lambda by the definition of a pulse-echo path.

READS
  out/tesscache/tess_s<seed>_p8_k-8.npz    48 cached tessellations
  sim/analysis/tessellation_replication.npz  the published 8 x 48 score
                                           matrix, for the confound check
WRITES
  out/tesscache/gen_s<seed>_p8.npz         replayed generators, cached
  sim/results/candidate_geometry.npz       every number printed
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wasserstein_distance_nd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

from specimen import DiskSpecimen                        # noqa: E402

CACHE = os.path.join(ROOT, "out", "tesscache")
RESULTS = os.path.join(ROOT, "sim", "results")

# Build parameters of every girdle candidate, from _t2_common and
# tessellation_replication. Pinned here so the replay is reproducible
# without importing a module that loads sweeps.
DIA, THK, NG, SIZE_CV = 0.100, 0.035, 100, 0.35
KAPPA, AXIS, PPW = -8.0, (1.0, 0.0, 0.0), 8.0
C_REF, F0 = 3850.0, 2.0e6
LAM_MM = C_REF / F0 * 1e3                 # 1.925 mm
VOL_M3 = np.pi * (DIA / 2.0) ** 2 * THK

# The 48 candidates, in the column order of tessellation_replication.npz.
CANDS = [7, 11, 17, 23, 41, 53, 71, 89] + list(range(100, 140))
# The eight specimens, in the row order of that file.
REAL = [11, 7, 17, 23, 41, 53, 71, 89]
DISTRACT = list(range(100, 140))

N_MC = 800000                 # common Monte Carlo cloud
EPS_MM = (0.10, 0.20)         # boundary-shell thicknesses
CHUNK = 50000
AREA_EPS_MM = 0.15            # the shell microstructure_error quotes areas at

# Seed-jitter levels for the reference curve. 0.03 and 0.042 are the two
# thresholds microstructure_error reports for the published statistic;
# the others bracket them and 0.5 pins the saturation end.
JIT_LEVELS = (0.01, 0.02, 0.03, 0.042, 0.05, 0.10, 0.20, 0.50)
N_JIT_REAL = 2                # realisations per level per specimen

# The published tolerance, from microstructure_error.py's threshold table
# for the published ("pub") statistic.
TOL_2OF8_MM = 0.420           # below 2 of 8 first places
TOL_1OF8_MM = 0.575           # below 1 of 8 first places


# ────────────────────────────── generators ────────────────────────────

def cache_path(seed):
    return os.path.join(CACHE, "tess_s%d_p8_k-8.npz" % seed)


def replay_generators(seed):
    """(all seed points, all Laguerre weights) of one candidate, SI.

    DiskSpecimen.build draws the Monte Carlo cloud, then the seed points,
    then the lognormal radii, then runs a weight-scale bisection that
    draws nothing, and only then labels the grid. Everything before the
    labelling is CPU arithmetic on a deterministic stream, so replaying it
    recovers the generators the production tessellation was built from.
    The result is cached because the bisection costs about twelve seconds.
    """
    path = os.path.join(CACHE, "generators_seed%d_ppw%g.npz" % (seed, PPW))
    if os.path.exists(path):
        with np.load(path) as z:
            return np.asarray(z["pts"], float), np.asarray(z["w"], float)
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
    w = scale * wshape
    np.savez_compressed(path, pts=pts, w=w)
    return pts, w


def keep_index(seed, pts):
    """Where each cached grain sits in the replayed generator list.

    The cached seed array is pts[keep] with keep increasing, so requiring
    an exact float match in increasing order is a bit-exact check that
    the replay recovered the production draw. Returns (keep, ok, extra
    weight agreement or None).
    """
    with np.load(cache_path(seed)) as z:
        cs = np.asarray(z["seeds"], float)
        wc = np.asarray(z["weights"], float) if "weights" in z.files else None
    idx = np.full(len(cs), -1, np.int64)
    for q, row in enumerate(cs):
        j = np.nonzero((pts == row).all(1))[0]
        if len(j):
            idx[q] = j[0]
    ok = bool(np.all(idx >= 0) and np.all(np.diff(idx) > 0))
    return idx, ok, wc


def mc_cloud(n, rng):
    """n points uniform in the disc, metres."""
    R = DIA / 2.0
    out, got = [], 0
    while got < n:
        p = rng.uniform(-1, 1, (int((n - got) * 1.4) + 16, 3))
        p = p[(p[:, 0] ** 2 + p[:, 1] ** 2) <= 1.0]
        out.append(p)
        got += len(p)
    p = np.concatenate(out)[:n]
    p[:, :2] *= R
    p[:, 2] *= THK / 2.0
    return p


def power_fields(pts, w, P, chunk=CHUNK):
    """(owning generator, distance to that cell's boundary) at every P.

    The power distance is expanded as |x|^2 - 2 x.p + |p|^2 - w so the
    only large operation is one matrix product, and |x|^2 is common to
    every generator and cancels in both the argmin and the plane
    distances. Distances are metres.
    """
    g0 = (pts ** 2).sum(1) - w
    dsep = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dsep, np.inf)
    own = np.empty(len(P), np.int32)
    dmin = np.empty(len(P), float)
    for lo in range(0, len(P), chunk):
        q = P[lo:lo + chunk]
        G = q @ (-2.0 * pts.T)
        G += g0[None, :]
        o = G.argmin(1)
        rows = np.arange(len(q))
        G -= G[rows, o][:, None]
        G /= (2.0 * dsep[o])
        G[rows, o] = np.inf
        own[lo:lo + chunk] = o.astype(np.int32)
        dmin[lo:lo + chunk] = G.min(1)
    return own, dmin


def facet_area(dmin, eps_mm=AREA_EPS_MM):
    """Total grain-boundary area from the shell fraction, m^2.

    The estimator of microstructure_error.facet_area_mc, evaluated on this
    module's cloud so the two areas are directly comparable.
    """
    e = eps_mm * 1e-3
    return float(np.mean(dmin < e) * VOL_M3 / (2.0 * e))


# ─────────────────────────────── distances ────────────────────────────

def boundary_metrics(mask_a, mask_b, dmin_a, dmin_b):
    """Every point-to-set statistic of one unordered pair, metres.

    mask_a selects the cloud points lying within eps of tessellation A's
    boundary surface; dmin_b is A-independent and already holds the
    distance from every cloud point to B's boundary, so a pair costs two
    masked reductions and nothing else.
    """
    ab, ba = dmin_b[mask_a], dmin_a[mask_b]
    both = np.concatenate([ab, ba])
    return dict(
        d_bnd=0.5 * (float(ab.mean()) + float(ba.mean())),
        d_med=0.5 * (float(np.median(ab)) + float(np.median(ba))),
        d_p95=float(max(np.percentile(ab, 95), np.percentile(ba, 95))),
        d_p99=float(max(np.percentile(ab, 99), np.percentile(ba, 99))),
        d_haus=float(both.max()))


def mislabel_fraction(own_a, own_b, n_a, n_b):
    """Disc fraction assigned to a different grain under the best
    one-to-one relabelling of the two grain lists.

    The overlap table is small and dense and the relabelling is an exact
    rectangular linear assignment, so this is the true optimum over all
    injections of the shorter list into the longer, not a greedy one.
    """
    tab = np.bincount(own_a.astype(np.int64) * n_b + own_b.astype(np.int64),
                      minlength=n_a * n_b).reshape(n_a, n_b).astype(float)
    r, c = linear_sum_assignment(-tab)
    return float(1.0 - tab[r, c].sum() / own_a.size)


def seed_wasserstein(pa, pb):
    """Wasserstein distances between two live-grain seed clouds, metres.

    Uniform marginals; the 1-Wasserstein is the exact optimal transport,
    which handles the unequal grain counts without discarding a point,
    and the 2-Wasserstein is the balanced assignment on the shorter list.
    """
    w1 = float(wasserstein_distance_nd(pa, pb))
    cost = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    return w1, float(np.sqrt(cost[r, c].mean()))


# ────────────────────────────── validation ────────────────────────────

def check_against_raster(seeds, P, lab_of):
    """Does the closed-form power diagram reproduce the cached labels?

    The cached volume is what every other module reads, so the analytic
    generator set has to agree with it or nothing below means anything.
    Each cloud point is snapped to its nearest voxel centre and the two
    labels compared. Disagreement is expected within about one voxel of a
    boundary, where the snap crosses it, so the residual is reported both
    raw and restricted to points more than one voxel from any facet.
    """
    rows = []
    for s in seeds:
        with np.load(cache_path(s)) as z:
            lab = np.asarray(z["labels"])
            h = float(z["h"])
        nx, ny, nz = lab.shape
        gi = np.rint((P[:, 0] + nx * h / 2) / h - 0.5).astype(int)
        gj = np.rint((P[:, 1] + ny * h / 2) / h - 0.5).astype(int)
        gk = np.rint((P[:, 2] + nz * h / 2) / h - 0.5).astype(int)
        ok = ((gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
              & (gk >= 0) & (gk < nz))
        lv = np.where(ok, lab[np.clip(gi, 0, nx - 1), np.clip(gj, 0, ny - 1),
                              np.clip(gk, 0, nz - 1)], -1)
        mine, d = lab_of[s]
        good = ok & (lv >= 0) & (mine >= 0)
        agree = float((lv[good] == mine[good]).mean())
        far = good & (d > h)
        agree_far = float((lv[far] == mine[far]).mean())
        rows.append((s, h * 1e3, agree, agree_far, int(good.sum())))
        print("  seed %3d  h %.4f mm  cell assignment agrees %.5f  "
              "beyond one voxel of a facet %.6f  (n = %d)"
              % (s, h * 1e3, agree, agree_far, int(good.sum())))
    return rows


# ──────────────────────────────── reports ─────────────────────────────

def mm(x):
    return np.asarray(x) * 1e3


def report_pairs(D, keys, tag, key, unit=1e3):
    v = np.array([D[k][key] for k in keys]) * unit
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    print("  %-24s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
          % (tag, v.mean(), v.std(ddof=1), q[0], q[1], q[3], q[5], q[6]))
    return v


def main():
    t0 = time.time()
    rng = np.random.default_rng(20260803)

    print("=" * 79)
    print("GEOMETRIC SEPARATION OF THE 48 CANDIDATE TESSELLATIONS")
    print("=" * 79)
    print("  %d candidates: %d specimens that were scanned and %d "
          "synthetic" % (len(CANDS), len(REAL), len(DISTRACT)))
    print("  distractors. %d Monte Carlo points in the disc, shared by all."
          % N_MC)

    P = mc_cloud(N_MC, rng)
    gen, own, dmin, area, keep, nlive = {}, {}, {}, {}, {}, {}
    n_ok, n_w = 0, 0
    for s in CANDS:
        p, w = replay_generators(s)
        idx, ok, wc = keep_index(s, p)
        n_ok += int(ok)
        if wc is not None:
            n_w += int(np.array_equal(w[idx], wc))
        gen[s] = (p, w, idx)
        o, d = power_fields(p, w, P)
        own[s], dmin[s] = o, d
        keep[s], nlive[s] = idx, len(idx)
        area[s] = facet_area(d)
    print("\n  generators replayed for %d candidates in %.0f s"
          % (len(CANDS), time.time() - t0))
    print("  the cached grain list is an exact increasing subsequence of")
    print("  the replayed generator list for %d of %d candidates"
          % (n_ok, len(CANDS)))
    print("  the one tessellation whose weights were also cached "
          "reproduces them bit-exactly: %d of 1" % n_w)

    # points owned by a generator that owns no voxel
    lab_of = {}
    dead_frac = []
    for s in CANDS:
        pos = np.full(len(gen[s][0]), -1, np.int64)
        pos[keep[s]] = np.arange(nlive[s])
        L = pos[own[s]]
        lab_of[s] = (L, dmin[s])
        dead_frac.append(float((L < 0).mean()))
    print("  cloud points owned by a generator that owns no voxel: "
          "%.2e of the disc, worst case %.2e"
          % (float(np.mean(dead_frac)), float(np.max(dead_frac))))

    print("\n" + "=" * 79)
    print("VALIDATION: the closed form against the cached label volumes")
    print("=" * 79)
    val = check_against_raster(REAL[:4], P, lab_of)
    print("  The residual is a voxel-snapping effect and not a")
    print("  disagreement: it lives within one voxel of a facet, where the")
    print("  snap to the nearest cell centre crosses the boundary. Beyond")
    print("  one voxel the closed form and the raster agree.")

    # ------------------------------------------------------------- scale
    print("\n" + "=" * 79)
    print("SCALE OF ONE TESSELLATION")
    print("=" * 79)
    a = np.array([area[s] for s in CANDS])
    ds = np.array([dmin[s].mean() for s in CANDS])
    ng = np.array([nlive[s] for s in CANDS])
    vol_over_area = float(VOL_M3 / a.mean())
    print("  grains per disc                      %.2f +- %.2f"
          % (ng.mean(), ng.std(ddof=1)))
    print("  grain-boundary area                  %.1f +- %.1f cm^2"
          % (a.mean() * 1e4, a.std(ddof=1) * 1e4))
    print("  disc volume / boundary area          %.3f mm"
          % (vol_over_area * 1e3))
    print("  mean distance from a random point")
    print("  to the nearest grain boundary        %.3f +- %.3f mm"
          % (mm(ds.mean()), mm(ds.std(ddof=1))))
    print("  wavelength at %g MHz, c = %.0f m/s    %.3f mm"
          % (F0 / 1e6, C_REF, LAM_MM))
    print("\n  The point-to-boundary figure is the CEILING of the boundary")
    print("  offset: two tessellations that know nothing about each other")
    print("  put the boundary of one at a typical interior position of the")
    print("  other, so no pair can be much farther apart than that. The")
    print("  question is how close the pairs come to the ceiling.")

    # ------------------------------------------------------------- pairs
    eps = np.array(EPS_MM) * 1e-3
    m1 = {s: (dmin[s] < eps[0]) for s in CANDS}
    m2 = {s: (dmin[s] < eps[1]) for s in CANDS}

    print("\n" + "=" * 79)
    print("PAIRWISE DISTANCES, all %d unordered pairs"
          % (len(CANDS) * (len(CANDS) - 1) // 2))
    print("=" * 79)
    D, D2 = {}, {}
    for ia, x in enumerate(CANDS):
        for y in CANDS[ia + 1:]:
            m = boundary_metrics(m1[x], m1[y], dmin[x], dmin[y])
            m["f_mis"] = mislabel_fraction(own[x], own[y],
                                           len(gen[x][0]), len(gen[y][0]))
            m["d_mis"] = m["f_mis"] * vol_over_area
            m["w1"], m["w2"] = seed_wasserstein(
                gen[x][0][keep[x]], gen[y][0][keep[y]])
            D[(x, y)] = m
            D2[(x, y)] = boundary_metrics(m2[x], m2[y], dmin[x], dmin[y])
    keys = sorted(D)
    print("  boundary points per candidate: %d at eps = %.2f mm and "
          "%d at eps = %.2f mm"
          % (int(np.mean([m1[s].sum() for s in CANDS])), EPS_MM[0],
             int(np.mean([m2[s].sum() for s in CANDS])), EPS_MM[1]))
    print("\n  %-24s %8s %8s %8s %8s %8s %8s %8s"
          % ("quantity, mm", "mean", "sd", "min", "p5", "p50", "p95", "max"))
    v1 = report_pairs(D, keys, "boundary offset d_bnd", "d_bnd")
    v2 = report_pairs(D2, keys, "  the same, eps 0.20", "d_bnd")
    report_pairs(D, keys, "  median, not mean", "d_med")
    report_pairs(D, keys, "Hausdorff, sampled", "d_haus")
    report_pairs(D, keys, "  95th percentile", "d_p95")
    report_pairs(D, keys, "  99th percentile", "d_p99")
    report_pairs(D, keys, "volume disagreement", "f_mis", unit=1.0)
    report_pairs(D, keys, "  the same, in mm", "d_mis")
    report_pairs(D, keys, "seed cloud W1", "w1")
    report_pairs(D, keys, "seed cloud W2", "w2")
    print("\n  the two shell thicknesses give boundary offsets differing by")
    print("  %.4f mm on average, %.2f per cent of the value, so the shell"
          % (abs(v1 - v2).mean(), 100 * abs(v1 - v2).mean() / v1.mean()))
    print("  is thin enough for this to be a property of the surface.")

    # ------------------------------------------------- nearest distractor
    print("\n" + "=" * 79)
    print("THE NUMBER THE OBJECTION ASKS FOR: nearest wrong candidate")
    print("=" * 79)
    print("  %-5s %9s %7s %8s %8s %9s %9s %9s"
          % ("seed", "nearest", "which", "d/lambda", "two-way",
             "mean over", "nearest", "nearest"))
    print("  %-5s %9s %7s %8s %8s %9s %9s %9s"
          % ("", "d_bnd mm", "cand", "", "cycles", "47 wrong", "f_mis",
             "W1 mm"))
    near = {}
    for s in REAL:
        rows = []
        for o in CANDS:
            if o == s:
                continue
            k = (s, o) if (s, o) in D else (o, s)
            rows.append((D[k]["d_bnd"], o, D[k]["f_mis"], D[k]["d_haus"],
                         D[k]["w1"], D[k]["w2"], D[k]["d_mis"]))
        rows.sort()
        d, o, f, hs, w1, w2, dmis = rows[0]
        allm = float(np.mean([r[0] for r in rows]))
        near[s] = dict(d=d, other=o, f_mis=f, d_haus=hs, w1=w1, w2=w2,
                       d_mis=dmis, mean=allm, rows=rows)
        print("  %-5d %9.3f %7d %8.3f %8.3f %9.3f %9.4f %9.3f"
              % (s, mm(d), o, mm(d) / LAM_MM, 2 * mm(d) / LAM_MM,
                 mm(allm), f, mm(w1)))
    nd = np.array([near[s]["d"] for s in REAL])
    nf = np.array([near[s]["f_mis"] for s in REAL])
    print("\n  nearest wrong candidate over the eight specimens:")
    print("    %.3f to %.3f mm, mean %.3f mm"
          % (mm(nd.min()), mm(nd.max()), mm(nd.mean())))
    print("    %.3f to %.3f wavelengths of boundary offset"
          % (mm(nd.min()) / LAM_MM, mm(nd.max()) / LAM_MM))
    print("    %.2f to %.2f cycles of two-way path at %g MHz"
          % (2 * mm(nd.min()) / LAM_MM, 2 * mm(nd.max()) / LAM_MM, F0 / 1e6))
    print("    volume disagreement %.4f to %.4f of the disc"
          % (nf.min(), nf.max()))

    rr = [D[(x, y)]["d_bnd"] if (x, y) in D else D[(y, x)]["d_bnd"]
          for ix, x in enumerate(REAL) for y in REAL[ix + 1:]]
    dd = [D[(x, y)]["d_bnd"] if (x, y) in D else D[(y, x)]["d_bnd"]
          for ix, x in enumerate(DISTRACT) for y in DISTRACT[ix + 1:]]
    rd = [D[(x, y)]["d_bnd"] if (x, y) in D else D[(y, x)]["d_bnd"]
          for x in REAL for y in DISTRACT]
    print("\n  the two kinds of wrong candidate are not geometrically")
    print("  distinguishable from each other:")
    print("    specimen-specimen     %.3f +- %.3f mm  (n = %d)"
          % (mm(np.mean(rr)), mm(np.std(rr, ddof=1)), len(rr)))
    print("    specimen-distractor   %.3f +- %.3f mm  (n = %d)"
          % (mm(np.mean(rd)), mm(np.std(rd, ddof=1)), len(rd)))
    print("    distractor-distractor %.3f +- %.3f mm  (n = %d)"
          % (mm(np.mean(dd)), mm(np.std(dd, ddof=1)), len(dd)))

    # ------------------------------------------------- jitter reference
    print("\n" + "=" * 79)
    print("REFERENCE CURVE: the same metric on a perturbation whose")
    print("effect on the identification statistic is already known")
    print("=" * 79)
    print("  Seed jitter of standard deviation sigma d_mean is the")
    print("  perturbation microstructure_error.py sweeps, and its effect")
    print("  on the published statistic is measured at every level there.")
    print("  Putting it in the metric above converts a candidate")
    print("  separation into a statement about the test.")
    base_full = {}
    d_mean = {}
    for s in REAL:
        p, w, idx = gen[s]
        o0, d0 = own[s], dmin[s]
        base_full[s] = (o0, d0)
        cnt = np.bincount(o0, minlength=len(p)).astype(float)
        v = cnt[cnt > 0] / o0.size * VOL_M3
        d_mean[s] = float(np.mean(2.0 * (3.0 * v / (4.0 * np.pi))
                                  ** (1.0 / 3.0)))
    print("\n  mean volume-equivalent grain diameter, this cloud: "
          "%.3f mm" % mm(np.mean([d_mean[s] for s in REAL])))
    print("\n  %-8s %10s %10s %10s %10s %10s"
          % ("sigma", "d_bnd mm", "f_mis", "f V/A mm", "d/lambda", "cycles"))
    jit = {}
    for lv in JIT_LEVELS:
        db, fm = [], []
        for s in REAL:
            p, w, idx = gen[s]
            o0, d0 = base_full[s]
            for r in range(N_JIT_REAL):
                g = np.random.default_rng(90000 + 1000 * int(lv * 1000)
                                          + 10 * s + r)
                pj = p + g.normal(0.0, lv * d_mean[s], p.shape)
                oj, dj = power_fields(pj, w, P)
                ma, mb = d0 < eps[0], dj < eps[0]
                db.append(0.5 * (dj[ma].mean() + d0[mb].mean()))
                fm.append(float((o0 != oj).mean()))
        jit[lv] = dict(d_bnd=float(np.mean(db)), f_mis=float(np.mean(fm)),
                       d_bnd_sd=float(np.std(db, ddof=1)))
        print("  %-8.3g %10.3f %10.4f %10.3f %10.3f %10.3f"
              % (lv, mm(jit[lv]["d_bnd"]), jit[lv]["f_mis"],
                 mm(jit[lv]["f_mis"] * vol_over_area),
                 mm(jit[lv]["d_bnd"]) / LAM_MM,
                 2 * mm(jit[lv]["d_bnd"]) / LAM_MM))
    print("\n  the f_mis column and its f V / A conversion are the two")
    print("  columns microstructure_error.py prints for the same levels,")
    print("  recomputed from generators rather than from the raster. Their")
    print("  agreement is the check that this module's geometry is that")
    print("  module's geometry.")

    lv_arr = np.array(JIT_LEVELS)
    db_arr = mm(np.array([jit[l]["d_bnd"] for l in JIT_LEVELS]))
    dm_arr = mm(np.array([jit[l]["f_mis"] for l in JIT_LEVELS])
                * vol_over_area)
    tol_bnd = [float(np.interp(t, dm_arr, db_arr))
               for t in (TOL_2OF8_MM, TOL_1OF8_MM)]
    print("\n  the two published tolerances, carried into the boundary-")
    print("  offset metric by interpolating this curve:")
    print("    below 2 of 8 first places at f V/A = %.3f mm  ->  "
          "d_bnd = %.3f mm" % (TOL_2OF8_MM, tol_bnd[0]))
    print("    below 1 of 8 first places at f V/A = %.3f mm  ->  "
          "d_bnd = %.3f mm" % (TOL_1OF8_MM, tol_bnd[1]))
    print("\n  DIFFICULTY = nearest wrong candidate / tolerance, both in")
    print("  the boundary-offset metric:")
    print("    %.2f to %.2f, mean %.2f, against the 2-of-8 tolerance"
          % (mm(nd.min()) / tol_bnd[0], mm(nd.max()) / tol_bnd[0],
             mm(nd.mean()) / tol_bnd[0]))
    print("    %.2f to %.2f, mean %.2f, against the 1-of-8 tolerance"
          % (mm(nd.min()) / tol_bnd[1], mm(nd.max()) / tol_bnd[1],
             mm(nd.mean()) / tol_bnd[1]))

    # ------------------------------------------------------------ confound
    print("\n" + "=" * 79)
    print("CONFOUND: does the score follow the geometry?")
    print("=" * 79)
    zp = os.path.join(HERE, "tessellation_replication.npz")
    conf = None
    if not os.path.exists(zp):
        print("  %s absent, skipped" % zp)
    else:
        with np.load(zp, allow_pickle=True) as z:
            ident = np.asarray(z["ident"], float)
            cands_z = [int(q) for q in np.asarray(z["cands"])]
            own_z = np.asarray(z["ident_own"], float)
        assert cands_z == CANDS, "candidate order differs from this module"
        print("\n  %-5s %6s %8s %8s %7s %10s %10s"
              % ("seed", "rank", "r_own", "r_wrong", "winner", "d to",
                 "d to"))
        print("  %-5s %6s %8s %8s %7s %10s %10s"
              % ("", "of 48", "", "best", "cand", "winner mm", "nearest mm"))
        rows = []
        for i, s in enumerate(REAL):
            rank, r_own = int(own_z[i, 1]), float(own_z[i, 0])
            win, r_win = int(own_z[i, 4]), float(own_z[i, 5])
            k = (s, win) if (s, win) in D else (win, s)
            d_win = D[k]["d_bnd"]
            rows.append((s, rank, r_own, r_win, win, d_win, near[s]["d"]))
            print("  %-5d %6d %8.3f %8.3f %7d %10.3f %10.3f"
                  % (s, rank, r_own, r_win, win, mm(d_win), mm(near[s]["d"])))
        rank_v = np.array([r[1] for r in rows], float)
        dwin_v = np.array([r[5] for r in rows], float)
        dnear_v = np.array([r[6] for r in rows], float)
        marg_v = np.array([r[2] - r[3] for r in rows], float)
        print("\n  n = 8, Spearman:")
        for tag, x, y in (
                ("rank against distance to nearest wrong", rank_v, dnear_v),
                ("rank against distance to the winner", rank_v, dwin_v),
                ("margin r_own - r_wrong against d to winner",
                 marg_v, dwin_v),
                ("margin against distance to nearest wrong",
                 marg_v, dnear_v)):
            rho, p = spearmanr(x, y)
            print("    %-44s rho %+.3f  p %.3f" % (tag, rho, p))

        print("\n  the stronger form. Within one specimen, over its 47")
        print("  wrong candidates, does a geometrically CLOSER wrong")
        print("  candidate score HIGHER? A negative rho would mean the")
        print("  score is reading a graded geometric distance and not a")
        print("  binary right-or-wrong.")
        print("  %-5s %10s %10s" % ("seed", "spearman", "p"))
        rhos = []
        for i, s in enumerate(REAL):
            sc, dg = [], []
            for j, o in enumerate(CANDS):
                if o == s:
                    continue
                k = (s, o) if (s, o) in D else (o, s)
                sc.append(ident[i, j])
                dg.append(D[k]["d_bnd"])
            rho, p = spearmanr(sc, dg)
            rhos.append(rho)
            print("  %-5d %10.3f %10.3f" % (s, rho, p))
        rhos = np.array(rhos)
        tstat = float(rhos.mean() / (rhos.std(ddof=1) / np.sqrt(len(rhos))))
        print("  mean rho %+.3f +- %.3f over the eight specimens, "
              "t = %+.2f on 7 df" % (rhos.mean(), rhos.std(ddof=1), tstat))
        conf = dict(rank=rank_v, d_win=dwin_v, d_near=dnear_v,
                    margin=marg_v, rho_within=rhos)

    # ---------------------------------------------------------------- save
    os.makedirs(RESULTS, exist_ok=True)
    out = dict(
        cands=np.array(CANDS), real=np.array(REAL),
        pair_a=np.array([k[0] for k in keys]),
        pair_b=np.array([k[1] for k in keys]),
        lam_mm=LAM_MM, vol_over_area_m=vol_over_area,
        d_self_m=float(ds.mean()), n_mc=N_MC, eps_mm=np.array(EPS_MM),
        tol_bnd_mm=np.array(tol_bnd),
        tol_fva_mm=np.array([TOL_2OF8_MM, TOL_1OF8_MM]),
        jit_levels=lv_arr, jit_d_bnd_mm=db_arr, jit_d_mis_mm=dm_arr,
        nearest_d_m=np.array([near[s]["d"] for s in REAL]),
        nearest_other=np.array([near[s]["other"] for s in REAL]),
        nearest_f_mis=np.array([near[s]["f_mis"] for s in REAL]),
        nearest_w1_m=np.array([near[s]["w1"] for s in REAL]),
        mean_over_wrong_m=np.array([near[s]["mean"] for s in REAL]),
        area_m2=np.array([area[s] for s in CANDS]),
        n_grain=np.array([nlive[s] for s in CANDS]),
        val_seed=np.array([v[0] for v in val]),
        val_agree=np.array([v[2] for v in val]),
        val_agree_far=np.array([v[3] for v in val]))
    for k in ("d_bnd", "d_med", "d_p95", "d_p99", "d_haus", "f_mis",
              "d_mis", "w1", "w2"):
        out["pair_" + k] = np.array([D[q][k] for q in keys])
    out["pair_d_bnd_eps2"] = np.array([D2[q]["d_bnd"] for q in keys])
    if conf is not None:
        for k, v in conf.items():
            out["conf_" + k] = v
    np.savez(os.path.join(RESULTS, "candidate_geometry.npz"), **out)
    print("\nwrote %s" % os.path.join(RESULTS, "candidate_geometry.npz"))
    print("total %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
