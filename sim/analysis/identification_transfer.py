"""Does the identification result transfer? An adjudication, not a rerun.

Two tests reported on the Sec. 5.2 identification claim: a perturbation
sweep (analysis/microstructure_error.py) and a noise sweep
(analysis/coda_noise_floor.py). Both returned claim-weakened. This module
does three things neither of them could do for itself.

FIRST, it re-derives every load-bearing number from the archives rather
than from the reporting code that wrote them. The counts, the binomial
tail probabilities and the level-to-millimetre conversion are all
recomputed here from the raw per-run scores in microstructure_error.npz,
so a bug in report_family or report_thresholds would show up as a
disagreement rather than propagate.

SECOND, it settles the one number the whole adjudication turns on. The
perturbation sweep placed the binding tolerance at a boundary
displacement of 0.42 mm, but it swept seed jitter at 0.02, 0.03 and then
0.05 with three realisations each, so the interval in which identifica-
tion collapses was crossed in one step and the "nothing identifies"
figure of 0.575 mm is an interpolation across that step, not a measure-
ment. Twenty-four binary outcomes also put a standard error of about 0.7
of 8 on a count of 2.00 of 8, which is the same size as the distance from
2 of 8 to chance. This module fills the gap with four levels and lifts
every level in the interval to six realisations, so the threshold is
measured, bracketed and given a bootstrap interval.

THIRD, it checks the millimetre conversion by a route that does not use
it. microstructure_error converts a changed-volume fraction into a mean
normal boundary displacement by delta = f V / A, which is exact to first
order but leans entirely on the total facet area A. Here the same
displacement is measured directly: points are drawn uniformly in the thin
shell around the unperturbed grain-boundary surface, which samples that
surface area-weighted, and the distance from each to the PERTURBED
boundary is evaluated in closed form. The mean of those distances is the
area-weighted mean normal displacement with no area estimate in it. A is
also re-estimated from the voxel face count with the Cauchy-Crofton
factor 2/3 for an isotropic surface on a cubic grid, which is an
independent estimator of the same quantity.

Everything is CPU. The tessellations come from out/tesscache through
tessellation_replication, the seeds and weights from the replay cache,
and no solver runs. Scoring is imported from microstructure_error so
that the refinement levels are scored by the identical code path as the
levels they interleave with; a private reimplementation would have made
the new levels incomparable with the old ones, which is the opposite of
what the refinement is for.

  python identification_transfer.py --audit      archives only, seconds
  python identification_transfer.py --geometry   the mm check, ~3 min
  python identification_transfer.py --refine     the sweep, ~10 min
  python identification_transfer.py --report     reprint from the archive
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

import microstructure_error as ME                        # noqa: E402
import tessellation_replication as TR                    # noqa: E402
from scipy.stats import binom                            # noqa: E402

TR.CENTRING = ME.CENTRING
RESULTS = ME.RESULTS
ARCH = os.path.join(RESULTS, "microstructure_error.npz")
NOISE = os.path.join(HERE, "coda_noise_floor.npz")
OUT = os.path.join(RESULTS, "identification_transfer.npz")
PARTIAL = os.path.join(RESULTS, "identification_transfer_partial.npz")

# The refinement. Two of the six levels already carry three realisations
# in the parent archive and are reused rather than rescored, so the new
# and old realisations of the same level differ only in their stream
# index and pool as one sample.
LV_REFINE = (0.02, 0.025, 0.03, 0.035, 0.04, 0.05)
N_REAL = 6
N_WORKER = 6                     # a GPU job holds cores; leave headroom
CKPT = 48

N_MC_SHELL = 600000              # points for the direct displacement
EPS_SHELL = 0.15e-3              # shell half-thickness, metres
GEOM_SEEDS = (11, 7, 17)         # three specimens is enough for a check
GEOM_LEVELS = (0.01, 0.02, 0.03, 0.05, 0.10)


# ─────────────────────────── archive arithmetic ───────────────────────
def load_archive():
    z = np.load(ARCH, allow_pickle=True)
    return {k: z[k] for k in z.files}


def counts(z, kind, level, variant=0, tol=1e-9):
    """Per-realisation count of tessellations ranking first, from raw scores.

    Rebuilt from `scores` and `foreign` rather than read from any summary,
    so this is an independent path to the numbers microstructure_error
    prints. Returns the per-(realisation) counts out of eight.
    """
    sel = (z["kinds"] == kind) & (np.abs(z["levels"] - level) < tol)
    names, sw = list(z["pert_sweep"][sel]), list(z["sweeps"])
    sc, ir = z["scores"][sel][:, variant], z["ireal"][sel]
    out = {}
    for q in range(len(sc)):
        i = sw.index(str(names[q]))
        first = int(np.sum(np.asarray(z["foreign"][i, variant]) > sc[q]) == 0)
        out.setdefault(int(ir[q]), []).append(first)
    return np.array([np.sum(out[k]) for k in sorted(out)], float)


def p_first(k, n=8, m=48):
    """Binomial tail for k of n specimens ranking first of m candidates."""
    return float(binom.sf(k - 1, n, 1.0 / m))


def audit(z):
    """Every load-bearing number of the two reports, recomputed here."""
    print("=" * 79)
    print("ARCHIVE AUDIT. numbers rebuilt from raw scores, not from the")
    print("code that printed them")
    print("=" * 79)
    ok = z["replay_ok"]
    print("  replay of the production build exact for %d of %d specimens "
          "on all three checks" % (int(ok.all(1).sum()), len(ok)))
    for q, v in enumerate(z["variants"]):
        k = int(np.sum(z["base_rank"][:, q] == 1))
        r = z["base_r"][:, q]
        print("  baseline %-5s %d of 8 first of %d   p = %.3g   "
              "mean r = %+.3f   sd %.3f"
              % (v, k, len(z["cands"]), p_first(k, 8, len(z["cands"])),
                 r.mean(), r.std(ddof=1)))
    print("  the published statistic is `pub`. The manuscript quotes four")
    print("  of eight and p = 1.2e-5.")

    print("\n  binomial tail of the count, n = 8, 48 candidates")
    print("  %-8s %12s" % ("k of 8", "p"))
    for k in (1, 2, 3, 4, 5):
        print("  %-8d %12.3g" % (k, p_first(k)))
    print("  a count of 2 of 8 is still significant at the five per cent")
    print("  level, so a threshold quoted as 'below 2 of 8' is the last")
    print("  significant point and not the first insignificant one.")

    print("\n  JITTER, published statistic, per realisation")
    print("  %-8s %10s %8s %8s %26s" % ("level", "counts", "mean", "sem",
                                        "frac changed"))
    for lv in ME.LV_JITTER:
        c = counts(z, "jitter", lv)
        f = z["frac_changed"][(z["kinds"] == "jitter")
                              & (np.abs(z["levels"] - lv) < 1e-9)]
        sem = c.std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else 0.0
        print("  %-8.4g %10s %8.2f %8.2f %26.4f"
              % (lv, "/".join("%d" % x for x in c), c.mean(), sem, f.mean()))
    print("  three realisations per level in the parent sweep. The step")
    print("  from 0.03 to 0.05 is where the collapse happens and it is one")
    print("  step wide, which is what --refine exists to fix.")


def geometry_from_archive(z):
    """The level-to-millimetre conversion, recomputed from the archive."""
    vol = np.pi * (ME.DIA / 2.0) ** 2 * ME.THK
    rng = np.random.default_rng(4)
    areas, raster, dmean = {}, {}, {}
    print("\n" + "=" * 79)
    print("BOUNDARY AREA, three estimators of the same surface")
    print("=" * 79)
    print("  %-6s %11s %11s %11s %11s %9s" % ("seed", "MC 0.15mm", "MC 0.30mm",
                                              "raster", "raster*2/3",
                                              "d_mean mm"))
    for s in z["owns"]:
        s = int(s)
        rep = ME.replay_build(s)
        a = ME.facet_area_mc(rep["pts"], rep["weights"], rng)
        lab, axes, _, h = TR.tessellation(s)
        ar = ME.facet_area_raster(lab, h)
        areas[s], raster[s] = float(np.mean(a)), ar
        dmean[s] = ME.mean_diameter(lab, len(axes), h)
        print("  %-6d %11.5g %11.5g %11.5g %11.5g %9.3f"
              % (s, a[0] * 1e4, a[1] * 1e4, ar * 1e4, ar * 2 / 3 * 1e4,
                 dmean[s] * 1e3))
    am, rm = np.mean(list(areas.values())), np.mean(list(raster.values()))
    print("  areas in cm^2. The voxel face count of an isotropic surface")
    print("  on a cubic grid overestimates its area by 3/2 (Cauchy), so")
    print("  raster*2/3 is an independent estimator of the Monte Carlo")
    print("  column: mean MC %.1f cm^2, mean raster*2/3 %.1f cm^2, ratio "
          "%.4f." % (am * 1e4, rm * 2 / 3 * 1e4, rm * 2 / 3 / am))
    print("  disc volume %.2f cm^3, mean grain diameter %.2f mm, "
          "lambda %.3f mm" % (vol * 1e6, np.mean(list(dmean.values())) * 1e3,
                              ME.C2.LAM * 1e3))
    return areas, vol, float(np.mean(list(dmean.values())))


# ──────────────────── direct displacement measurement ─────────────────
def boundary_distance(p, pts, w):
    """Exact distance from interior points to their own power cell's face.

    For power cells the separating surface of cells i and j is the plane
    (pow_j - pow_i) = 0, and the Euclidean distance from an interior point
    of cell i to that plane is (pow_j - pow_i) / (2 |s_j - s_i|). The
    distance to the cell boundary is the minimum over j, which is the same
    identity facet_area_mc uses; here it is evaluated for its own sake
    rather than histogrammed.
    """
    dsep = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dsep, np.inf)
    out = np.empty(len(p))
    rows_all = np.arange(20000)
    for lo in range(0, len(p), 20000):
        q = p[lo:lo + 20000]
        d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1) - w[None, :]
        own = d2.argmin(1)
        rows = rows_all[:len(q)]
        d = (d2 - d2[rows, own][:, None]) / (2.0 * dsep[own])
        d[rows, own] = np.inf
        out[lo:lo + 20000] = d.min(1)
    return out


def direct_displacement(seed, level, rng):
    """Mean normal displacement of the boundary, measured without an area.

    Points uniform in the disc are kept when they lie within EPS_SHELL of
    the unperturbed grain-boundary surface. Uniform in a thin shell is
    area-weighted on the surface, so the retained points sample the
    boundary the way the delta = f V / A identity weights it. Each point's
    distance to the PERTURBED boundary is then evaluated in closed form,
    and the mean of those distances is the area-weighted mean normal
    displacement. The estimate is biased high by at most EPS_SHELL and the
    reverse direction is reported as well, so the bias is visible.
    """
    rep = ME.replay_build(seed)
    pts0, w0 = rep["pts"], rep["weights"]
    lab0, axes0, seeds0, h = TR.tessellation(seed)
    d_mean = ME.mean_diameter(lab0, len(axes0), h)
    pts1 = pts0 + rng.normal(0.0, level * d_mean, pts0.shape)

    R = ME.DIA / 2.0
    p = rng.uniform(-1, 1, (int(N_MC_SHELL * 1.4), 3))
    p = p[(p[:, 0] ** 2 + p[:, 1] ** 2) <= 1.0][:N_MC_SHELL]
    p[:, :2] *= R
    p[:, 2] *= ME.THK / 2.0

    d0 = boundary_distance(p, pts0, w0)
    d1 = boundary_distance(p, pts1, w0)
    on0, on1 = d0 < EPS_SHELL, d1 < EPS_SHELL
    fwd = float(np.mean(d1[on0])) if on0.sum() else np.nan
    rev = float(np.mean(d0[on1])) if on1.sum() else np.nan
    return fwd, rev, int(on0.sum()), int(on1.sum())


def geometry_check():
    """Compare the swept-volume conversion with the direct measurement."""
    z = load_archive()
    areas, vol, d_mean = geometry_from_archive(z)
    print("\n" + "=" * 79)
    print("THE MILLIMETRE CONVERSION, CHECKED WITHOUT AN AREA ESTIMATE")
    print("=" * 79)
    print("  %-6s %-7s %11s %11s %11s %11s" %
          ("seed", "level", "f V / A mm", "direct mm", "reverse mm", "n shell"))
    rows = []
    for s in GEOM_SEEDS:
        for lv in GEOM_LEVELS:
            rng = np.random.default_rng([int(s), 909, int(lv * 1e6)])
            fwd, rev, n0, n1 = direct_displacement(int(s), lv, rng)
            sel = ((z["kinds"] == "jitter")
                   & (np.abs(z["levels"] - lv) < 1e-9)
                   & (z["pert_sweep"] == z["sweeps"][list(z["owns"]).index(s)]))
            f = float(np.mean(z["frac_changed"][sel])) if sel.sum() else np.nan
            swept = f * vol / areas[int(s)] * 1e3
            rows.append((s, lv, swept, fwd * 1e3, rev * 1e3))
            print("  %-6d %-7.4g %11.4f %11.4f %11.4f %11d"
                  % (s, lv, swept, fwd * 1e3, rev * 1e3, n0))
    a = np.array([[r[2], r[3]] for r in rows if np.isfinite(r[2])])
    print("  the two columns measure the same displacement by routes that")
    print("  share no arithmetic. ratio direct / swept-volume: mean %.3f, "
          "range %.3f to %.3f"
          % (np.mean(a[:, 1] / a[:, 0]), np.min(a[:, 1] / a[:, 0]),
             np.max(a[:, 1] / a[:, 0])))
    print("  the direct estimate is biased HIGH by up to the shell half-")
    print("  thickness of %.2f mm because a shell point is that far off"
          % (EPS_SHELL * 1e3))
    print("  the surface it stands for.")
    np.savez(os.path.join(RESULTS, "identification_transfer_geom.npz"),
             rows=np.array(rows), areas=np.array(list(areas.values())),
             vol=vol, d_mean=d_mean)


# ──────────────────────────── the refinement ──────────────────────────
def refine_jobs(z):
    """New (level, realisation) cells only; the parent archive is reused."""
    meas = {n: ME.measured(n) for n, _ in ME.SWEEPS}
    have = set()
    for i in range(len(z["kinds"])):
        if str(z["kinds"][i]) == "jitter":
            have.add((str(z["pert_sweep"][i]), round(float(z["levels"][i]), 6),
                      int(z["ireal"][i])))
    jobs = []
    for lv in LV_REFINE:
        for ir in range(N_REAL):
            for name, seed in ME.SWEEPS:
                if (name, round(lv, 6), ir) in have:
                    continue
                E, c_az = meas[name]
                jobs.append((name, seed, "jitter", lv, ir, E, c_az))
    return jobs


def save_partial(res):
    np.savez(PARTIAL,
             name=np.array([r["name"] for r in res]),
             seed=np.array([r["seed"] for r in res]),
             level=np.array([r["level"] for r in res]),
             ireal=np.array([r["ireal"] for r in res]),
             scores=np.array([r["scores"] for r in res]),
             frac_changed=np.array([r["frac_changed"] for r in res]),
             n_grain=np.array([r["n_grain"] for r in res]))


def load_partial():
    if not os.path.exists(PARTIAL):
        return {}
    out = {}
    with np.load(PARTIAL, allow_pickle=True) as z:
        for i in range(len(z["level"])):
            k = (str(z["name"][i]), round(float(z["level"][i]), 6),
                 int(z["ireal"][i]))
            out[k] = dict(name=k[0], seed=int(z["seed"][i]), kind="jitter",
                          level=k[1], ireal=k[2],
                          scores=tuple(z["scores"][i]),
                          frac_changed=float(z["frac_changed"][i]),
                          n_grain=int(z["n_grain"][i]))
    return out


def refine():
    t0 = time.time()
    z = load_archive()
    done = load_partial()
    jobs = [j for j in refine_jobs(z)
            if (j[0], round(j[3], 6), j[4]) not in done]
    print("refinement: %d new cells to score, %d in the checkpoint"
          % (len(jobs), len(done)))
    res = list(done.values())
    with Pool(N_WORKER) as pool:
        for q, r in enumerate(pool.imap_unordered(ME._pert_job, jobs,
                                                  chunksize=1)):
            r.pop("diag", None)
            res.append(r)
            if (q + 1) % CKPT == 0:
                save_partial(res)
                print("  %d of %d  %.0f s" % (q + 1, len(jobs),
                                              time.time() - t0), flush=True)
    save_partial(res)
    print("  scored in %.0f s" % (time.time() - t0))
    merge_and_save(z, res)


def merge_and_save(z, new):
    """Pool the new cells with the parent archive's jitter cells."""
    sw = list(z["sweeps"])
    rows = []
    for i in range(len(z["kinds"])):
        if str(z["kinds"][i]) != "jitter":
            continue
        rows.append((str(z["pert_sweep"][i]), float(z["levels"][i]),
                     int(z["ireal"][i]), float(z["scores"][i, 0]),
                     float(z["scores"][i, 1]), float(z["scores"][i, 2]),
                     float(z["frac_changed"][i]), 0))
    for r in new:
        rows.append((r["name"], r["level"], r["ireal"], r["scores"][0],
                     r["scores"][1], r["scores"][2], r["frac_changed"], 1))
    name = np.array([r[0] for r in rows])
    lev = np.array([r[1] for r in rows])
    ire = np.array([r[2] for r in rows])
    sc = np.array([[r[3], r[4], r[5]] for r in rows])
    fc = np.array([r[6] for r in rows])
    src = np.array([r[7] for r in rows])
    isw = np.array([sw.index(n) for n in name])
    first = np.zeros((len(rows), 3), int)
    for q in range(len(rows)):
        for v in range(3):
            first[q, v] = int(np.sum(
                np.asarray(z["foreign"][isw[q], v]) > sc[q, v]) == 0)
    np.savez(OUT, name=name, level=lev, ireal=ire, scores=sc, frac_changed=fc,
             source=src, isweep=isw, first=first, sweeps=z["sweeps"],
             owns=z["owns"], foreign=z["foreign"], base_rank=z["base_rank"],
             base_r=z["base_r"], cands=z["cands"])
    print("wrote %s" % OUT)


# ─────────────────── what a thin section actually gives ───────────────
# The perturbation sweep answers "how wrong may the microstructure be".
# It does not answer "how wrong would a real measurement be", because no
# perturbation family is a section. These variants are: the true
# tessellation; the true grain set with every seed moved to its own
# volume centroid, which is where a Laguerre model FITTED to a perfect
# three-dimensional grain map would put it; and a tessellation
# reconstructed from n equally spaced parallel sections through the disc,
# which is what serial sectioning supplies. n = 1 is an optical thin
# section. The reconstruction is given every advantage: the sections are
# noiseless, grain boundaries in them are exact, the production weight
# scale is handed to it, and each recovered grain keeps its true c-axis.
SECTION_VARIANTS = ("true", "c3d", "sec1", "sec2", "sec3", "sec5", "sec9")


def _grid_coords(lab, h):
    nx, nz = lab.shape[0], lab.shape[2]
    x = (np.arange(nx) - (nx - 1) / 2.0) * h
    z = (np.arange(nz) - (nz - 1) / 2.0) * h
    return x, z


def centroid_tess(seed):
    """Every grain kept, each seed moved to its own volume centroid.

    The upper bound on any method that recovers grains rather than
    Laguerre seeds. Nothing is missing, nothing is merged, the grain
    volumes are exact; the only error is that a fitted seed sits at the
    centroid of its cell and a Laguerre seed does not.
    """
    lab, axes, seeds0, h = TR.tessellation(seed)
    x, z = _grid_coords(lab, h)
    X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
    ins = lab >= 0
    li = lab[ins].astype(np.int64)
    n = len(axes)
    cnt = np.bincount(li, minlength=n).astype(float)
    c = np.stack([np.bincount(li, weights=q[ins], minlength=n) / cnt
                  for q in (X, Y, Z)], axis=1)
    v = cnt * h ** 3
    rad = (v / v.mean()) ** (1.0 / 3.0)
    rep = ME.replay_build(seed)
    w = float(rep["scale"]) * float(rep["base"]) * (rad ** 2 - 1.0)
    return c, w, axes, seeds0, h, lab


def section_tess(seed, n_sec):
    """Reconstruct a Laguerre model from n equally spaced plane sections.

    Each grain that any section cuts contributes the area centroid of its
    intersections, pooled over sections, as its estimated seed, and a
    Cavalieri volume estimate as its size. A grain that no section cuts is
    absent, exactly as it would be absent from the practitioner's map.
    The estimated radii are normalised to unit mean over the grains that
    were seen, which is what a practitioner would do and is the choice
    that flatters the reconstruction.
    """
    lab, axes, seeds0, h = TR.tessellation(seed)
    x, z = _grid_coords(lab, h)
    nz = lab.shape[2]
    idx = np.unique(np.round((np.arange(n_sec) + 0.5) / n_sec * nz
                             - 0.5).astype(int))
    n = len(axes)
    sx = np.zeros(n)
    sy = np.zeros(n)
    sz = np.zeros(n)
    cnt = np.zeros(n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    for iz in idx:
        pl = lab[:, :, iz]
        m = pl >= 0
        li = pl[m].astype(np.int64)
        cnt += np.bincount(li, minlength=n)
        sx += np.bincount(li, weights=X[m], minlength=n)
        sy += np.bincount(li, weights=Y[m], minlength=n)
        sz += np.bincount(li, weights=np.full(int(m.sum()), z[iz]),
                          minlength=n)
    seen = cnt > 0
    c = np.stack([sx[seen] / cnt[seen], sy[seen] / cnt[seen],
                  sz[seen] / cnt[seen]], axis=1)
    v = cnt[seen] * h ** 2 * (ME.THK / len(idx))       # Cavalieri
    rad = (v / v.mean()) ** (1.0 / 3.0)
    rep = ME.replay_build(seed)
    w = float(rep["scale"]) * float(rep["base"]) * (rad ** 2 - 1.0)
    return c, w, axes[seen], seeds0[seen], h, lab, int(seen.sum()), len(idx)


def _section_job(args):
    name, seed, variant, E, c_az = args
    if variant == "true":
        tess = TR.tessellation(seed)
        lab0 = tess[0]
        err = np.zeros(0)
        err_xy = np.zeros(0)
        n_keep, n_sec = len(tess[1]), 0
    else:
        if variant == "c3d":
            c, w, ax, s0, h, lab0 = centroid_tess(seed)
            n_keep, n_sec = len(ax), 0
        else:
            c, w, ax, s0, h, lab0, n_keep, n_sec = section_tess(
                seed, int(variant[3:]))
        err = np.linalg.norm(c - s0, axis=1)
        err_xy = np.linalg.norm(c[:, :2] - s0[:, :2], axis=1)
        lab, keep = ME.relabel(c, w, h)
        tess = (lab, ax[keep], c[keep], h)
    sc = ME.score_all(tess, E, c_az)
    ins = lab0 >= 0
    changed = float(np.mean(tess[0][ins] != lab0[ins])) if variant != "true" \
        else 0.0
    return dict(name=name, seed=seed, variant=variant, scores=sc,
                n_keep=n_keep, n_sec=n_sec, changed=changed,
                err_mm=float(err.mean() * 1e3) if len(err) else 0.0,
                err_xy_mm=float(err_xy.mean() * 1e3) if len(err) else 0.0,
                err_med_mm=float(np.median(err) * 1e3) if len(err) else 0.0)


def section_test():
    """Score the reconstructions a real microstructure measurement gives."""
    z = load_archive()
    meas = {n: ME.measured(n) for n, _ in ME.SWEEPS}
    jobs = [(n, s, v, meas[n][0], meas[n][1])
            for v in SECTION_VARIANTS for n, s in ME.SWEEPS]
    t0 = time.time()
    with Pool(N_WORKER) as pool:
        res = pool.map(_section_job, jobs)
    print("scored %d reconstructions in %.0f s" % (len(res), time.time() - t0))

    sw = list(z["sweeps"])
    print("\n" + "=" * 79)
    print("WHAT A SECTIONED SPECIMEN ACTUALLY SUPPLIES")
    print("=" * 79)
    print("  %-6s %7s %8s %9s %9s %9s %8s %8s %8s"
          % ("variant", "grains", "changed", "seed err", "in-plane",
             "median", "r_pub", "rank", "first/8"))
    rows = []
    for v in SECTION_VARIANTS:
        sel = [r for r in res if r["variant"] == v]
        rk, r1 = [], 0
        for r in sel:
            i = sw.index(r["name"])
            k = ME.rank_of(r["scores"][0], z["foreign"][i, 0])
            rk.append(k)
            r1 += int(k == 1)
        print("  %-6s %7.1f %8.3f %9.2f %9.2f %9.2f %8.3f %8.1f %8d"
              % (v, np.mean([r["n_keep"] for r in sel]),
                 np.mean([r["changed"] for r in sel]),
                 np.mean([r["err_mm"] for r in sel]),
                 np.mean([r["err_xy_mm"] for r in sel]),
                 np.mean([r["err_med_mm"] for r in sel]),
                 np.mean([r["scores"][0] for r in sel]),
                 np.mean(rk), r1))
        rows.append((v, np.mean([r["n_keep"] for r in sel]),
                     np.mean([r["changed"] for r in sel]),
                     np.mean([r["err_mm"] for r in sel]),
                     np.mean([r["err_xy_mm"] for r in sel]),
                     np.mean([r["scores"][0] for r in sel]),
                     np.mean(rk), r1,
                     p_first(r1, 8, len(z["cands"]))))
        print("        p for %d of 8 first of 48 = %.3g"
              % (r1, p_first(r1, 8, len(z["cands"]))))
    print("  seed err is the mean distance in mm from the reconstructed")
    print("  Laguerre seed to the production seed it stands for; in-plane")
    print("  is the same distance with the section-normal component")
    print("  dropped, which is the part a section could in principle fix by")
    print("  being thinner. changed is the fraction of the disc that")
    print("  changes grain identity, the same quantity the perturbation")
    print("  sweep reports. secN is N equally spaced parallel sections")
    print("  through a %.0f mm disc, so sec1 is one optical thin section."
          % (ME.THK * 1e3))
    np.savez(os.path.join(RESULTS, "identification_transfer_section.npz"),
             variant=np.array([r[0] for r in rows]),
             table=np.array([[float(q) for q in r[1:]] for r in rows]),
             per_run_variant=np.array([r["variant"] for r in res]),
             per_run_name=np.array([r["name"] for r in res]),
             per_run_scores=np.array([r["scores"] for r in res]),
             per_run_err=np.array([r["err_mm"] for r in res]),
             per_run_changed=np.array([r["changed"] for r in res]))


# ─────────────────────────────── reporting ────────────────────────────
def curve(w, variant=0):
    """Per-level count of eight, its standard error, and the mm abscissa."""
    z = load_archive()
    vol = np.pi * (ME.DIA / 2.0) ** 2 * ME.THK
    rng = np.random.default_rng(4)
    areas = {}
    for s in z["owns"]:
        rep = ME.replay_build(int(s))
        areas[int(s)] = float(np.mean(ME.facet_area_mc(rep["pts"],
                                                       rep["weights"], rng)))
    owns = {str(n): int(s) for n, s in zip(z["sweeps"], z["owns"])}
    out = []
    for lv in sorted(set(w["level"])):
        sel = np.abs(w["level"] - lv) < 1e-9
        f = w["first"][sel, variant]
        ir = w["ireal"][sel]
        per = np.array([f[ir == k].sum() for k in sorted(set(ir))], float)
        mm = np.mean([w["frac_changed"][sel][q] * vol
                      / areas[owns[str(w["name"][sel][q])]]
                      for q in range(int(sel.sum()))]) * 1e3
        out.append(dict(level=float(lv), mm=float(mm), n=int(sel.sum()),
                        k=int(f.sum()), n_real=len(per),
                        mean=float(per.mean()),
                        sem=float(per.std(ddof=1) / np.sqrt(len(per)))
                        if len(per) > 1 else 0.0,
                        p_hat=float(f.mean())))
    return out


def cross_mm(rows, floor, key="mean"):
    """Displacement at which the count curve crosses a floor, interpolated."""
    for q in range(1, len(rows)):
        if rows[q][key] < floor <= rows[q - 1][key]:
            x0, y0 = rows[q - 1]["mm"], rows[q - 1][key]
            x1, y1 = rows[q]["mm"], rows[q][key]
            return x0 + (y0 - floor) * (x1 - x0) / (y0 - y1)
    return None


def boot_cross(w, floor, variant=0, n_boot=2000, seed=0):
    """Bootstrap interval on the crossing, resampling specimens.

    The eight specimens are the independent unit, not the 48 candidates
    and not the realisations, so the resample is over specimens with all
    of a specimen's realisations moving together.
    """
    rng = np.random.default_rng(seed)
    z = load_archive()
    vol = np.pi * (ME.DIA / 2.0) ** 2 * ME.THK
    rg = np.random.default_rng(4)
    areas = {}
    for s in z["owns"]:
        rep = ME.replay_build(int(s))
        areas[int(s)] = float(np.mean(ME.facet_area_mc(rep["pts"],
                                                       rep["weights"], rg)))
    owns = {str(n): int(s) for n, s in zip(z["sweeps"], z["owns"])}
    levels = sorted(set(w["level"]))
    mm = {}
    for lv in levels:
        sel = np.abs(w["level"] - lv) < 1e-9
        mm[lv] = np.mean([w["frac_changed"][sel][q] * vol
                          / areas[owns[str(w["name"][sel][q])]]
                          for q in range(int(sel.sum()))]) * 1e3
    sw = list(z["sweeps"])
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(sw), len(sw))
        rows = []
        for lv in levels:
            tot = 0.0
            for i in pick:
                sel = ((np.abs(w["level"] - lv) < 1e-9)
                       & (w["name"] == sw[i]))
                tot += w["first"][sel, variant].mean()
            rows.append(dict(mm=mm[lv], mean=tot))
        c = cross_mm(rows, floor)
        if c is not None:
            out.append(c)
    return np.array(out)


def report():
    z = load_archive()
    audit(z)
    if not os.path.exists(OUT):
        print("\n(no refinement archive yet; run --refine)")
        return
    w = {k: v for k, v in np.load(OUT, allow_pickle=True).items()}
    print("\n" + "=" * 79)
    print("REFINED JITTER CURVE, published statistic, %d realisations per "
          "level" % N_REAL)
    print("=" * 79)
    rows = curve(w, 0)
    print("  %-8s %9s %8s %8s %8s %8s %8s"
          % ("level", "mm", "n_real", "count/8", "sem", "k/n", "p"))
    for r in rows:
        print("  %-8.4g %9.4f %8d %8.2f %8.2f %8s %8.3g"
              % (r["level"], r["mm"], r["n_real"], r["mean"], r["sem"],
                 "%d/%d" % (r["k"], r["n"]), p_first(int(round(r["mean"])))))
    for floor, lab in ((3.0, "three of eight"),
                       (2.5, "midway to the 2 of 8 line"),
                       (2.0, "half the baseline, 2 of 8"),
                       (1.5, "below 2 of 8 on rounding"),
                       (1.0, "one of eight"),
                       (8.0 / 48.0 * 8, "chance, 8/48 of 8")):
        c = cross_mm(rows, floor)
        b = boot_cross(w, floor)
        if c is None:
            print("  %-24s not crossed over the swept range" % lab)
            continue
        print("  %-24s at %.3f mm   bootstrap 90%% CI %.3f to %.3f mm "
              "(%d of %d resamples crossed)"
              % (lab, c, np.percentile(b, 5), np.percentile(b, 95),
                 len(b), 2000))
    print("  the bootstrap resamples the eight specimens, which are the")
    print("  independent unit. It is wide because eight is eight.")


if __name__ == "__main__":
    if "--refine" in sys.argv:
        refine()
    elif "--section" in sys.argv:
        section_test()
    elif "--geometry" in sys.argv:
        geometry_check()
    elif "--audit" in sys.argv:
        audit(load_archive())
    else:
        report()
