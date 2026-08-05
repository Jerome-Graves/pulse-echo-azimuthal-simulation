"""Edges 2 and 3, measured link by link on the stored data.

No solver, no CUDA, no DiskSpecimen.build. Reads only:
  out/tesscache/tess_s<seed>_p8_k-8.npz     labels, axes, seeds, weights
  sim/results/ensemble.npz                  coda levels, crossings, area
  sim/analysis/tessellation_replication.npz identification + field scores

For each of the eight girdle tessellations it forms, from the SAME
production-resolution label volume the solver saw:
  a1,a2,a3   volume-weighted orientation-tensor eigenvalues
  misor      boundary-area-weighted mean misorientation between the
             c-axes of face-adjacent grain pairs (degrees, folded to 90)
  dv_ray     boundary-area-weighted rms of |dv|/2vbar ACROSS those same
             adjacent pairs, for the beam direction, averaged over the
             30 common azimuths -- the contrast the ray actually sees
  dv_disc    the same quantity over ALL grain pairs in the disc,
             volume-weighted, i.e. the disc-averaged predictor
Then correlates each with the measured coda level and with the
identification score.
"""
import itertools
import os
import sys

import numpy as np
from scipy import stats

SIMROOT = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
sys.path.insert(0, os.path.join(SIMROOT, "sim"))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vendor"))))
import forward as FW                                     # noqa: E402

TESS = os.path.join(SIMROOT, "out", "tesscache")
ENS = os.path.join(SIMROOT, "sim", "results", "ensemble.npz")
REP = os.path.join(SIMROOT, "sim", "analysis",
                   "tessellation_replication.npz")

SEEDS = [11, 7, 17, 23, 41, 53, 71, 89]     # ensemble.npz order
AZ = np.arange(0, 360, 12)


def qp_velocity(cos_to_c):
    psi = np.arccos(np.clip(np.abs(cos_to_c), 0.0, 1.0))
    return np.interp(psi, FW._PSI, FW._VQP)


def beam_dirs(az_deg):
    a = np.radians(np.asarray(az_deg, float))
    return np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=-1)


def adjacency(lab):
    """Face-adjacent grain pairs and the number of shared voxel faces.

    Counts every 6-connected face between two DIFFERENT non-negative
    labels. The face count is proportional to the rasterised boundary
    area of that pair, so it is the natural weight.
    """
    pairs = {}
    for ax in range(3):
        lo = np.take(lab, np.arange(lab.shape[ax] - 1), axis=ax)
        hi = np.take(lab, np.arange(1, lab.shape[ax]), axis=ax)
        m = (lo != hi) & (lo >= 0) & (hi >= 0)
        a = lo[m].astype(np.int64)
        b = hi[m].astype(np.int64)
        i = np.minimum(a, b)
        j = np.maximum(a, b)
        key = i * 100000 + j
        u, c = np.unique(key, return_counts=True)
        for k, n in zip(u, c):
            pairs[int(k)] = pairs.get(int(k), 0) + int(n)
    i = np.array([k // 100000 for k in pairs], np.int64)
    j = np.array([k % 100000 for k in pairs], np.int64)
    w = np.array([pairs[k] for k in pairs], float)
    return i, j, w


def fold90(deg):
    d = np.abs(deg) % 180.0
    return np.minimum(d, 180.0 - d)


def per_seed(seed):
    p = os.path.join(TESS, "tess_s%d_p8_k-8.npz" % seed)
    with np.load(p) as z:
        lab = np.asarray(z["labels"])
        axes = np.asarray(z["axes"], float)
        h = float(z["h"])
    live = lab[lab >= 0]
    vol = np.bincount(live.ravel(), minlength=len(axes)).astype(float)
    w = vol / vol.sum()

    ev = np.sort(np.linalg.eigvalsh(
        np.einsum("g,gi,gj->ij", w, axes, axes)))[::-1]
    ev_u = np.sort(np.linalg.eigvalsh(
        np.einsum("gi,gj->ij", axes, axes) / len(axes)))[::-1]

    i, j, fw = adjacency(lab)
    cosm = np.abs(np.sum(axes[i] * axes[j], axis=1))
    mis = np.degrees(np.arccos(np.clip(cosm, 0.0, 1.0)))
    fwn = fw / fw.sum()

    n = beam_dirs(AZ)                       # (30, 3)
    v = qp_velocity(axes @ n.T)             # (ngrain, 30)
    dv = np.abs(v[i] - v[j])                # (npair, 30)
    vbar = float(w @ v.mean(axis=1))
    r_pair = dv / (2.0 * vbar)              # reflection coefficient
    # area-weighted mean square contrast per azimuth, then over azimuth
    ms_az = fwn @ (r_pair ** 2)             # (30,)
    dv_ray = float(np.sqrt(ms_az.mean()))

    # disc-averaged version: every grain pair, volume weighted
    ng = len(axes)
    ww = np.outer(w, w)
    np.fill_diagonal(ww, 0.0)
    ww /= ww.sum()
    tot = 0.0
    for q in range(v.shape[1]):
        d = np.abs(v[:, q][:, None] - v[:, q][None, :]) / (2.0 * vbar)
        tot += float((ww * d ** 2).sum())
    dv_disc = float(np.sqrt(tot / v.shape[1]))

    # misorientation over all pairs, volume weighted (the "raw angle"
    # analogue of dv_disc)
    cos_all = np.abs(axes @ axes.T)
    mis_all = np.degrees(np.arccos(np.clip(cos_all, 0.0, 1.0)))
    mis_disc = float((ww * mis_all).sum())

    return dict(seed=seed, ngrain=ng, h=h,
                a1=ev[0], a2=ev[1], a3=ev[2],
                a1u=ev_u[0], a3u=ev_u[2],
                npair=len(i), faces=float(fw.sum()),
                mis_mean=float(fwn @ mis),
                mis_rms=float(np.sqrt(fwn @ mis ** 2)),
                mis_disc=mis_disc,
                dv_ray=dv_ray, dv_disc=dv_disc,
                dv_ray_az=ms_az)


def rp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p = stats.pearsonr(x, y)
    rs, ps = stats.spearmanr(x, y)
    return float(r), float(p), float(rs), float(ps)


def main():
    with np.load(ENS) as z:
        assert list(z["girdle_seeds"]) == SEEDS
        level = z["girdle_levels"]
        cross = z["crossings"]
        area = z["area_cm2"]
        gaz = z["girdle_azimuth"]
    with np.load(REP) as z:
        names = list(z["names"])
        ident = z["ident_own"][:8, 0]
        identrank = z["ident_own"][:8, 1]
        field = z["field"][:8, 0]
        scal = z["scalar_full"][:8, 0]
    print("replication row order:", names[:8])

    rows = [per_seed(s) for s in SEEDS]

    print()
    print("PER-TESSELLATION TABLE (girdle, kappa = -8, ppw 8)")
    hd = ("seed ngr npair  a1     a3     a1_unw  mis_adj mis_disc "
          "dv_ray  dv_disc  cross  area   level   ident  rank  field")
    print(hd)
    for r, lv, cr, ar, idv, ik, fl in zip(rows, level, cross, area,
                                          ident, identrank, field):
        print("%4d %3d %5d %.4f %.4f %.4f  %6.2f %7.2f  %.5f %.5f "
              "%6.2f %6.1f %7.2f %+.4f %3d %+.4f"
              % (r["seed"], r["ngrain"], r["npair"], r["a1"], r["a3"],
                 r["a1u"], r["mis_mean"], r["mis_disc"], r["dv_ray"],
                 r["dv_disc"], cr, ar, lv, idv, ik, fl))

    def col(k):
        return np.array([r[k] for r in rows], float)

    print()
    print("SPREAD of each descriptor (sd / mean, per cent)")
    for k in ("a1", "a3", "mis_mean", "mis_disc", "dv_ray", "dv_disc"):
        v = col(k)
        print("  %-9s %8.4f +- %.4f   %5.2f %%"
              % (k, v.mean(), v.std(ddof=1),
                 100 * v.std(ddof=1) / abs(v.mean())))
    for k, v in (("crossings", cross), ("area_cm2", area)):
        print("  %-9s %8.4f +- %.4f   %5.2f %%"
              % (k, v.mean(), v.std(ddof=1),
                 100 * v.std(ddof=1) / abs(v.mean())))
    print("  %-9s %8.4f +- %.4f dB" % ("level", level.mean(),
                                       level.std(ddof=1)))
    print("  %-9s %8.4f +- %.4f" % ("ident", ident.mean(),
                                    ident.std(ddof=1)))

    print()
    print("EDGE 2 and EDGE 3 LINKS, n = 8 tessellations")
    print("  %-24s %-12s %8s %8s %8s %8s"
          % ("x", "y", "pearson", "p", "spearman", "p"))
    targets = [("level_dB", level), ("ident", ident), ("field", field),
               ("scalar", scal)]
    preds = [("a1", col("a1")), ("a3", col("a3")), ("a1_unw", col("a1u")),
             ("mis_adj", col("mis_mean")), ("mis_disc", col("mis_disc")),
             ("dv_ray", col("dv_ray")), ("dv_disc", col("dv_disc")),
             ("crossings", cross), ("area_cm2", area),
             ("n_grain", col("ngrain").astype(float))]
    for pn, pv in preds:
        for tn, tv in targets:
            r, p, rs, ps = rp(pv, tv)
            print("  %-24s %-12s %+8.3f %8.3f %+8.3f %8.3f"
                  % (pn, tn, r, p, rs, ps))

    print()
    print("INTERNAL LINKS of the edge-2 chain")
    for a, b in itertools.combinations(
            ["a1", "a3", "mis_adj", "dv_ray", "dv_disc"], 2):
        av = col({"mis_adj": "mis_mean"}.get(a, a))
        bv = col({"mis_adj": "mis_mean"}.get(b, b))
        r, p, rs, ps = rp(av, bv)
        print("  %-10s -> %-10s r %+.3f p %.3f  rho %+.3f p %.3f"
              % (a, b, r, p, rs, ps))

    print()
    print("WITHIN-SWEEP azimuthal link: does dv_ray(az) track level(az)?")
    for r, lv_az in zip(rows, gaz):
        x = r["dv_ray_az"]
        rr, pp = stats.pearsonr(10 * np.log10(x), lv_az)
        print("  seed %3d  r %+.3f (p %.3f, n=30, azimuths NOT independent)"
              % (r["seed"], rr, pp))

    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "edge23.npz"),
             seeds=np.array(SEEDS), level=level, cross=cross, area=area,
             ident=ident, field=field,
             **{k: col(k) for k in ("a1", "a3", "a1u", "mis_mean",
                                    "mis_disc", "dv_ray", "dv_disc",
                                    "ngrain")})


if __name__ == "__main__":
    main()
