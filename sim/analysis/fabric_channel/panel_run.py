import os as _os
import sys as _sys
# helper modules shared with sibling directories
for _d in ('..',):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import numpy as np, os, json
from observable_panel import *

NPERM = 20000

def zs(x):
    s = x.std(); return (x-x.mean())/(s if s > 0 else 1.0)

def design(rot, k=2):
    a = np.radians(rot.astype(float))
    return np.column_stack([np.ones_like(a), np.cos(k*a), np.sin(k*a)])

def harm_amp(z, D, Dpinv):
    """LSQ k-theta amplitude of standardised z (units of z's SD)."""
    c = Dpinv @ z
    return np.hypot(c[1], c[2])

def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    out = np.empty(m); run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (m-i)*p[idx]); out[idx] = min(run, 1.0)
    return out

def r_hi(r, n):
    z = np.arctanh(np.clip(abs(r), 0, 0.999)); return float(np.tanh(z + 1.96/np.sqrt(n-3)))

def n_distinct_shifts(rot):
    """Guard the circular-shift null and return the number of DISTINCT shifts.

    np.roll is a rigid rotation of the specimen ONLY when the azimuths are a
    uniform sampling of the whole circle: on a partial or ragged arc, rolling
    wraps a sample round to an azimuth the run never visited and the null is
    not a rotation of anything.  Refuse to run in that case.

    On a uniform full circle of n azimuths the predictor's 180 deg period
    makes shifts s and s + n/2 the same alignment (r(s + n/2) = r(s)
    identically), so only n/2 of the n shifts are distinct and n/2 is the
    denominator of the p-value.
    """
    r = np.asarray(rot, float); n = len(r)
    step = 360.0 / n
    gaps = np.diff(np.concatenate([r, [r[0] + 360.0]]))
    if n % 2 or not np.allclose(gaps, step, atol=1e-6):
        raise ValueError(
            "circular-shift null needs a uniformly sampled full circle with an "
            "even number of azimuths; got n=%d, step should be %.4g deg, "
            "azimuths %s" % (n, step, np.array2string(r, threshold=70)))
    return n // 2

def analyse(name, d=None, seed=7, verbose=True):
    cfg, rot, X, bw, dt = load_sweep(name, d)
    keys = sorted(k for k in X if not k.startswith("_"))
    pred, vbar = fabric_pred(cfg, rot)
    n = len(rot); rng = np.random.default_rng(seed)
    tof = rel_tof(bw, dt)
    D = design(rot); Dp = np.linalg.pinv(D)
    Z = np.column_stack([zs(X[k]) for k in keys])          # n x K
    K = len(keys)
    zpred = zs(pred)
    obs_r = Z.T @ zpred / n
    obs_h = np.array([harm_amp(Z[:, j], D, Dp) for j in range(K)])
    perms = np.array([rng.permutation(n) for _ in range(NPERM)])
    R = np.zeros((NPERM, K)); H = np.zeros((NPERM, K))
    for j in range(K):
        zp = Z[perms, j]
        R[:, j] = zp @ zpred / n
        C = zp @ Dp.T
        H[:, j] = np.hypot(C[:, 1], C[:, 2])
    nsh = n_distinct_shifts(rot)
    cyc = np.array([[float(np.mean(np.roll(Z[:, j], s)*zpred)) for j in range(K)] for s in range(nsh)])
    p_r = (np.sum(np.abs(R) >= np.abs(obs_r)-1e-12, 0)+1)/(NPERM+1)
    p_h = (np.sum(H >= obs_h-1e-12, 0)+1)/(NPERM+1)
    p_c = np.sum(np.abs(cyc) >= np.abs(obs_r)-1e-12, 0)/nsh
    # family-wise error rate of the LARGEST |r| in the panel under the same
    # circular-shift null (max statistic; the true alignment s=0 is one draw)
    p_cfw = float(np.mean(np.abs(cyc).max(1) >= np.abs(obs_r).max()-1e-12))
    maxR, maxH = np.abs(R).max(1), H.max(1)
    p_rmax = np.array([(np.sum(maxR >= abs(v)-1e-12)+1)/(NPERM+1) for v in obs_r])
    p_hmax = np.array([(np.sum(maxH >= v-1e-12)+1)/(NPERM+1) for v in obs_h])
    hall = holm(np.concatenate([p_r, p_h])); hr, hh = hall[:K], hall[K:]
    if verbose:
        print("\n"+"="*118)
        print("SWEEP %s  n=%d  ppw=%.0f  conc=%.2f  seed=%d  az %d..%d"
              % (name, n, cfg["ppw"], cfg["concentration"], cfg["seed"], rot.min(), rot.max()))
        print("  E[R^2] predictor: mean %.2f dB, sd %.4f dB, range %.3f dB" % (pred.mean(), pred.std(), np.ptp(pred)))
        print("  POSITIVE CONTROLS (declared separately, NOT in the coda family):")
        for lbl, y, pp, pl in (("rel_ToF (us)", tof, vbar, "vbar"), ("E1_level (dB)", X["_E1_db"], pred, "E[R^2]")):
            r = float(np.corrcoef(y, pp)[0, 1])
            zz = zs(y); h = harm_amp(zz, D, Dp)
            ph = (np.sum(np.hypot(*(( (zz[perms] @ Dp.T)[:, 1], (zz[perms] @ Dp.T)[:, 2] ))) >= h)+1)/(NPERM+1)
            print("    %-14s vs %-6s  r=%+.3f  sd=%.4g  2theta=%.3f sd (p_perm=%.4f)" % (lbl, pl, r, y.std(), h, ph))
        print("\n  CIRCULAR-SHIFT NULL: %d distinct alignments of %d azimuths."
              % (nsh, n))
        print("    largest |r| in the panel = %.3f (%s); family-wise p = %.4f"
              % (np.abs(obs_r).max(), keys[int(np.argmax(np.abs(obs_r)))], p_cfw))
        print("\n  CODA PANEL: %d observables x 2 tests = %d p-values; Holm over all %d. Ranked by |r|." % (K, 2*K, 2*K))
        print("  %-14s %9s %7s %8s %7s %7s %7s | %6s %8s %7s %7s | %7s"
              % ("observable", "sd", "r", "p_perm", "p_cyc", "p_holm", "p_max", "2th/sd", "p_perm", "p_holm", "p_max", "|r|95hi"))
        for j in np.argsort(-np.abs(obs_r)):
            print("  %-14s %9.4g %+7.3f %8.4f %7.3f %7.3f %7.3f | %6.3f %8.4f %7.3f %7.3f | %7.3f"
                  % (keys[j], X[keys[j]].std(), obs_r[j], p_r[j], p_c[j], hr[j], p_rmax[j],
                     obs_h[j], p_h[j], hh[j], p_hmax[j], r_hi(obs_r[j], n)))
    return dict(name=name, n=n, keys=keys, X=X, rot=rot, pred=pred, tof=tof, cfg=cfg,
                obs_r=obs_r, obs_h=obs_h, p_r=p_r, p_h=p_h, p_c=p_c, holm_r=hr, holm_h=hh,
                p_rmax=p_rmax, p_hmax=p_hmax, p_cfw=p_cfw, nsh=nsh, dt=dt, bw=bw)

SWEEPS = ("girdle_perp_ppw8", "girdle_perp", "gcheck_ppw8", "iso_gcal")

if __name__ == "__main__":
    res = {nm: analyse(nm) for nm in SWEEPS}
    np.save(os.path.join(CACHE, "res.npy"), res, allow_pickle=True)
    print("\n" + "="*118)
    print("FAMILY-WISE SUMMARY")
    print("="*118)
    print("  %-20s %4s %5s %8s %10s %12s" % ("sweep", "n", "shift", "max|r|",
                                             "p_fw(cyc)", "min p_cyc"))
    for nm in SWEEPS:
        d = res[nm]
        print("  %-20s %4d %5d %8.3f %10.4f %12.4f"
              % (nm, d["n"], d["nsh"], np.abs(d["obs_r"]).max(), d["p_cfw"],
                 d["p_c"].min()))
    allp = np.concatenate([np.concatenate([res[nm]["p_r"], res[nm]["p_h"]])
                           for nm in SWEEPS])
    adj = holm(allp)
    print("  Holm over the union family of %d permutation p-values "
          "(%d observables x 2 tests x %d sweeps): smallest adjusted p = %.5f"
          % (len(allp), len(res[SWEEPS[0]]["keys"]), len(SWEEPS), adj.min()))
