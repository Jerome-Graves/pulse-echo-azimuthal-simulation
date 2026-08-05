"""TASK 2: honest SIG_CODA_SWEEP = clean-band residual rms of the
two-stage fit with the axis FROZEN at each sweep's known sample axis,
calibration OFF.  Iterated to self-consistency (sigma enters the Huber
gain and the kappa it is measured from).

Also TASK 3: free refits (stage 1 as shipped, then stage 2) reported
without touching any fit_result.json.
"""
import json
import os
import sys

import numpy as np

SIM = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim")))
sys.path.insert(0, SIM)
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import fit_sweep as FS          # noqa: E402
import fit_fabric as FF         # noqa: E402
from scipy.optimize import minimize, minimize_scalar   # noqa: E402

AXIS = {"rigid_seed11": 28.8, "oos_seed23": 104.3, "iso_gcal": None}


def cfg_of(name):
    p = os.path.join(os.path.dirname(SIM), "out", "sweeps", name,
                     "config.json")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


SCRATCH = os.path.dirname(os.path.abspath(__file__))


def prep(name):
    """geo is cached HERE, not in the repo: the repo's own
    fit_geo_cache write loses its atomic-replace race against whatever
    is polling the sweep dir, so every run recomputed 285 envelopes."""
    cfg = cfg_of(name)
    data = FS.sweep_data(cfg)
    rots = [d["rot"] for d in data]
    gp = os.path.join(SCRATCH, f"geo_{name}.npz")
    geo = None
    if os.path.exists(gp):
        with np.load(gp) as z:
            if list(z["rots"]) == list(rots):
                geo = np.array(z["geo"])
    if geo is None:
        geo, _ = FS.sweep_geo(cfg, rots)
        np.savez(gp, rots=np.asarray(rots), geo=np.asarray(geo))
    tof_m = np.array([d["tof_us"] for d in data])
    return cfg, data, rots, geo, tof_m


def tof_chi2(alpha, kappa, rots, geo, tof_m, sig=0.3):
    _, _, tof_p, _ = FF.predict_basis(alpha, kappa, rots, geo)
    t = tof_p - tof_p.mean()
    den = float(t @ t)
    if den < 1e-12:
        return 1e9
    b = float(t @ (tof_m - tof_m.mean())) / den
    if b < 0.2:
        return 1e9
    return float(np.sum(FF.huber(((tof_m - tof_m.mean()) - b * t) / sig)))


def stage2_kappa(cfg, data, rots, geo, tof_m, alpha):
    psi = FS._psi_from_axis(rots, alpha)
    keep = (psi >= 15.0) & ~((psi > 65.0) & (psi < 85.0))

    def obj(logk):
        k = float(np.exp(np.clip(logk, np.log(0.3), np.log(30.0))))
        return (FS._coda_chi2(k, alpha, data, geo, keep, name=cfg["name"],
                              f0_mhz=cfg["f0_mhz"])[0]
                + tof_chi2(alpha, k, rots, geo, tof_m))

    ks = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    k0 = min(ks, key=lambda k: obj(np.log(k)))
    r = minimize(lambda p: obj(p[0]), x0=[np.log(k0)], method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-4, maxiter=500))
    k = float(np.exp(np.clip(r.x[0], np.log(0.3), np.log(30.0))))
    _, res = FS._coda_chi2(k, alpha, data, geo, keep, name=cfg["name"],
                           f0_mhz=cfg["f0_mhz"])
    return k, keep, np.asarray(res, float)


def stage1_axis(cfg, data, rots, geo, tof_m):
    """Stage 1 exactly as shipped (ToF template + 2-theta channel)."""
    c2t = None
    if FS.C2T_CHANNEL:
        try:
            c2t = FS.coherent_2theta(cfg)
        except Exception:
            c2t = None

    def s1(a, k):
        c = tof_chi2(a, k, rots, geo, tof_m)
        if c2t is None:
            return c
        d = (a - c2t["phi2_deg"] + 90.0) % 180.0 - 90.0
        return (c / FS.C2T_TOF_INFO
                + (d / max(c2t["se_deg"], FS.C2T_SE_FLOOR)) ** 2)

    best = min(((s1(a, k), a, k) for a in range(0, 180, 2)
                for k in (1., 2., 3., 4., 6., 8.)), key=lambda x: x[0])
    r = minimize(lambda p: s1(p[0], np.exp(p[1])),
                 x0=[best[1], np.log(best[2])], method="Nelder-Mead",
                 options=dict(xatol=1e-3, fatol=1e-4, maxiter=2000))
    return float(r.x[0] % 180.0), c2t


def main():
    print(f"estimator: FD_BANDS={FS.FD_BANDS} combine={FS.FD_COMBINE} "
          f"span={FS.FD_SPAN}  SIG_CODA_SWEEP={FS.SIG_CODA_SWEEP}")
    print(f"calibration file present: "
          f"{os.path.exists(FS.cal_path(2.0))}  (must be False)\n")
    P = {n: prep(n) for n in ("rigid_seed11", "oos_seed23", "iso_gcal")}

    print("== TASK 2: axis FROZEN at the known sample axis ==")
    sig = {}
    for n in ("rigid_seed11", "oos_seed23"):
        cfg, data, rots, geo, tof_m = P[n]
        a = AXIS[n]
        for it in range(4):
            k, keep, res = stage2_kappa(cfg, data, rots, geo, tof_m, a)
            rms = float(np.sqrt(np.mean(res ** 2)))
            mad = float(1.4826 * np.median(np.abs(res - np.median(res))))
            print(f"  {n:13s} iter{it} sigma_in {FS.SIG_CODA_SWEEP:.3f} "
                  f"-> kappa {k:6.3f}  n_keep {keep.sum():3d}  "
                  f"res rms {rms:.3f} dB  robust {mad:.3f} dB")
            if abs(FS.SIG_CODA_SWEEP - rms) < 0.01:
                break
            FS.SIG_CODA_SWEEP = rms
        sig[n] = (rms, mad, k)
        FS.SIG_CODA_SWEEP = 1.8
    print()

    rec = float(np.mean([sig[n][0] for n in sig]))
    print(f"  recommended SIG_CODA_SWEEP = {rec:.2f} dB "
          f"(mean of the two frozen-axis residual rms)\n")

    print("== TASK 3: free refits at the recommended sigma ==")
    FS.SIG_CODA_SWEEP = round(rec, 2)
    for n in ("rigid_seed11", "oos_seed23", "iso_gcal"):
        cfg, data, rots, geo, tof_m = P[n]
        a, c2t = stage1_axis(cfg, data, rots, geo, tof_m)
        k, keep, res = stage2_kappa(cfg, data, rots, geo, tof_m, a)
        ax = np.asarray(cfg["fabric_axis"], float)
        ta = float(np.degrees(np.arctan2(ax[1], ax[0])) % 180.0)
        print(f"  {n:13s} axis {a:6.1f} (nominal truth {ta:5.1f}, "
              f"sample {AXIS[n]})  KAPPA {k:6.3f} "
              f"(truth {cfg['concentration']})  res rms "
              f"{np.sqrt(np.mean(res**2)):.3f} dB  n_keep {keep.sum()}")
    FS.SIG_CODA_SWEEP = 1.8


if __name__ == "__main__":
    main()
