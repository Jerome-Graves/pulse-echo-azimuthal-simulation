"""Block bootstrap on kappa (blocks of 15 deg >> the 3-7 deg glint
correlation length), plus the FD1 frozen-axis residual rms for the
sigma comparison.
"""
import os
import sys

import numpy as np

SIM = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim")))
sys.path.insert(0, SIM)
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import fit_sweep as FS                                   # noqa: E402
from measure_sigma import prep, stage2_kappa, stagedge1_axis, AXIS  # noqa
from kappa_frozen_axis_compare import set_fd                               # noqa: E402

SW = ("singlemax_seed11_ppw6_rigid2", "singlemax_seed23_ppw6_heldout_axis", "isotropic_seed41_ppw6_calibration")
BLOCK = 15
NBOOT = 120


def boot(cfg, data, rots, geo, tof_m, alpha, rng):
    r = np.asarray(rots, float)
    starts = np.arange(0, 360, BLOCK)
    out = []
    for _ in range(NBOOT):
        pick = rng.choice(starts, size=len(starts), replace=True)
        idx = []
        for s in pick:
            idx += [i for i, rr in enumerate(r) if s <= rr < s + BLOCK]
        d2 = [data[i] for i in idx]
        g2 = geo[idx]
        t2 = tof_m[idx]
        r2 = [data[i]["rot"] for i in idx]
        try:
            k, _, _ = stage2_kappa(cfg, d2, r2, g2, t2, alpha)
            out.append(k)
        except Exception:
            pass
    return np.array(out)


def main():
    print("=== FD1 frozen-axis residual rms (sigma comparison) ===")
    set_fd(1)
    FS.SIG_CODA_SWEEP = 1.8
    P1 = {n: prep(n) for n in ("singlemax_seed11_ppw6_rigid2", "singlemax_seed23_ppw6_heldout_axis")}
    for n in P1:
        cfg, data, rots, geo, tof_m = P1[n]
        for _ in range(3):
            k, keep, res = stage2_kappa(cfg, data, rots, geo, tof_m,
                                        AXIS[n])
            rms = float(np.sqrt(np.mean(res ** 2)))
            if abs(FS.SIG_CODA_SWEEP - rms) < 0.01:
                break
            FS.SIG_CODA_SWEEP = rms
        print(f"  FD1 {n:13s} frozen axis {AXIS[n]}  kappa {k:.3f}  "
              f"res rms {rms:.3f} dB")
        FS.SIG_CODA_SWEEP = 1.8

    print("\n=== FD3geo block bootstrap on kappa "
          f"({NBOOT} resamples, {BLOCK} deg blocks) ===")
    set_fd(3, "geo")
    FS.SIG_CODA_SWEEP = 1.82
    rng = np.random.default_rng(7)
    P = {n: prep(n) for n in SW}
    for n in SW:
        cfg, data, rots, geo, tof_m = P[n]
        a, _ = stagedge1_axis(cfg, data, rots, geo, tof_m)
        k0, _, _ = stage2_kappa(cfg, data, rots, geo, tof_m, a)
        ks = boot(cfg, data, rots, geo, tof_m, a, rng)
        lo, hi = np.percentile(ks, [5, 95])
        print(f"  {n:13s} kappa {k0:6.3f}  boot median {np.median(ks):6.3f}"
              f"  5-95% [{lo:.2f}, {hi:.2f}]  frac<=1.0 "
              f"{(ks <= 1.0).mean():.2f}")
        np.save(f"boot_{n}.npy", ks)


if __name__ == "__main__":
    main()
