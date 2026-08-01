"""Harness validation (FD1 must reproduce the shipped fit_result.json)
plus the honest discriminating-power test: kappa vs FROZEN axis on all
three sweeps.  On iso_gcal there is no axis, so the spread of kappa
over frozen axes IS the null distribution of the coda channel's kappa.
"""
import json
import os
import sys

import numpy as np

SIM = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim"
sys.path.insert(0, SIM)
import fit_sweep as FS                      # noqa: E402
from measure_sigma import (cfg_of, prep, stage2_kappa,   # noqa: E402
                           stage1_axis, AXIS)

SW = ("rigid_seed11", "oos_seed23", "iso_gcal")


def set_fd(nb, combine="geo"):
    FS.FD_BANDS = nb
    FS.FD_COMBINE = combine
    FS.OBS_VER = (f"fd{nb}{combine}-le25-t4" if nb > 1 else "fd1-le25-t4")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "fd3"
    if mode == "fd1":
        set_fd(1)
    else:
        set_fd(3, "geo")
    FS.SIG_CODA_SWEEP = float(sys.argv[2]) if len(sys.argv) > 2 else 1.8
    print(f"MODE {mode}  FD_BANDS={FS.FD_BANDS} combine={FS.FD_COMBINE} "
          f"sigma={FS.SIG_CODA_SWEEP}")
    P = {n: prep(n) for n in SW}

    print("\n-- free two-stage fit --")
    for n in SW:
        cfg, data, rots, geo, tof_m = P[n]
        a, _ = stage1_axis(cfg, data, rots, geo, tof_m)
        k, keep, res = stage2_kappa(cfg, data, rots, geo, tof_m, a)
        print(f"  {n:13s} axis {a:6.1f}  kappa {k:6.3f}  "
              f"res rms {np.sqrt(np.mean(res**2)):.3f}")

    print("\n-- kappa vs FROZEN axis (deg) --")
    hdr = "  " + "".join(f"{a:7d}" for a in range(0, 180, 15))
    print(f"  {'sweep':13s}" + hdr)
    for n in SW:
        cfg, data, rots, geo, tof_m = P[n]
        ks = []
        for a in range(0, 180, 15):
            k, _, _ = stage2_kappa(cfg, data, rots, geo, tof_m, float(a))
            ks.append(k)
        ks = np.array(ks)
        print(f"  {n:13s}  " + "".join(f"{k:7.2f}" for k in ks)
              + f"   | min {ks.min():.2f} max {ks.max():.2f} "
                f"med {np.median(ks):.2f}")


if __name__ == "__main__":
    main()
