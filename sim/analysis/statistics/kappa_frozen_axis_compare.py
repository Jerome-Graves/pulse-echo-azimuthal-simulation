"""Harness validation (FD1 must reproduce the shipped fit_result.json)
plus the honest discriminating-power test: kappa vs FROZEN axis on all
three sweeps.  On isotropic_seed41_ppw6_calibration there is no axis, so the spread of kappa
over frozen axes IS the null distribution of the coda channel's kappa.
"""
import json
import os
import sys

import numpy as np

SIM = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim")))
sys.path.insert(0, SIM)
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import fit_sweep as FS                      # noqa: E402
from measure_sigma import (cfg_of, prep, stage2_kappa,   # noqa: E402
                           stagedge1_axis, AXIS)

SW = ("singlemax_seed11_ppw6_rigid2", "singlemax_seed23_ppw6_heldout_axis", "isotropic_seed41_ppw6_calibration")


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
        a, _ = stagedge1_axis(cfg, data, rots, geo, tof_m)
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
