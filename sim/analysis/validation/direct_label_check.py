"""Structural-fix validation: does labeling the grid DIRECTLY from
rotated seed points (single rasterisation) remove the 2-theta that the
production path (NN resample of the pre-rasterised build grid = DOUBLE
rasterisation) carries?

Uses an unweighted Voronoi of the kept seed-11 seeds as the test
specimen (same seeds for both paths, so the comparison is exact).
"""
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
from labforensics3 import rot_indices, H          # noqa: E402
from labforensics2 import harm360                 # noqa: E402
from mitigation_check import stats, ER            # noqa: E402

ROTS = np.arange(0, 360, 5)
R_DISK = 0.050


def main():
    d = np.load(os.path.join(SCRATCH, "std_build.npz"))   # seed 7: has seeds
    seeds = d["seeds"]                    # kept seeds, metres, centred
    nx_s, nz_s = 312, 110
    jc, cz = nx_s // 2, nz_s // 2

    u = (np.arange(nx_s) - (nx_s - 1) / 2) * H
    w = (np.arange(nz_s) - (nz_s - 1) / 2) * H
    X, Y = np.meshgrid(u, u, indexing="ij")
    inside = (X ** 2 + Y ** 2) <= R_DISK ** 2
    Pxy = np.stack([X[inside], Y[inside]], axis=1)
    npt = len(Pxy)

    def direct_label(rot):
        """Single rasterisation of the rotated continuum specimen."""
        a = np.radians(float(rot))
        ca, sa = np.cos(a), np.sin(a)
        s2 = seeds.copy()
        # specimen appears rotated by -rot in solver frame (rigid2)
        s2[:, 0] = ca * seeds[:, 0] + sa * seeds[:, 1]
        s2[:, 1] = -sa * seeds[:, 0] + ca * seeds[:, 1]
        lab = np.full((nx_s, nx_s, nz_s), -1, np.int16)
        tree = cKDTree(s2)
        for iz in range(nz_s):
            P = np.column_stack([Pxy, np.full(npt, w[iz])])
            _, idx = tree.query(P, workers=-1)
            plane = np.full((nx_s, nx_s), -1, np.int16)
            plane[inside] = idx.astype(np.int16)
            lab[:, :, iz] = plane
        return lab

    # build-time rasterisation at rot 0 (the reference build grid)
    base = direct_label(0)

    rows_nn, rows_dir = [], []
    t0 = time.time()
    for i, rot in enumerate(ROTS):
        gi, gj, ok = rot_indices(nx_s, int(rot), 0.0, 0.0)
        nn = np.where(ok[:, :, None], base[gi, gj, :], -1)   # production path
        rows_nn.append(stats(nn, jc, cz))
        rows_dir.append(stats(direct_label(int(rot)), jc, cz))
        if i % 12 == 0:
            print(f"  rot {rot} ({time.time()-t0:.0f} s)", flush=True)
    print(f"done ({time.time()-t0:.0f} s)")
    for k in ("ifc_ap", "cor_gx"):
        vn = np.array([r[k] for r in rows_nn], float)
        vd = np.array([r[k] for r in rows_dir], float)
        fn = harm360(vn, ROTS, orders=(1, 2, 4))
        fd = harm360(vd, ROTS, orders=(1, 2, 4))
        print(f"{k}: NN-resample  mean={fn['m']:.1f} "
              f"A2={fn['A2']:.2f}@{fn['phi2']:.1f} A4={fn['A4']:.2f}@{fn['phi4']:.1f}")
        print(f"{'':{len(k)}s}  direct-label mean={fd['m']:.1f} "
              f"A2={fd['A2']:.2f}@{fd['phi2']:.1f} A4={fd['A4']:.2f}@{fd['phi4']:.1f}"
              f"   (A2 ratio {fd['A2']/max(fn['A2'],1e-9):.2f})")
    np.savez(os.path.join(SCRATCH, "direct_label_check.npz"), rots=ROTS,
             **{f"nn_{k}": np.array([r[k] for r in rows_nn]) for k in rows_nn[0]},
             **{f"dir_{k}": np.array([r[k] for r in rows_dir]) for k in rows_dir[0]})


if __name__ == "__main__":
    main()
