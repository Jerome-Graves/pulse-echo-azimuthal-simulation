"""Exact facet table for a Laguerre tessellation, from the label grid.

For every neighbouring grain pair (i,j):
  n_ij   facet normal, EXACT: parallel to (seed_i - seed_j)
  A_ij   facet area.  Voxel face counting overestimates a plane's area by
         (|nx|+|ny|+|nz|) (the staircase), and that factor is known exactly
         because the normal is known, so it is divided out.
  x_ij   facet centroid
Cached to .npz because the build is ~13 s.
"""
import os
import sys

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vendor"))))
from specimen import DiskSpecimen             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def build_spec(h, kappa, axis, seed):
    return DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                        size_cv=0.35, concentration=kappa, spatial_corr=0.0,
                        fabric_axis=tuple(axis), seed=seed).build(h)


def facet_table(b):
    lab = np.asarray(b["labels"])
    h = float(b["h"])
    seeds = np.asarray(b["seeds"], float)
    nx, ny, nz = lab.shape
    cx = (np.arange(nx) + 0.5) * h - nx * h / 2
    cy = (np.arange(ny) + 0.5) * h - ny * h / 2
    cz = (np.arange(nz) + 0.5) * h - nz * h / 2
    ns = len(seeds)
    cnt = {}          # (i,j) -> face count
    pos = {}          # (i,j) -> summed midpoint
    for ax in range(3):
        A = np.take(lab, np.arange(0, lab.shape[ax] - 1), axis=ax)
        B = np.take(lab, np.arange(1, lab.shape[ax]), axis=ax)
        m = (A != B) & (A >= 0) & (B >= 0)
        if not m.any():
            continue
        ii, jj, kk = np.nonzero(m)
        a, bb = A[m].astype(np.int64), B[m].astype(np.int64)
        lo, hi = np.minimum(a, bb), np.maximum(a, bb)
        key = lo * ns + hi
        px = cx[ii] + (h / 2 if ax == 0 else 0.0)
        py = cy[jj] + (h / 2 if ax == 1 else 0.0)
        pz = cz[kk] + (h / 2 if ax == 2 else 0.0)
        u, inv = np.unique(key, return_inverse=True)
        c = np.bincount(inv)
        sx = np.bincount(inv, weights=px)
        sy = np.bincount(inv, weights=py)
        sz = np.bincount(inv, weights=pz)
        for q, k in enumerate(u):
            cnt[k] = cnt.get(k, 0) + c[q]
            p = pos.get(k)
            v = np.array([sx[q], sy[q], sz[q]])
            pos[k] = v if p is None else p + v
    keys = np.array(sorted(cnt))
    i = (keys // ns).astype(int)
    j = (keys % ns).astype(int)
    n = seeds[i] - seeds[j]
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    nfaces = np.array([cnt[k] for k in keys], float)
    stair = np.abs(n).sum(1)                       # |nx|+|ny|+|nz|
    area = nfaces * h * h / stair
    cen = np.array([pos[k] for k in keys]) / nfaces[:, None]
    return dict(i=i, j=j, n=n, area=area, cen=cen, nfaces=nfaces, h=h)


def get(kappa, axis, seed, h, tag):
    p = os.path.join(HERE, f"t3_facets_{tag}.npz")
    if os.path.exists(p):
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        return d
    b = build_spec(h, kappa, axis, seed)
    f = facet_table(b)
    f["axes"] = np.asarray(b["axes"], float)
    f["seeds"] = np.asarray(b["seeds"], float)
    np.savez(p, **f)
    return f


if __name__ == "__main__":
    f = get(3.93, (1.0, 0.0, 0.0), 7, 3850.0 / 2e6 / 6, "seed7")
    a = f["area"]
    print(f"seed 7: {len(a)} facets, total area {a.sum()*1e4:.1f} cm^2")
    print(f"  facet equivalent diameter sqrt(4A/pi): "
          f"mean {np.sqrt(4*a.mean()/np.pi)*1e3:.2f} mm, "
          f"area-weighted {np.sqrt(4*(a*a).sum()/a.sum()/np.pi)*1e3:.2f} mm")
    print(f"  area percentiles (mm^2): "
          f"{np.percentile(a*1e6,[10,50,90]).round(1)}")
    V = np.pi * 0.05 ** 2 * 0.035
    print(f"  facet area per unit volume {a.sum()/V:.1f} 1/m "
          f"(mean free path 1/that = {V/a.sum()*1e3:.1f} mm)")
    print(f"  |n_z| distribution: mean {np.abs(f['n'][:,2]).mean():.3f} "
          f"(0.5 = isotropic)")
