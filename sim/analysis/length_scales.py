"""Corrected geometry: the true diverging cone, and exact facet normals.

TWO ERRORS IN MY EARLIER DESCRIPTOR, both exposed by asking what the
wavefield actually is:

1. BEAM WIDTH.  The near field ends at 5.2 mm, so the coda gate (46-69 mm)
   is far field and the beam has spread to 14-22 mm diameter.  I sampled a
   6.35 mm cylinder, i.e. roughly a tenth of the insonified area.  Here the
   rays diverge properly at the 8.9 deg half angle.

2. FACET ORIENTATION.  Monostatic backscatter from a planar facet is
   controlled by how close its NORMAL points back at the transducer, and
   the lobe is only lambda/D_grain = 6.3 deg wide.  A Laguerre tessellation
   gives this exactly for free: the boundary between grains i and j is
   perpendicular to (seed_i - seed_j).  So the facet directivity can be
   computed without estimating anything from the voxels.

Descriptors:
  born_iso   sum of (dv/2vbar)^2 over facets, no orientation weighting
  born_spec  same, weighted by the facet directivity |2J1(x)/x|^2,
             x = k*Dg*sin(angle between beam and facet normal)
  n_spec     number of facets inside the specular lobe
  v_var      velocity spread over the cone (for comparison with before)
"""
import os
import sys

import numpy as np
from scipy.special import j1

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward as F                                   # noqa: E402
from beam_descriptors import OUT, measure                  # noqa: E402
from specimen import DiskSpecimen                     # noqa: E402

C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
CODA_W = (24e-6, 36e-6)
LAM = C_REF / F0
HALF = np.radians(8.9)
DG = 17.4e-3
K = 2 * np.pi / LAM

SW = [("girdle_perp_ppw8", 8.0, -8.0, (1.0, 0.0, 0.0), 11, False),
      ("rigid_seed11", 6.0, 3.93, (0.866, 0.5, 0.0), 11, False),
      ("iso_gcal", 6.0, 0.001, (1.0, 0.0, 0.0), 41, True)]


def direc(ct):
    """facet directivity power, ct = |cos(beam, facet normal)|"""
    st = np.sqrt(np.clip(1 - ct ** 2, 0, 1))
    x = K * DG * st
    d = np.where(x < 1e-9, 1.0, 2 * j1(x) / np.where(x < 1e-9, 1.0, x))
    return d ** 2


def build(ppw, kappa, axis, seed):
    h = C_REF / F0 / ppw
    b = DiskSpecimen(diameter_m=DIA, thickness_m=0.035, n_grains=100,
                     size_cv=0.35, concentration=kappa, spatial_corr=0.0,
                     fabric_axis=axis, seed=seed).build(h)
    return b["labels"], np.asarray(b["axes"], float), \
        np.asarray(b["seeds"], float), h


def cone(lab, ax, seeds, h, rots, nlat=7):
    nx, ny, nz = lab.shape
    d0, d1 = CODA_W[0] * C_REF / 2, CODA_W[1] * C_REF / 2
    s = np.arange(d0, d1, h)
    uu = np.linspace(-1, 1, nlat)
    U, W = np.meshgrid(uu, uu, indexing="ij")
    keep = U ** 2 + W ** 2 <= 1.0
    U, W = U[keep], W[keep]              # normalised cone coords
    out = {k: [] for k in ("born_iso", "born_spec", "n_spec", "v_var")}
    for r in rots:
        a = np.radians(r)
        n = np.array([np.cos(a), np.sin(a), 0.0])
        t1 = np.array([-np.sin(a), np.cos(a), 0.0])
        t2 = np.array([0.0, 0.0, 1.0])
        vg = np.interp(np.arccos(np.clip(np.abs(ax @ n), 0, 1)),
                       F._PSI, F._VQP)
        vbar = vg.mean()
        rad = (s * np.tan(HALF))[:, None]                    # cone radius
        P = (DIA / 2 * n - s[:, None] * n)[:, None, :] \
            + (rad * U[None, :])[:, :, None] * t1 \
            + (rad * W[None, :])[:, :, None] * t2
        gi = np.rint((P[..., 0] + nx * h / 2) / h - 0.5).astype(int)
        gj = np.rint((P[..., 1] + ny * h / 2) / h - 0.5).astype(int)
        gk = np.rint((P[..., 2] + nz * h / 2) / h - 0.5).astype(int)
        ok = ((gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
              & (gk >= 0) & (gk < nz))
        L = np.where(ok, lab[np.clip(gi, 0, nx - 1), np.clip(gj, 0, ny - 1),
                             np.clip(gk, 0, nz - 1)], -1)
        A, B = L[:-1], L[1:]
        cr = (A != B) & (A >= 0) & (B >= 0)
        ia, ib = A[cr], B[cr]
        dv = vg[ia] - vg[ib]
        R2 = (dv / (2 * vbar)) ** 2
        nm = seeds[ia] - seeds[ib]
        nm /= np.linalg.norm(nm, axis=1, keepdims=True) + 1e-30
        ct = np.abs(nm @ n)
        w = direc(ct)
        good = L[L >= 0]
        out["born_iso"].append(float(R2.sum()))
        out["born_spec"].append(float((R2 * w).sum()))
        out["n_spec"].append(float((w > 0.25).sum()))
        out["v_var"].append(float(vg[good].var()))
    return {k: np.array(v, float) for k, v in out.items()}


def shift_p(x, y):
    n = len(y)
    r0 = np.corrcoef(x, y)[0, 1]
    rs = [np.corrcoef(x, np.roll(y, k))[0, 1] for k in range(n)]
    return r0, float(np.sum(np.abs(rs) >= abs(r0)) / n)


def strip(y, az):
    t = np.radians(az)
    A = np.column_stack([np.ones_like(t), np.cos(2 * t), np.sin(2 * t),
                         np.cos(4 * t), np.sin(4 * t)])
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


print(f"{'specimen':<19}{'n':>5}  {'descriptor':<11}"
      f"{'r':>8}{'p':>7}  {'r resid':>9}{'p':>7}")
for name, ppw, kap, axis, seed, ctrl in SW:
    d = os.path.join(OUT, name)
    if not os.path.isdir(d):
        continue
    rots, coda = measure(d)
    if len(rots) > 120:                       # subsample the 360-az sweeps
        sel = np.arange(0, len(rots), len(rots) // 120)
        rots, coda = rots[sel], coda[sel]
    lab, ax, seeds, h = build(ppw, kap, axis, seed)
    D = cone(lab, ax, seeds, h, rots)
    for k in ("born_spec", "born_iso", "n_spec", "v_var"):
        rf, pf = shift_p(D[k], coda)
        rr, pr = shift_p(strip(D[k], rots), strip(coda, rots))
        st = " *" if pf < 0.05 else ""
        print(f"{name:<19}{len(rots):>5}  {k:<11}{rf:>8.3f}{pf:>7.3f}"
              f"{rr:>9.3f}{pr:>7.3f}{st}"
              f"{'  [CONTROL]' if ctrl else ''}")
    print()
