"""TEST 1: does the facet-geometry model predict the coda on ALL FOUR
tessellations, or only on seed 11?

Predictors (exactly as in cone_specular.py / decisive.py):
  geom_only  SUM over beam-crossed facets of the specular directivity
             |2J1(k Dg sin th)/x|^2 -- pure GEOMETRY, no fabric at all
  born_spec  same, weighted by (dv/2vbar)^2 -- geometry x fabric
  born_iso   SUM of (dv/2vbar)^2, no orientation weighting -- fabric only
  n_cross    raw count of facet crossings (naive geometry baseline)

Scoring: circular-shift null (floor 1/n), plus harmonic stripping
(mean, 2-theta, 4-theta removed from BOTH series).
"""
import json
import os
import sys
import time

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import j1

SIM = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim")))
sys.path.insert(0, SIM)
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
import forward as F                                   # noqa: E402
from specimen import DiskSpecimen                     # noqa: E402

OUT = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "sweeps")))
SCR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCR, "t1cache")
os.makedirs(CACHE, exist_ok=True)

C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
CODA_W, BAND = (24e-6, 36e-6), (0.8e6, 3.0e6)
LAM = C_REF / F0
HALF = np.radians(8.9)          # far-field divergence half-angle
DG = 17.4e-3                    # mean grain diameter -> specular lobe width
K = 2 * np.pi / LAM
KEYS = ("geom_only", "born_spec", "born_iso", "n_cross")

# name, ppw, kappa, axis, seed, is_isotropic, raster
SWEEPS = [
    ("rigid_seed11",     6.0,  3.93, (0.866, 0.5, 0.0),    11, False, "double"),
    ("girdle_perp",      6.0, -8.00, (1.0, 0.0, 0.0),      11, False, "single"),
    ("girdle_par",       6.0, -8.00, (0.0, 0.0, 1.0),      11, False, "single"),
    ("kappa8_seed17",    6.0,  8.00, (0.5, 0.866, 0.0),    17, False, "double"),
    ("oos_seed23",       6.0,  3.93, (-0.342, 0.94, 0.0),  23, False, "double"),
    ("iso_gcal",         6.0, 0.001, (1.0, 0.0, 0.0),      41, True,  "double"),
    ("girdle_perp_ppw8", 8.0, -8.00, (1.0, 0.0, 0.0),      11, False, "single"),
    ("singlemax_ppw8",   8.0,  3.93, (0.866, 0.5, 0.0),    11, False, "single"),
]
SMALL = [
    ("girdle_20",        6.0, -8.00, (0.342, 0.0, 0.94),   11, False, "single"),
    ("gcheck_ppw8",      8.0,  3.93, (0.866, 0.5, 0.0),    11, False, "double"),
]
MAXN = 120


def measure(d):
    rots, cd, e1 = [], [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        k0, w = int(2 * DIA / C_REF * fs), int(2e-6 * fs)
        if k0 + w >= len(tr):
            continue
        sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                     btype="band", output="sos")
        ef = np.abs(hilbert(sosfiltfilt(sos, tr)))        # filter WHOLE trace
        e1.append(np.abs(hilbert(tr))[max(k0 - w, 0):k0 + w].max())
        cd.append(np.sqrt((ef[int(CODA_W[0] * fs):int(CODA_W[1] * fs)] ** 2)
                          .mean()))
        rots.append(int(f[2:5]))
    return np.array(rots), 20 * np.log10(np.array(cd) / np.mean(e1))


def build_cached(ppw, kappa, axis, seed):
    h = C_REF / F0 / ppw
    tag = f"s{seed}_p{ppw:g}_k{kappa:g}_a{axis[0]:g}_{axis[1]:g}_{axis[2]:g}"
    p = os.path.join(CACHE, tag + ".npz")
    if os.path.exists(p):
        with np.load(p) as z:
            return z["labels"], z["axes"], z["seeds"], h
    t = time.time()
    b = DiskSpecimen(diameter_m=DIA, thickness_m=0.035, n_grains=100,
                     size_cv=0.35, concentration=kappa, spatial_corr=0.0,
                     fabric_axis=axis, seed=seed).build(h)
    lab = np.asarray(b["labels"])
    ax = np.asarray(b["axes"], float)
    sd = np.asarray(b["seeds"], float)
    np.savez_compressed(p, labels=lab, axes=ax, seeds=sd)
    print(f"    built {tag} in {time.time()-t:.0f}s  "
          f"grid{lab.shape} ngrain={len(sd)}", flush=True)
    return lab, ax, sd, h


def preds(lab, ax, seeds, h, rots, nlat=7, sign=1.0):
    """Diverging-cone sampling; exact Laguerre facet normals."""
    nx, ny, nz = lab.shape
    d0, d1 = CODA_W[0] * C_REF / 2, CODA_W[1] * C_REF / 2
    s = np.arange(d0, d1, h)
    uu = np.linspace(-1, 1, nlat)
    U, W = np.meshgrid(uu, uu, indexing="ij")
    keep = U ** 2 + W ** 2 <= 1.0
    U, W = U[keep], W[keep]
    out = {k: [] for k in KEYS}
    for r in rots:
        a = sign * np.radians(r)
        n = np.array([np.cos(a), np.sin(a), 0.0])
        t1 = np.array([-np.sin(a), np.cos(a), 0.0])
        t2 = np.array([0.0, 0.0, 1.0])
        vg = np.interp(np.arccos(np.clip(np.abs(ax @ n), 0, 1)),
                       F._PSI, F._VQP)
        vbar = vg.mean()
        rad = (s * np.tan(HALF))[:, None]
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
        nm = seeds[ia] - seeds[ib]
        nm /= np.linalg.norm(nm, axis=1, keepdims=True) + 1e-30
        st = np.sqrt(np.clip(1 - (nm @ n) ** 2, 0, 1))
        x = K * DG * st
        D = np.where(x < 1e-9, 1.0,
                     2 * j1(np.where(x < 1e-9, 1.0, x))
                     / np.where(x < 1e-9, 1.0, x)) ** 2
        R2 = ((vg[ia] - vg[ib]) / (2 * vbar)) ** 2
        out["geom_only"].append(float(D.sum()))
        out["born_spec"].append(float((R2 * D).sum()))
        out["born_iso"].append(float(R2.sum()))
        out["n_cross"].append(float(cr.sum()))
    return {k: np.array(v, float) for k, v in out.items()}


def shift_p(x, y):
    """Circular-shift null.  Two-sided on |r|.  Floor 1/n."""
    n = len(y)
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return np.nan, np.nan
    r0 = np.corrcoef(x, y)[0, 1]
    rs = np.array([np.corrcoef(x, np.roll(y, k))[0, 1] for k in range(n)])
    return float(r0), float(np.sum(np.abs(rs) >= abs(r0)) / n)


def strip(y, az):
    t = np.radians(az)
    A = np.column_stack([np.ones_like(t), np.cos(2 * t), np.sin(2 * t),
                         np.cos(4 * t), np.sin(4 * t)])
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def neff(y):
    """effective n from the circular autocorrelation integral length"""
    n = len(y)
    z = y - y.mean()
    ac = np.array([np.dot(z, np.roll(z, k)) for k in range(n)]) / np.dot(z, z)
    k = 1
    while k < n // 2 and ac[k] > 0:
        k += 1
    tau = 1 + 2 * ac[1:k].sum()
    return max(2.0, n / max(tau, 1.0)), tau


def fisher_ci(r, ne):
    if ne <= 3 or not np.isfinite(r):
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    s = 1.96 / np.sqrt(ne - 3)
    return (float(np.tanh(z - s)), float(np.tanh(z + s)))


def holm(pv):
    pv = np.asarray(pv, float)
    m = len(pv)
    o = np.argsort(pv)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (m - i) * pv[idx])
        adj[idx] = min(1.0, run)
    return adj


