"""Is born_spec carrying FABRIC, or only microstructure GEOMETRY?

born_spec = SUM over facets of (dv/2vbar)^2 * D(theta) mixes two things:
  GEOMETRY  which facets exist and which point back at the transducer
  FABRIC    the qP velocity jump dv across each facet, which depends on
            the two grains' c-axis orientations relative to this beam

singlemax_ppw8 and girdle_perp_ppw8 share tessellation seed 11, so their
facet geometry is IDENTICAL and only the c-axes differ.  That makes two
clean discriminating tests possible:

  A. GEOMETRY-ONLY predictor: same facets, same directivity, but every
     dv set to a constant.  If this predicts as well, the fabric term is
     decoration.
  B. CROSS-PREDICTION: use the girdle specimen's born_spec to predict the
     single-max coda and vice versa.  Geometry is shared, so a
     geometry-driven correlation survives the swap; a fabric-driven one
     must degrade.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import j1

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward as F                                   # noqa: E402
from length_scales import DG, HALF, K, build, shift_p, strip   # noqa: E402
from beam_descriptors import OUT                           # noqa: E402

C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
CODA_W, BAND = (24e-6, 36e-6), (0.8e6, 3.0e6)
PAIR = [("singlemax_ppw8", 3.93, (0.866, 0.5, 0.0)),
        ("girdle_perp_ppw8", -8.0, (1.0, 0.0, 0.0))]


def coda_of(d):
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
        ef = np.abs(hilbert(sosfiltfilt(sos, tr)))
        e1.append(np.abs(hilbert(tr))[max(k0 - w, 0):k0 + w].max())
        cd.append(np.sqrt((ef[int(CODA_W[0] * fs):int(CODA_W[1] * fs)] ** 2)
                          .mean()))
        rots.append(int(f[2:5]))
    return np.array(rots), 20 * np.log10(np.array(cd) / np.mean(e1))


def preds(lab, ax, seeds, h, rots, nlat=7):
    """returns dict of per-azimuth predictors, incl. a geometry-only one"""
    nx, ny, nz = lab.shape
    d0, d1 = CODA_W[0] * C_REF / 2, CODA_W[1] * C_REF / 2
    s = np.arange(d0, d1, h)
    uu = np.linspace(-1, 1, nlat)
    U, W = np.meshgrid(uu, uu, indexing="ij")
    keep = U ** 2 + W ** 2 <= 1.0
    U, W = U[keep], W[keep]
    out = {"born_spec": [], "geom_only": []}
    for r in rots:
        a = np.radians(r)
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
        D = np.where(x < 1e-9, 1.0, 2 * j1(np.where(x < 1e-9, 1.0, x))
                     / np.where(x < 1e-9, 1.0, x)) ** 2
        R2 = ((vg[ia] - vg[ib]) / (2 * vbar)) ** 2
        out["born_spec"].append(float((R2 * D).sum()))
        out["geom_only"].append(float(D.sum()))       # dv set constant
    return {k: np.array(v, float) for k, v in out.items()}


S = {}
for name, kap, axis in PAIR:
    rots, coda = coda_of(os.path.join(OUT, name))
    lab, ax, seeds, h = build(8.0, kap, axis, 11)
    S[name] = (rots, coda, preds(lab, ax, seeds, h, rots))

print("A. GEOMETRY-ONLY vs FULL predictor (residual = harmonics removed)")
print(f"{'specimen':<19}{'predictor':<11}{'r':>8}{'p':>7}"
      f"{'  r resid':>10}{'p':>7}")
for name, _, _ in PAIR:
    rots, coda, P = S[name]
    for k in ("born_spec", "geom_only"):
        rf, pf = shift_p(P[k], coda)
        rr, pr = shift_p(strip(P[k], rots), strip(coda, rots))
        print(f"{name:<19}{k:<11}{rf:>8.3f}{pf:>7.3f}{rr:>10.3f}{pr:>7.3f}"
              f"{' *' if pr < 0.05 else ''}")

print("\nB. CROSS-PREDICTION (same tessellation, wrong fabric)")
print(f"{'predictor from':<19}{'applied to':<19}{'r resid':>9}{'p':>7}")
for pn, _, _ in PAIR:
    for cn, _, _ in PAIR:
        rots, coda, _ = S[cn]
        P = S[pn][2]
        rr, pr = shift_p(strip(P["born_spec"], rots), strip(coda, rots))
        tag = "  <- correct" if pn == cn else "  <- WRONG fabric"
        print(f"{pn:<19}{cn:<19}{rr:>9.3f}{pr:>7.3f}{tag}")
