"""Score the zero-contrast control against the facet model.

Written BEFORE the sweep finished, so the processing is fixed in advance.

Three numbers to beat, all from the real specimen (girdle_perp_ppw8):
  scalar azimuthal, geometry-only predictor   residual r = +0.465
  scalar azimuthal, full facet model          residual r = +0.511
  time-resolved field, full facet model                r = +0.366

The zero-contrast run has the SAME tessellation, so the geometry-only
predictor is identical for it.  The full model is built from the REAL
specimen's c-axes, which asks whether the artefact can imitate the
fabric-weighted version too.

A near-zero score confirms the facet result.  A score near 0.465 means
the effect was staircase all along.
"""
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import j1

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
import forward as F                                    # noqa: E402
from specimen import DiskSpecimen                      # noqa: E402

SWD = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "sweeps")))
C, F0, DIA = 3850.0, 2.0e6, 0.100
GATE, BAND = (24e-6, 36e-6), (0.8e6, 3.0e6)
DG, HALF = 17.4e-3, np.radians(8.9)
KP = 2 * np.pi * F0 / C
NT = 240
TG = np.linspace(GATE[0], GATE[1], NT)
h = C / F0 / 8.0
T0 = 1.2 / F0                                       # ricker source delay

b = DiskSpecimen(diameter_m=DIA, thickness_m=0.035, n_grains=100,
                 size_cv=0.35, concentration=-8.0, spatial_corr=0.0,
                 fabric_axis=(1.0, 0.0, 0.0), seed=11).build(h)
LAB, AXG = b["labels"], np.asarray(b["axes"], float)
SEEDS = np.asarray(b["seeds"], float)
NX, NY, NZ = LAB.shape


def facets(rot, nlat=7):
    a = np.radians(rot)
    n = np.array([np.cos(a), np.sin(a), 0.0])
    t1 = np.array([-np.sin(a), np.cos(a), 0.0])
    t2 = np.array([0.0, 0.0, 1.0])
    vg = np.interp(np.arccos(np.clip(np.abs(AXG @ n), 0, 1)), F._PSI, F._VQP)
    s = np.arange(GATE[0] * C / 2, GATE[1] * C / 2, h)
    uu = np.linspace(-1, 1, nlat)
    U, W = np.meshgrid(uu, uu, indexing="ij")
    k = U ** 2 + W ** 2 <= 1.0
    U, W = U[k], W[k]
    rad = (s * np.tan(HALF))[:, None]
    P = (DIA / 2 * n - s[:, None] * n)[:, None, :] \
        + (rad * U[None, :])[:, :, None] * t1 \
        + (rad * W[None, :])[:, :, None] * t2
    gi = np.rint((P[..., 0] + NX * h / 2) / h - 0.5).astype(int)
    gj = np.rint((P[..., 1] + NY * h / 2) / h - 0.5).astype(int)
    gk = np.rint((P[..., 2] + NZ * h / 2) / h - 0.5).astype(int)
    ok = ((gi >= 0) & (gi < NX) & (gj >= 0) & (gj < NY)
          & (gk >= 0) & (gk < NZ))
    L = np.where(ok, LAB[np.clip(gi, 0, NX - 1), np.clip(gj, 0, NY - 1),
                         np.clip(gk, 0, NZ - 1)], -1)
    A, B = L[:-1], L[1:]
    cr = (A != B) & (A >= 0) & (B >= 0)
    si = np.broadcast_to(s[:-1, None], A.shape)[cr]
    ia, ib = A[cr], B[cr]
    nm = SEEDS[ia] - SEEDS[ib]
    nm /= np.linalg.norm(nm, axis=1, keepdims=True) + 1e-30
    st = np.sqrt(np.clip(1 - (nm @ n) ** 2, 0, 1))
    x = KP * DG * st
    D = np.where(x < 1e-9, 1.0, 2 * j1(np.where(x < 1e-9, 1.0, x))
                 / np.where(x < 1e-9, 1.0, x)) ** 2
    v = 0.5 * (vg[ia] + vg[ib])
    R2 = ((vg[ia] - vg[ib]) / (2 * vg.mean())) ** 2
    return 2 * si / v + T0, R2, D


def bin_train(t, w):
    y = np.zeros(NT)
    m = (t >= GATE[0]) & (t < GATE[1])
    if m.any():
        np.add.at(y, np.clip(np.searchsorted(TG, t[m]), 0, NT - 1), w[m])
    return uniform_filter1d(y, max(int(NT / (GATE[1] - GATE[0]) / F0), 3))


def load(name):
    d = os.path.join(SWD, name)
    rots = sorted(int(f[2:5]) for f in os.listdir(d)
                  if f.startswith("az") and f.endswith(".npz"))
    scal, field = [], []
    for r in rots:
        with np.load(os.path.join(d, f"az{r:03d}.npz")) as z:
            tr, dt = np.asarray(z["trace"], float).ravel(), float(z["dt"])
        fs = 1.0 / dt
        sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                     btype="band", output="sos")
        e = np.abs(hilbert(sosfiltfilt(sos, tr)))
        k0, w = int(2 * DIA / C * fs), int(2e-6 * fs)
        scal.append((np.sqrt((e[int(GATE[0]*fs):int(GATE[1]*fs)] ** 2).mean()),
                     e[max(k0 - w, 0):k0 + w].max()))
        es = uniform_filter1d(e, int(1.0e-6 * fs))
        field.append(np.interp(TG, np.arange(len(es)) / fs, es))
    s = np.array(scal)
    return rots, 20 * np.log10(s[:, 0] / s[:, 1].mean()), np.array(field)


def strip(y, az, orders=(2, 4)):
    t = np.radians(az)
    cols = [np.ones_like(t)]
    for k in orders:
        cols += [np.cos(k * t), np.sin(k * t)]
    A = np.column_stack(cols)
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def cen(X):
    X = X - X.mean(0, keepdims=True)
    return X - X.mean(1, keepdims=True)


def shift_p(x, y):
    n = len(y)
    r0 = np.corrcoef(x, y)[0, 1]
    rs = [np.corrcoef(x, np.roll(y, k))[0, 1] for k in range(n)]
    return r0, float(np.sum(np.abs(rs) >= abs(r0)) / n)


REF = {"geom_only": 0.465, "born_spec": 0.511, "field": 0.366}
print(f"{'sweep':<21}{'n':>4}{'geom_only':>11}{'p':>7}"
      f"{'born_spec':>11}{'p':>7}{'field':>9}")
for name in ("girdle_perp_ppw8", "zerocontrast_ppw8"):
    d = os.path.join(SWD, name)
    if not os.path.isdir(d):
        continue
    rots, scal, field = load(name)
    if len(rots) < 10:
        print(f"{name}: only {len(rots)} azimuths so far, skipping")
        continue
    FT = [facets(r) for r in rots]
    go = np.array([bin_train(t, dd).sum() for t, _, dd in FT])
    bs = np.array([bin_train(t, r2 * dd).sum() for t, r2, dd in FT])
    Pf = np.array([bin_train(t, r2 * dd) for t, r2, dd in FT])
    rg, pg = shift_p(strip(go, rots), strip(scal, rots))
    rb, pb = shift_p(strip(bs, rots), strip(scal, rots))
    Mc, Pc = cen(field), cen(Pf)
    rf = np.corrcoef(Mc.ravel(), Pc.ravel())[0, 1]
    print(f"{name:<21}{len(rots):>4}{rg:>11.3f}{pg:>7.3f}"
          f"{rb:>11.3f}{pb:>7.3f}{rf:>9.3f}")
print(f"\nreal-specimen reference: geom_only {REF['geom_only']:+.3f}, "
      f"born_spec {REF['born_spec']:+.3f}, field {REF['field']:+.3f}")
print("near zero on the zero-contrast row CONFIRMS the facet result;")
print("near the reference values means it was staircase all along.")
