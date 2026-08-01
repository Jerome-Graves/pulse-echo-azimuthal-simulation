"""Every directional property of the specimen the beam actually samples,
tested against the measured coda.

The predictor used until now averaged over EVERY grain in the disc.  The
beam does not do that: it samples the grains along its own chord, in the
depth range the coda gate listens to (46-69 mm).  A bulk average
therefore has no realisation-specific content, which is why it collapsed
to 89% a single 2-theta cosine and why the phase control failed.

These descriptors are computed inside the beam column only:

  n_grain     distinct grains the beam column intersects
  n_cross     grain boundaries crossed along the beam axis
  born        SUM over crossings of (dv/2vbar)^2  -- the physically
              correct monostatic Born backscatter: a boundary crossed
              while stepping ALONG the beam has its normal aligned with
              the beam, which is the specular backscatter condition, and
              dv is the qP velocity jump ACROSS that boundary FOR THIS
              beam direction.  This is fabric x geometry, not a template.
  v_var       velocity variance among the grains in the beam (the old
              E[R^2], but local)
  v_mean      mean qP velocity along the beam (drives ToF, included as a
              cross-check)
  cos_c       mean |cos| between beam and the c-axes it passes through

Scored with a CIRCULAR-SHIFT null (preserves azimuthal autocorrelation)
and Holm-corrected across the panel.  iso_gcal is the negative control:
nothing may survive there.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
import forward as F                              # noqa: E402
from specimen import DiskSpecimen                # noqa: E402

OUT = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
C_REF, F0 = 3850.0, 2.0e6
CODA_W, BAND = (24e-6, 36e-6), (0.8e6, 3.0e6)
ELEM, DIA = 6.35e-3, 0.100
KEYS = ["n_grain", "n_cross", "born", "v_var", "v_mean", "cos_c"]

SW = [("girdle_perp_ppw8", 8.0, -8.0, (1.0, 0.0, 0.0), 11, False),
      ("girdle_perp", 6.0, -8.0, (1.0, 0.0, 0.0), 11, False),
      ("rigid_seed11", 6.0, 3.93, (0.866, 0.5, 0.0), 11, False),
      ("iso_gcal", 6.0, 0.001, (1.0, 0.0, 0.0), 41, True)]


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
        ef = np.abs(hilbert(sosfiltfilt(sos, tr)))
        e1.append(np.abs(hilbert(tr))[max(k0 - w, 0):k0 + w].max())
        cd.append(np.sqrt((ef[int(CODA_W[0] * fs):int(CODA_W[1] * fs)] ** 2)
                          .mean()))
        rots.append(int(f[2:5]))
    e1 = np.array(e1)
    return np.array(rots), 20 * np.log10(np.array(cd) / e1.mean())


def descriptors(ppw, kappa, axis, seed, rots):
    h = C_REF / F0 / ppw
    b = DiskSpecimen(diameter_m=DIA, thickness_m=0.035, n_grains=100,
                     size_cv=0.35, concentration=kappa, spatial_corr=0.0,
                     fabric_axis=axis, seed=seed).build(h)
    lab = b["labels"]
    ax = np.asarray(b["axes"], float)
    nx, ny, nz = lab.shape
    d0, d1 = CODA_W[0] * C_REF / 2, CODA_W[1] * C_REF / 2
    s = np.arange(d0, d1, h / 2)                     # along beam, 2x sampled
    off = np.arange(-ELEM / 2, ELEM / 2 + h, h)      # lateral
    zc = (np.arange(nz) + 0.5) * h - nz * h / 2
    zk = np.arange(0, nz, max(1, int(round(2e-3 / h))))   # 2 mm in z
    out = {k: [] for k in KEYS}
    for r in rots:
        a = np.radians(r)
        n = np.array([np.cos(a), np.sin(a), 0.0])
        t = np.array([-np.sin(a), np.cos(a), 0.0])
        # grain velocity along THIS beam direction
        vg = np.interp(np.arccos(np.clip(np.abs(ax @ n), 0, 1)),
                       F._PSI, F._VQP)
        P = (DIA / 2 * n - s[:, None] * n)[:, None, None, :] \
            + off[None, :, None, None] * t \
            + np.zeros(3) + np.stack([np.zeros_like(zc[zk]),
                                      np.zeros_like(zc[zk]),
                                      zc[zk]], 1)[None, None, :, :]
        gi = np.rint((P[..., 0] + nx * h / 2) / h - 0.5).astype(int)
        gj = np.rint((P[..., 1] + ny * h / 2) / h - 0.5).astype(int)
        gk = np.rint((P[..., 2] + nz * h / 2) / h - 0.5).astype(int)
        ok = ((gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
              & (gk >= 0) & (gk < nz))
        L = np.where(ok, lab[np.clip(gi, 0, nx - 1), np.clip(gj, 0, ny - 1),
                             np.clip(gk, 0, nz - 1)], -1)
        good = L >= 0
        ids = L[good]
        vbar = vg.mean()
        out["n_grain"].append(len(np.unique(ids)))
        out["v_mean"].append(vg[ids].mean())
        out["v_var"].append(vg[ids].var())
        out["cos_c"].append(np.abs(ax[ids] @ n).mean())
        # boundaries crossed stepping ALONG the beam (axis 0)
        A, B = L[:-1], L[1:]
        cross = (A != B) & (A >= 0) & (B >= 0)
        out["n_cross"].append(int(cross.sum()))
        dv = vg[A[cross]] - vg[B[cross]]
        out["born"].append(float(((dv / (2 * vbar)) ** 2).sum()))
    return {k: np.array(v, float) for k, v in out.items()}


def shift_p(x, y):
    n = len(y)
    r0 = np.corrcoef(x, y)[0, 1]
    rs = np.array([np.corrcoef(x, np.roll(y, k))[0, 1] for k in range(n)])
    return r0, float(np.sum(np.abs(rs) >= abs(r0)) / n)


for name, ppw, kap, axis, seed, ctrl in SW:
    d = os.path.join(OUT, name)
    if not os.path.isdir(d):
        continue
    rots, coda = measure(d)
    if len(rots) < 8:
        continue
    D = descriptors(ppw, kap, axis, seed, rots)
    print(f"\n=== {name}  ppw{ppw:.0f}  n={len(rots)}"
          f"{'   [ISOTROPIC CONTROL]' if ctrl else ''}")
    res = []
    for k in KEYS:
        r, p = shift_p(D[k], coda)
        res.append((k, r, p, D[k].mean(), D[k].std()))
    order = np.argsort([x[2] for x in res])
    m = len(res)
    print(f"  {'descriptor':<10}{'mean':>9}{'sd':>8}{'r':>8}"
          f"{'shift p':>9}{'Holm':>8}")
    for rank, i in enumerate(order):
        k, r, p, mu, sd = res[i]
        holm = min(1.0, p * (m - rank))
        star = " *" if holm < 0.05 else ""
        print(f"  {k:<10}{mu:>9.3g}{sd:>8.3g}{r:>8.3f}{p:>9.3f}"
              f"{holm:>8.3f}{star}")
