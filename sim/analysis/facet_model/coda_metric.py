"""How much of the measured coda modulation is FABRIC, at ppw 6 vs ppw 8?

Same specimen (seed 11, girdle k=-8, normal along x), same everything except
the grid.  Predict each azimuth's coda level from the ACTUAL grains, regress
the measurement on it, and split the measured variance into a
fabric-explained part and a residual (speckle + numerical) part.

Prediction: a beam along n sees a volume-weighted distribution of qP
velocities v(angle between n and each grain's c-axis).  Single-scattering
backscatter power goes as E[R^2] = Var(v) / (2 vbar^2).
"""
import os
import sys

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
import forward as F                              # noqa: E402
from specimen import DiskSpecimen                # noqa: E402

OUT = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
C_REF, F0 = 3850.0, 2.0e6
CODA_W = (24e-6, 36e-6)


def predict(ppw):
    """Volume-weighted E[R^2] per azimuth, in dB, from the real grains."""
    h = C_REF / F0 / ppw
    sp = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                      size_cv=0.35, concentration=-8.0, spatial_corr=0.0,
                      fabric_axis=(1.0, 0.0, 0.0), seed=11)
    b = sp.build(h)
    lab, axes = b["labels"], np.asarray(b["axes"], float)
    ok = lab >= 0
    vol = np.bincount(lab[ok].ravel(), minlength=len(axes)).astype(float)
    vol /= vol.sum()
    keep = vol > 0
    axes, vol = axes[keep], vol[keep] / vol[keep].sum()

    def one(rot):
        a = np.radians(rot)
        n = np.array([np.cos(a), np.sin(a), 0.0])
        psi = np.arccos(np.clip(np.abs(axes @ n), 0.0, 1.0))
        v = np.interp(psi, F._PSI, F._VQP)
        vb = float(vol @ v)
        var = float(vol @ (v - vb) ** 2)
        return 10.0 * np.log10(var / (2.0 * vb ** 2))
    return one


def measured(d):
    rots, lv = [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        e = np.abs(hilbert(tr))
        k0 = int(2 * 0.100 / C_REF * fs)
        w = int(2e-6 * fs)
        E1 = e[max(k0 - w, 0):k0 + w].max()
        c = np.sqrt((e[int(CODA_W[0] * fs):int(CODA_W[1] * fs)] ** 2).mean())
        rots.append(int(f[2:5]))
        lv.append(20.0 * np.log10(c / E1))
    return np.array(rots), np.array(lv)


def perm_p(x, y, n=20000, seed=0):
    r0 = np.corrcoef(x, y)[0, 1]
    rng = np.random.default_rng(seed)
    hits = sum(abs(np.corrcoef(rng.permutation(x), y)[0, 1]) >= abs(r0)
               for _ in range(n))
    return r0, (hits + 1) / (n + 1)


print(f"{'sweep':<22}{'n':>4}{'coda/E1':>10}{'r':>8}{'p':>9}"
      f"{'fabric':>9}{'resid':>8}")
for name, ppw in (("girdle_perp", 6.0), ("girdle_perp_ppw8", 8.0)):
    d = os.path.join(OUT, name)
    if not os.path.isdir(d):
        continue
    rot, meas = measured(d)
    if len(rot) < 8:
        continue
    fn = predict(ppw)
    pred = np.array([fn(r) for r in rot])
    r, p = perm_p(pred.copy(), meas)
    s = np.polyfit(pred, meas, 1)[0]
    fab = abs(s) * pred.std()
    res = np.sqrt(max(meas.var() - fab ** 2, 0.0))
    print(f"{name:<22}{len(rot):>4}{meas.mean():>9.1f}d{r:>8.3f}{p:>9.4f}"
          f"{fab:>8.2f}d{res:>7.2f}d")
print("\ncoda/E1 = mean backscatter level below the backwall echo")
print("fabric  = rms of the modulation explained by the real grains (dB)")
print("resid   = rms of everything else: speckle + numerics (dB)")
