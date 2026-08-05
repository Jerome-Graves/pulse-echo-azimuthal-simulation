"""Edge 3: how tightly does this ensemble hold grain size fixed?

Grain diameters straight from the cached production label volumes, so
this is the same microstructure the solver saw.
"""
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
TESS = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "out", "tesscache")))
SEEDS = [11, 7, 17, 23, 41, 53, 71, 89]
E = np.load(os.path.join(HERE, "edge23.npz"))

rows = []
for s in SEEDS:
    with np.load(os.path.join(TESS, "tess_s%d_p8_k-8.npz" % s)) as z:
        lab = np.asarray(z["labels"])
        h = float(z["h"])
        nax = len(np.asarray(z["axes"]))
    live = lab[lab >= 0]
    cells = np.bincount(live.ravel(), minlength=nax)
    v = cells[cells > 0] * h ** 3
    d = 2.0 * (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0)
    dv_eq = 2.0 * (3.0 * v.sum() / len(v) / (4.0 * np.pi)) ** (1.0 / 3.0)
    rows.append(dict(seed=s, n=len(v), h=h,
                     d_mean=float(d.mean()) * 1e3,
                     d_vol_eq=float(dv_eq) * 1e3,
                     d_sd=float(d.std(ddof=1)) * 1e3,
                     vol_cm3=float(v.sum()) * 1e6))

lam = 3850.0 / 2.0e6
print("GRAIN SIZE ACROSS THE EIGHT GIRDLE TESSELLATIONS")
print("wavelength at 2 MHz and 3850 m/s: %.3f mm" % (lam * 1e3))
print("  %4s %5s %10s %12s %10s %10s %9s"
      % ("seed", "ngr", "d_mean_mm", "d_vol_eq_mm", "d_sd_mm", "vol_cm3",
         "D/lambda"))
for r in rows:
    print("  %4d %5d %10.3f %12.3f %10.3f %10.2f %9.2f"
          % (r["seed"], r["n"], r["d_mean"], r["d_vol_eq"], r["d_sd"],
             r["vol_cm3"], r["d_vol_eq"] * 1e-3 / lam))

for k in ("d_mean", "d_vol_eq", "n"):
    v = np.array([r[k] for r in rows], float)
    print("  %-10s %9.4f +- %.4f   spread %.2f %% of the mean, "
          "full range %.2f %%"
          % (k, v.mean(), v.std(ddof=1),
             100 * v.std(ddof=1) / v.mean(),
             100 * (v.max() - v.min()) / v.mean()))

print()
print("does the crossing count track grain size within this ensemble?")
for k in ("d_mean", "d_vol_eq", "n"):
    v = np.array([r[k] for r in rows], float)
    r1, p1 = stats.pearsonr(v, E["cross"])
    r2, p2 = stats.pearsonr(v, E["level"])
    print("  %-10s -> n_cross r %+.3f p %.3f | -> coda level r %+.3f "
          "p %.3f" % (k, r1, p1, r2, p2))
print()
print("n_cross varies by %.1f %% of its mean while the grain diameter")
print("varies by %.1f %%, so within this ensemble n_cross is a")
print("realisation-specific property of where the boundaries fall along")
print("the beam, not a proxy for grain size.")
c = E["cross"]
d = np.array([r["d_vol_eq"] for r in rows])
print("  n_cross %.2f %%   d_vol_eq %.2f %%"
      % (100 * c.std(ddof=1) / c.mean(), 100 * d.std(ddof=1) / d.mean()))
