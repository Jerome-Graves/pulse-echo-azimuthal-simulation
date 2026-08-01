"""EMPIRICAL sigma from the real data, no surrogates and no spectral model.

rigid_seed11 has all 360 azimuths at ppw6.  gcheck_ppw8 has 12 azimuths at
step 30 of the SAME specimen at ppw8.  There are 30 distinct 12-azimuth
uniform subsets of the 360-az series (offsets 0..29).  The spread of the
known-phase projection across those 30 subsets is a direct, model-free
measurement of how noisy a 12-azimuth estimate is.
"""
import os, sys
import numpy as np
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "skep_levels.npz"))
from skep_project import ls_harm, inphase, predictor  # noqa

print()
print("=" * 92)
print("A) empirical spread of a 12-azimuth (step 30) projection, from the "
      "real 360-az series")
print("=" * 92)
for nm, phi in (("rigid_seed11", 65.0), ("oos_seed23", 212.0),
                ("iso_gcal", 158.0)):
    rot = Z[f"{nm}__rot"].astype(float)
    y = Z[f"{nm}__un"]
    full = inphase(rot, y, phi)
    vals = []
    for off in range(30):
        idx = (np.arange(12) * 30 + off) % 360
        sel = np.isin(rot, idx)
        vals.append(inphase(rot[sel], y[sel], phi))
    vals = np.array(vals)
    print(f"  {nm:<14} full-360 proj {full:+.2f} | 12-az subsets: "
          f"mean {vals.mean():+.2f}  sd {vals.std(ddof=1):.2f}  "
          f"range {vals.min():+.2f} to {vals.max():+.2f}")
print()
print("  The strand's surrogate sigma for a 12-azimuth sweep was 1.76 dB.")
print("  Compare with the sd column above (measured, same n, real data).")

print()
print("=" * 92)
print("B) does the k=2 noise floor really SATURATE with n?  Direct "
      "subsampling of iso_gcal, no model.")
print("=" * 92)
print("  For each decimation step s, take every s-th azimuth at every "
      "offset, fit mean+k2+k4, record A2.")
print(f"  {'n':>5}{'step':>6}{'mean A2':>10}{'sd A2':>8}   "
      f"{'UN-normalised':>0}      {'| normalised metric':>0}")
rot = Z["iso_gcal__rot"].astype(float)
for s in (30, 18, 12, 8, 6, 4, 3, 2, 1):
    n = 360 // s
    outs_u, outs_n = [], []
    for off in range(s):
        idx = (np.arange(n) * s + off) % 360
        sel = np.isin(rot, idx)
        ou, _, _ = ls_harm(rot[sel], Z["iso_gcal__un"][sel], (2, 4))
        on, _, _ = ls_harm(rot[sel], Z["iso_gcal__norm"][sel], (2, 4))
        outs_u.append(ou[2][0]); outs_n.append(on[2][0])
    u = np.array(outs_u); v = np.array(outs_n)
    print(f"  {n:>5}{s:>6}{u.mean():>10.2f}{u.std():>8.2f}"
          f"        |  {v.mean():>7.2f}{v.std():>8.2f}")
print()
print("  -> A2 converges to the realisation's OWN value; it does not shrink.")
print("     The strand's structural claim (a) is CORRECT.  But note what it "
      "converges TO:")
print("     un-normalised A2 -> 0.37 dB for THIS realisation, not 1.07 dB.")
print("     1.07 dB is an ACROSS-realisation floor, and it is estimated from "
      "one realisation.")

print()
print("=" * 92)
print("C) the 'sqrt(M) materialises' check is arithmetic, not evidence")
print("=" * 92)
s1, s2 = 0.85, 0.75
print(f"  strand: rigid_seed11 sigma {s1}, oos_seed23 sigma {s2}, "
      f"combined 0.56, 'factor 1.42 vs ideal 1.41'")
comb = 1.0 / np.sqrt(1 / s1 ** 2 + 1 / s2 ** 2)
print(f"  inverse-variance combination of two numbers with those sigmas "
      f"gives {comb:.3f} BY DEFINITION.")
print(f"  No data enter this. It tests nothing about whether the fabric term "
      f"is common across realisations.")
print(f"  The actual empirical content is whether the two projections AGREE: "
      f"+1.50+-0.85 vs +1.99+-0.75,")
print(f"  difference {1.99-1.50:+.2f} +- {np.hypot(s1,s2):.2f} "
      f"({(1.99-1.50)/np.hypot(s1,s2):.2f} sigma) - consistent, but that is a "
      f"weak 1-dof check.")
