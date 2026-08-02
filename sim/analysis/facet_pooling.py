"""Final arithmetic: pooling, power, and the model-free tessellation test.

The awkward fact this has to confront: the three INDEPENDENT tessellations
(17, 23, 41) exist only at ppw6 with double rasterisation / legacy FD, and
on seed 11 the geometry correlation is itself attenuated from 0.465 (ppw8,
single raster) to 0.164 (ppw6, double raster).  So the honest question is
not "is 0.465 excluded elsewhere" (it is) but "is 0.164 excluded elsewhere".
"""
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facet_predictors import (SWEEPS, OUT, build_cached, measure, preds,  # noqa
                     strip)

INFO = {s[0]: dict(ppw=s[1], kap=s[2], ax=s[3], seed=s[4], iso=s[5],
                   ras=s[6]) for s in SWEEPS}
R = {}
for nm, i in INFO.items():
    rots, coda = measure(os.path.join(OUT, nm))
    lab, ax, sd, h = build_cached(i["ppw"], i["kap"], i["ax"], i["seed"])
    P = preds(lab, ax, sd, h, rots)
    R[nm] = dict(rots=rots, coda=coda,
                 r=np.corrcoef(strip(P["geom_only"], rots),
                               strip(coda, rots))[0, 1], n=len(rots))

R465 = R["girdle_perp_ppw8"]["r"]
R164 = R["rigid_seed11"]["r"]
print("=== reference effect sizes (geom_only, residual space) ===")
print(f"   seed 11, ppw8, single raster  (girdle_perp_ppw8) r = {R465:.3f}")
print(f"   seed 11, ppw6, DOUBLE raster  (rigid_seed11)     r = {R164:.3f}")
print(f"   attenuation factor from the ppw6/double numerics: "
      f"{R164/R465:.2f}")

IND = ["kappa8_seed17", "oos_seed23", "iso_gcal"]
print(f"\n   independent tessellations (all ppw6 double raster):")
for nm in IND:
    print(f"     {nm:<16} seed {INFO[nm]['seed']:>2}  r = {R[nm]['r']:+.3f}")

print("\n=== POOLED ONE-SIDED TESTS over the 3 independent tessellations ===")
z = np.array([np.arctanh(R[nm]["r"]) for nm in IND])
for ne in (25.0, 36.0, 60.0, 90.0):
    se = 1 / np.sqrt(ne - 3) / np.sqrt(len(z))
    zm = z.mean()
    lo, hi = np.tanh(zm - 1.96 * se), np.tanh(zm + 1.96 * se)
    out = [f"   n_eff/sweep={ne:>5.0f}  pooled r = {np.tanh(zm):+.3f} "
           f"[{lo:+.3f},{hi:+.3f}]"]
    for tgt, lab in ((R465, "0.465 (ppw8)"), (R164, "0.164 (ppw6-double)")):
        p = norm.cdf((zm - np.arctanh(tgt)) / se)
        out.append(f"  p(rho<{lab})={p:.4f}")
    print("".join(out))

print("\n=== POWER of each independent sweep to detect r = 0.164 ===")
print("   (two-sided alpha=0.05, Fisher z)")
for nm in IND:
    for ne in (36.0, 90.0):
        s = 1 / np.sqrt(ne - 3)
        pw = norm.cdf(np.arctanh(R164) / s - 1.96) + \
            norm.cdf(-np.arctanh(R164) / s - 1.96)
        print(f"   {nm:<16} n={R[nm]['n']:>3} n_eff={ne:>4.0f}  "
              f"power = {pw:.2f}")
        break
ne = 36.0
s = 1 / np.sqrt(ne - 3)
for tgt in (0.164, 0.30, 0.465):
    pw = norm.cdf(np.arctanh(tgt) / s - 1.96)
    print(f"   single sweep, n_eff=36: power to detect r={tgt:.3f} "
          f"is {pw:.2f}")
pw3 = norm.cdf(np.arctanh(0.164) / (s / np.sqrt(3)) - 1.96)
print(f"   POOLED over the 3, n_eff=36 each: power at r=0.164 is {pw3:.2f}")

print("\n=== MODEL-FREE TEST (no descriptor at all) ===")
print("   The hypothesis says the coda is set by grain-boundary geometry.")
print("   Then two sweeps sharing a TESSELLATION but with DIFFERENT fabric")
print("   must have strongly correlated coda-vs-azimuth curves.")
pairs = [("girdle_perp", "girdle_par"), ("girdle_perp", "singlemax_ppw8"),
         ("girdle_par", "singlemax_ppw8"), ("girdle_par", "girdle_perp_ppw8"),
         ("girdle_perp_ppw8", "singlemax_ppw8")]
vals = []
print(f"   {'same tessellation (seed 11), different fabric':<48}"
      f"{'n':>4}{'r':>8}{'p':>8}")
for a, b in pairs:
    ca = set(R[a]["rots"].tolist()) & set(R[b]["rots"].tolist())
    ca = np.array(sorted(ca))
    ia = [int(np.where(R[a]["rots"] == x)[0][0]) for x in ca]
    ib = [int(np.where(R[b]["rots"] == x)[0][0]) for x in ca]
    x, y = strip(R[a]["coda"][ia], ca), strip(R[b]["coda"][ib], ca)
    r0 = np.corrcoef(x, y)[0, 1]
    rs = np.array([abs(np.corrcoef(x, np.roll(y, k))[0, 1])
                   for k in range(len(ca))])
    vals.append(r0)
    print(f"   {a+' / '+b:<48}{len(ca):>4}{r0:>8.3f}"
          f"{np.mean(rs >= abs(r0)):>8.3f}")
print(f"   mean r = {np.mean(vals):+.3f}  -> shared variance "
      f"{100*np.mean(vals):.0f}% (r itself is the shared-variance fraction "
      f"when the shared term is common)")
print("   the hypothesis ('fabric is a small second-order term') predicts")
print("   these should be near 1.  They are not.")
print("   NOTE: there is NO different-tessellation single-raster pair in")
print("   existence, so the matched negative baseline cannot be formed.")
