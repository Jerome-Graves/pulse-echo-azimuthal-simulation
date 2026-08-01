"""Unplanned but necessary: the residual codas of DIFFERENT tessellations
appear to correlate with each other at r ~ 0.55.  Nothing physical is shared
between them (different seeds, different fabrics), so that would be a
common-mode azimuthal artefact of the rotation/rasterisation, and it would
contaminate every coda-vs-azimuth test in this project.

Quantify it, find its harmonic content, and re-run the geometry test after
projecting out a LEAVE-ONE-OUT estimate of the common mode.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_core import (SWEEPS, OUT, build_cached, measure, preds,  # noqa
                     strip)

AZ30 = np.arange(0, 360, 12)      # exactly present in EVERY sweep, no interp

D = {}
for name, ppw, kap, axis, seed, iso, ras in SWEEPS:
    rots, coda = measure(os.path.join(OUT, name))
    lab, ax, sd, h = build_cached(ppw, kap, axis, seed)
    D[name] = dict(rots=rots, coda=coda, seed=seed, ras=ras, ppw=ppw,
                   lab=lab, ax=ax, sd=sd, h=h)
    miss = [a for a in AZ30 if a not in set(rots.tolist())]
    if miss:
        print(f"  WARNING {name} missing azimuths {miss}")

names = [s[0] for s in SWEEPS]
Y = np.zeros((len(names), len(AZ30)))
for i, nm in enumerate(names):
    d = D[nm]
    idx = [int(np.where(d["rots"] == a)[0][0]) for a in AZ30]
    Y[i] = strip(d["coda"][idx], AZ30)

print("=== A. CODA-CODA CROSS-CORRELATION, exact common 30-azimuth grid ===")
print("   (residual: mean/2-theta/4-theta stripped).  seeds in brackets")
C = np.corrcoef(Y)
print(f"   {'':<20}" + "".join(f"{nm[:8]:>9}" for nm in names))
for i, nm in enumerate(names):
    print(f"   {nm+' ['+str(D[nm]['seed'])+']':<20}"
          + "".join(f"{C[i,j]:>9.3f}" for j in range(len(names))))

iu = np.triu_indices(len(names), 1)
same = np.array([D[names[i]]["seed"] == D[names[j]]["seed"]
                 for i, j in zip(*iu)])
ras = np.array([D[names[i]]["ras"] == D[names[j]]["ras"]
                for i, j in zip(*iu)])
v = C[iu]
print(f"\n   pairs sharing a tessellation seed : n={same.sum():2d}  "
      f"mean r = {v[same].mean():+.3f}  range [{v[same].min():+.3f},"
      f"{v[same].max():+.3f}]")
print(f"   pairs with DIFFERENT tessellations: n={(~same).sum():2d}  "
      f"mean r = {v[~same].mean():+.3f}  range [{v[~same].min():+.3f},"
      f"{v[~same].max():+.3f}]")
for tag, m in (("same raster", ras), ("diff raster", ~ras)):
    sel = (~same) & m
    if sel.sum():
        print(f"     different tessellation, {tag:<11}: n={sel.sum():2d}  "
              f"mean r = {v[sel].mean():+.3f}")

# circular-shift null for the mean cross-tessellation correlation
n = len(AZ30)
obs = v[~same].mean()
null = []
for k in range(1, n):
    Yr = np.array([np.roll(Y[i], (i * k) % n) for i in range(len(names))])
    Cr = np.corrcoef(Yr)
    null.append(Cr[iu][~same].mean())
null = np.array(null)
print(f"   circular-shift null for that mean: {null.mean():+.3f} "
      f"+- {null.std():.3f};  observed {obs:+.3f}, "
      f"p = {np.mean(np.abs(null) >= abs(obs)):.3f}")

# ---------------------------------------------------------------- B
print("\n=== B. HARMONIC CONTENT of the cross-tessellation common mode ===")
IND = ["rigid_seed11", "kappa8_seed17", "oos_seed23", "iso_gcal"]
sel = [names.index(x) for x in IND]
Z = Y[sel] / Y[sel].std(axis=1, keepdims=True)
cm = Z.mean(axis=0)
F = np.abs(np.fft.rfft(cm)) ** 2
F /= F.sum()
print("   power fraction of the common mode by wavenumber k:")
print("   " + "".join(f"{'k=%d' % k:>8}" for k in range(1, 13)))
print("   " + "".join(f"{F[k] if k < len(F) else 0:>8.3f}"
                      for k in range(1, 13)))
print(f"   common mode explains {np.mean([np.corrcoef(cm,z)[0,1]**2 for z in Z])*100:.0f}%"
      f" of each independent sweep's residual coda variance (mean r^2)")

# ---------------------------------------------------------------- C
print("\n=== C. GEOMETRY TEST AFTER REMOVING A LEAVE-ONE-OUT COMMON MODE ===")
print("   common mode estimated from the OTHER sweeps only (no circularity)")
print(f"   {'sweep':<18}{'seed':>5}{'r before':>10}{'r after':>9}"
      f"{'shift p':>9}")
for i, nm in enumerate(names):
    d = D[nm]
    idx = [int(np.where(d["rots"] == a)[0][0]) for a in AZ30]
    y = Y[i]
    others = [j for j in range(len(names))
              if D[names[j]]["seed"] != d["seed"]]
    cm_i = np.mean([Y[j] / Y[j].std() for j in others], axis=0)
    P = preds(d["lab"], d["ax"], d["sd"], d["h"], AZ30)
    x = strip(P["geom_only"], AZ30)
    r0 = np.corrcoef(x, y)[0, 1]
    # project the leave-one-out common mode out of BOTH series
    def proj(a, b):
        return a - b * (a @ b) / (b @ b)
    ya, xa = proj(y, cm_i), proj(x, cm_i)
    r1 = np.corrcoef(xa, ya)[0, 1]
    rs = [abs(np.corrcoef(xa, np.roll(ya, k))[0, 1]) for k in range(len(AZ30))]
    print(f"   {nm:<18}{d['seed']:>5}{r0:>10.3f}{r1:>9.3f}"
          f"{np.mean(np.array(rs) >= abs(r1)):>9.3f}")
