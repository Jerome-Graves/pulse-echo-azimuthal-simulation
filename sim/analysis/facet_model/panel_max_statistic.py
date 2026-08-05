"""TEST 1, part 2: proper statistics on the all-tessellation panel.

Fixes the structural problem in the Holm run: with a circular-shift null the
p floor is 1/n, so a family of 24 tests on n=60 sweeps CANNOT produce a
Holm-adjusted p below 24/60 = 0.4.  Here the family-wise correction is done
with a MAX-STATISTIC permutation instead: one common fractional rotation is
applied to every sweep at once (360 fractions), so the null preserves both
the azimuthal autocorrelation within a sweep and the dependence between
sweeps that share a tessellation.  Family-wise p floor 1/360.

Also:
  * full azimuthal resolution (no subsampling) wherever available
  * rotation-sign diagnostic on every sweep (rules out a per-sweep
    convention bug as the cause of the nulls)
  * correct-vs-wrong tessellation rank test
  * equivalence / power test: is the seed-11 effect size EXCLUDED elsewhere?
"""
import os as _os
import sys as _sys
# helper modules shared with sibling directories
for _d in ('..',):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facet_predictors import (SWEEPS, OUT, build_cached, measure, preds,  # noqa
                     shift_p, strip, neff, fisher_ci)

FRAC = 360


def rload(name, ppw, kap, axis, seed):
    rots, coda = measure(os.path.join(OUT, name))
    lab, ax, sd, h = build_cached(ppw, kap, axis, seed)
    return rots, coda, lab, ax, sd, h


D = {}
for name, ppw, kap, axis, seed, iso, ras in SWEEPS:
    rots, coda, lab, ax, sd, h = rload(name, ppw, kap, axis, seed)
    P = preds(lab, ax, sd, h, rots, sign=+1.0)
    M = preds(lab, ax, sd, h, rots, sign=-1.0)
    D[name] = dict(rots=rots, coda=coda, P=P, M=M, seed=seed, iso=iso,
                   ppw=ppw, ras=ras, lab=lab, ax=ax, sd=sd, h=h)
    print(f"{name:<18} FULL n={len(rots)}", flush=True)

KEYS3 = ("geom_only", "born_spec", "born_iso")

# ------------------------------------------------ full-resolution panel
print("\n=== 1b. FULL-RESOLUTION PANEL (no subsampling) ===")
print(f"{'sweep':<18}{'seed':>5}{'n':>5}  {'predictor':<10}"
      f"{'r_res':>8}{'p_shift':>9}{'  95%CI':>16}{'  maxr_shift':>12}")
CELLS = []
for name, *_ in SWEEPS:
    d = D[name]
    ne, _ = neff(strip(d["coda"], d["rots"]))
    yc = strip(d["coda"], d["rots"])
    for k in KEYS3:
        xc = strip(d["P"][k], d["rots"])
        n = len(yc)
        rs = np.array([np.corrcoef(xc, np.roll(yc, i))[0, 1]
                       for i in range(n)])
        r0 = rs[0]
        p = float(np.sum(np.abs(rs) >= abs(r0)) / n)
        lo, hi = fisher_ci(r0, ne)
        CELLS.append(dict(sweep=name, pred=k, r=r0, p=p, ne=ne,
                          seed=d["seed"], iso=d["iso"], n=n))
        print(f"{name:<18}{d['seed']:>5}{n:>5}  {k:<10}{r0:>8.3f}{p:>9.4f}"
              f"   [{lo:+.2f},{hi:+.2f}]{np.abs(rs).max():>12.3f}"
              f"{'   ISO' if d['iso'] else ''}")
    print()

# ------------------------------------- 2. max-statistic family-wise null
print("=== 2. MAX-STATISTIC FAMILY-WISE CORRECTION ===")
print(f"   family = {len(CELLS)} cells ({len(KEYS3)} predictors x "
      f"{len(SWEEPS)} sweeps); {FRAC} common fractional rotations")
obs = np.array([c["r"] for c in CELLS])
nullmax = np.zeros(FRAC)
for t, f in enumerate(np.arange(FRAC) / FRAC):
    vals = []
    for c in CELLS:
        d = D[c["sweep"]]
        yc = strip(d["coda"], d["rots"])
        xc = strip(d["P"][c["pred"]], d["rots"])
        k = int(round(f * len(yc))) % len(yc)
        vals.append(abs(np.corrcoef(xc, np.roll(yc, k))[0, 1]))
    nullmax[t] = max(vals)
print(f"   null max|r| distribution: median {np.median(nullmax):.3f}  "
      f"95th {np.percentile(nullmax,95):.3f}  max {nullmax.max():.3f}")
print(f"   {'sweep':<18}{'predictor':<11}{'r_res':>8}{'p_FWE':>8}")
order = np.argsort(-np.abs(obs))
for i in order:
    c = CELLS[i]
    pf = float(np.sum(nullmax >= abs(c["r"])) / FRAC)
    print(f"   {c['sweep']:<18}{c['pred']:<11}{c['r']:>8.3f}{pf:>8.4f}"
          f"{' *' if pf < 0.05 else ''}")

# --------------------------------------------- 3. rotation-sign diagnostic
print("\n=== 3. ROTATION-SIGN DIAGNOSTIC (geom_only) ===")
print("   if a null sweep were merely mis-registered, sign -1 would rescue it")
print(f"   {'sweep':<18}{'r(+1)':>8}{'r(-1)':>8}{'max|r| over shifts, +1':>26}")
for name, *_ in SWEEPS:
    d = D[name]
    yc = strip(d["coda"], d["rots"])
    rp = np.corrcoef(strip(d["P"]["geom_only"], d["rots"]), yc)[0, 1]
    rm = np.corrcoef(strip(d["M"]["geom_only"], d["rots"]), yc)[0, 1]
    xc = strip(d["P"]["geom_only"], d["rots"])
    mx = max(abs(np.corrcoef(xc, np.roll(yc, i))[0, 1])
             for i in range(len(yc)))
    print(f"   {name:<18}{rp:>8.3f}{rm:>8.3f}{mx:>26.3f}")

# ------------------------------- 4. correct-vs-wrong tessellation rank test
print("\n=== 4. CORRECT-vs-WRONG TESSELLATION RANK TEST (geom_only) ===")
GEO = {11: ("singlemax_seed11_ppw6_rigid2", 6.0, 3.93, (0.866, 0.5, 0.0)),
       17: ("singlemax_seed17_ppw6_kappa8", 6.0, 8.00, (0.5, 0.866, 0.0)),
       23: ("singlemax_seed23_ppw6_heldout_axis", 6.0, 3.93, (-0.342, 0.94, 0.0)),
       41: ("isotropic_seed41_ppw6_calibration", 6.0, 0.001, (1.0, 0.0, 0.0))}
GB = {}
for s, (nm, ppw, kap, axis) in GEO.items():
    GB[s] = build_cached(ppw, kap, axis, s)
print(f"   {'coda sweep':<18}{'true':>6}" +
      "".join(f"{'s%d' % s:>9}" for s in GEO) + f"{'  rank':>7}")
wins, tot = 0, 0
for name, *_ in SWEEPS:
    d = D[name]
    yc = strip(d["coda"], d["rots"])
    rr = {}
    for s in GEO:
        lab, ax, sd, h = GB[s]
        Pi = preds(lab, ax, sd, h, d["rots"])
        rr[s] = np.corrcoef(strip(Pi["geom_only"], d["rots"]), yc)[0, 1]
    ordr = sorted(GEO, key=lambda s: -abs(rr[s]))
    rk = ordr.index(d["seed"]) + 1
    wins += (rk == 1)
    tot += 1
    print(f"   {name:<18}{d['seed']:>6}" +
          "".join(f"{rr[s]:>9.3f}" for s in GEO) + f"{rk:>7}")
print(f"   correct tessellation ranked 1st in {wins}/{tot} sweeps "
      f"(chance = {tot/4:.1f}/{tot})")

# ------------------------------------------- 5. equivalence / power test
print("\n=== 5. IS THE SEED-11 EFFECT EXCLUDED ELSEWHERE? ===")
R11 = 0.465
print(f"   H0: rho = {R11} (the seed-11 ppw8 geom_only value).")
print(f"   {'sweep':<18}{'seed':>5}{'r_res':>8}{'n_eff':>7}"
      f"{'z':>8}{'p(one-sided rho<%.2f)' % R11:>24}")
from scipy.stats import norm  # noqa: E402
for name, *_ in SWEEPS:
    d = D[name]
    c = [x for x in CELLS if x["sweep"] == name
         and x["pred"] == "geom_only"][0]
    ne = c["ne"]
    if ne <= 3:
        continue
    z = (np.arctanh(c["r"]) - np.arctanh(R11)) * np.sqrt(ne - 3)
    print(f"   {name:<18}{d['seed']:>5}{c['r']:>8.3f}{ne:>7.1f}"
          f"{z:>8.2f}{norm.cdf(z):>24.4f}")
