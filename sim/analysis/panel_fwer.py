"""D. Properly matched nulls + the arithmetic check on the headline."""
import os, sys, json
import numpy as np
from scipy.stats import t as tdist
sys.path.insert(0, r"C:\Users\Jerome\AppData\Local\Temp\claude\C--Users-Jerome\72aa31c0-c0c1-48de-881e-9470fe03e8ba\scratchpad")
import observable_panel as P
from observable_panel import load_sweep, fabric_pred, CACHE

OUT = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
SNAP = os.path.join(CACHE, "snap_girdle_perp_ppw8")
P.CODA = (24e-6, 36e-6)

def zs(x):
    s = x.std(); return (x-x.mean())/(s if s > 0 else 1.0)

cfgA, rotA, XA, _, _ = load_sweep("girdle_perp")       # ppw6, fabric in plane
cfgB, rotB, XB, _, _ = load_sweep("girdle_par")        # ppw6, SAME config, no in-plane fabric
cfg8, rot8, X8, _, _ = load_sweep("girdle_perp_ppw8", SNAP)
predA, _ = fabric_pred(cfgA, rotA)
pred8, _ = fabric_pred(cfg8, rot8)
keys = sorted(k for k in XA if not k.startswith("_"))
assert np.array_equal(rotA, rotB)
zpA = zs(predA)

print("="*100)
print("D1. MATCHED-SPECIMEN NULL for the ppw6 headline (lvl_rms r=+0.693).")
print("    Null = girdle_par: identical ppw/az/seed/conc/rasterise/fd_kh_max,")
print("    girdle normal along z so the TRUE in-plane E[R^2] is flat.")
print("    Null draws = its 22 observables x 30 distinct cyclic rotations vs the")
print("    girdle_perp predictor (rotation is an exact fabric-phase rotation).")
print("="*100)
null = []
for k in keys:
    z = zs(XB[k])
    for s in range(30):
        null.append(float(np.mean(np.roll(z, s)*zpA)))
null = np.abs(np.array(null))
print("  null |r| draws: %d   median %.3f  90th %.3f  95th %.3f  99th %.3f  max %.3f"
      % (len(null), np.median(null), *[np.quantile(null, q) for q in (.90, .95, .99)], null.max()))
print()
print("  %-14s %8s %10s %12s" % ("observable", "r(perp)", "p_matched", "p_matched(own-obs null)"))
for k in ("lvl_rms", "lvl_early", "f_centroid", "lvl_late", "inst_freq", "early_late", "decay", "t_centroid"):
    r = float(np.corrcoef(XA[k], predA)[0, 1])
    p = float((np.abs(null) >= abs(r)).mean())
    own = np.abs([float(np.mean(np.roll(zs(XB[k]), s)*zpA)) for s in range(30)])
    print("  %-14s %+8.3f %10.4f %12.4f" % (k, r, p, float((own >= abs(r)).mean())))
print("  (the same-observable null has only 30 draws -> floor 1/30 = 0.033)")

print()
print("="*100)
print("D2. ARITHMETIC CHECK ON THE HEADLINE SENTENCE")
print("="*100)
print("  reported: 'the largest effect (|r|=0.717) is SMALLER than the median of")
print("  the chance maximum (0.599)'   ->  0.717 > 0.599 is %s" % (0.717 > 0.599))
print("  their own final.py prints FWER p = 0.300, i.e. the observed max sits at the")
print("  70th percentile of the null max distribution, ABOVE the median. The prose")
print("  and the computed p-value contradict each other.")

print()
print("="*100)
print("D3. THE CYCLIC-SHIFT TEST IS A PHASE-ONLY TEST (conditional on A2)")
print("="*100)
def A2(z, rot):
    a = np.radians(rot.astype(float))
    D = np.column_stack([np.ones_like(a), np.cos(2*a), np.sin(2*a)])
    c = np.linalg.pinv(D) @ z
    return np.hypot(c[1], c[2])
m8 = rot8 <= 174
print("  ppw8, n=30. A2 of the observable is invariant under cyclic shift, so the")
print("  cyclic null only randomises the PHASE. Check invariance and the r-range:")
for k in ("t_centroid", "decay", "early_late", "lvl_early", "lvl_rms"):
    z = zs(X8[k][m8])
    a0 = A2(z, rot8[m8]); a5 = A2(np.roll(z, 7), rot8[m8])
    rs = np.array([float(np.mean(np.roll(z, s)*zs(pred8[m8]))) for s in range(30)])
    r0 = float(np.mean(z*zs(pred8[m8])))
    print("  %-12s A2=%.3f (shifted %.3f)  r_obs=%+.3f  shift-null r in [%+.3f,%+.3f] rank %d/30"
          % (k, a0, a5, r0, rs.min(), rs.max(), int(np.sum(np.abs(rs) >= abs(r0)))))
print("  => a large 2-theta amplitude cannot ever be 'significant' under this null;")
print("     only its phase can be. The unconditional information is discarded.")

print()
print("="*100)
print("D4. UNCONDITIONAL 2-THETA TEST vs the CONFIG-MATCHED ppw6 null (girdle_par)")
print("="*100)
nullA2 = np.array([A2(zs(XB[k]), rotB) for k in keys])
print("  girdle_par A2 across 22 observables: median %.3f  90th %.3f  95th %.3f max %.3f"
      % (np.median(nullA2), np.quantile(nullA2, .9), np.quantile(nullA2, .95), nullA2.max()))
obsA2 = np.array([A2(zs(XA[k]), rotA) for k in keys])
o = np.argsort(-obsA2)
for j in o[:6]:
    print("  %-14s A2=%.3f  exceeds %d/22 of the matched null (p=%.3f)"
          % (keys[j], obsA2[j], int((nullA2 < obsA2[j]).sum()), float((nullA2 >= obsA2[j]).mean())))

print()
print("="*100)
print("D5. BARTLETT n_eff (uses BOTH series' autocorrelation) vs their one-series n_eff")
print("="*100)
def ac_circ(z):
    n = len(z); F = np.fft.rfft(z); a = np.fft.irfft(np.abs(F)**2, n); return a/a[0]
def neff_theirs(y):
    z = zs(y); n = len(z); ac = ac_circ(z)
    zc = np.argmax(ac < 0) if np.any(ac[:n//2] < 0) else n//2
    return n/max(1+2*ac[1:zc].sum(), 1.0)
def neff_bartlett(y, p):
    zy, zp = zs(y), zs(p); n = len(zy)
    ay, ap = ac_circ(zy), ac_circ(zp)
    L = n//2
    s = 1 + 2*np.sum(ay[1:L]*ap[1:L])
    return n/max(s, 1.0)
print("  ppw6 girdle_perp, n=60:")
for k in ("lvl_rms", "lvl_early", "f_centroid", "decay"):
    r = float(np.corrcoef(XA[k], predA)[0, 1])
    n1, n2 = neff_theirs(XA[k]), neff_bartlett(XA[k], predA)
    def pp(ne):
        tt = abs(r)*np.sqrt(max(ne-2, 1))/np.sqrt(max(1-r*r, 1e-9))
        return 2*(1-tdist.cdf(tt, max(ne-2, 1)))
    print("  %-12s r=%+.3f | n_eff(theirs)=%5.1f p=%.4f | n_eff(Bartlett)=%5.1f p=%.5f"
          % (k, r, n1, pp(n1), n2, pp(n2)))
