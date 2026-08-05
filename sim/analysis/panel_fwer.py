"""D. Properly matched nulls + the arithmetic check on the headline."""
import os, sys, json
import numpy as np
from scipy.stats import t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import observable_panel as P
from observable_panel import load_sweep, fabric_pred, CACHE

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "out", "sweeps")
P.CODA = (24e-6, 36e-6)

def zs(x):
    s = x.std(); return (x-x.mean())/(s if s > 0 else 1.0)

cfgA, rotA, XA, _, _ = load_sweep("girdle_seed11_ppw6_axis_perp")       # ppw6, fabric in plane
cfgB, rotB, XB, _, _ = load_sweep("girdle_seed11_ppw6_axis_par")        # ppw6, SAME config, no in-plane fabric
cfg8, rot8, X8, _, _ = load_sweep("girdle_seed11_ppw8_dev")
predA, _ = fabric_pred(cfgA, rotA)
pred8, _ = fabric_pred(cfg8, rot8)
keys = sorted(k for k in XA if not k.startswith("_"))
assert np.array_equal(rotA, rotB)
zpA = zs(predA)

print("="*100)
print("D1. MATCHED-SPECIMEN NULL for the ppw6 headline (lvl_rms r=+0.693).")
print("    Null = girdle_seed11_ppw6_axis_par: identical ppw/az/seed/conc/rasterise/fd_kh_max,")
print("    girdle normal along z so the TRUE in-plane E[R^2] is flat.")
print("    Null draws = its 22 observables x 30 distinct cyclic rotations vs the")
print("    girdle_seed11_ppw6_axis_perp predictor (rotation is an exact fabric-phase rotation).")
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
print("D2. FAMILY-WISE ERROR RATE OF THE LARGEST |r|, CIRCULAR-SHIFT MAX STATISTIC")
print("    Recomputed here from the sweeps on disc. The superseded 0.300 was printed")
print("    by a module pointed at an unfinished 33-azimuth snapshot of the ppw8 run;")
print("    the completed 60-azimuth sweep is used below.")
print("="*100)

def fw_cyc(Xd, predd, kk):
    """observed max |r| over the panel, and its FWER under the shift null."""
    Z = np.column_stack([zs(Xd[k]) for k in kk]); zp = zs(predd)
    n = len(zp); nd = n // 2          # predictor 180 deg periodic
    obs = np.abs(Z.T @ zp / n)
    cyc = np.array([[float(np.mean(np.roll(Z[:, j], s)*zp)) for j in range(Z.shape[1])]
                    for s in range(nd)])
    mx = np.abs(cyc).max(1)
    return obs.max(), float(np.mean(mx >= obs.max()-1e-12)), float(np.median(mx)), nd

print("  %-22s %5s %7s %9s %9s %10s" % ("sweep", "n", "shifts", "max|r|", "p_fw", "med null"))
for lab, Xd, predd in (("girdle_seed11_ppw6_axis_perp (ppw6)", XA, predA),
                       ("girdle_seed11_ppw8_dev", X8, pred8)):
    mo, pf, md, nd = fw_cyc(Xd, predd, keys)
    print("  %-22s %5d %7d %9.3f %9.4f %10.3f   observed max %s the median chance max"
          % (lab, len(predd), nd, mo, pf, md, "exceeds" if mo > md else "is below"))

print()
print("="*100)
print("D3. THE CYCLIC-SHIFT TEST IS A PHASE-ONLY TEST (conditional on A2)")
print("="*100)
def A2(z, rot):
    a = np.radians(rot.astype(float))
    D = np.column_stack([np.ones_like(a), np.cos(2*a), np.sin(2*a)])
    c = np.linalg.pinv(D) @ z
    return np.hypot(c[1], c[2])
ND = len(rot8)//2      # predictor is 180 deg periodic -> n/2 distinct alignments
print("  ppw8, the COMPLETE n=%d full circle (%d distinct alignments). A2 of the" % (len(rot8), ND))
print("  observable is invariant under cyclic shift, so the cyclic null only")
print("  randomises the PHASE. Check invariance and the r-range:")
for k in ("t_centroid", "decay", "early_late", "lvl_early", "lvl_rms"):
    z = zs(X8[k])
    a0 = A2(z, rot8); a5 = A2(np.roll(z, 7), rot8)
    rs = np.array([float(np.mean(np.roll(z, s)*zs(pred8))) for s in range(ND)])
    r0 = float(np.mean(z*zs(pred8)))
    print("  %-12s A2=%.3f (shifted %.3f)  r_obs=%+.3f  shift-null r in [%+.3f,%+.3f] rank %d/%d"
          % (k, a0, a5, r0, rs.min(), rs.max(), int(np.sum(np.abs(rs) >= abs(r0))), ND))
print("  => a large 2-theta amplitude cannot ever be 'significant' under this null;")
print("     only its phase can be. The unconditional information is discarded.")

print()
print("="*100)
print("D4. UNCONDITIONAL 2-THETA TEST vs the CONFIG-MATCHED ppw6 null (girdle_seed11_ppw6_axis_par)")
print("="*100)
nullA2 = np.array([A2(zs(XB[k]), rotB) for k in keys])
print("  girdle_seed11_ppw6_axis_par A2 across 22 observables: median %.3f  90th %.3f  95th %.3f max %.3f"
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
print("  ppw6 girdle_seed11_ppw6_axis_perp, n=60:")
for k in ("lvl_rms", "lvl_early", "f_centroid", "decay"):
    r = float(np.corrcoef(XA[k], predA)[0, 1])
    n1, n2 = neff_theirs(XA[k]), neff_bartlett(XA[k], predA)
    def pp(ne):
        tt = abs(r)*np.sqrt(max(ne-2, 1))/np.sqrt(max(1-r*r, 1e-9))
        return 2*(1-tdist.cdf(tt, max(ne-2, 1)))
    print("  %-12s r=%+.3f | n_eff(theirs)=%5.1f p=%.4f | n_eff(Bartlett)=%5.1f p=%.5f"
          % (k, r, n1, pp(n1), n2, pp(n2)))
