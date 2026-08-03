import os as _os
import sys as _sys
# helper modules shared with sibling directories
for _d in ('..',):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import numpy as np, os
import observable_panel as P
from observable_panel import load_sweep, fabric_pred, CACHE
from panel_run import zs, design, harm_amp, n_distinct_shifts

def cyc_one_sided(y, pred, half=True):
    """one-sided cyclic-shift p: fraction of distinct rotations with r >= r_obs."""
    z, zp = zs(y), zs(pred); n=len(z)
    r0=float(np.mean(z*zp)); s=np.sign(r0) or 1.0
    nd = n//2 if half else n            # predictor has period 180 -> n/2 distinct
    rs=np.array([float(np.mean(np.roll(z,k)*zp)) for k in range(nd)])
    return r0, float(np.sum(s*rs >= s*r0-1e-12)/nd), nd

def neff(y):
    z=zs(y); n=len(z)
    F=np.fft.rfft(z); ac=np.fft.irfft(np.abs(F)**2,n); ac/=ac[0]
    zc=np.argmax(ac<0) if np.any(ac[:n//2]<0) else n//2
    return n/max(1+2*ac[1:zc].sum(),1.0)

print("RE-EXAMINATION OF THE PREVIOUSLY REPORTED ppw6 RESULT (coda level, girdle_perp)")
P.CODA=(24e-6,36e-6)
cfg,rot,X,bw,dt = load_sweep("girdle_perp")
pred,_=fabric_pred(cfg,rot)
for k in ("lvl_rms","lvl_early","f_centroid","decay"):
    r,p,nd = cyc_one_sided(X[k],pred)
    ne=neff(X[k])
    from scipy.stats import t as tdist
    tt=abs(r)*np.sqrt(max(ne-2,1))/np.sqrt(max(1-r*r,1e-9))
    pp=2*(1-tdist.cdf(tt,max(ne-2,1)))
    print("  %-12s n=%d  r=%+.3f | naive-perm p<1e-4 | cyclic(1-sided,%d rot) p=%.3f | n_eff=%.1f -> parametric p=%.3f"
          %(k,len(rot),r,nd,p,ne,pp))

print("\nSAME TEST ON THE ppw8 SWEEP (the COMPLETE 60-azimuth full circle, az 0-354),")
print("pre-registered 24-36 us window.  A cyclic shift is a rigid rotation of the")
print("specimen only on a uniformly sampled full circle, so no half-circle subset.")
cfg8,rot8,X8,_,_ = load_sweep("girdle_perp_ppw8")
nd0 = n_distinct_shifts(rot8)
pred8,_=fabric_pred(cfg8,rot8)
keys=sorted(k for k in X8 if not k.startswith("_"))
from scipy.stats import t as tdist
print("  %-14s %8s %10s %10s %9s %10s"%("observable","r","p_cyc1","n_eff","p_param","p_holm(22)"))
res=[]
for k in keys:
    y=X8[k]; r,p,nd=cyc_one_sided(y,pred8); ne=neff(y)
    tt=abs(r)*np.sqrt(max(ne-2,1))/np.sqrt(max(1-r*r,1e-9)); pp=2*(1-tdist.cdf(tt,max(ne-2,1)))
    res.append((k,r,p,ne,pp))
res.sort(key=lambda z:-abs(z[1]))
pv=np.array([x[4] for x in res]); mN=len(pv); o=np.argsort(pv); hol=np.empty(mN); run=0
for i,idx in enumerate(o):
    run=max(run,(mN-i)*pv[idx]); hol[idx]=min(run,1)
for (k,r,p,ne,pp),h in zip(res,hol):
    print("  %-14s %+8.3f %10.3f %10.1f %9.3f %10.3f"%(k,r,p,ne,pp,h))
print("\n  (p_cyc1 floor with %d distinct rotations = %.3f)"%(nd0,1.0/nd0))
print("  Holm across the 22 pre-registered observables at the single 24-36 us window.")
print("  With %d distinct alignments the smallest attainable raw p is %.3f, so no"%(nd0,1.0/nd0))
print("  Holm-adjusted p over 22 observables can fall below 22/%d = %.3f."%(nd0,22.0/nd0))
