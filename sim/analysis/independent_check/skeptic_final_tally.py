"""Final independent tally: the 4 INDEPENDENT tessellations, all statistics."""
import os as _os
import sys as _sys
# helper modules shared with sibling directories
for _d in ('..',):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import os, numpy as np
from scipy.stats import rankdata
import _skeptic_lib as S
CANDS=[11,17,23,41]+list(range(100,140))
tgrid=np.arange(12e-6,48e-6,S.TBIN)
SW=[("girdle_seed11_ppw8_dev",11,1),("singlemax_seed11_ppw8_twin",11,1),("girdle_seed11_ppw6_axis_par",11,1),
    ("singlemax_seed17_ppw6_kappa8",17,1),("singlemax_seed23_ppw6_heldout_axis",23,4),("isotropic_seed41_ppw6_calibration",41,4),("singlemax_seed11_ppw6_rigid2",11,4)]
def perazim(A,B):
    A=A-A.mean(1,keepdims=True); B=B-B.mean(1,keepdims=True)
    na=np.linalg.norm(A,axis=1); nb=np.linalg.norm(B,axis=1); ok=(na>1e-12)&(nb>1e-12)
    return float(np.mean((A[ok]*B[ok]).sum(1)/(na[ok]*nb[ok])))
def rev(M): return np.roll(M[::-1],1,axis=0)
print(f"{'sweep':<18}{'own':>4}{'gate':>9}{'r':>8}{'rank':>7}{'p':>8}{'perazim r':>10}{'rank':>6}"
      f"{'r_rev':>8}{'shift p':>9}")
rows=[]
for nm,own,sub in SW:
    f=f"skP_{nm}_{sub}_1.npz"
    if not os.path.exists(f): print(f"{nm:<18} MISSING"); continue
    az,E,t1,dt=S.measured(nm,tgrid,sub)
    z=np.load(f); P={int(k[1:]):z[k] for k in z.files if k.startswith('s')}
    for g in [(12,48),(24,36)]:
        sel=(tgrid>=g[0]*1e-6)&(tgrid<=g[1]*1e-6); Em=E[:,sel]
        r={c:S.corr(Em,P[c][:,sel]) for c in CANDS}
        v=np.array([r[c] for c in CANDS]); oth=np.array([r[c] for c in CANDS if c!=own])
        rank=int(np.sum(v>=r[own])); p=(1+np.sum(oth>=r[own]))/44
        pa={c:perazim(Em,P[c][:,sel]) for c in CANDS}
        va=np.array([pa[c] for c in CANDS]); ranka=int(np.sum(va>=pa[own]))
        rr=S.corr(Em,rev(P[own])[:,sel])
        rs=np.array([S.corr(np.roll(Em,k,axis=0),P[own][:,sel]) for k in range(1,len(az))])
        ps=(1+np.sum(rs>=r[own]))/len(az)
        print(f"{nm:<18}{own:>4}{str(g):>9}{r[own]:>8.3f}{rank:>4}/44{p:>8.4f}"
              f"{pa[own]:>10.3f}{ranka:>3}/44{rr:>8.3f}{ps:>9.4f}",flush=True)
        rows.append((nm,own,g,rank,p,ranka,ps))
np.save("skeptic_final_tally.npy",np.array(rows,dtype=object),allow_pickle=True)
