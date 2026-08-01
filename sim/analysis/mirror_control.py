"""Frame-registration control done properly: apply the azimuth-REVERSAL
operator to EVERY candidate, not just the true one.

r_fwd = corr(E, P_c) ; r_rev = corr(E, Rev P_c).
The antisymmetric part  d_c = (r_fwd - r_rev)/2  is the only piece of the
correlation that knows the specimen's handedness.  If the identification is
"the coda locates THIS specimen's boundaries at THIS azimuth", the true
candidate must win on d, not only on r_fwd.
Also: azimuth jackknife (how many azimuths is rank-1 hostage to).
"""
import numpy as np, sk_lib as S
CANDS=[11,17,23,41]+list(range(100,140))
tgrid=np.arange(12e-6,48e-6,S.TBIN)

def rev(M):   # az -> -az on a uniform 0..360 grid
    return np.roll(M[::-1], 1, axis=0)

for nm,own,sub in [("girdle_perp_ppw8",11,1),("singlemax_ppw8",11,1)]:
    az,E,t1,dt=S.measured(nm,tgrid,sub)
    assert np.allclose(np.diff(az), az[1]-az[0])
    z=np.load(f"skP_{nm}_{sub}_1.npz")
    P={int(k[1:]):z[k] for k in z.files if k.startswith('s')}
    for g in [(24,36),(12,48),(12,24),(36,48)]:
        sel=(tgrid>=g[0]*1e-6)&(tgrid<=g[1]*1e-6)
        Em=E[:,sel]
        rf={c:S.corr(Em,P[c][:,sel]) for c in CANDS}
        rr={c:S.corr(Em,rev(P[c])[:,sel]) for c in CANDS}
        d={c:0.5*(rf[c]-rr[c]) for c in CANDS}
        for lbl,dd in [("r_fwd",rf),("d_anti",d)]:
            vals=np.array([dd[c] for c in CANDS]); oth=np.array([dd[c] for c in CANDS if c!=own])
            rank=int(np.sum(vals>=dd[own])); p=(1+np.sum(oth>=dd[own]))/44
            print(f"{nm:<18}{str(g):>10}{lbl:>8} own={dd[own]:+.4f} rank {rank:2d}/44 p={p:.4f} "
                  f"nullmax={oth.max():+.4f}")
        print(f"{'':<18}{'':>10}{'rev(own)':>8} = {rr[own]:+.4f}   sym frac of r_fwd = "
              f"{0.5*(rf[own]+rr[own])/max(rf[own],1e-9):.2f}")
    # azimuth jackknife on the wide gate
    sel=(tgrid>=12e-6)&(tgrid<=48e-6); Em=E[:,sel]
    ranks=[]
    for q in range(len(az)):
        k=[i for i in range(len(az)) if i!=q]
        r={c:S.corr(Em[k],P[c][:,sel][k]) for c in CANDS}
        v=np.array([r[c] for c in CANDS]); ranks.append(int(np.sum(v>=r[own])))
    ranks=np.array(ranks)
    print(f"{nm}: wide-gate leave-one-azimuth-out rank: min {ranks.min()} max {ranks.max()} "
          f"#!=1 {int((ranks!=1).sum())}/{len(az)}")
    # single most influential azimuth
    base={c:S.corr(Em,P[c][:,sel]) for c in CANDS}
    drop=[(base[own]-S.corr(Em[[i for i in range(len(az)) if i!=q]],
            P[own][:,sel][[i for i in range(len(az)) if i!=q]]),az[q]) for q in range(len(az))]
    drop.sort(reverse=True)
    print("   biggest single-azimuth contributions (dr, az):",
          [(f"{d:+.4f}",int(a)) for d,a in drop[:4]])
    print()
