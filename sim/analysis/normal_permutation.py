"""Are the 43 null tessellations exchangeable with the true one?
If candidates with more grains (denser boundaries) systematically score
higher, and the true seeds happen to be grain-rich, the 1/44 rank p is
biased.  Check n_grains for every candidate and its correlation with r."""
import numpy as np, sk_lib as S
CANDS=[11,17,23,41]+list(range(100,140))
tgrid=np.arange(12e-6,48e-6,S.TBIN)
ng={c:len(S.cand_sw(c)[0]) for c in CANDS}
v=np.array([ng[c] for c in CANDS])
print(f"n_grains over 44 candidates: min {v.min()} max {v.max()} mean {v.mean():.1f} sd {v.std():.1f}")
print("  real seeds:", {c:ng[c] for c in [11,17,23,41]})
print(f"  percentile of seed 11 among all 44: {100*np.mean(v<=ng[11]):.0f}%")
for nm,own,sub in [("girdle_perp_ppw8",11,1),("singlemax_ppw8",11,1),
                   ("kappa8_seed17",17,1),("oos_seed23",23,4)]:
    az,E,t1,dt=S.measured(nm,tgrid,sub)
    z=np.load(f"skP_{nm}_{sub}_1.npz"); P={int(k[1:]):z[k] for k in z.files if k.startswith('s')}
    sel=(tgrid>=12e-6)&(tgrid<=48e-6); Em=E[:,sel]
    r=np.array([S.corr(Em,P[c][:,sel]) for c in CANDS])
    msk=np.array([c!=own for c in CANDS])
    print(f"{nm:<18} corr(r, n_grains) over the 43 nulls = "
          f"{np.corrcoef(r[msk],v[msk])[0,1]:+.3f}   r_own={r[CANDS.index(own)]:+.3f} "
          f"(own n_grains {ng[own]})")
