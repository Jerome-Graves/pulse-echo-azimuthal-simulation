"""Does the PROCESSING manufacture the identification?

Surrogate measurements: 2-D FFT of the double-centred envelope matrix in
(azimuth, time), randomise the phases keeping Hermitian symmetry, inverse.
That preserves the full 2-D power spectrum - hence the azimuthal 1/e
correlation length, the red temporal spectrum and the harmonic content -
while destroying any link to the specimen.  Run the identical 44-candidate
identification on each surrogate and count how often the TRUE candidate
ranks 1st.  Expectation under a valid null: 1/44 = 2.27 %.
"""
import numpy as np, sk_lib as S
rng=np.random.default_rng(7)
CANDS=[11,17,23,41]+list(range(100,140))
tgrid=np.arange(12e-6,48e-6,S.TBIN)
NS=300

def surrogate(M,rng):
    F=np.fft.rfft2(M)
    ph=rng.uniform(0,2*np.pi,F.shape)
    G=np.abs(F)*np.exp(1j*ph)
    out=np.fft.irfft2(G,s=M.shape)
    return out

for nm,own,sub in [("girdle_perp_ppw8",11,1),("singlemax_ppw8",11,1)]:
    az,E,t1,dt=S.measured(nm,tgrid,sub)
    z=np.load(f"skP_{nm}_{sub}_1.npz"); P={int(k[1:]):z[k] for k in z.files if k.startswith('s')}
    for g in [(12,48),(24,36)]:
        sel=(tgrid>=g[0]*1e-6)&(tgrid<=g[1]*1e-6)
        Em=S.dcent(E[:,sel]); Pc={c:S.dcent(P[c][:,sel]) for c in CANDS}
        Pn={c:Pc[c]/np.linalg.norm(Pc[c]) for c in CANDS}
        r_true={c:float((Em*Pn[c]).sum()/np.linalg.norm(Em)) for c in CANDS}
        v=np.array([r_true[c] for c in CANDS]); rk=int(np.sum(v>=r_true[own]))
        hits=0; rks=[]
        for _ in range(NS):
            Sg=S.dcent(surrogate(Em,rng)); ns=np.linalg.norm(Sg)
            rr=np.array([float((Sg*Pn[c]).sum()/ns) for c in CANDS])
            k=int(np.sum(rr>=rr[CANDS.index(own)])); rks.append(k)
            hits+= (k==1)
        rks=np.array(rks)
        print(f"{nm:<18}{str(g):>10} true rank {rk}/44 | surrogates: rank1 in "
              f"{hits}/{NS} = {100*hits/NS:.1f}% (expect 2.3%)  median rank {np.median(rks):.0f}"
              f"  frac rank<=3 {100*np.mean(rks<=3):.1f}% (expect 6.8%)")
