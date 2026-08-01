import os,sys,numpy as np, sk_lib as S
CANDS=[11,17,23,41]+list(range(100,140))
tgrid=np.arange(12e-6,48e-6,S.TBIN)
nm,own,sub,spec = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4]))
key=f"skP_{nm}_{sub}_{int(spec)}.npz"
az,E,t1,dt=S.measured(nm,tgrid,sub)
c_az=2.0*S.T.DIA/t1
if not os.path.exists(key):
    P={}
    for c in CANDS:
        sd,wt=S.cand_sw(c); P[c]=S.predict(az,sd,wt,c_az,tgrid,spec)
    sd,wt=S.cand_sw(own); Pm=S.predict(-az,sd,wt,c_az,tgrid,spec)
    np.savez_compressed(key,mirror=Pm,**{f"s{c}":P[c] for c in CANDS})
print("done",key,flush=True)
