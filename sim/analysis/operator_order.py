import numpy as np
from dispersion_coeffs import optimised_coeffs, taylor_coeffs, keff, dkeff
print(" ord | CFL_lim | ph.err@ppw6  ph.err@ppw3  ph.err@ppw2.5  vg@2.5 | ppw for 0.1% err")
for o in (4,6,8,10,12,16,20):
    for tag,c in (("opt",optimised_coeffs(o)),("tay",taylor_coeffs(o))):
        row=[]
        for ppw in (6,3,2.5):
            kh=2*np.pi/ppw; row.append((keff(c,kh)/kh-1)*100)
        kh=np.linspace(1e-3,3.14,6000)
        err=np.abs(keff(c,kh)/kh-1)
        ok=kh[err<1e-3]
        ppw01 = 2*np.pi/ok.max() if len(ok) else np.inf
        cl = 1/(np.sqrt(3)*np.abs(c).sum())
        print(f"{o:3d} {tag} | {cl:.4f} | {row[0]:9.4f}% {row[1]:9.4f}% {row[2]:9.4f}% "
              f"{dkeff(c,2*np.pi/2.5):6.3f} | {ppw01:.2f}")
