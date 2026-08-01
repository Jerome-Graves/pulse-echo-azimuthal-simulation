import numpy as np
from scipy.optimize import minimize
def taylor_sg(M):
    A=np.zeros((M,M)); b=np.zeros(M); b[0]=1.0
    for j in range(1,M+1):
        for m in range(1,M+1): A[j-1,m-1]=(2*m-1)**(2*j-1)
    return np.linalg.solve(A,b)
def K(c,kh):
    kh=np.atleast_1d(kh); s=np.zeros_like(kh,float)
    for m in range(1,len(c)+1): s+=c[m-1]*np.sin((2*m-1)*kh/2.0)
    return 2.0*s                      # = k_eff * h
def dK(c,kh):
    kh=np.atleast_1d(kh); s=np.zeros_like(kh,float)
    for m in range(1,len(c)+1): s+=c[m-1]*(2*m-1)*np.cos((2*m-1)*kh/2.0)
    return s                          # = d(k_eff h)/d(kh)

print("Do OPTIMISED (Holberg/Liu-Sen minimax) 8th-order operators give NEGATIVE group velocity?")
for khmax in [2.0,2.2,2.4,2.6,2.8,3.0]:
    kh=np.linspace(1e-6,khmax,4000)
    r=minimize(lambda c: np.max(np.abs(K(c,kh)/kh-1.0)), taylor_sg(4), method='Nelder-Mead',
               options={'xatol':1e-13,'fatol':1e-15,'maxiter':400000,'maxfev':400000})
    c=r.x; grid=np.linspace(1e-7,np.pi,500001); g=dK(c,grid)
    neg=g<0
    first = grid[np.argmax(neg)] if neg.any() else None
    tag = f"NEGATIVE from kh={first:.4f} (ppw={2*np.pi/first:.2f}), min vg/c0={g.min():.4f}" if neg.any() else f"stays >=0 (min {g.min():.2e})"
    print(f"  band kh<={khmax}: maxPhaseErr={r.fun:.2e}  ceiling 2*sum|c|={2*np.sum(np.abs(c)):.4f}  -> {tag}")

print("\nppw table, 8th-order Taylor  (kh = 2*pi/ppw)")
c4=taylor_sg(4)
print(f"{'ppw':>6} {'kh':>8} {'vp/c0':>10} {'vp err %':>10} {'vg/c0':>10} {'vg err %':>10}")
for ppw in [20,12,10,8,6,5,4.5,4,3.5,3,2.5,2.2,2.0]:
    kh=2*np.pi/ppw; vp=float(K(c4,kh)[0]/kh); vg=float(dK(c4,kh)[0])
    print(f"{ppw:>6} {kh:>8.4f} {vp:>10.6f} {100*(vp-1):>10.4f} {vg:>10.6f} {100*(vg-1):>10.4f}")

print("\nCEILING on apparent normalised wavenumber  Kmax = 2*sum_m |c_m|  (attained at kh=pi)")
for M in range(1,9):
    c=taylor_sg(M); print(f"  spatial order {2*M:2d}: Kmax = {2*np.sum(np.abs(c)):.5f}")
print(f"  spectral/exact limit: pi = {np.pi:.5f}")

print("\nWith 2nd-order leapfrog time stepping, Courant r=c*dt/h:")
print("  omega*h/c = (2/r)*arcsin( r*K(kh)/2 ) ; at kh=pi with 8th-order Taylor K=2.57262")
for r_ in [0.1,0.2,0.3,0.4,0.5]:
    print(f"   r={r_}: apparent omega*h/c ceiling = {(2.0/r_)*np.arcsin(r_*2.57262/2.0):.5f}")
