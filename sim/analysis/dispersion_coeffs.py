import numpy as np
from numpy.polynomial import polynomial as P

# ---- Taylor staggered-grid coefficients, 2M-th order ----
def taylor_sg(M):
    # Solve sum_m c_m * (2m-1)^(2j-1) = delta_{j1}, j=1..M  (standard SG Taylor system)
    A = np.zeros((M, M)); b = np.zeros(M); b[0] = 1.0
    for j in range(1, M+1):
        for m in range(1, M+1):
            A[j-1, m-1] = (2*m-1)**(2*j-1)
    return np.linalg.solve(A, b)

def keff_over_k(c, kh):
    """ k_eff*h = 2*sum_m c_m sin((2m-1)kh/2) ; return ratio k_eff/k """
    M = len(c)
    s = np.zeros_like(np.atleast_1d(kh), dtype=float)
    for m in range(1, M+1):
        s += c[m-1]*np.sin((2*m-1)*np.atleast_1d(kh)/2.0)
    return 2.0*s/np.atleast_1d(kh)

def keff_h(c, kh):
    M = len(c); s = np.zeros_like(np.atleast_1d(kh), dtype=float)
    for m in range(1, M+1):
        s += c[m-1]*np.sin((2*m-1)*np.atleast_1d(kh)/2.0)
    return 2.0*s

def dkeff_h_dkh(c, kh):
    M = len(c); s = np.zeros_like(np.atleast_1d(kh), dtype=float)
    for m in range(1, M+1):
        s += c[m-1]*(2*m-1)*np.cos((2*m-1)*np.atleast_1d(kh)/2.0)
    return s   # d/d(kh) of 2*sum c_m sin((2m-1)kh/2) = sum c_m (2m-1) cos(...)

for M in [1,2,3,4,5]:
    c = taylor_sg(M)
    print(f"M={M} (order {2*M}) Taylor coeffs:", np.array2string(c, precision=10))

print()
c4 = taylor_sg(4)   # 8th order
kh = np.linspace(1e-6, np.pi, 400001)

# --- SPATIAL-ONLY (dt -> 0) analysis ---
for M,label in [(4,'8th-order Taylor')]:
    c = taylor_sg(M)
    g = dkeff_h_dkh(c, kh)          # group velocity ratio vg/c0 (spatial-only, dt->0)
    ph = keff_h(c, kh)/kh           # phase velocity ratio vp/c0
    # first zero crossing of group velocity
    idx = np.where(np.diff(np.sign(g)))[0]
    print(label)
    print("  phase-vel ratio at kh=1.0,1.257(ppw5),1.6,2.0,2.55,3.0:",
          [round(float(keff_h(c,np.array([x]))[0]/x),4) for x in [1.0,2*np.pi/5,1.6,2.0,2.55,3.0]])
    print("  group-vel ratio at same kh:",
          [round(float(dkeff_h_dkh(c,np.array([x]))[0]),4) for x in [1.0,2*np.pi/5,1.6,2.0,2.55,3.0]])
    if len(idx):
        for i in idx[:4]:
            # refine
            lo,hi = kh[i], kh[i+1]
            for _ in range(200):
                mid=0.5*(lo+hi)
                if np.sign(dkeff_h_dkh(c,np.array([lo]))[0])==np.sign(dkeff_h_dkh(c,np.array([mid]))[0]): lo=mid
                else: hi=mid
            z=0.5*(lo+hi)
            print(f"  group velocity zero at kh = {z:.6f}  (ppw = 2*pi/kh = {2*np.pi/z:.4f}), vp/c0 there = {float(keff_h(c,np.array([z]))[0]/z):.4f}")
    # minimum (most negative) group velocity
    imin = np.argmin(g); print(f"  min group-vel ratio {g[imin]:.4f} at kh={kh[imin]:.4f}")
    # where keff_h is maximum (stationary point of dispersion curve)
    imax = np.argmax(keff_h(c,kh)); print(f"  max k_eff*h = {keff_h(c,kh)[imax]:.4f} at kh={kh[imax]:.4f}")
