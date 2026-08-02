"""Step 12: is the scalar impedance-jump reflection coefficient
(v_2 - v_1)/(v_2 + v_1) the right one for an ice/ice grain boundary?

The FDTD solves the full anisotropic elastic problem, in which the qP
polarisation is not exactly longitudinal, so a normal-incidence P wave
generates BOTH a reflected qP and two reflected qS.  Solve the exact
6x6 interface problem (displacement + traction continuity for the three
Christoffel modes each side) with the SAME stiffness tensors the solver
uses, and compare |R_PP| with the scalar formula over the real boundary
population.  Pure CPU linear algebra - no CUDA, no solver import.
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
from ringfwi import anisotropy as an                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
G = np.load(os.path.join(HERE, "po_src_geom.npz"))
AXES, FI, FJ, FA = G["axes"], G["fi"], G["fj"], G["farea"]

ids = np.arange(-1, len(AXES), dtype=np.int32).reshape(-1, 1, 1)
Cd, rho = an.polycrystal_stiffness_3d(ids, AXES)
NM = len(AXES) + 1
Cfull = np.zeros((NM, 6, 6))
for I in range(1, 7):
    for J in range(1, 7):
        a = np.asarray(Cd[f"C{min(I,J)}{max(I,J)}"], float)
        Cfull[:, I - 1, J - 1] = np.broadcast_to(a, (NM, 1, 1)).reshape(-1)
RHO = np.array(np.broadcast_to(np.asarray(rho, float),
                               (NM, 1, 1)).reshape(-1))
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def c4(m):
    C = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VOIGT):
        for b, (k, l) in enumerate(VOIGT):
            v = Cfull[m, a, b]
            for ii, jj in ((i, j), (j, i)):
                for kk, ll in ((k, l), (l, k)):
                    C[ii, jj, kk, ll] = v
    return C


C4 = [c4(m) for m in range(NM)]


def modes(m, n):
    """Christoffel: speeds and polarisations along n, sorted qP first."""
    Gm = np.einsum("ijkl,j,l->ik", C4[m], n, n)
    w, V = np.linalg.eigh(Gm)
    v = np.sqrt(np.maximum(w, 0) / RHO[m])
    o = np.argsort(v)[::-1]
    return v[o], V[:, o]


def r_pp(m1, m2, n):
    """Exact normal-incidence P->P amplitude reflection coefficient."""
    v1, P1 = modes(m1, n)
    v2, P2 = modes(m2, n)
    A = np.zeros((6, 6))
    b = np.zeros(6)
    for a in range(3):                       # reflected, travelling along -n
        A[0:3, a] = P1[:, a]
        A[3:6, a] = RHO[m1] * v1[a] * P1[:, a]
    for a in range(3):                       # transmitted, along +n
        A[0:3, 3 + a] = -P2[:, a]
        A[3:6, 3 + a] = RHO[m2] * v2[a] * P2[:, a]
    b[0:3] = -P1[:, 0]
    b[3:6] = RHO[m1] * v1[0] * P1[:, 0]
    x = np.linalg.solve(A, b)
    return x[0], x[1:3], v1[0], v2[0]


# --- sanity: two ISOTROPIC media must give (Z2-Z1)/(Z2+Z1)
_Cs, _R4, _RH = Cfull.copy(), list(C4), RHO.copy()
for m, (vp, vs, rr) in enumerate([(4000.0, 2000.0, 917.0),
                                  (4200.0, 2100.0, 917.0)]):
    lam = rr * (vp ** 2 - 2 * vs ** 2)
    mu = rr * vs ** 2
    M = np.zeros((6, 6))
    M[:3, :3] = lam
    M[0, 0] = M[1, 1] = M[2, 2] = lam + 2 * mu
    M[3, 3] = M[4, 4] = M[5, 5] = mu
    Cfull[m] = M
    C4[m] = c4(m)
    RHO[m] = rr
nn = np.array([0.3, 0.5, np.sqrt(1 - 0.09 - 0.25)])
rp, rs, v1, v2 = r_pp(0, 1, nn)
Z1, Z2 = 917.0 * 4000.0, 917.0 * 4200.0
print(f"isotropic sanity: |R_PP| exact {abs(rp):.6f}  vs "
      f"|(Z2-Z1)/(Z2+Z1)| {abs((Z2-Z1)/(Z2+Z1)):.6f}   "
      f"|R_PS| {np.linalg.norm(rs):.2e} (must be 0)\n")
Cfull, C4, RHO = _Cs, _R4, _RH

rng = np.random.default_rng(1)
sel = rng.choice(len(FI), 250, replace=False)
az = np.radians(rng.uniform(0, 360, 250))
ex, sc, sh = [], [], []
for k, a in zip(sel, az):
    n = np.array([np.cos(a), np.sin(a), 0.0])
    m1, m2 = int(FI[k]) + 1, int(FJ[k]) + 1
    rp, rs, v1, v2 = r_pp(m1, m2, n)
    ex.append(abs(rp))
    sc.append(abs((v2 - v1) / (v2 + v1)))
    sh.append(np.linalg.norm(rs))
ex, sc, sh = np.array(ex), np.array(sc), np.array(sh)
print("normal-incidence reflection at real ice/ice grain boundaries")
print(f"  n = {len(ex)} (boundary, look-direction) pairs")
print(f"  exact  |R_PP| : rms {np.sqrt((ex**2).mean())*100:.4f} %  "
      f"median {np.median(ex)*100:.4f} %")
print(f"  scalar |dv/2v|: rms {np.sqrt((sc**2).mean())*100:.4f} %  "
      f"median {np.median(sc)*100:.4f} %")
print(f"  ratio of rms (exact/scalar) = "
      f"{np.sqrt((ex**2).mean()/(sc**2).mean()):.4f}  "
      f"-> {20*np.log10(np.sqrt((ex**2).mean()/(sc**2).mean())):+.2f} dB "
      f"on the coda level")
print(f"  reflected qS at normal incidence: rms "
      f"{np.sqrt((sh**2).mean())*100:.4f} % "
      f"({np.sqrt((sh**2).mean()/(ex**2).mean()):.2f} x the qP reflection "
      f"in amplitude)")
print(f"  worst-case per-pair ratio exact/scalar: "
      f"{np.percentile(ex/np.maximum(sc,1e-9),[5,50,95]).round(3)}")

# --- how much PRESSURE does a returning qS wave make?  The receiver
#     records -(s1+s2+s3)/3, and in a rotated TI crystal a quasi-shear
#     wave carries a non-zero stress trace, so it is NOT invisible.
print("\npressure visibility of the converted (qS) return")


def pfac(m, n, p, v):
    """|trace(sigma)| per unit displacement amplitude, /(i omega)."""
    return abs(np.einsum("kkjl,j,l->", C4[m], p, n)) / v


pp_amp, ps_amp = [], []
for k, a in zip(sel, az):
    n = np.array([np.cos(a), np.sin(a), 0.0])
    m1 = int(FI[k]) + 1
    rp, rs, v1, v2 = r_pp(m1, int(FJ[k]) + 1, n)
    vv, PP = modes(m1, n)
    pp_amp.append(abs(rp) * pfac(m1, n, PP[:, 0], vv[0]))
    ps_amp.append(np.hypot(abs(rs[0]) * pfac(m1, n, PP[:, 1], vv[1]),
                           abs(rs[1]) * pfac(m1, n, PP[:, 2], vv[2])))
pp_amp, ps_amp = np.array(pp_amp), np.array(ps_amp)
pr = ps_amp / np.maximum(pp_amp, 1e-30)
rat = np.sqrt((ps_amp ** 2).mean() / (pp_amp ** 2).mean())
print(f"  ENSEMBLE ratio of recorded pressure, P->S channel / P->P "
      f"channel = {rat:.4f}  ({20*np.log10(rat):+.1f} dB)")
print(f"    per-pair ratio percentiles 5/50/95: "
      f"{np.percentile(pr,[5,50,95]).round(4)}")
print(f"  -> the P->S channel returns "
      f"{20*np.log10(rat):+.1f} dB of pressure relative to the "
      f"P->P channel that physical optics models,")
print(f"     and it arrives from boundaries at 31-47 mm rather than "
      f"46-69 mm, i.e. INSIDE the same 24-36 us gate.")
vs = np.array([modes(int(FI[k]) + 1,
                     np.array([np.cos(a), np.sin(a), 0.0]))[0][1:].mean()
               for k, a in zip(sel[:60], az[:60])])
print(f"  mean quasi-shear speed {vs.mean():.0f} m/s, Vp/Vs = "
      f"{3850/vs.mean():.2f}")
