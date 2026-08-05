"""Part 4: independent verification via the Stroh sextic at oblique incidence.

Completely different code path from exact_rc.py: instead of Christoffel
eigenvectors at n, this solves the sextic for the vertical slowness s3 at
fixed horizontal slowness s1, selects up/down-going roots by energy flux,
and matches displacement + traction. At s1 -> 0 it must reproduce the
normal-incidence answer.
"""
import os
import sys
import numpy as np
from scipy.linalg import eig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # siblings
from exact_rc import setup, exact_normal_RT, RHO, v2t, banner  # noqa: E402


def modes(C6, rho, s1):
    """All six Stroh solutions for horizontal slowness s1 (sagittal plane).

    Returns list of (s3, u(3,), t(3,), P3) with t the traction on x3=const
    divided by i*omega*U, and P3 the x3 energy flux (sign only matters).
    """
    T4 = v2t(C6)
    Q = T4[:, 0, :, 0] * s1 ** 2
    Rm = T4[:, 0, :, 2] * s1
    Tm = T4[:, 2, :, 2]
    I = np.eye(3)
    A = np.zeros((6, 6)); B = np.zeros((6, 6))
    A[0:3, 3:6] = I
    A[3:6, 0:3] = -(Q - rho * I)
    A[3:6, 3:6] = -(Rm + Rm.T)
    B[0:3, 0:3] = I
    B[3:6, 3:6] = Tm
    w, V = eig(A, B)
    out = []
    for s3, y in zip(w, V.T):
        u = y[0:3]
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            continue
        u = u / nu
        t = (Rm.T + s3 * Tm) @ u
        P3 = np.real(np.vdot(u, t))          # ~ 0.5 w^2 |U|^2 * this
        out.append((complex(s3), u, t, float(P3)))
    return out


def pick(ms, up):
    """Select the three modes leaving the interface.

    up=True  -> into medium B (+x3): propagating with P3>0, evanescent Im>0
    up=False -> into medium A (-x3): propagating with P3<0, evanescent Im<0
    """
    sel = []
    for s3, u, t, P3 in ms:
        if abs(s3.imag) > 1e-9 * max(1.0, abs(s3.real)):
            ok = (s3.imag > 0) if up else (s3.imag < 0)
        else:
            ok = (P3 > 0) if up else (P3 < 0)
        if ok:
            sel.append((s3, u, t, P3))
    return sel


def oblique_RT(CA, CB, rho, theta_deg):
    """Exact qP->qP reflection at incidence angle theta (deg) in medium A."""
    # incident qP: phase direction (sin th, 0, cos th) in the interface frame
    th = np.radians(theta_deg)
    n = np.array([np.sin(th), 0.0, np.cos(th)])
    G = np.einsum("ijkl,j,l->ik", v2t(CA), n, n)
    ev, P = np.linalg.eigh(0.5 * (G + G.T))
    v = np.sqrt(ev[-1] / rho)                     # qP
    s1 = np.sin(th) / v
    mA, mB = modes(CA, rho, s1), modes(CB, rho, s1)
    inc = [m for m in pick(mA, True)]
    # incident is the fastest down-going (qP)
    inc = sorted(inc, key=lambda m: abs(m[0].real))[0]
    dn = pick(mA, False)
    tr = pick(mB, True)
    if len(dn) != 3 or len(tr) != 3:
        raise RuntimeError(f"root selection failed: {len(dn)},{len(tr)}")
    dn = sorted(dn, key=lambda m: abs(m[0].real))
    tr = sorted(tr, key=lambda m: abs(m[0].real))
    M = np.zeros((6, 6), complex)
    for j, (s3, u, t, _) in enumerate(dn):
        M[0:3, j] = u
        M[3:6, j] = t
    for j, (s3, u, t, _) in enumerate(tr):
        M[0:3, 3 + j] = -u
        M[3:6, 3 + j] = -t
    rhs = np.concatenate([-inc[1], -inc[2]])
    x = np.linalg.solve(M, rhs)
    return x[0], np.abs(M @ x - rhs).max(), inc, dn, tr


banner("9.  INDEPENDENT CHECK: STROH SEXTIC vs CHRISTOFFEL MATCHING")
print("Two independent formulations of the same exact problem, at theta -> 0.")
print(f"{'psi_a/psi_b':>12}{'|R| Christoffel':>18}{'|R| Stroh(1e-6 deg)':>22}"
      f"{'rel.diff':>12}")
CASES = [(0, 20), (0, 35), (0, 51), (0, 90), (90, 51), (0, 27)]
for pa, pb in CASES:
    CA, CB, *_ = setup(pa, pb)
    r1 = abs(exact_normal_RT(CA, CB, RHO)['R_by']['qP'])
    r2, res, *_ = oblique_RT(CA, CB, RHO, 1e-6)
    print(f"{pa}/{pb:<9}{r1:>18.12f}{abs(r2):>22.12f}"
          f"{abs(abs(r2)-r1)/r1:>12.2e}")

banner("10.  R_PP AGAINST INCIDENCE ANGLE (the finite aperture)")
print("The 6.35 mm element at 40 mm range is NOT a plane wave: its far-field")
print("half-angle to the first null is asin(0.61*lambda/a) ~ 23 deg.")
print(f"\n{'psi_a/psi_b':>12}" + "".join(f"{t:>11}" for t in
      [0, 5, 10, 15, 20, 25]) + f"{'0->20 deg':>12}")
for pa, pb in CASES:
    CA, CB, *_ = setup(pa, pb)
    vals = []
    for t in [0, 5, 10, 15, 20, 25]:
        r, _, *_ = oblique_RT(CA, CB, RHO, max(t, 1e-6))
        vals.append(abs(r))
    ch = 20 * np.log10(vals[4] / vals[0])
    print(f"{pa}/{pb:<9}" + "".join(f"{v:>11.6f}" for v in vals)
          + f"{ch:>11.2f}dB")
