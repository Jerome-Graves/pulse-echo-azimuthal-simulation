"""Exact anisotropic reflection coefficient for an ice/ice grain boundary.

CPU numpy only. No FDTD, no CUDA, no simulation.

Geometry follows tilt_testbed.py / the bicrystal validation exactly:
  interface normal  n = R_y(tilt) @ xhat
  c-axis of grain A c_a = R_y(tilt) @ (cos psi_a, 0, sin psi_a)
  c-axis of grain B c_b = R_y(tilt) @ (cos psi_b, 0, sin psi_b)
so psi is the angle between the c-axis and the interface normal, which at
normal incidence is also the angle to the propagation direction.
"""
import os
import sys
import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
from ringfwi import anisotropy as an  # noqa: E402

np.set_printoptions(precision=6, suppress=False, linewidth=140)

MAT = an.ICE_MATERIAL
RHO = float(MAT["rho"])
C0 = an.ti_stiffness_6(**MAT)          # TI, c-axis along z

V6 = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def v2t(C):
    T = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(V6):
        for b, (k, l) in enumerate(V6):
            v = C[a, b]
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    T[ii, jj, kk, ll] = v
    return T


def t2v(T):
    return np.array([[T[i, j, k, l] for (k, l) in V6] for (i, j) in V6])


def rot6(C, R):
    return t2v(np.einsum("pi,qj,rk,sl,ijkl->pqrs", R, R, R, R, v2t(C)))


def rot_z_to(a):
    """Rotation taking zhat onto unit vector a (Rodrigues)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, a)
    c = float(z @ a)
    s = np.linalg.norm(v)
    if s < 1e-14:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def crystal_C(caxis):
    """6x6 stiffness of ice Ih with c-axis along caxis, in the lab frame."""
    return rot6(C0, rot_z_to(caxis))


# ---------------------------------------------------------------- part 0
def banner(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


banner("0.  CONSTANTS AS USED (read from ringfwi.anisotropy)")
print("ICE_MATERIAL =", MAT)
print("rho =", RHO, "kg/m^3")
print("ti_stiffness_6 (GPa):\n", C0 / 1e9)
print("C66 = (C11-C12)/2 =", (MAT['C11'] - MAT['C12']) / 2 / 1e9, "GPa")
# cross-check my own Bond rotation against the project's
for ax in [(1, 0, 0), (0.6, 0.2, np.sqrt(1 - .36 - .04))]:
    d = np.abs(crystal_C(ax) - an.ti_stiffness_3d(np.array(ax, float), MAT)).max()
    print(f"  rotation cross-check axis {ax}: max |mine - ringfwi| = {d:.3e} Pa")

# qP speed on axis / off axis, cross-check against the module
for psi in (0.0, 51.0, 90.0):
    n = np.array([np.cos(np.radians(psi)), 0.0, np.sin(np.radians(psi))])
    # propagation along x, c-axis at psi -> equivalently c along z, prop at psi
    vq = an.christoffel_3d(C0, RHO, (np.sin(np.radians(psi)), 0,
                                     np.cos(np.radians(psi))))[0]
    print(f"  psi={psi:5.1f} deg  v_qP = {vq:9.3f} m/s   "
          f"(ice_qp_vs_caxis = {an.ice_qp_vs_caxis(np.radians(psi)):9.3f})")
print("  sqrt(C33/rho) =", np.sqrt(MAT['C33'] / RHO),
      "  sqrt(C11/rho) =", np.sqrt(MAT['C11'] / RHO))


# ------------------------------------------------- part 1: normal incidence
def modes_at_normal(C6, rho):
    """Christoffel modes for propagation along +z (the interface normal).

    Returns (speeds v[3], polarisations P[3,3] as columns, labels).
    Gamma_ik = C_i3k3.  Traction on the plane z=const for a wave U p
    travelling along +z is  t = +i w rho v U p ,  along -z it is
    -i w rho v U p.  So each mode is self-impedant with Z = rho v.
    """
    T = v2t(C6)
    G = T[:, 2, :, 2]
    G = 0.5 * (G + G.T)
    w, P = np.linalg.eigh(G)                 # ascending
    v = np.sqrt(w / rho)
    order = np.argsort(-v)
    v, P = v[order], P[:, order]
    # fix a sign convention: make the largest component positive
    for m in range(3):
        if P[np.argmax(np.abs(P[:, m])), m] < 0:
            P[:, m] *= -1.0
    lab = []
    nz = np.array([0.0, 0.0, 1.0])
    for m in range(3):
        p = P[:, m]
        if abs(p @ nz) > 0.5:
            lab.append("qP")
        elif abs(p[1]) > 0.5:
            lab.append("SH")
        else:
            lab.append("qSV")
    return v, P, lab


def exact_normal_RT(CA, CB, rho, incident="qP"):
    """Exact plane-wave R/T at normal incidence, 3 modes each side.

    Unknowns [R1,R2,R3,T1,T2,T3] (displacement amplitudes, U_inc = 1).
      displacement:  p_inc + sum R_m pA_m = sum T_m pB_m
      traction    :  Z_inc p_inc - sum Z^A_m R_m pA_m = sum Z^B_m T_m pB_m
    """
    vA, PA, lA = modes_at_normal(CA, rho)
    vB, PB, lB = modes_at_normal(CB, rho)
    ZA, ZB = rho * vA, rho * vB
    i0 = lA.index(incident)
    pinc, Zinc = PA[:, i0], ZA[i0]

    M = np.zeros((6, 6))
    M[0:3, 0:3] = PA
    M[0:3, 3:6] = -PB
    M[3:6, 0:3] = -PA * ZA[None, :]
    M[3:6, 3:6] = -PB * ZB[None, :]
    rhs = np.concatenate([-pinc, -Zinc * pinc])
    x = np.linalg.solve(M, rhs)
    R, T = x[:3], x[3:]

    resid = np.abs(M @ x - rhs).max()
    # energy: flux along n is 0.5 w^2 Z |U|^2
    ebal = (Zinc - (ZA * R ** 2).sum() - (ZB * T ** 2).sum()) / Zinc
    out = dict(vA=vA, vB=vB, PA=PA, PB=PB, lA=lA, lB=lB,
               R=R, T=T, resid=resid, ebal=ebal,
               R_by=dict(zip(lA, R)), T_by=dict(zip(lB, T)),
               vP_a=vA[lA.index("qP")], vP_b=vB[lB.index("qP")])
    return out


def setup(psi_a, psi_b, tilt_deg=0.0, extra_rot=None):
    """Build the two stiffnesses expressed in a frame with n along +z."""
    Rt = rot_y(np.radians(tilt_deg))
    n = Rt @ np.array([1.0, 0.0, 0.0])
    ca = Rt @ np.array([np.cos(np.radians(psi_a)), 0.0, np.sin(np.radians(psi_a))])
    cb = Rt @ np.array([np.cos(np.radians(psi_b)), 0.0, np.sin(np.radians(psi_b))])
    if extra_rot is not None:
        n, ca, cb = extra_rot @ n, extra_rot @ ca, extra_rot @ cb
    W = rot_z_to(n).T                       # takes n onto +z
    return rot6(crystal_C(ca), W), rot6(crystal_C(cb), W), n, ca, cb


def polar_tilt_deg(C6, rho):
    """Angle between the qP polarisation and +z (the propagation direction)."""
    v, P, lab = modes_at_normal(C6, rho)
    p = P[:, lab.index("qP")]
    return np.degrees(np.arccos(min(1.0, abs(p[2]))))


banner("1.  WHAT COUPLES AT NORMAL INCIDENCE")
CA, CB, n, ca, cb = setup(0.0, 51.0)
r = exact_normal_RT(CA, CB, RHO)
print("interface-frame normal is +z; both c-axes lie in the x-z plane")
print("medium A modes:", [f"{l}:{vv:.1f} m/s p={np.round(r['PA'][:,i],5)}"
                          for i, (l, vv) in enumerate(zip(r['lA'], r['vA']))])
print("medium B modes:", [f"{l}:{vv:.1f} m/s p={np.round(r['PB'][:,i],5)}"
                          for i, (l, vv) in enumerate(zip(r['lB'], r['vB']))])
print("\nR (displacement) by mode:", {k: f"{v:+.6e}" for k, v in r['R_by'].items()})
print("T (displacement) by mode:", {k: f"{v:+.6e}" for k, v in r['T_by'].items()})
print(f"linear-system residual {r['resid']:.2e}   energy imbalance {r['ebal']:.3e}")

print("\nqP polarisation tilt off the propagation direction, vs psi:")
for psi in range(0, 91, 5):
    C, _, _, _, _ = setup(psi, psi)
    print(f"   psi={psi:3d} deg   delta = {polar_tilt_deg(C, RHO):7.4f} deg")
grid = np.linspace(0, 90, 9001)
dl = np.array([polar_tilt_deg(setup(p, p)[0], RHO) for p in grid])
print(f"   maximum polarisation tilt {dl.max():.4f} deg at psi = "
      f"{grid[np.argmax(dl)]:.2f} deg")
