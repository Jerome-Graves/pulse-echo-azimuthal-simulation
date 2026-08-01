"""Acceptance tests against ANISOTROPIC theory, not isotropic intuition.

Every earlier check in this project compared a simulated arrival against a
phase velocity. That is the isotropic habit and it is wrong in general: a
point source radiates ENERGY, which travels at the group velocity, and in an
anisotropic medium the group velocity differs from the phase velocity in both
magnitude and direction. They coincide only where dv/dtheta = 0, which for ice
means along the c-axis and in the basal plane. Anywhere else, testing against
phase velocity builds the error into the acceptance criterion.

Ice Ih is transversely isotropic with 6.9% qP anisotropy, so at 45 degrees the
two differ by around a percent, which is the same size as the numerical
dispersion we are trying to measure. A test that cannot separate them is not a
test.

What is checked here:
  A. phase velocities from the Christoffel equation, all three modes
  B. group velocity, computed exactly, and where it does/does not equal phase
  C. simulated point-source arrivals against GROUP velocity at several angles
  D. all three modes present at their predicted times
"""
import numpy as np

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def voigt_to_tensor(C6):
    """(6,6) Voigt matrix -> (3,3,3,3) stiffness tensor."""
    C = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(VOIGT):
        for J, (k, l) in enumerate(VOIGT):
            v = C6[I, J]
            for (a, b) in ((i, j), (j, i)):
                for (c, d) in ((k, l), (l, k)):
                    C[a, b, c, d] = v
    return C


def ice_voigt(axis=(0.0, 0.0, 1.0)):
    """Ice Ih stiffness in Voigt form with the c-axis along `axis`."""
    C11, C33, C44, C12, C13 = 13.93e9, 15.01e9, 3.01e9, 7.08e9, 5.77e9
    C66 = 0.5 * (C11 - C12)
    C6 = np.array([
        [C11, C12, C13, 0, 0, 0],
        [C12, C11, C13, 0, 0, 0],
        [C13, C13, C33, 0, 0, 0],
        [0, 0, 0, C44, 0, 0],
        [0, 0, 0, 0, C44, 0],
        [0, 0, 0, 0, 0, C66]], float)
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, a)
    s = np.linalg.norm(v)
    if s < 1e-12:
        R = np.eye(3) if a[2] > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        c = float(z @ a)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / s ** 2)
    T = voigt_to_tensor(C6)
    return np.einsum("ai,bj,ck,dl,ijkl->abcd", R, R, R, R, T)


def phase_velocities(Ct, rho, n):
    """All three phase velocities and polarisations for propagation along `n`.

    Solves the Christoffel eigenproblem  (C_ijkl n_j n_l) u_k = rho v^2 u_i.
    Returned fastest first: qP, then the two quasi-shear modes.
    """
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    G = np.einsum("ijkl,j,l->ik", Ct, n, n)
    w, U = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    return np.sqrt(np.maximum(w[order], 0.0) / rho), U[:, order]


def group_velocity(Ct, rho, n, mode=0):
    """Exact group-velocity VECTOR for one mode.

        Vg_i = C_ijkl n_j u_k u_l / (rho * v_phase)

    This is what a point source's energy actually follows. It equals the phase
    velocity only where the slowness surface is locally spherical.
    """
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    v, U = phase_velocities(Ct, rho, n)
    u = U[:, mode]
    Vg = np.einsum("ijkl,j,k,l->i", Ct, n, u, u) / (rho * v[mode])
    return Vg, float(v[mode])


def group_vs_phase_table(axis=(0, 0, 1), rho=917.0, angles=(0, 15, 30, 45, 60, 75, 90)):
    """Where group and phase velocity agree, and where they must not."""
    Ct = ice_voigt(axis)
    rows = []
    for th in angles:
        t = np.radians(th)
        n = np.array([np.sin(t), 0.0, np.cos(t)])       # angle from the c-axis
        Vg, vp = group_velocity(Ct, rho, n, 0)
        vg = np.linalg.norm(Vg)
        skew = np.degrees(np.arccos(np.clip(Vg @ n / vg, -1, 1)))
        rows.append(dict(angle=th, v_phase=vp, v_group=vg,
                         pct=100 * (vg - vp) / vp, skew_deg=skew))
    return rows


def expected_arrival(Ct, rho, r_vec, mode=0, n_search=2001):
    """Arrival time of `mode` energy at offset `r_vec` from a point source.

    The receiver sees the phase direction whose GROUP velocity points at it,
    which is not generally the straight line to the receiver. Found by search
    over propagation directions in the plane containing r_vec and the c-axis.
    """
    r = np.asarray(r_vec, float)
    R = np.linalg.norm(r)
    rhat = r / R
    # search directions in the plane spanned by rhat and z
    z = np.array([0.0, 0.0, 1.0])
    e2 = z - (z @ rhat) * rhat
    if np.linalg.norm(e2) < 1e-9:
        e2 = np.array([1.0, 0.0, 0.0]) - rhat[0] * rhat
    e2 /= np.linalg.norm(e2)
    best_t, best_a = np.inf, np.nan
    for a in np.linspace(-np.pi / 2, np.pi / 2, n_search):
        n = np.cos(a) * rhat + np.sin(a) * e2
        Vg, _ = group_velocity(Ct, rho, n, mode)
        proj = Vg @ rhat
        if proj <= 0:
            continue
        t = R / proj                     # time for energy to reach the receiver
        if t < best_t:
            best_t, best_a = t, a
    return best_t, np.degrees(best_a)
