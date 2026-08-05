"""Tilted-interface testbed: measures staircase error with NO analytic solver.

THE TRICK. A point source and a PLANE reflector give an image source at 2d
along the plane normal, so the echo returns to the source at t = 2d/v for
ANY plane orientation.  Therefore:

  * put the monopole source at a FIXED grid location (identically
    discretised at every tilt),
  * put the interface at distance d with normal n(theta),
  * rotate BOTH c-axes by the same R(theta) so the angle between the
    propagation direction and each c-axis is preserved.

The physics is then invariant under theta by rigid rotation.  The ONLY
thing that changes is how the plane is represented on the grid.  So

    error(theta) = 20 log10( amp(theta) / amp(0) )

is PURE discretisation error, with theta = 0 (plane on a cell face, no
staircase at all) as exact truth.  No analytic anisotropic reflection
coefficient is needed, which is what made this test awkward before.

A-B DESIGN.  At each theta we run the bicrystal AND a homogeneous
reference (both halves = grain A).  Their difference isolates the
interface echo and cancels the source signature, the sponge and any
direct-field discretisation to first order.

TREATMENTS (selected by --treat):
  naive : nearest-neighbour labelling, cell centre decides   [current code]
  aa    : anti-aliased, stiffness linear in the cell's volume fraction
  sm    : Schoenberg-Muir equivalent medium using the exact plane normal
Each is realised as EXTRA MATERIAL LABELS, one per volume-fraction bin, so
nothing in the CUDA kernel changes.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
import fdtd                                            # noqa: E402
from ringfwi import anisotropy as an                   # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
C_REF, F0, ORDER, KHM = 3850.0, 2.0e6, 8, 2.0
# L 30->36 mm and sponge 12->20 cells: at the old size the sponge's inner
# edge returned at ~5.8 us, only 1.3 us after the echo, which put the
# echo-to-noise ratio at 1.9 for RSG and 5-11 for SSG. Bigger domain plus a
# wider, stronger absorber pushes that return clear of the gate.
L, DEPTH, SPONGE = 0.036, 0.008, 20
NBIN = 21                                              # volume-fraction bins
SUP = 4                                                # supersampling per axis


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def voigt6(C):
    """dict of Cij -> 6x6 for a single orientation (scalars)."""
    M = np.zeros((6, 6))
    for I in range(1, 7):
        for J in range(1, 7):
            M[I - 1, J - 1] = float(C[f"C{min(I,J)}{max(I,J)}"])
    return M


def grain_C(axis):
    """6x6 stiffness of an ice grain with the given c-axis (unit vector)."""
    C, _ = an.polycrystal_stiffness_3d(np.zeros((1, 1, 1), np.int32),
                                       np.asarray([axis], float),
                                       c_couplant=1500.0, rho_couplant=300.0)
    return voigt6({k: np.asarray(v).ravel()[0] for k, v in C.items()})


# ---- Schoenberg-Muir / Backus average for layering normal to x3 ----------
_N = [2, 3, 4]          # Voigt indices 3,4,5 -> the "normal" group
_T = [0, 1, 5]          # Voigt indices 1,2,6 -> the "tangential" group


def sm_average(Cs, fs):
    """Schoenberg & Muir (1989) equivalent medium, layering normal to x3."""
    inn = np.zeros((3, 3))
    tn_nn = np.zeros((3, 3))
    tt_res = np.zeros((3, 3))
    nn_nt = np.zeros((3, 3))
    for C, f in zip(Cs, fs):
        Cnn = C[np.ix_(_N, _N)]
        Ctn = C[np.ix_(_T, _N)]
        Ctt = C[np.ix_(_T, _T)]
        Ci = np.linalg.inv(Cnn)
        inn += f * Ci
        tn_nn += f * (Ctn @ Ci)
        nn_nt += f * (Ci @ Ctn.T)
        tt_res += f * (Ctt - Ctn @ Ci @ Ctn.T)
    Cnn_e = np.linalg.inv(inn)
    Ctn_e = tn_nn @ Cnn_e
    Ctt_e = tt_res + tn_nn @ Cnn_e @ nn_nt
    E = np.zeros((6, 6))
    E[np.ix_(_N, _N)] = Cnn_e
    E[np.ix_(_T, _N)] = Ctn_e
    E[np.ix_(_N, _T)] = Ctn_e.T
    E[np.ix_(_T, _T)] = Ctt_e
    return 0.5 * (E + E.T)


def bond(R):
    """6x6 Bond matrix for rotating a Voigt stiffness by R."""
    M = np.zeros((6, 6))
    p = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for a, (i, j) in enumerate(p):
        for b, (k, l) in enumerate(p):
            if b < 3:
                M[a, b] = R[i, k] * R[j, k]
            else:
                m, n = p[b]
                M[a, b] = R[i, m] * R[j, n] + R[i, n] * R[j, m]
    return M


def rot_C(C, R):
    B = bond(R)
    return B @ C @ B.T


def frac_field(shape, h, nrm, x0, sup=SUP):
    """volume fraction of the NEAR half-space (n.(x-x0) < 0) in each cell."""
    nz, ny, nx = shape
    o = (np.arange(sup) + 0.5) / sup - 0.5
    acc = np.zeros(shape, np.float32)
    for dz in o:
        for dy in o:
            for dx in o:
                z = (np.arange(nz) + 0.5 + dz) * h
                y = (np.arange(ny) + 0.5 + dy) * h
                x = (np.arange(nx) + 0.5 + dx) * h
                s = (nrm[2] * (z - x0[2])[:, None, None]
                     + nrm[1] * (y - x0[1])[None, :, None]
                     + nrm[0] * (x - x0[0])[None, None, :])
                acc += (s < 0).astype(np.float32)
    return acc / sup ** 3


def build_tables(CA, CB, nrm, treat):
    """(n_bin,36) stiffness table + rho table for the chosen treatment."""
    f = np.linspace(0.0, 1.0, NBIN)              # fraction of A
    out = np.zeros((NBIN, 36), np.float32)
    # rotate so the plane normal is along x3, average there, rotate back
    a = np.array([0.0, 0.0, 1.0])
    v = np.cross(nrm, a)
    s, c = np.linalg.norm(v), float(nrm @ a)
    if s < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / s ** 2)
    CAr, CBr = rot_C(CA, R), rot_C(CB, R)
    for i, fa in enumerate(f):
        if treat == "naive":
            E = CA if fa > 0.5 else CB
        elif treat == "aa":
            E = fa * CA + (1.0 - fa) * CB
        elif treat == "sm":
            E = rot_C(sm_average([CAr, CBr], [fa, 1.0 - fa]), R.T)
        else:
            raise ValueError(treat)
        out[i] = np.asarray(E, np.float32).reshape(-1)
    # forward_fused_labels does mats = lab + 1, so row 0 is the label -1
    # material. No cell carries -1 here, but the row must exist and must
    # have a non-zero density or safe_dt divides by zero.
    out = np.vstack([out[:1], out])
    rho = np.full(NBIN + 1, float(an.ICE_MATERIAL["rho"]), np.float32)
    return out, rho


def run(theta_deg, ppw, treat, psi_a, psi_b, homog=False, dt_in=None):
    th = np.radians(theta_deg)
    R = rot_y(th)
    nrm = R @ np.array([1.0, 0.0, 0.0])
    ca = R @ np.array([np.cos(np.radians(psi_a)), 0.0,
                       np.sin(np.radians(psi_a))])
    cb = R @ np.array([np.cos(np.radians(psi_b)), 0.0,
                       np.sin(np.radians(psi_b))])
    CA, CB = grain_C(ca), grain_C(cb if not homog else ca)

    h = C_REF / F0 / ppw
    n = int(round(L / h))
    m = SPONGE + 3
    N = n + 2 * m
    ctr = np.array([N * h / 2] * 3)
    # Snap the plane to a cell FACE at theta = 0, so that case is EXACTLY
    # staircase-free: every cell is then purely A or purely B, the volume
    # fraction is exactly 0 or 1, and all three treatments must return the
    # identical answer. (Snapping to round(DEPTH/h)*h put the plane through
    # cell CENTRES for odd N, leaving a half-filled layer that naive snaps
    # and the treatments smear - which made the reference case differ by
    # treatment and destroyed the comparison.)
    d = round((ctr[0] + DEPTH) / h) * h - ctr[0]
    x0 = ctr + d * nrm

    fr = frac_field((N, N, N), h, nrm, x0)
    lab = np.rint(fr * (NBIN - 1)).astype(np.int32)
    Ct, rho_t = build_tables(CA, CB, nrm, treat)

    co = fdtd.optimised_coeffs(ORDER, kh_max=KHM, multistart=True)
    # the bicrystal and its homogeneous reference MUST share dt, or their
    # traces have different lengths and different sampling and the A-B
    # subtraction is meaningless
    dt = dt_in if dt_in is not None else fdtd.safe_dt_labels(
        (lab + 1).astype(np.uint8), Ct, rho_t, h, co, safety=0.5)
    nt = int(7.0e-6 / dt)
    wav = fdtd.ricker(F0, dt, nt)
    c = N // 2
    src = [((c, c, c), 1.0)]
    tr = np.asarray(fdtd.forward_fused_labels(
        lab, None, h, dt, nt, src, wav, [([(c, c, c)], np.array([1.0]))],
        order=ORDER, coeffs=co, sponge_width=SPONGE,
        mat_tables=(Ct, rho_t)), float).ravel()
    return tr, dt, d


def vA(psi_a):
    """qP speed of the NEAR grain along the beam. The propagation direction
    and that grain's c-axis are both rotated by R(theta), so the angle
    between them stays psi_a and this is tilt-independent. Using C_REF here
    instead put the gate centre 0.2 us late at psi_a = 0, where the true
    speed is 4046 m/s rather than 3850."""
    return float(an.ice_qp_vs_caxis(np.radians(psi_a)))


def echo_amp(tr_bi, tr_hom, dt, d, v, half_us=0.5):
    e = np.abs(hilbert(tr_bi - tr_hom))
    fs = 1.0 / dt
    t0 = 2 * d / v + 1.2 / F0
    w = int(half_us * 1e-6 * fs)
    i0 = int(t0 * fs)
    return e[max(i0 - w, 0):i0 + w].max()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--treat", default="naive",
                    choices=["naive", "aa", "sm"])
    ap.add_argument("--ppw", type=float, nargs="+", default=[8.0])
    ap.add_argument("--tilts", type=float, nargs="+",
                    default=[0, 15, 22.5, 30, 45])
    ap.add_argument("--psi", type=float, nargs=2, default=[0.0, 51.0],
                    help="c-axis angles of the two grains, in the "
                         "pre-rotation frame")
    a = ap.parse_args()
    print(f"treatment {a.treat}   c-axes {a.psi[0]:.0f}/{a.psi[1]:.0f} deg\n")
    print(f"{'ppw':>5}{'tilt':>8}{'echo':>12}{'vs tilt0 (dB)':>16}{'s':>7}")
    for ppw in a.ppw:
        base = None
        for t in a.tilts:
            t0 = time.time()
            bi, dt, d = run(t, ppw, a.treat, a.psi[0], a.psi[1])
            ho, _, _ = run(t, ppw, a.treat, a.psi[0], a.psi[1],
                           homog=True, dt_in=dt)
            amp = echo_amp(bi, ho, dt, d, vA(a.psi[0]))
            if base is None:
                base = amp
            print(f"{ppw:>5.0f}{t:>8.1f}{amp:>12.4e}"
                  f"{20*np.log10(amp/base):>15.2f} {time.time()-t0:>6.0f}")
