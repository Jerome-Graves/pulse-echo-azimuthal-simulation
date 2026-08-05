"""A reference that tests the TWO-WAY path, not the source signature.

WHY THIS EXISTS. Every error in the tilt testbed is a bicrystal run
differenced against a homogeneous run at the same tilt. The differencing
is exact: the two runs share the source, the grid, the time step and the
near half-space, and differ ONLY in the stiffness of the far half, so
bi - hom IS the scattered field, with nothing approximated. What the
differencing does NOT do is remove the tilt dependence of the propagator
itself. The measured echo factorises as

    A(theta) = G(theta) * R(theta)

with G the gain of the discretised source injection and two-way
propagation along the beam direction n(theta), and R the effective
reflection strength, which carries the staircase error the test is
about. The rigid-rotation argument makes the PHYSICS invariant under
theta; it does not make the GRID invariant, and the beam direction
rotates with respect to the grid exactly as the interface does. The
existing check compares hom_peak, the maximum of the homogeneous trace,
which occurs at the source cell and is the source signature: it bounds a
tilt dependence of injection at t ~ 0 and says nothing about G.

WHAT THIS MODULE ADDS. Extra receivers on the beam axis in the
homogeneous run, at fractions of the IMAGE RANGE R2 = 2d. The image-source
construction the testbed is built on is what makes that the right
reference: a plane reflector at perpendicular distance d returns the echo
to the source exactly as a source at range 2d would, so a homogeneous
direct arrival at range 2d accumulates the same geometric spreading and
the same numerical dispersion as the echo, along the same direction,
through the same medium, with NO interface anywhere and therefore no
staircase of any kind. Its ratio across tilt is G(theta), and

    S(theta) = 20 log10 A(theta)/A(0)  -  20 log10 G(theta)/G(0)

is what belongs to the interface.

WHAT IT DOES NOT ESTABLISH. A specular reflection and a spherically
spreading direct wave do not weight the operator's wavenumber error
identically; G measured this way is the propagator's gain on the direct
field, and the reflection additionally involves the field's interaction
with the plane at the specular point. It is a reference for the path,
not a full forward model of the echo.

Nothing here changes tilt_testbed.py, tilt_rsg.py or rsg.py. The two
solver wrappers reproduce theirs step for step and are gated against them
(--gate) to a bit-identical source-cell trace; the only difference is
that more receivers are recorded, which cannot change the field.
"""
import os
import sys

import numpy as np
from scipy.signal import hilbert

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('.', '..'):
    _p = os.path.normpath(os.path.join(_HERE, _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))

import fdtd                                                # noqa: E402
import rsg                                                 # noqa: E402
import tilt_testbed as TB                                  # noqa: E402
import tilt_rsg                                            # noqa: E402
from ringfwi import anisotropy as an                       # noqa: E402

# fractions of the image range 2d at which the beam-axis receivers sit.
# 1.0 is the reference; the shorter ones show WHERE any tilt dependence
# accumulates and let the 1/r spreading be checked rather than assumed.
FRACS = (0.25, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------- geometry

def geometry(theta_deg, ppw):
    """Exactly the geometry tilt_testbed.run and tilt_rsg.run_rsg build.

    Returned in the (x, y, z) component order those modules use for `nrm`
    and `x0`, with array indices in the (z, y, x) order CuPy sees.
    """
    th = np.radians(theta_deg)
    R = TB.rot_y(th)
    nrm = R @ np.array([1.0, 0.0, 0.0])
    h = TB.C_REF / TB.F0 / ppw
    n = int(round(TB.L / h))
    m = TB.SPONGE + 3
    N = n + 2 * m
    ctr = np.array([N * h / 2] * 3)
    d = round((ctr[0] + TB.DEPTH) / h) * h - ctr[0]
    x0 = ctr + d * nrm
    c = N // 2
    p_src = np.array([(c + 0.5) * h] * 3)
    return dict(th=th, R=R, nrm=nrm, h=h, N=N, ctr=ctr, d=d, x0=x0,
                c=c, p_src=p_src)


def ray_receivers(g, fracs=FRACS):
    """Cells nearest the beam axis at `fracs` times the image range 2d.

    Returns (idx_zyx, r_true, r_nom).  r_true is the ACTUAL distance from
    the source cell centre to the receiver cell centre, which the caller
    must use to remove geometric spreading: snapping to the grid moves the
    range by up to h*sqrt(3)/2, which is 0.75 per cent of 2d at ppw 8 and
    would otherwise be read as a tilt-dependent amplitude.
    """
    h, N = g['h'], g['N']
    R2 = 2.0 * g['d']
    idx, r_true, r_nom = [], [], []
    for f in fracs:
        p = g['p_src'] + f * R2 * g['nrm']
        q = np.rint(p / h - 0.5).astype(int)            # (x, y, z)
        if q.min() < 0 or q.max() >= N:
            raise ValueError("beam receiver outside the grid")
        pc = (q + 0.5) * h
        idx.append((int(q[2]), int(q[1]), int(q[0])))   # -> (z, y, x)
        r_true.append(float(np.linalg.norm(pc - g['p_src'])))
        r_nom.append(float(f * R2))
    return idx, np.asarray(r_true), np.asarray(r_nom)


def check_outside_sponge(g, idx):
    """The recorded stress is pre-damp, so a receiver inside the sponge is
    silently wrong. forward_fused_labels refuses one; forward_rsg does not,
    so refuse here for both."""
    N, w = g['N'], TB.SPONGE
    for q in idx:
        for a in q:
            if a < w or a >= N - w:
                raise ValueError(
                    "beam receiver at %s lies in the %d-cell sponge of a "
                    "%d-cell grid" % (str(q), w, N))
    return min(min(min(a, N - 1 - a) for a in q) for q in idx) - w


# ------------------------------------------------------ standard grid (SSG)

def run_ssg(theta_deg, ppw, treat, psi_a, psi_b, homog=False, dt_in=None,
            extra_rec=()):
    """tilt_testbed.run with additional receiver groups.

    Reproduces that function line for line; extra receivers are appended
    to rec_groups, which the solver gathers from the same fields and
    cannot feed back into them, so the source-cell trace is unchanged.
    """
    g = geometry(theta_deg, ppw)
    R, nrm, h, N, c = g['R'], g['nrm'], g['h'], g['N'], g['c']
    ca = R @ np.array([np.cos(np.radians(psi_a)), 0.0,
                       np.sin(np.radians(psi_a))])
    cb = R @ np.array([np.cos(np.radians(psi_b)), 0.0,
                       np.sin(np.radians(psi_b))])
    CA, CB = TB.grain_C(ca), TB.grain_C(cb if not homog else ca)

    fr = TB.frac_field((N, N, N), h, nrm, g['x0'])
    lab = np.rint(fr * (TB.NBIN - 1)).astype(np.int32)
    Ct, rho_t = TB.build_tables(CA, CB, nrm, treat)

    co = fdtd.optimised_coeffs(TB.ORDER, kh_max=TB.KHM, multistart=True)
    dt = dt_in if dt_in is not None else fdtd.safe_dt_labels(
        (lab + 1).astype(np.uint8), Ct, rho_t, h, co, safety=0.5)
    nt = int(7.0e-6 / dt)
    wav = fdtd.ricker(TB.F0, dt, nt)
    src = [((c, c, c), 1.0)]
    groups = [([(c, c, c)], np.array([1.0]))]
    groups += [([q], np.array([1.0])) for q in extra_rec]
    rec = np.asarray(fdtd.forward_fused_labels(
        lab, None, h, dt, nt, src, wav, groups,
        order=TB.ORDER, coeffs=co, sponge_width=TB.SPONGE,
        mat_tables=(Ct, rho_t)), float)
    return rec, dt, g['d']


# ------------------------------------------------------- rotated grid (RSG)

def forward_rsg_multi(Cfield, rho, h, dt, nt, src_idx, wavelet, rec_pts,
                      coeffs, sponge_width=12, sponge_strength=0.16):
    """rsg.forward_rsg with a LIST of receiver points.

    The time loop is a verbatim copy of rsg.forward_rsg's; only the
    recording line differs, gathering each point separately instead of
    summing them.  Gated in __main__ against the original to a
    bit-identical source-cell trace.
    """
    import cupy as cp
    xp = cp
    c = np.asarray(coeffs, float)
    C = xp.asarray(np.asarray(Cfield, np.float32))
    shp = C.shape[2:]
    rho_a = xp.asarray(np.broadcast_to(np.asarray(rho, np.float32),
                                       shp).copy())
    v = [xp.zeros(shp, cp.float32) for _ in range(3)]
    s = [xp.zeros(shp, cp.float32) for _ in range(6)]

    w = xp.ones(shp, cp.float32)
    if sponge_width > 0:
        ramp = xp.asarray(
            (1.0 - sponge_strength
             * (0.5 * (1 + np.cos(np.pi * np.arange(sponge_width)
                                  / sponge_width)))[::-1]).astype(np.float32))
        for d in range(3):
            sl = [slice(None)] * 3
            sh_ = [1, 1, 1]
            sh_[d] = sponge_width
            r = ramp.reshape(sh_)
            sl[d] = slice(0, sponge_width)
            w[tuple(sl)] *= r
            sl[d] = slice(shp[d] - sponge_width, shp[d])
            w[tuple(sl)] *= r[::-1] if d == 0 else xp.flip(r, axis=d)

    rp = np.asarray(rec_pts, int).reshape(-1, 3)
    zi = xp.asarray(rp[:, 0])
    yi = xp.asarray(rp[:, 1])
    xi = xp.asarray(rp[:, 2])
    out = np.zeros((nt, len(rp)), np.float64)
    srcd = xp.asarray(src_idx)
    for it in range(nt):
        gx = rsg._grad(xp, v[0], c, h, True)
        gy = rsg._grad(xp, v[1], c, h, True)
        gz = rsg._grad(xp, v[2], c, h, True)
        e = [gx[0], gy[1], gz[2],
             gy[2] + gz[1], gx[2] + gz[0], gx[1] + gy[0]]
        for I in range(6):
            acc = C[I, 0] * e[0]
            for J in range(1, 6):
                acc = acc + C[I, J] * e[J]
            s[I] += dt * acc
        s[0] += dt * wavelet[it] * srcd
        s[1] += dt * wavelet[it] * srcd
        s[2] += dt * wavelet[it] * srcd
        for I in range(6):
            s[I] *= w
        d0 = rsg._grad(xp, s[0], c, h, False)
        d1 = rsg._grad(xp, s[1], c, h, False)
        d2 = rsg._grad(xp, s[2], c, h, False)
        d3 = rsg._grad(xp, s[3], c, h, False)
        d4 = rsg._grad(xp, s[4], c, h, False)
        d5 = rsg._grad(xp, s[5], c, h, False)
        v[0] += dt / rho_a * (d0[0] + d5[1] + d4[2])
        v[1] += dt / rho_a * (d5[0] + d1[1] + d3[2])
        v[2] += dt / rho_a * (d4[0] + d3[1] + d2[2])
        for d in range(3):
            v[d] *= w
        out[it] = cp.asnumpy(
            -(s[0][zi, yi, xi] + s[1][zi, yi, xi] + s[2][zi, yi, xi]) / 3.0)
    return out


def run_rsg(theta_deg, ppw, psi_a, psi_b, homog=False, dt_in=None,
            extra_rec=()):
    """tilt_rsg.run_rsg with additional receiver points."""
    g = geometry(theta_deg, ppw)
    R, nrm, h, N, c = g['R'], g['nrm'], g['h'], g['N'], g['c']
    ca = R @ np.array([np.cos(np.radians(psi_a)), 0.0,
                       np.sin(np.radians(psi_a))])
    cb = R @ np.array([np.cos(np.radians(psi_b)), 0.0,
                       np.sin(np.radians(psi_b))])
    CA = TB.grain_C(ca)
    CB = TB.grain_C(cb if not homog else ca)

    fr = TB.frac_field((N, N, N), h, nrm, g['x0'], sup=1)
    sel = (fr > 0.5)
    C = np.empty((6, 6, N, N, N), np.float32)
    for I in range(6):
        for J in range(6):
            C[I, J] = np.where(sel, CA[I, J], CB[I, J]).astype(np.float32)

    co = fdtd.optimised_coeffs(TB.ORDER, kh_max=TB.KHM, multistart=True)
    rho = float(an.ICE_MATERIAL["rho"])
    dt = dt_in if dt_in is not None else rsg.stable_dt(
        float(max(CA.max(), CB.max())), rho, h, co, safety=0.4)
    nt = int(7.0e-6 / dt)
    wav = fdtd.ricker(TB.F0, dt, nt)
    src = np.zeros((N, N, N), np.float32)
    src[c, c, c] = 1.0
    pts = [(c, c, c)] + [tuple(q) for q in extra_rec]
    rec = forward_rsg_multi(C, rho, h, dt, nt, src, wav, pts, co,
                            sponge_width=TB.SPONGE)
    return np.asarray(rec, float), dt, g['d']


# ---------------------------------------------------------------- measures

def env(x):
    return np.abs(hilbert(np.asarray(x, float)))


def gate_peak(x, dt, t0, half_us=0.5):
    """Envelope peak of `x` in a +-half_us window about t0, and its time."""
    e = env(x)
    fs = 1.0 / dt
    w = int(half_us * 1e-6 * fs)
    i0 = int(t0 * fs)
    a, b = max(i0 - w, 0), min(i0 + w, len(e))
    j = int(np.argmax(e[a:b])) + a
    return float(e[j]), float(j * dt)


def out_of_gate_peak(x, dt, t0, half_us=0.5, guard_us=1.0,
                     skip_us=0.5, tail_us=0.3):
    """Largest envelope value OUTSIDE the gate and its guard band.

    This is the denominator of the echo-to-noise ratio. `skip_us` drops
    the first half microsecond, where the differenced field is exactly
    zero by causality and float underflow makes the ratio meaningless,
    and `tail_us` drops the Hilbert transform's end taper.
    """
    e = env(x)
    fs = 1.0 / dt
    n = len(e)
    i0 = int(t0 * fs)
    gw = int(guard_us * 1e-6 * fs)
    a = int(skip_us * 1e-6 * fs)
    b = n - int(tail_us * 1e-6 * fs)
    pre = e[a:max(i0 - gw, a)]
    post = e[min(i0 + gw, b):b]
    pk_pre = float(pre.max()) if pre.size else 0.0
    pk_post = float(post.max()) if post.size else 0.0
    return max(pk_pre, pk_post), pk_pre, pk_post


# -------------------------------------------------------------------- gate

def _gate():
    """Prove the two wrappers reproduce the published solvers exactly.

    Small grid, few steps, single receiver: the source-cell trace must be
    bit-identical to tilt_testbed.run and tilt_rsg.run_rsg. If it is not,
    nothing measured with this module may be compared with the published
    numbers.
    """
    ok = True
    for th in (0.0, 45.0):
        a, _, _ = TB.run(th, 4.0, 'naive', 0.0, 51.0)
        b, _, _ = run_ssg(th, 4.0, 'naive', 0.0, 51.0)
        same = np.array_equal(a, b[:, 0])
        ok &= same
        print('SSG  tilt %5.1f  bit-identical %s  max|diff| %.3e'
              % (th, same, np.abs(a - b[:, 0]).max()))
    for th in (0.0, 45.0):
        a, _, _ = tilt_rsg.run_rsg(th, 4.0, 0.0, 51.0)
        b, _, _ = run_rsg(th, 4.0, 0.0, 51.0)
        same = np.array_equal(a, b[:, 0])
        ok &= same
        print('RSG  tilt %5.1f  bit-identical %s  max|diff| %.3e'
              % (th, same, np.abs(a - b[:, 0]).max()))
    print('GATE', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--gate':
        sys.exit(0 if _gate() else 1)
    for ppw in (6.0, 8.0, 10.0):
        for th in (0.0, 30.0, 45.0, 60.0):
            g = geometry(th, ppw)
            idx, rt, rn = ray_receivers(g)
            marg = check_outside_sponge(g, idx)
            print('ppw %4.1f  tilt %5.1f  N %4d  h %.4f mm  d %.4f mm  '
                  '2d %.4f mm  r_true %s  sponge margin %d cells'
                  % (ppw, th, g['N'], g['h'] * 1e3, g['d'] * 1e3,
                     2e3 * g['d'], np.array2string(rt * 1e3, precision=3),
                     marg))
