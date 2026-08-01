"""Same tilted-interface testbed, but solved on a ROTATED staggered grid.

Identical geometry, identical truth definition (rigid rotation of the whole
problem, so tilt 0 is exact), identical A-B design and echo metric as
tilt_testbed.py, so the two are directly comparable.

The RSG claim: because all six stress components share one location, no
material interpolation is needed anywhere, and a non-grid-aligned interface
is represented without the component-by-component mixing that the standard
staggered grid suffers.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.signal import hilbert

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
import fdtd                                            # noqa: E402
import rsg                                             # noqa: E402
import tilt_testbed as TB                              # noqa: E402
from ringfwi import anisotropy as an                   # noqa: E402


def run_rsg(theta_deg, ppw, psi_a, psi_b, homog=False, dt_in=None):
    th = np.radians(theta_deg)
    R = TB.rot_y(th)
    nrm = R @ np.array([1.0, 0.0, 0.0])
    ca = R @ np.array([np.cos(np.radians(psi_a)), 0.0,
                       np.sin(np.radians(psi_a))])
    cb = R @ np.array([np.cos(np.radians(psi_b)), 0.0,
                       np.sin(np.radians(psi_b))])
    CA = TB.grain_C(ca)
    CB = TB.grain_C(cb if not homog else ca)

    h = TB.C_REF / TB.F0 / ppw
    n = int(round(TB.L / h))
    m = TB.SPONGE + 3
    N = n + 2 * m
    ctr = np.array([N * h / 2] * 3)
    d = round((ctr[0] + TB.DEPTH) / h) * h - ctr[0]
    x0 = ctr + d * nrm

    # RSG needs NO interpolation: assign whichever grain owns the stress
    # point. That is the whole point of the scheme, so use the plain
    # nearest-point rule rather than an effective medium.
    fr = TB.frac_field((N, N, N), h, nrm, x0, sup=1)
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
    c = N // 2
    src = np.zeros((N, N, N), np.float32)
    src[c, c, c] = 1.0
    tr = rsg.forward_rsg(C, rho, h, dt, nt, src, wav, (c, c, c), co,
                         sponge_width=TB.SPONGE)
    return np.asarray(tr, float), dt, d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppw", type=float, nargs="+", default=[8.0])
    ap.add_argument("--tilts", type=float, nargs="+",
                    default=[0, 15, 22.5, 30, 45])
    ap.add_argument("--psi", type=float, nargs=2, default=[0.0, 51.0])
    a = ap.parse_args()
    print(f"ROTATED staggered grid   c-axes {a.psi[0]:.0f}/{a.psi[1]:.0f}\n")
    print(f"{'ppw':>5}{'tilt':>8}{'echo':>12}{'vs tilt0 (dB)':>16}{'s':>7}")
    for ppw in a.ppw:
        base = None
        for t in a.tilts:
            t0 = time.time()
            bi, dt, d = run_rsg(t, ppw, a.psi[0], a.psi[1])
            ho, _, _ = run_rsg(t, ppw, a.psi[0], a.psi[1], homog=True,
                               dt_in=dt)
            if not (np.isfinite(bi).all() and np.isfinite(ho).all()):
                print(f"{ppw:>5.0f}{t:>8.1f}   UNSTABLE")
                continue
            amp = TB.echo_amp(bi, ho, dt, d, TB.vA(a.psi[0]))
            if base is None:
                base = amp
            print(f"{ppw:>5.0f}{t:>8.1f}{amp:>12.4e}"
                  f"{20*np.log10(amp/base):>15.2f} {time.time()-t0:>6.0f}")
