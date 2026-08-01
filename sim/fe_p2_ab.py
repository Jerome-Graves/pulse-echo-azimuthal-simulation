"""THE arbiter shot: staircased FDTD vs conforming HRZ-TET10, A-B coda.

Runs BOTH solvers' A (real random c-axes) and B (uniform crystal) on the
same 12-grain specimen (fe_crosscheck seeds/geometry), and scores the
A-B difference coda in the SAME edge-safe window with the SAME analysis
- no remembered numbers, everything measured in one harness.

Only meaningful if fe_p2_floor.py returned a floor comfortably below
the FDTD grain signal; the P2 floor context is printed alongside.
"""
import time

import numpy as np
from scipy.signal import hilbert

import fdtd                                     # noqa: E402
import fe_solver_p2 as p2                       # noqa: E402
from fe_crosscheck import (F0, SRC, REC,                    # noqa: E402
                           axes_sets, run_fdtd)

W = (4.9e-6, 6.2e-6)
T = 7.0e-6


def analyse(a, b, dt, name):
    fs = 1.0 / dt
    t_dir = np.linalg.norm(REC - SRC) / 4046.0 + 1.2 / F0
    e_dir = np.abs(hilbert(b))
    k = int(t_dir * fs)
    direct = e_dir[max(k - int(1e-6 * fs), 0):k + int(1e-6 * fs)].max()
    cw = np.abs(hilbert(a - b))[int(W[0] * fs):int(W[1] * fs)]
    lvl = 20 * np.log10(np.sqrt((cw ** 2).mean()) / direct + 1e-30)
    print(f"{name}: direct {direct:.3e}, A-B coda RMS {lvl:.1f} dB "
          f"re direct (window {W[0]*1e6:.1f}-{W[1]*1e6:.1f} us)",
          flush=True)
    return lvl


def run_p2():
    d = np.load("fe_p2_floor1.npz")
    nodes, tets10, grain = d["nodes"], d["tets"], d["grain"]
    nodes = p2.straighten(nodes, tets10)
    aA, aB = axes_sets()
    dt = p2.stable_dt_p2(nodes, tets10, 4046.0)
    nt = int(T / dt)
    wav = fdtd.ricker(F0, dt, nt)
    out = {}
    for tag, ax in (("A", aA), ("B", aB)):
        t0 = time.time()
        out[tag] = p2.forward_fe_p2(nodes, tets10, grain, ax, dt, nt,
                                    wav, SRC, [REC])[:, 0]
        print(f"P2 {tag}: {time.time()-t0:.0f}s "
              f"({len(tets10)} tets, {nt} steps)", flush=True)
    return out["A"], out["B"], dt


def main():
    fa, fb, dtf = run_fdtd()
    lf = analyse(fa, fb, dtf, "FDTD (staircase)")
    ea, eb, dte = run_p2()
    le = analyse(ea, eb, dte, "P2-FE (conforming)")
    print(f"\nCROSS-CHECK: FDTD {lf:.1f} vs P2-FE {le:.1f} dB -> "
          f"difference {lf-le:+.1f} dB "
          f"({'PASS' if abs(lf-le) < 3.0 else 'INVESTIGATE'} at 3 dB)",
          flush=True)


if __name__ == "__main__":
    main()
