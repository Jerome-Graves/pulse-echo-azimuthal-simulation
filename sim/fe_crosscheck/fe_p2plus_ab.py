import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
"""THE arbiter shot, ML2n15 edition: staircased FDTD vs conforming
mass-lumped-P2 FE on the same 12-grain specimen, A-B coda in the same
edge-safe window, both measured by this one harness.

A = real random c-axes, B = uniform crystal; each solver's A-B coda is
normalised by its OWN direct so source-type differences common-mode.
Run only after fe_p2plus_floor.py reports a floor comfortably below the
FDTD grain signal (-42.8 dB): the P2+ floor is what makes this shot
meaningful where CST (-26.8) and HRZ-TET10 could not referee.

Input: fe_p2_floor1.npz (the 0.54 mm TET10 remesh) in the working
directory. The mesh npz files are archived out of the repo as
regenerable bulk; np.load here does not rebuild, so run fe_p2_floor.py
first (its get_mesh remeshes and saves both floor meshes) or restore
the file from pulse-echo-cof-sim_archive.
"""
import time

import numpy as np
from scipy.signal import hilbert

import fdtd                                     # noqa: E402
import fe_solver_p2 as p2                       # noqa: E402
import fe_solver_p2plus as pp                   # noqa: E402
from fe_crosscheck import (F0, SRC, REC,        # noqa: E402
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


def run_p2plus():
    d = np.load("fe_p2_floor1.npz")
    nodes10, tets10, grain = d["nodes"], d["tets"], d["grain"]
    nodes10 = p2.straighten(nodes10, tets10)
    nodes15, tets15 = pp.build_nodes(nodes10, tets10)
    aA, aB = axes_sets()
    out = {}
    for tag, ax in (("A", aA), ("B", aB)):
        vol, D = pp.precompute(nodes15, tets15, grain, ax)
        md = pp.lumped_mass(nodes15, tets15, vol)
        dt = pp.stable_dt(tets15, D, md)
        nt = int(T / dt)
        wav = fdtd.ricker(F0, dt, nt)
        t0 = time.time()
        out[tag] = (pp.forward(nodes15, tets15, grain, ax, dt, nt, wav,
                               SRC, [REC], D=D, vol=vol)[:, 0], dt)
        print(f"P2+ {tag}: {time.time()-t0:.0f}s ({len(tets15)} tets, "
              f"{nt} steps, dt {dt*1e9:.2f} ns)", flush=True)
    # common 2 ns clock (A and B dt differ slightly via power iteration)
    fs = 5.0e8
    tg = np.arange(0.0, T, 1.0 / fs)
    za = np.interp(tg, np.arange(len(out["A"][0])) * out["A"][1],
                   out["A"][0])
    zb = np.interp(tg, np.arange(len(out["B"][0])) * out["B"][1],
                   out["B"][0])
    return za, zb, 1.0 / fs


def main():
    fa, fb, dtf = run_fdtd()
    lf = analyse(fa, fb, dtf, "FDTD (staircase)")
    ea, eb, dte = run_p2plus()
    np.savez("fe_p2plus_ab_traces.npz", fa=fa, fb=fb, dtf=dtf,
             ea=ea, eb=eb, dte=dte)      # every expensive run reanalysable
    le = analyse(ea, eb, dte, "P2+ (conforming ML2n15)")
    print(f"\nCROSS-CHECK: FDTD {lf:.1f} vs P2+ {le:.1f} dB -> "
          f"difference {lf-le:+.1f} dB "
          f"({'PASS' if abs(lf-le) < 3.0 else 'INVESTIGATE'} at 3 dB)",
          flush=True)


if __name__ == "__main__":
    main()
