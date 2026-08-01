"""Arbiter shot round 3: SOURCE-MATCHED. Round-2 forensics found the
remaining -19 dB 'floor' is S-WAVE physics the FDTD cannot see: the FE
point-force source radiates shear (direct S arrives 5.94 us = the loud
5.5-6.1 us feature; S-grain coda fills the window), while the FDTD
source is an isotropic stress monopole (verified in fdtd.py: wavelet
added equally to sxx/syy/szz; record = pressure = -mean stress).

Round 3 = round 2's padded mesh (cached) + quartic shell, with the FE
source switched to a centre of dilatation (f_a = M grad N_a, P-only)
and the receiver to dilatation div u - the FE counterparts of the
FDTD's monopole/pressure. Own-direct normalisation absorbs scales.
"""
import time

import numpy as np

import fdtd                                     # noqa: E402
import fe_solver_p2plus as pp                   # noqa: E402
from fe_crosscheck import F0, axes_sets, run_fdtd  # noqa: E402
from fe_crosscheck import SRC, REC              # noqa: E402
from fe_p2plus_ab import analyse, T             # noqa: E402
from fe_p2plus_ab2 import get_mesh, SHELL_FRAC, SHELL_RATE  # noqa: E402


def run_p2plus():
    nodes10, tets10, grain = get_mesh()
    nodes15, tets15 = pp.build_nodes(nodes10, tets10)
    aA, aB = axes_sets()
    out = {}
    for tag, ax in (("A", aA), ("B", aB)):
        ax13 = np.vstack([ax, ax[:1]])
        vol, D = pp.precompute(nodes15, tets15, grain, ax13)
        md = pp.lumped_mass(nodes15, tets15, vol)
        dt = pp.stable_dt(tets15, D, md)
        nt = int(T / dt)
        wav = fdtd.ricker(F0, dt, nt)
        t0 = time.time()
        out[tag] = (pp.forward(nodes15, tets15, grain, ax13, dt, nt, wav,
                               SRC, [REC], shell_frac=SHELL_FRAC,
                               shell_rate=SHELL_RATE, shell_pow=4,
                               src_type="monopole", rec_type="div",
                               D=D, vol=vol)[:, 0], dt)
        print(f"P2+ {tag}: {time.time()-t0:.0f}s ({len(tets15)} tets, "
              f"{nt} steps, dt {dt*1e9:.2f} ns)", flush=True)
    fs = 5.0e8
    tg = np.arange(0.0, T, 1.0 / fs)
    za = np.interp(tg, np.arange(len(out["A"][0])) * out["A"][1],
                   out["A"][0])
    zb = np.interp(tg, np.arange(len(out["B"][0])) * out["B"][1],
                   out["B"][0])
    return za, zb, 1.0 / fs


def main():
    fa, fb, dtf = run_fdtd()
    lf = analyse(fa, fb, dtf, "FDTD (staircase, monopole/pressure)")
    ea, eb, dte = run_p2plus()
    np.savez("fe_p2plus_ab3_traces.npz", fa=fa, fb=fb, dtf=dtf,
             ea=ea, eb=eb, dte=dte)
    le = analyse(ea, eb, dte, "P2+ (padded, monopole/div)")
    print(f"\nCROSS-CHECK 3: FDTD {lf:.1f} vs P2+ {le:.1f} dB -> "
          f"difference {lf-le:+.1f} dB "
          f"({'PASS' if abs(lf-le) < 3.0 else 'INVESTIGATE'} at 3 dB)",
          flush=True)


if __name__ == "__main__":
    main()
