"""Arbiter shot round 2: FE box enlarged with a uniform-crystal pad and
a weak-graded quartic absorber, so the FE finally simulates the SAME
problem as the FDTD reference (whose margin is ALREADY uniform grain-0
crystal + sponge - checked in run_fdtd: lab fills with 0 -> grain 0).

Round-1 forensics (fe_p2plus_ab_traces.npz): FE uniform-B carried
-19.7 dB of in-window box reverberation (FDTD-B: -52.3) because the
0.35/step-equivalent shell is an overdamped brick wall reflecting
~-16 dB at its entrance; A-B then measured shifted reverberation, not
grain scattering. Fix: pad 8 mm (~2 wavelengths) of uniform crystal
(grain id 12 by centroid rule) on every face, quartic shell over
exactly the pad, rate 1.15e7/s (one-way -40 dB, adiabatic entry).
Mesh frame -8..38 mm so specimen coordinates, src/rec, window and the
analyse() of round 1 are reused verbatim. FDTD side unchanged.
"""
import time

import numpy as np

import fdtd                                     # noqa: E402
import fe_mesh                                  # noqa: E402
import fe_solver_p2 as p2                       # noqa: E402
import fe_solver_p2plus as pp                   # noqa: E402
from fe_crosscheck import (L, F0, SRC, REC,     # noqa: E402
                           seeds, axes_sets, run_fdtd)
from fe_p2plus_ab import analyse, T             # noqa: E402

PAD = 0.008
H = 0.54e-3
SHELL_FRAC = PAD / (L + 2 * PAD)                # shell covers exactly the pad
SHELL_RATE = 1.15e7                             # int q^4 -> one-way ~ -40 dB


def get_mesh():
    import os
    path = "fe_p2plus_ab2_mesh.npz"
    if os.path.exists(path):
        d = np.load(path)
        nodes, tets, grain = d["nodes"], d["tets"], d["grain"]
        print(f"reusing {path}: {len(tets)} tets", flush=True)
    else:
        t0 = time.time()
        nodes, tets, grain = fe_mesh.mesh_polycrystal(
            seeds(), ((-PAD, L + PAD),) * 3, H, path, order=2)
        print(f"meshed: {len(tets)} tets in {time.time()-t0:.0f}s",
              flush=True)
    cent = nodes[tets[:, :4]].mean(axis=1)
    pad = ((cent < 0) | (cent > L)).any(axis=1)
    grain = grain.copy()
    grain[pad] = 12                             # uniform-crystal surround
    print(f"pad tets: {pad.sum()} of {len(tets)} "
          f"({100*pad.mean():.0f}%)", flush=True)
    return p2.straighten(nodes, tets), tets, grain


def run_p2plus():
    nodes10, tets10, grain = get_mesh()
    nodes15, tets15 = pp.build_nodes(nodes10, tets10)
    aA, aB = axes_sets()
    out = {}
    for tag, ax in (("A", aA), ("B", aB)):
        ax13 = np.vstack([ax, ax[:1]])          # 12 grains + pad = axis 0
        vol, D = pp.precompute(nodes15, tets15, grain, ax13)
        md = pp.lumped_mass(nodes15, tets15, vol)
        dt = pp.stable_dt(tets15, D, md)
        nt = int(T / dt)
        wav = fdtd.ricker(F0, dt, nt)
        t0 = time.time()
        out[tag] = (pp.forward(nodes15, tets15, grain, ax13, dt, nt, wav,
                               SRC, [REC], shell_frac=SHELL_FRAC,
                               shell_rate=SHELL_RATE, shell_pow=4,
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
    lf = analyse(fa, fb, dtf, "FDTD (staircase)")
    ea, eb, dte = run_p2plus()
    np.savez("fe_p2plus_ab2_traces.npz", fa=fa, fb=fb, dtf=dtf,
             ea=ea, eb=eb, dte=dte)
    le = analyse(ea, eb, dte, "P2+ (padded, quartic shell)")
    print(f"\nCROSS-CHECK 2: FDTD {lf:.1f} vs P2+ {le:.1f} dB -> "
          f"difference {lf-le:+.1f} dB "
          f"({'PASS' if abs(lf-le) < 3.0 else 'INVESTIGATE'} at 3 dB)",
          flush=True)


if __name__ == "__main__":
    main()
