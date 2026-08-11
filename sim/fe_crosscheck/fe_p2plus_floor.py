import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
"""ML2n15 mesh-texture floor: same specimen, independent remeshes.

Reuses the TET10 remesh pair from fe_p2_floor (straightened + face/
interior nodes built - no new meshing). Lessons applied from the P2-HRZ
floor attempt (-16.2 dB, invalidated): (a) the shell is RATE-scaled in
this solver, so unequal dt between meshes no longer changes absorption;
(b) each trace is gain-normalised by its OWN direct before differencing
(source coupling is mesh-local and common-modes away in the same-mesh
A-B this floor is a proxy for; leaving it in fabricated most of the
-16.2). Arrival-time texture is NOT aligned away - that is the signal.

Context chain: P1 floor -26.8 dB (0.5% dispersion) -> ML2n15 dispersion
0.046% at lambda/8 -> projected floor ~ -47 dB. Target <= -45.

Inputs: fe_p2_floor1.npz and fe_p2_floor2.npz in the working
directory. Both mesh npz files are archived out of the repo as
regenerable bulk; run() loads without rebuilding, so run
fe_p2_floor.py first (its get_mesh remeshes and saves both) or
restore them from pulse-echo-azimuthal-simulation_archive.
"""
import time

import numpy as np
from scipy.signal import hilbert

import fdtd                                     # noqa: E402
import fe_solver_p2 as p2                       # noqa: E402
import fe_solver_p2plus as pp                   # noqa: E402
from fe_crosscheck import F0, SRC, REC, axes_sets  # noqa: E402

W = (4.9e-6, 6.2e-6)
T = 7.0e-6


def run(path, axes):
    d = np.load(path)
    nodes10, tets10, grain = d["nodes"], d["tets"], d["grain"]
    nodes10 = p2.straighten(nodes10, tets10)
    nodes15, tets15 = pp.build_nodes(nodes10, tets10)
    vol, D = pp.precompute(nodes15, tets15, grain, axes)
    md = pp.lumped_mass(nodes15, tets15, vol)
    dt = pp.stable_dt(tets15, D, md)
    nt = int(T / dt)
    wav = fdtd.ricker(F0, dt, nt)
    t0 = time.time()
    rec = pp.forward(nodes15, tets15, grain, axes, dt, nt, wav,
                     SRC, [REC], D=D, vol=vol)[:, 0]
    print(f"{path}: {time.time()-t0:.0f}s ({len(tets15)} tets, "
          f"{nt} steps, dt {dt*1e9:.2f} ns)", flush=True)
    return rec, dt


def main():
    aB = axes_sets()[1]                          # uniform crystal
    fs = 5.0e8
    tg = np.arange(0.0, T, 1.0 / fs)
    t_dir = np.linalg.norm(REC - SRC) / 4046.0 + 1.2 / F0
    z, directs = [], []
    for path in ("fe_p2_floor1.npz", "fe_p2_floor2.npz"):
        r, dt = run(path, aB)
        zi = np.interp(tg, np.arange(len(r)) * dt, r)
        env = np.abs(hilbert(zi))
        k = int(t_dir * fs)
        directs.append(env[k - int(1e-6 * fs):k + int(1e-6 * fs)].max())
        z.append(zi / directs[-1])               # own-direct gain norm
    print(f"directs: {directs[0]:.4e} / {directs[1]:.4e} "
          f"(ratio {directs[1]/directs[0]:.4f})", flush=True)
    d = z[0] - z[1]                              # both now direct == 1
    w0, w1 = int(W[0] * fs), int(W[1] * fs)
    cw = np.abs(hilbert(d))[w0:w1]
    lvl = 20 * np.log10(np.sqrt((cw ** 2).mean()) + 1e-30)
    print(f"\nML2n15 TEXTURE FLOOR: {lvl:.1f} dB re direct "
          f"(window 4.9-6.2 us, gain-normalised)\n"
          f"context: P1 -26.8 | FDTD grain signal -42.8 | target <= -45 "
          f"-> {'GO' if lvl <= -45.0 else 'NO-GO'}", flush=True)


if __name__ == "__main__":
    main()
