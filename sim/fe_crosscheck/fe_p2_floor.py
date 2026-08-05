import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
"""HRZ-lumped TET10 mesh-texture floor: the go/no-go for the P2 arbiter.

Protocol (same as the P1 floor measurement that returned -26.8 dB):
one specimen geometry (fe_crosscheck seeds, 30 mm box), UNIFORM crystal
everywhere (no grain physics), two INDEPENDENT remeshes (h and 1.03h so
every node moves); the difference trace in the edge-safe 4.9-6.2 us
window is pure mesh texture. Exact-coordinate src/rec via barycentric
placement (placement already exonerated on P1: -25.0 -> -26.8).

Node spacing matched to the P1 floor test: TET10 at h = 0.54 mm has
nodes every 0.27 mm, same as P1 at h = 0.27 mm.
Target: floor <= -45 dB re direct (grain signal to referee is -42.8).
The structured-bar probe showed HRZ-P2 dispersion ~= P1 at equal h,
so this measures whether P2's smoother error field scatters less
anyway. Traces are interpolated onto a common 2 ns clock before
differencing (meshes have different stable dt).
"""
import time

import numpy as np
from scipy.signal import hilbert

import fdtd                                     # noqa: E402
import fe_mesh                                  # noqa: E402
import fe_solver_p2 as p2                       # noqa: E402
from fe_crosscheck import L, F0, SRC, REC, seeds, axes_sets  # noqa: E402

W = (4.9e-6, 6.2e-6)
T = 7.0e-6


def get_mesh(h, path):
    import os
    if os.path.exists(path):
        d = np.load(path)
        nodes, tets, grain = d["nodes"], d["tets"], d["grain"]
        print(f"reusing {path}: {len(tets)} tets", flush=True)
    else:
        t0 = time.time()
        nodes, tets, grain = fe_mesh.mesh_polycrystal(
            seeds(), ((0, L), (0, L), (0, L)), h, path, order=2)
        print(f"meshed {path}: {len(tets)} tets in {time.time()-t0:.0f}s",
              flush=True)
    return p2.straighten(nodes, tets), tets, grain


def run(nodes, tets10, grain, axes):
    dt = p2.stable_dt_p2(nodes, tets10, 4046.0)
    nt = int(T / dt)
    wav = fdtd.ricker(F0, dt, nt)
    t0 = time.time()
    rec = p2.forward_fe_p2(nodes, tets10, grain, axes, dt, nt, wav,
                           SRC, [REC])[:, 0]
    print(f"run: {time.time()-t0:.0f}s ({len(tets10)} tets, {nt} steps, "
          f"dt {dt*1e9:.2f} ns)", flush=True)
    return rec, dt


def main():
    aB = axes_sets()[1]                          # uniform crystal
    meshes = [get_mesh(0.54e-3, "fe_p2_floor1.npz"),
              get_mesh(0.556e-3, "fe_p2_floor2.npz")]
    fs = 5.0e8                                   # common 2 ns clock
    tg = np.arange(0.0, T, 1.0 / fs)
    z, directs = [], []
    t_dir = np.linalg.norm(REC - SRC) / 4046.0 + 1.2 / F0
    for nodes, tets10, grain in meshes:
        r, dt = run(nodes, tets10, grain, aB)
        zi = np.interp(tg, np.arange(len(r)) * dt, r)
        env = np.abs(hilbert(zi))
        k = int(t_dir * fs)
        directs.append(env[k - int(1e-6 * fs):k + int(1e-6 * fs)].max())
        z.append(zi)
    print(f"directs: {directs[0]:.4e} / {directs[1]:.4e} "
          f"(ratio {directs[1]/directs[0]:.4f})", flush=True)
    d = z[0] - z[1]
    w0, w1 = int(W[0] * fs), int(W[1] * fs)
    cw = np.abs(hilbert(d))[w0:w1]
    lvl = 20 * np.log10(np.sqrt((cw ** 2).mean()) / np.mean(directs) + 1e-30)
    print(f"\nP2-HRZ TEXTURE FLOOR: {lvl:.1f} dB re direct "
          f"(window 4.9-6.2 us)\n"
          f"context: P1 floor -26.8 | FDTD grain signal -42.8 | "
          f"target <= -45 -> "
          f"{'GO' if lvl <= -45.0 else 'NO-GO for HRZ-P2 arbiter'}",
          flush=True)


if __name__ == "__main__":
    main()
