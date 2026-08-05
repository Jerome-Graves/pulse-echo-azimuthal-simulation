import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
"""Order-of-convergence check for ML2n15: same bar, F0 = 1.0 vs 0.5 MHz.

A healthy 4th-order element drops the group-velocity error ~16x when
nE doubles (8 -> 16 elements/wavelength); any residual k^2-type bug
would only drop ~4x. This discriminates 'harsh broadband-group metric'
from 'element defect' after the +0.87% spatial reading at lambda/8.
"""
import time

import numpy as np

import fdtd                                     # noqa: E402
import fe_solver_p2plus as pp                   # noqa: E402
from fe_p2_probe import kuhn_bar, xcorr_dt      # noqa: E402

RHO = 917.0
h = 0.5e-3
nodes4, tets4, nodes10, tets10 = kuhn_bar(80, 24, 24, h)
nodes15, tets15 = pp.build_nodes(nodes10, tets10)
axes = np.array([[1.0, 0.0, 0.0]])
grain = np.zeros(len(tets15), np.int32)
vol, D = pp.precompute(nodes15, tets15, grain, axes)
md = pp.lumped_mass(nodes15, tets15, vol)
Ct, _ = fdtd.material_tables(axes)
v_exp = np.sqrt(Ct.reshape(-1, 6, 6)[1][0, 0] / RHO)
dt0 = pp.stable_dt(tets15, D, md)
print(f"dt = {dt0*1e9:.2f} ns", flush=True)

src = np.array([8e-3, 6e-3, 6e-3])
r1 = np.array([16e-3, 6e-3, 6e-3])
r2 = np.array([28e-3, 6e-3, 6e-3])
gap = np.linalg.norm(r2 - r1)
errs = {}
for F0 in (1.0e6, 0.5e6):
    t0w = 1.2 / F0
    t1e = np.linalg.norm(r1 - src) / v_exp + t0w
    t2e = np.linalg.norm(r2 - src) / v_exp + t0w
    T = t2e + 2.5 / F0
    nt = int(T / dt0)
    wav = fdtd.ricker(F0, dt0, nt)
    t0 = time.time()
    rec = pp.forward(nodes15, tets15, grain, axes, dt0, nt, wav,
                     src, [r1, r2], shell_rate=0.0, D=D, vol=vol)
    lag = xcorr_dt(rec[:, 0], rec[:, 1], dt0, t1e, t2e, half=1.2 / F0)
    v = gap / lag
    errs[F0] = v / v_exp - 1
    print(f"F0={F0/1e6:.1f} MHz (nE={v_exp/F0/h:.1f}): "
          f"{time.time()-t0:.0f}s, v = {v:.1f} m/s "
          f"({errs[F0]*100:+.3f}%)", flush=True)
r = errs[1.0e6] / errs[0.5e6]
print(f"\nerror ratio (nE 8 -> 16): {r:.1f}x  "
      f"(4th order -> ~16x, k^2 bug -> ~4x)", flush=True)
