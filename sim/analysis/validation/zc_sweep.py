"""The decisive control: can an ACOUSTICALLY INVISIBLE tessellation fake it?

Identical specimen, identical grid, identical solver settings, all 60
azimuths - but every grain is given the SAME c-axis.  The grain boundaries
are still there geometrically, and the solver still staircases them, but
they carry no acoustic contrast, so a perfect solver returns nothing.

Whatever coda comes back is pure staircase artefact, and it is scattered
by EXACTLY the facets the facet model uses.  So this asks the one question
the bicrystal could not: does the artefact alone reproduce the facet
correlation?

  geometry-only predictor on the REAL specimen  ->  residual r = +0.465
  full facet model on the REAL specimen         ->  residual r = +0.511

If the zero-contrast run scores near 0.465, the facet result is staircase
and the whole line collapses.  If it scores near 0, the result is
confirmed against the one confound that could plausibly fake it.

Writes a proper sweep directory so the control is reusable.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vendor"))))
import fdtd                                            # noqa: E402
from rotation_test import rotated_grid                 # noqa: E402
from specimen import DiskSpecimen                      # noqa: E402

OUT = ((os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "out", "sweeps")) +
       r"\zerocontrast_ppw8"))
C_REF, F0, PPW, ORDER = 3850.0, 2.0e6, 8.0, 8
SPONGE, DAMP, ELEM, RECF, KHM = 10, 0.02, 6.35e-3, 2.7, 2.0
DIA, THK = 0.100, 0.035
os.makedirs(OUT, exist_ok=True)
json.dump({"f0_mhz": 2.0, "ppw": PPW, "order": ORDER, "az_start": 0,
           "az_stop": 360, "az_step": 6, "record_factor": RECF,
           "fluid_damp": DAMP, "sponge": SPONGE, "element_d_mm": 6.35,
           "diameter_mm": 100.0, "thickness_mm": 35.0, "n_grains": 100,
           "size_cv": 0.35, "concentration": -8.0, "spatial_corr": 0.0,
           "fabric_axis": [1.0, 0.0, 0.0], "seed": 11,
           "axes_convention": "rigid2", "fd_kh_max": KHM,
           "rasterise": "single", "name": "zerocontrast_ppw8",
           "NOTE": "ZERO-CONTRAST CONTROL: every grain given the SAME "
                   "c-axis, so the tessellation is acoustically invisible. "
                   "Any coda here is pure staircase artefact."},
          open(os.path.join(OUT, "config.json"), "w"), indent=1)

h = C_REF / F0 / PPW
CO = fdtd.optimised_coeffs(ORDER, kh_max=KHM, multistart=True)
sp = DiskSpecimen(diameter_m=DIA, thickness_m=THK, n_grains=100,
                  size_cv=0.35, concentration=-8.0, spatial_corr=0.0,
                  fabric_axis=(1.0, 0.0, 0.0), seed=11)
build = sp.build(h)
print(f"specimen built, h={h*1e3:.3f} mm", flush=True)

AZ = list(range(0, 360, 6))
times = []
for i, az in enumerate(AZ):
    f = os.path.join(OUT, f"az{az:03d}.npz")
    if os.path.exists(f):
        continue
    t0 = time.time()
    lab, nd, m, axr = rotated_grid(build, h, SPONGE, az, single_raster=True)
    uni = np.tile(axr[0][None, :], (len(axr), 1))      # EVERY grain the same
    nz, nyy = lab.shape[0], lab.shape[1]
    Ct, rho_t = fdtd.material_tables(uni)
    dt = fdtd.safe_dt_labels((lab + 1).astype(np.uint8), Ct, rho_t, h, CO,
                             safety=0.5)
    D = nd * h
    nt = int(RECF * D / C_REF / dt)
    wav = fdtd.ricker(F0, dt, nt)
    dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    dm = np.exp(-DAMP * dist).astype(np.float32)
    cz, cy, cx = nz // 2, nyy // 2, nyy // 2
    ixp = next(cx + k - 1 for k in range(nd // 2 - 1, 0, -1)
               if lab[cz, cy, cx + k] >= 0)
    er = max(int(ELEM / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
           for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er and lab[cz + dz, cy + dy, ixp] >= 0]
    w = 1.0 / len(pts)
    tr = np.asarray(fdtd.forward_fused_labels(
        lab, uni, h, dt, nt, [(p, w) for p in pts], wav,
        [(pts, np.full(len(pts), w))], order=ORDER, coeffs=CO,
        sponge_width=SPONGE, damp_mask=dm), float).ravel()
    np.savez_compressed(f, trace=tr.astype(np.float32), dt=dt, az=az)
    times.append(time.time() - t0)
    left = (len(AZ) - i - 1) * float(np.mean(times)) / 60
    print(f"  az {az:3d}  ({i+1}/{len(AZ)})  {times[-1]:.0f} s   "
          f"~{left:.0f} min left", flush=True)
print("SWEEP DONE", flush=True)
