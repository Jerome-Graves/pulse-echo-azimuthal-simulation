"""Paper A dataset: a resolution ladder with a MEASURED numerical floor at
each rung, all on the current code path.

Three sweeps, run back to back:
  zerocontrast_ppw6    60 az, ppw 6   -> floor at ppw 6
  girdle_perp_ppw10    30 az, ppw 10  -> is ppw 8 converged?
  zerocontrast_ppw10   30 az, ppw 10  -> floor at ppw 10

12 deg steps at ppw 10 so every azimuth has an exact partner in the
existing 6 deg ppw 6 and ppw 8 sweeps: the comparison is PAIRED, so the
speckle is common between resolutions and cancels.

Zero-contrast = identical tessellation, identical grid, every grain given
the SAME c-axis, so the boundaries are geometrically present and still
staircased but carry no acoustic contrast.  Whatever coda comes back is
the numerical floor, measured rather than inferred.

Everything else is the production recipe: order 8, multistart coefficients
at kh_max 2.0, single rasterisation, sponge 10, fluid damp 0.02,
record_factor 2.7, 6.35 mm element.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
import fdtd                                            # noqa: E402
from rotation_test import rotated_grid                 # noqa: E402
from specimen import DiskSpecimen                      # noqa: E402

SWD = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "sweeps")))
C_REF, F0, ORDER = 3850.0, 2.0e6, 8
SPONGE, DAMP, ELEM, RECF, KHM = 10, 0.02, 6.35e-3, 2.7, 2.0
DIA, THK, SEED, KAPPA, AXIS = 0.100, 0.035, 11, -8.0, (1.0, 0.0, 0.0)

#     name                 ppw   step  zero-contrast?
JOBS = [("zerocontrast_ppw6", 6.0, 6, True),
        ("girdle_perp_ppw10", 10.0, 12, False),
        ("zerocontrast_ppw10", 10.0, 12, True)]


def run(name, ppw, step, zero):
    out = os.path.join(SWD, name)
    os.makedirs(out, exist_ok=True)
    cfg = {"f0_mhz": 2.0, "ppw": ppw, "order": ORDER, "az_start": 0,
           "az_stop": 360, "az_step": step, "record_factor": RECF,
           "fluid_damp": DAMP, "sponge": SPONGE, "element_d_mm": 6.35,
           "diameter_mm": 100.0, "thickness_mm": 35.0, "n_grains": 100,
           "size_cv": 0.35, "concentration": KAPPA, "spatial_corr": 0.0,
           "fabric_axis": list(AXIS), "seed": SEED,
           "axes_convention": "rigid2", "fd_kh_max": KHM,
           "rasterise": "single", "name": name}
    if zero:
        cfg["NOTE"] = ("ZERO-CONTRAST CONTROL: every grain given the SAME "
                       "c-axis, so the tessellation is acoustically "
                       "invisible. Any coda here is numerical floor.")
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)

    h = C_REF / F0 / ppw
    co = fdtd.optimised_coeffs(ORDER, kh_max=KHM, multistart=True)
    build = DiskSpecimen(diameter_m=DIA, thickness_m=THK, n_grains=100,
                         size_cv=0.35, concentration=KAPPA,
                         spatial_corr=0.0, fabric_axis=AXIS,
                         seed=SEED).build(h)
    az = list(range(0, 360, step))
    print(f"\n=== {name}: ppw {ppw:.0f}, {len(az)} az, h={h*1e3:.3f} mm, "
          f"{'ZERO-CONTRAST' if zero else 'real fabric'}", flush=True)
    times = []
    for i, a in enumerate(az):
        f = os.path.join(out, f"az{a:03d}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        lab, nd, m, axr = rotated_grid(build, h, SPONGE, a,
                                       single_raster=True)
        axes = np.tile(axr[0][None, :], (len(axr), 1)) if zero else axr
        nz, nyy = lab.shape[0], lab.shape[1]
        Ct, rho_t = fdtd.material_tables(axes)
        dt = fdtd.safe_dt_labels((lab + 1).astype(np.uint8), Ct, rho_t, h,
                                 co, safety=0.5)
        nt = int(RECF * nd * h / C_REF / dt)
        wav = fdtd.ricker(F0, dt, nt)
        dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
        dm = np.exp(-DAMP * dist).astype(np.float32)
        cz, cy, cx = nz // 2, nyy // 2, nyy // 2
        ixp = next(cx + k - 1 for k in range(nd // 2 - 1, 0, -1)
                   if lab[cz, cy, cx + k] >= 0)
        er = max(int(ELEM / 2 / h), 1)
        pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
               for dz in range(-er, er + 1)
               if dy * dy + dz * dz <= er * er
               and lab[cz + dz, cy + dy, ixp] >= 0]
        w = 1.0 / len(pts)
        tr = np.asarray(fdtd.forward_fused_labels(
            lab, axes, h, dt, nt, [(p, w) for p in pts], wav,
            [(pts, np.full(len(pts), w))], order=ORDER, coeffs=co,
            sponge_width=SPONGE, damp_mask=dm), float).ravel()
        np.savez_compressed(f, trace=tr.astype(np.float32), dt=dt, az=a)
        times.append(time.time() - t0)
        left = (len(az) - i - 1) * float(np.mean(times)) / 60
        print(f"  {name} az {a:3d} ({i+1}/{len(az)}) grid {lab.shape} "
              f"{nt} steps {times[-1]:.0f} s  ~{left:.0f} min left",
              flush=True)
    print(f"=== {name} DONE", flush=True)


for nm, p, s, z in JOBS:
    run(nm, p, s, z)
print("\nALL LADDER RUNS COMPLETE", flush=True)
