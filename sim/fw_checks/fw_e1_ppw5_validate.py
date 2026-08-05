"""Validate ppw5 for the E1 LEG of the split (corridor + coarser grid).

The coda needs ppw6 (convergence series flattens there; ppw5 is 8 dB
contaminated - settled). E1 is a SPECULAR arrival integrated over a
~6 mm Fresnel zone, so its ppw requirement is its own question,
answered here by measurement: corridor grid at ppw5, full record, vs
the ppw6 full-disk reference. License criteria: ToF within 20 ns and
amplitude offset < 0.5 dB treated as a CONSTANT calibration (the same
grid serves every azimuth; an azimuth-dependence spot check at a
walk-off azimuth is noted as future hardening). Combined leg cost
~0.25x -> split total ~0.70x per azimuth.
"""
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys
import time

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
from scipy import ndimage                     # noqa: E402

import coda_convergence as CC                  # noqa: E402
import fdtd                                    # noqa: E402
from specimen import DiskSpecimen              # noqa: E402
from fw_e1_corridor_validate import e1_of, REF, C_REF, F0, HALF_W  # noqa: E402


def main():
    d = np.load(os.path.join(REF, "fw_reference_5mhz_fabric00.npz"))
    E1r, t1r = e1_of(d["trace"], float(d["dt"]))

    h = C_REF / F0 / 5.0                       # ppw 5
    sp = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                      size_cv=0.35, concentration=3.93, spatial_corr=0.0,
                      fabric_axis=(1, 0, 0), seed=7)
    print("building ppw5 specimen...", flush=True)
    build = sp.build(h)
    lab, nd, m = CC.build_grid(build, h, 10)
    nz, nyy = lab.shape[0], lab.shape[1]
    co = fdtd.optimised_coeffs(8)
    mats = (lab + 1).astype(np.uint8)
    Ct, rho_t = fdtd.material_tables(build["axes"])
    dt = fdtd.safe_dt_labels(mats, Ct, rho_t, h, co, safety=0.5)
    D = nd * h
    nt = int(2.2 * D / C_REF / dt)
    wav = fdtd.ricker(F0, dt, nt)
    dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    dm = np.exp(-0.02 * dist).astype(np.float32)

    cz, cy, cx = nz // 2, nyy // 2, nyy // 2
    Rc = nd // 2
    ixp = next(cx + k - 1 for k in range(Rc - 1, 0, -1)
               if lab[cz, cy, cx + k] >= 0)
    er = max(int(6.35e-3 / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
           for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er
           and lab[cz + dz, cy + dy, ixp] >= 0]
    w = 1.0 / len(pts)
    ky = int(HALF_W / h)
    y0, y1 = cy - ky, cy + ky + 1
    lab_c = np.ascontiguousarray(lab[:, y0:y1, :])
    dm_c = np.ascontiguousarray(dm[:, y0:y1, :])
    pts_c = [(a, b - y0, c) for a, b, c in pts]
    print(f"ppw5 corridor {lab_c.shape}, nt {nt}", flush=True)

    t0 = time.time()
    tr = np.asarray(fdtd.forward_fused_labels(
        lab_c, build["axes"], h, dt, nt, [(p, w) for p in pts_c], wav,
        [(pts_c, np.full(len(pts_c), w))], order=8, coeffs=co,
        sponge_width=10, damp_mask=dm_c), float).ravel()
    el = time.time() - t0
    print(f"ppw5 corridor run: {el/60:.1f} min "
          f"(vs 15-17 full ppw6)", flush=True)

    E1c, t1c = e1_of(tr, dt)
    off = 20 * np.log10(E1c / E1r + 1e-30)
    print(f"\nE1 amplitude: ppw6-full {E1r:.5g} vs ppw5-corridor "
          f"{E1c:.5g} -> offset {off:+.3f} dB")
    print(f"E1 ToF: {t1r*1e6:.4f} vs {t1c*1e6:.4f} us "
          f"-> shift {abs(t1c-t1r)*1e9:.1f} ns")
    ok = abs(off) < 0.5 and abs(t1c - t1r) < 20e-9
    print(f"-> {'PASS (ppw5 E1 leg licensed, offset = calibration const)'
          if ok else 'FAIL (E1 leg stays ppw6-corridor)'}", flush=True)


if __name__ == "__main__":
    main()
