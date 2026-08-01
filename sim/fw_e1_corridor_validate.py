"""Validate the E1 BEAM CORRIDOR at 5 MHz vs ref/fw_5mhz_fabric00.npz.

The coda/E1 split's second half: E1 (the far-rim specular echo) forms
in the stationary-phase region around the beam axis - first Fresnel
zone ~6.3 mm at 100 mm range, plus anisotropic walk-off up to ~14 mm.
Cropping the grid to |y| <= 27 mm about the axis keeps ~0.55x the
cells for the full-length record. UNLIKE the coda crop this is NOT
provable by causality (the cut faces are parallel to the beam), so it
is licensed by MEASUREMENT: E1 amplitude within 1% and ToF within a
few ns of the full-disk reference, else FAIL and the split ships
coda-crop-only. Coda levels from corridor runs are NOT valid and are
printed only to document how wrong they are.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
from scipy import ndimage                     # noqa: E402
from scipy.signal import hilbert              # noqa: E402

import coda_convergence as CC                  # noqa: E402
import fdtd                                    # noqa: E402
from specimen import DiskSpecimen              # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")
C_REF = 3850.0
F0 = 5.0e6
HALF_W = 27e-3                                 # corridor half-width in y


def e1_of(tr, dt):
    fs = 1.0 / dt
    e = np.abs(hilbert(tr))
    k0 = int(2 * 0.100 / C_REF * fs)
    a, b = max(k0 - int(2e-6 * fs), 0), k0 + int(2e-6 * fs)
    k = a + int(np.argmax(e[a:b]))
    return e[k], k * dt


def main():
    d = np.load(os.path.join(REF, "fw_5mhz_fabric00.npz"))
    tr_ref, dt_ref = d["trace"], float(d["dt"])
    E1r, t1r = e1_of(tr_ref, dt_ref)

    h = C_REF / F0 / 6.0
    sp = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                      size_cv=0.35, concentration=3.93, spatial_corr=0.0,
                      fabric_axis=(1, 0, 0), seed=7)
    print("building 5 MHz specimen (GPU labels)...", flush=True)
    build = sp.build(h)
    lab, nd, m = CC.build_grid(build, h, 10)
    nz, nyy = lab.shape[0], lab.shape[1]
    co = fdtd.optimised_coeffs(8)
    mats = (lab + 1).astype(np.uint8)
    Ct, rho_t = fdtd.material_tables(build["axes"])
    dt = fdtd.safe_dt_labels(mats, Ct, rho_t, h, co, safety=0.5)
    assert abs(dt - dt_ref) < 1e-15
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
    print(f"full {lab.shape} -> corridor {lab_c.shape} "
          f"({lab_c.size/lab.size:.2f}x cells), nt {nt}", flush=True)

    t0 = time.time()

    def prog(f):
        if f > 0.02:
            el = time.time() - t0
            print(f"   {100*f:5.1f}%  eta {el/f*(1-f):4.0f}s", flush=True)

    tr_c = np.asarray(fdtd.forward_fused_labels(
        lab_c, build["axes"], h, dt, nt, [(p, w) for p in pts_c], wav,
        [(pts_c, np.full(len(pts_c), w))], order=8, coeffs=co,
        sponge_width=10, damp_mask=dm_c, progress=prog), float).ravel()
    print(f"corridor run: {(time.time()-t0)/60:.1f} min", flush=True)
    np.savez(os.path.join(REF, "fw_5mhz_e1corridor.npz"), trace=tr_c,
             dt=dt, half_w=HALF_W)

    E1c, t1c = e1_of(tr_c, dt)
    fs = 1.0 / dt
    e_r = np.abs(hilbert(tr_ref)); e_c = np.abs(hilbert(tr_c))
    lo, hi = int(24e-6 * fs), int(36e-6 * fs)
    cr = 20*np.log10(np.sqrt((e_r[lo:hi]**2).mean())/E1r + 1e-30)
    cc = 20*np.log10(np.sqrt((e_c[lo:hi]**2).mean())/E1c + 1e-30)
    damp_ratio = E1c / E1r
    print(f"\nE1 amplitude: full {E1r:.5g} vs corridor {E1c:.5g} "
          f"-> ratio {damp_ratio:.4f} ({20*np.log10(damp_ratio):+.3f} dB)")
    print(f"E1 ToF: full {t1r*1e6:.4f} vs corridor {t1c*1e6:.4f} us "
          f"-> shift {abs(t1c-t1r)*1e9:.1f} ns")
    print(f"(coda re own E1, NOT licensed: full {cr:.1f} vs "
          f"corridor {cc:.1f} dB)")
    ok = abs(damp_ratio - 1) < 0.01 and abs(t1c - t1r) < 20e-9
    print(f"-> {'PASS (E1 corridor licensed)' if ok else 'FAIL'}",
          flush=True)


if __name__ == "__main__":
    main()
