"""First full-wave trace at the REAL probe frequency: 5 MHz, full disk.

Runs on the label-indexed solver (forward_fused_labels): 194M cells in
~8 GB, ~2.6 h on the RTX 4070. Saves ref/fw_5mhz_fabric00.npz and prints
the two numbers every extrapolation in this project rests on:
  * clean-window coda level (fast-model validation at 5 MHz)
  * per-boundary blip visibility vs the model prediction and the
    3 MHz-measured trend (freq_blips.py)
"""
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys
import time

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor"))))
from scipy import ndimage                     # noqa: E402
from scipy.signal import hilbert              # noqa: E402

import coda_convergence as CC                  # noqa: E402
import fdtd                                    # noqa: E402
from specimen import DiskSpecimen              # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")
C_REF = 3850.0
F0 = 5.0e6


def main():
    h = C_REF / F0 / 6.0
    sp = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                      size_cv=0.35, concentration=3.93, spatial_corr=0.0,
                      fabric_axis=(1, 0, 0), seed=7)
    print("building 5 MHz specimen + grid (CPU)...", flush=True)
    build = sp.build(h)
    lab, nd, m = CC.build_grid(build, h, 10)
    nz, nx = lab.shape[0], lab.shape[1]
    co = fdtd.optimised_coeffs(8)
    mats = (lab + 1).astype(np.uint8)
    Ct, rho_t = fdtd.material_tables(build["axes"])
    dt = fdtd.safe_dt_labels(mats, Ct, rho_t, h, co, safety=0.5)
    D = nd * h
    # record to 2.2 D/c = 57 us: covers E1 (52 us) with margin; the old
    # 2.6 factor bought nothing but 15% more steps
    nt = int(2.2 * D / C_REF / dt)
    wav = fdtd.ricker(F0, dt, nt)
    dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    dm = np.exp(-0.02 * dist).astype(np.float32)   # 0.02: rim instability dies (0.015 lethal, 0.02 adds margin) (see validate_labels.setup)
    cz, cy, cx = nz // 2, nx // 2, nx // 2
    Rc = nd // 2
    ixp = next(cx + k - 1 for k in range(Rc - 1, 0, -1)
               if lab[cz, cy, cx + k] >= 0)
    er = max(int(6.35e-3 / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
           for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er and lab[cz + dz, cy + dy, ixp] >= 0]
    w = 1.0 / len(pts)
    print(f"grid {lab.shape} = {lab.size/1e6:.0f}M, {nt} steps, "
          f"dt {dt*1e9:.2f} ns, element {len(pts)} cells", flush=True)

    t0 = time.time()

    def prog(f):
        el = time.time() - t0
        if f > 0.005:
            print(f"   {100*f:5.1f}%  {el:6.0f}s elapsed, "
                  f"eta {el/f*(1-f):5.0f}s", flush=True)

    tr = np.asarray(fdtd.forward_fused_labels(
        lab, build["axes"], h, dt, nt, [(p, w) for p in pts], wav,
        [(pts, np.full(len(pts), w))], order=8, coeffs=co, sponge_width=10,
        damp_mask=dm, progress=prog), float).ravel()
    el = time.time() - t0
    print(f"trace done in {el/60:.0f} min", flush=True)
    np.savez(os.path.join(REF, "fw_5mhz_fabric00.npz"), trace=tr, dt=dt,
             f0=F0, h=h, order=8, ppw=6, seed_specimen=7)

    fs = 1.0 / dt
    e = np.abs(hilbert(tr))
    t1 = 2 * 0.100 / C_REF
    k0 = int(t1 * fs)
    E1 = e[max(k0 - int(2e-6 * fs), 0):k0 + int(2e-6 * fs)].max()
    lo, hi = int(24e-6 * fs), int(36e-6 * fs)
    rms = 20 * np.log10(np.sqrt((e[lo:hi] ** 2).mean()) / E1 + 1e-30)
    mx = 20 * np.log10(e[int(22e-6*fs):int(42e-6*fs)].max() / E1 + 1e-30)
    print(f"E1 {E1:.4g}; coda 24-36us RMS {rms:.1f} dB re E1; "
          f"22-42us max {mx:.1f} dB re E1")
    # decontaminated 2 MHz fabric00 reference (fw_reference.py, damp
    # 0.02); the old print quoted contaminated-era numbers (-36.2/-28.1)
    print("(2 MHz clean reference: coda RMS -51.5, max -41.1)")

    # blip visibility at the predicted boundary times, with speckle null
    import acquisition as A
    from config import Config
    cfg = Config(); cfg.err.enabled = False
    trd = A.truth_along_diameter(build, cfg, 0.0)
    t_bnd = 2.0 * np.cumsum(trd["length_m"] / trd["speed"])[:-1]
    half, k4 = int(1.0e-6 * fs), int(4e-6 * fs)
    print(f"\n{'bnd t us':>9} {'above local bg dB':>18}")
    for tb in t_bnd:
        k = int(tb * fs)
        if k - half < 0 or k + half >= len(e):
            continue
        pk = e[k - half:k + half].max()
        ring = np.r_[e[max(k - k4, 0):k - half],
                     e[k + half:min(k + k4, len(e))]]
        bg = np.median(ring) if len(ring) else 1e-30
        print(f"{tb*1e6:>9.1f} {20*np.log10(pk/bg+1e-30):>+18.1f}")
    rng = np.random.default_rng(3)
    null = []
    while len(null) < 200:
        t = rng.uniform(8e-6, 46e-6)
        if np.abs(t - t_bnd).min() < 2e-6:
            continue
        k = int(t * fs)
        pk = e[k - half:k + half].max()
        ring = np.r_[e[max(k - k4, 0):k - half],
                     e[k + half:min(k + k4, len(e))]]
        null.append(20 * np.log10(pk / np.median(ring) + 1e-30))
    null = np.array(null)
    print(f"speckle null: median {np.median(null):+.1f}, "
          f"99th pct {np.percentile(null, 99):+.1f} dB "
          f"(a blip is real above the 99th)")


if __name__ == "__main__":
    main()
