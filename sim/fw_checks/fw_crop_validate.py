"""Validate the CAUSAL CODA CROP against the saved 2 MHz reference.

Claim to prove: for a monostatic transducer, grid cells farther than
R = c*t/2 behind the causal horizon cannot influence the trace before
t, so cropping them (and stopping the clock at the window end) changes
the scored coda window by NOTHING. R uses c = 4200 m/s (above the
fastest physical qP 4046 AND the order-8 grid's slightly superluminal
numerical precursors) plus a 2 mm margin; even the cut-face reflection
and the relocated sponge live behind the horizon.

Protocol: exact fw_reference_2mhz setup (ladder.standard_build, fabric00,
h = lambda/6 at 2 MHz, damp 0.02), cropped run vs ref/fw_fabric00.npz.
PASS if the window-region difference is <= -70 dB re E1 (a -70 dB
perturbation moves the -51 dB coda RMS by < 0.1 dB).
The mechanism is frequency-independent: a PASS here licenses the same
crop at 5 MHz (fw_reference_5mhz coda runs -> ~0.45x cost).
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
from scipy.signal import hilbert              # noqa: E402

import coda_convergence as CC                  # noqa: E402
import fdtd                                    # noqa: E402
import ladder                                  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")
T_WIN_END = 36e-6                              # scored window ends here
T_KEEP = 38e-6                                 # record the crop run to this
C_CAUSAL = 4200.0                              # > qP max AND numerical vmax
MARGIN = 2e-3


def main():
    d = np.load(os.path.join(REF, "fw_fabric00.npz"))
    tr_full, dt_ref = d["trace"], float(d["dt"])
    E1 = float(d["E1"])

    build = ladder.standard_build()
    axes = ladder.standard_fabrics(build["axes"])[0]
    h = 3850.0 / 2e6 / 6.0
    lab, nd, m = CC.build_grid(build, h, 10)
    nz, nyy = lab.shape[0], lab.shape[1]
    co = fdtd.optimised_coeffs(8)
    mats = (lab + 1).astype(np.uint8)
    Ct, rho_t = fdtd.material_tables(axes)
    dt = fdtd.safe_dt_labels(mats, Ct, rho_t, h, co, safety=0.5)
    assert abs(dt - dt_ref) < 1e-15, (dt, dt_ref)
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

    R = C_CAUSAL * T_KEEP / 2 + MARGIN
    ix0 = max(int(ixp - R / h) - 1, 0)
    lab_c = np.ascontiguousarray(lab[:, :, ix0:])
    dm_c = np.ascontiguousarray(dm[:, :, ix0:])
    pts_c = [(a, b, c - ix0) for a, b, c in pts]
    nt = int(T_KEEP / dt)
    wav = fdtd.ricker(2e6, dt, nt)
    print(f"full grid {lab.shape} -> cropped {lab_c.shape} "
          f"({lab_c.size/lab.size:.2f}x cells), nt {nt} "
          f"({nt*dt*1e6:.1f} us vs full {len(tr_full)*dt*1e6:.1f} us) "
          f"-> cost {lab_c.size/lab.size*nt/len(tr_full):.2f}x", flush=True)

    t0 = time.time()
    tr_c = np.asarray(fdtd.forward_fused_labels(
        lab_c, axes, h, dt, nt, [(p, w) for p in pts_c], wav,
        [(pts_c, np.full(len(pts_c), w))], order=8, coeffs=co,
        sponge_width=10, damp_mask=dm_c), float).ravel()
    print(f"cropped run: {time.time()-t0:.0f}s", flush=True)

    n_win = int(T_WIN_END / dt)
    diff = tr_c[:n_win] - tr_full[:n_win]
    env = np.abs(hilbert(diff))
    mx = 20 * np.log10(env.max() / E1 + 1e-30)
    # where the coda actually gets scored
    lo = int(24e-6 / dt)
    rms = 20 * np.log10(np.sqrt((env[lo:] ** 2).mean()) / E1 + 1e-30)
    print(f"\nCROP DIFF through {T_WIN_END*1e6:.0f} us: "
          f"max {mx:.1f} dB re E1, coda-window RMS {rms:.1f} dB re E1\n"
          f"-> {'PASS (causal crop licensed)' if mx <= -70 else 'FAIL'}",
          flush=True)


if __name__ == "__main__":
    main()
