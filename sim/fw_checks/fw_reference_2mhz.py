"""Regenerate and SAVE the full-wave reference traces for the ladder.

The ladder's four reference levels (-28.1 / -32.9 / -31.6 / -43.4 dB) were
measured from traces that were never written to disk, which blocked every
later shape diagnostic. This script reproduces them with the validated
solver settings (order 8, ppw 6, safety 0.5, fluid damping 0.01, Ricker)
and the EXACT fabric protocol from ladder.py, then saves each trace to
sim/ref/fw_fabric{00,25,50,100}.npz.

Validation: the printed levels must come out at the recorded values above
(within a fraction of a dB); if they do not, the run does NOT reproduce the
reference and must not be saved over it.

Takes about 275 s per trace on the GPU; four traces ~ 18 minutes.
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
from ringfwi import anisotropy as an          # noqa: E402
from scipy import ndimage                     # noqa: E402
from scipy.signal import hilbert              # noqa: E402

import coda_convergence as CC                  # noqa: E402
import fdtd                                    # noqa: E402
import ladder                                  # noqa: E402

REF_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ref")


def coda_level(x, dt):
    """The ladder metric, self-referenced to the trace's own E1."""
    t1 = 2 * 0.100 / 3850.0
    e = np.abs(hilbert(x))
    k0 = int(t1 / dt)
    E1 = e[max(k0 - int(2e-6 / dt), 0):min(k0 + int(2e-6 / dt), len(e) - 1)].max()
    lo, hi = int(22e-6 / dt), min(int(42e-6 / dt), len(e))
    return 20 * np.log10(e[lo:hi].max() / E1 + 1e-30), E1


def main():
    os.makedirs(REF_DIR, exist_ok=True)
    build = ladder.standard_build()
    fabrics = ladder.standard_fabrics(build["axes"])

    h = 3850.0 / 2e6 / 6.0
    lab, nd, m = CC.build_grid(build, h, 10)
    nz, nx = lab.shape[0], lab.shape[1]
    co = fdtd.optimised_coeffs(8)
    C0, rho = an.polycrystal_stiffness_3d(lab, build["axes"],
                                          c_couplant=1500.0, rho_couplant=300.0)
    dt = fdtd.safe_dt(C0, rho, h, co, safety=0.5)
    D = nd * h
    nt = int(2.6 * D / 3850.0 / dt)
    wav = fdtd.ricker(2e6, dt, nt)
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
    print(f"grid {lab.shape} = {lab.size/1e6:.1f}M, {nt} steps, "
          f"dt {dt*1e9:.2f} ns (protocol says 21.64)", flush=True)

    # The pre-2026-07-25 recorded levels (-28.1/-32.9/-31.6/-43.4) were
    # measured on instability-contaminated traces (fluid_damp 0.01); with
    # the 0.02 damping the coda comes out LOWER by design. The pass
    # criterion is now a DEAD TAIL (last-10% max < 0.1 x E1), not matching
    # the old numbers. Runs on the labels solver (bit-identical, faster).
    recorded = ladder.FW
    for frac, axes, tag, rec in zip(ladder.FRACS, fabrics,
                                    ("00", "25", "50", "100"), recorded):
        t0 = time.time()
        tr = np.asarray(fdtd.forward_fused_labels(
            lab, axes, h, dt, nt, [(p, w) for p in pts], wav,
            [(pts, np.full(len(pts), w))], order=8, coeffs=co,
            sponge_width=10, damp_mask=dm), float).ravel()
        lvl, E1 = coda_level(tr, dt)
        tail = float(np.abs(tr[int(0.9 * len(tr)):]).max() / max(E1, 1e-30))
        print(f"fabric {tag:>3}%: coda {lvl:6.1f} dB re E1 "
              f"(contaminated-era {rec}), E1 {E1:.5f}, tail/E1 {tail:.3f} "
              f"({'STABLE' if tail < 0.1 else 'UNSTABLE'}), "
              f"{time.time()-t0:.0f}s", flush=True)
        np.savez(os.path.join(REF_DIR, f"fw_fabric{tag}.npz"),
                 trace=tr, dt=dt, f0=2e6, E1=E1, level_db=lvl,
                 frac=frac, h=h, order=8, ppw=6, seed_specimen=7,
                 seed_fabric=11, fluid_damp=0.02)
    print("saved to", REF_DIR, flush=True)


if __name__ == "__main__":
    main()
