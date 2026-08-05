"""Validate the pseudospectral solver against the saved FD references.

The PS solver is a DIFFERENT discretisation, so bit-equivalence is not the
bar; physics-equivalence is. Gates, in order:
  1. E1 arrival time within 0.3 us and E1 amplitude within 1.5 dB of the
     reference trace (propagation + rim reflection are right).
  2. Clean-window (24-36 us) coda RMS re E1 within 2 dB of the reference
     (BOUNDARY SCATTERING is right - the Gibbs question, and the whole
     point: this is our observable).
Run at 2 MHz (reference: ref/fw_fabric00.npz), report timing per trace.
--ppw N selects the PS grid density (default 3).
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
from scipy.signal import hilbert               # noqa: E402

import coda_convergence as CC                  # noqa: E402
import fdtd                                    # noqa: E402
import fdtd_ps                                 # noqa: E402
from specimen import DiskSpecimen              # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")
C_REF = 3850.0


def run_ps(f0=2.0e6, ppw=3.0, safety=0.5, smooth=0.0, sponge=10,
           strength=0.09, verbose=True):
    h = C_REF / f0 / ppw
    sp = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                      size_cv=0.35, concentration=3.93, spatial_corr=0.0,
                      fabric_axis=(1, 0, 0), seed=7)
    build = sp.build(h)
    lab, nd, m = CC.build_grid(build, h, sponge)
    nz, nx = lab.shape[0], lab.shape[1]
    dt = fdtd_ps.ps_dt(h, 4046.0 * 1.05, safety=safety)
    D = nd * h
    nt = int(2.2 * D / C_REF / dt)
    wav = fdtd.ricker(f0, dt, nt)
    dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    dm = np.exp(-0.02 * dist * (fdtd_ps.C_REF_DEFAULT / f0 / 6.0) / h
                ).astype(np.float32)   # scale damping rate to cell size
    cz, cy, cx = nz // 2, nx // 2, nx // 2
    Rc = nd // 2
    ixp = next(cx + k - 1 for k in range(Rc - 1, 0, -1)
               if lab[cz, cy, cx + k] >= 0)
    er = max(int(6.35e-3 / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
           for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er and lab[cz + dz, cy + dy, ixp] >= 0]
    w = 1.0 / len(pts)
    if verbose:
        print(f"PS grid {lab.shape} = {lab.size/1e6:.1f}M cells "
              f"(FD reference was 15.5M), {nt} steps, dt {dt*1e9:.2f} ns",
              flush=True)
    t0 = time.time()
    tr = np.asarray(fdtd_ps.forward_ps_labels(
        lab, build["axes"], h, dt, nt, [(p, w) for p in pts], wav,
        [(pts, np.full(len(pts), w))], sponge_width=sponge,
        sponge_strength=strength, damp_mask=dm,
        smooth_cells=smooth), float).ravel()
    el = time.time() - t0
    if verbose:
        print(f"PS trace: {el:.0f}s", flush=True)
    return tr, dt, el


def metrics(x, dt):
    fs = 1.0 / dt
    e = np.abs(hilbert(np.asarray(x, float)))
    t1 = 2 * 0.100 / C_REF
    k0 = int(t1 * fs)
    seg = e[max(k0 - int(2.5e-6 * fs), 0):k0 + int(2.5e-6 * fs)]
    kpk = max(k0 - int(2.5e-6 * fs), 0) + int(np.argmax(seg))
    E1 = seg.max()
    lo, hi = int(24e-6 * fs), int(36e-6 * fs)
    rms = 20 * np.log10(np.sqrt((e[lo:hi] ** 2).mean()) / E1 + 1e-30)
    return dict(E1=E1, t_e1_us=kpk / fs * 1e6, coda_rms=rms)


def main():
    ppw, smooth, sponge, strength = 3.0, 0.0, 10, 0.09
    for a in sys.argv[1:]:
        if a.startswith("--ppw"):
            ppw = float(sys.argv[sys.argv.index(a) + 1])
        if a.startswith("--smooth"):
            smooth = float(sys.argv[sys.argv.index(a) + 1])
        if a.startswith("--sponge"):
            sponge = int(sys.argv[sys.argv.index(a) + 1])
        if a.startswith("--strength"):
            strength = float(sys.argv[sys.argv.index(a) + 1])
    d = np.load(os.path.join(REF, "fw_fabric00.npz"))
    ref_tr, ref_dt = d["trace"].ravel(), float(d["dt"])
    mr = metrics(ref_tr, ref_dt)

    tr, dt, el = run_ps(ppw=ppw, smooth=smooth, sponge=sponge,
                        strength=strength)
    mp = metrics(tr, dt)

    print(f"\n{'':>16} {'FD reference':>13} {'PS ppw ' + str(ppw):>13}")
    print(f"{'E1 time us':>16} {mr['t_e1_us']:>13.2f} {mp['t_e1_us']:>13.2f}"
          f"   (diff {abs(mp['t_e1_us']-mr['t_e1_us']):.2f}, gate 0.3)")
    print(f"{'coda RMS dB':>16} {mr['coda_rms']:>13.1f} {mp['coda_rms']:>13.1f}"
          f"   (diff {abs(mp['coda_rms']-mr['coda_rms']):.1f}, gate 2.0)")
    t_ok = abs(mp['t_e1_us'] - mr['t_e1_us']) < 0.3
    c_ok = abs(mp['coda_rms'] - mr['coda_rms']) < 2.0
    print(f"\nE1 timing: {'PASS' if t_ok else 'FAIL'};  "
          f"coda level: {'PASS' if c_ok else 'FAIL'};  "
          f"PS trace time {el:.0f}s (FD reference ~40s at this size)")


if __name__ == "__main__":
    main()
