"""Azimuth sweep of the full-wave coda on ONE specimen.

WHY: two independent 25%-randomisation draws moved the full-wave clean-window
coda by -5.6 and +4.9 dB while the incoherent model responded -0.7 and -0.9
to both. The full-wave window level is glint-dominated: a few partially
coherent facets set it, so it carries ~+/-5 dB of realisation noise that a
mean-field model cannot track per-look. The experiment, however, records 180
azimuths. If the fw level fluctuates across azimuths on a fixed specimen
while its azimuth MEAN approaches the model, then (a) the model predicts the
ensemble mean as intended and (b) the inversion observable must be
azimuth-aggregated coda, not single-look coda.

Reuses rotation_test.rotated_grid: the specimen (grains AND c-axes) is
rotated against the fixed probe, which is exactly a probe azimuth change in
the specimen frame; the model is evaluated at azimuth_rad = +rot on the
unrotated build (sign verified at rot=0).

MEASURED OUTCOMES (2026-07-24, ref/fw_az*.npz):
  * az 76 is a 20 dB outlier BOTH in coda/E1 and in absolute window energy,
    and its E1 is 4-6x weaker than neighbours. This is anisotropic BEAM
    WALK-OFF: group-velocity skew peaks near psi ~ 70-75 deg (zero at 0,
    50.4, 90), so the round-trip return walks ~14 mm off the 6.35 mm
    element; the missed energy rattles on as multipath and floods the coda
    window. Consequences: (1) never normalise coda by the SAME azimuth's E1
    under a single-max fabric - use a common calibration E1; (2) coda at
    skew-max azimuths measures bulk-fabric multipath, not grain scattering,
    and the skew angle is computable from the fabric estimate, so those
    azimuths can be predicted and excluded or used as their own observable;
    (3) the E1(az) fade pattern itself encodes the mean c-axis direction -
    a free extra fabric observable the rig already records.
  * Over the six walk-off-clean azimuths (0/17/31/47/62/90), fw vs rung 3
    in the clean window (24-36 us, RMS, common E1): mean offset +2.4 dB,
    per-azimuth scatter 2.1 dB, correlation +0.76 (the model tracks 58% of
    the azimuth-to-azimuth variance). The mean-field model predicts the
    azimuth-resolved coda to ~2 dB once walk-off azimuths are handled,
    which is the accuracy the amplitude inversion needs.
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

import born                                    # noqa: E402
import fdtd                                    # noqa: E402
import ladder                                  # noqa: E402
from rotation_test import rotated_grid         # noqa: E402

REF_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ref")
W0, W1 = ladder.W_CLEAN


def fw_at_rotation(build, rot_deg, order=8, ppw=6.0, f0=2.0e6, sponge=10):
    c_ref = 3850.0
    h = c_ref / f0 / ppw
    lab, nd, m = None, None, None
    lab, nd, m, axes_rot = rotated_grid(build, h, sponge, rot_deg)
    nz, nx = lab.shape[0], lab.shape[1]
    co = fdtd.optimised_coeffs(order)
    # labels solver: bit-identical to the dense path (gated), ~7x faster
    mats = (lab + 1).astype(np.uint8)
    Ct, rho_t = fdtd.material_tables(axes_rot)
    dt = fdtd.safe_dt_labels(mats, Ct, rho_t, h, co, safety=0.5)
    D = nd * h
    nt = int(2.6 * D / c_ref / dt)
    wav = fdtd.ricker(f0, dt, nt)
    dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    dm = np.exp(-0.02 * dist).astype(np.float32)
    cz, cy, cx = nz // 2, nx // 2, nx // 2
    Rc = nd // 2
    ixp = next(cx + k - 1 for k in range(Rc - 1, 0, -1)
               if lab[cz, cy, cx + k] >= 0)
    er = max(int(6.35e-3 / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
           for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er and lab[cz + dz, cy + dy, ixp] >= 0]
    w = 1.0 / len(pts)
    tr = np.asarray(fdtd.forward_fused_labels(
        lab, axes_rot, h, dt, nt, [(p, w) for p in pts], wav,
        [(pts, np.full(len(pts), w))], order=order, coeffs=co,
        sponge_width=sponge, damp_mask=dm), float).ravel()
    return tr, dt


def clean_level(tr, dt):
    e = np.abs(hilbert(tr))
    t1 = 2 * 0.100 / 3850.0
    k0 = int(t1 / dt)
    E1 = e[max(k0 - int(2e-6 / dt), 0):k0 + int(2e-6 / dt)].max()
    w = e[int(W0 / dt):int(W1 / dt)]
    return 20 * np.log10(np.sqrt((w ** 2).mean()) / E1 + 1e-30)


def main():
    build = ladder.standard_build()
    cfg = ladder.standard_cfg()
    dt_m = ladder.DT
    lo, hi = int(W0 / dt_m), int(W1 / dt_m)
    rows = []
    for rot in (17, 31, 47, 62, 76, 90):
        t0 = time.time()
        fp = os.path.join(REF_DIR, f"fw_az{rot:02d}.npz")
        d0 = np.load(fp) if os.path.exists(fp) else None
        if d0 is not None and "fluid_damp" in d0.files:
            tr, dt = d0["trace"].ravel(), float(d0["dt"])   # already clean
        else:
            tr, dt = fw_at_rotation(build, rot)
        lvl = clean_level(tr, dt)
        np.savez(os.path.join(REF_DIR, f"fw_az{rot:02d}.npz"),
                 trace=tr, dt=dt, f0=2e6, rot_deg=rot, level_db=lvl,
                 fluid_damp=0.02)
        _, env = born.boundary_scatter(dict(build), cfg,
                                       azimuth_rad=np.radians(rot))
        w = np.asarray(env, float)[lo:hi]
        m3 = 20 * np.log10(np.sqrt((w ** 2).mean()) / 0.776 + 1e-30)
        rows.append((rot, lvl, m3))
        print(f"az {rot:>3} deg: fw {lvl:6.1f} dB, rung3 {m3:6.1f} dB "
              f"({time.time()-t0:.0f}s)", flush=True)
    fw = np.array([r[1] for r in rows])
    m3 = np.array([r[2] for r in rows])
    print(f"\nfw   across azimuths: mean {fw.mean():.1f}, spread "
          f"{fw.max()-fw.min():.1f} dB, std {fw.std():.1f}")
    print(f"m3   across azimuths: mean {m3.mean():.1f}, spread "
          f"{m3.max()-m3.min():.1f} dB, std {m3.std():.1f}")
    print(f"azimuth-mean offset (fw - m3): {fw.mean()-m3.mean():+.1f} dB")
    if len(fw) > 2:
        print(f"per-azimuth correlation: {np.corrcoef(fw, m3)[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
