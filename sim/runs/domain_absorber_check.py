"""Is the coda independent of the domain size and the absorber width?

Supports Section 4. Every level reported in the paper is measured inside a
box that is larger than the specimen by a fixed pad, the outer cells of
which are damped so that energy leaving the specimen does not return. Two
things could invalidate that. If the pad were too thin, a reflection from
the outer face could arrive inside the coda gate. If the absorber were too
weak, energy would return from the damped region itself; if too strong, it
would present an impedance step and reflect at its entrance, which is the
first error catalogued in Appendix A.

The check is to vary the pad and the damping and confirm the coda does not
move. Three configurations are run on the same specimen and azimuths:

    sponge 10, damping 0.02   the production setting
    sponge 20, damping 0.02   twice the pad, so both the domain and the
                              absorber are wider
    sponge 10, damping 0.01   half the damping at the production pad,
                              which separates absorber STRENGTH from
                              absorber WIDTH

If the production setting is converged, all three agree to well inside
the 0.60 dB ensemble spread of Section 4.

Reads:  nothing. Traces are generated here.
Writes: ../results/domain_absorber.log and .npz
Needs a CUDA device. Six azimuths per configuration, about 40 minutes.
"""
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.signal import butter, hilbert, sosfiltfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('..',):
    _p = os.path.normpath(os.path.join(_HERE, _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, r'C:\Users\Jerome\Documents\GitHub\openUSCT\simulation')

import fdtd                                                # noqa: E402
from rotation_test import rotated_grid                     # noqa: E402
from specimen import DiskSpecimen                          # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))

# Specimen and acquisition of Section 3.
C_REF, F0, ORDER, KH_MAX = 3850.0, 2.0e6, 8, 2.0
DIA, THK, N_GRAINS, SIZE_CV = 0.100, 0.035, 100, 0.35
KAPPA, AXIS, SEED = -8.0, (1.0, 0.0, 0.0), 11
ELEMENT_D, RECORD_FACTOR, PPW = 6.35e-3, 2.7, 8.0

# Coda gate and band of Section 3.
GATE_S, BAND_HZ = (24e-6, 36e-6), (0.8e6, 3.0e6)

AZIMUTHS = (0, 60, 120, 180, 240, 300)
CONFIGS = (('production', 10, 0.02),
           ('wide pad', 20, 0.02),
           ('weak absorber', 10, 0.01))


def note(msg):
    line = '%s  %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(os.path.join(RESULTS, 'domain_absorber.log'), 'a',
              encoding='utf-8') as fh:
        fh.write(line + '\n')


def coda_level_db(trace, dt):
    """Coda RMS referenced to the source, in dB.

    Referenced to the source and not the backwall, because the backwall is
    itself attenuated by scattering and so is not a fixed reference when
    the absorber changes (Section 4, choice of reference).
    """
    fs = 1.0 / dt
    sos = butter(4, [BAND_HZ[0] / (fs / 2), BAND_HZ[1] / (fs / 2)],
                 btype='band', output='sos')
    envelope = np.abs(hilbert(sosfiltfilt(sos, trace)))
    a, b = int(GATE_S[0] * fs), int(GATE_S[1] * fs)
    coda = np.sqrt((envelope[a:b] ** 2).mean())
    return 20.0 * np.log10(coda / np.abs(hilbert(trace)).max())


def run_azimuth(build, h, sponge, damping, azimuth, coeffs):
    lab, n_diam, _m, axes = rotated_grid(build, h, sponge, azimuth,
                                         single_raster=True)
    tables = fdtd.material_tables(axes)
    dt = fdtd.safe_dt_labels((lab + 1).astype(np.uint8), tables[0],
                             tables[1], h, coeffs, safety=0.5)
    n_t = int(RECORD_FACTOR * n_diam * h / C_REF / dt)
    wavelet = fdtd.ricker(F0, dt, n_t)
    distance = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    damp_mask = np.exp(-damping * distance).astype(np.float32)

    nz, ny = lab.shape[0], lab.shape[1]
    cz, cy, cx = nz // 2, ny // 2, ny // 2
    ix = next(cx + k - 1 for k in range(n_diam // 2 - 1, 0, -1)
              if lab[cz, cy, cx + k] >= 0)
    radius = max(int(ELEMENT_D / 2 / h), 1)
    points = [(cz + dz, cy + dy, ix)
              for dy in range(-radius, radius + 1)
              for dz in range(-radius, radius + 1)
              if dy * dy + dz * dz <= radius * radius
              and lab[cz + dz, cy + dy, ix] >= 0]
    weight = 1.0 / len(points)
    trace = np.asarray(fdtd.forward_fused_labels(
        lab, axes, h, dt, n_t, [(p, weight) for p in points], wavelet,
        [(points, np.full(len(points), weight))], order=ORDER,
        coeffs=coeffs, sponge_width=sponge, damp_mask=damp_mask,
        mat_tables=tables), float).ravel()
    return coda_level_db(trace, dt), lab.shape


def main():
    os.makedirs(RESULTS, exist_ok=True)
    note('=' * 66)
    note('Domain-size and absorber-width independence, seed %d, ppw %.0f'
         % (SEED, PPW))
    h = C_REF / F0 / PPW
    coeffs = fdtd.optimised_coeffs(ORDER, kh_max=KH_MAX, multistart=True)
    build = DiskSpecimen(diameter_m=DIA, thickness_m=THK,
                         n_grains=N_GRAINS, size_cv=SIZE_CV,
                         concentration=KAPPA, spatial_corr=0.0,
                         fabric_axis=AXIS, seed=SEED).build(h)

    levels = {}
    for name, sponge, damping in CONFIGS:
        got = []
        for az in AZIMUTHS:
            t0 = time.time()
            db, shape = run_azimuth(build, h, sponge, damping, az, coeffs)
            got.append(db)
            note('  %-14s sponge %2d damp %.3f  az %3d  %8.2f dB  '
                 '%s  %.0f s'
                 % (name, sponge, damping, az, db, shape, time.time() - t0))
        levels[name] = np.array(got)

    note('')
    note('%-16s %10s %8s %12s' % ('configuration', 'mean dB', 'sd', 'vs prod'))
    base = levels['production'].mean()
    for name, _s, _d in CONFIGS:
        v = levels[name]
        note('%-16s %10.2f %8.2f %+12.2f'
             % (name, v.mean(), v.std(ddof=1), v.mean() - base))
    note('')
    note('The ensemble spread across independent tessellations is '
         '0.60 dB (Section 4).')
    note('A configuration difference well inside that is not resolvable '
         'and the')
    note('production setting is converged in that respect.')
    np.savez(os.path.join(RESULTS, 'domain_absorber.npz'),
             azimuths=np.array(AZIMUTHS),
             **{n.replace(' ', '_'): levels[n] for n, _s, _d in CONFIGS})


if __name__ == '__main__':
    main()
