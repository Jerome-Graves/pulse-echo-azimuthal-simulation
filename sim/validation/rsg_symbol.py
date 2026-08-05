"""Analytic symbol of the rotated and standard operators, no GPU.

WHY. The two-way reference reports that the rotated staggered grid's
HOMOGENEOUS direct arrival changes by 8-11 dB with the tilt of the
problem, and rsg_twoway_inspect reports its arrival time moving by up to
0.32 us over 16 mm, which is a 7 per cent phase-velocity swing. Both are
enormous for an eighth-order operator optimised to kh_max = 2. Before any
of that is believed, the operator's own symbol should be asked what it
predicts, because that is free and it discriminates two very different
stories:

  * if the symbol predicts a large direction dependence at these
    wavenumbers, the measured swing is the scheme behaving as designed on
    this grid and the testbed's tilt-0 normalisation is simply invalid;
  * if the symbol predicts a small one, the measured swing is an
    implementation fault or a measurement fault, and the diagnosis has to
    say which.

THE SYMBOLS. Both schemes use the SAME staggered coefficients c_n.

  standard grid, taps along each axis:
      d/dx_i -> (i/h) S(h k_i),      S(p) = 2 sum_n c_n sin((n-1/2) p)

  rotated grid (sim/core/rsg.py), taps along the four body diagonals a_m:
      D_m    -> (i/h) S(h k.a_m)
      d/dx_i -> (i/4h) sum_m (a_m)_i S(h k.a_m)

The numerical phase velocity along a direction is v_num/v = |k| / |k_eff|
with k_eff built from the symbol. Reported here against the propagation
angle in the x-z plane, which is the plane every wave in the testbed
lives in.

Reads nothing. Writes ../results/rsg_symbol.log.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('.', '..'):
    _p = os.path.normpath(os.path.join(_HERE, _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))

import fdtd                                                 # noqa: E402
import rsg                                                  # noqa: E402
import tilt_testbed as TB                                   # noqa: E402
from ringfwi import anisotropy as an                        # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOG = os.path.join(RESULTS, 'rsg_symbol.log')

ANGLES = np.array([0.0, 15.0, 22.5, 30.0, 40.0, 45.0, 50.0, 54.7356, 60.0,
                   75.0, 90.0])


def note(msg):
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def S(p, c):
    """Symbol of the staggered first difference, real and odd in p."""
    n = np.arange(1, len(c) + 1)
    return 2.0 * np.sum(np.asarray(c)[None, :]
                        * np.sin((n[None, :] - 0.5) * np.asarray(p)[:, None]),
                        axis=1)


def keff_ssg(kvec, h, c):
    """(3,) effective wavevector of the standard operator."""
    return np.array([S(np.array([h * kvec[i]]), c)[0] / h for i in range(3)])


def keff_rsg(kvec, h, c):
    """(3,) effective wavevector of the body-diagonal operator."""
    out = np.zeros(3)
    for a in rsg.DIAG:
        phi = h * float(np.dot(kvec, a))
        Sm = S(np.array([phi]), c)[0]
        out += a * Sm
    return out / (4.0 * h)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    co = fdtd.optimised_coeffs(TB.ORDER, kh_max=TB.KHM, multistart=True)
    note('=' * 78)
    note('OPERATOR SYMBOL, order %d, kh_max %.1f' % (TB.ORDER, TB.KHM))
    note('coefficients: %s' % np.array2string(np.asarray(co), precision=6))
    note('sum|c| = %.6f' % float(np.abs(np.asarray(co)).sum()))

    # the wave actually propagating in the testbed: qP along the c-axis of
    # grain A, because psi_a = 0 puts that axis along the beam.
    v = TB.vA(0.0)
    note('')
    note('qP speed along the beam %.1f m/s;  h = C_REF/F0/ppw with C_REF '
         '%.0f m/s' % (v, TB.C_REF))
    note('so the true points per wavelength is ppw * %.4f' % (v / TB.C_REF))

    for ppw in (6.0, 8.0, 10.0):
        h = TB.C_REF / TB.F0 / ppw
        note('')
        note('-' * 78)
        note('ppw %.0f   h %.4f mm   true ppw %.2f' % (ppw, h * 1e3,
                                                       v / TB.F0 / h))
        note('%9s %12s %12s %12s %12s'
             % ('angle', 'ssg v/v0', 'rsg v/v0', 'ssg abs err', 'rsg abs err'))
        for fmul, tag in ((1.0, 'f0'), (1.6, '1.6 f0')):
            k = 2.0 * np.pi * TB.F0 * fmul / v
            note('  at %s, kh = %.4f' % (tag, k * h))
            row_s, row_r = [], []
            for a in ANGLES:
                t = np.radians(a)
                kv = k * np.array([np.cos(t), 0.0, np.sin(t)])
                ks = np.linalg.norm(keff_ssg(kv, h, co))
                kr = np.linalg.norm(keff_rsg(kv, h, co))
                row_s.append(k / ks)
                row_r.append(k / kr)
            row_s, row_r = np.array(row_s), np.array(row_r)
            for a, s_, r_ in zip(ANGLES, row_s, row_r):
                note('%9.4f %12.6f %12.6f %11.3f%% %11.3f%%'
                     % (a, s_ / row_s[0], r_ / row_r[0],
                        100 * (s_ - 1.0), 100 * (r_ - 1.0)))
            note('    peak-to-peak across angle: ssg %.3f%%   rsg %.3f%%'
                 % (100 * (row_s.max() - row_s.min()),
                    100 * (row_r.max() - row_r.min())))

    # THE SHEAR BAND. The echo is a qP reflection, but a tilted interface
    # converts qP to qS, and ice's qS is far shorter than its qP. The
    # rotated operator's argument along a body diagonal is up to sqrt(2)
    # times the axial one in this plane, so it runs out of design band
    # first, and it runs out on the SHEAR wave rather than on the qP wave
    # the ladder is specified in.
    vs = 1950.0
    note('')
    note('SHEAR BAND. qS speed %0.0f m/s, so lambda_S = %.3f mm.' % (vs, 1e3 * vs / TB.F0))
    note('The operator is optimised to kh_max = %.1f. Reported below is the'
         % TB.KHM)
    note('largest argument each scheme presents it with, at f0 and at')
    note('1.6 f0, on the qP wave and on the qS wave.')
    note('%5s %8s %10s %10s %10s %10s'
         % ('ppw', 'wave', 'ssg f0', 'rsg f0', 'ssg 1.6f0', 'rsg 1.6f0'))
    for ppw in (6.0, 8.0, 10.0):
        h = TB.C_REF / TB.F0 / ppw
        for nm, cc in (('qP', v), ('qS', vs)):
            row = []
            for fm in (1.0, 1.6):
                k = 2.0 * np.pi * TB.F0 * fm / cc
                row.append(k * h)                      # ssg, axial
                row.append(np.sqrt(2.0) * k * h)       # rsg, face diagonal
            note('%5.0f %8s %10.3f %10.3f %10.3f %10.3f'
                 % (ppw, nm, row[0], row[1], row[2], row[3]))
    note('Entries above %.1f are outside the band the coefficients were'
         % TB.KHM)
    note('fitted on. That is a reason a coarse grid can be bad on the')
    note('rotated scheme; it is not a reason a FINE one can be worse,')
    note('because every argument falls as h falls.')

    # how far does a 1 per cent velocity error move a 16 mm two-way arrival?
    note('')
    note('SCALE. A 1 per cent phase-velocity error moves the %0.0f mm arrival'
         % (2e3 * 0.008))
    note('by %.3f us, which is %.0f per cent of the +-0.5 us echo gate.'
         % (0.01 * 0.016 / v * 1e6, 100 * 0.01 * 0.016 / v * 1e6 / 0.5))

    # amplitude consequence of dispersion alone: propagate a Ricker over
    # 2d with each scheme's numerical velocity and read the envelope peak.
    note('')
    note('DISPERSION-ONLY AMPLITUDE. A Ricker at F0 is propagated 2d = %.1f '
         'mm' % (2e3 * 0.008))
    note('through each scheme by giving every frequency its own numerical')
    note('phase velocity, with NO geometric spreading and NO attenuation.')
    note('The envelope peak that survives is what pulse spreading alone')
    note('costs, so it bounds how much of the measured tilt swing can be')
    note('dispersion rather than source directivity.')
    note('%5s %9s %12s %12s %12s'
         % ('ppw', 'angle', 'ssg dB', 'rsg dB', 'rsg vs 0'))
    from scipy.signal import hilbert
    for ppw in (6.0, 8.0, 10.0):
        h = TB.C_REF / TB.F0 / ppw
        dt = 2.0e-9
        nt = 4096
        t = np.arange(nt) * dt
        w = fdtd.ricker(TB.F0, dt, nt)
        f = np.fft.rfftfreq(nt, dt)
        W = np.fft.rfft(w)
        r2 = 2.0 * 0.008
        base = {}
        for a in ANGLES:
            th = np.radians(a)
            amps = {}
            for name, fn in (('ssg', keff_ssg), ('rsg', keff_rsg)):
                ke = np.zeros_like(f)
                for i, fi in enumerate(f):
                    if fi <= 0:
                        continue
                    k = 2.0 * np.pi * fi / v
                    kv = k * np.array([np.cos(th), 0.0, np.sin(th)])
                    ke[i] = np.linalg.norm(fn(kv, h, co))
                # drop everything past the operator's usable band
                ok = (f > 0) & (f < 0.5 / dt) & (ke * h < np.pi)
                Wp = np.where(ok, W * np.exp(-1j * ke * r2), 0.0)
                amps[name] = float(np.abs(hilbert(
                    np.fft.irfft(Wp, nt))).max())
            ref = float(np.abs(hilbert(np.fft.irfft(
                np.where(f > 0, W * np.exp(
                    -1j * 2 * np.pi * f / v * r2), 0.0), nt))).max())
            if not base:
                base = dict(ssg=amps['ssg'], rsg=amps['rsg'])
            note('%5.0f %9.4f %12.3f %12.3f %12.3f'
                 % (ppw, a, 20 * np.log10(amps['ssg'] / ref),
                    20 * np.log10(amps['rsg'] / ref),
                    20 * np.log10(amps['rsg'] / base['rsg'])))


if __name__ == '__main__':
    main()
