"""Audit the two-way reference on the traces it was measured from.

The two-way reference reports that the rotated grid's homogeneous direct
arrival changes by 8 to 11 dB with the tilt of the problem. That number
is an envelope PEAK read at ONE cell. Three things could produce it, and
they have different consequences for the manuscript:

  A a real change in radiated energy along the beam, which would be a
    property of the discretised source and propagator;
  B a change in the SHAPE of the arrival, with the energy unchanged: the
    envelope peak then moves between lobes of a wavetrain and the ratio
    of peaks is not a ratio of gains;
  C the receiver cell landing in a different class of a grid-scale
    modulation of the recorded field as the beam direction rotates, in
    which case the number is a property of the snapping and of nothing
    else.

They are separated here at no computational cost, from the archives the
reference already wrote.

  A vs B  by measuring the same arrival two ways: envelope peak, which is
          what the reference used, and ENERGY in the same gate, which is
          blind to shape. If the two agree the change is energy; if the
          peak moves and the energy does not, it was shape.
  C       by the parity class of the receiver's offset from the source
          cell. The rotated operator differences along the four body
          diagonals, and the lattice those generate is the index-4
          sublattice of offsets whose three components share a parity.
          If tx sorts by that class rather than by tilt, C is the
          explanation.

Also reproduces the published echo amplitudes from the archived traces
and reports where in the gate the echo sits.

Reads ../results/rsg_twoway_*.npz. Writes ../results/rsg_ref_audit.log.
"""
import os
import sys

import numpy as np
from scipy.signal import hilbert

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('.', '..'):
    _p = os.path.normpath(os.path.join(_HERE, _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

import tilt_testbed as TB                                   # noqa: E402
import twoway_ref as TW                                     # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOG = os.path.join(RESULTS, 'rsg_ref_audit.log')
PSI_A = 0.0


def note(msg):
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def gate(x, dt, t0, half_us=0.5):
    e = np.abs(hilbert(np.asarray(x, float)))
    w = int(half_us * 1e-6 / dt)
    i0 = int(t0 / dt)
    a, b = max(i0 - w, 0), min(i0 + w, len(e))
    j = int(np.argmax(e[a:b])) + a
    seg = np.asarray(x, float)[a:b]
    en = float(np.sum(seg ** 2) * dt)
    esub = e[a:b]
    lobes = int(np.sum((esub[1:-1] > esub[:-2]) & (esub[1:-1] > esub[2:])
                       & (esub[1:-1] > 0.25 * esub.max())))
    ctr = float(np.sum(esub * np.arange(a, b) * dt) / max(esub.sum(), 1e-300))
    return dict(pk=float(e[j]), tpk=j * dt, en=en, lobes=lobes, ctr=ctr,
                seg=seg, a=a, b=b)


def parity_class(off):
    return len({int(off[0]) % 2, int(off[1]) % 2, int(off[2]) % 2}) == 1


def main():
    note('=' * 100)
    note('AUDIT OF THE TWO-WAY REFERENCE, from its own archives')
    v = TB.vA(PSI_A)
    for scheme in ('ssg', 'rsg'):
        for ppw in (6.0, 8.0, 10.0):
            f = os.path.join(RESULTS, 'rsg_twoway_%s_ppw%.0f.npz'
                             % (scheme, ppw))
            if not os.path.exists(f):
                continue
            z = np.load(f)
            tilts = sorted({float(k.split('_')[1]) for k in z.files
                            if k.startswith('bi_')})
            g = TW.geometry(0.0, ppw)
            c = g['c']
            note('')
            note('-' * 100)
            note('%s ppw %.0f   N %d  h %.4f mm  source cell (%d,%d,%d)'
                 % (scheme.upper(), ppw, g['N'], g['h'] * 1e3, c, c, c))
            note('%-7s %11s %11s %11s %11s %9s %9s %7s %7s %9s %8s %8s'
                 % ('tilt', 'echo', 'err dB', 'tx peak', 'G pk dB',
                    'G en dB', 'ctr ns', 'lobes', 'par', 'echo off', 'ENR', 'gate/rec'))
            base_pk = base_en = base_echo = None
            for th in tilts:
                k = ('%g' % th)
                bi = z['bi_' + k]
                ho = z['ho_' + k]
                dt = float(z['dt_' + k])
                d = float(z['d_' + k])
                rt = z['rtrue_' + k]
                idx = z['idx_' + k]
                echo = TB.echo_amp(bi[:, 0], ho[:, 0], dt, d, v)
                # the beam receiver at the image range, radius-corrected
                r = float(rt[-1])
                gg = gate(ho[:, -1], dt, r / v + 1.2 / TB.F0)
                pk = gg['pk'] * r
                en = gg['en'] * r ** 2
                off = np.asarray(idx[-1]) - c
                # echo placement inside its own gate
                t0e = 2 * d / v + 1.2 / TB.F0
                ge = gate(bi[:, 0] - ho[:, 0], dt, t0e)
                if base_pk is None:
                    base_pk, base_en, base_echo = pk, en, echo
                # echo-to-noise on the same differenced trace, and how much
                # of the whole record the gate actually holds
                x = bi[:, 0] - ho[:, 0]
                og, _, _ = TW.out_of_gate_peak(x, dt, t0e)
                e = TW.env(x)
                ia = int(0.5e-6 / dt)
                ib = len(e) - int(0.3e-6 / dt)
                gmax = float(e[ia:ib].max())
                note('%-7.4g %11.4e %11.2f %11.4e %11.2f %9.2f %9.1f '
                     '%7d %7s %9.1f %8.1f %8.2f'
                     % (th, echo, 20 * np.log10(echo / base_echo), gg['pk'],
                        20 * np.log10(pk / base_pk),
                        10 * np.log10(en / base_en),
                        (gg['ctr'] - r / v - 1.2 / TB.F0) * 1e9, gg['lobes'],
                        'same' if parity_class(off) else 'mixed',
                        (ge['tpk'] - t0e) * 1e9,
                        ge['pk'] / max(og, 1e-300), ge['pk'] / gmax))
            note('  columns: G pk is the reference as published (envelope '
                 'peak); G en is the same arrival measured by ENERGY in the')
            note('  same gate. par is the parity class of the receiver '
                 'offset from the source cell: "same" means all three')
            note('  components share a parity, which is the sublattice the '
                 'four body diagonals generate.')
            # the parity class of every beam receiver, all ranges
            note('  parity class of all four beam receivers, and tx at each:')
            for th in tilts:
                k = ('%g' % th)
                ho = z['ho_' + k]
                dt = float(z['dt_' + k])
                rt = z['rtrue_' + k]
                idx = z['idx_' + k]
                cls, val = [], []
                for i in range(4):
                    off = np.asarray(idx[i]) - c
                    cls.append('S' if parity_class(off) else 'm')
                    gg = gate(ho[:, i + 1], dt,
                              float(rt[i]) / v + 1.2 / TB.F0)
                    val.append(gg['pk'] * float(rt[i]))
                note('    tilt %-7.4g offsets %s  class %s  A*r %s'
                     % (th, np.array2string(
                         np.asarray(idx) - c).replace('\n', ''),
                        ''.join(cls),
                        np.array2string(np.asarray(val), precision=4)))


if __name__ == '__main__':
    main()
