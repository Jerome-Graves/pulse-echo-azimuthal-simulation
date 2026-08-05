"""Read the Green-function probe and say what each scheme's point source
actually does, with no interface anywhere.

Three questions, in the order that decides the manuscript.

  Q1 IS THE ARC FLAT. In the isotropic medium a monopole radiates a
     spherical qP wave, so at a fixed radius the recorded amplitude and
     the arrival time must not depend on the direction. Reported as dB
     and ns against the 0-degree arc point, after removing the small
     radius differences that snapping to cells introduces. Two amplitude
     measures are given because they fail differently: the ENVELOPE PEAK,
     which is what the testbed uses and which moves if the waveform
     changes shape, and the ENERGY in the same gate, which does not. If
     the two disagree the peak was reading distortion, not gain.

  Q2 DOES IT SPREAD AS 1/r. Amplitude times radius along each ray,
     normalised at the arc radius.

  Q3 DOES A SINGLE CELL MEAN ANYTHING. Spread of the recorded amplitude
     over cells one and two steps transverse to the same arc point. A
     scheme whose neighbouring cells disagree by decibels cannot be read
     at one cell.

Then the same three in the ANISOTROPIC medium of the testbed, where the
arc is not flat physically and only the ray and neighbour tests are
clean, plus the arc compared BETWEEN schemes, which is.

Reads ../results/rsg_green_*.npz. Writes ../results/rsg_green_report.log.
"""
import glob
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

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOG = os.path.join(RESULTS, 'rsg_green_report.log')
HALF_US = 0.5


def note(msg):
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def measures(x, dt, t0, half_us=HALF_US):
    """(envelope peak, peak time, energy in the gate) for one trace."""
    e = np.abs(hilbert(np.asarray(x, float)))
    fs = 1.0 / dt
    w = int(half_us * 1e-6 * fs)
    i0 = int(t0 * fs)
    a, b = max(i0 - w, 0), min(i0 + w, len(e))
    j = int(np.argmax(e[a:b])) + a
    en = float(np.sum(np.asarray(x, float)[a:b] ** 2) * dt)
    return float(e[j]), float(j * dt), en


def load(scheme, med, ppw):
    f = os.path.join(RESULTS, 'rsg_green_%s_%s_ppw%.0f.npz'
                     % (scheme, med, ppw))
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    return dict(rec=z['rec'], dt=float(z['dt']), tag=[str(t) for t in z['tag']],
                ang=z['ang'], rad=z['rad'], vp=float(z['vp']),
                h=float(z['h']), N=int(z['N']))


def table(d):
    """Per-receiver measures, radius-corrected to the arc radius."""
    dt, vp = d['dt'], d['vp']
    out = []
    for i, tg in enumerate(d['tag']):
        r = d['rad'][i]
        t0 = r / vp + 1.2 / TB.F0
        pk, tp, en = measures(d['rec'][:, i + 1], dt, t0)
        out.append(dict(tag=tg, ang=float(d['ang'][i]), r=r,
                        pk=pk, tp=tp, en=en,
                        pk_r=pk * r, en_r=en * r ** 2,
                        v=r / max(tp - 1.2 / TB.F0, 1e-12)))
    return out


def arc_rows(t):
    return [r for r in t if r['tag'].startswith('arc')]


def report_arc(name, d):
    t = arc_rows(table(d))
    b = t[0]
    note('  %s arc, dB and ns against 0 deg (radius-corrected)' % name)
    note('    %7s %9s %9s %11s %11s'
         % ('angle', 'peak dB', 'energy dB', 'dt peak ns', 'v m/s'))
    for r in t:
        note('    %7.1f %9.2f %9.2f %11.1f %11.1f'
             % (r['ang'], 20 * np.log10(r['pk_r'] / b['pk_r']),
                10 * np.log10(r['en_r'] / b['en_r']),
                (r['tp'] - r['r'] / d['vp'] - 1.2 / TB.F0) * 1e9, r['v']))
    pk = np.array([20 * np.log10(r['pk_r'] / b['pk_r']) for r in t])
    en = np.array([10 * np.log10(r['en_r'] / b['en_r']) for r in t])
    tv = np.array([(r['tp'] - r['r'] / d['vp'] - 1.2 / TB.F0) * 1e9
                   for r in t])
    note('    peak-to-peak: %.2f dB (peak)  %.2f dB (energy)  %.1f ns (time)'
         % (pk.max() - pk.min(), en.max() - en.min(), tv.max() - tv.min()))
    return pk, en, tv


def report_rays(name, d):
    t = table(d)
    note('  %s rays, amplitude x radius normalised at the arc radius '
         '(flat = 1/r)' % name)
    for a in sorted({r['ang'] for r in t if r['tag'].startswith('ray')}):
        ray = [r for r in t if r['tag'].startswith('ray')
               and abs(r['ang'] - a) < 1e-6]
        arc = [r for r in arc_rows(t) if abs(r['ang'] - a) < 1e-6]
        if not arc:
            continue
        ref = arc[0]['pk_r']
        note('    %7.2f deg  r = %s mm   A*r = %s'
             % (a, np.array2string(np.array([r['r'] for r in ray]
                                            + [arc[0]['r']]) * 1e3,
                                   precision=2),
                np.array2string(np.array([r['pk_r'] / ref for r in ray]
                                         + [1.0]), precision=4)))


def report_nbr(name, d):
    t = table(d)
    for kind in ('nbz', 'nby'):
        g = [r for r in t if r['tag'].startswith(kind)]
        if not g:
            continue
        ref = [r for r in arc_rows(t) if abs(r['ang']) < 1e-6][0]
        v = np.array([20 * np.log10(r['pk_r'] / ref['pk_r']) for r in g])
        note('  %s neighbours %s of the 0-deg arc cell: %s dB  (spread %.2f)'
             % (name, kind,
                np.array2string(v, precision=2), v.max() - v.min()))


def main():
    ppws = [float(x) for x in sys.argv[1:]] or [4.0, 6.0, 10.0]
    note('=' * 78)
    note('GREEN-FUNCTION PROBE REPORT')
    note('The isotropic arc is the decisive one: it must be flat.')
    summary = []
    for ppw in ppws:
        for med in ('iso', 'ani'):
            for scheme in ('ssg', 'rsg'):
                d = load(scheme, med, ppw)
                if d is None:
                    continue
                name = '%s %s ppw%.0f' % (scheme.upper(), med, ppw)
                note('')
                note('-' * 78)
                note('%s   N %d  h %.4f mm  dt %.4f ns'
                     % (name, d['N'], d['h'] * 1e3, d['dt'] * 1e9))
                pk, en, tv = report_arc(name, d)
                report_rays(name, d)
                report_nbr(name, d)
                summary.append((scheme, med, ppw, pk.max() - pk.min(),
                                en.max() - en.min(), tv.max() - tv.min()))
    note('')
    note('=' * 78)
    note('SUMMARY. Peak-to-peak variation around a fixed-radius arc in a')
    note('HOMOGENEOUS medium with no interface. For "iso" the physical')
    note('answer is exactly zero at every entry.')
    note('%-7s %-5s %6s %12s %12s %12s'
         % ('scheme', 'med', 'ppw', 'peak dB', 'energy dB', 'time ns'))
    for s, m, p, a, b, c in summary:
        note('%-7s %-5s %6.0f %12.2f %12.2f %12.1f' % (s, m, p, a, b, c))


if __name__ == '__main__':
    main()
