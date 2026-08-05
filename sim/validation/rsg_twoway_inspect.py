"""Look at the raw beam traces before believing anything measured on them.

The two-way reference is only worth what its arrival measurement is
worth. If the rotated grid's direct arrival at the image range were
merely shifted out of the gate, or split into two features, an envelope
peak in a fixed window would report an amplitude loss that is really a
timing or shape effect and the verdict would be wrong.

So this prints, for every beam receiver in the homogeneous run, the
envelope profile against time in units of the expected arrival, the peak
time, the peak against the trace's own global maximum, and the
first-break time at a fixed fraction of that peak. A clean arrival has
its peak within a fraction of a period of the prediction, carries the
whole record's maximum, and its first break scales with range.

Reads the archives rsg_twoway_run.py writes; no GPU, no new runs.

Usage: python rsg_twoway_inspect.py {ssg|rsg} PPW [TILT ...]
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import tilt_testbed as TB                                   # noqa: E402
import twoway_ref as TW                                     # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    scheme, ppw = sys.argv[1], float(sys.argv[2])
    z = dict(np.load(os.path.join(
        RESULTS, 'rsg_twoway_%s_ppw%g.npz' % (scheme, ppw)),
        allow_pickle=True))
    tilts = ([float(t) for t in sys.argv[3:]]
             or sorted(float(k[3:]) for k in z if k.startswith('ho_')))
    v = TB.vA(0.0)
    print('%s  n_lambda %g   qP speed %.1f m/s   period %.3f us'
          % (scheme.upper(), ppw, v, 1e6 / TB.F0))
    for th in tilts:
        ho = z['ho_%g' % th]
        dt = float(z['dt_%g' % th])
        rt = z['rtrue_%g' % th]
        print('')
        print('  tilt %g' % th)
        for j in range(1, ho.shape[1]):
            r = rt[j - 1]
            e = TW.env(ho[:, j])
            t_pred = r / v + 1.2 / TB.F0
            jp = int(np.argmax(e))
            fb = np.argmax(e > 0.1 * e[jp])
            # envelope in 0.25 us bins around the prediction, dB below peak
            fs = 1.0 / dt
            prof = []
            for k in range(-4, 9):
                i1 = int((t_pred + k * 0.25e-6) * fs)
                i2 = int((t_pred + (k + 1) * 0.25e-6) * fs)
                seg = e[max(i1, 0):min(i2, len(e))]
                prof.append('%4.0f' % (20 * np.log10(seg.max() / e[jp]))
                            if seg.size and seg.max() > 0 else '   .')
            print('    r %6.2f mm  t_pred %6.3f  t_peak %6.3f  '
                  'first break %6.3f us  peak %.4e'
                  % (r * 1e3, t_pred * 1e6, jp * dt * 1e6, fb * dt * 1e6,
                     e[jp]))
            print('      %s' % ' '.join(prof))


if __name__ == '__main__':
    main()
