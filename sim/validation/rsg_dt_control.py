"""Could the time step account for the two-way reference's tilt swing?

The rotated grid picks dt from the largest stiffness component, which
rotates with tilt, so its Courant number varies by about 2.6 per cent
across the tilts examined. The seven-tilt diagnosis already argued that
away on the grounds that 2.6 per cent cannot make 13 decibels, but the
two-way reference is a new measurement and the argument should be made
against it directly rather than inherited.

So: run the HOMOGENEOUS case at tilt 0, unchanged in every other
respect, at the time steps the other tilts chose. If the reference
amplitude barely moves, the time step is excluded and the swing belongs
to the operator's response to the rotated tensor and to the rotated
beam direction.

Usage: python rsg_dt_control.py PPW
Writes ../results/rsg_dt_control.log.
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import tilt_testbed as TB                                   # noqa: E402
import twoway_ref as TW                                     # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOG = os.path.join(RESULTS, 'rsg_dt_control.log')
PSI_A, PSI_B = 0.0, 51.0


def note(msg):
    line = '%s  %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def main():
    ppw = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    z = dict(np.load(os.path.join(
        RESULTS, 'rsg_twoway_rsg_ppw%g.npz' % ppw), allow_pickle=True))
    tilts = sorted(float(k[3:]) for k in z if k.startswith('ho_'))
    g = TW.geometry(0.0, ppw)
    idx, r_true, _ = TW.ray_receivers(g)
    v = TB.vA(PSI_A)
    note('=' * 72)
    note('time-step control, RSG, n_lambda = %g: tilt 0 run at the time '
         'steps the other tilts chose' % ppw)
    note('%-12s %10s %14s %10s %8s'
         % ('dt from tilt', 'dt (ns)', 'tx at 2d', 'vs own dt', 's'))
    base = None
    for th in tilts:
        dt = float(z['dt_%g' % th])
        t0 = time.time()
        ho, _, _ = TW.run_rsg(0.0, ppw, PSI_A, PSI_B, homog=True,
                              dt_in=dt, extra_rec=idx)
        tx, _ = TW.gate_peak(ho[:, -1], dt,
                             r_true[-1] / v + 1.2 / TB.F0, half_us=0.6)
        if base is None:
            base = tx
        note('%-12.4g %10.4f %14.4e %10.2f %8.0f'
             % (th, dt * 1e9, tx, 20 * np.log10(tx / base),
                time.time() - t0))
    note('The tilt-0 geometry is identical in every run above, so any '
         'spread here is the time step alone.')


if __name__ == '__main__':
    main()
