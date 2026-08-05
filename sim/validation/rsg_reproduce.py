"""Independent reproduction of the published rotated-grid numbers.

Nothing here is new physics. It calls tilt_rsg.run_rsg and
tilt_testbed.echo_amp, the published functions, unmodified, and asks
whether the numbers in sim/results/rsg_diagnose.log and
sim/results/tilt_table.log come back. If they do not, no verdict about
them can be reached and this adjudication stops.

It also records, from the SAME runs and at no extra cost, the three
things the published diagnosis did not:

  * where inside the +-0.5 us gate the echo envelope actually peaks, and
    how much of the whole differenced record the gate holds. A gate that
    is not centred on the echo, or that does not contain the largest
    thing in the record, is not measuring an echo;
  * the echo-to-noise ratio at the present domain size, under a stated
    definition, which is the second of the two outstanding checks;
  * the late-time growth and Courant number, which the published
    diagnosis reports and which must come back unchanged.

usage:  python rsg_reproduce.py PPW TILT [TILT ...]
Writes ../results/rsg_reproduce.log and one npz per (ppw, tilt), so a
kill loses at most the run in flight.
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('.', '..'):
    _p = os.path.normpath(os.path.join(_HERE, _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tilt_testbed as TB                                   # noqa: E402
import tilt_rsg                                             # noqa: E402
import twoway_ref as TW                                     # noqa: E402
import rsg_diagnose as DG                                   # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOG = os.path.join(RESULTS, 'rsg_reproduce.log')
PSI_A, PSI_B = 0.0, 51.0


def note(msg):
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def one(ppw, theta):
    out = os.path.join(RESULTS, 'rsg_repro_ppw%.0f_t%.4f.npz' % (ppw, theta))
    if os.path.exists(out):
        z = np.load(out)
        return {k: (z[k] if z[k].ndim else float(z[k])) for k in z.files}
    t0 = time.time()
    bi, dt, d = tilt_rsg.run_rsg(theta, ppw, PSI_A, PSI_B)
    ho, _, _ = tilt_rsg.run_rsg(theta, ppw, PSI_A, PSI_B, homog=True,
                                dt_in=dt)
    r = dict(theta=theta, ppw=ppw, dt=dt, d=d, bi=bi, ho=ho,
             secs=time.time() - t0,
             amp=TB.echo_amp(bi, ho, dt, d, TB.vA(PSI_A)),
             courant=TB.C_REF * dt / (TB.C_REF / TB.F0 / ppw),
             hom_peak=float(np.max(np.abs(ho))),
             growth_bi=DG.late_time_growth_db(bi, dt),
             growth_ho=DG.late_time_growth_db(ho, dt))
    np.savez(out, **r)
    return r


def extras(r):
    """Gate placement and echo-to-noise on the differenced trace."""
    dt, d = r['dt'], r['d']
    x = np.asarray(r['bi']) - np.asarray(r['ho'])
    t0 = 2 * d / TB.vA(PSI_A) + 1.2 / TB.F0
    pk, tpk = TW.gate_peak(x, dt, t0)
    out, _, _ = TW.out_of_gate_peak(x, dt, t0)
    e = TW.env(x)
    a = int(0.5e-6 / dt)
    b = len(e) - int(0.3e-6 / dt)
    j = int(np.argmax(e[a:b])) + a
    return dict(t0=t0, pk=pk, tpk=tpk, enr=pk / max(out, 1e-300),
                gmax=float(e[j]), tgmax=j * dt, hold=pk / float(e[j]))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    ppw = float(sys.argv[1])
    tilts = [float(x) for x in sys.argv[2:]] or [0.0, 45.0]
    note('=' * 100)
    note('REPRODUCTION via tilt_rsg.run_rsg and tilt_testbed.echo_amp, '
         'ppw %.0f   %s' % (ppw, time.strftime('%Y-%m-%d %H:%M:%S')))
    note('%-9s %9s %8s %12s %11s %10s %10s %8s'
         % ('tilt', 'dt (ns)', 'Courant', 'echo amp', 'hom peak',
            'growth bi', 'growth ho', 's'))
    rows = []
    for th in tilts:
        r = one(ppw, th)
        rows.append(r)
        note('%-9.4f %9.4f %8.4f %12.4e %11.4e %+10.2f %+10.2f %8.0f'
             % (th, r['dt'] * 1e9, r['courant'], r['amp'], r['hom_peak'],
                r['growth_bi'], r['growth_ho'], r['secs']))
    base = rows[0]['amp']
    note('')
    note('%-9s %12s %14s %16s' % ('tilt', 'error dB', 'Courant vs 0',
                                  'hom peak vs 0'))
    for r in rows:
        note('%-9.4f %12.2f %14.4f %16.3e'
             % (r['theta'], 20 * np.log10(r['amp'] / base),
                r['courant'] - rows[0]['courant'],
                r['hom_peak'] / rows[0]['hom_peak'] - 1.0))
    note('')
    note('GATE PLACEMENT AND ECHO-TO-NOISE (new; the gate is the published')
    note('one, +-0.5 us about 2d/v + 1.2/f0, and ENR divides the gated')
    note('envelope peak by the largest envelope value outside the gate and')
    note('a +-1.0 us guard).')
    note('%-9s %11s %11s %11s %9s %11s %9s'
         % ('tilt', 'gate ctr us', 'echo pk us', 'offset ns', 'ENR',
            'rec max us', 'gate/rec'))
    for r in rows:
        x = extras(r)
        note('%-9.4f %11.3f %11.3f %11.1f %9.1f %11.3f %9.2f'
             % (r['theta'], x['t0'] * 1e6, x['tpk'] * 1e6,
                (x['tpk'] - x['t0']) * 1e9, x['enr'], x['tgmax'] * 1e6,
                x['hold']))


if __name__ == '__main__':
    main()
