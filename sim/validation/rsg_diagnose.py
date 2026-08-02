"""Is the rotated staggered grid's divergence at 45 degrees real?

Section 4 reports that the rotated staggered grid, unlike every other
treatment, gets WORSE as the grid is refined, and that the behaviour is
concentrated at an interface tilt of 45 degrees. That is a strong
negative claim about a scheme two cited papers recommend, so before it is
published the obvious implementation faults have to be excluded.

Three specific suspicions, each with its own test here.

  STABILITY. The rotated operator differences along body diagonals, so
  its stability limit differs from the standard grid's. If the time step
  chosen at 45 degrees sits closer to that limit than at other tilts, the
  extra error could be marginal instability rather than discretisation.
  Tested by recording the Courant number actually used at every tilt and
  by looking for late-time growth in the trace, which is what a marginally
  unstable scheme does and a merely inaccurate one does not.

  THE REFERENCE. Every error here is a bicrystal differenced against a
  homogeneous run. If the homogeneous reference itself misbehaves at 45
  degrees the differencing is contaminated and the interface is innocent.
  Tested by comparing the homogeneous traces across tilt: the medium is
  uniform, so nothing about them should depend on the tilt at all.

  THE ORIENTATION. 45 degrees in the x-z plane is a FACE diagonal. The
  rotated operator is aligned with BODY diagonals, at 54.7 degrees. If
  the error is a property of how far the interface sits from an aligned
  direction, it should vary smoothly through 45 rather than spike there.
  Tested by sampling tilts either side, including 54.7 itself.

Reads:  nothing. Writes ../results/rsg_diagnose.log and .npz.
Needs a CUDA device. About one hour at n_lambda = 8.
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

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOG = os.path.join(RESULTS, 'rsg_diagnose.log')

PSI_A, PSI_B = 0.0, 51.0
PPW = 8.0

# 0 is the exact reference. 54.7356 is the body diagonal, the direction
# the rotated operator is actually aligned with, and the one orientation
# at which it should do best if the alignment argument holds.
TILTS = (0.0, 30.0, 40.0, 45.0, 50.0, 54.7356, 60.0)


def note(msg):
    line = '%s  %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def late_time_growth_db(trace, dt):
    """Energy in the last tenth of the record against the middle tenth.

    A stable scheme leaves a decaying tail. A marginally unstable one
    leaves a growing one, and that shows here as a positive number long
    before it becomes visible as an obviously wrong waveform.
    """
    n = len(trace)
    mid = trace[int(0.45 * n):int(0.55 * n)]
    end = trace[int(0.90 * n):]
    e_mid = float(np.mean(mid ** 2)) + 1e-300
    e_end = float(np.mean(end ** 2)) + 1e-300
    return 10.0 * np.log10(e_end / e_mid)


def run_tilt(theta):
    """Bicrystal and homogeneous run at one tilt, with diagnostics."""
    bi, dt, d = tilt_rsg.run_rsg(theta, PPW, PSI_A, PSI_B)
    ho, _, _ = tilt_rsg.run_rsg(theta, PPW, PSI_A, PSI_B, homog=True,
                                dt_in=dt)
    h = TB.C_REF / TB.F0 / PPW
    courant = TB.C_REF * dt / h
    return dict(
        theta=theta, dt=dt, courant=courant,
        amp=TB.echo_amp(bi, ho, dt, d, TB.vA(PSI_A)),
        hom_peak=float(np.max(np.abs(ho))),
        hom_energy=float(np.sum(ho ** 2)),
        growth_bi=late_time_growth_db(bi, dt),
        growth_ho=late_time_growth_db(ho, dt))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    note('=' * 72)
    note('Rotated staggered grid diagnosis, n_lambda = %.0f, psi_b = %.0f'
         % (PPW, PSI_B))
    note('%-9s %9s %8s %12s %11s %10s %10s'
         % ('tilt', 'dt (ns)', 'Courant', 'echo amp', 'hom peak',
            'growth bi', 'growth ho'))
    rows = []
    for theta in TILTS:
        t0 = time.time()
        r = run_tilt(theta)
        rows.append(r)
        note('%-9.4f %9.4f %8.4f %12.4e %11.4e %+10.2f %+10.2f   (%.0f s)'
             % (r['theta'], r['dt'] * 1e9, r['courant'], r['amp'],
                r['hom_peak'], r['growth_bi'], r['growth_ho'],
                time.time() - t0))

    base = rows[0]['amp']
    note('')
    note('%-9s %14s %16s %16s' % ('tilt', 'error (dB)', 'Courant vs 0',
                                  'hom peak vs 0'))
    for r in rows:
        note('%-9.4f %14.2f %16.4f %16.3e'
             % (r['theta'], 20 * np.log10(r['amp'] / base),
                r['courant'] - rows[0]['courant'],
                r['hom_peak'] / rows[0]['hom_peak'] - 1.0))

    note('')
    note('READING THIS TABLE.')
    note('  If the Courant column is flat, the time step is not the cause.')
    note('  If hom peak is flat, the homogeneous reference is sound and')
    note('  the differencing is clean, so the error belongs to the')
    note('  interface and not to the harness.')
    note('  If growth is negative everywhere the scheme is stable, and')
    note('  marginal instability is excluded.')
    note('  If the error varies smoothly through 45 and is no better at')
    note('  54.74, the body-diagonal alignment argument does not explain')
    note('  it. If it dips at 54.74 the alignment argument does explain')
    note('  it, and 45 degrees is simply the worst case rather than a bug.')

    np.savez(os.path.join(RESULTS, 'rsg_diagnose.npz'),
             **{k: np.array([r[k] for r in rows]) for k in rows[0]})


if __name__ == '__main__':
    main()
