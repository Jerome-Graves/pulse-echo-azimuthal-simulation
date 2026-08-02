"""Regenerate Table 2 of the manuscript: discretisation error against
interface tilt, for four interface treatments at three grid resolutions.

Why this exists. The published table has twelve cells, but only two of
them had surviving raw output: the standard staggered grid and the
rotated staggered grid, both at n_lambda = 6. The other ten appeared in
the manuscript source and nowhere else, and the seven-point convergence
ladder disagreed with two of them. This script recomputes all twelve on
one code path in one run, so the table and the convergence order come
from the same experiment.

Method. For each treatment and resolution the tilted bicrystal is run at
five interface tilts. The physics is invariant under tilt by rigid
rotation (both c-axes rotate with the interface), so the echo amplitude
must not change; whatever change is measured is discretisation error. The
tilt-zero case has the interface on a cell face, every cell wholly within
one grain, and is therefore exact truth. The reported figure is the mean
absolute error in decibels over the four non-zero tilts.

Reads:  nothing on disk. Everything is computed.
Writes: sim/results/tilt_table.npz   per-cell errors and per-tilt detail
        sim/results/tilt_table.log   human-readable transcript

Resumable: a cell already present in the .npz is skipped, so the run can
be stopped and restarted without losing work.

Requires a CUDA device. Roughly one hour for the standard-grid
treatments; the rotated staggered grid is far more expensive per run and
is timed and reported separately so the cost is visible before it is
paid.
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('..', os.path.join('..', 'validation')):
    _p = os.path.normpath(os.path.join(_HERE, _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tilt_testbed as TB                                   # noqa: E402
import tilt_rsg                                             # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))
NPZ = os.path.join(RESULTS, 'tilt_table.npz')
LOG = os.path.join(RESULTS, 'tilt_table.log')

# psi_a = 0 and psi_b = 51 degrees is the strongest contrast available in
# ice: 50.81 degrees is the quasi-longitudinal speed minimum. See Sec. 4.2.
PSI_A, PSI_B = 0.0, 51.0
TILTS = (0.0, 15.0, 22.5, 30.0, 45.0)
PPWS = (6.0, 8.0, 10.0)

# Cheapest first, so a cost estimate for the expensive rotated-grid cells
# is available before committing to them.
TREATMENTS = ('naive', 'aa', 'sm', 'rsg')

LABEL = {'naive': 'standard staggered grid, nearest-neighbour',
         'aa': 'anti-aliased medium',
         'sm': 'Schoenberg-Muir equivalent medium',
         'rsg': 'rotated staggered grid'}


def note(msg):
    """Print and append, so a background run is followable while it runs."""
    line = '%s  %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def one_cell(treat, ppw):
    """Mean absolute error in dB over the non-zero tilts, plus the detail.

    The homogeneous reference is run at the bicrystal's own time step so
    that differencing the two isolates the interface echo rather than a
    sampling difference.
    """
    amps, per_tilt = [], []
    for theta in TILTS:
        if treat == 'rsg':
            bi, dt, d = tilt_rsg.run_rsg(theta, ppw, PSI_A, PSI_B)
            ho, _, _ = tilt_rsg.run_rsg(theta, ppw, PSI_A, PSI_B,
                                        homog=True, dt_in=dt)
        else:
            bi, dt, d = TB.run(theta, ppw, treat, PSI_A, PSI_B)
            ho, _, _ = TB.run(theta, ppw, treat, PSI_A, PSI_B,
                              homog=True, dt_in=dt)
        amps.append(TB.echo_amp(bi, ho, dt, d, TB.vA(PSI_A)))
    base = amps[0]                      # tilt zero is exact by construction
    for theta, a in zip(TILTS, amps):
        per_tilt.append(20.0 * np.log10(a / base))
    err = float(np.mean(np.abs(per_tilt[1:])))
    return err, np.array(per_tilt), np.array(amps)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    store = {}
    if os.path.exists(NPZ):
        with np.load(NPZ, allow_pickle=True) as z:
            store = {k: z[k] for k in z.files}
        note('resuming, %d cell(s) already done' % (len(store) // 3))

    note('=' * 64)
    note('Table 2 regeneration: %d treatments x %d resolutions x %d tilts'
         % (len(TREATMENTS), len(PPWS), len(TILTS)))

    for treat in TREATMENTS:
        for ppw in PPWS:
            key = '%s_ppw%d' % (treat, int(ppw))
            if key + '_err' in store:
                note('  %-34s already done, %.2f dB'
                     % (key, float(store[key + '_err'])))
                continue
            t0 = time.time()
            try:
                err, per_tilt, amps = one_cell(treat, ppw)
            except Exception as exc:                  # keep going
                note('  %-34s FAILED: %s' % (key, exc))
                continue
            dt_min = (time.time() - t0) / 60.0
            store[key + '_err'] = np.array(err)
            store[key + '_tilt'] = per_tilt
            store[key + '_amp'] = amps
            np.savez(NPZ, **store)
            note('  %-34s %5.2f dB   (%s)   %.1f min'
                 % (key, err,
                    ' '.join('%+.2f' % v for v in per_tilt), dt_min))

    note('')
    note('TABLE 2, mean absolute error in dB over tilts >= 15 degrees')
    note('%-44s %8s %8s %8s' % ('', 'ppw 6', 'ppw 8', 'ppw 10'))
    for treat in TREATMENTS:
        cells = []
        for ppw in PPWS:
            k = '%s_ppw%d_err' % (treat, int(ppw))
            cells.append('%.2f' % float(store[k]) if k in store else '  --')
        note('%-44s %8s %8s %8s' % (LABEL[treat], *cells))
    note('')
    note('LaTeX rows:')
    for treat in TREATMENTS:
        cells = []
        for ppw in PPWS:
            k = '%s_ppw%d_err' % (treat, int(ppw))
            cells.append('\\SI{%.2f}{\\decibel}' % float(store[k])
                         if k in store else '\\RUN{}')
        note('%-42s & %s \\\\' % (LABEL[treat].replace('-', '--'),
                                  ' & '.join(cells)))


if __name__ == '__main__':
    main()
