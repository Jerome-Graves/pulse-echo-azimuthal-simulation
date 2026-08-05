"""Read the two-way batch and decide the two outstanding checks.

DEFINITIONS, stated because the verdict turns on them.

  echo(theta)      envelope peak of bi - ho at the source cell in the
                   +-0.5 us gate centred on 2d/v + 1.2/f0. This is
                   TB.echo_amp verbatim, so it reproduces Table 2.
  error(theta)     20 log10 echo(theta)/echo(0). Table 2's quantity.
  tx(theta)        envelope peak of the HOMOGENEOUS trace at the beam
                   receiver nearest the image range 2d, scaled by
                   r_true/r_ref to remove the grid-snapping difference in
                   range. No interface exists in that run.
  G(theta)         20 log10 tx(theta)/tx(0): the gain of the discretised
                   source injection plus two-way propagation, over the
                   path length, direction and medium the echo uses.
  S(theta)         error(theta) - G(theta): what is left for the
                   interface once the propagator is accounted for.
  ENR(theta)       echo(theta) divided by the largest envelope value of
                   the same differenced trace outside the gate and a
                   +-1.0 us guard, from 0.5 us to 0.3 us before the end.
                   Reported split into the causally empty part before the
                   gate and the part after it.
  rho_wf(theta)    peak normalised cross-correlation of the differenced
                   trace in the gate against the tilt-0 one, over lags up
                   to +-0.5 us. A real echo keeps its shape under rigid
                   rotation; grid noise does not.

Usage: python rsg_twoway_report.py [ssg|rsg ...]
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
LOG = os.path.join(RESULTS, 'rsg_twoway_report.log')
# Table 2 averages the absolute error over the tilts at or above 15 deg
TAB2 = (15.0, 22.5, 30.0, 45.0)


def note(msg):
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def xcorr_peak(a, b):
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a) * np.linalg.norm(b)
    if na == 0:
        return 0.0
    return float(np.abs(np.correlate(a, b, 'full')).max() / na)


def load(scheme, ppw):
    f = os.path.join(RESULTS, 'rsg_twoway_%s_ppw%g.npz' % (scheme, ppw))
    if not os.path.exists(f):
        return None
    z = dict(np.load(f, allow_pickle=True))
    tilts = sorted(float(k[3:]) for k in z if k.startswith('bi_'))
    return z, tilts


def measure(z, tilts):
    v = TB.vA(0.0)
    rows = []
    ref = None
    for th in tilts:
        dt = float(z['dt_%g' % th])
        d = float(z['d_%g' % th])
        bi, ho = z['bi_%g' % th], z['ho_%g' % th]
        rt = z['rtrue_%g' % th]
        dif = bi[:, 0] - ho[:, 0]
        t_echo = 2 * d / v + 1.2 / TB.F0
        echo, t_pk = TW.gate_peak(dif, dt, t_echo)
        noise, npre, npost = TW.out_of_gate_peak(dif, dt, t_echo)
        # beam receivers, homogeneous run, spreading removed
        tx, ttx = [], []
        for j in range(1, ho.shape[1]):
            a, t = TW.gate_peak(ho[:, j], dt, rt[j - 1] / v + 1.2 / TB.F0,
                                half_us=0.6)
            tx.append(a * rt[j - 1])          # amplitude x range
            ttx.append(t)
        fs = 1.0 / dt
        w = int(0.5e-6 * fs)
        i0 = int(t_echo * fs)
        seg = dif[max(i0 - w, 0):i0 + w]
        if ref is None:
            ref = seg
        # where the differenced field is largest anywhere in the record:
        # if the gate were missing a dispersed or delayed echo, this would
        # sit far from t_echo and carry most of the energy
        e = TW.env(dif)
        a = int(0.5e-6 * fs)
        jg = int(np.argmax(e[a:])) + a
        prof = []
        for k in range(1, 14):
            i1, i2 = int(k * 0.5e-6 * fs), int((k + 1) * 0.5e-6 * fs)
            seg2 = e[i1:min(i2, len(e))]
            prof.append('%5.0f' % (20 * np.log10(seg2.max() / echo))
                        if seg2.size and seg2.max() > 0 else '    .')
        rows.append(dict(prof=' '.join(prof),
                         t_glob=jg * dt, a_glob=float(e[jg]),
                         t_gate=t_echo,
                         theta=th, dt=dt, d=d, echo=echo, t_pk=t_pk,
                         noise=noise, npre=npre, npost=npost,
                         tx=np.asarray(tx), ttx=np.asarray(ttx),
                         rt=rt, rho=xcorr_peak(seg, ref)))
    return rows


def report(scheme, ppw):
    got = load(scheme, ppw)
    if got is None:
        return None
    z, tilts = got
    if 0.0 not in tilts:
        note('  %s ppw %g: no tilt 0, cannot reference' % (scheme, ppw))
        return None
    rows = measure(z, tilts)
    b = rows[tilts.index(0.0)]
    note('')
    note('%s, n_lambda = %g   (grid %d, h = %.4f mm)'
         % (scheme.upper(), ppw, int(z['meta'][1]), z['meta'][2] * 1e3))
    note('%-8s %11s %9s %11s %9s %9s %9s %8s %8s %8s %7s'
         % ('tilt', 'echo', 'err dB', 'tx(2d)', 'G dB', 'R dB', 'S dB',
            'ENR', 'ENRpre', 'dt_pk ns', 'rho_wf'))
    out = []
    for r in rows:
        err = 20 * np.log10(r['echo'] / b['echo'])
        gdb = 20 * np.log10(r['tx'][-1] / b['tx'][-1])
        # R is the echo referenced to the direct arrival that travelled the
        # SAME path, so it is dimensionless and comparable ACROSS schemes,
        # unlike the raw amplitudes, whose source scalings differ by dt.
        rdb = 20 * np.log10(r['echo'] * r['rt'][-1] / r['tx'][-1])
        enr = r['echo'] / r['noise'] if r['noise'] > 0 else np.inf
        enrp = r['echo'] / r['npre'] if r['npre'] > 0 else np.inf
        note('%-8.4g %11.4e %9.2f %11.4e %9.2f %9.2f %9.2f %8.1f %8.1f '
             '%8.1f %7.3f'
             % (r['theta'], r['echo'], err, r['tx'][-1] / r['rt'][-1], gdb,
                rdb, err - gdb, enr, enrp,
                (r['t_pk'] - b['t_pk']) * 1e9, r['rho']))
        out.append(dict(theta=r['theta'], err=err, g=gdb, s=err - gdb,
                        r=rdb, enr=enr, rho=r['rho']))
    # spreading check: amplitude x range should be flat along the beam
    note('  largest envelope value anywhere in the differenced record '
         '(gate centre %.3f us):' % (rows[0]['t_gate'] * 1e6))
    for r in rows:
        note('    tilt %-7.4g  %.4e at %.3f us   gate holds %.2f of it'
             % (r['theta'], r['a_glob'], r['t_glob'] * 1e6,
                r['echo'] / r['a_glob']))
    note('  envelope of the differenced trace, peak in each 0.5 us bin '
         'from 0.5 us, dB below the gate peak:')
    for r in rows:
        note('    tilt %-7.4g %s' % (r['theta'], r['prof']))
    note('  amplitude x range along the beam (should be flat if 1/r):')
    for r in rows:
        note('    tilt %-7.4g  r = %s mm   A*r = %s'
             % (r['theta'],
                np.array2string(r['rt'] * 1e3, precision=2),
                np.array2string(r['tx'] / r['tx'][-1], precision=4)))
    note('  G at each range separately, dB vs tilt 0. Four independent '
         'receiver cells: a single-cell sampling accident cannot')
    note('  reproduce itself at all four, and a value already large at '
         'the nearest range is injection, not accumulated propagation.')
    note('    %-8s %s' % ('tilt', ''.join(
        '%12s' % ('r=%.1f mm' % (x * 1e3)) for x in rows[0]['rt'])))
    for r in rows:
        note('    %-8.4g %s' % (r['theta'], ''.join(
            '%12.2f' % (20 * np.log10(r['tx'][j] / b['tx'][j]))
            for j in range(len(r['tx'])))))
    sel = [o for o in out if o['theta'] in TAB2 and o['theta'] > 0]
    if sel:
        note('  mean |error| over %s = %.2f dB, mean |S| = %.2f dB'
             % (str([o['theta'] for o in sel]),
                np.mean([abs(o['err']) for o in sel]),
                np.mean([abs(o['s']) for o in sel])))
    return out


def main():
    schemes = sys.argv[1:] or ['ssg', 'rsg']
    note('=' * 78)
    note('TWO-WAY REFERENCE AND ECHO-TO-NOISE, present domain size '
         '(L = %g mm, sponge %d)' % (TB.L * 1e3, TB.SPONGE))
    summary = {}
    for sc in schemes:
        for ppw in (6.0, 8.0, 10.0):
            o = report(sc, ppw)
            if o:
                summary[(sc, ppw)] = o
    note('')
    note('CONVERGENCE OF THE TWO PARTS, tilts common to all resolutions')
    note('%-6s %-8s %9s %9s %9s %9s' % ('scheme', 'tilt', 'ppw6', 'ppw8',
                                        'ppw10', 'trend'))
    for sc in schemes:
        have = [p for p in (6.0, 8.0, 10.0) if (sc, p) in summary]
        if len(have) < 2:
            continue
        tset = set.intersection(*[{o['theta'] for o in summary[(sc, p)]}
                                  for p in have])
        for th in sorted(t for t in tset if t > 0):
            for lbl, key in (('err', 'err'), ('G', 'g'), ('S', 's'),
                             ('R', 'r')):
                vals = []
                for p in (6.0, 8.0, 10.0):
                    if (sc, p) in summary:
                        vals.append(next(o[key] for o in summary[(sc, p)]
                                         if o['theta'] == th))
                    else:
                        vals.append(np.nan)
                note('%-6s %-8.4g %9.2f %9.2f %9.2f   %s'
                     % (sc, th, vals[0], vals[1], vals[2], lbl))
        for th in sorted(t for t in tset if t > 0):
            vals = []
            for p in (6.0, 8.0, 10.0):
                if (sc, p) in summary:
                    vals.append(next(o['enr'] for o in summary[(sc, p)]
                                     if o['theta'] == th))
                else:
                    vals.append(np.nan)
            note('%-6s %-8.4g %9.1f %9.1f %9.1f   ENR'
                 % (sc, th, vals[0], vals[1], vals[2]))


if __name__ == '__main__':
    main()
