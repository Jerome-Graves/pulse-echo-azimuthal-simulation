"""Why the rotated grid's ZERO-TILT case is its worst, not its exact one.

The testbed defines truth as the zero-tilt run, on the grounds that the
interface then lies on a cell face and every cell is wholly within one
grain. That argument is about CELLS, and it is sound for the standard
staggered operator, whose taps run along the axes: at tilt 0 no tap
crosses the interface except normal to it, and the volume fraction is
exactly 0 or 1.

It does not transfer to the rotated operator, which takes no axial
differences at all. Its taps run along the four body diagonals

    a_m in {(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1)},

at offsets (n - 1/2) h a_m. What a tap does to an interface with unit
normal nrm depends on the projection nrm . a_m: a branch with a zero
projection lies IN the interface and never crosses it, while a branch
with a large projection smears the discontinuity over its whole reach.
The relevant quantity is therefore the mean projected reach

    reach(theta) = <|nrm . a_m|>_m * (M - 1/2) h,   M = len(coeffs),

and the prediction is the opposite of the cell argument: the rotated
operator represents the interface WORST where the cell picture says it
is exact.

At tilt 0, nrm = (1,0,0) and |nrm . a_m| = 1 for all four branches, so
every branch straddles the interface. At 45 degrees two of the four
branches lie exactly in the plane and contribute no smearing at all.
Whether that ordering is what the measured reflection strengths do is
the test; this script prints the two side by side.

Reads the archives rsg_twoway_run.py writes, if present. No GPU.

Usage: python rsg_stencil_reach.py [TILT ...]
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

import fdtd                                                 # noqa: E402
import rsg                                                  # noqa: E402
import tilt_testbed as TB                                   # noqa: E402
import twoway_ref as TW                                     # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))


def measured_R(scheme, ppw, theta):
    """20 log10 (echo / direct arrival at the image range), or nan."""
    f = os.path.join(RESULTS, 'rsg_twoway_%s_ppw%g.npz' % (scheme, ppw))
    if not os.path.exists(f):
        return np.nan
    z = dict(np.load(f, allow_pickle=True))
    if 'bi_%g' % theta not in z:
        return np.nan
    dt = float(z['dt_%g' % theta])
    d = float(z['d_%g' % theta])
    rt = z['rtrue_%g' % theta]
    v = TB.vA(0.0)
    dif = z['bi_%g' % theta][:, 0] - z['ho_%g' % theta][:, 0]
    echo, _ = TW.gate_peak(dif, dt, 2 * d / v + 1.2 / TB.F0)
    tx, _ = TW.gate_peak(z['ho_%g' % theta][:, -1], dt,
                         rt[-1] / v + 1.2 / TB.F0, half_us=0.6)
    return 20 * np.log10(echo / tx)


def main():
    tilts = ([float(t) for t in sys.argv[1:]]
             or [0.0, 15.0, 22.5, 30.0, 45.0, 54.7356, 60.0])
    co = fdtd.optimised_coeffs(TB.ORDER, kh_max=TB.KHM, multistart=True)
    M = len(co)
    print('rotated operator, %d coefficients, taps to %.1f h along each '
          'body diagonal' % (M, (M - 0.5) * np.sqrt(3.0)))
    print('%-9s %28s %6s %9s %9s %9s %9s'
          % ('tilt', '|n.a| for the four diagonals', 'zero',
             'mean|n.a|', 'reach/h', 'R ppw6', 'R ppw8'))
    for th in tilts:
        nrm = TB.rot_y(np.radians(th)) @ np.array([1.0, 0.0, 0.0])
        p = np.abs(rsg.DIAG @ nrm)
        print('%-9.4g %28s %6d %9.4f %9.4f %9.2f %9.2f'
              % (th, np.array2string(p, precision=3), int((p < 1e-9).sum()),
                 p.mean(), p.mean() * (M - 0.5),
                 measured_R('rsg', 6.0, th), measured_R('rsg', 8.0, th)))
    print('')
    print('For comparison the standard operator differences along the '
          'axes, so its')
    print('projected reach at tilt 0 is (M - 1/2) h normal to the '
          'interface on one')
    print('axis and zero on the other two, and no cell is mixed at all.')
    for ppw in (6.0, 8.0, 10.0):
        print('  standard grid R at tilt 0, ppw %g: %6.2f dB'
              % (ppw, measured_R('ssg', ppw, 0.0)))

    # The seven-tilt diagnosis already on disk is an independent test of
    # the same ordering, and a much better one, because it has two pairs
    # of tilts with EQUAL reach (30 with 60, and 40 with 50) that the
    # body-diagonal reading of that experiment treated as unrelated.
    f = os.path.join(RESULTS, 'rsg_diagnose.npz')
    if os.path.exists(f):
        z = np.load(f)
        th, amp = np.asarray(z['theta']), np.asarray(z['amp'])
        err = 20 * np.log10(amp / amp[0])
        reach = np.array([
            np.abs(rsg.DIAG @ (TB.rot_y(np.radians(t))
                               @ np.array([1.0, 0.0, 0.0]))).mean()
            * (M - 0.5) for t in th])
        o = np.argsort(reach)
        print('')
        print('The seven-tilt diagnosis on disk, sorted by projected '
              'reach:')
        print('  %-10s %10s %12s' % ('tilt', 'reach/h', 'error dB'))
        for i in o:
            print('  %-10.4g %10.4f %12.2f' % (th[i], reach[i], err[i]))
        nz = th > 0
        print('  Pearson correlation of error with reach: %+.4f over all '
              'seven, %+.4f over the six non-zero tilts.'
              % (np.corrcoef(reach, err)[0, 1],
                 np.corrcoef(reach[nz], err[nz])[0, 1]))
        print('  Tilt 0 is the outlier of that fit, and it is the tilt '
              'the test calls exact. Reach alone')
        print('  predicts every non-zero tilt to reflect MORE than tilt '
              '0, and two of them read as less;')
        print('  the missing term is the two-way reference, which is '
              'measured separately.')

    # Reach against the interface term S = error - G, the part left once
    # the two-way reference is divided out. That is the quantity the reach
    # argument is actually about.
    print('')
    print('Interface term S = error - G against reach, from the two-way '
          'batch:')
    print('  %-6s %-8s %10s %10s %10s %10s'
          % ('ppw', 'tilt', 'reach/h', 'error dB', 'G dB', 'S dB'))
    for ppw in (6.0, 8.0, 10.0):
        f = os.path.join(RESULTS, 'rsg_twoway_rsg_ppw%g.npz' % ppw)
        if not os.path.exists(f):
            continue
        z = dict(np.load(f, allow_pickle=True))
        ts = sorted(float(k[3:]) for k in z if k.startswith('bi_'))
        if 0.0 not in ts:
            continue
        base = None
        for t in ts:
            dt = float(z['dt_%g' % t])
            d = float(z['d_%g' % t])
            rt = z['rtrue_%g' % t]
            v = TB.vA(0.0)
            dif = z['bi_%g' % t][:, 0] - z['ho_%g' % t][:, 0]
            e, _ = TW.gate_peak(dif, dt, 2 * d / v + 1.2 / TB.F0)
            x, _ = TW.gate_peak(z['ho_%g' % t][:, -1], dt,
                                rt[-1] / v + 1.2 / TB.F0, half_us=0.6)
            x *= rt[-1]
            if base is None:
                base = (e, x)
            er = 20 * np.log10(e / base[0])
            gd = 20 * np.log10(x / base[1])
            rc = np.abs(rsg.DIAG @ (TB.rot_y(np.radians(t))
                                    @ np.array([1.0, 0.0, 0.0]))).mean()
            print('  %-6g %-8.4g %10.4f %10.2f %10.2f %10.2f'
                  % (ppw, t, rc * (M - 0.5), er, gd, er - gd))


if __name__ == '__main__':
    main()
