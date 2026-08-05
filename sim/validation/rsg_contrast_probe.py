"""Is the measured echo proportional to the contrast that makes it?

A second, independent way to ask whether the rotated grid's echo is near
a numerical floor. The echo-to-noise ratio measures how well the echo
stands out in TIME; this measures whether it behaves like a reflection
in AMPLITUDE. Reduce the c-axis misorientation of the far grain and the
true reflection coefficient falls with it. Anything that is a genuine
specular echo must fall by the same factor on both schemes, because the
physics is the same; anything sitting on a floor of grid noise or
round-off will not fall at all.

No theory is needed: the standard grid is the control, and only the
RATIO of the two schemes' responses to the same contrast change is read.

The homogeneous reference depends only on psi_a, so the runs already
stored by rsg_twoway_run.py are reused and only the bicrystal is run
again, at the stored dt so the traces stay commensurate.

Usage: python rsg_contrast_probe.py {ssg|rsg} PPW TILT [PSI_B ...]
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
LOG = os.path.join(RESULTS, 'rsg_contrast_probe.log')
PSI_A = 0.0
LEVELS = (51.0, 25.5, 12.75)


def note(msg):
    line = '%s  %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    scheme, ppw, theta = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    levels = [float(x) for x in sys.argv[4:]] or list(LEVELS)

    src = os.path.join(RESULTS, 'rsg_twoway_%s_ppw%g.npz' % (scheme, ppw))
    if not os.path.exists(src):
        raise SystemExit('need %s first' % src)
    z = dict(np.load(src, allow_pickle=True))
    ho = z['ho_%g' % theta][:, 0]
    dt = float(z['dt_%g' % theta])
    d = float(z['d_%g' % theta])
    g = TW.geometry(theta, ppw)
    idx, _, _ = TW.ray_receivers(g)
    v = TB.vA(PSI_A)
    t_echo = 2 * d / v + 1.2 / TB.F0

    note('=' * 72)
    note('contrast probe, %s, n_lambda = %g, tilt %g, dt held at %.4f ns'
         % (scheme.upper(), ppw, theta, dt * 1e9))
    note('%-8s %12s %10s %10s %8s' % ('psi_b', 'echo', 'vs 51 dB',
                                      'ENR', 's'))
    out = []
    for pb in levels:
        t0 = time.time()
        if scheme == 'rsg':
            bi, _, _ = TW.run_rsg(theta, ppw, PSI_A, pb, dt_in=dt,
                                  extra_rec=idx)
        else:
            bi, _, _ = TW.run_ssg(theta, ppw, 'naive', PSI_A, pb,
                                  dt_in=dt, extra_rec=idx)
        dif = bi[:, 0] - ho
        amp, _ = TW.gate_peak(dif, dt, t_echo)
        nz, _, _ = TW.out_of_gate_peak(dif, dt, t_echo)
        out.append((pb, amp))
        note('%-8.4g %12.4e %10.2f %10.1f %8.0f'
             % (pb, amp, 20 * np.log10(amp / out[0][1]),
                amp / nz if nz > 0 else np.inf, time.time() - t0))
    np.savez(os.path.join(
        RESULTS, 'rsg_contrast_%s_ppw%g_t%g.npz' % (scheme, ppw, theta)),
        psi=np.array([p for p, _ in out]),
        amp=np.array([a for _, a in out]))


if __name__ == '__main__':
    main()
