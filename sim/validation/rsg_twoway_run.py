"""Collect the two-way reference and the echo-to-noise ratio.

Runs the SAME pair the tilt diagnosis runs at each tilt, a bicrystal and
its homogeneous reference, and adds four receivers on the beam axis at
0.25, 0.5, 0.75 and 1.0 times the image range 2d. The extra receivers
cost nothing: they are gathered from fields the run computes anyway, and
the source-cell trace is bit-identical to the published solvers
(twoway_ref.py --gate).

That buys both outstanding checks from one batch.

  TWO-WAY REFERENCE. The homogeneous trace at the image range is the
  amplitude of a direct arrival that has travelled the echo's path
  length, in the echo's direction, through the echo's medium, with no
  interface anywhere. Its variation with tilt is the propagation and
  injection gain G(theta) the differencing cannot remove.

  ECHO TO NOISE. The differenced trace at the source cell is the
  scattered field. The ratio of its envelope peak inside the echo gate
  to its largest envelope value outside the gate and its guard band is
  measured at the PRESENT domain size, which is what the flag asks for.

Usage:  python rsg_twoway_run.py {ssg|rsg} PPW [TILT ...]
Writes ../results/rsg_twoway.log and rsg_twoway_{scheme}_ppw{N}.npz,
saving after every tilt so a killed run loses at most one pair.
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
    sys.path[:0] = [os.path.join(sys.path[0], _d)
                    for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

import tilt_testbed as TB                                   # noqa: E402
import twoway_ref as TW                                     # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOG = os.path.join(RESULTS, 'rsg_twoway.log')

PSI_A, PSI_B = 0.0, 51.0
TREAT = 'naive'            # the standard-grid treatment Table 2 compares


def note(msg):
    line = '%s  %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def timestep(scheme, theta, ppw):
    """The dt the bicrystal run would choose, WITHOUT running it.

    Both solvers set dt from the stiffness tables alone, so the reference
    can be run at the bicrystal's time step without paying for the
    bicrystal. That is what makes the hom-only mode useful: the two-way
    reference needs no interface, and the plane echo at the tilts it is
    wanted for is already published.
    """
    import fdtd
    g = TW.geometry(theta, ppw)
    R, nrm, h, N = g['R'], g['nrm'], g['h'], g['N']
    ca = R @ np.array([np.cos(np.radians(PSI_A)), 0.0,
                       np.sin(np.radians(PSI_A))])
    cb = R @ np.array([np.cos(np.radians(PSI_B)), 0.0,
                       np.sin(np.radians(PSI_B))])
    CA, CB = TB.grain_C(ca), TB.grain_C(cb)
    co = fdtd.optimised_coeffs(TB.ORDER, kh_max=TB.KHM, multistart=True)
    if scheme == 'rsg':
        import rsg
        from ringfwi import anisotropy as an
        return rsg.stable_dt(float(max(CA.max(), CB.max())),
                             float(an.ICE_MATERIAL["rho"]), h, co,
                             safety=0.4)
    fr = TB.frac_field((N, N, N), h, nrm, g['x0'])
    lab = np.rint(fr * (TB.NBIN - 1)).astype(np.int32)
    Ct, rho_t = TB.build_tables(CA, CB, nrm, TREAT)
    return fdtd.safe_dt_labels((lab + 1).astype(np.uint8), Ct, rho_t, h,
                               co, safety=0.5)


def one_tilt(scheme, theta, ppw, hom_only=False):
    g = TW.geometry(theta, ppw)
    idx, r_true, r_nom = TW.ray_receivers(g)
    TW.check_outside_sponge(g, idx)
    bi = None
    dt = timestep(scheme, theta, ppw) if hom_only else None
    if scheme == 'rsg':
        if not hom_only:
            bi, dt, _ = TW.run_rsg(theta, ppw, PSI_A, PSI_B, extra_rec=idx)
        ho, _, d = TW.run_rsg(theta, ppw, PSI_A, PSI_B, homog=True,
                              dt_in=dt, extra_rec=idx)
    else:
        if not hom_only:
            bi, dt, _ = TW.run_ssg(theta, ppw, TREAT, PSI_A, PSI_B,
                                   extra_rec=idx)
        ho, _, d = TW.run_ssg(theta, ppw, TREAT, PSI_A, PSI_B, homog=True,
                              dt_in=dt, extra_rec=idx)
    return dict(theta=theta, dt=dt, d=d, bi=bi, ho=ho,
                r_true=r_true, r_nom=r_nom,
                idx=np.asarray(idx, int), N=g['N'], h=g['h'])


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    argv = sys.argv[1:]
    hom_only = '--hom' in argv
    argv = [a for a in argv if a != '--hom']
    scheme = argv[0]
    ppw = float(argv[1])
    tilts = [float(t) for t in argv[2:]] or [0.0, 30.0, 45.0, 60.0]
    if scheme not in ('ssg', 'rsg'):
        raise SystemExit('scheme must be ssg or rsg')

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS,
                       'rsg_twoway_%s_ppw%g.npz' % (scheme, ppw))
    store = {}
    if os.path.exists(out):
        store = dict(np.load(out, allow_pickle=True))

    note('=' * 72)
    note('two-way reference, %s, n_lambda = %g, psi_b = %g%s'
         % (scheme.upper(), ppw, PSI_B, '  (reference only)'
            if hom_only else ''))
    note('%-9s %9s %12s %12s %12s %8s'
         % ('tilt', 'dt (ns)', 'echo', 'tx at 2d', 'noise', 's'))
    v = TB.vA(PSI_A)
    for theta in tilts:
        key = 'ho_%g' % theta if hom_only else 'bi_%g' % theta
        if key in store:
            note('%-9.4f   already done' % theta)
            continue
        t0 = time.time()
        r = one_tilt(scheme, theta, ppw, hom_only=hom_only)
        t_echo = 2 * r['d'] / v + 1.2 / TB.F0
        if r['bi'] is None:
            echo, noise = np.nan, np.nan
        else:
            dif = r['bi'][:, 0] - r['ho'][:, 0]
            echo, _ = TW.gate_peak(dif, r['dt'], t_echo)
            noise, _, _ = TW.out_of_gate_peak(dif, r['dt'], t_echo)
            store['bi_%g' % theta] = r['bi']
        t_tx = r['r_true'][-1] / v + 1.2 / TB.F0
        tx, _ = TW.gate_peak(r['ho'][:, -1], r['dt'], t_tx, half_us=0.6)
        store['ho_%g' % theta] = r['ho']
        store['dt_%g' % theta] = r['dt']
        store['d_%g' % theta] = r['d']
        store['rtrue_%g' % theta] = r['r_true']
        store['rnom_%g' % theta] = r['r_nom']
        store['idx_%g' % theta] = r['idx']
        store['meta'] = np.array([ppw, r['N'], r['h'], PSI_A, PSI_B])
        np.savez(out, **store)
        note('%-9.4f %9.4f %12.4e %12.4e %12.4e %8.0f'
             % (theta, r['dt'] * 1e9, echo, tx, noise, time.time() - t0))
    note('saved %s' % out)


if __name__ == '__main__':
    main()
