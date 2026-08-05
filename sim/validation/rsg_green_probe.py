"""The discrete Green's function of each scheme, with NO interface at all.

WHY THIS AND NOT THE TWO-WAY REFERENCE. The two-way reference measures a
homogeneous direct arrival at four cells along ONE ray, and compares that
ray across separate runs at different tilts. Two things it cannot
separate:

  * a genuine direction dependence of the discrete source and propagator,
    which would be a property of the scheme, from
  * a per-cell sampling accident or a waveform distortion that moves the
    envelope peak between lobes, which would be a property of the
    measurement. All four of its receivers lie on the SAME ray, so
    agreement between them excludes a single-cell accident and nothing
    wider.

It also cannot say whether whatever it sees CONVERGES, because the
interface runs it is attached to only exist at three resolutions and two
useful tilts.

WHAT THIS DOES. One homogeneous run per scheme per medium per resolution,
with receivers at

  ARC   a fixed radius R_ARC, every ARC_STEP degrees in the x-z plane,
        which is the plane every wave in the testbed lives in. In an
        ISOTROPIC medium the physical amplitude and arrival time are the
        same at every one of them, so any variation is the scheme.
  RAY   four radii along each of a few directions, which tests 1/r
        spreading per direction rather than assuming it.
  NBR   cells one and two steps transverse to the arc point at 0 degrees,
        which measures the cell-to-cell scatter of the recorded field and
        so says whether a single-cell reading means anything.

Two media are used. The ISOTROPIC one is the clean instrument: a monopole
in a homogeneous isotropic solid radiates a spherical qP wave, so the arc
must be flat in both amplitude and time, and anything else is
discretisation. The ANISOTROPIC one is grain A of the testbed with its
c-axis along x, i.e. exactly the medium the tilt-0 testbed run uses; its
arc is NOT flat physically, and it is included so that a directional
artefact seen in the isotropic case can be checked to survive in the
medium the published numbers were measured in.

Runs are staged and each one is saved as it finishes, so a kill loses at
most the run in flight.

Nothing here edits tilt_testbed.py, tilt_rsg.py or rsg.py. The RSG time
loop is twoway_ref.forward_rsg_multi, which is gated bit-identical to
rsg.forward_rsg.

usage:  python rsg_green_probe.py [ppw ...]
Writes ../results/rsg_green.log and ../results/rsg_green_<scheme>_<med>_
ppw<n>.npz
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
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))

import fdtd                                                 # noqa: E402
import tilt_testbed as TB                                   # noqa: E402
import twoway_ref as TW                                     # noqa: E402
from ringfwi import anisotropy as an                        # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOG = os.path.join(RESULTS, 'rsg_green.log')

R_ARC = 0.016                       # the testbed image range 2d, to the cell
ARC_STEP = 5.0                      # degrees
RAY_DIRS = (0.0, 30.0, 45.0, 54.7356, 60.0, 90.0)
RAY_FRACS = (0.25, 0.5, 0.75, 1.0)
NBR_OFF = (-2, -1, 1, 2)


def note(msg):
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


# ------------------------------------------------------------------ media

def iso_C():
    """Isotropic ice: qP matched to the testbed beam speed, qS to ice."""
    rho = float(an.ICE_MATERIAL["rho"])
    vp = TB.vA(0.0)
    vs = 1950.0
    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2.0 * mu
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    for i in range(3):
        C[i, i] = lam + 2 * mu
    for i in range(3, 6):
        C[i, i] = mu
    return C, rho, vp


def aniso_C():
    """Grain A of the testbed at tilt 0: c-axis along x."""
    C = TB.grain_C(np.array([1.0, 0.0, 0.0]))
    return C, float(an.ICE_MATERIAL["rho"]), TB.vA(0.0)


MEDIA = dict(iso=iso_C, ani=aniso_C)


# --------------------------------------------------------------- receivers

def build_receivers(N, h, c):
    """(list of (z,y,x), list of labels, arrays of angle and radius).

    Every receiver is a real cell, so the radius it actually sits at is
    recorded and used; snapping moves it by up to h*sqrt(3)/2 and that
    must not be read as an amplitude change.
    """
    p_src = np.array([(c + 0.5) * h] * 3)                   # (x, y, z)
    idx, tag, ang, rad = [], [], [], []

    def add(p, label, a):
        q = np.rint(np.asarray(p) / h - 0.5).astype(int)    # (x, y, z)
        if q.min() < TB.SPONGE or q.max() >= N - TB.SPONGE:
            raise ValueError('receiver %s in the sponge' % label)
        pc = (q + 0.5) * h
        idx.append((int(q[2]), int(q[1]), int(q[0])))
        tag.append(label)
        ang.append(a)
        rad.append(float(np.linalg.norm(pc - p_src)))

    for a in np.arange(0.0, 90.0 + 1e-9, ARC_STEP):
        t = np.radians(a)
        add(p_src + R_ARC * np.array([np.cos(t), 0.0, np.sin(t)]),
            'arc%.1f' % a, a)
    for a in RAY_DIRS:
        t = np.radians(a)
        for f in RAY_FRACS[:-1]:                            # 1.0 is on the arc
            add(p_src + f * R_ARC * np.array([np.cos(t), 0.0, np.sin(t)]),
                'ray%.1f_%.2f' % (a, f), a)
    for o in NBR_OFF:                                       # transverse in z
        add(p_src + np.array([R_ARC, 0.0, o * h]), 'nbz%+d' % o, 0.0)
    for o in NBR_OFF:                                       # transverse in y
        add(p_src + np.array([R_ARC, o * h, 0.0]), 'nby%+d' % o, 0.0)
    return idx, tag, np.asarray(ang), np.asarray(rad)


# ------------------------------------------------------------------- runs

def run_probe(scheme, med, ppw):
    C6, rho, vp = MEDIA[med]()
    h = TB.C_REF / TB.F0 / ppw
    n = int(round(TB.L / h))
    N = n + 2 * (TB.SPONGE + 3)
    c = N // 2
    idx, tag, ang, rad = build_receivers(N, h, c)
    co = fdtd.optimised_coeffs(TB.ORDER, kh_max=TB.KHM, multistart=True)

    if scheme == 'rsg':
        import rsg
        dt = rsg.stable_dt(float(C6.max()), rho, h, co, safety=0.4)
        nt = int(7.0e-6 / dt)
        wav = fdtd.ricker(TB.F0, dt, nt)
        Cf = np.empty((6, 6, N, N, N), np.float32)
        for I in range(6):
            for J in range(6):
                Cf[I, J] = np.float32(C6[I, J])
        src = np.zeros((N, N, N), np.float32)
        src[c, c, c] = 1.0
        pts = [(c, c, c)] + list(idx)
        rec = TW.forward_rsg_multi(Cf, rho, h, dt, nt, src, wav, pts, co,
                                   sponge_width=TB.SPONGE)
    else:
        Ct = np.tile(np.asarray(C6, np.float32).reshape(1, 36), (3, 1))
        rho_t = np.full(3, np.float32(rho))
        lab = np.zeros((N, N, N), np.int32)
        dt = fdtd.safe_dt_labels((lab + 1).astype(np.uint8), Ct, rho_t, h,
                                 co, safety=0.5)
        nt = int(7.0e-6 / dt)
        wav = fdtd.ricker(TB.F0, dt, nt)
        groups = [([(c, c, c)], np.array([1.0]))]
        groups += [([q], np.array([1.0])) for q in idx]
        rec = np.asarray(fdtd.forward_fused_labels(
            lab, None, h, dt, nt, [((c, c, c), 1.0)], wav, groups,
            order=TB.ORDER, coeffs=co, sponge_width=TB.SPONGE,
            mat_tables=(Ct, rho_t)), float)
    return dict(rec=rec, dt=dt, h=h, N=N, tag=np.asarray(tag), ang=ang,
                rad=rad, vp=vp)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    ppws = [float(x) for x in sys.argv[1:]] or [6.0, 10.0]
    note('=' * 78)
    note('GREEN-FUNCTION PROBE  %s   arc radius %.1f mm, step %.1f deg'
         % (time.strftime('%Y-%m-%d %H:%M:%S'), R_ARC * 1e3, ARC_STEP))
    for ppw in ppws:
        for med in ('iso', 'ani'):
            for scheme in ('ssg', 'rsg'):
                out = os.path.join(
                    RESULTS, 'rsg_green_%s_%s_ppw%.0f.npz'
                    % (scheme, med, ppw))
                if os.path.exists(out):
                    note('  %s %s ppw %.0f  already done' % (scheme, med, ppw))
                    continue
                t0 = time.time()
                r = run_probe(scheme, med, ppw)
                np.savez(out, **r)
                note('  %s %s ppw %.0f  N %d  dt %.4f ns  nt %d  %.0f s'
                     % (scheme, med, ppw, r['N'], r['dt'] * 1e9,
                        r['rec'].shape[0], time.time() - t0))
    note('done')


if __name__ == '__main__':
    main()
