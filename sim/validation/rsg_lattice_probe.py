"""Does the recorded field vary smoothly from cell to cell?

WHY. The two-way reference, and the echo metric it is meant to police,
both read the field at ONE cell. That is only meaningful if the field is
smooth on the scale of a cell. The Green-function probe found that on the
rotated grid the cells one step either side of a receiver are more than
13 dB down on it, and the cells two steps either side are only 4 dB down,
which is the signature of a grid-scale (2h) modulation rather than of a
resolved wave.

If that is what is happening then the tilt dependence the two-way
reference measures is not a propagation gain at all. As the tilt changes,
the snapped receiver cell moves between the classes of that modulation,
and the amplitude jumps by whatever the modulation depth is. Nothing
about the physics need change.

WHAT THIS DOES. One homogeneous ISOTROPIC run per scheme per resolution,
with a receiver at EVERY cell along two lines from the source: the x axis
and the x-z face diagonal, out to the testbed's image range. A resolved
wave gives a smooth 1/r decay along both. A grid-scale mode gives a
sawtooth. The modulation depth is then reported directly, and against
resolution, because a modulation that does not shrink as h shrinks is not
a discretisation error of a consistent scheme.

The cells are also sorted by the parity class of their offset from the
source cell, since the rotated operator differences along body diagonals
and the lattice those diagonals generate is the index-4 sublattice of
offsets whose three components share a parity.

usage:  python rsg_lattice_probe.py [ppw ...]
Reads nothing. Writes ../results/rsg_lattice.log and
../results/rsg_lattice_<scheme>_ppw<n>.npz
"""
import os
import sys
import time

import numpy as np
from scipy.signal import hilbert

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
import rsg_green_probe as GP                                # noqa: E402
from ringfwi import anisotropy as an                        # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))
LOG = os.path.join(RESULTS, 'rsg_lattice.log')
R_MAX = 0.016                       # the testbed image range 2d


def note(msg):
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def build_lines(N, h, c):
    """Every cell along +x and along the +x+z face diagonal, out to R_MAX."""
    idx, tag, off = [], [], []
    nx = int(R_MAX / h)
    for m in range(1, nx + 1):
        if c + m >= N - TB.SPONGE:
            break
        idx.append((c, c, c + m))
        tag.append('x%d' % m)
        off.append((0, 0, m))
    nd = int(R_MAX / (h * np.sqrt(2.0)))
    for m in range(1, nd + 1):
        if c + m >= N - TB.SPONGE:
            break
        idx.append((c + m, c, c + m))
        tag.append('d%d' % m)
        off.append((m, 0, m))
    return idx, tag, np.asarray(off)


def run_line(scheme, ppw):
    C6, rho, vp = GP.iso_C()
    h = TB.C_REF / TB.F0 / ppw
    n = int(round(TB.L / h))
    N = n + 2 * (TB.SPONGE + 3)
    c = N // 2
    idx, tag, off = build_lines(N, h, c)
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
        rec = TW.forward_rsg_multi(Cf, rho, h, dt, nt, src, wav,
                                   [(c, c, c)] + idx, co,
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
    return dict(rec=rec, dt=dt, h=h, N=N, vp=vp, off=off,
                tag=np.asarray(tag))


def amps(d):
    """Envelope peak in a gate about the geometric arrival, per receiver."""
    dt, h, vp = d['dt'], d['h'], d['vp']
    r = np.linalg.norm(d['off'] * h, axis=1)
    out = np.zeros(len(r))
    for i in range(len(r)):
        e = np.abs(hilbert(d['rec'][:, i + 1]))
        t0 = r[i] / vp + 1.2 / TB.F0
        w = int(0.5e-6 / dt)
        i0 = int(t0 / dt)
        a, b = max(i0 - w, 0), min(i0 + w, len(e))
        out[i] = e[a:b].max()
    return r, out


def report(scheme, ppw, d):
    r, a = amps(d)
    tags = [str(t) for t in d['tag']]
    note('')
    note('%s ppw %.0f   N %d  h %.4f mm  dt %.4f ns'
         % (scheme.upper(), ppw, d['N'], d['h'] * 1e3, d['dt'] * 1e9))
    for pre, name in (('x', 'along +x'), ('d', 'along the +x+z diagonal')):
        sel = [i for i, t in enumerate(tags) if t.startswith(pre)]
        if not sel:
            continue
        rr, aa = r[sel], a[sel] * r[sel]        # 1/r removed
        off = d['off'][sel]
        # parity class of the offset: the body-diagonal lattice is the set
        # of offsets whose three components share a parity.
        same = np.array([len({int(o[0]) % 2, int(o[1]) % 2,
                              int(o[2]) % 2}) == 1 for o in off])
        far = rr > 0.25 * R_MAX                 # drop the near field
        ref = aa[far].max()
        note('  %s, A*r in dB against the largest, cells beyond %.1f mm'
             % (name, 0.25 * R_MAX * 1e3))
        db = 20 * np.log10(np.maximum(aa, 1e-300) / ref)
        line = '   '
        for i in range(len(rr)):
            if not far[i]:
                continue
            line += '%7.1f' % db[i]
            if len(line) > 95:
                note(line)
                line = '   '
        if line.strip():
            note(line)
        g1 = db[far & same]
        g0 = db[far & ~same]
        note('  cell-to-cell: mean |step| %.2f dB, largest step %.2f dB'
             % (float(np.mean(np.abs(np.diff(db[far])))),
                float(np.max(np.abs(np.diff(db[far]))))))
        if g1.size and g0.size:
            note('  offsets with all three components of one parity: '
                 'mean %+.2f dB (n %d)' % (g1.mean(), g1.size))
            note('  offsets with mixed parity:                        '
                 'mean %+.2f dB (n %d)' % (g0.mean(), g0.size))
            note('  SPLIT BETWEEN THE TWO CLASSES: %.2f dB'
                 % (g1.mean() - g0.mean()))
        note('  spread over all cells beyond the near field: %.2f dB'
             % (db[far].max() - db[far].min()))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    ppws = [float(x) for x in sys.argv[1:]] or [6.0, 8.0, 10.0]
    note('=' * 96)
    note('LATTICE PROBE  %s   isotropic homogeneous medium, no interface'
         % time.strftime('%Y-%m-%d %H:%M:%S'))
    for ppw in ppws:
        for scheme in ('ssg', 'rsg'):
            out = os.path.join(RESULTS,
                               'rsg_lattice_%s_ppw%.0f.npz' % (scheme, ppw))
            if os.path.exists(out):
                d = dict(np.load(out, allow_pickle=True))
                d = {k: (float(v) if v.ndim == 0 and k in
                         ('dt', 'h', 'vp') else
                         (int(v) if k == 'N' else v)) for k, v in d.items()}
            else:
                t0 = time.time()
                d = run_line(scheme, ppw)
                np.savez(out, **d)
                note('  %s ppw %.0f ran in %.0f s' % (scheme, ppw,
                                                      time.time() - t0))
            report(scheme, ppw, d)
    note('done')


if __name__ == '__main__':
    main()
