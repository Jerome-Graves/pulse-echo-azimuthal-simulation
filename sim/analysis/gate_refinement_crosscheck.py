"""Third-path check on the seed-23 refinement ladder.

WHY THIS EXISTS. The supplement was being rewritten to say that the
analysis gate strengthens under refinement on seed 23, at the time
believed to be the only specimen existing at ppw 6, 8 and 10. (Both
beliefs fell later: seed 11 carries a full ladder too, scored by
gate_refinement_second_ladder.py, and the rise proved to be a property
of the per-rung clock, adjudicated by gate_timebase_cross.py. This
module's own variations are what first broke the rise.) That sentence
was positive evidence for the paper rather than a caveat, so it would
be attacked, and two measurements of it already agreed: a first check
built on tessellation_replication, since removed from the release, and
gate_refinement_audit.py, which rebuilt the estimator from scratch and
reproduced the stored published value for mx_girdle_s23_ppw8 exactly. Agreement between those two establishes that
the number is what the published estimator returns. It does not
establish that the RISE survives the arbitrary choices the estimator
makes, and the rise is the whole claim.

This module therefore asks a different question from either of them: not
"is this the published number" but "does the direction of the ladder
depend on anything that was chosen freely". It imports neither
gate_refinement nor gate_refinement_audit. It varies, one at a time:

  the prediction ray march   n_ray 600 and 1200, rayseed 0 and 1, so
                             that the rise cannot be a fixed draw of
                             cone rays that happens to favour the fine
                             grid
  the azimuth set            all 30, then the two disjoint halves of 15,
                             so that the rise cannot be carried by a few
                             azimuths
  the backwall normalisation  the sweep-mean level divided out, and not
                             divided out at all
  the range-to-time map      the specimen's own per-azimuth backwall
                             speed, and a single constant speed, so that
                             the rise cannot be an artefact of a speed
                             estimate that itself sharpens with the grid

The published gate and centring are held fixed throughout: 24 to 36 us,
harmonic centring, one ppw-8 prediction geometry scored against all
three solutions. The chain is gated on reproducing the stored published
value before any perturbation is reported.

READS  out/sweeps/{lad_girdle_s23_ppw6, mx_girdle_s23_ppw8,
       lad_girdle_s23_ppw10}/az*.npz
       out/tesscache/tess_s<seed>_p8_k-8.npz, 48 candidates
       sim/analysis/tessellation_replication.npz, published value only
WRITES sim/analysis/gate_refinement_crosscheck.npz. Prints four tables.
NO CUDA, no solver, no tessellation build: every input is on disk.

Run:  python gate_refinement_crosscheck.py
"""
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _t2_common as C2                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
SWEEPS = os.path.join(ROOT, "out", "sweeps")
CACHE = os.path.join(ROOT, "out", "tesscache")

C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
BAND = (0.8e6, 3.0e6)
T0_SRC = 1.2 / F0
DT = 20e-9
GATE = (24e-6, 36e-6)
CONE_DEG = 8.9
PPW_PRED = 8

SEED = 23
LADDER = [("lad_girdle_s23_ppw6", 6),
          ("mx_girdle_s23_ppw8", 8),
          ("lad_girdle_s23_ppw10", 10)]
AZ_ALL = np.arange(0, 360, 12)
CANDS = sorted(set([7, 11, 17, 23, 41, 53, 71, 89] + list(range(100, 140))))

PUB = dict(sweep="mx_girdle_s23_ppw8", r=0.112120, z=6.5362)
TGRID = np.arange(20e-6, 42e-6, DT)


# ───────────────────────────── measurement ────────────────────────────
def _sos(fs):
    return butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                  btype="band", output="sos")


def read_sweep(sweep, az_keep):
    """Envelope rows on TGRID, backwall peak level and time, per azimuth."""
    d = os.path.join(SWEEPS, sweep)
    az, rows, lev, tbw = [], [], [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        a = int(f[2:5])
        if a not in az_keep:
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        env = np.abs(hilbert(sosfiltfilt(_sos(fs), tr)))
        k0, w = int(2 * DIA / C_REF * fs), int(3e-6 * fs)
        lo = max(k0 - w, 0)
        seg = env[lo:k0 + w]
        lev.append(seg.max())
        tbw.append((lo + int(np.argmax(seg))) * dt)
        rows.append(np.interp(TGRID, np.arange(len(tr)) * dt, env))
        az.append(a)
    o = np.argsort(az)
    return (np.array(rows)[o], np.array(lev)[o], np.array(tbw)[o],
            np.array(az)[o])


# ───────────────────────────── prediction ─────────────────────────────
def kernel(dt=DT, half_us=2.0):
    fs = 1e9
    tc = np.arange(int(6e-6 * fs)) / fs
    a = (np.pi * F0 * (tc - 3e-6)) ** 2
    e = np.abs(hilbert(sosfiltfilt(_sos(fs), (1.0 - 2.0 * a)
                                   * np.exp(-a)))) ** 2
    n = int(round(half_us * 1e-6 / dt))
    k = np.interp(np.arange(-n, n + 1) * dt + 3e-6, tc, e)
    return k / k.sum()


def load_tess(seed, ppw=PPW_PRED, kappa=-8.0):
    p = os.path.join(CACHE, f"tess_s{seed}_p{ppw:g}_k{kappa:g}.npz")
    with np.load(p) as z:
        return (z["labels"].astype(np.int32), z["axes"], z["seeds"],
                float(z["h"]))


def march_all(az_deg, n_ray, rayseed):
    """Weight and range of every beam-crossed boundary, all candidates."""
    ev = {}
    for c in CANDS:
        lab, axes, seeds, h = load_tess(c)
        ev[c] = [C2.facet_events(lab, axes, seeds, h, a, use_dv=False,
                                 use_direc=True, n_ray=n_ray,
                                 th_max_deg=CONE_DEG, rayseed=rayseed,
                                 const_v=True)[1:]
                 for a in az_deg]
    return ev


def bin_events(events, c_az, kern):
    edges = np.concatenate([TGRID - DT / 2, [TGRID[-1] + DT / 2]])
    rows = [np.histogram(2.0 * s / c_az[q] + T0_SRC, edges, weights=w)[0]
            for q, (w, s) in enumerate(events)]
    return np.array([np.convolve(r, kern, mode="same") for r in rows])


# ──────────────────────────── the statistic ───────────────────────────
def centre_harm(X, kmax=4):
    """Harmonic centring: the published convention, written out here."""
    t = np.arange(X.shape[0]) / X.shape[0] * 2 * np.pi
    cols = [np.ones(X.shape[0])]
    for k in range(1, kmax + 1):
        cols += [np.cos(k * t), np.sin(k * t)]
    A = np.column_stack(cols)
    Y = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]
    return Y - Y.mean(1, keepdims=True)


def corr(M, P):
    a = centre_harm(M).ravel()
    b = centre_harm(P).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def rung(rows, preds, win=GATE):
    """r, rank of 48 and z for the true tessellation in one window."""
    g = (TGRID >= win[0]) & (TGRID < win[1])
    Mw = rows[:, g] ** 2
    v = np.array([corr(Mw, P[:, g]) for P in preds])
    i = CANDS.index(SEED)
    other = np.delete(v, i)
    return (float(v[i]), int(np.sum(v >= v[i])),
            float((v[i] - other.mean()) / other.std(ddof=1)))


def ladder(ev, kern, az_keep, norm=True, const_speed=False):
    """One ladder: r, rank and z at ppw 6, 8 and 10 under one setting."""
    out = []
    for sweep, ppw in LADDER:
        rows, lev, tbw, az = read_sweep(sweep, az_keep)
        if norm:
            rows = rows / float(np.mean(lev))
        c_az = (np.full(len(az), C_REF) if const_speed
                else 2.0 * DIA / (tbw - T0_SRC))
        preds = [bin_events(ev[c], c_az, kern) for c in CANDS]
        out.append((ppw,) + rung(rows, preds))
    return out


def show(tag, lad):
    r = [row[1] for row in lad]
    trend = "rising" if r[0] < r[1] < r[2] else "NOT monotone"
    cells = "".join(f"{x:>9.4f}{k:>4}/48{z:>7.2f}" for _, x, k, z in lad)
    print(f"{tag:>28}{cells}   {trend}")


# ────────────────────────────── report ────────────────────────────────
def main():
    kern = kernel()
    hdr = f"{'':>28}" + "".join(f"{'ppw ' + str(p):>20}" for _, p in LADDER)

    print("\n" + "=" * 88)
    print("0. GATE ON THE PUBLISHED SETTING, AGAINST THE STORED VALUE")
    print("=" * 88)
    ev = march_all(AZ_ALL, 600, 0)
    base = ladder(ev, kern, AZ_ALL)
    z = np.load(os.path.join(HERE, "tessellation_replication.npz"),
                allow_pickle=True)
    names = [str(x) for x in z["names"]]
    row = z["ident_own"][names.index(PUB["sweep"])]
    got = [t for t in base if t[0] == 8][0]
    print(f"  stored npz   r {row[0]:.6f}  rank {int(row[1]):d}  "
          f"z {row[3]:.4f}")
    print(f"  this module  r {got[1]:.6f}  rank {got[2]:d}  z {got[3]:.4f}")
    d = max(abs(got[1] - row[0]), abs(got[3] - row[3]))
    print(f"  max abs difference: {d:.2e}")
    if d > 1e-6:
        raise SystemExit("does not reproduce the published value")

    print("\n" + "=" * 88)
    print("1. DOES THE RISE DEPEND ON THE RAY MARCH?")
    print("=" * 88)
    print(hdr)
    show("n_ray 600, rayseed 0", base)
    rows_out = {"base": np.array(base, float)}
    for n_ray, rs in ((600, 1), (1200, 0)):
        e = march_all(AZ_ALL, n_ray, rs)
        lad = ladder(e, kern, AZ_ALL)
        show(f"n_ray {n_ray}, rayseed {rs}", lad)
        rows_out[f"ray_{n_ray}_{rs}"] = np.array(lad, float)

    print("\n" + "=" * 88)
    print("2. DOES THE RISE DEPEND ON WHICH AZIMUTHS ARE USED?")
    print("=" * 88)
    print(hdr)
    for tag, keep in (("azimuths 0, 24, 48 ...", AZ_ALL[0::2]),
                      ("azimuths 12, 36, 60 ...", AZ_ALL[1::2])):
        e = march_all(keep, 600, 0)
        lad = ladder(e, kern, keep)
        show(tag, lad)
        rows_out["az_" + tag.split()[1].rstrip(",")] = np.array(lad, float)

    print("\n" + "=" * 88)
    print("3. DOES THE RISE DEPEND ON THE NORMALISATION OR THE SPEED?")
    print("=" * 88)
    print(hdr)
    for tag, kw in (("no backwall normalisation", dict(norm=False)),
                    ("one constant speed", dict(const_speed=True)),
                    ("neither", dict(norm=False, const_speed=True))):
        lad = ladder(ev, kern, AZ_ALL, **kw)
        show(tag, lad)
        rows_out["set_" + tag.split()[0]] = np.array(lad, float)

    np.savez(os.path.join(HERE, "gate_refinement_crosscheck.npz"),
             cands=np.array(CANDS), ppw=np.array([p for _, p in LADDER]),
             **rows_out)
    print("\nwrote gate_refinement_crosscheck.npz")


if __name__ == "__main__":
    main()
