"""Independent re-measurement of the seed-23 refinement ladder.

WHY THIS EXISTS. The supplement asserts that on seed 23, which exists at
ppw 6, 8 and 10 against one prediction geometry, the earlier window
(10 to 22 us) returns r = 0.557, 0.586 and 0.622 and ranks first of 48 at
every resolution, "while over the same three resolutions this gate falls,
reaching rank 12 of 48 at ppw = 10". Neither the falling rank nor those
three correlations appears in any stored result. A first check, since
removed from the release, already contradicted the falling rank, but it
was built by importing tessellation_replication wholesale, so it could
not be used to check itself: any error in the shared wrapper was
invisible to it.

This module therefore rebuilds the published identification estimator
from the ground up and does not import either of them. It writes its own
band-pass, its own analytic envelope, its own backwall normalisation and
speed estimate, its own pulse power kernel, its own event binning, its
own four centring conventions and its own correlation. Only two things
are taken from the release: _t2_common.facet_events, which is the ray
march that defines the prediction rather than the statistic under test,
and the cached label volumes in out/tesscache, which are data. The
locally built pulse kernel is checked against the shared one and the
whole chain is checked against the stored published number for
mx_girdle_s23_ppw8 before any other row is believed.

Four things are then varied, one at a time, so that the direction of the
result can be attributed:

  the resolution      ppw 6, 8, 10, one prediction geometry throughout
  the centring        raw, col, double, harm; the manuscript quotes harm
                      and the module default is double, which is the
                      likeliest way to obtain a number that reproduces
                      nothing
  the gate edges      the published gate nudged by +-1 and +-2 us on each
                      edge separately and together
  the window          the published gate against the earlier window of
                      the window-sensitivity section, both on the same
                      whole-trace envelope estimator

READS  out/sweeps/{lad_girdle_s23_ppw6, mx_girdle_s23_ppw8,
       lad_girdle_s23_ppw10}/az*.npz
       out/tesscache/tess_s<seed>_p8_k-8.npz, 48 candidates
       sim/analysis/tessellation_replication.npz, published values only
WRITES sim/analysis/gate_refinement_audit.npz. Prints five tables.
NO CUDA, no solver, no tessellation build: every input is on disk.

Run:  python gate_refinement_audit.py
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

# Physics and acquisition constants, restated here rather than imported so
# that a change in the shared module cannot silently change this audit.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
BAND = (0.8e6, 3.0e6)
T0_SRC = 1.2 / F0
DT = 20e-9
GATE = (24e-6, 36e-6)                      # the published analysis gate
EARLY = (10e-6, 22e-6)                     # the earlier window
CONE_DEG = np.degrees(np.radians(8.9))     # far-field half-angle, Sec. 3
N_RAY = 600
PPW_PRED = 8                               # one prediction geometry

SEED = 23
LADDER = [("lad_girdle_s23_ppw6", 6),
          ("mx_girdle_s23_ppw8", 8),
          ("lad_girdle_s23_ppw10", 10)]
AZ_COMMON = np.arange(0, 360, 12)
CANDS = sorted(set([7, 11, 17, 23, 41, 53, 71, 89] + list(range(100, 140))))
MODES = ("raw", "col", "double", "harm")

# Published value this chain must return before anything else is believed.
PUB = dict(sweep="mx_girdle_s23_ppw8", r=0.1121, rank=1, z=6.54)


# ───────────────────────────── measurement ────────────────────────────
def _sos(fs):
    return butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                  btype="band", output="sos")


def measure(sweep, tgrid, az_keep=AZ_COMMON):
    """Envelope on tgrid re the sweep-mean backwall, and the two-way speed.

    Written out in full rather than called: band-pass at the trace's own
    sample rate, analytic envelope at that rate, then interpolate, which is
    the order every published number was computed in. The backwall peak is
    searched in a 3 us half-width around the nominal arrival, the level is
    the SWEEP mean of those peaks so that azimuthal variation of the coda
    survives, and the source delay is removed before the speed is formed.
    """
    d = os.path.join(SWEEPS, sweep)
    az, rows, e1, t1 = [], [], [], []
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
        e1.append(seg.max())
        t1.append((lo + int(np.argmax(seg))) * dt)
        rows.append(np.interp(tgrid, np.arange(len(tr)) * dt, env))
        az.append(a)
    o = np.argsort(az)
    return (np.array(rows)[o] / float(np.mean(e1)),
            2.0 * DIA / (np.array(t1)[o] - T0_SRC),
            np.array(az)[o])


# ───────────────────────────── prediction ─────────────────────────────
def pulse_kernel(dt=DT, half_us=2.0):
    """Power envelope of the band-passed Ricker, unit sum.

    Rebuilt here from the wavelet definition. Checked against the shared
    implementation in main(); the audit stops if they disagree.
    """
    fs = 1e9
    tc = np.arange(int(6e-6 * fs)) / fs
    a = (np.pi * F0 * (tc - 3e-6)) ** 2
    e = np.abs(hilbert(sosfiltfilt(_sos(fs), (1.0 - 2.0 * a)
                                   * np.exp(-a)))) ** 2
    n = int(round(half_us * 1e-6 / dt))
    k = np.interp(np.arange(-n, n + 1) * dt + 3e-6, tc, e)
    return k / k.sum()


def load_tess(seed, ppw=PPW_PRED, kappa=-8.0):
    """Cached label volume, c-axes and seed points. Never builds."""
    p = os.path.join(CACHE, f"tess_s{seed}_p{ppw:g}_k{kappa:g}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with np.load(p) as z:
        return (z["labels"].astype(np.int32), z["axes"], z["seeds"],
                float(z["h"]))


def march(tess, az_deg=AZ_COMMON):
    """Beam-crossed boundaries of a candidate: weight and range, per azimuth.

    Geometry only. use_dv False and const_v True, so the candidate supplies
    no velocity and cannot be penalised for carrying the wrong fabric; the
    range-to-time map comes from the specimen's own backwall speed later.
    """
    lab, axes, seeds, h = tess
    return [C2.facet_events(lab, axes, seeds, h, a, use_dv=False,
                            use_direc=True, n_ray=N_RAY,
                            th_max_deg=CONE_DEG, const_v=True)[1:]
            for a in az_deg]


def bin_events(events, c_az, tgrid, kern):
    """Marched crossings mapped to two-way time and smeared by the pulse."""
    edges = np.concatenate([tgrid - DT / 2, [tgrid[-1] + DT / 2]])
    rows = [np.histogram(2.0 * s / c_az[q] + T0_SRC, edges, weights=w)[0]
            for q, (w, s) in enumerate(events)]
    return np.array([np.convolve(r, kern, mode="same") for r in rows])


# ──────────────────────────── the statistic ───────────────────────────
def harm_basis(n, kmax=4):
    t = np.arange(n) / n * 2 * np.pi
    cols = [np.ones(n)]
    for k in range(1, kmax + 1):
        cols += [np.cos(k * t), np.sin(k * t)]
    return np.column_stack(cols)


def centre(X, mode):
    """The four conventions, written out independently of the release."""
    if mode == "raw":
        return X - X.mean()
    if mode == "col":
        return X - X.mean(0, keepdims=True)
    if mode == "double":
        Y = X - X.mean(0, keepdims=True)
        return Y - Y.mean(1, keepdims=True)
    if mode == "harm":
        A = harm_basis(X.shape[0])
        Y = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]
        return Y - Y.mean(1, keepdims=True)
    raise ValueError(mode)


def score(M, P, mode):
    """Pearson r of the two centred fields, flattened. No log."""
    a = centre(M, mode).ravel()
    b = centre(P, mode).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def identify(M, preds_binned, mode, win, tgrid):
    """r, rank, z and best wrong candidate for the true tessellation."""
    g = (tgrid >= win[0]) & (tgrid < win[1])
    Mw = M[:, g] ** 2
    v = np.array([score(Mw, P[:, g], mode) for P in preds_binned])
    i = CANDS.index(SEED)
    other = np.delete(v, i)
    j = int(np.argmax(other))
    runner = [c for c in CANDS if c != SEED][j]
    return dict(r=float(v[i]), rank=int(np.sum(v >= v[i])),
                z=float((v[i] - other.mean()) / other.std(ddof=1)),
                best_wrong=float(other[j]), which=int(runner))


def shift_rank(M, P, mode, win, tgrid):
    """Rank of the true azimuthal registration among the 30 rigid rotations."""
    g = (tgrid >= win[0]) & (tgrid < win[1])
    Mw, Pw = M[:, g] ** 2, P[:, g]
    r = np.array([score(np.roll(Mw, s, axis=0), Pw, mode)
                  for s in range(Mw.shape[0])])
    return int(np.sum(r >= r[0])), len(r)


# ────────────────────────────── report ────────────────────────────────
def main():
    k_mine = pulse_kernel()
    k_theirs = C2.pulse_power_kernel(DT)
    dk = float(np.max(np.abs(k_mine - k_theirs)))
    print(f"pulse kernel, locally rebuilt against the shared one: "
          f"max abs difference {dk:.2e}")
    if dk > 1e-12:
        raise SystemExit("kernel mismatch; the chains are not comparable")

    # One time grid, wide enough for both windows plus the kernel's 2 us
    # reach, so gate and early window are measured on the same object.
    tgrid = np.arange(6e-6, 44e-6, DT)
    tnarrow = np.arange(20e-6, 42e-6, DT)      # the published grid

    print(f"\nmarching {len(CANDS)} candidate tessellations at ppw "
          f"{PPW_PRED} on {len(AZ_COMMON)} azimuths", flush=True)
    ev = {c: march(load_tess(c)) for c in CANDS}

    meas, measn = {}, {}
    for sweep, _ in LADDER:
        meas[sweep] = measure(sweep, tgrid)
        measn[sweep] = measure(sweep, tnarrow)

    binned, binnedn = {}, {}
    for sweep, _ in LADDER:
        _, c_az, _ = meas[sweep]
        binned[sweep] = [bin_events(ev[c], c_az, tgrid, k_mine)
                         for c in CANDS]
        _, c_azn, _ = measn[sweep]
        binnedn[sweep] = [bin_events(ev[c], c_azn, tnarrow, k_mine)
                          for c in CANDS]

    # ---- 0. reproduce the stored published value before anything else
    print("\n" + "=" * 74)
    print("0. DOES THIS CHAIN REPRODUCE THE PUBLISHED NUMBER?")
    print("=" * 74)
    z = np.load(os.path.join(HERE, "tessellation_replication.npz"),
                allow_pickle=True)
    names = [str(x) for x in z["names"]]
    row = z["ident_own"][names.index(PUB["sweep"])]
    M, _, _ = measn[PUB["sweep"]]
    got = identify(M, binnedn[PUB["sweep"]], "harm", GATE, tnarrow)
    gotd = identify(M, binnedn[PUB["sweep"]], "double", GATE, tnarrow)
    print(f"  stored npz  r {row[0]:.4f}  rank {int(row[1]):d}  "
          f"z {row[3]:.4f}  best wrong s{int(row[4]):d} {row[5]:.4f}")
    print(f"  this module r {got['r']:.4f}  rank {got['rank']:d}  "
          f"z {got['z']:.4f}  best wrong s{got['which']:d} "
          f"{got['best_wrong']:.4f}   (harm, published grid)")
    print(f"  max abs difference in r and z: "
          f"{max(abs(got['r'] - row[0]), abs(got['z'] - row[3])):.2e}")
    print(f"  the module default 'double' would give r {gotd['r']:.4f} "
          f"rank {gotd['rank']:d} z {gotd['z']:.2f}, which is not the "
          f"published number")
    # the wide grid must not move the gate result
    Mw, _, _ = meas[PUB["sweep"]]
    wide = identify(Mw, binned[PUB["sweep"]], "harm", GATE, tgrid)
    print(f"  same on the wide time grid: r {wide['r']:.4f} rank "
          f"{wide['rank']:d} z {wide['z']:.2f}  (delta r "
          f"{abs(wide['r'] - got['r']):.2e})")

    out = {}

    # ---- 1 and 2 and 6. the ladder, published gate, published centring
    print("\n" + "=" * 74)
    print("1/2/6. THE LADDER ON THE PUBLISHED ESTIMATOR (harm, "
          f"{GATE[0] * 1e6:.0f} to {GATE[1] * 1e6:.0f} us)")
    print("=" * 74)
    print(f"{'ppw':>5}{'r_true':>10}{'rank':>7}{'z':>8}{'best wrong':>14}"
          f"{'shift rank':>12}")
    lad = []
    for sweep, ppw in LADDER:
        M, _, _ = meas[sweep]
        h = identify(M, binned[sweep], "harm", GATE, tgrid)
        sr, ns = shift_rank(M, binned[sweep][CANDS.index(SEED)], "harm",
                            GATE, tgrid)
        lad.append([ppw, h["r"], h["rank"], h["z"], h["best_wrong"], sr])
        print(f"{ppw:>5}{h['r']:>10.4f}{h['rank']:>5}/48{h['z']:>8.2f}"
              f"   s{h['which']:<4}{h['best_wrong']:>7.4f}{sr:>8}/{ns}")
    out["ladder_harm_gate"] = np.array(lad, float)

    # ---- 3. all four centring conventions
    print("\n" + "=" * 74)
    print("3. ROBUSTNESS TO THE CENTRING CONVENTION, published gate")
    print("=" * 74)
    print(f"{'centring':>10}" + "".join(
        f"{'ppw ' + str(p):>21}" for _, p in LADDER))
    print(f"{'':>10}" + "".join(f"{'r':>9}{'rank':>6}{'z':>6}"
                                for _ in LADDER))
    cen = []
    for mode in MODES:
        cells, line = [], f"{mode:>10}"
        for sweep, _ in LADDER:
            M, _, _ = meas[sweep]
            h = identify(M, binned[sweep], mode, GATE, tgrid)
            cells += [h["r"], h["rank"], h["z"]]
            line += f"{h['r']:>9.4f}{h['rank']:>4}/48{h['z']:>6.2f}"
        cen.append(cells)
        print(line)
    out["centring"] = np.array(cen, float)

    # ---- 4. nudged gate edges
    print("\n" + "=" * 74)
    print("4. ROBUSTNESS TO THE GATE EDGES, harm centring")
    print("=" * 74)
    nudges = [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
              (0, -2), (0, -1), (0, 1), (0, 2),
              (-2, -2), (-1, -1), (1, 1), (2, 2), (-2, 2), (2, -2)]
    print(f"{'window us':>14}" + "".join(
        f"{'ppw ' + str(p):>21}" for _, p in LADDER) + "   trend")
    nud = []
    for dlo, dhi in nudges:
        win = (GATE[0] + dlo * 1e-6, GATE[1] + dhi * 1e-6)
        cells, line = [], f"{win[0] * 1e6:>7.0f}{win[1] * 1e6:>7.0f}"
        rs = []
        for sweep, _ in LADDER:
            M, _, _ = meas[sweep]
            h = identify(M, binned[sweep], "harm", win, tgrid)
            cells += [h["r"], h["rank"], h["z"]]
            rs.append(h["r"])
            line += f"{h['r']:>9.4f}{h['rank']:>4}/48{h['z']:>6.2f}"
        nud.append([win[0], win[1]] + cells)
        line += "   rising" if rs[0] < rs[1] < rs[2] else "   NOT monotone"
        print(line)
    out["nudged"] = np.array(nud, float)

    # ---- 5. the earlier window on the same estimator
    print("\n" + "=" * 74)
    print(f"5. THE EARLIER WINDOW ({EARLY[0] * 1e6:.0f} to "
          f"{EARLY[1] * 1e6:.0f} us) ON THE SAME ESTIMATOR")
    print("=" * 74)
    print("   the supplement quotes r = 0.557, 0.586, 0.622 for this window")
    for mode in MODES:
        print(f"\n   centring {mode}")
        print(f"{'ppw':>5}{'r_true':>10}{'rank':>7}{'z':>8}"
              f"{'best wrong':>14}{'shift rank':>12}")
        rows = []
        for sweep, ppw in LADDER:
            M, _, _ = meas[sweep]
            h = identify(M, binned[sweep], mode, EARLY, tgrid)
            sr, ns = shift_rank(M, binned[sweep][CANDS.index(SEED)], mode,
                                EARLY, tgrid)
            rows.append([ppw, h["r"], h["rank"], h["z"], h["best_wrong"], sr])
            print(f"{ppw:>5}{h['r']:>10.4f}{h['rank']:>5}/48{h['z']:>8.2f}"
                  f"   s{h['which']:<4}{h['best_wrong']:>7.4f}"
                  f"{sr:>8}/{ns}")
        out[f"early_{mode}"] = np.array(rows, float)

    np.savez(os.path.join(HERE, "gate_refinement_audit.npz"),
             cands=np.array(CANDS), ppw=np.array([p for _, p in LADDER]),
             modes=np.array(MODES), **out)
    print("\nwrote gate_refinement_audit.npz")


if __name__ == "__main__":
    main()
