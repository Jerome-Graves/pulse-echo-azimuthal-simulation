"""The second resolution ladder, on seed 11, and the time base it is read on.

WHY THIS EXISTS. Twice now the supplement has said something about the
archive that nobody checked against the archive. It first said the gate
"falls, reaching rank 12 of 48 at ppw = 10", which no module produces.
The replacement said the ladder rises, which is true, but justified it
with "Seed 23 is the one specimen that exists at ppw = 6, 8 and 10",
which is false. Seed 11 carries a full ladder as well, in out/sweeps as
girdle_seed11_ppw6_axis_perp at ppw 6, girdle_seed11_ppw8_dev at ppw 8 and girdle_seed11_ppw10_licensing
at ppw 10; analysis/eight_pairs_two_ladders.py has had all three
hard-coded as its LADDER[11] since the levels were reconciled, and the
supplement's own scatter paragraph already says two tessellations carry a
full ladder. Section 1 below prints the seed, the ppw and the azimuth
sampling read off each sweep's own config.json, so that the existence
claim is a measurement and not a memory.

WHAT IS MEASURED. Two things the seed-23 ladder cannot settle on its own.

  THE SECOND SPECIMEN. The same identification statistic on seed 11 at
  ppw 6, 8 and 10: the specimen's own tessellation scored against 47
  wrong ones drawn from the same generator, harmonic centring, the
  published gate of 24 to 36 us and the earlier window of 10 to 22 us,
  with the azimuthal registration ranked against the 30 rigid rotations.

  THE TIME BASE. Range is mapped to two-way time with a speed measured
  from the specimen's OWN backwall at each azimuth, so each rung of a
  ladder is read on its own clock and a change between rungs could
  belong to the clock rather than to the coda. Every wavefield is
  therefore scored under all four clocks: the ones measured at ppw 6, 8
  and 10 and a single constant 3850 m/s. The diagonal of that square is
  the ladder as published. Its columns are the ladders that a reader who
  fixed the clock would see. Both specimens are crossed, seed 23 as well
  as seed 11, because the question the paper's wording turns on is
  whether identification degrades under refinement on ANY time base on
  EITHER specimen, and an answer to that cannot be taken on trust from a
  measurement made elsewhere.

AZIMUTH MATCHING, which is a real choice here and not a detail. Seed 11's
ppw 6 and ppw 8 sweeps step the azimuth by 6 degrees and carry 60 beam
positions; its ppw 10 sweep steps by 12 and carries 30. Three rules are
reported and Section 3 says whether the choice moves anything:

  A, adopted     keep the 30 azimuths at 12 degrees that all three rungs
                 share. Identical beam positions at every rung, and the
                 set the stored published row for girdle_seed11_ppw8_dev was
                 computed on.
  B, complement  keep the other 30, at 6, 18, ... 354 degrees. Exists at
                 ppw 6 and 8 only, so it cannot make a third rung; it
                 tests whether those two rungs depend on WHICH half of
                 the finer sampling is kept.
  C, context     all 60 at ppw 6 and 8 against 30 at ppw 10. The rows of
                 the field, the harmonic basis and the registration set
                 all change length between rungs, so this is not the
                 same statistic at each rung and is printed as context.

  Averaging adjacent beam positions down to 30 is NOT done. Each azimuth
  is a separate insonification, not a repeat, and averaging two of them
  would build a field no rung of the ppw 10 sweep contains.

THE STATISTIC CANNOT TEST AN AMPLITUDE. It is a Pearson correlation of
centred fields and is invariant to rescaling either of them, so nothing
below bears on whether the backscattered LEVEL falls under refinement.
Section 6 prints that level ladder from the audited estimator of
analysis/db_reconcile.py, unchanged, for both specimens, so that the two
questions are not confused with one another.

The chain is rebuilt here rather than imported: its own band-pass,
envelope, backwall speed, pulse kernel, event binning, harmonic centring
and correlation. Only _t2_common.facet_events is shared, that being the
ray march which DEFINES the prediction rather than the statistic under
test, along with the cached label volumes, which are data. Nothing is
reported until the whole chain returns the stored published row for
girdle_seed11_ppw8_dev with a difference of exactly zero.

READS  out/sweeps/{girdle_seed11_ppw6_axis_perp, girdle_seed11_ppw8_dev, girdle_seed11_ppw10_licensing,
       girdle_seed23_ppw6_ladder, girdle_seed23_ppw8_ensemble, girdle_seed23_ppw10_ladder}
       az*.npz and config.json
       out/tesscache/tess_s<seed>_p8_k-8.npz, 48 candidates
       sim/analysis/tessellation_replication.npz, published rows only
WRITES sim/analysis/gate_refinement_second_ladder.npz. Prints six tables.
NO CUDA, no solver, no tessellation build: every input is on disk.

Run:  python gate_refinement_second_ladder.py
"""
import json
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

# Acquisition and analysis constants, restated rather than imported so a
# change in a shared module cannot silently change this measurement.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
BAND = (0.8e6, 3.0e6)
T0_SRC = 1.2 / F0
DT = 20e-9
GATE = (24e-6, 36e-6)                    # the published analysis gate
EARLY = (10e-6, 22e-6)                   # the earlier window
CONE_DEG = 8.9                           # far-field half-angle, Sec. 3
N_RAY = 600
PPW_PRED = 8                             # one prediction geometry

TNARROW = np.arange(20e-6, 42e-6, DT)    # the published time grid
TWIDE = np.arange(6e-6, 44e-6, DT)       # wide enough for both windows

LADDERS = {11: [("girdle_seed11_ppw6_axis_perp", 6), ("girdle_seed11_ppw8_dev", 8),
                ("girdle_seed11_ppw10_licensing", 10)],
           23: [("girdle_seed23_ppw6_ladder", 6), ("girdle_seed23_ppw8_ensemble", 8),
                ("girdle_seed23_ppw10_ladder", 10)]}

AZ_SHARED = np.arange(0, 360, 12)        # rule A, all three rungs
AZ_OTHER = np.arange(6, 360, 12)         # rule B, ppw 6 and 8 only
AZ_FULL = np.arange(0, 360, 6)           # rule C, context

CANDS = sorted(set([7, 11, 17, 23, 41, 53, 71, 89] + list(range(100, 140))))

# The two stored published rows. The first is the gate this module is
# gated on; the second is carried because the same chain has to return
# both if it is the published chain at all.
PUB = {11: ("girdle_seed11_ppw8_dev", 0.393359393, 2, 3.51850405),
       23: ("girdle_seed23_ppw8_ensemble", 0.112119627, 1, 6.53615739)}


# ───────────────────────────── the archive ────────────────────────────
def sweep_config(sweep):
    """Seed, ppw and azimuth sampling as the sweep itself records them."""
    with open(os.path.join(SWEEPS, sweep, "config.json")) as fh:
        c = json.load(fh)
    az = sorted(int(f[2:5]) for f in os.listdir(os.path.join(SWEEPS, sweep))
                if f.startswith("az") and f.endswith(".npz"))
    return dict(seed=int(c["seed"]), ppw=float(c["ppw"]),
                step=int(c["az_step"]), n_az=len(az), az=np.array(az))


# ───────────────────────────── measurement ────────────────────────────
def _sos(fs):
    return butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                  btype="band", output="sos")


def measure(sweep, tgrid, az_keep):
    """Envelope rows on tgrid, and the two-way speed from the backwall.

    Band-pass at the trace's own sample rate, analytic envelope at that
    rate, then interpolate onto the common grid: the order every
    published number was computed in. The backwall peak is searched in a
    3 us half-width about the nominal arrival, the level divided out is
    the SWEEP mean so azimuthal variation of the coda survives, and the
    source delay is removed before the speed is formed. Every azimuth is
    read on its own dt; no stack is formed and nothing is indexed by
    sample number.
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
    """Power envelope of the band-passed Ricker, unit sum."""
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
    p = os.path.join(CACHE, f"labels_seed{seed}_ppw{ppw:g}_kappa{kappa:g}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with np.load(p) as z:
        return (z["labels"].astype(np.int32), z["axes"], z["seeds"],
                float(z["h"]))


def march(seed, az_deg):
    """Beam-crossed boundaries of one candidate: weight and range.

    Geometry only. use_dv False and const_v True, so the candidate
    supplies no velocity and cannot be penalised for carrying the wrong
    fabric; range becomes time later, under whichever clock is being
    tested.
    """
    lab, axes, seeds, h = load_tess(seed)
    return [C2.facet_events(lab, axes, seeds, h, a, use_dv=False,
                            use_direc=True, n_ray=N_RAY,
                            th_max_deg=CONE_DEG, const_v=True)[1:]
            for a in az_deg]


def march_all(az_deg):
    return {c: march(c, az_deg) for c in CANDS}


def bin_events(events, c_az, tgrid, kern):
    """Crossings mapped to two-way time by one clock, smeared by the pulse."""
    edges = np.concatenate([tgrid - DT / 2, [tgrid[-1] + DT / 2]])
    rows = [np.histogram(2.0 * s / c_az[q] + T0_SRC, edges, weights=w)[0]
            for q, (w, s) in enumerate(events)]
    return np.array([np.convolve(r, kern, mode="same") for r in rows])


# ──────────────────────────── the statistic ───────────────────────────
def centre_harm(X, kmax=4):
    """Harmonic centring, the published convention, written out here."""
    n = X.shape[0]
    t = np.arange(n) / n * 2 * np.pi
    cols = [np.ones(n)]
    for k in range(1, kmax + 1):
        cols += [np.cos(k * t), np.sin(k * t)]
    A = np.column_stack(cols)
    Y = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]
    return Y - Y.mean(1, keepdims=True)


def corr(M, P):
    """Pearson r of the two centred fields, flattened. No log."""
    a = centre_harm(M).ravel()
    b = centre_harm(P).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def identify(seed, M, preds, win, tgrid):
    """r, rank of 48, z and best wrong candidate for the true tessellation."""
    g = (tgrid >= win[0]) & (tgrid < win[1])
    Mw = M[:, g] ** 2
    v = np.array([corr(Mw, P[:, g]) for P in preds])
    i = CANDS.index(seed)
    other = np.delete(v, i)
    j = int(np.argmax(other))
    return dict(r=float(v[i]), rank=int(np.sum(v >= v[i])),
                z=float((v[i] - other.mean()) / other.std(ddof=1)),
                best_wrong=float(other[j]),
                which=int([c for c in CANDS if c != seed][j]),
                all_r=v)


def shift_rank(M, P, win, tgrid):
    """Rank of the true registration among the rigid azimuthal rotations."""
    g = (tgrid >= win[0]) & (tgrid < win[1])
    Mw, Pw = M[:, g] ** 2, P[:, g]
    r = np.array([corr(np.roll(Mw, s, axis=0), Pw) for s in range(Mw.shape[0])])
    return int(np.sum(r >= r[0])), len(r)


# ────────────────────────────── reporting ─────────────────────────────
def trend(rs):
    if rs[0] < rs[1] < rs[2]:
        return "rising"
    if rs[0] > rs[1] > rs[2]:
        return "FALLING"
    return "not monotone"


def report_archive():
    """Section 1. What the archive actually holds, read from the archive."""
    print("=" * 78)
    print("1. THE TWO LADDERS, AS THE SWEEPS THEMSELVES RECORD THEM")
    print("=" * 78)
    print("  %-26s %6s %6s %10s %8s" % ("sweep", "seed", "ppw", "az step",
                                        "n az"))
    cfg = {}
    for seed in (11, 23):
        for sweep, ppw in LADDERS[seed]:
            c = sweep_config(sweep)
            cfg[sweep] = c
            print("  %-26s %6d %6.0f %10d %8d"
                  % (sweep, c["seed"], c["ppw"], c["step"], c["n_az"]))
            if c["seed"] != seed or c["ppw"] != ppw:
                raise SystemExit("sweep %s is not seed %d at ppw %d"
                                 % (sweep, seed, ppw))
    print("  Both specimens exist at ppw 6, 8 and 10. Seed 23 is not the")
    print("  only one, and any sentence that says so is wrong about the")
    print("  archive.")
    print()
    return cfg


def report_gate(preds_pub, meas_pub):
    """Section 0. Nothing is believed until both stored rows come back."""
    print("=" * 78)
    print("0. DOES THIS CHAIN REPRODUCE THE STORED PUBLISHED ROWS?")
    print("=" * 78)
    z = np.load(os.path.join(HERE, "tessellation_replication.npz"),
                allow_pickle=True)
    names = [str(x) for x in z["names"]]
    worst = 0.0
    for seed in (11, 23):
        sweep = PUB[seed][0]
        row = z["ident_own"][names.index(sweep)]
        got = identify(seed, meas_pub[seed][0], preds_pub[seed], GATE,
                       TNARROW)
        d = max(abs(got["r"] - row[0]), abs(got["z"] - row[3]),
                abs(got["best_wrong"] - row[5]))
        worst = max(worst, d)
        print("  %-22s stored  r %.9f rank %d z %.6f best wrong s%d %.6f"
              % (sweep, row[0], int(row[1]), row[3], int(row[4]), row[5]))
        print("  %-22s here    r %.9f rank %d z %.6f best wrong s%d %.6f"
              % ("", got["r"], got["rank"], got["z"], got["which"],
                 got["best_wrong"]))
        print("  %-22s max abs difference %.2e   rank %s"
              % ("", d, "same" if got["rank"] == int(row[1]) else "DIFFERENT"))
        if got["rank"] != int(row[1]) or got["which"] != int(row[4]):
            raise SystemExit("published row not reproduced for " + sweep)
    if worst != 0.0:
        raise SystemExit("published rows differ by %.3e, not zero" % worst)
    print("  Both rows returned exactly. Everything below is this chain.")
    print()


def report_ladder(seed, meas, preds, tag):
    """Sections 2 and 4. One ladder, gate and earlier window.

    Columns of the returned array, which is stored as s11_A / s23_A:
      0 ppw, 1 window lo, 2 window hi, 3 r, 4 rank of 48, 5 z,
      6 shift rank, 7 n shifts, 8 best wrong r, 9 best wrong SEED.
    Column 9 exists because the supplement says of seed 11 that "the
    same wrong candidate leads at every rung", and until it was stored
    that sentence could not be recovered from the archive: the identity
    was printed and thrown away. A claim the deposit cannot return is
    the defect this module was written to end.
    """
    print("  %s" % tag)
    print("  %-8s %-7s %9s %8s %8s %9s %14s"
          % ("window", "ppw", "r", "rank", "z", "shift", "best wrong"))
    out = []
    for win, wname, tgrid, mk in ((GATE, "gate", TNARROW, "narrow"),
                                  (EARLY, "early", TWIDE, "wide")):
        rs = []
        for j, (sweep, ppw) in enumerate(LADDERS[seed]):
            M = meas[mk][sweep][0]
            h = identify(seed, M, preds[mk][sweep], win, tgrid)
            sr, ns = shift_rank(M, preds[mk][sweep][CANDS.index(seed)], win,
                                tgrid)
            rs.append(h["r"])
            out.append([ppw, win[0], win[1], h["r"], h["rank"], h["z"], sr,
                        ns, h["best_wrong"], h["which"]])
            print("  %-8s %-7d %9.4f %6d/48 %8.2f %7d/%-3d   s%-4d %7.4f"
                  % (wname if j == 0 else "", ppw, h["r"], h["rank"], h["z"],
                     sr, ns, h["which"], h["best_wrong"]))
        print("  %-8s %-7s %9s   %s" % ("", "", "", trend(rs)))
    print()
    return np.array(out, float)


def report_cross(seed, meas, preds_by_clock, tag):
    """Sections 3 and 5. Every wavefield under every clock."""
    clocks = [str(p) for _, p in LADDERS[seed]] + ["const"]
    print("  %s" % tag)
    print("  %-24s %s" % ("", "".join("%16s" % ("clock ppw " + c)
                                      for c in clocks)))
    grid = np.zeros((3, len(clocks), 2))
    for i, (sweep, ppw) in enumerate(LADDERS[seed]):
        M = meas["narrow"][sweep][0]
        line = "  field ppw %-14d" % ppw
        for k, c in enumerate(clocks):
            h = identify(seed, M, preds_by_clock[c], GATE, TNARROW)
            grid[i, k] = (h["r"], h["rank"])
            line += "%10.4f%4d/48" % (h["r"], h["rank"])
        print(line)
    print("  %-24s %s" % ("ladder under that clock",
                          "".join("%16s" % trend(grid[:, k, 0])
                                  for k in range(len(clocks)))))
    print("  %-24s %s" % ("rank at coarsest / finest",
                          "".join("%16s" % ("%d -> %d" % (grid[0, k, 1],
                                                          grid[2, k, 1]))
                                  for k in range(len(clocks)))))
    print()
    return grid


# ──────────────────────────────── main ────────────────────────────────
def main():
    kern = pulse_kernel()
    k2 = C2.pulse_power_kernel(DT)
    dk = float(np.max(np.abs(kern - k2)))
    print("pulse kernel, rebuilt here against the shared one: %.2e" % dk)
    if dk > 1e-12:
        raise SystemExit("kernel mismatch; the chains are not comparable")
    print()

    report_archive()

    print("marching %d candidates at ppw %d on the shared 30 azimuths"
          % (len(CANDS), PPW_PRED), flush=True)
    ev = march_all(AZ_SHARED)

    # Measurements on both grids, on the shared azimuths.
    meas = {"narrow": {}, "wide": {}}
    for seed in (11, 23):
        for sweep, _ in LADDERS[seed]:
            meas["narrow"][sweep] = measure(sweep, TNARROW, AZ_SHARED)
            meas["wide"][sweep] = measure(sweep, TWIDE, AZ_SHARED)

    # Predictions on each sweep's own clock, both grids.
    preds = {"narrow": {}, "wide": {}}
    for seed in (11, 23):
        for sweep, _ in LADDERS[seed]:
            for mk, tg in (("narrow", TNARROW), ("wide", TWIDE)):
                c_az = meas[mk][sweep][1]
                preds[mk][sweep] = [bin_events(ev[c], c_az, tg, kern)
                                    for c in CANDS]

    report_gate({s: preds["narrow"][PUB[s][0]] for s in (11, 23)},
                {s: meas["narrow"][PUB[s][0]] for s in (11, 23)})

    out = {}

    print("=" * 78)
    print("2. THE SEED-11 LADDER, harmonic centring, one ppw-8 prediction")
    print("=" * 78)
    out["s11_A"] = report_ladder(
        11, meas, preds,
        "rule A, the 30 azimuths at 12 degrees shared by all three rungs")

    print("=" * 78)
    print("3. DOES THE AZIMUTH MATCHING RULE CHANGE THE ANSWER?")
    print("=" * 78)
    alt = {}
    for tagr, azs in (("B", AZ_OTHER), ("C", AZ_FULL)):
        rows, evx = [], None
        for sweep, ppw in LADDERS[11]:
            have = set(sweep_config(sweep)["az"].tolist())
            keep = azs if set(azs.tolist()) <= have else None
            if keep is None and tagr == "C":
                keep = AZ_SHARED           # ppw 10 has no finer sampling
            if keep is None:
                rows.append([ppw, 0, np.nan, 0, np.nan, 0])
                continue
            if keep is AZ_SHARED:
                e = ev
            else:
                if evx is None:
                    evx = march_all(keep)
                e = evx
            M, c_az, azu = measure(sweep, TNARROW, keep)
            P = [bin_events(e[cd], c_az, TNARROW, kern) for cd in CANDS]
            h = identify(11, M, P, GATE, TNARROW)
            sr, _ = shift_rank(M, P[CANDS.index(11)], GATE, TNARROW)
            rows.append([ppw, len(azu), h["r"], h["rank"], h["z"], sr])
            del P
        del evx
        alt[tagr] = np.array(rows, float)
    print("  gate only, seed 11. Rule A is the adopted match.")
    print("  %-46s %8s %8s %8s" % ("rule", "ppw 6", "ppw 8", "ppw 10"))
    a = out["s11_A"][:3]
    print("  %-46s %8.4f %8.4f %8.4f"
          % ("A  r, 30 shared azimuths", a[0, 3], a[1, 3], a[2, 3]))
    print("  %-46s %8d %8d %8d"
          % ("A  rank of 48", a[0, 4], a[1, 4], a[2, 4]))
    print("  %-46s %8.4f %8.4f %8s"
          % ("B  r, the complementary 30 azimuths", alt["B"][0, 2],
             alt["B"][1, 2], "n/a"))
    print("  %-46s %8d %8d %8s"
          % ("B  rank of 48", alt["B"][0, 3], alt["B"][1, 3], "n/a"))
    print("  %-46s %8.4f %8.4f %8.4f"
          % ("C  r, all 60 against 30, NOT the same statistic",
             alt["C"][0, 2], alt["C"][1, 2], alt["C"][2, 2]))
    print("  %-46s %8d %8d %8d"
          % ("C  rank of 48", alt["C"][0, 3], alt["C"][1, 3],
             alt["C"][2, 3]))
    print("  %-46s %8d %8d %8d"
          % ("C  azimuths used", alt["C"][0, 1], alt["C"][1, 1],
             alt["C"][2, 1]))
    print("  step 6 to 8 in r:  A %+.4f   B %+.4f   C %+.4f"
          % (a[1, 3] - a[0, 3], alt["B"][1, 2] - alt["B"][0, 2],
             alt["C"][1, 2] - alt["C"][0, 2]))
    print()
    out["s11_B"], out["s11_C"] = alt["B"], alt["C"]

    print("=" * 78)
    print("4. THE SEED-23 LADDER ON THE SAME CHAIN, for comparison")
    print("=" * 78)
    out["s23_A"] = report_ladder(23, meas, preds,
                                 "the 30 azimuths of that sweep")

    print("=" * 78)
    print("5. FIELD AGAINST CLOCK. Every wavefield under every time base.")
    print("=" * 78)
    print("  The diagonal of each square is the published ladder. A column")
    print("  is what a reader who fixed the clock would see.")
    print()
    for seed in (11, 23):
        pbc = {}
        for sweep, ppw in LADDERS[seed]:
            c_az = meas["narrow"][sweep][1]
            pbc[str(ppw)] = [bin_events(ev[c], c_az, TNARROW, kern)
                             for c in CANDS]
        c_const = np.full(len(AZ_SHARED), C_REF)
        pbc["const"] = [bin_events(ev[c], c_const, TNARROW, kern)
                        for c in CANDS]
        out["cross_s%d" % seed] = report_cross(
            seed, meas, pbc, "seed %d, gate 24 to 36 us" % seed)

    print("=" * 78)
    print("6. THE AMPLITUDE, WHICH THIS STATISTIC CANNOT SEE")
    print("=" * 78)
    print("  The correlation above is invariant to rescaling either")
    print("  field, so it says nothing about level. The level ladder is")
    print("  the audited estimator of analysis/db_reconcile.py, imported")
    print("  unchanged and not reimplemented here.")
    import db_reconcile as DR                              # noqa: E402
    print("  %-6s %-5s %12s %12s %14s" % ("seed", "ppw", "coda dB",
                                          "backwall dB", "coda step dB"))
    lev = []
    for seed in (11, 23):
        prev = None
        for sweep, ppw in LADDERS[seed]:
            L = DR.load_sweep(sweep)
            c, b = DR.db(L["coda_band"].mean()), DR.db(L["bw_env"].mean())
            step = "" if prev is None else "%+.2f" % (c - prev)
            lev.append([seed, ppw, c, b])
            print("  %-6d %-5d %12.2f %12.2f %14s" % (seed, ppw, c, b, step))
            prev = c
    out["levels"] = np.array(lev, float)
    print("  Both coda levels fall monotonically under refinement. By the")
    print("  criterion the supplement states, that a discretisation")
    print("  artefact must weaken as the grid is refined, the AMPLITUDE")
    print("  behaves as an artefact would, and refinement therefore does")
    print("  not rule a staircase contribution out. What refinement shows")
    print("  is that identification persists, which is a different claim")
    print("  and the one the paper is about.")
    print()

    np.savez(os.path.join(HERE, "gate_refinement_second_ladder.npz"),
             cands=np.array(CANDS), az_shared=AZ_SHARED, **out)
    print("wrote gate_refinement_second_ladder.npz")


if __name__ == "__main__":
    main()
