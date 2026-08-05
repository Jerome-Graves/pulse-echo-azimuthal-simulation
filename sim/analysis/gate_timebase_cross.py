"""Is the seed-23 gate rise a property of the coda or of the clock?

WHY THIS EXISTS. The supplement is being rewritten a second time. The
first version claimed the analysis gate WEAKENS under refinement,
"reaching rank 12 of 48 at ppw = 10", and was withdrawn as
unreproducible. The replacement claims it STRENGTHENS, and the three
correlations in it are right: gate_refinement_audit.py rebuilt the
estimator from scratch and reproduced the stored published value for
girdle_seed23_ppw8_ensemble exactly, and gate_refinement_crosscheck.py varied
the free choices. The rise did not in fact survive all of them.
gate_refinement_crosscheck.npz holds 0.1814, 0.1148, 0.1338 for the
single constant speed and 0.2458, 0.1484, 0.2436 for one azimuth half,
and neither rises; the supplement calls that flattening, but 0.181
against 0.134 end to end is a reversal and not a flattening.

The constant-speed row is a hint at the ingredient neither module
isolates, because it is not a free choice but a measurement: the clock.

Range is mapped to two-way time with a speed measured from the
specimen's OWN backwall, per azimuth. Each rung of the ladder therefore
carries its own clock, measured on its own wavefield, and the ladder
compares three fields under three different clocks at once. If the
finer grid places its backwall more faithfully, the finer rung is scored
against a better-registered prediction, and r would rise with refinement
even if the coda itself carried nothing extra. The rise and the clock
are confounded in the published ladder and cannot be separated by
anything the two existing modules vary.

This module separates them by crossing the two. Every one of the three
wavefields is scored under every one of the three clocks and under a
single constant speed, all 48 candidates in every cell, harmonic
centring, the published gate, one ppw-8 prediction geometry. The
diagonal of that square IS the published ladder and must reproduce it.
The columns are the question: at a COMMON clock, does the ladder still
rise?

It then asks what the rise is worth as a statement. Three rising numbers
are only impressive against the chance of three rising numbers, so the
same three-rung ladder is run for all 48 candidates and the true
tessellation's end-to-end gain is placed in the distribution of the 47
wrong ones. Both the count of wrong candidates that also rise and the
position of the true gain are reported, because they are different
statements and the weaker one is the one a referee will reach for.

Finally the mechanism is measured rather than asserted: the per-azimuth
backwall pick and the two-way speed it defines, at each resolution, and
the two-way time shift that swapping one rung's clock for another's puts
into the middle of the gate.

The amplitude question is NOT tested here and cannot be. The statistic
is a Pearson correlation of centred fields, invariant to rescaling of
either field, so it is blind to amplitude by construction. The one
amplitude ladder that exists is the decibel ladder of the validation
section, and section 4 below recomputes it from the audited estimator so
that the direction of the level can be stated from measurement.

READS  out/sweeps/{girdle_seed23_ppw6_ladder, girdle_seed23_ppw8_ensemble,
       girdle_seed23_ppw10_ladder}/az*.npz
       out/sweeps/{girdle_seed11_ppw6_axis_perp, girdle_seed11_ppw8_dev, girdle_seed11_ppw10_licensing}
       out/tesscache/tess_s<seed>_p8_k-8.npz, 48 candidates
       sim/analysis/tessellation_replication.npz, published value only
WRITES sim/analysis/gate_timebase_cross.npz. Prints five tables.
NO CUDA, no solver, no tessellation build: every input is on disk.

Run:  python gate_timebase_cross.py
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

# Restated rather than imported, so a change in a shared module cannot
# silently change this measurement.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
BAND = (0.8e6, 3.0e6)
T0_SRC = 1.2 / F0
DT = 20e-9
GATE = (24e-6, 36e-6)                      # the published analysis gate
CONE_DEG = 8.9
PPW_PRED = 8                               # one prediction geometry
N_RAY = 600

SEED = 23
LADDER = [("girdle_seed23_ppw6_ladder", 6),
          ("girdle_seed23_ppw8_ensemble", 8),
          ("girdle_seed23_ppw10_ladder", 10)]
AZ_ALL = np.arange(0, 360, 12)
CANDS = sorted(set([7, 11, 17, 23, 41, 53, 71, 89] + list(range(100, 140))))
TGRID = np.arange(20e-6, 42e-6, DT)        # the published grid

# The stored published row this chain must return before anything else
# is believed, and the ladder the diagonal of the cross must reproduce.
PUB = dict(sweep="girdle_seed23_ppw8_ensemble", r=0.11211963, z=6.53615739)
PUB_LADDER = {6: 0.09563330, 8: 0.11211963, 10: 0.13181387}

# The decibel ladders the validation section states, for section 4. Coda
# level referenced to the source, audited estimator, steps in dB.
#   sections/04-validation.tex and supplementary.tex, sec on whether the
#   ladder is a property of the regime.
PUB_DB_STEPS = {11: (-4.02, -2.68), 23: (-3.72, -2.54)}
DB_SPECIMEN = {11: {6: "girdle_seed11_ppw6_axis_perp", 8: "girdle_seed11_ppw8_dev",
                    10: "girdle_seed11_ppw10_licensing"},
               23: {6: "girdle_seed23_ppw6_ladder", 8: "girdle_seed23_ppw8_ensemble",
                    10: "girdle_seed23_ppw10_ladder"}}


# ───────────────────────────── measurement ────────────────────────────
def _sos(fs):
    return butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                  btype="band", output="sos")


def read_sweep(sweep, az_keep=AZ_ALL):
    """Envelope rows on TGRID, backwall peak level and time, per azimuth.

    Written out in full rather than called: band-pass at the trace's own
    sample rate, analytic envelope at that rate, then interpolate onto
    the common grid, which is the order every published number was
    computed in. The backwall peak is searched in a 3 us half-width
    about the nominal arrival. Every azimuth keeps its own dt.
    """
    d = os.path.join(SWEEPS, sweep)
    az, rows, lev, tbw, dts = [], [], [], [], []
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
        dts.append(dt)
    o = np.argsort(az)
    return dict(rows=np.array(rows)[o], lev=np.array(lev)[o],
                tbw=np.array(tbw)[o], az=np.array(az)[o],
                dt=np.array(dts)[o])


# ───────────────────────────── prediction ─────────────────────────────
def kernel(dt=DT, half_us=2.0):
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


def march_all(az_deg=AZ_ALL, n_ray=N_RAY, rayseed=0):
    """Weight and RANGE of every beam-crossed boundary, all candidates.

    Geometry only, and range rather than time: const_v True, so the
    candidate supplies no velocity at all. The map from range to two-way
    time is applied afterwards by whichever clock is under test, which
    is what makes the cross below possible.
    """
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
    """Marched crossings mapped to two-way time by one clock."""
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
    """Pearson r of the two centred fields, flattened. No log."""
    a = centre_harm(M).ravel()
    b = centre_harm(P).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def all_scores(rows, preds, win=GATE):
    """r against every one of the 48 candidates, in CANDS order."""
    g = (TGRID >= win[0]) & (TGRID < win[1])
    Mw = rows[:, g] ** 2
    return np.array([corr(Mw, P[:, g]) for P in preds])


def cell(v):
    """r, rank of 48 and z for the true tessellation from a score vector."""
    i = CANDS.index(SEED)
    other = np.delete(v, i)
    return (float(v[i]), int(np.sum(v >= v[i])),
            float((v[i] - other.mean()) / other.std(ddof=1)))


# ────────────────────────────── report ────────────────────────────────
def clocks_and_fields(kern):
    """Read the three rungs, build the four clocks, score the square.

    Returns the measured sweeps, the clocks and a dict keyed
    (field ppw, clock tag) holding the full 48-vector of correlations.
    """
    meas = {ppw: read_sweep(sweep) for sweep, ppw in LADDER}
    clk = {str(ppw): 2.0 * DIA / (meas[ppw]["tbw"] - T0_SRC)
           for _, ppw in LADDER}
    clk["const"] = np.full(len(AZ_ALL), C_REF)

    ev = march_all()
    preds = {tag: [bin_events(ev[c], c_az, kern) for c in CANDS]
             for tag, c_az in clk.items()}

    v = {}
    for _, ppw in LADDER:
        rows = meas[ppw]["rows"] / float(np.mean(meas[ppw]["lev"]))
        for tag in preds:
            v[(ppw, tag)] = all_scores(rows, preds[tag])
    return meas, clk, v


def report_gate(v):
    """Nothing is believed until the published number comes back."""
    print("\n" + "=" * 78)
    print("0. DOES THIS CHAIN REPRODUCE THE PUBLISHED NUMBER?")
    print("=" * 78)
    z = np.load(os.path.join(HERE, "tessellation_replication.npz"),
                allow_pickle=True)
    names = [str(x) for x in z["names"]]
    row = z["ident_own"][names.index(PUB["sweep"])]
    r, rank, zz = cell(v[(8, "8")])
    print(f"  stored npz   r {row[0]:.8f}  rank {int(row[1]):d}  "
          f"z {row[3]:.6f}")
    print(f"  this module  r {r:.8f}  rank {rank:d}  z {zz:.6f}")
    d = max(abs(r - row[0]), abs(zz - row[3]))
    print(f"  max abs difference: {d:.2e}")
    if d > 0.0:
        raise SystemExit("does not reproduce the published value exactly")

    bad = []
    for _, ppw in LADDER:
        got = cell(v[(ppw, str(ppw))])[0]
        if abs(got - PUB_LADDER[ppw]) > 5e-8:
            bad.append((ppw, got, PUB_LADDER[ppw]))
    if bad:
        print("\n  *** THE DIAGONAL DOES NOT REPRODUCE THE PUBLISHED "
              "LADDER ***")
        for ppw, got, want in bad:
            print(f"  *** ppw {ppw}: this module {got:.8f} against "
                  f"{want:.8f} ***")
    else:
        print("  the field-clock diagonal reproduces the published ladder "
              "at all three rungs")


def report_cross(v):
    """The 3 x 4 square: every field under every clock."""
    print("\n" + "=" * 78)
    print("1. THE TIME-BASE CROSS, seed 23, published gate, harm centring")
    print("=" * 78)
    print("   rows are the WAVEFIELD, columns the CLOCK that maps range")
    print("   to two-way time. The diagonal is the published ladder.")
    tags = ["6", "8", "10", "const"]
    print(f"\n{'field':>7}" + "".join(f"{'clk ' + t:>21}" for t in tags))
    print(f"{'':>7}" + "".join(f"{'r':>9}{'rank':>6}{'z':>6}" for _ in tags))
    tab = []
    for _, ppw in LADDER:
        line, cells = f"{'ppw ' + str(ppw):>7}", []
        for t in tags:
            r, rank, zz = cell(v[(ppw, t)])
            cells += [r, rank, zz]
            mark = "*" if t == str(ppw) else " "
            line += f"{r:>8.4f}{mark}{rank:>4}/48{zz:>6.2f}"
        tab.append(cells)
        print(line)
    print("   * marks the cell where field and clock are the same rung")

    print(f"\n{'clock':>9}{'r(6)':>9}{'r(8)':>9}{'r(10)':>9}{'6->8':>9}"
          f"{'8->10':>9}{'6->10':>9}{'highest':>9}   shape")
    col = []
    for t in tags:
        rs = [cell(v[(ppw, t)])[0] for _, ppw in LADDER]
        hi = [6, 8, 10][int(np.argmax(rs))]
        if rs[0] < rs[1] < rs[2]:
            d = "monotone rise"
        elif rs[0] > rs[1] > rs[2]:
            d = "monotone fall"
        elif rs[1] < rs[0] and rs[1] < rs[2]:
            d = "V, minimum at ppw 8"
        else:
            d = "not monotone"
        d += f", end to end {rs[2] - rs[0]:+.4f}"
        col.append(rs + [rs[1] - rs[0], rs[2] - rs[1], rs[2] - rs[0], hi])
        print(f"{'clk ' + t:>9}{rs[0]:>9.4f}{rs[1]:>9.4f}{rs[2]:>9.4f}"
              f"{rs[1] - rs[0]:>+9.4f}{rs[2] - rs[1]:>+9.4f}"
              f"{rs[2] - rs[0]:>+9.4f}{'ppw ' + str(hi):>9}   {d}")
    print("   the diagonal is the only place the series rises end to end.")
    print("   Every common clock puts the coarsest grid highest, and every")
    print("   column has the same shape: the 6 to 8 step reverses sign")
    print("   under a common clock, the 8 to 10 step does not.")
    return np.array(tab, float), np.array(col, float)


def report_mechanism(meas, clk):
    """What the clock actually is, at each resolution."""
    print("\n" + "=" * 78)
    print("2. THE CLOCK ITSELF: backwall pick and two-way speed, 30 az")
    print("=" * 78)
    print(f"{'ppw':>5}{'dt ns':>8}{'t_bw us':>11}{'sd ns':>9}{'ptp ns':>9}"
          f"{'sd/dt':>8}{'c m/s':>10}{'sd':>8}{'ptp':>8}{'cv %':>8}")
    rowsout = []
    for _, ppw in LADDER:
        m = meas[ppw]
        dt = float(m["dt"][0])
        t = m["tbw"]
        c = clk[str(ppw)]
        rowsout.append([ppw, dt, t.mean(), t.std(ddof=1), np.ptp(t),
                        c.mean(), c.std(ddof=1), np.ptp(c),
                        100 * c.std(ddof=1) / c.mean()])
        print(f"{ppw:>5}{dt * 1e9:>8.2f}{t.mean() * 1e6:>11.4f}"
              f"{t.std(ddof=1) * 1e9:>9.2f}{np.ptp(t) * 1e9:>9.2f}"
              f"{t.std(ddof=1) / dt:>8.2f}{c.mean():>10.2f}"
              f"{c.std(ddof=1):>8.2f}{np.ptp(c):>8.2f}"
              f"{100 * c.std(ddof=1) / c.mean():>8.3f}")
    print("   sd/dt is the azimuthal scatter of the pick in samples of")
    print("   that rung's own dt: above 1 the pick is moving physically,")
    print("   not just quantising.")

    print("\n   what swapping one clock for another does to the middle of")
    print("   the gate. Entries are the two-way time shift at 30 us,")
    print("   |(t - t0)(1 - c_row/c_col)|, rms and max over 30 azimuths, ns")
    tags = ["6", "8", "10", "const"]
    tmid = 30e-6 - T0_SRC
    print(f"\n{'':>8}" + "".join(f"{'clk ' + t:>16}" for t in tags))
    shift = []
    for a in tags:
        line, cells = f"{'clk ' + a:>8}", []
        for b in tags:
            d = tmid * (1.0 - clk[a] / clk[b])
            rms = float(np.sqrt(np.mean(d ** 2)))
            cells += [rms, float(np.max(np.abs(d)))]
            line += f"{rms * 1e9:>8.1f}{np.max(np.abs(d)) * 1e9:>8.1f}"
        shift.append(cells)
        print(line)
    k = kernel()
    print("   each pair is rms then max. For scale, the pulse power kernel")
    print(f"   that smears the prediction has an equivalent width of "
          f"{DT / np.sum(k ** 2) * 1e9:.0f} ns")
    print(f"   and the analysis grid step is {DT * 1e9:.0f} ns.")

    print("\n   are the three clocks the same clock. Pearson r between the")
    print("   per-azimuth speed vectors, and the largest single-azimuth")
    print("   disagreement in m/s")
    print(f"\n{'':>10}" + "".join(f"{'clk ' + t:>16}" for t in tags[:3]))
    for a in tags[:3]:
        line = f"{'clk ' + a:>10}"
        for b in tags[:3]:
            line += (f"{np.corrcoef(clk[a], clk[b])[0, 1]:>9.4f}"
                     f"{np.max(np.abs(clk[a] - clk[b])):>7.1f}")
        print(line)
    print("   the ppw 8 and ppw 10 clocks agree azimuth by azimuth; the")
    print("   ppw 6 clock is a different vector, not a rescaled one, and")
    print("   that is the whole of the difference between the rungs.")
    return np.array(rowsout, float), np.array(shift, float)


def report_permutation(v):
    """What a rising triple is worth against 47 wrong tessellations."""
    print("\n" + "=" * 78)
    print("3. THE PERMUTATION ON THE GAIN")
    print("=" * 78)
    print("   the same three-rung ladder, own clock per rung, run for all")
    print("   48 candidates. gain = r(ppw 10) - r(ppw 6).")
    tags = ["own", "6", "8", "10", "const"]
    out = {}
    print(f"\n{'clock':>8}{'rise/47':>9}{'p_mono':>9}{'gain_true':>11}"
          f"{'mean_wrong':>12}{'sd_wrong':>10}{'z':>8}{'rank/48':>9}"
          f"{'level':>8}")
    for t in tags:
        R = np.array([v[(ppw, str(ppw) if t == "own" else t)]
                      for _, ppw in LADDER])          # 3 x 48
        mono = (R[0] < R[1]) & (R[1] < R[2])
        gain = R[2] - R[0]
        i = CANDS.index(SEED)
        wrong_mono = int(np.sum(np.delete(mono, i)))
        gw = np.delete(gain, i)
        rank = int(np.sum(gain >= gain[i]))
        z = (gain[i] - gw.mean()) / gw.std(ddof=1)
        out[t] = np.concatenate([[wrong_mono, gain[i], gw.mean(),
                                  gw.std(ddof=1), z, rank], gain])
        print(f"{t:>8}{wrong_mono:>6}/47{wrong_mono / 47:>9.3f}"
              f"{gain[i]:>+11.5f}{gw.mean():>+12.5f}{gw.std(ddof=1):>10.5f}"
              f"{z:>8.2f}{rank:>6}/48{rank / 48:>8.3f}")
        if t == "own":
            o = np.argsort(-gain)
            top = ", ".join(f"s{CANDS[k]} {gain[k]:+.5f}" for k in o[:5])
            print(f"   five largest gains under the published clocks: {top}")
            print(f"   the true tessellation is s{SEED}, and it holds rank "
                  f"{rank} of 48 on the gain")
    print("\n   rise/47 counts WRONG candidates whose own r rises at both")
    print("   steps. p_mono is what a bare monotone triple is worth: the")
    print("   share of wrong tessellations that also produce one. level is")
    print("   the one-sided permutation level of the gain itself.")
    return out


def report_db_ladder():
    """The amplitude ladder, which the correlation cannot test.

    Taken from analysis/db_reconcile.py unchanged rather than
    reimplemented, because the claim under check is about the decibels
    of the validation section and those are that module's estimator:
    local in-band gate power referenced to the source, averaged over the
    revolution as energies. Importing it is the only way to be sure this
    is the same decibel.
    """
    print("\n" + "=" * 78)
    print("4. THE AMPLITUDE LADDER, WHICH THE CORRELATION CANNOT TEST")
    print("=" * 78)
    print("   r is a Pearson correlation of centred fields and is exactly")
    print("   invariant to rescaling either field, so no rank and no z in")
    print("   the tables above says anything about how loud the coda is.")
    print("   The level ladder is recomputed here from the audited")
    print("   estimator and checked against the published steps.")
    import db_reconcile as DR                              # noqa: E402
    print(f"\n{'seed':>6}{'ppw':>5}{'coda/src dB':>14}{'step':>9}"
          f"{'published':>11}{'diff':>8}")
    out = {}
    for seed in (11, 23):
        lv = {}
        for ppw, name in sorted(DB_SPECIMEN[seed].items()):
            sw = DR.load_sweep(name)
            lv[ppw] = DR.level(sw, "coda_band")
        st = [lv[8] - lv[6], lv[10] - lv[8]]
        pub = PUB_DB_STEPS[seed]
        for j, ppw in enumerate((6, 8, 10)):
            s = f"{st[j - 1]:>+9.2f}{pub[j - 1]:>+11.2f}" \
                f"{st[j - 1] - pub[j - 1]:>+8.2f}" if j else f"{'':>28}"
            print(f"{seed:>6}{ppw:>5}{lv[ppw]:>14.2f}{s}")
        out[seed] = np.array([lv[6], lv[8], lv[10], st[0], st[1]], float)
    print("\n   Both ladders fall at both steps. The passage's own stated")
    print("   criterion is that a discretisation artefact must WEAKEN as")
    print("   the grid is refined, and on the amplitude that is what")
    print("   happens, so refinement does not rule a staircase")
    print("   contribution out. What survives refinement is the")
    print("   IDENTIFICATION, which is a different claim.")
    return out


def main():
    kern = kernel()
    print(f"marching {len(CANDS)} candidates at ppw {PPW_PRED} on "
          f"{len(AZ_ALL)} azimuths, then binning under 4 clocks",
          flush=True)
    meas, clk, v = clocks_and_fields(kern)

    report_gate(v)
    tab, col = report_cross(v)
    mech, shift = report_mechanism(meas, clk)
    perm = report_permutation(v)
    dbl = report_db_ladder()

    np.savez(os.path.join(HERE, "gate_timebase_cross.npz"),
             cands=np.array(CANDS), ppw=np.array([p for _, p in LADDER]),
             az=AZ_ALL, cross=tab, cross_by_clock=col, mechanism=mech,
             clock_shift=shift,
             scores=np.array([[v[(p, t)] for t in ("6", "8", "10", "const")]
                              for _, p in LADDER]),
             clocks=np.array([clk[t] for t in ("6", "8", "10", "const")]),
             db_11=dbl[11], db_23=dbl[23],
             **{f"perm_{k}": val for k, val in perm.items()})
    print("\nwrote gate_timebase_cross.npz")


if __name__ == "__main__":
    main()
