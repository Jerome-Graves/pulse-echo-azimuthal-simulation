"""Fabric type from one rotational acquisition, and what it costs.

THE CLAIM THIS SUPPORTS, sec:fabrictype in Section 5.1. The azimuthal
time-of-flight pattern carries which of the two fabrics a specimen
holds, in the 180-degree-periodic component and not in the four-fold to
two-fold ratio that the velocity surface first suggests. The separation
is reported as an observation with a measured margin, NOT as a
capability, and this module is the margin budget.

THE DECISION, and why it is the middle one of three.

  Not "no". The mechanism is prior, not fitted. Section 5.1 already
  identifies the two-fold null at kappa = -6.53 as the reason the girdle
  AXIS is not recovered. The same fact predicts a fabric-type
  separation, because it makes the ensemble two-fold amplitude
  0.0119 us at kappa = -8 against 0.3181 us at kappa = +3.93. On the
  thirteen specimens simulated the measured statistic separates the two
  fabrics completely, at the smallest rank-sum level attainable with
  eight and five, and it does so at all three leading-edge thresholds
  the paper uses. Reporting nothing would discard a result the theory
  section already contains.

  Not "a capability" either. The margin is 0.0411 us, which is
  0.08 per cent of the two-way time. Repeating ONE specimen under ONE
  fabric at ppw 6, 8 and 10 moves the statistic by up to 0.0324 us,
  which is 79 per cent of that margin. The separating threshold is
  fitted to these thirteen; a threshold taken from the model instead
  classifies ten of thirteen. The statistic was chosen after five were
  tried on the same data. And it tests for the ABSENCE of a two-fold,
  not for a girdle: the isotropic control is called a girdle.

  So: an observation, with the mechanism, the n, the margin budget and
  the failure modes, and nothing in the abstract.

THE FOUR-FOLD TO TWO-FOLD RATIO, which was the proposal, does not work
and is reported as a negative. The model separates the fabrics by a
factor of 25.8 in it, but only because the girdle denominator is nearly
null; a hundred-grain tessellation does not realise that null, and the
measured ratios overlap between 1.063 and 2.128. Dividing by the
quantity the realisation swamps converts a clean numerator into noise.

WHAT IS INDEPENDENT HERE. The arrival time is re-picked from the raw
traces by this file, and the harmonics are obtained by least squares
rather than by the transform tof_axis_recovery uses, so the per-sweep
table is a second code path and not a second call. Both agree to the
printed precision; the module prints the largest disagreement.

WHAT IT READS
  out/sweeps/<name>/az*.npz        stored traces, no simulation run
  analysis/tofaxis_build_s*.npz    cached tessellations beside
                                   tof_axis_recovery.py
  tof_axis_recovery                geometry, template, oracle, axis fit
  ensemble_stats.measure_trace     the audited coda level of
                                   tab:reconcile, for the comparison

WHAT IT WRITES
  stdout only. No figure, no file, no .tex.

IT TOUCHES NO CUDA. tof_axis_recovery forces the CPU labeller at import
and every tessellation this module needs is already cached, so nothing
is built. Sweep membership is filtered by completeness at load time, so
re-running it as the outstanding single-maximum sweeps land updates
every number with no edit.
"""
import os
import sys

import numpy as np
from scipy import stats as ST
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

import ensemble_stats as ES                            # noqa: E402
import tof_axis_recovery as TAR                        # noqa: E402
# tof_axis_recovery puts sim/ on the path and forces the CPU labeller,
# so specimen must be imported after it and never before.
import specimen as SP                                  # noqa: E402

SWD = ((os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) +
       r"\out\sweeps"))

# Section 3.3. Repeated here rather than imported so that the pick below
# is a second statement of the method and not a second call to it.
C_REF, DIA = 3850.0, 0.100
TOF_FRAC, TOF_HALF_W = 0.25, 2.0e-6
AZ_STEP = 12
AZ30 = np.arange(0.0, 360.0, float(AZ_STEP))

# The reference phase of the four-fold component of the template. It is
# 45 degrees at every concentration, which is what lets the frame be
# fixed without knowing the fabric; check_reference_phase re-derives it.
PHI4_REF = 45.0

GIRDLE = ("girdle_perp_ppw8", "mx_girdle_s7_ppw8", "mx_girdle_s17_ppw8",
          "mx_girdle_s23_ppw8", "mx_girdle_s41_ppw8",
          "mx_girdle_s53_ppw8", "mx_girdle_s71_ppw8",
          "mx_girdle_s89_ppw8")
SINGLE = ("singlemax_ppw8", "mx_single_s7_ppw8", "mx_single_s17_ppw8",
          "mx_single_s23_ppw8", "mx_single_s41_ppw8",
          "mx_single_s53_ppw8", "mx_single_s71_ppw8",
          "mx_single_s89_ppw8")
SEED = {"girdle_perp_ppw8": 11, "singlemax_ppw8": 11}
for _s in (7, 17, 23, 41, 53, 71, 89):
    SEED["mx_girdle_s%d_ppw8" % _s] = _s
    SEED["mx_single_s%d_ppw8" % _s] = _s
KAPPA = {"girdle": -8.0, "single": 3.93}
AXIS = {"girdle": (1.0, 0.0, 0.0), "single": (0.866, 0.5, 0.0)}

# Repeats that hold the specimen AND the fabric fixed and change only
# something carrying no fabric information. These are the margin budget.
RESOLUTION = (("seed 11 girdle, ppw 6 / 8 / 10",
               ("girdle_perp", "girdle_perp_ppw8",
                "lic_girdle_s11_ppw10")),
              ("seed 23 girdle, ppw 6 / 8 / 10",
               ("lad_girdle_s23_ppw6", "mx_girdle_s23_ppw8",
                "lad_girdle_s23_ppw10")),
              ("seed 11 single crystal, ppw 6 / 8 / 10",
               ("zc_s11_ppw6", "zerocontrast_ppw8", "zc_s11_ppw10")))
LADDER = (("cs_f000_s11_ppw8", 0.00), ("cs_f025_s11_ppw8", 0.25),
          ("cs_f050_s11_ppw8", 0.50), ("cs_f075_s11_ppw8", 0.75),
          ("girdle_perp_ppw8", 1.00))
CONTROLS = (("iso_gcal", "isotropic, no fabric type, ppw 6"),
            ("zerocontrast_ppw8", "one crystal, the extreme single max"))
FRACS = (0.15, 0.25, 0.35)
COUNTS = (30, 20, 15, 12, 10, 6)
NDRAW, MC_SEED = 3000, 20260802


# -- load ------------------------------------------------------------
def pick_trace(path, frac=TOF_FRAC):
    """Leading edge at frac of the envelope peak, sub-sample.

    Written from the Section 3.3 description. tof_axis_recovery.load_tof
    is the module Section 5.1 uses; check_pick compares the two.
    """
    with np.load(path) as handle:
        trace = np.asarray(handle["trace"], float).ravel()
        dt = float(handle["dt"])
    fs = 1.0 / dt
    envelope = np.abs(hilbert(trace))
    centre = int(2.0 * DIA / C_REF * fs)
    half = int(TOF_HALF_W * fs)
    lo = max(centre - half, 0)
    seg = envelope[lo:centre + half]
    peak = int(np.argmax(seg))
    thr = frac * seg[peak]
    above = np.nonzero(seg[:peak + 1] >= thr)[0]
    j = int(above[0]) if above.size else peak
    if j > 0 and seg[j] > seg[j - 1]:
        sub = (thr - seg[j - 1]) / (seg[j] - seg[j - 1])
    else:
        sub = 0.0
    return (lo + j - 1 + sub) / fs * 1e6


def azimuths_on_disk(name):
    folder = os.path.join(SWD, name)
    if not os.path.isdir(folder):
        return []
    return sorted(int(f[2:5]) for f in os.listdir(folder)
                  if f.startswith("az") and f.endswith(".npz"))


def complete(name):
    """True if the sweep holds the whole common 12-degree revolution."""
    have = set(azimuths_on_disk(name))
    return all(a in have for a in AZ30.astype(int))


def load_tof(name, frac=TOF_FRAC):
    """Azimuth and arrival time on the common 30-azimuth grid."""
    folder = os.path.join(SWD, name)
    az, tof = [], []
    for deg in azimuths_on_disk(name):
        if deg % AZ_STEP or deg >= 360:
            continue
        az.append(float(deg))
        tof.append(pick_trace(os.path.join(folder, "az%03d.npz" % deg),
                              frac))
    return np.asarray(az), np.asarray(tof)


def load_specimen(name):
    """Realised c-axes and grain volumes. Cached; nothing is built."""
    fabric = "girdle" if "girdle" in name else "single"
    return TAR.load_specimen(SEED[name], KAPPA[fabric],
                             np.asarray(AXIS[fabric]))


def available():
    """Which sweeps are complete, and which are still filling."""
    used_g = [n for n in GIRDLE if complete(n)]
    used_s = [n for n in SINGLE if complete(n)]
    skipped = [(n, len(azimuths_on_disk(n)))
               for n in GIRDLE + SINGLE if not complete(n)]
    return used_g, used_s, skipped


# -- compute ---------------------------------------------------------
def harmonics(az, y, orders=(2, 4)):
    """Amplitude and phase of each order, by least squares.

    A regression rather than a transform, so that it does not share a
    code path with tof_axis_recovery.harmonic. On the uniform grid the
    two agree to rounding; check_harmonics reports by how much.
    """
    th = np.radians(np.asarray(az, float))
    cols = [np.ones_like(th)]
    for k in orders:
        cols += [np.cos(k * th), np.sin(k * th)]
    beta = np.linalg.lstsq(np.stack(cols, axis=1),
                           np.asarray(y, float), rcond=None)[0]
    out = {}
    for i, k in enumerate(orders):
        a, b = beta[1 + 2 * i], beta[2 + 2 * i]
        out[k] = (float(np.hypot(a, b)),
                  float((np.degrees(np.arctan2(b, a)) / k)
                        % (360.0 / k)))
    return out


def locked_two_fold(az, y):
    """The 180-degree-periodic amplitude in the four-fold frame, us.

    The four-fold component is strong under BOTH fabrics and fixes the
    axis modulo 90 degrees. Projecting the two-fold onto that frame
    keeps the part of it the fabric put there and discards the
    random-phase part the realisation put there. The sign is not
    available, because the frame is fixed only modulo 90 degrees and a
    quarter turn reverses a two-fold; the magnitude is.
    """
    h = harmonics(az, y)
    amp2, phi2 = h[2]
    phi4 = h[4][1]
    frame = (phi4 - PHI4_REF) % 90.0
    return abs(amp2 * np.cos(np.radians(2.0 * (phi2 - frame))))


def model_harmonics(kappa):
    """(A2, phi2, A4, phi4) of the ensemble template, axis at zero."""
    grid = np.arange(0.0, 360.0, 1.0)
    h = harmonics(grid, TAR.model_tof(grid, 0.0, float(kappa)))
    return h[2][0], h[2][1], h[4][0], h[4][1]


def girdle_ceiling(lo=0.3, hi=1000.0, n=60):
    """Largest two-fold the model gives anywhere on the girdle branch.

    Quoted over the range evaluated and not to the perfect-girdle limit:
    beyond kappa = -1000 the orientation quadrature loses resolution and
    the returned amplitude falls again, which is numerical.
    """
    best = (0.0, np.nan)
    for kappa in -np.geomspace(lo, hi, n):
        amp = model_harmonics(kappa)[0]
        if amp > best[0]:
            best = (amp, float(kappa))
    return best


def single_crossing(ceiling):
    """Single-maximum concentration whose two-fold clears that ceiling."""
    for kappa in np.linspace(0.2, 8.0, 79):
        if model_harmonics(kappa)[0] > ceiling:
            return float(kappa)
    return np.nan


def sweep_rows(names_g, names_s, frac=TOF_FRAC):
    """Per-sweep measured and noise-free statistics."""
    rows = []
    for name in list(names_g) + list(names_s):
        az, tof = load_tof(name, frac)
        axes, vol = load_specimen(name)
        oracle = TAR.oracle_tof(axes, vol, az)
        h, ho = harmonics(az, tof), harmonics(az, oracle)
        rows.append(dict(
            name=name, girdle=name in names_g, n=len(az),
            mean_tof=float(np.mean(tof)),
            a2=h[2][0], a4=h[4][0], ratio=h[4][0] / h[2][0],
            locked=locked_two_fold(az, tof),
            oracle_a2=ho[2][0], oracle_a4=ho[4][0],
            oracle_ratio=ho[4][0] / ho[2][0],
            oracle_locked=locked_two_fold(az, oracle)))
    return rows


def separation(rows, key):
    """Class ranges, gap and rank-sum level for one statistic."""
    g = np.array([r[key] for r in rows if r["girdle"]])
    s = np.array([r[key] for r in rows if not r["girdle"]])
    u, p = ST.mannwhitneyu(s, g, alternative="two-sided")
    return dict(g=g, s=s, gap=float(s.min() - g.max()),
                auc=float(u / (len(g) * len(s))), p=float(p))


def floor_level(ng, ns):
    """Smallest rank-sum level attainable at these two counts."""
    return float(ST.mannwhitneyu(np.arange(ng, ng + ns),
                                 np.arange(ng),
                                 alternative="two-sided")[1])


def leave_one_out(rows, key):
    """Threshold refitted on the other twelve, applied to the held out.

    With a complete separation this cannot fail, which is exactly why it
    is a weak check and is reported beside the model-set threshold.
    """
    x = np.array([r[key] for r in rows])
    y = np.array([not r["girdle"] for r in rows])
    ok = 0
    for i in range(len(x)):
        keep = np.ones(len(x), bool)
        keep[i] = False
        lo, hi = x[keep][~y[keep]], x[keep][y[keep]]
        thr = 0.5 * (lo.max() + hi.min())
        ok += int((x[i] >= thr) == y[i])
    return ok, len(x)


def fixed_threshold(rows, key, thr):
    ok = sum(int((r[key] >= thr) != r["girdle"]) for r in rows)
    wrong = [r["name"] for r in rows if (r[key] >= thr) == r["girdle"]]
    return ok, len(rows), wrong


def repeat_spread(names, frac=TOF_FRAC):
    """Locked two-fold over repeats of one specimen under one fabric."""
    out = []
    for name in names:
        if not complete(name):
            continue
        az, tof = load_tof(name, frac)
        out.append((name, locked_two_fold(az, tof)))
    vals = [v for _, v in out]
    return out, (max(vals) - min(vals) if len(vals) > 1 else np.nan)


def threshold_sensitivity(names_g, names_s):
    """Class gap at each of the three leading-edge fractions."""
    out = []
    for frac in FRACS:
        g = [locked_two_fold(*load_tof(n, frac)) for n in names_g]
        s = [locked_two_fold(*load_tof(n, frac)) for n in names_s]
        out.append((frac, max(g), min(s), min(s) - max(g)))
    return out


def spread_subsets(n_total, count):
    """Every maximally spread subset of that count, as index arrays."""
    base = np.round(np.linspace(0, n_total, count,
                                endpoint=False)).astype(int)
    seen, out = set(), []
    for off in range(n_total):
        idx = tuple(sorted((base + off) % n_total))
        if len(set(idx)) == count and idx not in seen:
            seen.add(idx)
            out.append(np.array(idx))
    return out


def azimuth_counts(rows_names_g, rows_names_s, thr):
    """Class gap and error rate against azimuth count, threshold held."""
    data = {n: load_tof(n) for n in list(rows_names_g) +
            list(rows_names_s)}
    out = []
    for count in COUNTS:
        subs = spread_subsets(len(AZ30), count)
        gaps, right, total = [], 0, 0
        for idx in subs:
            vals = {n: locked_two_fold(az[idx], tof[idx])
                    for n, (az, tof) in data.items()}
            gaps.append(min(vals[n] for n in rows_names_s) -
                        max(vals[n] for n in rows_names_g))
            for n, v in vals.items():
                total += 1
                right += int((v >= thr) == (n in rows_names_s))
        out.append((count, len(subs), float(np.median(gaps)),
                    100.0 * right / total))
    return out


def draw_fresh(kappa, axis, vols, ndraw=NDRAW, seed=MC_SEED):
    """Locked two-fold of fresh Watson tessellations. No wave at all.

    This is the out-of-sample question the thirteen specimens cannot
    answer: given the fabric, where does the statistic land on a
    tessellation nobody has simulated? Only the specimen enters, so it
    bounds the separation from below, not the measurement.
    """
    rng = np.random.default_rng(seed)
    beams = np.atleast_2d(TAR.beam_direction(AZ30))
    out = np.empty(ndraw)
    for i in range(ndraw):
        vol = vols[i % len(vols)]
        axes = SP.sample_watson(len(vol), np.asarray(axis, float),
                                float(kappa), rng)
        w = vol / vol.sum()
        speed = TAR.qp_velocity(axes @ beams.T)
        out[i] = locked_two_fold(AZ30,
                                 2.0 * DIA * (w @ (1.0 / speed)) * 1e6)
    return out


def coda_levels(names):
    """Audited level of tab:reconcile, energy-averaged over azimuth."""
    out = {}
    for name in names:
        folder = os.path.join(SWD, name)
        power = [ES.measure_trace(os.path.join(folder, "az%03d.npz" % d))
                 [ES.AUDITED] for d in azimuths_on_disk(name)
                 if d % AZ_STEP == 0 and d < 360]
        power = np.asarray(power)
        out[name] = (10.0 * np.log10(power.mean()),
                     100.0 * power.max() / power.sum())
    return out


def check_pick(names):
    """Largest disagreement with tof_axis_recovery's own pick, us."""
    worst = 0.0
    for name in names:
        az, mine = load_tof(name)
        ref_az, ref = TAR.load_tof(name)
        keep = np.isin(ref_az, az)
        worst = max(worst, float(np.max(np.abs(mine - ref[keep]))))
    return worst


def check_harmonics(names):
    """Largest disagreement between the regression and the transform."""
    worst = 0.0
    for name in names:
        az, tof = load_tof(name)
        mine = harmonics(az, tof)
        for k in (2, 4):
            worst = max(worst, abs(mine[k][0] - TAR.harmonic(az, tof,
                                                             k)[0]))
    return worst


def check_reference_phase():
    """Four-fold reference phase across the whole concentration range."""
    ks = (-1000.0, -100.0, -20.0, -8.0, -2.0, -0.3, 0.3, 1.8, 3.93,
          10.0, 100.0, 400.0)
    return [(k, model_harmonics(k)[3]) for k in ks]


# -- draw ------------------------------------------------------------
def draw_header(used_g, used_s, skipped, worst_pick, worst_harm):
    print("FABRIC TYPE FROM ONE ROTATIONAL ACQUISITION, AND ITS MARGIN")
    print("  girdle sweeps used  (%d): %s" % (len(used_g),
                                              ", ".join(used_g)))
    print("  single-max sweeps   (%d): %s" % (len(used_s),
                                              ", ".join(used_s)))
    for name, have in skipped:
        print("  SKIPPED, %2d of 30 azimuths on the common grid: %s"
              % (have, name))
    print("  every statistic on the %d shared azimuths at %d degrees"
          % (len(AZ30), AZ_STEP))
    print("  arrival time re-picked here from the raw traces; largest")
    print("    disagreement with tof_axis_recovery.load_tof %.3e us"
          % worst_pick)
    print("  harmonics by least squares, not by the transform; largest")
    print("    disagreement in amplitude %.3e us" % worst_harm)


def draw_model(ceiling, crossing, phases):
    print("\nWHAT THE ENSEMBLE PREDICTS, from Section 5.1's template")
    print("  %-16s %10s %10s %10s %10s" %
          ("fabric", "A2 us", "A4 us", "A4/A2", "phi2 deg"))
    for tag, kappa in (("single maximum", KAPPA["single"]),
                       ("girdle", KAPPA["girdle"])):
        a2, p2, a4, _ = model_harmonics(kappa)
        print("  %-16s %10.5f %10.5f %10.3f %10.3f"
              % ("%s k %+.2f" % (tag, kappa), a2, a4, a4 / a2, p2))
    a2s = model_harmonics(KAPPA["single"])[0]
    a2g, _, a4g, _ = model_harmonics(KAPPA["girdle"])
    a4s = model_harmonics(KAPPA["single"])[2]
    print("  the ratio separates them by a factor of %.1f, but only"
          % ((a4g / a2g) / (a4s / a2s)))
    print("    because the girdle two-fold is nearly null: %.5f us"
          % a2g)
    print("  the two-fold separates them by a factor of %.1f"
          % (a2s / a2g))
    print("  two-fold null at kappa %.4f" % TAR.two_fold_null_kappa())
    print("  four-fold reference phase against concentration:")
    print("    " + "  ".join("%g:%.2f" % kp for kp in phases))
    print("  largest model two-fold on the girdle branch %.4f us at"
          " kappa %.0f" % ceiling)
    print("  the single-maximum branch clears it at kappa %+.2f"
          % crossing)


def draw_rows(rows):
    print("\nPER SWEEP, measured on the shared grid, and noise free")
    print("  %-22s %7s %7s %7s %7s | %7s %7s"
          % ("sweep", "A2", "A4", "A4/A2", "|c2|", "or A4/A2", "or|c2|"))
    for tag, want in (("girdle, kappa -8", True),
                      ("single maximum, kappa +3.93", False)):
        print("  -- %s" % tag)
        for r in rows:
            if r["girdle"] is not want:
                continue
            print("  %-22s %7.4f %7.4f %7.3f %7.4f | %7.3f %7.4f"
                  % (r["name"], r["a2"], r["a4"], r["ratio"],
                     r["locked"], r["oracle_ratio"],
                     r["oracle_locked"]))


def draw_validation(rows, ceiling):
    print("\nVALIDATION")
    for key, tag in (("ratio", "A4/A2, as proposed"),
                     ("a2", "two-fold amplitude A2"),
                     ("locked", "two-fold locked to the four-fold"),
                     ("oracle_ratio", "A4/A2, noise free"),
                     ("oracle_locked", "locked two-fold, noise free")):
        sep = separation(rows, key)
        print("  %-32s girdle %7.4f to %7.4f  single %7.4f to %7.4f"
              % (tag, sep["g"].min(), sep["g"].max(), sep["s"].min(),
                 sep["s"].max()))
        print("  %-32s gap %+8.4f   AUC %.3f   rank p %.5f"
              % ("", sep["gap"], sep["auc"], sep["p"]))
    ng = sum(r["girdle"] for r in rows)
    print("  smallest rank-sum level attainable at %d and %d is %.5f"
          % (ng, len(rows) - ng, floor_level(ng, len(rows) - ng)))
    ok, tot = leave_one_out(rows, "locked")
    print("  locked two-fold, threshold refitted per fold: %d of %d"
          % (ok, tot))
    print("    (a complete separation cannot fail this; it is reported")
    print("     only so that the weaker check is on the record)")
    ok, tot, wrong = fixed_threshold(rows, "locked", ceiling)
    print("  locked two-fold at the MODEL threshold %.4f us: %d of %d"
          % (ceiling, ok, tot))
    print("    misread: %s" % ", ".join(wrong))
    print("    this is the out-of-sample number, not the %d of %d"
          % (leave_one_out(rows, "locked")[0], tot))
    mean_tof = float(np.mean([r["mean_tof"] for r in rows]))
    gap = separation(rows, "locked")["gap"]
    print("  the gap is %.4f us on a two-way time of %.1f us, %.3f "
          "per cent" % (gap, mean_tof, 100.0 * gap / mean_tof))


def draw_margin(res, ladder, sens, gap, coda):
    print("\nMARGIN BUDGET. The gap to beat is %.4f us." % gap)
    print("  RESOLUTION, one specimen and one fabric at ppw 6 / 8 / 10")
    for tag, items, spread in res:
        print("    %s" % tag)
        for name, val in items:
            print("      %-24s |c2| %.4f" % (name, val))
        print("      spread %.4f us, which is %.0f per cent of the gap"
              % (spread, 100.0 * spread / gap))
    print("  CONTRAST, one tessellation and one fabric, f = 0 to 1")
    for (name, f), val in ladder[0]:
        print("    f %.2f  %-22s |c2| %.4f" % (f, name, val))
    print("    spread %.4f us, %.0f per cent of the gap"
          % (ladder[1], 100.0 * ladder[1] / gap))
    print("  LEADING-EDGE THRESHOLD, the same traces")
    for frac, gmax, smin, g in sens:
        print("    %2.0f per cent: girdle max %.4f  single min %.4f  "
              "gap %+.4f  %s" % (100 * frac, gmax, smin, g,
                                 "separated" if g > 0 else "NOT"))
    print("  FOR COMPARISON, the coda under the same resolution test:")
    print("    class separation %.2f dB against a same-specimen drift "
          "of %.2f dB" % coda)
    print("    the time-of-flight statistic passes this control; the")
    print("    coda does not, and that is the reason to report one and")
    print("    not the other")


def draw_out_of_sample(lg, ls, thr_fit, thr_model):
    print("\nOUT OF SAMPLE, %d fresh Watson tessellations per fabric,"
          % len(lg))
    print("  realised grain volumes reused, NO wave propagated")
    for tag, v in (("girdle  kappa -8.00", lg),
                   ("single  kappa +3.93", ls)):
        q = np.percentile(v, [5, 50, 95, 99])
        print("  %s  median %.4f  5 to 95 pc %.4f to %.4f  99 pc %.4f"
              % (tag, q[1], q[0], q[2], q[3]))
    for tag, thr in (("fitted to the thirteen", thr_fit),
                     ("set from the model", thr_model)):
        fp, fn = float(np.mean(lg >= thr)), float(np.mean(ls < thr))
        print("  threshold %.4f us (%s): girdle read as single %.1f per"
              " cent," % (thr, tag, 100 * fp))
        print("    single read as girdle %.1f per cent, balanced %.1f"
              % (100 * fn, 100 * 0.5 * (fp + fn)))
    pair = float(np.mean(lg[:, None] > ls[None, :]))
    print("  P(a girdle draw exceeds a single-maximum draw) = %.4f"
          % pair)
    print("  the SPECIMEN therefore separates the fabrics; what this")
    print("  cannot show is whether the measurement follows it, which")
    print("  is what the margin budget above is for")


def draw_counts(table, thr):
    print("\nAZIMUTH COUNT, threshold held at %.4f us" % thr)
    print("  %-9s %-9s %-11s %s" % ("azimuths", "subsets",
                                    "median gap", "correct"))
    for count, nsub, gap, pct in table:
        print("  %-9d %-9d %+-11.4f %.1f per cent" % (count, nsub, gap,
                                                      pct))
    print("  on N equally spaced azimuths harmonic k cannot be told")
    print("  from N - k, so the four-fold collapses onto the two-fold")
    print("  at N = 6 and sits on its own Nyquist at N = 8. The frame")
    print("  the statistic is projected onto does not exist below 10.")


def draw_controls(vals):
    print("\nCONTROLS")
    for name, note, a2, a4, c2, call in vals:
        print("  %-20s |c2| %.4f  A4/A2 %6.3f  read as %-14s (%s)"
              % (name, c2, a4 / a2, call, note))
    print("  the isotropic specimen has NO fabric type and no two-fold")
    print("  either, so it is read as a girdle. The statistic tests for")
    print("  the absence of a two-fold, not for a girdle.")


def draw_coda(levels, pairs):
    print("\nTHE CODA, on the same sweeps, for the comparison")
    diffs = np.array([levels[s][0] - levels[g][0] for _, g, s in pairs])
    for (seed, g, s), d in zip(pairs, diffs):
        print("  seed %-3d girdle %7.2f  single %7.2f  difference %+6.2f"
              % (seed, levels[g][0], levels[s][0], d))
    n = len(diffs)
    t = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(n))
    print("  n %d pairs, single louder by %+.2f dB (sd %.2f), t %+.2f,"
          % (n, diffs.mean(), diffs.std(ddof=1), t))
    print("    p %.4f, 95 per cent interval %.2f to %.2f dB"
          % (2 * (1 - ST.t.cdf(abs(t), n - 1)),
             diffs.mean() - ST.t.ppf(0.975, n - 1) * diffs.std(ddof=1)
             / np.sqrt(n),
             diffs.mean() + ST.t.ppf(0.975, n - 1) * diffs.std(ddof=1)
             / np.sqrt(n)))
    print("  brightest azimuth share, single maxima: %s"
          % ", ".join("%.1f" % levels[s][1] for _, _, s in pairs))
    print("  brightest azimuth share, girdles:       %s"
          % ", ".join("%.1f" % levels[g][1] for _, g, _ in pairs))


def main():
    used_g, used_s, skipped = available()
    names = used_g + used_s
    draw_header(used_g, used_s, skipped, check_pick(names),
                check_harmonics(names))

    ceiling, ceiling_k = girdle_ceiling()
    draw_model((ceiling, ceiling_k), single_crossing(ceiling),
               check_reference_phase())

    rows = sweep_rows(used_g, used_s)
    draw_rows(rows)
    draw_validation(rows, ceiling)

    sep = separation(rows, "locked")
    gap, thr_fit = sep["gap"], 0.5 * (sep["g"].max() + sep["s"].min())

    res = []
    for tag, group in RESOLUTION:
        items, spread = repeat_spread(group)
        res.append((tag, items, spread))
    ladder_items, ladder_spread = repeat_spread([n for n, _ in LADDER])
    ladder = ([((n, f), v) for (n, f), (_, v)
               in zip(LADDER, ladder_items)], ladder_spread)
    rungs = [n for n in RESOLUTION[0][1] if n not in names]
    coda_all = coda_levels(names + rungs)
    g_lv = np.array([coda_all[n][0] for n in used_g])
    s_lv = np.array([coda_all[n][0] for n in used_s])
    drift = (max(coda_all[n][0] for n in RESOLUTION[0][1]) -
             min(coda_all[n][0] for n in RESOLUTION[0][1]))
    draw_margin(res, ladder, threshold_sensitivity(used_g, used_s), gap,
                (float(s_lv.mean() - g_lv.mean()), float(drift)))

    vols = [load_specimen(n)[1] for n in used_g]
    draw_out_of_sample(draw_fresh(KAPPA["girdle"], AXIS["girdle"], vols),
                       draw_fresh(KAPPA["single"], AXIS["single"], vols),
                       thr_fit, ceiling)

    draw_counts(azimuth_counts(used_g, used_s, thr_fit), thr_fit)

    control_vals = []
    for name, note in CONTROLS:
        az, tof = load_tof(name)
        h = harmonics(az, tof)
        c2 = locked_two_fold(az, tof)
        control_vals.append((name, note, h[2][0], h[4][0], c2,
                             "girdle" if c2 < thr_fit
                             else "single maximum"))
    draw_controls(control_vals)

    pairs = [(SEED[g], g, s) for g in used_g for s in used_s
             if SEED[g] == SEED[s]]
    draw_coda(coda_all, pairs)


if __name__ == "__main__":
    main()
