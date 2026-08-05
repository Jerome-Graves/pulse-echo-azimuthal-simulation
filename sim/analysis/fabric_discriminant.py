"""What a single pulse-echo revolution can say about the fabric.

Supports the two capability claims of Section 5.1: recovery of the
predominant orientation, and recognition of which fabric type a specimen
carries, single maximum against girdle.

THE DECISIONS THIS MODULE MAKES, and the evidence for each.

  1. THE AXIS IS FIXED MODULO 90 DEGREES FOR A GIRDLE, NOT MODULO 45.
  Every one of the eight girdle tessellations lands either on its own
  sample axis or 71 to 87 degrees from it, and folding the error onto
  [0, 45] gives exactly the same eight numbers as folding it onto
  [0, 22.5]. The mod-45 statement is true but weak: the data support the
  stronger one. The single maximum needs no such qualification.

  2. THE GIRDLE BRANCH IS NOT RESOLVABLE, AND NOT BY ADDING AZIMUTHS.
  Which of the two 90-degree branches is correct is carried by the
  two-fold component of the velocity surface, whose ensemble amplitude at
  kappa = -8 is 0.0119 us. The two-fold a 100-grain tessellation actually
  realises is 0.022 to 0.168 us with a phase unrelated to the axis, so
  the informative term is buried under the specimen's own realisation
  before any wave is launched. The measured branch is right in four of
  eight and the noise-free orientation average is right in five of eight,
  both chance. Azimuths do not help: the realisation term is a property
  of the specimen and does not average down over beam positions. The
  grain count is what would have to change, and the realised two-fold
  reaches the ensemble value only near 25600 grains against the 100 here.

  3. THE FOUR-FOLD TO TWO-FOLD RATIO IS NOT THE DISCRIMINANT IT LOOKS.
  In the model A4/A2 is 20.57 at kappa = -8 and 0.798 at kappa = 3.93, a
  factor of 25.8. Measured, the two families overlap outright: 1.06 to
  8.17 for the girdle against 0.30 to 2.13 for the single maximum. The
  ratio fails for the reason above. Its denominator is the quantity the
  realisation swamps, so dividing by it converts a clean numerator into
  noise.

  4. WHAT DOES SEPARATE IS THE TWO-FOLD AMPLITUDE PHASE-LOCKED TO THE
  FOUR-FOLD FRAME. The four-fold is strong in both fabrics and fixes the
  axis modulo 90 degrees; projecting the two-fold onto that frame keeps
  the part of it the fabric put there and discards the random-phase part
  the realisation put there. On the common 30-azimuth grid the eight
  girdles give 0.029 to 0.138 us and the five single maxima 0.179 to
  0.652 us, separated with a gap. This is reported as a capability with
  three honest qualifications, all printed below: the threshold that
  separates them was read off these thirteen specimens and is not an
  out-of-sample number; the model-set threshold, the largest two-fold any
  girdle concentration can produce, gets ten of thirteen; and the
  statistic is a two-fold-amplitude test, so an isotropic specimen is
  classified as a girdle, which the iso_gcal control confirms.

  5. THE CODA LEVEL IS NOT A USABLE FABRIC DISCRIMINANT. On the five
  matched pairs now available the single maximum is louder by 3.53 dB,
  paired t = 2.09 on 4 degrees of freedom, p = 0.104. The apparent
  contrast is carried by two of the five single-maximum realisations,
  seeds 7 and 17, in each of which one azimuth of thirty carries 46 per
  cent of the revolution energy. As a classifier the level ranks eight
  girdles against five single maxima at p = 0.0062 but a leave-one-out
  threshold gets only nine of thirteen, and the seed-41 pair is inverted.

THE ESTIMATORS. Time of flight is the manuscript's leading-edge pick,
taken through analysis/tof_axis_recovery.py so that one module owns it.
The coda level is the AUDITED estimator of tab:reconcile, taken through
analysis/ensemble_stats.py: local, band-limited to 0.8 to 3.0 MHz before
gating, referenced to the source amplitude, averaged over azimuth as
ENERGIES. Every sweep is read on the 30 azimuths at 12 degree spacing
that all of them share, so no specimen is measured differently from any
other; the two seed-11 sweeps hold 60 and are decimated to it.

THE FOUR-FOLD FRAME SMUGGLES IN NO KNOWLEDGE OF THE FABRIC. The phase of
the four-fold component of the template, measured from the template's own
axis, is 45.00 degrees at every concentration from kappa = -1000 to +400.
The frame therefore needs the sign and size of nothing. This is printed
rather than asserted.

READS, all under out/sweeps, each directory az*.npz plus config.json:
  girdle_perp_ppw8, mx_girdle_s{7,17,23,41,53,71,89}_ppw8
      eight girdle tessellations, kappa = -8, nominal axis 0 deg
  singlemax_ppw8, mx_single_s{7,17,23,41}_ppw8
      the single-maximum sweeps that have landed, kappa = +3.93,
      nominal axis 30 deg, each sharing a bit-identical Laguerre
      tessellation with the girdle sweep of the same seed. The matched
      set is discovered at run time and named in the output, so this
      module reports whatever the GPU queue has finished.
  iso_gcal, zerocontrast_ppw8, cs_f000_s11_ppw8, cs_f050_s11_ppw8
      controls on the two-fold statistic.
The realised c-axes and grain volumes come from the CPU rebuild cached
beside tof_axis_recovery.py. Nothing here touches CUDA.

WRITES stdout only.
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) +
                   r"\sim\analysis"))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) +
                   r"\sim"))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

import ensemble_stats as ES                       # noqa: E402
import specimen as SP                             # noqa: E402
import tof_axis_recovery as TAR                   # noqa: E402

# The eight girdle tessellations, and every single-maximum sweep the
# production queue has ever been asked for. Availability is checked at
# run time so that a partly finished batch is reported honestly.
GIRDLE = ("girdle_perp_ppw8", "mx_girdle_s7_ppw8", "mx_girdle_s17_ppw8",
          "mx_girdle_s23_ppw8", "mx_girdle_s41_ppw8",
          "mx_girdle_s53_ppw8", "mx_girdle_s71_ppw8",
          "mx_girdle_s89_ppw8")
SINGLE_ALL = ("singlemax_ppw8", "mx_single_s7_ppw8", "mx_single_s17_ppw8",
              "mx_single_s23_ppw8", "mx_single_s41_ppw8",
              "mx_single_s53_ppw8", "mx_single_s71_ppw8",
              "mx_single_s89_ppw8")
CONTROLS = (("iso_gcal", "isotropic, no fabric, ppw 6"),
            ("zerocontrast_ppw8", "one crystal everywhere, ppw 8"),
            ("cs_f000_s11_ppw8", "girdle bulk, no grain contrast"),
            ("cs_f050_s11_ppw8", "girdle, half grain contrast"))

# Every sweep in the matrix carries these thirty azimuths.
AZ_STEP = 12
# A sweep is usable only once it holds all of them.
AZ_REQUIRED = 360 // AZ_STEP

# The two simulated concentrations, and the azimuth grid the model
# harmonics are read on. One degree is far finer than any sweep and makes
# the model numbers independent of the acquisition.
KAPPA_GIRDLE, KAPPA_SINGLE = -8.0, 3.93
MODEL_AZ = np.arange(0.0, 360.0, 1.0)

# Concentrations the model curve is tabulated at. The negative arm runs
# to -1000 because the girdle two-fold rises with |kappa| and the claim
# below is a bound over ALL girdle concentrations.
MODEL_KAPPA = (-1000.0, -400.0, -100.0, -40.0, -20.0, -10.0, -8.0,
               -6.53, -4.0, -2.0, -1.0, 1.0, 1.8, 2.0, 3.0, 3.93, 5.0,
               8.0, 20.0, 100.0, 400.0)
# Concentrations searched for the largest two-fold any girdle can give.
GIRDLE_SEARCH = -np.geomspace(0.3, 1000.0, 120)

# Azimuth counts the classification is retried at, as in Section 5.1.
COUNTS = (30, 20, 15, 12, 10, 6)

# Grain counts the realised two-fold is measured at, and how many draws
# per count. 60 draws resolves the median to about 10 per cent, which is
# all the argument needs.
GRAIN_COUNTS = (100, 400, 1600, 6400, 25600)
GRAIN_DRAWS = 60
GRAIN_SEED = 2026


# ---------------------------------------------------------------- load

def have_sweep(name):
    """Number of azimuth records on disk, 0 if the sweep does not exist."""
    folder = ES.SWEEPS / name
    if not folder.is_dir():
        return 0
    return len([f.name for f in folder.iterdir()
                if f.name.startswith("az") and f.name.endswith(".npz")])


def available_single():
    """Single-maximum sweeps that have finished, in queue order."""
    return tuple(n for n in SINGLE_ALL if have_sweep(n) >= AZ_REQUIRED)


def load_tof_common(name):
    """Azimuth and leading-edge time of flight on the common grid."""
    az, tof = TAR.load_tof(name)
    keep = (az.astype(int) % AZ_STEP) == 0
    return az[keep], tof[keep]


def load_specimen_axis(name):
    """Volume-weighted sample axis, realised c-axes and grain volumes."""
    cfg = TAR.load_config(name)
    kappa = float(cfg["concentration"])
    nominal = np.asarray(cfg["fabric_axis"], float)
    axes, vol = TAR.load_specimen(cfg["seed"], kappa, tuple(nominal))
    svec, _ev = TAR.sample_axis(axes, vol, kappa)
    if float(np.dot(svec, nominal)) < 0.0:
        svec = -svec
    return dict(seed=int(cfg["seed"]), kappa=kappa, axes=axes, vol=vol,
                nominal=nominal, svec=svec,
                sample=TAR.azimuth_of(svec))


def load_level(name):
    """Audited coda level of one sweep, dB re source, energies."""
    _az, powers, _cfg = ES.sweep_powers(name)
    return ES.level(powers[ES.AUDITED])


# ------------------------------------------------------------- compute

def fold(delta, period):
    """Angular difference folded onto [0, period/2]."""
    d = abs(float(delta)) % period
    return min(d, period - d)


def project(az, y, alpha, order):
    """Component of y at `order`-fold, in the frame whose axis is alpha."""
    theta = np.radians(np.asarray(az, float) - alpha)
    x = np.asarray(y, float)
    x = x - x.mean()
    return 2.0 / len(theta) * float(np.sum(x * np.cos(order * theta)))


def four_fold_frame(az, tof):
    """Axis modulo 90 degrees, from the four-fold component alone.

    The reference phase is the template's own, which is 45.00 degrees at
    every concentration, so no knowledge of the fabric enters here.
    """
    phase = TAR.harmonic(az, tof, 4)[1]
    return (phase - 45.0) % 90.0


def two_fold_statistic(az, tof):
    """Magnitude of the two-fold locked to the four-fold frame, in us.

    The sign is not available: the frame is fixed only modulo 90 degrees
    and a 90 degree rotation reverses a two-fold component. The magnitude
    is, and it is what the fabric type is read from.
    """
    return abs(project(az, tof, four_fold_frame(az, tof), 2))


def model_harmonics(kappa):
    """(A2, phi2, A4, phi4) of the predicted pattern, axis at zero."""
    pattern = TAR.model_tof(MODEL_AZ, 0.0, float(kappa))
    a2, p2 = TAR.harmonic(MODEL_AZ, pattern, 2)
    a4, p4 = TAR.harmonic(MODEL_AZ, pattern, 4)
    return a2, p2, a4, p4


def girdle_ceiling():
    """Largest two-fold the model gives over the girdle branch.

    Searched over GIRDLE_SEARCH and not to the perfect-girdle limit: the
    orientation quadrature loses resolution beyond kappa = -1000, where
    the returned amplitude starts to fall again, so the bound is quoted
    over the range actually evaluated.
    """
    best = (0.0, np.nan)
    for kappa in GIRDLE_SEARCH:
        a2 = model_harmonics(kappa)[0]
        if a2 > best[0]:
            best = (a2, float(kappa))
    return best


def single_crossing(ceiling):
    """Single-maximum concentration whose two-fold clears that ceiling."""
    for kappa in np.linspace(0.2, 8.0, 79):
        if model_harmonics(kappa)[0] > ceiling:
            return float(kappa)
    return np.nan


def axis_rows(names):
    """Recovered axis and its error, per sweep, on the common grid."""
    rows = []
    for name in names:
        spec = load_specimen_axis(name)
        az, tof = load_tof_common(name)
        alpha = TAR.fit_axis_template(az, tof, spec["kappa"])[0]
        rows.append(dict(name=name, seed=spec["seed"], n=len(az),
                         kappa=spec["kappa"], sample=spec["sample"],
                         recovered=alpha,
                         err180=fold(alpha - spec["sample"], 180.0),
                         err90=fold(alpha - spec["sample"], 90.0),
                         err45=fold(alpha - spec["sample"], 45.0)))
    return rows


def table_rows(names):
    """Every column of tab:axisrecovery, on the AS-ACQUIRED azimuths.

    The published table reads the two seed-11 sweeps on all sixty of
    their azimuths, so this function does too and reproduces it exactly.
    The classification below uses the common thirty instead, and says so.
    """
    rows = []
    for name in names:
        cfg = TAR.load_config(name)
        spec = load_specimen_axis(name)
        nominal = TAR.azimuth_of(np.asarray(cfg["fabric_axis"], float))
        az, tof = TAR.load_tof(name)
        oracle = TAR.oracle_tof(spec["axes"], spec["vol"], az)
        alpha, _gain, rms = TAR.fit_axis_template(az, tof, spec["kappa"])
        tpl_orc = TAR.fit_axis_template(az, oracle, spec["kappa"])[0]
        two_orc = TAR.axis_from_two_fold(az, oracle, spec["kappa"])[0]
        two = TAR.axis_from_two_fold(az, tof, spec["kappa"])[0]
        rows.append(dict(
            name=name, seed=spec["seed"], n=len(az), nominal=nominal,
            sample=spec["sample"],
            floor=fold(spec["sample"] - nominal, 180.0),
            recovered=alpha, err=fold(alpha - spec["sample"], 180.0),
            err90=fold(alpha - spec["sample"], 90.0),
            err45=fold(alpha - spec["sample"], 45.0),
            err_two=fold(two - spec["sample"], 180.0),
            oracle_two=fold(two_orc - spec["sample"], 180.0),
            oracle_tpl=fold(tpl_orc - spec["sample"], 180.0),
            rms=rms, A2=TAR.harmonic(az, tof, 2)[0],
            floor3d=TAR.angle_between(spec["svec"], spec["nominal"])))
    return rows


def axis_count_rows(names):
    """Axis error against azimuth count, as tab:azcount defines it.

    Median and ninetieth percentile pooled over the specimens of a family
    and over every maximally spread subset of that size, on the
    AS-ACQUIRED azimuths, which is how the published table was built: the
    two seed-11 sweeps hold sixty and so contribute more subsets than the
    others. Reproduces the published girdle row exactly, which is the
    check that the single-maximum row below is a like-for-like update.
    """
    out = {}
    for name in names:
        spec = load_specimen_axis(name)
        az, tof = TAR.load_tof(name)
        row = {}
        for count in COUNTS:
            errs = []
            for idx in TAR.subsets(len(az), count):
                alpha = TAR.fit_axis_template(az[idx], tof[idx],
                                              spec["kappa"])[0]
                errs.append(fold(alpha - spec["sample"], 180.0))
            row[count] = np.array(errs)
        out[name] = row
    return out


def branch_rows(names):
    """Sign of the two-fold in the TRUE axis frame, measured and exact.

    The model puts a negative coefficient there at every concentration
    outside the interval between the two-fold null and zero, so a correct
    branch is a negative coefficient and the test needs no threshold.
    """
    rows = []
    for name in names:
        spec = load_specimen_axis(name)
        az, tof = load_tof_common(name)
        oracle = TAR.oracle_tof(spec["axes"], spec["vol"], az)
        rows.append(dict(
            name=name, seed=spec["seed"],
            measured=project(az, tof, spec["sample"], 2),
            oracle=project(az, oracle, spec["sample"], 2),
            model=-model_harmonics(spec["kappa"])[0]))
    return rows


def realised_two_fold():
    """Two-fold a finite Watson draw realises, against the ensemble one.

    Equal weights, so this is the grain count alone. The simulated
    tessellations are volume weighted and their effective count is lower,
    which is printed beside it.
    """
    rng = np.random.default_rng(GRAIN_SEED)
    az = np.arange(0.0, 360.0, 6.0)
    out = []
    for n in GRAIN_COUNTS:
        got = []
        for _ in range(GRAIN_DRAWS):
            axes = SP.sample_watson(n, (1.0, 0.0, 0.0), KAPPA_GIRDLE, rng)
            got.append(TAR.harmonic(az, TAR.oracle_tof(
                axes, np.ones(n), az), 2)[0])
        got = np.array(got)
        out.append((n, float(np.median(got)),
                    float(np.percentile(got, 5)),
                    float(np.percentile(got, 95))))
    return out


def effective_grain_counts(names):
    """1 / sum w^2 on the volume weights, per tessellation."""
    out = []
    for name in names:
        spec = load_specimen_axis(name)
        w = spec["vol"] / spec["vol"].sum()
        out.append((name, len(spec["axes"]),
                    float(1.0 / np.sum(w ** 2))))
    return out


def candidate_statistics(names):
    """Every candidate fabric-type statistic, per sweep."""
    rows = []
    for name in names:
        cfg = TAR.load_config(name)
        az, tof = load_tof_common(name)
        a2 = TAR.harmonic(az, tof, 2)[0]
        a4 = TAR.harmonic(az, tof, 4)[0]
        rms_g = TAR.fit_axis_template(az, tof, KAPPA_GIRDLE)[2]
        rms_s = TAR.fit_axis_template(az, tof, KAPPA_SINGLE)[2]
        rows.append(dict(name=name, seed=int(cfg["seed"]), n=len(az),
                         A2=a2, A4=a4, ratio=a4 / a2,
                         locked=two_fold_statistic(az, tof),
                         shape=float(np.log(rms_g / rms_s)),
                         level=load_level(name)))
    return rows


def separation(low, high):
    """Rank-sum p, best threshold, and whether the two groups overlap.

    `low` is the group expected to take the smaller values. The best
    threshold is the one maximising the count classified correctly, and
    it is fitted to these data, which is why it is reported beside the
    leave-one-out count and not instead of it.
    """
    low, high = np.asarray(low, float), np.asarray(high, float)
    _u, p = stats.mannwhitneyu(low, high, alternative="two-sided",
                               method="exact")
    best = (0, np.nan)
    for cut in np.sort(np.r_[low, high]):
        got = int((low < cut).sum() + (high >= cut).sum())
        if got > best[0]:
            best = (got, float(cut))
    return dict(p=float(p), n=len(low) + len(high), correct=best[0],
                cut=best[1], gap=float(min(high) - max(low)),
                lo=(float(low.min()), float(low.max())),
                hi=(float(high.min()), float(high.max())))


def leave_one_out(low, high):
    """Held-out classification with the threshold refitted each time."""
    low, high = list(map(float, low)), list(map(float, high))
    right = 0
    for i in range(len(low) + len(high)):
        is_low = i < len(low)
        a = low[:i] + low[i + 1:] if is_low else low
        b = high if is_low else high[:i - len(low)] + high[i - len(low)+1:]
        cut = (0.5 * (max(a) + min(b)) if max(a) < min(b)
               else 0.5 * (np.mean(a) + np.mean(b)))
        value = low[i] if is_low else high[i - len(low)]
        right += int((value < cut) == is_low)
    return right, len(low) + len(high)


def count_rows(names_low, names_high, threshold):
    """Classification against azimuth count, over every spread subset."""
    store = {}
    for name in names_low + names_high:
        az, tof = load_tof_common(name)
        row = []
        for count in COUNTS:
            values = [two_fold_statistic(az[i], tof[i])
                      for i in TAR.subsets(len(az), count)]
            row.append(np.array(values))
        store[name] = row
    out = []
    for j, count in enumerate(COUNTS):
        lows = np.concatenate([store[n][j] for n in names_low])
        highs = np.concatenate([store[n][j] for n in names_high])
        worst_low = max(store[n][j].max() for n in names_low)
        worst_high = min(store[n][j].min() for n in names_high)
        median_low = max(np.median(store[n][j]) for n in names_low)
        median_high = min(np.median(store[n][j]) for n in names_high)
        out.append(dict(count=count, worst_low=worst_low,
                        worst_high=worst_high, median_low=median_low,
                        median_high=median_high,
                        correct=float(np.mean(
                            np.r_[lows < threshold, highs >= threshold]))))
    return store, out


def level_pairs(single_names):
    """Matched girdle and single-maximum levels, one pair per seed."""
    by_seed = {}
    for name in GIRDLE:
        by_seed[int(TAR.load_config(name)["seed"])] = name
    pairs = []
    for name in single_names:
        seed = int(TAR.load_config(name)["seed"])
        if seed not in by_seed:
            continue
        pairs.append(dict(seed=seed, girdle=by_seed[seed], single=name,
                          g=load_level(by_seed[seed]),
                          s=load_level(name)))
    for row in pairs:
        row["d"] = row["s"] - row["g"]
    return pairs


def paired_test(diffs):
    """Mean, spread, t, p and interval of a set of paired differences."""
    d = np.asarray(diffs, float)
    n = len(d)
    if n < 2:
        return dict(n=n, mean=float(d.mean()) if n else np.nan,
                    sd=np.nan, t=np.nan, p=np.nan, lo=np.nan, hi=np.nan)
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / np.sqrt(n)))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return dict(n=n, mean=float(d.mean()), sd=sd, t=t,
                p=float(2 * stats.t.sf(abs(t), n - 1)),
                lo=float(d.mean()) - half, hi=float(d.mean()) + half)


def bright_share(name):
    """Share of the revolution energy carried by the loudest azimuth."""
    az, powers, _cfg = ES.sweep_powers(name)
    power = powers[ES.AUDITED]
    keep = (az.astype(int) % AZ_STEP) == 0
    az, power = az[keep], power[keep]
    j = int(np.argmax(power))
    return int(az[j]), float(power[j] / power.sum())


# ---------------------------------------------------------------- draw

def report_availability(single_names):
    print("SWEEPS AVAILABLE, ppw 8, %d azimuths at %d degrees required"
          % (AZ_REQUIRED, AZ_STEP))
    for name in GIRDLE:
        print("  girdle  %-24s %3d az" % (name, have_sweep(name)))
    for name in SINGLE_ALL:
        n = have_sweep(name)
        print("  single  %-24s %3d az%s"
              % (name, n, "" if n >= AZ_REQUIRED else "   NOT USED"))
    print("  using %d girdle and %d single-maximum tessellations: %s"
          % (len(GIRDLE), len(single_names),
             ", ".join(n.replace("_ppw8", "") for n in single_names)))
    print()


def report_axis(single_names):
    print("AXIS RECOVERY, template estimator, common %d-azimuth grid"
          % AZ_REQUIRED)
    print("  %-24s %7s %10s %8s %8s %8s"
          % ("sweep", "sample", "recovered", "mod 180", "mod 90",
             "mod 45"))
    rows = axis_rows(GIRDLE + tuple(single_names))
    for row in rows:
        print("  %-24s %7.2f %10.2f %8.2f %8.2f %8.2f"
              % (row["name"], row["sample"], row["recovered"],
                 row["err180"], row["err90"], row["err45"]))
    out = {}
    for tag, sel in (("girdle", rows[:len(GIRDLE)]),
                     ("single maximum", rows[len(GIRDLE):])):
        e180 = np.array([r["err180"] for r in sel])
        e90 = np.array([r["err90"] for r in sel])
        e45 = np.array([r["err45"] for r in sel])
        out[tag] = (e180, e90)
        print("  %-16s n = %d   mod 180 median %5.2f mean %5.2f "
              "worst %5.2f   within 10 deg %d/%d"
              % (tag, len(sel), np.median(e180), e180.mean(), e180.max(),
                 int((e180 < 10).sum()), len(sel)))
        print("  %-16s %9s mod  90 median %5.2f mean %5.2f worst %5.2f"
              "   within 10 deg %d/%d"
              % ("", "", np.median(e90), e90.mean(), e90.max(),
                 int((e90 < 10).sum()), len(sel)))
        print("  %-16s %9s mod  45 median %5.2f, identical to mod 90 in "
              "%d of %d"
              % ("", "", np.median(e45),
                 int(np.sum(np.isclose(e45, e90))), len(sel)))
    print("  The mod-45 fold returns the same number as the mod-90 fold")
    print("  for every specimen, so the girdle ambiguity is 90 degrees")
    print("  and the manuscript's modulo 45 understates the result.")
    print()
    return out


def report_table(single_names):
    """tab:axisrecovery, extended to every tessellation now available."""
    rows = table_rows(GIRDLE + tuple(single_names))
    print("TAB:AXISRECOVERY, as acquired (seed 11 holds 60 azimuths)")
    print("  %-24s %3s %6s %8s %6s %10s %7s %7s %7s %7s %7s"
          % ("sweep", "n", "nom", "sample", "floor", "recovered", "error",
             "orc 2f", "orc tpl", "mod 90", "rms us"))
    for row in rows:
        print("  %-24s %3d %6.1f %8.2f %6.2f %10.2f %7.2f %7.2f %7.2f "
              "%7.2f %7.3f"
              % (row["name"], row["n"], row["nominal"], row["sample"],
                 row["floor"], row["recovered"], row["err"],
                 row["oracle_two"], row["oracle_tpl"], row["err90"],
                 row["rms"]))
    for tag, sel in (("girdle", rows[:len(GIRDLE)]),
                     ("single maximum", rows[len(GIRDLE):])):
        err = np.array([r["err"] for r in sel])
        e90 = np.array([r["err90"] for r in sel])
        two = np.array([r["err_two"] for r in sel])
        o_t = np.array([r["oracle_tpl"] for r in sel])
        o_2 = np.array([r["oracle_two"] for r in sel])
        rms = np.array([r["rms"] for r in sel])
        a2 = np.array([r["A2"] for r in sel])
        flo = np.array([r["floor"] for r in sel])
        print("  %-16s n = %d" % (tag, len(sel)))
        print("    template  median %5.2f mean %5.2f worst %5.2f, "
              "modulo 90 median %5.2f worst %5.2f"
              % (np.median(err), err.mean(), err.max(), np.median(e90),
                 e90.max()))
        print("    two-fold  median %5.2f mean %5.2f" % (np.median(two),
                                                         two.mean()))
        print("    on the exact orientation average: template median "
              "%5.2f (%4.2f to %4.2f), two-fold median %5.2f"
              % (np.median(o_t), o_t.min(), o_t.max(), np.median(o_2)))
        print("    fit residual %.2f to %.2f us, two-fold amplitude "
              "%.2f to %.2f us" % (rms.min(), rms.max(), a2.min(),
                                   a2.max()))
        print("    sampling floor %.2f +- %.2f deg, %.2f to %.2f"
              % (flo.mean(), flo.std(ddof=1), flo.min(), flo.max()))
    flo = np.array([r["floor"] for r in rows])
    f3d = np.array([r["floor3d"] for r in rows])
    print("  sampling floor over all %d specimens: in plane "
          "%.1f +- %.1f deg, in three dimensions %.1f +- %.1f deg"
          % (len(rows), flo.mean(), flo.std(ddof=1), f3d.mean(),
             f3d.std(ddof=1)))
    print("    extremes %.1f and %.1f deg in plane, %.1f and %.1f in "
          "three dimensions"
          % (flo.min(), flo.max(), f3d.min(), f3d.max()))
    print()


def report_axis_counts(single_names):
    """tab:azcount rebuilt on whatever tessellations are available."""
    table = axis_count_rows(GIRDLE + tuple(single_names))
    print("TAB:AZCOUNT, axis error against azimuth count, degrees")
    print("  %-28s" % "row" + "".join("%8d" % c for c in COUNTS))
    for tag, sel in (("single maximum, median", single_names),
                     ("girdle, median", GIRDLE)):
        cells = ""
        tail = ""
        for count in COUNTS:
            pooled = np.concatenate([table[n][count] for n in sel])
            cells += "%8.1f" % np.median(pooled)
            tail += "%8.1f" % np.percentile(pooled, 90)
        print("  %-28s%s" % ("%s (n %d)" % (tag, len(sel)), cells))
        print("  %-28s%s" % ("  90th percentile", tail))
    print("  pooled over specimens and over every maximally spread "
          "subset of that size,")
    print("  on the as-acquired azimuths, as the published table is.")
    print()


def report_branch(single_names):
    print("THE GIRDLE BRANCH: is the two-fold big enough to choose?")
    a2g, p2g, a4g, _p4g = model_harmonics(KAPPA_GIRDLE)
    a2s, _p2s, a4s, _p4s = model_harmonics(KAPPA_SINGLE)
    print("  model at kappa %+.2f: A2 %.4f us, A4 %.4f us, A4/A2 %.2f"
          % (KAPPA_GIRDLE, a2g, a4g, a4g / a2g))
    print("  model at kappa %+.2f: A2 %.4f us, A4 %.4f us, A4/A2 %.2f"
          % (KAPPA_SINGLE, a2s, a4s, a4s / a2s))
    rows = branch_rows(GIRDLE + tuple(single_names))
    print("  %-24s %10s %10s %8s %8s"
          % ("sweep", "measured", "exact", "model", "branch"))
    for row in rows:
        print("  %-24s %+10.4f %+10.4f %+8.4f %8s"
              % (row["name"], row["measured"], row["oracle"],
                 row["model"],
                 "right" if row["measured"] < 0 else "WRONG"))
    for tag, sel in (("girdle", rows[:len(GIRDLE)]),
                     ("single maximum", rows[len(GIRDLE):])):
        meas = int(sum(1 for r in sel if r["measured"] < 0))
        orc = int(sum(1 for r in sel if r["oracle"] < 0))
        mag = np.abs([r["oracle"] for r in sel])
        print("  %-16s branch right %d/%d measured, %d/%d on the exact "
              "orientation average" % (tag, meas, len(sel), orc, len(sel)))
        print("  %-16s exact two-fold magnitude %.4f to %.4f us, "
              "median %.4f" % ("", mag.min(), mag.max(),
                               float(np.median(mag))))
    print("  The girdle's informative two-fold is %.4f us and what its"
          % a2g)
    print("  own realisation supplies is an order of magnitude larger,")
    print("  with a phase unrelated to the axis. Both counts are chance.")
    print()
    print("  REALISED TWO-FOLD AGAINST GRAIN COUNT, girdle kappa %+.1f,"
          % KAPPA_GIRDLE)
    print("  equal weights, %d draws per count" % GRAIN_DRAWS)
    print("  %10s %10s %10s %10s" % ("grains", "median", "5 pc", "95 pc"))
    for n, med, lo, hi in realised_two_fold():
        print("  %10d %10.4f %10.4f %10.4f" % (n, med, lo, hi))
    print("  the ensemble value it has to reach is %.4f us" % a2g)
    print("  effective grain count of the simulated tessellations, "
          "1/sum w^2:")
    eff = effective_grain_counts(GIRDLE + tuple(single_names))
    counts = np.array([e[2] for e in eff])
    print("    %d grains realised, effective %.1f to %.1f"
          % (eff[0][1], counts.min(), counts.max()))
    print("  No azimuth count repairs this. The realisation term belongs")
    print("  to the specimen and is common to every beam position, so it")
    print("  does not average down as azimuths are added.")
    print()


def report_type(single_names):
    print("FABRIC TYPE: which statistic separates girdle from single "
          "maximum?")
    rows = candidate_statistics(GIRDLE + tuple(single_names))
    print("  %-24s %8s %8s %8s %9s %9s %9s"
          % ("sweep", "A2", "A4", "A4/A2", "|c2|", "log LR", "level"))
    for row in rows:
        print("  %-24s %8.4f %8.4f %8.2f %9.4f %+9.4f %9.2f"
              % (row["name"], row["A2"], row["A4"], row["ratio"],
                 row["locked"], row["shape"], row["level"]))
    girdle, single = rows[:len(GIRDLE)], rows[len(GIRDLE):]
    print()
    print("  %-24s %-19s %-19s %8s %8s %7s %7s"
          % ("statistic", "girdle", "single maximum", "rank p", "cut",
             "at cut", "held"))
    order = (("A4/A2, as proposed", "ratio", False),
             ("two-fold amplitude A2", "A2", True),
             ("two-fold locked to 4f", "locked", True),
             ("template log ratio", "shape", True),
             ("coda level, audited", "level", True))
    summary = {}
    for label, key, girdle_low in order:
        g = [r[key] for r in girdle]
        s = [r[key] for r in single]
        low, high = (g, s) if girdle_low else (s, g)
        sep = separation(low, high)
        right, total = leave_one_out(low, high)
        summary[key] = sep
        print("  %-24s %8.3f to %-8.3f %8.3f to %-8.3f %8.4f %8.3f "
              "%4d/%-2d %3d/%-2d"
              % (label, min(g), max(g), min(s), max(s), sep["p"],
                 sep["cut"], sep["correct"], total, right, total))
    print("  best cut is fitted to these %d specimens and is not an "
          "out-of-sample number" % len(rows))
    print("  smallest rank-sum p attainable at n = %d and %d is %.5f"
          % (len(girdle), len(single),
             2.0 / _comb(len(rows), len(single))))
    print()
    ceiling, at_kappa = girdle_ceiling()
    crossing = single_crossing(ceiling)
    print("  MODEL CURVE, two-fold and four-fold against concentration")
    print("  %10s %10s %8s %10s %8s %8s"
          % ("kappa", "A2 (us)", "phi2", "A4 (us)", "phi4", "A4/A2"))
    for kappa in MODEL_KAPPA:
        a2, p2, a4, p4 = model_harmonics(kappa)
        print("  %10.2f %10.4f %8.2f %10.4f %8.2f %8.2f"
              % (kappa, a2, p2, a4, p4, a4 / a2))
    print("  phi4 is 45.00 at every concentration, so the four-fold "
          "frame")
    print("  is fixed without knowing the fabric.")
    print("  largest two-fold over girdle kappa %.1f to %.0f: %.4f us "
          "at kappa %.0f"
          % (GIRDLE_SEARCH.max(), GIRDLE_SEARCH.min(), ceiling,
             at_kappa))
    print("  a single maximum clears it at kappa %+.2f" % crossing)
    locked_g = [r["locked"] for r in girdle]
    locked_s = [r["locked"] for r in single]
    wrong_g = [r["name"] for r in girdle if r["locked"] >= ceiling]
    wrong_s = [r["name"] for r in single if r["locked"] < ceiling]
    print("  threshold set by that model ceiling: %d of %d correct"
          % (len(rows) - len(wrong_g) - len(wrong_s), len(rows)))
    if wrong_g:
        print("    girdles above it: %s"
              % ", ".join(n.replace("_ppw8", "") for n in wrong_g))
    if wrong_s:
        print("    single maxima below it: %s"
              % ", ".join(n.replace("_ppw8", "") for n in wrong_s))
    print("  these data are separated by any threshold in %.4f to %.4f "
          "us" % (max(locked_g), min(locked_s)))
    mean_tof = float(np.mean([load_tof_common(r["name"])[1].mean()
                              for r in rows]))
    print("  as a fraction of the %.1f us two-way time, %.3f to %.3f "
          "per cent" % (mean_tof, 100 * max(locked_g) / mean_tof,
                        100 * min(locked_s) / mean_tof))
    print()
    print("  CONTROLS on the locked two-fold")
    for name, note in CONTROLS:
        if not have_sweep(name):
            continue
        az, tof = load_tof_common(name)
        print("    %-22s n %3d  |c2| %7.4f  A4 %7.4f   %s"
              % (name, len(az), two_fold_statistic(az, tof),
                 TAR.harmonic(az, tof, 4)[0], note))
    print("    The statistic tests for a two-fold and not for a girdle.")
    print("    An isotropic specimen has none either, and is classified")
    print("    as a girdle. Telling those two apart needs the modulation")
    print("    depth, which Section 5.1 already reports as weak.")
    print()
    threshold = 0.5 * (max(locked_g) + min(locked_s))
    _store, counts = count_rows(GIRDLE, tuple(single_names), threshold)
    print("  AZIMUTH COUNT, threshold held at %.4f us" % threshold)
    print("  %8s %14s %14s %14s %14s %10s"
          % ("azimuths", "girdle worst", "single worst",
             "girdle median", "single median", "correct"))
    for row in counts:
        print("  %8d %14.4f %14.4f %14.4f %14.4f %9.1f%%"
              % (row["count"], row["worst_low"], row["worst_high"],
                 row["median_low"], row["median_high"],
                 100 * row["correct"]))
    print("  Worst columns are over every maximally spread subset of "
          "that size.")
    print("  The full sweep separates every specimen; a single subset "
          "of 20 or")
    print("  fewer can already fail, so the classification is a "
          "property of the")
    print("  %d-azimuth revolution and not of an arbitrary short arc."
          % AZ_REQUIRED)
    print()
    return summary


def report_level(single_names):
    print("CODA LEVEL ON MATCHED PAIRS, audited estimator, energies")
    pairs = level_pairs(single_names)
    print("  %-6s %-22s %10s %10s %12s"
          % ("seed", "tessellation", "girdle", "single", "difference"))
    for row in pairs:
        print("  %-6d %-22s %10.2f %10.2f %+12.2f"
              % (row["seed"], row["girdle"].replace("_ppw8", ""),
                 row["g"], row["s"], row["d"]))
    diffs = [row["d"] for row in pairs]
    full = paired_test(diffs)
    print("  n = %d pairs, single maximum louder by %+.2f dB (sd %.2f)"
          % (full["n"], full["mean"], full["sd"]))
    print("  paired t = %+.2f on %d dof, p = %.4f, 95%% interval "
          "%+.2f to %+.2f dB"
          % (full["t"], full["n"] - 1, full["p"], full["lo"], full["hi"]))
    positive = int(sum(1 for d in diffs if d > 0))
    print("  sign test %d of %d positive, exact p = %.4f"
          % (positive, len(diffs),
             stats.binomtest(positive, len(diffs), 0.5).pvalue))
    for drop in (17, 7):
        kept = [row["d"] for row in pairs if row["seed"] != drop]
        if len(kept) == len(pairs):
            continue
        cut = paired_test(kept)
        print("  DIAGNOSTIC, seed %d removed: n = %d, %+.2f dB, "
              "t = %+.2f, p = %.4f"
              % (drop, cut["n"], cut["mean"], cut["t"], cut["p"]))
    kept = [row["d"] for row in pairs if row["seed"] not in (7, 17)]
    if len(kept) >= 2:
        cut = paired_test(kept)
        print("  DIAGNOSTIC, seeds 7 and 17 both removed: n = %d, "
              "%+.2f dB, t = %+.2f, p = %.4f"
              % (cut["n"], cut["mean"], cut["t"], cut["p"]))
    print("  Removing a pair is a diagnostic and not the headline. The")
    print("  headline is the %d-pair line above." % full["n"])
    print()
    g = np.array([load_level(n) for n in GIRDLE])
    s = np.array([load_level(n) for n in single_names])
    print("  families, unpaired: girdle %.2f +- %.2f dB (n %d, span "
          "%.2f)" % (g.mean(), g.std(ddof=1), len(g), g.max() - g.min()))
    print("  %20s single %.2f +- %.2f dB (n %d, span %.2f)"
          % ("", s.mean(), s.std(ddof=1), len(s), s.max() - s.min()))
    print("  A single realisation of the single-maximum family spans")
    print("  %.2f dB, against a %.2f dB fabric difference, so the level"
          % (s.max() - s.min(), s.mean() - g.mean()))
    print("  is not a usable discriminant on one specimen.")
    print()
    print("  WHERE THE DIFFERENCE COMES FROM: brightest azimuth share")
    print("  %-24s %10s %10s %10s" % ("sweep", "level", "azimuth",
                                      "share"))
    for name in GIRDLE + tuple(single_names):
        az, share = bright_share(name)
        print("  %-24s %10.2f %10d %9.1f%%"
              % (name, load_level(name), az, 100 * share))
    print("  Two of the %d single-maximum realisations put nearly half"
          % len(single_names))
    print("  the revolution energy in one azimuth of thirty. No girdle")
    print("  does. The fabric difference in the mean level is those two")
    print("  specular azimuths, not a bulk property of the fabric.")
    print()


def _comb(n, k):
    """Binomial coefficient, without depending on numpy internals."""
    out = 1
    for i in range(k):
        out = out * (n - i) // (i + 1)
    return out


def main():
    single_names = available_single()
    report_availability(single_names)
    report_table(single_names)
    report_axis(single_names)
    report_axis_counts(single_names)
    report_branch(single_names)
    report_type(single_names)
    report_level(single_names)


if __name__ == "__main__":
    main()
