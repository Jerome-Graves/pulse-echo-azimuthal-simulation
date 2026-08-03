"""Fabric TYPE from one rotational acquisition: single maximum or girdle.

Written to test a specific proposal for Section 5.1, and it does not
support it. The proposal was that the ratio of the four-fold to the
two-fold amplitude of the azimuthal time-of-flight pattern separates the
two fabrics, because the ENSEMBLE velocity surface makes that ratio
0.80 at kappa = +3.93 and 20.6 at kappa = -8, a factor of 26 in one
dimensionless number costing nothing beyond the harmonic regression that
Section 5.1 already runs.

THE DECISION. Do not present the ratio as a fabric-type classifier. Its
leave-one-out accuracy on the thirteen labelled sweeps is 8 of 13, which
is exactly the accuracy of ignoring the data and calling every specimen a
girdle, since 8 of the 13 are girdles. Its resubstitution accuracy is
11 of 13, and the difference between those two numbers is the whole
reason this module exists.

WHY IT FAILS, and the failure is not measurement. The ensemble ratio of
20.6 is large only because the ensemble two-fold amplitude at kappa = -8
is nearly null, 0.0119 us, the simulated girdle sitting close to the
two-fold null at kappa = -6.53 that Section 5.1 already identifies. A
tessellation of about 100 grains does not realise that null. Applying the
harmonic regression to the EXACT volume-weighted orientation average of
the realised c-axes, which carries no measurement noise whatever, the
eight girdles return two-fold amplitudes of 0.045 to 0.168 us, four to
fourteen times the ensemble value, and the noise-free ratio falls from
20.6 to a range of 1.45 to 6.04 against 0.62 to 1.50 for the single
maxima. The two ranges touch before a single wave has been simulated.
The discriminant divides by a quantity that is sampling scatter rather
than physics, and no improvement in the instrument recovers it.

WHAT DOES SURVIVE, and it is a weaker claim than a classifier. In the
MATCHED pairs, where a single-maximum sweep and a girdle sweep share a
bit-identical Laguerre tessellation so that the specimen is held fixed,
two statistics order all five available pairs correctly: the template
model-selection statistic and the two-fold depth, each at a sign-test
p = 0.031. The harmonic ratio orders 4 of 5, p = 0.19. The
between-tessellation scatter, not the measurement, is what a fixed
threshold cannot cross. The best time-of-flight statistic on the
unpaired test, the model-selection ratio between the two templates,
reaches 10 of 13 leave-one-out, two above the majority-class baseline
and not significant against it (p = 0.20 before any allowance for the
six statistics tried).

THE CODA IS NOT THE ANSWER EITHER, though on the unpaired test it looks
like one. The audited coda level of Section 4.6 reaches 12 of 13
leave-one-out with an AUC of 0.950. Three things stop that being a
capability. It is not significant in the matched pairs, which is the
comparison the design supports: 4 of 5 correct, mean +3.53 dB with a
standard deviation of 3.77, t = 2.09, p = 0.10, and the two pairs that
carry it (seeds 7 and 17) contribute +7.8 and +7.4 dB against +1.5,
+1.3 and -0.3 for the rest. The mechanism check passes only in sign:
theory has the level tracking the grain-to-grain velocity variance one
for one, and sigma_d^2 differs between the pairs by +2.18 dB, so those
two pairs overshoot the predicted difference by five and six decibels.
And a 3.5 dB separation is smaller than the drift of the SAME estimator
on the SAME specimen across the grid resolutions this paper uses:
-81.09, -85.11 and -87.79 dB at ppw 6, 8 and 10. An absolute decibel
level is not a transferable discriminant. Every time-of-flight statistic
here is a shape statistic and needs no such calibration.

CONTROLS. The observable is coherent and it does survive f = 0, which is
the one positive result here and does not depend on any classifier
working: across the whole contrast ladder from f = 0 to f = 1 the
two-fold amplitude moves only from 0.112 to 0.137 us and the four-fold
from 0.178 to 0.200 us, so whatever fabric-type information time of
flight carries is carried with no grain-to-grain contrast at all, by the
bulk anisotropy alone. Fabric TYPE, on this evidence, would be
recoverable without any scattering.

Three controls fail, and they do not fail for the same statistic, which
is the reason nothing here should be presented as a capability. The
ladder is five rungs of ONE tessellation under ONE fabric, so a
fabric-type statistic must not move along it: the harmonic ratio passes
it, calling all five rungs girdle, but the model-selection statistic,
which is the best of them on the unpaired test, moves by 0.42 along the
ladder against a median leave-one-out margin of 0.56 and calls four of
the five rungs a single maximum. The isotropic sweep iso_gcal, which
has NO fabric type at all, is called a girdle by both, because an
isotropic specimen has no two-fold term either. And the single-crystal
sweep zerocontrast_ppw8, which is the extreme single maximum, is called
a girdle by the harmonic ratio, though the model selection gets it
right. Between them the two statistics fail every control; neither
fails all of them, and no combination was fitted, because with thirteen
sweeps a fitted combination would only be the resubstitution mistake in
a smarter form.

READS, all under out/sweeps, on the 30 azimuths at 12 degree spacing that
every sweep shares:
  girdle, kappa = -8, nominal axis 0 deg
      girdle_perp_ppw8 and mx_girdle_s{7,17,23,41,53,71,89}_ppw8
  single maximum, kappa = +3.93, nominal axis 30 deg
      singlemax_ppw8 and mx_single_s{7,17,23,41}_ppw8
  controls
      cs_f000..cs_f075_s11_ppw8 contrast ladder, zerocontrast_ppw8
      single crystal, iso_gcal isotropic at ppw 6
Any sweep of the candidate lists below that does not yet hold all 30
common azimuths is skipped and named in the header, so this module is
safe to re-run as the batch completes. The numbers quoted above and in
the manuscript are the thirteen sweeps complete on 2 August 2026: eight
girdles and five single maxima. mx_single_s53_ppw8 was then part run,
and s71 and s89 had not started. Every count is out of thirteen and
every matched-pair test out of five, so re-running with the full eight
pairs will change them.

WRITES stdout only. It touches no GPU: the arrival time comes from
analysis/tof_axis_recovery.py, which forces the CPU labeller at import,
and the oracle uses that module's cached tessellation builds.

THE ARRIVAL TIME IS NOT RE-IMPLEMENTED HERE. Section 5.1's pick is the
leading edge of the backwall arrival, the first crossing of 25 per cent
of the envelope peak with sub-sample interpolation, inside a +-2 us
window about the isotropic arrival 2D/c. This module calls
tof_axis_recovery.load_tof for it, so the two cannot be different
observables wearing one name. check_pick below additionally re-derives
the pick from the raw traces along an independent code path and prints
the largest disagreement over every sweep, which is a guard against a
future edit to either file rather than a measurement.
"""
import math
import os
import sys

import numpy as np
from scipy import stats as SPS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db_reconcile as DB                        # noqa: E402
import tof_axis_recovery as TOF                  # noqa: E402

SWD = TOF.SWD

# The 30 azimuths at 12 degree spacing that every ppw 8 sweep holds. The
# two 60-azimuth sweeps are decimated onto it so that no statistic is
# compared across different azimuth counts.
AZ_COMMON = tuple(range(0, 360, 12))

# Candidate sweeps. Membership is filtered by completeness at load time.
GIRDLE = ("girdle_perp_ppw8", "mx_girdle_s7_ppw8", "mx_girdle_s17_ppw8",
          "mx_girdle_s23_ppw8", "mx_girdle_s41_ppw8",
          "mx_girdle_s53_ppw8", "mx_girdle_s71_ppw8",
          "mx_girdle_s89_ppw8")
SINGLE = ("singlemax_ppw8", "mx_single_s7_ppw8", "mx_single_s17_ppw8",
          "mx_single_s23_ppw8", "mx_single_s41_ppw8",
          "mx_single_s53_ppw8", "mx_single_s71_ppw8",
          "mx_single_s89_ppw8")

# The contrast ladder shares one tessellation, seed 11 girdle. f = 1 is
# the production sweep. f = 0 replaces every grain tensor by the ARITHMETIC
# MEAN of the grain tensors, so the medium keeps the fabric's bulk
# stiffness and loses every grain-to-grain contrast. zerocontrast_ppw8 is
# a different treatment: it copies the FIRST grain's tensor everywhere, so
# that specimen is a single crystal and its fabric is the extreme single
# maximum, not a girdle.
LADDER = (("cs_f000_s11_ppw8", 0.00), ("cs_f025_s11_ppw8", 0.25),
          ("cs_f050_s11_ppw8", 0.50), ("cs_f075_s11_ppw8", 0.75),
          ("girdle_perp_ppw8", 1.00))
UNLABELLED = (("zerocontrast_ppw8", "single crystal, s11 grain 1"),
              ("iso_gcal", "isotropic, no type, ppw 6"))

# The two hypotheses the classifier chooses between. These are the
# simulated concentrations, not fitted values.
K_GIRDLE, K_SINGLE = -8.0, 3.93

# Azimuth counts for the operating-limit table, matching Section 5.1's
# tab:azcount. On N equally spaced azimuths the harmonic k is
# indistinguishable from N - k, so the four-fold term is separable from
# the two-fold only for N > 6, and is degenerate at its own Nyquist
# N = 8. Every statistic here therefore has a hard floor at 10 azimuths.
COUNTS = (30, 20, 15, 12, 10, 6)

# Section 5.1's leading-edge fractions, for the sensitivity table.
PICK_FRACTIONS = (0.15, 0.25, 0.35)

# Larger means MORE GIRDLE for every statistic, so one threshold rule
# serves all of them.
STATS = (("ratio", "log10 A4/A2, the proposal"),
         ("select", "log SSR(single)/SSR(girdle), template selection"),
         ("depth", "-A2, two-fold depth alone"),
         ("extremes", "-(max to min azimuth separation), deg"),
         ("noncos", "fraction of variance not in a single two-fold"),
         ("coda", "-(audited coda level), dB"))


# ---------------------------------------------------------------- load

def available(names):
    """Those candidate sweeps holding every azimuth of the common grid."""
    keep, miss = [], []
    for n in names:
        root = os.path.join(SWD, n)
        have = all(os.path.exists(os.path.join(root, "az%03d.npz" % a))
                   for a in AZ_COMMON)
        (keep if have else miss).append(n)
    return keep, miss


def load_series(name, level=TOF.TOF_FRAC):
    """Azimuth (deg) and Section 5.1's leading-edge arrival (us), on the
    common grid. The pick itself is tof_axis_recovery's, called and not
    copied."""
    az, tof = TOF.load_tof(name, level=level)
    keep = np.isin(az.astype(int), np.asarray(AZ_COMMON))
    return az[keep], tof[keep]


def load_oracle(name):
    """The exact volume-weighted orientation average of the realised
    c-axes, at the same azimuths. No measurement noise and no fitted
    parameter, so it isolates what the tessellation alone can do."""
    cfg = TOF.load_config(name)
    az, _ = load_series(name)
    axes, vol = TOF.load_specimen(cfg["seed"], float(cfg["concentration"]),
                                  tuple(cfg["fabric_axis"]))
    return az, TOF.oracle_tof(axes, vol, az)


def load_coda(name):
    """Mean coda level in dB on the audited estimator of Section 4.6:
    local in-band gate power referenced to the source amplitude, averaged
    over azimuth as an energy."""
    vals = [DB.measure_trace(os.path.join(SWD, name, "az%03d.npz" % a))
            ["coda_band"] for a in AZ_COMMON]
    return 10.0 * np.log10(float(np.mean(vals)))


def check_pick():
    """Re-derive the arrival independently and compare with Section 5.1's.

    This exists so that a later edit to either file cannot silently make
    the classifier and the axis recovery two different observables. It
    reports the largest disagreement in microseconds, which is 0 while
    the two agree exactly.
    """
    from scipy.signal import hilbert
    worst, tested = 0.0, 0
    complete = available(GIRDLE)[0] + available(SINGLE)[0]
    for name in sorted(complete):
        root = os.path.join(SWD, name)
        az, ref = load_series(name)
        mine = []
        for a in az:
            with np.load(os.path.join(root, "az%03d.npz" % int(a))) as z:
                trace = np.asarray(z["trace"], float).ravel()
                dt = float(z["dt"])
            fs = 1.0 / dt
            env = np.abs(hilbert(trace))
            k0 = int(2.0 * TOF.DIA / TOF.C_REF * fs)
            half = int(TOF.TOF_HALF_W * fs)
            lo = max(k0 - half, 0)
            seg = env[lo:k0 + half]
            ipk = int(np.argmax(seg))
            thr = TOF.TOF_FRAC * seg[ipk]
            above = np.nonzero(seg[:ipk + 1] >= thr)[0]
            j = int(above[0]) if above.size else ipk
            sub = 0.0
            if j > 0 and seg[j] > seg[j - 1]:
                sub = (thr - seg[j - 1]) / (seg[j] - seg[j - 1])
            mine.append((lo + j - 1 + sub) / fs * 1e6)
        worst = max(worst, float(np.max(np.abs(np.asarray(mine) - ref))))
        tested += 1
    return tested, worst


# ----------------------------------------------------------- statistics

def amplitudes(az, tof):
    """Two-fold and four-fold amplitude of the pattern, in us.

    tof_axis_recovery.harmonic, so the regression is Section 5.1's. The
    amplitudes are rotation invariant, which is the one attraction the
    proposal genuinely has: no axis has to be recovered first.
    """
    return TOF.harmonic(az, tof, 2)[0], TOF.harmonic(az, tof, 4)[0]


def template_ssr(az, tof, kappa):
    """Best residual sum of squares of the Section 5.1 template at this
    concentration, over a free axis, a free offset and a free
    non-negative gain. Only the SHAPE of the pattern enters."""
    yc = np.asarray(tof, float) - np.mean(tof)
    best = np.inf
    for alpha in TOF.ALPHAS:
        m = TOF.model_tof(az, alpha, kappa)
        mc = m - m.mean()
        den = float(mc @ mc)
        gain = max(float(mc @ yc) / den, 0.0) if den > 0.0 else 0.0
        best = min(best, float(np.sum((yc - gain * mc) ** 2)))
    return best


def extreme_separation(az, tof):
    """Folded azimuthal separation of the fastest and slowest directions.

    A pure two-fold surface puts them 90 degrees apart and a four-fold
    one 45. Read off a mean-plus-two-fold-plus-four-fold reconstruction
    on a quarter-degree grid rather than off the samples, so the answer
    is not quantised by the azimuth spacing.
    """
    th = np.radians(np.asarray(az, float))
    design = np.stack([np.ones_like(th), np.cos(2 * th), np.sin(2 * th),
                       np.cos(4 * th), np.sin(4 * th)], axis=1)
    c = np.linalg.lstsq(design, np.asarray(tof, float), rcond=None)[0]
    fine = np.arange(0.0, 360.0, 0.25)
    ft = np.radians(fine)
    rec = (c[0] + c[1] * np.cos(2 * ft) + c[2] * np.sin(2 * ft)
           + c[3] * np.cos(4 * ft) + c[4] * np.sin(4 * ft))
    d = abs(fine[int(np.argmax(rec))] - fine[int(np.argmin(rec))]) % 180.0
    return min(d, 180.0 - d)


def statistics(az, tof, coda=None):
    """Every discriminant tried, all signed so that larger means girdle."""
    a2, a4 = amplitudes(az, tof)
    out = dict(a2=a2, a4=a4, r=a4 / a2,
               sep=extreme_separation(az, tof),
               ratio=math.log10(a4 / a2),
               depth=-a2,
               extremes=-extreme_separation(az, tof),
               noncos=1.0 - 0.5 * a2 * a2 / float(np.var(tof)),
               select=math.log(template_ssr(az, tof, K_SINGLE)
                               / template_ssr(az, tof, K_GIRDLE)))
    out["coda_db"] = coda if coda is not None else float("nan")
    out["coda"] = -out["coda_db"]
    return out


# ----------------------------------------------------------- classifier

def fit_threshold(values, labels):
    """Threshold minimising training errors, ties broken by margin.

    Predicts girdle above the threshold. Candidates are the midpoints
    between adjacent sorted values plus the two degenerate all-one-class
    thresholds, so a statistic that carries nothing is allowed to say so
    by collapsing onto the majority class.
    """
    v = np.asarray(values, float)
    order = np.sort(v)
    cands = list((order[:-1] + order[1:]) / 2.0)
    cands += [order[0] - 1.0, order[-1] + 1.0]
    best = None
    for t in cands:
        pred = np.where(v > t, "G", "S")
        key = (int((pred != labels).sum()),
               -float(np.abs(v - t).min()))
        if best is None or key < best[0]:
            best = (key, float(t))
    return best[1]


def resubstitution(values, labels):
    """Accuracy of the threshold fitted on the same data. Reported only
    so that the gap to the leave-one-out figure is visible."""
    t = fit_threshold(values, labels)
    return int((np.where(np.asarray(values) > t, "G", "S")
                == labels).sum()), t


def leave_one_out(values, labels):
    """Refit the threshold on the others, classify the held-out sweep.

    Returns the per-sweep verdict and the signed margin of the held-out
    value from the threshold it was judged by, in the units of the
    statistic. A count without a margin cannot distinguish a decisive
    separation from a coincidental one.
    """
    v, lab = np.asarray(values, float), np.asarray(labels)
    ok, margin = [], []
    for i in range(len(v)):
        keep = np.arange(len(v)) != i
        t = fit_threshold(v[keep], lab[keep])
        ok.append(("G" if v[i] > t else "S") == lab[i])
        margin.append(v[i] - t)
    return np.array(ok), np.array(margin)


def auc(values, labels):
    """Mann-Whitney statistic: the chance a random girdle outranks a
    random single maximum. Threshold free, so it separates the quality of
    the ordering from the quality of the cut."""
    v = np.asarray(values, float)
    g, s = v[np.asarray(labels) == "G"], v[np.asarray(labels) == "S"]
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0
               for a in g for b in s)
    return wins / (len(g) * len(s))


def binom_tail(k, n, p):
    """P(X >= k) for X ~ Bin(n, p). Exact, no approximation."""
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                     for i in range(k, n + 1)))


def majority_baseline(labels):
    """Accuracy of the classifier that reads nothing and answers with the
    commoner class. Any count at or below this is not a measurement."""
    lab = np.asarray(labels)
    return max(int((lab == "G").sum()), int((lab == "S").sum())), len(lab)


# -------------------------------------------------------------- compute

def compute_sweeps():
    """Every statistic for every labelled sweep, plus its oracle twin."""
    gird, miss_g = available(GIRDLE)
    sing, miss_s = available(SINGLE)
    rows = []
    for name, lab in ([(n, "G") for n in gird]
                      + [(n, "S") for n in sing]):
        cfg = TOF.load_config(name)
        az, tof = load_series(name)
        st = statistics(az, tof, coda=load_coda(name))
        _, orc = load_oracle(name)
        o2, o4 = amplitudes(az, orc)
        st.update(name=name, label=lab, seed=int(cfg["seed"]),
                  kappa=float(cfg["concentration"]), n=len(az),
                  o2=o2, o4=o4, o_r=o4 / o2,
                  o_sep=extreme_separation(az, orc))
        rows.append(st)
    return rows, miss_g + miss_s


def compute_ensemble():
    """The ensemble amplitudes the proposal rests on, from the same
    template Section 5.1 fits, so the 26 is recomputed and not quoted."""
    az = np.arange(0.0, 360.0, 1.0)
    out = {}
    for kappa in (K_SINGLE, K_GIRDLE):
        m = TOF.model_tof(az, 0.0, kappa)
        a2, a4 = TOF.harmonic(az, m, 2)[0], TOF.harmonic(az, m, 4)[0]
        out[kappa] = (a2, a4, a4 / a2)
    return out


def compute_scores(rows):
    """Resubstitution, leave-one-out, margins and AUC per statistic."""
    labels = np.array([r["label"] for r in rows])
    base, n = majority_baseline(labels)
    out = []
    for key, blurb in STATS:
        v = np.array([r[key] for r in rows])
        if not np.all(np.isfinite(v)):
            continue
        res, thr = resubstitution(v, labels)
        ok, margin = leave_one_out(v, labels)
        out.append(dict(key=key, blurb=blurb, resub=res, thr=thr,
                        loo=int(ok.sum()), n=n, base=base,
                        auc=auc(v, labels),
                        p_vs_base=binom_tail(int(ok.sum()), n,
                                             base / float(n)),
                        min_margin=float(np.abs(margin).min()),
                        med_margin=float(np.median(np.abs(margin))),
                        wrong=[rows[i]["name"] for i in range(n)
                               if not ok[i]]))
    return out


def compute_pairs(rows):
    """Matched pairs: the same Laguerre tessellation under both fabrics.

    This is the only comparison in which the specimen is held fixed, so
    it separates a discriminant that works from one defeated by
    tessellation-to-tessellation scatter.
    """
    by = {}
    for r in rows:
        by.setdefault(r["seed"], {})[r["label"]] = r
    pairs = [(s, by[s]["G"], by[s]["S"]) for s in sorted(by)
             if "G" in by[s] and "S" in by[s]]
    out = []
    for key, blurb in STATS:
        wins = [int(g[key] > s[key]) for _, g, s in pairs]
        out.append(dict(key=key, blurb=blurb, wins=sum(wins),
                        n=len(pairs), signs=wins,
                        p=binom_tail(sum(wins), len(pairs), 0.5)))
    orc = [int(g["o_r"] > s["o_r"]) for _, g, s in pairs]
    return pairs, out, dict(wins=sum(orc), n=len(pairs),
                            p=binom_tail(sum(orc), len(pairs), 0.5))


def compute_controls(rows):
    """The classifier's verdict on the ladder and on the unlabelled
    sweeps, under thresholds fitted on all thirteen labelled sweeps."""
    labels = np.array([r["label"] for r in rows])
    thr = {k: fit_threshold(np.array([r[k] for r in rows]), labels)
           for k, _ in STATS if np.all(np.isfinite(
               [r[k] for r in rows]))}
    out = []
    for name, tag in ([(n, "ladder f = %.2f" % f) for n, f in LADDER]
                      + list(UNLABELLED)):
        az, tof = load_series(name)
        st = statistics(az, tof, coda=load_coda(name))
        st.update(name=name, tag=tag,
                  verdict={k: ("girdle" if st[k] > t else "single")
                           for k, t in thr.items()})
        out.append(st)
    return thr, out


def compute_counts(rows):
    """Accuracy against azimuth count, over every maximally spread
    subset of the common grid."""
    labels = np.array([r["label"] for r in rows])
    series = {r["name"]: load_series(r["name"]) for r in rows}
    table = {}
    for key, _ in STATS:
        if key == "coda":
            continue
        row = {}
        for c in COUNTS:
            accs, aucs = [], []
            for idx in TOF.subsets(len(AZ_COMMON), c):
                v = []
                for r in rows:
                    az, tof = series[r["name"]]
                    v.append(statistics(az[idx], tof[idx])[key])
                v = np.array(v)
                if not np.all(np.isfinite(v)):
                    continue
                accs.append(leave_one_out(v, labels)[0].mean())
                aucs.append(auc(v, labels))
            row[c] = (float(np.mean(accs)), float(np.median(aucs)),
                      len(accs))
        table[key] = row
    return table


def grain_velocity_variance(name, az):
    """sigma_d^2, the volume-weighted fractional variance of the
    quasi-longitudinal speed across grains for each beam direction.

    Section 4 has the coda level tracking this one for one in decibels,
    so it is the quantity the coda discriminant is supposed to be
    reading. Computed from the same cached tessellation the oracle uses.
    """
    cfg = TOF.load_config(name)
    axes, vol = TOF.load_specimen(cfg["seed"], float(cfg["concentration"]),
                                  tuple(cfg["fabric_axis"]))
    w = vol / vol.sum()
    out = []
    for n in np.atleast_2d(TOF.beam_direction(az)):
        v = TOF.qp_velocity(axes @ n)
        vbar = float(w @ v)
        out.append(float(w @ (v - vbar) ** 2) / vbar ** 2)
    return 10.0 * np.log10(float(np.mean(out)))


def compute_coda_evidence(rows, pairs):
    """Everything needed to judge the coda discriminant honestly.

    The unpaired count flatters it, so three further things are measured:
    the paired difference and its t test, the sigma_d^2 the theory says
    the level should be tracking, and the drift of the same estimator on
    one specimen across the three grid resolutions the paper uses. If
    the class separation is smaller than that drift, the discriminant is
    not transferable however well it scores here.
    """
    az = np.asarray(AZ_COMMON, float)
    sig = {r["name"]: grain_velocity_variance(r["name"], az) for r in rows}
    diffs = np.array([s["coda_db"] - g["coda_db"] for _, g, s in pairs])
    sdiff = np.array([sig[s["name"]] - sig[g["name"]]
                      for _, g, s in pairs])
    t = float(diffs.mean() / (diffs.std(ddof=1) / math.sqrt(len(diffs))))
    labels = np.array([r["label"] for r in rows])
    hot = [r["name"] for r in sorted(
        [r for r in rows if r["label"] == "S"],
        key=lambda r: -r["coda_db"])[:2]]
    keep = [i for i, r in enumerate(rows) if r["name"] not in hot]
    v = np.array([rows[i]["coda"] for i in keep])
    lab_k = labels[keep]
    ok_k, marg_k = leave_one_out(v, lab_k)
    drift = {p: DB.level(DB.load_sweep(n), "coda_band")
             for p, n in sorted(DB.SPECIMEN.items())}
    return dict(sig=sig, diffs=diffs, sdiff=sdiff, t=t,
                p=float(2.0 * SPS.t.sf(abs(t), len(diffs) - 1)),
                seeds=[s for s, _, _ in pairs],
                jack=[float(np.delete(diffs, i).mean())
                      for i in range(len(diffs))],
                hot=hot, loo_drop=int(ok_k.sum()), n_drop=len(keep),
                base_drop=majority_baseline(lab_k)[0],
                gap_drop=float(np.abs(marg_k).min()),
                auc_drop=auc(v, lab_k), drift=drift)


def compute_pick_sensitivity(rows):
    """Does the verdict depend on where on the leading edge the pick is
    taken? Section 5.1 asks the same question of the axis."""
    labels = np.array([r["label"] for r in rows])
    out = []
    for frac in PICK_FRACTIONS:
        vals = {}
        for r in rows:
            az, tof = load_series(r["name"], level=frac)
            st = statistics(az, tof)
            for key, _ in STATS:
                if key != "coda":
                    vals.setdefault(key, []).append(st[key])
        row = dict(frac=frac, n=len(labels))
        for key, v in vals.items():
            v = np.array(v)
            row[key] = (int(leave_one_out(v, labels)[0].sum()),
                        auc(v, labels))
        out.append(row)
    return out


# ----------------------------------------------------------------- draw

def draw_header(rows, missing, tested, worst):
    print("FABRIC TYPE FROM ONE ROTATIONAL ACQUISITION")
    g = [r["name"] for r in rows if r["label"] == "G"]
    s = [r["name"] for r in rows if r["label"] == "S"]
    print("  girdle sweeps used  (%d): %s" % (len(g), ", ".join(g)))
    print("  single-max sweeps   (%d): %s" % (len(s), ", ".join(s)))
    if missing:
        print("  SKIPPED, incomplete on the common grid: %s"
              % ", ".join(missing))
    print("  all statistics on the %d shared azimuths at %d degree "
          "spacing" % (len(AZ_COMMON), 360 // len(AZ_COMMON)))
    print("  arrival time: Section 5.1's leading edge at %.0f per cent "
          "of the" % (100 * TOF.TOF_FRAC))
    print("  envelope peak, taken from tof_axis_recovery.load_tof "
          "itself.")
    print("  independent re-derivation over %d sweeps disagrees by at "
          "most %.3e us" % (tested, worst))
    print()


def draw_ensemble(ens):
    print("WHAT THE ENSEMBLE PREDICTS, from Section 5.1's own template")
    print("  %-24s %9s %9s %9s" % ("fabric", "A2 us", "A4 us", "A4/A2"))
    for kappa, tag in ((K_SINGLE, "single maximum"), (K_GIRDLE, "girdle")):
        a2, a4, r = ens[kappa]
        print("  %-16s k %+5.2f %9.5f %9.5f %9.3f"
              % (tag, kappa, a2, a4, r))
    print("  predicted separation in the ratio: a factor of %.1f"
          % (ens[K_GIRDLE][2] / ens[K_SINGLE][2]))
    print()


def draw_sweeps(rows):
    print("PER SWEEP. Oracle columns are the exact orientation average of")
    print("the realised c-axes at the same azimuths, with no measurement.")
    print("  %-22s %2s %7s %7s %7s %7s %7s %7s %7s %8s"
          % ("sweep", "l", "A2", "A4", "A4/A2", "orc A2", "orc R", "sep",
             "noncos", "coda dB"))
    for r in rows:
        print("  %-22s %2s %7.4f %7.4f %7.3f %7.4f %7.3f %7.2f %7.3f "
              "%8.2f" % (r["name"], r["label"], r["a2"], r["a4"], r["r"],
                         r["o2"], r["o_r"], r["sep"], r["noncos"],
                         r["coda_db"]))
    for lab, tag in (("G", "girdle"), ("S", "single max")):
        sel = [r for r in rows if r["label"] == lab]
        v = np.array([r["r"] for r in sel])
        o = np.array([r["o_r"] for r in sel])
        print("  %-12s measured A4/A2 %.3f to %.3f (median %.3f); "
              "oracle %.3f to %.3f"
              % (tag, v.min(), v.max(), np.median(v), o.min(), o.max()))
    gr = [r["r"] for r in rows if r["label"] == "G"]
    sr = [r["r"] for r in rows if r["label"] == "S"]
    print("  the two measured ranges OVERLAP between %.3f and %.3f"
          % (max(min(gr), min(sr)), min(max(gr), max(sr))))
    go = [r["o_r"] for r in rows if r["label"] == "G"]
    so = [r["o_r"] for r in rows if r["label"] == "S"]
    print("  and the two noise-free oracle ranges overlap between %.3f "
          "and %.3f" % (max(min(go), min(so)), min(max(go), max(so))))
    o2g = [r["o2"] for r in rows if r["label"] == "G"]
    print("  realised girdle two-fold amplitude %.4f to %.4f us against "
          "an" % (min(o2g), max(o2g)))
    print("  ensemble %.4f us, so the denominator of the ratio is "
          "sampling scatter" % TOF.harmonic(
              np.arange(0.0, 360.0, 1.0),
              TOF.model_tof(np.arange(0.0, 360.0, 1.0), 0.0, K_GIRDLE),
              2)[0])
    print()


def draw_scores(scores):
    print("VALIDATION. Leave one out refits the threshold on the other")
    print("sweeps and classifies the held-out one. Resubstitution is")
    print("printed only so the gap is visible; it is not a result.")
    b, n = scores[0]["base"], scores[0]["n"]
    print("  majority-class baseline: %d/%d = %.3f, the accuracy of "
          "calling" % (b, n, b / float(n)))
    print("  every specimen a girdle without reading the data")
    print("  %-10s %7s %7s %7s %8s %9s %9s"
          % ("statistic", "resub", "LOO", "AUC", "p vs base", "min marg",
             "med marg"))
    for s in scores:
        print("  %-10s %5d/%2d %5d/%2d %7.3f %8.3f %9.4f %9.4f"
              % (s["key"], s["resub"], s["n"], s["loo"], s["n"],
                 s["auc"], s["p_vs_base"], s["min_margin"],
                 s["med_margin"]))
    print()
    for s in scores:
        print("  %-10s %s" % (s["key"], s["blurb"]))
        if s["wrong"]:
            print("  %-10s misclassified: %s"
                  % ("", ", ".join(s["wrong"])))
    print()
    print("  Margins are in the units of the statistic and are not")
    print("  comparable between rows. Within a row, compare the median")
    print("  margin with the minimum: a run of correct calls resting on")
    print("  one hair-thin margin is not the same result as a run with")
    print("  room to spare.")
    print("  Six statistics were tried on one data set, so the smallest")
    print("  p above should be multiplied by six before it is believed:")
    best = min(s["p_vs_base"] for s in scores)
    print("  %.3f becomes %.3f." % (best, min(1.0, 6.0 * best)))
    print()


def draw_pairs(pairs, pair_scores, oracle_pairs):
    print("MATCHED PAIRS. Each pair shares a bit-identical Laguerre")
    print("tessellation, so the specimen is held fixed and only the")
    print("fabric changes. This is the comparison the design supports.")
    print("  seeds available: %s"
          % ", ".join(str(s) for s, _, _ in pairs))
    print("  %-10s %8s %10s %-28s" % ("statistic", "correct", "sign p",
                                      "per pair, girdle first"))
    for s in pair_scores:
        print("  %-10s %5d/%2d %10.4f %s"
              % (s["key"], s["wins"], s["n"], s["p"],
                 " ".join("+" if w else "-" for w in s["signs"])))
    print("  %-10s %5d/%2d %10.4f  (noise-free orientation average)"
          % ("ratio orc", oracle_pairs["wins"], oracle_pairs["n"],
             oracle_pairs["p"]))
    print()
    print("  %-6s %9s %9s %9s %9s" % ("seed", "R girdle", "R single",
                                      "D girdle", "D single"))
    for seed, g, s in pairs:
        print("  %-6d %9.3f %9.3f %9.3f %9.3f"
              % (seed, g["r"], s["r"], g["select"], s["select"]))
    print()


def draw_controls(thr, controls):
    print("CONTROLS, under thresholds fitted on all labelled sweeps.")
    items = sorted(thr.items())
    for i in range(0, len(items), 3):
        print("  thresholds: %s"
              % ", ".join("%s %+.4f" % (k, t) for k, t in items[i:i + 3]))
    print("  %-20s %-28s %7s %7s %7s %7s %7s"
          % ("sweep", "what", "A2", "A4", "A4/A2", "ratio", "select"))
    for c in controls:
        print("  %-20s %-28s %7.4f %7.4f %7.3f %7s %7s"
              % (c["name"], c["tag"], c["a2"], c["a4"], c["r"],
                 c["verdict"]["ratio"], c["verdict"]["select"]))
    lad = [c for c in controls if c["tag"].startswith("ladder")]
    a2 = [c["a2"] for c in lad]
    a4 = [c["a4"] for c in lad]
    print("  Across the ladder the two-fold amplitude runs %.4f to %.4f "
          "us and" % (min(a2), max(a2)))
    print("  the four-fold %.4f to %.4f us. Time of flight is coherent, "
          "so the" % (min(a4), max(a4)))
    print("  pattern survives f = 0 intact: whatever fabric-type content "
          "it")
    print("  carries is carried by the bulk anisotropy with no "
          "grain-to-grain")
    print("  contrast at all, and needs no scattering. That is the one")
    print("  positive result in this module and it does not depend on "
          "any")
    print("  classifier working.")
    print("  But the same ladder is a control the discriminants fail. "
          "Every")
    print("  rung is ONE tessellation under ONE fabric, so a statistic "
          "that")
    print("  reads fabric type must not move along it:")
    for key, _ in STATS:
        if key == "coda":
            continue
        v = [c[key] for c in lad]
        print("    %-10s ladder spread %8.4f, verdicts %s"
              % (key, max(v) - min(v),
                 "".join("g" if c["verdict"][key] == "girdle" else "s"
                         for c in lad)))
    print("  Compare each spread with that statistic's median "
          "leave-one-out")
    print("  margin above. Where they are comparable, a change carrying "
          "NO")
    print("  fabric-type information moves the statistic as far as the "
          "fabric")
    print("  does, and the verdict flips along the ladder.")
    print("  iso_gcal has NO fabric type and is nonetheless called a "
          "girdle,")
    print("  because an isotropic specimen has no two-fold term either. "
          "The")
    print("  single-crystal sweep is the extreme single maximum and the "
          "ratio")
    print("  misreads it too.")
    print()


def draw_coda_evidence(ev):
    print("THE CODA DISCRIMINANT, examined separately because the")
    print("unpaired count above flatters it.")
    print("  %-6s %10s %12s" % ("seed", "coda dB", "sigma_d^2 dB"))
    for seed, d, s in zip(ev["seeds"], ev["diffs"], ev["sdiff"]):
        print("  %-6d %+10.2f %+12.2f" % (seed, d, s))
    print("  paired, single minus girdle: mean %+.2f dB, sd %.2f, "
          "t %.2f, p %.4f" % (ev["diffs"].mean(),
                              ev["diffs"].std(ddof=1), ev["t"], ev["p"]))
    print("  median %+.2f dB; leave-one-pair-out means %s"
          % (float(np.median(ev["diffs"])),
             " ".join("%+.2f" % j for j in ev["jack"])))
    print("  the theory has the level tracking sigma_d^2 one for one, "
          "and that")
    print("  differs by %+.2f dB in the mean, so the sign is right and "
          "two of" % ev["sdiff"].mean())
    print("  the five pairs overshoot the predicted size by about five "
          "decibels.")
    print("  dropping the two hottest single-maximum sweeps,")
    print("  %s," % " and ".join(ev["hot"]))
    print("  leaves leave-one-out at %d/%d against a baseline of %d, "
          "AUC %.3f," % (ev["loo_drop"], ev["n_drop"], ev["base_drop"],
                         ev["auc_drop"]))
    print("  on a smallest held-out margin of %.2f dB."
          % ev["gap_drop"])
    print("  The SAME estimator on ONE specimen reads")
    print("  %s dB at ppw %s,"
          % (", ".join("%.2f" % v for _, v in sorted(ev["drift"].items())),
             ", ".join(str(p) for p in sorted(ev["drift"]))))
    print("  a drift of %.2f dB, larger than the separation it is asked"
          % (max(ev["drift"].values()) - min(ev["drift"].values())))
    print("  to carry. An absolute decibel level is not a transferable")
    print("  discriminant; every time-of-flight statistic here is a "
          "shape")
    print("  statistic and needs no such calibration.")
    print()


def draw_counts(table):
    print("OPERATING LIMIT: azimuth count, mean leave-one-out accuracy")
    print("over every maximally spread subset of the common grid.")
    head = "".join("%8d" % c for c in COUNTS)
    print("  %-10s%s" % ("statistic", head))
    for key, row in table.items():
        print("  %-10s%s"
              % (key, "".join("%8.3f" % row[c][0] for c in COUNTS)))
    print("  %-10s%s" % ("subsets", "".join("%8d" % table["ratio"][c][2]
                                            for c in COUNTS)))
    print()
    print("  median AUC over the same subsets")
    for key, row in table.items():
        print("  %-10s%s"
              % (key, "".join("%8.3f" % row[c][1] for c in COUNTS)))
    print()
    print("  On N equally spaced azimuths the harmonic k cannot be told "
          "from")
    print("  N - k, so the four-fold term collapses onto the two-fold at "
          "N = 6")
    print("  and onto its own Nyquist at N = 8. Every statistic here has "
          "a")
    print("  hard floor at 10 azimuths for that reason, and the 6-azimuth")
    print("  column is not a measurement of anything.")
    print()


def draw_pick_sensitivity(sens):
    print("PICK-THRESHOLD SENSITIVITY, leave-one-out count and AUC")
    keys = [k for k, _ in STATS if k != "coda"]
    print("  %-6s%s" % ("frac", "".join("%14s" % k for k in keys)))
    for row in sens:
        cells = "".join("%8s%6.3f" % ("%d/%d" % (row[k][0], row["n"]),
                                      row[k][1]) for k in keys)
        print("  %-6.2f%s" % (row["frac"], cells))
    print()
    print("  The verdict should not depend on where on the leading edge")
    print("  the pick is taken. Where a row moves by two or more counts")
    print("  across this table, the statistic is being carried by the")
    print("  processing choice and not by the fabric.")
    print()


def main():
    tested, worst = check_pick()
    rows, missing = compute_sweeps()
    draw_header(rows, missing, tested, worst)
    draw_ensemble(compute_ensemble())
    draw_sweeps(rows)
    draw_scores(compute_scores(rows))
    pairs, pair_scores, oracle_pairs = compute_pairs(rows)
    draw_pairs(pairs, pair_scores, oracle_pairs)
    draw_coda_evidence(compute_coda_evidence(rows, pairs))
    draw_controls(*compute_controls(rows))
    draw_counts(compute_counts(rows))
    draw_pick_sensitivity(compute_pick_sensitivity(rows))


if __name__ == "__main__":
    main()
