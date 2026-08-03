"""Does the fabric axis become recoverable where the Fresnel zone finally
exceeds the grain? The one window of this record that can test the
length-scale mechanism instead of assuming it.

THE QUESTION. Sec. 5.2 explains a negative result by a separation of
length scales: the first Fresnel zone is no larger than a mean grain, so
the beam interrogates about one grain boundary at a time and the return
is dominated by the geometry of that single facet rather than by an
average over many. That account has never been tested, because every
window analysed satisfies its inequality. The ratio of Fresnel diameter
to grain diameter at the window centre is 0.50 at 4-16 us, 0.63 at
10-22 us and 0.87 in the analysed 24-36 us gate, and moving the window
EARLIER only strengthens the inequality. An inequality is not tested
where it holds.

It fails late. With lambda = 1.925 mm and a realised volume-equivalent
grain of 17.2 mm the two lengths are equal at 39.7 us of two-way time,
and the coda is usable to about 52 us, so a window in which the beam
interrogates MORE than one boundary at a time exists in data already
recorded. Scoring the fabric-axis tests there tests the mechanism:

  axis recoverable where the inequality FAILS  -> mechanism SUPPORTED
  axis still unrecoverable there               -> mechanism REFUTED
  window inadmissible                          -> nothing is decided

The third outcome is a real outcome and this module is built to be able
to return it.

THE MIRROR PROBLEM, WHICH IS THE PRIMARY THREAT. The published gate is
bounded above at 36 us for a reason. The backwall arrival is at 52.2 us
measured, 51.95 us nominal from 2D/c, and stands about 48 dB above the
coda. Sec. sec:window already establishes the failure mode at the other
end of the record: the analytic envelope of a whole trace is not a local
operator and 53.7 per cent of the apparent envelope power at 10-22 us is
imported from the front arrival ten microseconds earlier, which is why
the fabric predictor there tracks specimens that cannot backscatter from
a grain boundary at all BETTER than it tracks the specimen (|r| 0.90 and
0.73 against 0.28 at 4-16 us, 0.70 and 0.56 against 0.25 at 10-22 us,
against 0.22 and 0.15 against 0.27 in the gate). A late window has the
same problem with the sign of time reversed. Backwall leakage is
therefore measured BEFORE any fabric number is looked at, and the two
zero-scattering controls are carried through every cell.

WHAT IS MEASURED, IN ORDER.

  0. HARNESS. The published fabric-axis numbers in the analysed gate,
     through this module's loading and this module's null, printed beside
     Sec. 5.2. Test one r = 0.41 with the true alignment tenth of thirty
     at ppw 6 and r = 0.21 at ppw 8; test two family-wise error rates
     0.100, 0.367, 0.667, 0.344. Nothing below is read if these fail.
  1. THE WINDOW CHOICE, with the Fresnel diameter and the ratio to grain
     at the centre of every candidate, and the two-way time at which the
     ratio reaches one.
  2. BACKWALL LEAKAGE, three ways, before any fabric result:
       (a) the imported share of Sec. sec:window's own convention, the
           envelope power lost when the trace is tapered to its window
           before the Hilbert transform;
       (b) the drop in windowed power when the backwall is excised from
           the RAW trace before the band-pass, which is the only excision
           that also removes the filter's own ringing;
       (c) the share the backwall contributes on its own, from a trace
           holding nothing else.
  3. THE ZERO-SCATTERING MARGIN, revolution level of the specimen over
     both controls, in dB, on the convention the other windows are
     reported on (22.9 and 21.2 dB in the gate; 0.51 and 0.02 dB at
     4-16 us untreated).
  4. THE TWO AXIS TESTS, T1 and T2, for the specimen AND both controls,
     then the twelve-tessellation first-rank count.

THE GRID. cs_f000_s11_ppw8 holds 30 azimuths where the production sweeps
hold 60, so every cell carrying both controls runs on the 30 common
azimuths: 15 DISTINCT alignments, because the bulk predictor is a
function of |c . n| alone and is exactly 180 degree periodic, and an
exact shift-null p floor of 1/15 = 0.0667. That is NOT the 30 alignments
and 1/30 of the published numbers. Every table names its grid and no rank
out of 15 is ever compared against a rank out of 30.

WHAT IS MEASURED AND WHAT IS INFERRED.
  MEASURED   every r, rank, p, level, margin and leakage share printed.
  INFERRED   the closing verdict of report_verdict, which is labelled
             where it is drawn.

WHAT IS IMPORTED AND WHAT IS NEW. The audited primitives are imported
UNCHANGED from analysis/fabric_axis_windows.py and nothing in it is
edited: the E[R^2] predictor and its label-volume cache reader, the
22-observable panel, the circular-shift null and its family-wise error
rate, the published-path level and panel used for the harness, and the
Fresnel convention of figures/fig_scales.py. Everything else here is new:
the raw-domain backwall excision, the asymmetric window taper, the late
candidates and every table below section 1.

TOUCHES NO CUDA. No fdtd.forward_*, no DiskSpecimen.build, no label
build. Every tessellation quantity comes from out/tesscache through the
imported reader, which raises rather than build a missing one. Reads
out/sweeps traces. Writes late_window_mechanism.npz beside this file.

Run with `python late_window_mechanism.py`.
"""
import os
import sys

import numpy as np
from scipy import stats
from scipy.signal import butter, hilbert, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SWD = os.path.join(ROOT, "out", "sweeps")
sys.path.insert(0, HERE)

import fabric_axis_windows as FAW                            # noqa: E402

C_REF, F0, DIA, THK = FAW.C_REF, FAW.F0, FAW.DIA, FAW.THK
BAND = FAW.BAND
LAM = FAW.LAM
DT_C = FAW.DT_C
T_MAX = FAW.T_MAX
AZ30 = FAW.AZ_COMMON

SPECIMEN = FAW.SPECIMEN
CONTROLS = FAW.CONTROLS
SEEDS = FAW.SEEDS

# The backwall is the far rim of the disc: 2D/c = 51.95 us nominal, 52.2 us
# measured on the seed-11 sweep. T_CUT is where the raw-domain excision
# takes it out, placed below the measured -40 dB leading edge of the
# arrival so that the excision removes the whole event and not its peak
# alone. CUT_PAD is the cosine skirt of the excision, which is applied to
# the RAW trace so that the band-pass never sees the backwall at all.
T_BACKWALL = 2.0 * DIA / C_REF
T_CUT = 51.0e-6
CUT_PAD = 0.5e-6
CUT_ALT = (49.0e-6, 50.0e-6, 51.0e-6, 51.5e-6)

# The low half of the recorded band. Lengthening the wavelength raises the
# Fresnel diameter at every range, so this moves the ratio the mechanism
# rests on without moving the window and therefore without moving any
# range-accumulated quantity.
SUBBAND = (0.8e6, 1.4e6)

# Candidates. (tag, lo, hi, upper skirt). The lower skirt is 2 us
# everywhere, matching analysis/median_trace_removal.py. The upper skirt
# is named per candidate because at a late window it is the skirt and not
# the window that decides how close the estimator reaches to the backwall:
# a 12 us gate ending at 50 us carries its skirt to 52 us, which is the
# backwall peak itself.
PAD_LO = 2.0e-6
CANDIDATES = (
    ("24-36", 24e-6, 36e-6, 2.0e-6, "published gate, reference"),
    ("30-42", 30e-6, 42e-6, 2.0e-6, "intermediate, reference"),
    ("38-50", 38e-6, 50e-6, 2.0e-6, "the manuscript's named late gate"),
    ("38-50c", 38e-6, 50e-6, 1.0e-6, "same gate, skirt clipped at the cut"),
    ("40-48", 40e-6, 48e-6, 2.0e-6, "8 us, same centre, 3 us more guard"),
    ("38-44", 38e-6, 44e-6, 2.0e-6, "6 us, centre 41 us, most guard"),
)
LATE_TAGS = ("38-50", "38-50c", "40-48", "38-44")

# The twelve tessellations with a cached label volume at ppw 8, each with
# the seed and concentration its own predictor is built from.
ENSEMBLE = ([(FAW.GIRDLE[s], s, -8.0) for s in SEEDS]
            + [(FAW.SINGLE[s], s, 3.93) for s in (11, 17, 23, 41)])

# Sec. 5.2, for the harness.
REF_T1 = FAW.REF_T1
REF_T2_FWER = FAW.REF_T2_FWER
REF_T2_MAXR = FAW.REF_T2_MAXR

_RAW = {}
_PREP = {}


# ─────────────────────────────── loading ──────────────────────────────
def _sos(fs, band=BAND):
    return butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)],
                  btype="band", output="sos")


def raw(name, az_keep=AZ30):
    """(azimuth, trace, dt) for one sweep, untouched, cached."""
    key = (name, az_keep)
    if key in _RAW:
        return _RAW[key]
    d = os.path.join(SWD, name)
    rows = []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        a = int(f[2:5])
        if az_keep is not None and a not in az_keep:
            continue
        with np.load(os.path.join(d, f)) as z:
            rows.append((a, np.asarray(z["trace"], float).ravel(),
                         float(z["dt"])))
    rows.sort(key=lambda r: r[0])
    _RAW[key] = rows
    return rows


def cut_mask(t, mode, tcut=T_CUT):
    """Raw-domain backwall excision, applied before the band-pass.

    mode 'full' keeps everything, 'drop' removes the backwall and all that
    follows it, 'only' keeps nothing else. The two are exact complements,
    so the two measurements they support cannot both be flattering.
    """
    if mode == "full":
        return np.ones_like(t)
    w = np.ones_like(t)
    lo = tcut - CUT_PAD
    w[t >= tcut] = 0.0
    s = (t > lo) & (t < tcut)
    w[s] = 0.5 * (1 + np.cos(np.pi * (t[s] - lo) / CUT_PAD))
    return w if mode == "drop" else 1.0 - w


def prep(name, az_keep=AZ30, mode="full", tcut=T_CUT, band=BAND):
    """Band-passed RF on the common grid, with the source and backwall
    references always taken from the UNMODIFIED trace.

    az    azimuths, ascending
    R     band-passed trace resampled onto tg, one row per azimuth
    src2  squared peak of the unfiltered analytic envelope, the reference
          every level in this module is quoted on
    e1    backwall envelope peak, the reference the 22-observable panel
          uses, measured on the unmodified trace even when the backwall
          has been excised from R, so that an excision can never move a
          normalisation.
    """
    key = (name, az_keep, mode, tcut, band)
    if key in _PREP:
        return _PREP[key]
    tg = np.arange(0.0, T_MAX, DT_C)
    az, rows, src2, e1 = [], [], [], []
    for a, tr, dt in raw(name, az_keep):
        fs = 1.0 / dt
        t = np.arange(len(tr)) * dt
        env = np.abs(hilbert(tr))
        k0, w = int(T_BACKWALL * fs), int(2e-6 * fs)
        az.append(a)
        src2.append(env.max() ** 2)
        e1.append(env[max(k0 - w, 0):k0 + w].max())
        rows.append(np.interp(tg, t, sosfiltfilt(_sos(fs, band),
                                                 tr * cut_mask(t, mode,
                                                               tcut))))
    out = dict(az=np.array(az), tg=tg, R=np.array(rows),
               src2=np.array(src2), e1=np.array(e1))
    _PREP[key] = out
    return out


def taper(tg, lo, hi, pad_lo=PAD_LO, pad_hi=PAD_LO):
    """Unity on [lo, hi], raised-cosine skirts of the named widths, zero
    outside. Asymmetric because a late window needs a shorter skirt at the
    top than at the bottom and the width of that skirt is the whole
    question."""
    w = np.zeros_like(tg)
    w[(tg >= lo) & (tg <= hi)] = 1.0
    a = (tg > lo - pad_lo) & (tg < lo)
    w[a] = 0.5 * (1 - np.cos(np.pi * (tg[a] - (lo - pad_lo)) / pad_lo))
    b = (tg > hi) & (tg < hi + pad_hi)
    w[b] = 0.5 * (1 + np.cos(np.pi * (tg[b] - hi) / pad_hi))
    return w


# ────────────────────────────── estimators ────────────────────────────
def power(name, cand, est, az_keep=AZ30, mode="full", tapered=True,
          tcut=T_CUT, band=BAND):
    """Per-azimuth windowed power, referenced to the source peak.

    est 'tf'   mean square of the band-passed trace over the window, with
               no transform at all. E|z|^2 = 2 E x^2 for a locally
               stationary segment, so this is an envelope power without an
               envelope, and it cannot import anything from outside the
               window except through the band-pass filter itself.
    est 'env'  the trace tapered to the window BEFORE the Hilbert
               transform, then the mean square of the envelope. Cannot
               import from beyond [lo - pad_lo, hi + pad_hi].
    tapered=False on 'env' is the untapered whole-trace envelope, which is
    only used to measure what tapering removes.
    """
    _, lo, hi, pad_hi, _ = cand
    s = prep(name, az_keep, mode, tcut, band)
    tg, R = s["tg"], s["R"]
    g = (tg >= lo) & (tg < hi)
    if est == "tf":
        return 2.0 * np.mean(R[:, g] ** 2, axis=1) / s["src2"]
    X = R * taper(tg, lo, hi, PAD_LO, pad_hi)[None, :] if tapered else R
    return np.mean(np.abs(hilbert(X, axis=1))[:, g] ** 2, axis=1) / s["src2"]


def level(name, cand, est, az_keep=AZ30, mode="full", band=BAND):
    """Per-azimuth level in dB re the per-azimuth source peak."""
    return 10 * np.log10(power(name, cand, est, az_keep, mode, True,
                               T_CUT, band))


def revolution(name, cand, est, az_keep=AZ30, mode="full", band=BAND):
    """Revolution level in dB. A revolution is a quadrature over a closed
    rotation and not a draw from a population, so the average is over
    powers."""
    return float(10 * np.log10(np.mean(power(name, cand, est, az_keep, mode,
                                             True, T_CUT, band))))


def panel(name, cand, az_keep=AZ30, mode="full"):
    """The 22-observable panel of Sec. 5.2 on the tapered window.

    The observables are computed by the imported analysis/
    fabric_axis_windows.observables, unchanged; what is new here is only
    the asymmetric taper it is handed.
    """
    _, lo, hi, pad_hi, _ = cand
    s = prep(name, az_keep, mode)
    tg = s["tg"]
    g = (tg >= lo) & (tg < hi)
    X = s["R"] * taper(tg, lo, hi, PAD_LO, pad_hi)[None, :]
    A = hilbert(X, axis=1)
    rows = [FAW.observables(A[i][g], X[i][g], s["e1"][i], DT_C, hi - lo)
            for i in range(len(X))]
    keys = sorted(rows[0])
    return s["az"], {k: np.array([r[k] for r in rows]) for k in keys}


# ───────────────────────────── length scales ──────────────────────────
def fresnel_ratio(cand, dg):
    """First Fresnel diameter at the window centre and its ratio to grain.

    figures/fig_scales.py's convention exactly, through the imported
    FAW.fresnel: two-way, so the first zone ends at sqrt(lambda L / 2) and
    its diameter is sqrt(2 lambda L), with L the ONE-WAY range at the
    window centre taken straight from the centre time.
    """
    d, L = FAW.fresnel((cand[1], cand[2]))
    return d, L, d / dg


def equality_time(dg):
    """Two-way time at which the Fresnel diameter equals the grain."""
    return dg ** 2 / (LAM * C_REF)


# ────────────────────────────── validation ────────────────────────────
def report_harness():
    """The published fabric-axis numbers, this path, printed beside them.

    Test one is scored on the published estimator and the published grid:
    the mean square of the band-passed envelope in the gate over the
    sweep-mean source peak, at each azimuth's own sample rate, on all 60
    azimuths, which is 30 distinct alignments. Test two is the published
    panel path: whole-trace Hilbert of the unfiltered trace. Both come
    through the imported FAW primitives, so what this section checks is
    that this module's grid, predictor and null return Sec. 5.2 and not
    that the panel has been rewritten.
    """
    ok = True
    print("=" * 79)
    print("0. HARNESS. Sec. 5.2's two axis tests in the published 24-36 us")
    print("   gate, on the published 60-azimuth grid: 30 distinct")
    print("   alignments, floor 1/30. Published against measured.")
    print("=" * 79)
    print("\n  T1  coda level against the E[R^2] predictor, circular-shift null")
    print(f"  {'sweep':<20}{'r':>9}{'pub':>7}{'rank':>10}{'pub':>8}")
    for name, ppw in (("girdle_perp", 6), ("girdle_perp_ppw8", 8)):
        s = FAW.sweep(name, None)
        y = FAW.level_native(name, FAW.PUBLISHED, None, ref="src_mean")
        pr = FAW.predictor(11, s["az"], -8.0, 0.0, ppw)
        r, rk, nd = FAW.shift_test(y, pr, s["az"])
        rr, rrk, rnd = REF_T1[name]
        good = abs(r - rr) < 0.006 and abs(rk - rrk) <= 1 and nd == rnd
        ok &= good
        print(f"  {name:<20}{r:>+9.3f}{rr:>+7.2f}{rk:>7d}/{nd:<2}"
              f"{rrk:>5d}/{rnd:<2}   {'ok' if good else 'MISMATCH'}")

    print("\n  T2  panel of 22 observables, exact circular-shift family-wise")
    print("      error rate of the largest |r|")
    print(f"  {'sweep':<20}{'n':>5}{'shifts':>7}{'max|r|':>9}{'pub':>7}"
          f"{'p_fw':>9}{'pub':>9}  {'argmax':<14}")
    for name, seed, kappa, _ in FAW.PANEL_SWEEPS:
        ppw = 6 if name in ("girdle_perp", "iso_gcal") else 8
        az, X = FAW.panel_native(name, FAW.PUBLISHED, None)
        keys = sorted(X)
        pr = FAW.predictor_for(name, seed, kappa, az, ppw)
        mr, arg, pf, nd = FAW.fwer(X, keys, pr, az)
        rf = REF_T2_FWER[name]
        good = abs(pf - rf) < 0.002
        if name in REF_T2_MAXR:
            good &= abs(mr - REF_T2_MAXR[name]) < 0.004
        ok &= good
        print(f"  {name:<20}{len(az):>5}{nd:>7}{mr:>9.3f}"
              f"{REF_T2_MAXR.get(name, np.nan):>7.3f}{pf:>9.4f}{rf:>9.4f}"
              f"  {arg:<14}{'ok' if good else 'MISMATCH'}")
    print("\n   " + ("HARNESS REPRODUCES" if ok
                     else "HARNESS DOES NOT REPRODUCE"))
    return ok


# ─────────────────────────── 1. the window choice ─────────────────────
def report_choice(store):
    print("\n" + "=" * 79)
    print("1. THE WINDOW CHOICE. Two requirements pull against each other:")
    print("   the centre must lie past the equality time so the Fresnel")
    print("   diameter EXCEEDS the grain, and the window and its skirt must")
    print("   end far enough before the backwall to be readable.")
    print("=" * 79)
    dg, n = FAW.grain_diameter()
    teq = equality_time(dg)
    print(f"   wavelength {LAM * 1e3:.3f} mm, {n} grains realised on the")
    print(f"   seed-11 tessellation, volume-equivalent grain {dg * 1e3:.2f} mm")
    print(f"   D_Fresnel equals the grain at {teq * 1e6:.1f} us of two-way")
    print(f"   time. Backwall {T_BACKWALL * 1e6:.2f} us nominal 2D/c,")
    print("   52.2 us measured. Excision cut at "
          f"{T_CUT * 1e6:.1f} us with a {CUT_PAD * 1e6:.1f} us skirt.")
    print(f"\n  {'window':<8}{'centre':>8}{'range':>9}{'D_Fres':>9}"
          f"{'ratio':>8}{'skirt to':>10}{'guard':>8}  {'note':<38}")
    for c in CANDIDATES:
        tag, lo, hi, ph, note = c
        d, L, rat = fresnel_ratio(c, dg)
        top = (hi + ph)
        print(f"  {tag:<8}{0.5 * (lo + hi) * 1e6:>7.1f}u{L * 1e3:>8.1f}m"
              f"{d * 1e3:>8.2f}m{rat:>8.3f}{top * 1e6:>9.1f}u"
              f"{(T_CUT - top) * 1e6:>7.1f}u  {note:<38}")
        store["scale_" + tag] = np.array([0.5 * (lo + hi), L, d, rat,
                                          T_CUT - top])
    print("\n   'guard' is the gap between the top of the taper skirt and")
    print("   the excision cut. Where it is negative the estimator's own")
    print("   window function reaches into the backwall arrival.")
    return dg, teq


# ───────────────────── 2. backwall leakage, measured first ────────────
def report_leakage(store):
    print("\n" + "=" * 79)
    print("2. BACKWALL LEAKAGE. Measured before any fabric result, because")
    print("   this is the mirror of the failure that disqualified the early")
    print("   windows. Sec. sec:window's convention (a); the raw-domain")
    print("   excision (b) and (c), which are new here.")
    print("=" * 79)
    print("\n   (a) IMPORTED SHARE on the untapered whole-trace envelope, the")
    print("       quantity Sec. sec:window reports as 53.7 per cent at")
    print("       10-22 us, 15.6 per cent in the gate and 3.0 per cent at")
    print("       30-42 us: the envelope power lost when the trace is")
    print("       tapered to its window before the transform.")
    print(f"  {'window':<8}", end="")
    for nm in (SPECIMEN, CONTROLS[0][0], CONTROLS[1][0]):
        print(f"{nm[:16]:>18}", end="")
    print()
    for c in CANDIDATES:
        print(f"  {c[0]:<8}", end="")
        for nm in (SPECIMEN, CONTROLS[0][0], CONTROLS[1][0]):
            pf = power(nm, c, "env", AZ30, "full", tapered=False)
            pt = power(nm, c, "env", AZ30, "full", tapered=True)
            v = float(1.0 - np.mean(pt) / np.mean(pf))
            store[f"imp_{c[0]}_{nm}"] = np.array([v])
            print(f"{100 * v:>17.1f}%", end="")
        print()

    print("\n   (b) BACKWALL-ATTRIBUTED SHARE. The backwall and everything")
    print("       after it are removed from the RAW trace before the")
    print("       band-pass, so the filter never sees the arrival either;")
    print("       the number is the drop in windowed power. This is the")
    print("       only excision that also removes the filter's own ringing.")
    print("       Both leakage-safe estimators, specimen and both controls.")
    for est in ("tf", "env"):
        print(f"\n       estimator '{est}'")
        print(f"  {'window':<8}", end="")
        for nm in (SPECIMEN, CONTROLS[0][0], CONTROLS[1][0]):
            print(f"{nm[:16]:>18}", end="")
        print()
        for c in CANDIDATES:
            print(f"  {c[0]:<8}", end="")
            for nm in (SPECIMEN, CONTROLS[0][0], CONTROLS[1][0]):
                pf = float(np.mean(power(nm, c, est, AZ30, "full")))
                pd = float(np.mean(power(nm, c, est, AZ30, "drop")))
                v = 1.0 - pd / pf
                store[f"drop_{est}_{c[0]}_{nm}"] = np.array([v])
                print(f"{100 * v:>17.1f}%", end="")
            print()

    print("\n   (c) BACKWALL-ONLY SHARE. A trace holding the backwall and")
    print("       nothing else, put through the same band-pass, the same")
    print("       taper and the same window. This is what the arrival")
    print("       contributes on its own and is the exact complement of (b).")
    for est in ("tf", "env"):
        print(f"\n       estimator '{est}'")
        print(f"  {'window':<8}", end="")
        for nm in (SPECIMEN, CONTROLS[0][0], CONTROLS[1][0]):
            print(f"{nm[:16]:>18}", end="")
        print()
        for c in CANDIDATES:
            print(f"  {c[0]:<8}", end="")
            for nm in (SPECIMEN, CONTROLS[0][0], CONTROLS[1][0]):
                pf = float(np.mean(power(nm, c, est, AZ30, "full")))
                po = float(np.mean(power(nm, c, est, AZ30, "only")))
                v = po / pf
                store[f"only_{est}_{c[0]}_{nm}"] = np.array([v])
                print(f"{100 * v:>17.1f}%", end="")
            print()
    print("\n   A window in which (b) or (c) is a large fraction is not")
    print("   measuring the coda. Read the zero-contrast column first: it")
    print("   has no coda to dilute the arrival, so it is the cleanest")
    print("   measurement of how far the backwall reaches.")

    print("\n   (d) SENSITIVITY OF (c) TO WHERE THE CUT IS PLACED. The")
    print("       excision is only as honest as the cut: a cut placed after")
    print("       a precursor leaves the precursor in and understates the")
    print("       arrival's reach. Zero-contrast control, estimator 'env'.")
    print(f"  {'window':<8}", end="")
    for tc in CUT_ALT:
        print(f"{('cut %.1f us' % (tc * 1e6)):>14}", end="")
    print()
    for c in CANDIDATES:
        if c[0] not in LATE_TAGS:
            continue
        print(f"  {c[0]:<8}", end="")
        for tc in CUT_ALT:
            if c[2] >= tc:
                print(f"{'in window':>14}", end="")
                continue
            pf = float(np.mean(power(CONTROLS[0][0], c, "env", AZ30, "full")))
            po = float(np.mean(power(CONTROLS[0][0], c, "env", AZ30, "only",
                                     tcut=tc)))
            store[f"cutsens_{c[0]}_{tc:.1e}"] = np.array([po / pf])
            print(f"{100 * po / pf:>13.1f}%", end="")
        print()


# ──────────────────── 3. the zero-scattering margin ───────────────────
def report_margin(store):
    print("\n" + "=" * 79)
    print("3. THE ZERO-SCATTERING MARGIN. Revolution level, dB re the")
    print("   per-azimuth source peak, and the margin of the specimen over")
    print("   two specimens that cannot backscatter from a grain boundary")
    print("   at all. The gate carries 22.9 and 21.2 dB; at 4-16 us")
    print("   untreated they collapse to 0.51 and 0.02 dB. A window with no")
    print("   margin cannot carry a level test.")
    print("=" * 79)
    for est in ("tf", "env"):
        for mode, mtag in (("full", "as recorded"),
                           ("drop", "backwall excised")):
            print(f"\n   estimator '{est}', {mtag}")
            print(f"  {'window':<8}{'specimen':>10}{'zerocontr':>11}"
                  f"{'margin':>9}{'cs f=0':>10}{'margin':>9}{'null pair':>11}")
            for c in CANDIDATES:
                sp = revolution(SPECIMEN, c, est, AZ30, mode)
                zc = revolution(CONTROLS[0][0], c, est, AZ30, mode)
                cs = revolution(CONTROLS[1][0], c, est, AZ30, mode)
                store[f"margin_{est}_{mode}_{c[0]}"] = np.array([sp, zc, cs])
                print(f"  {c[0]:<8}{sp:>10.2f}{zc:>11.2f}{sp - zc:>9.2f}"
                      f"{cs:>10.2f}{sp - cs:>9.2f}{zc - cs:>11.2f}")
    print("\n   'null pair' is zerocontrast minus cs f = 0: two records with")
    print("   the same geometry and no scattering in either, so it is what")
    print("   this estimator returns for a difference that must be zero.")


# ───────────────────────── 4. the two axis tests ──────────────────────
def report_axis(store):
    print("\n" + "=" * 79)
    print("4. THE TWO AXIS TESTS. All three sweeps on the 30 azimuths every")
    print("   ppw 8 sweep shares, so the shift null has 15 DISTINCT")
    print("   alignments and the exact p floor is 1/15 = 0.0667. This is not")
    print("   the 30 alignments and 1/30 of the published numbers and no")
    print("   rank here is comparable with one there.")
    print("=" * 79)
    az = np.array(AZ30)
    pr = FAW.predictor(11, az, -8.0, 0.0, 8)
    for mode, mtag in (("full", "as recorded"), ("drop", "backwall excised")):
        print(f"\n   {mtag}")
        for nm, note in ((SPECIMEN, "specimen, girdle k = -8"),
                         (CONTROLS[0][0], "CONTROL, " + CONTROLS[0][1]),
                         (CONTROLS[1][0], "CONTROL, " + CONTROLS[1][1])):
            print(f"\n   {nm}   {note}")
            print(f"  {'window':<8}{'T1 r tf':>10}{'rank':>7}"
                  f"{'T1 r env':>10}{'rank':>7}{'T2 max|r|':>11}{'p_fw':>8}"
                  f"  {'argmax':<13}")
            for c in CANDIDATES:
                row = []
                for est in ("tf", "env"):
                    y = level(nm, c, est, AZ30, mode)
                    r, rk, nd = FAW.shift_test(y, pr, az)
                    row += [r, f"{rk}/{nd}"]
                    store[f"t1_{est}_{mode}_{c[0]}_{nm}"] = np.array(
                        [r, rk, nd])
                a2, X = panel(nm, c, AZ30, mode)
                keys = sorted(X)
                mr, arg, pf, nd = FAW.fwer(X, keys, pr, a2)
                store[f"t2_{mode}_{c[0]}_{nm}"] = np.array([mr, pf, nd])
                print(f"  {c[0]:<8}{row[0]:>+10.3f}{row[1]:>7}"
                      f"{row[2]:>+10.3f}{row[3]:>7}{mr:>11.3f}{pf:>8.4f}"
                      f"  {arg:<13}")
    print("\n   Read the two control blocks first. A window in which the")
    print("   FABRIC predictor tracks a specimen with NO fabric contrast as")
    print("   well as it tracks the real one decides nothing, whatever the")
    print("   specimen does in it.")


def report_grid60(store):
    """The production grid, where a rank is out of thirty and not fifteen.

    cs_f000_s11_ppw8 holds 30 azimuths, so any cell carrying it runs on
    the 30 common azimuths. The specimen and zerocontrast_ppw8 both hold
    60, and on those the null has 30 distinct alignments and a floor of
    1/30 = 0.033, which is twice the resolution section 4 can offer. The
    price is that only ONE of the two zero-scattering controls is
    available, so this table is a supplement to section 4 and not a
    replacement for it. A rank out of 30 here is never compared with a
    rank out of 15 there.
    """
    print("\n" + "=" * 79)
    print("5. THE PRODUCTION GRID. Specimen and zerocontrast_ppw8 only,")
    print("   60 azimuths, 30 DISTINCT alignments, exact floor")
    print("   1/30 = 0.0333. cs_f000_s11_ppw8 has no rows here: it holds")
    print("   30 azimuths and cannot be put on this grid.")
    print("=" * 79)
    for nm, note in ((SPECIMEN, "specimen, girdle k = -8"),
                     (CONTROLS[0][0], "CONTROL, " + CONTROLS[0][1])):
        print(f"\n   {nm}   {note}")
        print(f"  {'window':<8}{'T1 r tf':>10}{'rank':>8}{'p':>8}"
              f"{'T1 r env':>10}{'rank':>8}{'T2 max|r|':>11}{'p_fw':>8}")
        for c in CANDIDATES:
            s = prep(nm, None, "full")
            pr = FAW.predictor(11, s["az"], -8.0, 0.0, 8)
            row = []
            for est in ("tf", "env"):
                y = level(nm, c, est, None, "full")
                r, rk, nd = FAW.shift_test(y, pr, s["az"])
                row += [r, rk, nd]
                store[f"g60_t1_{est}_{c[0]}_{nm}"] = np.array([r, rk, nd])
            a2, X = panel(nm, c, None, "full")
            keys = sorted(X)
            mr, arg, pf, nd = FAW.fwer(X, keys, pr, a2)
            store[f"g60_t2_{c[0]}_{nm}"] = np.array([mr, pf, nd])
            print(f"  {c[0]:<8}{row[0]:>+10.3f}{row[1]:>5}/{row[2]:<2}"
                  f"{row[1] / row[2]:>8.4f}{row[3]:>+10.3f}"
                  f"{row[4]:>5}/{row[5]:<2}{mr:>11.3f}{pf:>8.4f}")


def report_ensemble(store):
    print("\n" + "=" * 79)
    print("6. THE TWELVE TESSELLATIONS. The predictor is built from each")
    print("   specimen's own realised c-axes, so the true alignment is the")
    print("   zero shift by construction: this is an ORACLE. If it cannot")
    print("   find the axis, no estimator of this class can. First-rank")
    print("   count out of twelve against a chance 12/15 = 0.8, on the 30")
    print("   common azimuths, 15 alignments.")
    print("=" * 79)
    az = np.array(AZ30)
    keep = {}
    for mode in ("full", "drop"):
        print(f"\n   {'as recorded' if mode == 'full' else 'backwall excised'}")
        print(f"  {'window':<8}{'first/12 tf':>13}{'p exact':>10}"
              f"{'rank sum':>10}{'z':>7}{'first/12 env':>14}{'rank sum':>10}")
        for c in CANDIDATES:
            cell = {}
            for est in ("tf", "env"):
                rks = []
                for nm, seed, kappa in ENSEMBLE:
                    pr = FAW.predictor(seed, az, kappa, 0.0, 8)
                    y = level(nm, c, est, AZ30, mode)
                    _, rk, _ = FAW.shift_test(y, pr, az)
                    rks.append(rk)
                cell[est] = np.array(rks)
                store[f"ens_{est}_{mode}_{c[0]}"] = np.array(rks)
            keep[(c[0], mode)] = cell["tf"]
            k = int((cell["tf"] == 1).sum())
            pex = float(stats.binom.sf(k - 1, 12, 1.0 / 15.0))
            ssum = int(cell["tf"].sum())
            z = (ssum - 96.0) / np.sqrt((15.0 ** 2 - 1) / 12.0 * 12)
            print(f"  {c[0]:<8}{k:>10}/12{pex:>10.4f}{ssum:>10}{z:>7.2f}"
                  f"{int((cell['env'] == 1).sum()):>11}/12"
                  f"{cell['env'].sum():>10}")
    print("\n   Chance is 0.8 first ranks of twelve and a rank sum of 96,")
    print("   with a rank-sum standard deviation of 14.97.")

    print("\n   Per-tessellation ranks, estimator 'tf', as recorded")
    print(f"  {'window':<8}" + " ".join(f"{n.split('_')[1][:4]:>5}"
                                        for n, _, _ in ENSEMBLE))
    for c in CANDIDATES:
        print(f"  {c[0]:<8}"
              + " ".join(f"{v:>5d}" for v in keep[(c[0], "full")]))

    print("\n   Paired on the tessellation, which is the unit of")
    print("   replication, every candidate against the published gate:")
    g = keep[("24-36", "full")]
    print(f"  {'window':<8}{'better':>8}{'worse':>7}{'tied':>6}"
          f"{'sign p':>9}{'wilcoxon p':>12}")
    for c in CANDIDATES:
        if c[0] == "24-36":
            continue
        v = keep[(c[0], "full")]
        d = v - g
        nz = d[d != 0]
        sp = (float(stats.binomtest(int((nz < 0).sum()), len(nz)).pvalue)
              if len(nz) else 1.0)
        wp = float(stats.wilcoxon(v, g).pvalue) if np.any(d) else 1.0
        store[f"paired_{c[0]}"] = np.array([int((d < 0).sum()),
                                            int((d > 0).sum()),
                                            int((d == 0).sum()), sp, wp])
        print(f"  {c[0]:<8}{int((d < 0).sum()):>8}{int((d > 0).sum()):>7}"
              f"{int((d == 0).sum()):>6}{sp:>9.3f}{wp:>12.3f}")


def report_degrees(store):
    """The axis itself, in degrees, which is what 'recoverable' means.

    The estimator is T1's, taken off the azimuth lattice: the trial axis
    rotates the predictor continuously and the SIGNED correlation is
    maximised, because the forward model has the level rising with
    E[R^2] and a rotation that anticorrelates is not a candidate axis.
    alpha = 0 is the true registration by construction, so the error is
    fold_180(alpha). Chance for an axial angle folded onto [0, 90] is a
    mean of 45 degrees; tab:axisrecovery's time-of-flight channel reaches
    5.9 degrees over these same twelve specimens.
    """
    print("\n" + "=" * 79)
    print("7. THE AXIS IN DEGREES, over the twelve tessellations, on the")
    print("   convention of tab:axisrecovery. This is what a recovered axis")
    print("   would have to look like, and it is the number that can be put")
    print("   beside the time-of-flight channel's 5.9 degrees.")
    print("=" * 79)
    az = np.array(AZ30)
    print(f"  {'window':<8}{'mean |err|':>12}{'median':>9}{'worst':>8}"
          f"{'mean mod45':>12}   {'controls, |err| deg':<28}")
    for c in CANDIDATES:
        errs = []
        for nm, seed, kappa in ENSEMBLE:
            y = level(nm, c, "tf", AZ30, "full")
            al, r, _ = FAW.axis_fit(y, az, seed, kappa)
            errs.append(FAW.fold_180(al))
        e = np.array(errs)
        ce = []
        for cn, _ in CONTROLS:
            yc = level(cn, c, "tf", AZ30, "full")
            alc, _, _ = FAW.axis_fit(yc, az, 11, -8.0)
            ce.append(FAW.fold_180(alc))
        store[f"deg_{c[0]}"] = e
        store[f"degctl_{c[0]}"] = np.array(ce)
        print(f"  {c[0]:<8}{e.mean():>11.2f}d{np.median(e):>8.2f}"
              f"{e.max():>8.2f}"
              f"{np.mean([FAW.fold_45(v) for v in errs]):>12.2f}"
              f"   {ce[0]:>7.2f} {ce[1]:>7.2f}   (chance 45.0)")
    print("\n   A control's error is a draw from the chance distribution and")
    print("   is printed so the specimens' errors are read against")
    print("   something measured rather than against 45.0 asserted.")


def coda_centroid(name, cand, az_keep=AZ30, band=BAND):
    """Power-weighted spectral centroid of the windowed coda, Hz.

    The Fresnel diameter is set by the wavelength the coda actually
    carries, not by the nominal drive frequency, so a sub-band's ratio
    must be quoted on its measured centroid. On the full band this
    returns the number the published 1.925 mm wavelength assumes, which
    is the check that the convention has not been changed.
    """
    _, lo, hi, pad_hi, _ = cand
    s = prep(name, az_keep, "full", T_CUT, band)
    tg = s["tg"]
    g = (tg >= lo) & (tg < hi)
    X = (s["R"] * taper(tg, lo, hi, PAD_LO, pad_hi)[None, :])[:, g]
    P = np.abs(np.fft.rfft(X * np.hanning(X.shape[1])[None, :],
                           axis=1)) ** 2
    f = np.fft.rfftfreq(X.shape[1], DT_C)
    m = (f >= band[0]) & (f <= band[1])
    return float((P[:, m] * f[m]).sum() / P[:, m].sum())


def report_subband(store, dg):
    """The one lever this record has that moves the ratio WITHOUT moving
    the range, and so breaks the collinearity that makes section 4
    ambiguous.

    D_Fresnel = sqrt(2 lambda L) rises with the wavelength as well as with
    the range. Restricting the analysis band to the low half of the
    recorded band raises lambda by the ratio of the centroids and so
    raises the Fresnel diameter at EVERY window, including the published
    gate, where the range and therefore any range-accumulated bulk
    anisotropy is unchanged. A mechanism that is about the Fresnel zone
    predicts an improvement here; a channel that is about accumulated
    path length predicts none. Nothing is re-simulated: this is a filter
    applied to traces already recorded.
    """
    print("\n" + "=" * 79)
    print("8. THE SUB-BAND TEST, which moves the Fresnel diameter without")
    print("   moving the range. D_Fresnel = sqrt(2 lambda L), so the ratio")
    print("   can be raised by lengthening the wavelength instead of by")
    print("   going late. Restricting the analysis band to its low half")
    print("   raises the ratio at the PUBLISHED gate, where the range and")
    print("   any range-accumulated bulk anisotropy are untouched. This")
    print("   separates the two accounts that section 4 cannot.")
    print("=" * 79)
    print(f"\n   full band {BAND[0] / 1e6:.1f}-{BAND[1] / 1e6:.1f} MHz,"
          f" low sub-band {SUBBAND[0] / 1e6:.1f}-{SUBBAND[1] / 1e6:.1f} MHz")
    print(f"\n  {'window':<8}{'band':<10}{'centroid':>10}{'lambda':>9}"
          f"{'D_Fres':>9}{'ratio':>8}{'specimen':>10}{'margin zc':>11}"
          f"{'margin cs':>11}")
    cells = [c for c in CANDIDATES if c[0] in ("24-36", "30-42", "38-50",
                                               "40-48")]
    for c in cells:
        for band, btag in ((BAND, "full"), (SUBBAND, "low")):
            fc = coda_centroid(SPECIMEN, c, AZ30, band)
            lam = C_REF / fc
            L = 0.5 * C_REF * 0.5 * (c[1] + c[2])
            d = np.sqrt(2.0 * lam * L)
            sp = revolution(SPECIMEN, c, "tf", AZ30, "full", band)
            zc = revolution(CONTROLS[0][0], c, "tf", AZ30, "full", band)
            cs = revolution(CONTROLS[1][0], c, "tf", AZ30, "full", band)
            store[f"sub_{btag}_{c[0]}"] = np.array([fc, lam, d, d / dg,
                                                    sp, sp - zc, sp - cs])
            print(f"  {c[0]:<8}{btag:<10}{fc / 1e6:>9.3f}M{lam * 1e3:>8.3f}m"
                  f"{d * 1e3:>8.2f}m{d / dg:>8.3f}{sp:>10.2f}{sp - zc:>11.2f}"
                  f"{sp - cs:>11.2f}")

    print("\n   T1 in the sub-band, 30 common azimuths, 15 alignments,")
    print("   specimen and both zero-scattering controls.")
    az = np.array(AZ30)
    pr = FAW.predictor(11, az, -8.0, 0.0, 8)
    print(f"  {'window':<8}{'band':<8}{'specimen r':>12}{'rank':>7}"
          f"{'zerocontr r':>13}{'rank':>7}{'cs f=0 r':>11}{'rank':>7}"
          f"{'first/12':>10}")
    for c in cells:
        for band, btag in ((BAND, "full"), (SUBBAND, "low")):
            row = []
            for nm in (SPECIMEN, CONTROLS[0][0], CONTROLS[1][0]):
                y = level(nm, c, "tf", AZ30, "full", band)
                r, rk, nd = FAW.shift_test(y, pr, az)
                row += [r, f"{rk}/{nd}"]
            rks = []
            for nm, seed, kappa in ENSEMBLE:
                p2 = FAW.predictor(seed, az, kappa, 0.0, 8)
                y = level(nm, c, "tf", AZ30, "full", band)
                _, rk, _ = FAW.shift_test(y, p2, az)
                rks.append(rk)
            rks = np.array(rks)
            store[f"subt1_{btag}_{c[0]}"] = np.array(
                [row[0], row[2], row[4], int((rks == 1).sum()), rks.sum()])
            print(f"  {c[0]:<8}{btag:<8}{row[0]:>+12.3f}{row[1]:>7}"
                  f"{row[2]:>+13.3f}{row[3]:>7}{row[4]:>+11.3f}{row[5]:>7}"
                  f"{int((rks == 1).sum()):>7}/12")
    print("\n   Read this against section 4. If the ratio is what matters,")
    print("   the low sub-band should improve the published gate, whose")
    print("   range never changed. If it does not, the late improvement of")
    print("   section 4 is about range and not about the Fresnel zone.")


def report_verdict(store, dg, teq):
    """INFERRED. Everything above is measured; this paragraph is not."""
    print("\n" + "=" * 79)
    print("9. VERDICT. INFERRED from the tables above, and labelled so.")
    print("=" * 79)
    print("\n   Admissibility is decided by section 2 and section 3 and by")
    print("   the two control blocks of section 4, in that order. A late")
    print("   window is admissible only if the backwall contributes a small")
    print("   share of its power, the specimen keeps a margin over both")
    print("   zero-scattering controls, and neither control tracks the")
    print("   fabric predictor. If it is not admissible, the axis result in")
    print("   it decides nothing about the length-scale mechanism, and the")
    print("   honest answer to the question this module was written for is")
    print("   window-inadmissible.")
    for tag in LATE_TAGS:
        m = store.get(f"margin_tf_full_{tag}")
        b = store.get(f"only_env_{tag}_{CONTROLS[0][0]}")
        t1 = store.get(f"t1_tf_full_{tag}_{SPECIMEN}")
        c1 = store.get(f"t1_tf_full_{tag}_{CONTROLS[0][0]}")
        c2 = store.get(f"t1_tf_full_{tag}_{CONTROLS[1][0]}")
        print(f"\n   {tag}: margins {m[0] - m[1]:+.2f} and {m[0] - m[2]:+.2f}"
              f" dB, backwall-only share of the zero-contrast envelope"
              f" {100 * b[0]:.1f} %,")
        print(f"          specimen |r| {abs(t1[0]):.3f} rank {int(t1[1])}/15,"
              f" controls |r| {abs(c1[0]):.3f} and {abs(c2[0]):.3f}")


def main():
    if not report_harness():
        raise SystemExit("harness does not reproduce; nothing below is valid")
    store = {}
    dg, teq = report_choice(store)
    report_leakage(store)
    report_margin(store)
    report_axis(store)
    report_grid60(store)
    report_ensemble(store)
    report_degrees(store)
    report_subband(store, dg)
    report_verdict(store, dg, teq)
    np.savez(os.path.join(HERE, "late_window_mechanism.npz"), **store)
    print("\nwrote late_window_mechanism.npz")


if __name__ == "__main__":
    main()
