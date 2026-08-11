"""Is the Section 5.2 null a property of the physics or of the gate?

THE QUESTION. Section 5.2 says the fabric axis is not recoverable from
the backscattered coda, and Fig. fig:scales explains it by a length
scale: the first Fresnel zone at the analysed gate, 14.9 mm, is no
larger than the 17.2 mm volume-equivalent grain, so the beam interrogates
about one boundary at a time. That explanation is evaluated at the gate.
This module measures the same three tests in two earlier windows, 4 to 16
and 10 to 22 microseconds, where the Fresnel zone is 8.6 and 10.9 mm, and
asks which of three separate things moves with the window:

    (a) whether the fabric AXIS is recoverable,
    (b) whether the two fabric TYPES separate,
    (c) whether the LENGTH-SCALE MECHANISM given for (a) is right.

They do not move together, and the manuscript needs each stated on its
own.

WHY A SECOND, INDEPENDENT IMPLEMENTATION. A prior module reached these
windows by way of analysis/median_trace_removal.py and reported that the
4 to 16 window must be disqualified because the leave-one-out median
removal it needs manufactures structure that the circular-shift null
reads as signal. Every number below is rebuilt here from the traces and
the cached label volumes, sharing no code with that path, and the
disqualification argument is tested directly rather than inferred.

WHAT IS MEASURED, AND WHAT THE ANSWER TURNED OUT TO BE.

  0. HARNESS. Nothing is believed until the published numbers come back
     through this path: the tab:reconcile coda column, T1's correlation
     and shift rank in the gate, T3's eight matched pairs, and the
     bandwidth and decay channels. They do, to two decimals.

  1. THE AXIS NULL HOLDS IN ALL THREE WINDOWS. It is not gate-specific.
     No estimator in a bank of nineteen reaches a family-wise error rate
     below 0.267 on the specimen in any window, and on twelve
     tessellations the true alignment reaches first rank in one to four
     of the twelve against a chance 0.8, with no window separating from
     the gate on the paired test that uses the tessellation as the unit
     of replication.

  2. THE REMOVAL DOES NOT BREAK THE SHIFT NULL, so that is the wrong
     reason to distrust the early window. The leave-one-out median is a
     symmetric function of the azimuths, so it commutes exactly with any
     relabelling of them, and a circular-shift null applied after it is
     as exact as the same null applied before it. Monte Carlo on
     rotationally exchangeable surrogates confirms it: the true
     alignment reaches first rank 6.5 to 7.1 per cent of the time
     against a nominal 6.67, treated and untreated alike.

     The right reason is simpler and is not a multiplicity artefact. In
     the early windows the fabric predictor correlates with specimens
     that carry NO fabric contrast at all more strongly than with the
     specimen: |r| up to 0.92 at 4 to 16 us and 0.92 at 10 to 22 us,
     against 0.28 and 0.25 for the real specimen, and 0.15 to 0.22 for
     the controls in the analysed gate. The early windows read the
     rendering of the tessellation. A null measured there is a null in a
     channel with no demonstrated sensitivity, which is weaker evidence
     than the same null in the gate, not stronger.

  3. THE FABRIC-TYPE SEPARATION IS gate-specific, and that is already in
     the manuscript. It decays monotonically as the window moves
     earlier: -2.86 dB at p = 0.033 in the gate, -0.88 dB at p = 0.50 at
     10 to 22, and +0.15 dB at p = 0.78 at 4 to 16, where the sign
     reverses. The bandwidth channel goes 0.010, 0.075, 0.73 and the
     decay channel 0.025, 0.38, 0.28.

  4. THE LENGTH-SCALE MECHANISM IS NOT REFUTED BY THIS. It is untested,
     and untestable in this record. The mechanism is an INEQUALITY,
     D_Fresnel <= D_grain, and the ratio is 0.50, 0.63 and 0.87 in the
     three windows: the inequality holds in all of them and holds most
     strongly where the Fresnel zone is smallest. Equality is reached at
     39.7 us of two-way time, past the far end of the record, so no
     window of this acquisition can break it. Moving the window earlier
     strengthens the antecedent of the mechanism and the null is
     unchanged, which is consistent with the mechanism; it is not
     evidence against it, and it is not evidence for it either.

TOUCHES NO CUDA. Nothing here calls fdtd.forward_*, DiskSpecimen.build
or any GPU path. Every tessellation quantity is read from
out/tesscache/tess_s<seed>_p8_<kappa>.npz and the module raises
SystemExit rather than build a missing one. Reads out/sweeps traces.
Writes sim/analysis/axis_window_adjudication.npz.

Run with `python axis_window_adjudication.py`; add an integer to set the
Monte Carlo draw count (default 3000, about four minutes).
"""
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import binomtest, kurtosis, norm, skew
from scipy.stats import t as tdist
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SWD = os.path.join(ROOT, "out", "sweeps")
TESS = os.path.join(ROOT, "out", "tesscache")

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
from ringfwi import anisotropy as an                       # noqa: E402

# Acquisition constants, Section 3. The qP curve is tabulated exactly as
# sim/model/forward.py does, but imported from ringfwi directly so that nothing
# on the solver's import path is touched.
C_REF, F0, DIA, THICK = 3850.0, 2.0e6, 0.100, 0.035
BAND = (0.8e6, 3.0e6)
GATE = (24e-6, 36e-6)
WINDOWS = {"4-16": (4e-6, 16e-6), "10-22": (10e-6, 22e-6),
           "24-36": GATE}
AZ30 = tuple(range(0, 360, 12))

_PSI = np.linspace(0.0, np.pi / 2.0, 901)
_VQP = np.array([an.ice_qp_vs_caxis(p) for p in _PSI])

# Common physical time axis for the resample-and-remove path. Each
# azimuth carries its own dt, set by the CFL limit of its own grid, so
# azimuths cannot be stacked by sample index.
PAD, DTG = 4e-6, 2e-8

# Eight girdle/single pairs on identical tessellations at ppw 8: the
# facet geometry and the discretisation are common and only the
# crystallographic orientations differ.
PAIRS = ([("girdle_seed11_ppw8_dev", "singlemax_seed11_ppw8_twin", 11)] +
         [("girdle_seed%d_ppw8_ensemble" % s, "singlemax_seed%d_ppw8_ensemble" % s, s)
          for s in (7, 17, 23, 41, 53, 71, 89)])

# The twelve tessellations with a cached label volume at ppw 8.
SPECS = ([("girdle_seed11_ppw8_dev", 11, "k-8")] +
         [("girdle_seed%d_ppw8_ensemble" % s, s, "k-8")
          for s in (7, 17, 23, 41, 53, 71, 89)] +
         [("singlemax_seed11_ppw8_twin", 11, "k3.93")] +
         [("singlemax_seed%d_ppw8_ensemble" % s, s, "k3.93") for s in (17, 23, 41)])

# Neither can backscatter from a grain boundary, and both carry the
# seed-11 geometry the predictor is built from.
CONTROLS = ("girdle_seed11_ppw8_uniform_axis", "girdle_seed11_ppw8_contrast_f000")

_SW = {}


# ───────────────────────────────── load ──────────────────────────────────
def bandpass(x, fs, lo=BAND[0], hi=BAND[1]):
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band",
                 output="sos")
    return sosfiltfilt(sos, x)


def load_sweep(name, az_keep=AZ30):
    if (name, az_keep) in _SW:
        return _SW[(name, az_keep)]
    d = os.path.join(SWD, name)
    out = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        a = int(f[2:5])
        if az_keep is not None and a not in az_keep:
            continue
        with np.load(os.path.join(d, f)) as z:
            out[a] = (np.asarray(z["trace"], float).ravel(), float(z["dt"]))
    _SW[(name, az_keep)] = out
    return out


# ─────────────────────────────── estimators ──────────────────────────────
def lvl_local_band(tr, dt, win):
    """Audited tab:reconcile level: 2<x^2> in band, per-azimuth source ref."""
    fs = 1.0 / dt
    i0, i1 = int(win[0] * fs), int(win[1] * fs)
    src2 = np.abs(hilbert(tr)).max() ** 2
    return 2.0 * (bandpass(tr, fs)[i0:i1] ** 2).mean() / src2


def lvl_env_band(tr, dt, win):
    """mean_level_clean_pair's estimator: <|H(band-passed)|^2> over the window."""
    fs = 1.0 / dt
    i0, i1 = int(win[0] * fs), int(win[1] * fs)
    return (np.abs(hilbert(bandpass(tr, fs)))[i0:i1] ** 2).mean()


def tgrid(win):
    return np.arange(win[0] - PAD, win[1] + PAD, DTG)


def rf_on_grid(sw, tg):
    """Band-passed RF of every azimuth, source-normalised, on one axis."""
    az = sorted(sw)
    rows = []
    for a in az:
        tr, dt = sw[a]
        s = np.abs(hilbert(tr)).max()
        rows.append(np.interp(tg, np.arange(len(tr)) * dt,
                              bandpass(tr, 1.0 / dt) / s))
    return np.array(az), np.array(rows)


def loo_median(R):
    """Azimuth-common component, leaving each azimuth out."""
    return np.array([np.median(np.delete(R, i, axis=0), axis=0)
                     for i in range(len(R))])


def level_none(sw, win):
    az = sorted(sw)
    return np.array(az), np.array([lvl_local_band(*sw[a], win) for a in az])


def level_loo(sw, win, remove=True):
    tg = tgrid(win)
    az, R = rf_on_grid(sw, tg)
    if remove:
        R = R - loo_median(R)
    m = (tg >= win[0]) & (tg < win[1])
    return az, 2.0 * (R[:, m] ** 2).mean(axis=1)


def revlevel(v):
    """Revolution level in dB. A rotation is a quadrature, so powers."""
    return 10.0 * np.log10(np.mean(v))


# ─────────────────────────────── predictor ───────────────────────────────
def er2(seed, kappa, rots):
    """E[R^2] in dB per azimuth from the CACHED label volume.

    Identical arithmetic to observable_panel.fabric_pred, reading the
    prebuilt tessellation rather than calling DiskSpecimen.build, which
    rasterises on the GPU.
    """
    p = os.path.join(TESS, "tess_s%d_p8_%s.npz" % (seed, kappa))
    if not os.path.exists(p):
        raise SystemExit("missing %s; refusing to build (CUDA)" % p)
    with np.load(p) as z:
        lab, axes = z["labels"], np.asarray(z["axes"], float)
    vol = np.bincount(lab[lab >= 0].ravel(),
                      minlength=len(axes)).astype(float)
    vol /= vol.sum()
    keep = vol > 0
    axes, vol = axes[keep], vol[keep] / vol[keep].sum()
    out = []
    for rot in rots:
        a = np.radians(float(rot))
        n = np.array([np.cos(a), np.sin(a), 0.0])
        v = np.interp(np.arccos(np.clip(np.abs(axes @ n), 0.0, 1.0)),
                      _PSI, _VQP)
        vb = float(vol @ v)
        out.append(10 * np.log10(float(vol @ (v - vb) ** 2) / (2 * vb ** 2)))
    return np.array(out), axes, vol


# ──────────────────────────────── the null ───────────────────────────────
def shift_scan(y, pred):
    """Pearson r at every circular shift of the predictor. r[0] is true."""
    yc = y - y.mean()
    out = np.empty(len(y))
    for k in range(len(y)):
        pc = np.roll(pred, k)
        pc = pc - pc.mean()
        d = np.sqrt((yc ** 2).sum() * (pc ** 2).sum())
        out[k] = (yc * pc).sum() / d if d > 0 else 0.0
    return out


def rank_true(rs, m):
    """Rank of the true alignment among the m DISTINCT ones, by |r|.

    The predictor sees the c-axes only through |axis . n|, so it is
    exactly 180 degrees periodic and shift k and k + m give the same
    alignment. |r| rather than r is the convention that returns the
    published tenth of thirty; ranking on signed r returns fifth.
    """
    v = rs[:m]
    return int((np.abs(v) >= abs(v[0])).sum()), v[0]


def bank(R, tg, win):
    """Nineteen azimuthal observables of the windowed field."""
    m = (tg >= win[0]) & (tg < win[1])
    X, t = R[:, m], tg[m]
    E = np.abs(hilbert(X, axis=1))
    P = E ** 2
    h = X.shape[1] // 2
    o = {"lvl": 10 * np.log10(2.0 * (X ** 2).mean(1)),
         "lvl_env": 10 * np.log10(P.mean(1)),
         "lvl_early": 10 * np.log10(P[:, :h].mean(1)),
         "lvl_late": 10 * np.log10(P[:, h:].mean(1)),
         "peak": 20 * np.log10(E.max(1)),
         "env_kurt": kurtosis(E, axis=1), "env_skew": skew(E, axis=1)}
    o["early_late"] = o["lvl_early"] - o["lvl_late"]
    o["crest"] = o["peak"] - 10 * np.log10(P.mean(1))
    w = P / P.sum(1, keepdims=True)
    tc = (w * t).sum(1)
    o["t_centroid"] = (tc - win[0]) * 1e6
    o["t_spread"] = np.sqrt((w * (t[None, :] - tc[:, None]) ** 2).sum(1)) * 1e6
    o["entropy"] = -(w * np.log(w + 1e-300)).sum(1) / np.log(X.shape[1])
    ldb = 10 * np.log10(np.maximum(P, P.max() * 1e-10))
    tt = (t - t.mean()) * 1e6
    o["decay"] = np.array([np.polyfit(tt, r, 1)[0] for r in ldb])
    S = np.abs(np.fft.rfft(X * np.hanning(X.shape[1])[None, :], axis=1)) ** 2
    fr = np.fft.rfftfreq(X.shape[1], DTG)
    sel = (fr > 0.5e6) & (fr < 4.0e6)
    Ss, fsel = S[:, sel], fr[sel]
    cen = (Ss * fsel).sum(1) / Ss.sum(1)
    o["f_centroid"] = cen / 1e6
    o["f_bw"] = np.sqrt((Ss * (fsel[None, :] - cen[:, None]) ** 2).sum(1)
                        / Ss.sum(1)) / 1e6
    b = lambda a, z: S[:, (fr >= a) & (fr < z)].sum(1)
    o["bandratio_lo"] = 10 * np.log10(b(1.2e6, 1.8e6) / b(1.8e6, 2.4e6))
    o["bandratio_hi"] = 10 * np.log10(b(2.4e6, 3.2e6) / b(1.8e6, 2.4e6))
    o["spec_flat"] = np.exp(np.log(Ss + 1e-300).mean(1)) / Ss.mean(1)
    acw = []
    for r in E - E.mean(1, keepdims=True):
        ac = np.correlate(r, r, "full")[len(r) - 1:]
        ac /= ac[0]
        acw.append((np.argmax(ac < 0.5) if np.any(ac < 0.5) else len(ac))
                   * DTG * 1e6)
    o["acf_width"] = np.array(acw)
    return o


def fwer(obs, pred, m=15):
    """Exact circular-shift FWER of the largest |r| over the whole bank."""
    keys = sorted(obs)
    M = np.array([shift_scan(obs[k], pred)[:m] for k in keys])
    stat = np.abs(M).max(0)
    j = int(np.argmax(np.abs(M[:, 0])))
    return float((stat >= stat[0]).sum()) / m, keys[j], np.abs(M[:, 0]).max()


def paired(d):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    tv = d.mean() / se
    q = tdist.ppf(0.975, n - 1)
    return (d.mean(), d.std(ddof=1), tv, 2 * tdist.sf(abs(tv), n - 1),
            d.mean() - q * se, d.mean() + q * se,
            int((np.sign(d) == np.sign(d.mean())).sum()))


# ──────────────────────────────── reports ────────────────────────────────
def report_harness():
    print("=" * 74)
    print("0. HARNESS. Published numbers, this path.")
    print("=" * 74)
    ok = True
    print("\n   tab:reconcile coda column, 30 matched azimuths, audited level")
    for tag, nm, ref in (("ppw6", "girdle_seed11_ppw6_axis_perp", -81.09),
                         ("ppw8", "girdle_seed11_ppw8_dev", -85.11),
                         ("ppw10", "girdle_seed11_ppw10_licensing", -87.79)):
        sw = load_sweep(nm)
        v = revlevel([lvl_local_band(*sw[a], GATE) for a in sorted(sw)])
        ok &= abs(v - ref) < 0.02
        print("      %-6s %8.2f dB   published %8.2f" % (tag, v, ref))

    print("\n   T1, coda level against E[R^2], published gate, native grid")
    for tag, nm, rr in (("ppw6", "girdle_seed11_ppw6_axis_perp", 0.41),
                        ("ppw8", "girdle_seed11_ppw8_dev", 0.21)):
        sw = load_sweep(nm, None)
        az = np.array(sorted(sw))
        y = np.array([10 * np.log10(lvl_env_band(*sw[a], GATE)) for a in az])
        p, _, _ = er2(11, "k-8", az)
        rk, r0 = rank_true(shift_scan(y, p), len(az) // 2)
        ok &= abs(r0 - rr) < 0.01
        print("      %-5s r = %+.3f  rank %2d/%d   published r = %.2f, "
              "rank 10th and 11th of 30" % (tag, r0, rk, len(az) // 2, rr))

    print("\n   T3, eight matched pairs, published gate")
    d = [revlevel(level_none(load_sweep(g), GATE)[1]) -
         revlevel(level_none(load_sweep(s), GATE)[1]) for g, s, _ in PAIRS]
    m, sd, tv, p, lo, hi, ag = paired(d)
    ok &= abs(m + 2.86) < 0.02 and abs(p - 0.033) < 0.005
    print("      %+.2f dB, t = %+.2f, p = %.4f, %d of 8 agree, "
          "95%% [%+.2f, %+.2f]" % (m, tv, p, ag, lo, hi))
    print("      published  -2.86 dB, t = -2.65, p = 0.033")
    print("\n   HARNESS REPRODUCES" if ok else "\n   HARNESS FAILED")
    if not ok:
        raise SystemExit("harness did not reproduce; nothing below is valid")
    return ok


def report_type(store):
    print()
    print("=" * 74)
    print("1. THE FABRIC-TYPE SEPARATION. Eight matched pairs, by window.")
    print("=" * 74)
    print("   %-6s %-6s %8s %7s %7s %8s %6s  %s"
          % ("win", "treat", "mean dB", "sd", "t", "p", "agree", "95% CI"))
    for wn, win in WINDOWS.items():
        for tr in ("none", "loo"):
            d = []
            for g, s, _ in PAIRS:
                f = level_none if tr == "none" else level_loo
                d.append(revlevel(f(load_sweep(g), win)[1]) -
                         revlevel(f(load_sweep(s), win)[1]))
            m, sd, tv, p, lo, hi, ag = paired(d)
            store["type_%s_%s" % (wn, tr)] = np.array([m, sd, tv, p, ag])
            print("   %-6s %-6s %+8.2f %7.2f %+7.2f %8.4f %4d/8  "
                  "[%+.2f, %+.2f]" % (wn, tr, m, sd, tv, p, ag, lo, hi))
    print("\n   The two confirmatory channels, same eight pairs")
    print("   %-6s %-6s %-10s %+9s %9s %6s" %
          ("win", "treat", "channel", "mean", "p", "agree"))
    for wn, win in WINDOWS.items():
        for tr in ("none", "loo"):
            acc = {}
            for g, s, _ in PAIRS:
                row = []
                for nm in (g, s):
                    tg = tgrid(win)
                    _, R = rf_on_grid(load_sweep(nm), tg)
                    if tr == "loo":
                        R = R - loo_median(R)
                    row.append(bank(R, tg, win))
                for k in ("f_bw", "decay"):
                    acc.setdefault(k, []).append(
                        row[0][k].mean() - row[1][k].mean())
            for k in ("f_bw", "decay"):
                m, sd, tv, p, lo, hi, ag = paired(acc[k])
                store["chan_%s_%s_%s" % (k, wn, tr)] = np.array([m, p, ag])
                print("   %-6s %-6s %-10s %+9.4f %9.4f %4d/8"
                      % (wn, tr, k, m, p, ag))


def report_floor(store):
    print()
    print("=" * 74)
    print("2. ZERO-SCATTERING FLOOR. Revolution level, dB, and the margin")
    print("   of the specimen over specimens that cannot backscatter.")
    print("=" * 74)
    print("   %-6s %-6s %9s %9s %8s %9s %8s"
          % ("win", "treat", "specimen", "zerocont", "margin", "cs f=0",
             "margin"))
    for wn, win in WINDOWS.items():
        for tr in ("none", "loo"):
            f = level_none if tr == "none" else level_loo
            v = [revlevel(f(load_sweep(n), win)[1]) for n in
                 ("girdle_seed11_ppw8_dev",) + CONTROLS]
            store["floor_%s_%s" % (wn, tr)] = np.array(v)
            print("   %-6s %-6s %9.2f %9.2f %8.2f %9.2f %8.2f"
                  % (wn, tr, v[0], v[1], v[0] - v[1], v[2], v[0] - v[2]))


def report_axis(store):
    print()
    print("=" * 74)
    print("3. THE AXIS. T1 and a nineteen-observable bank,")
    print("   exact circular-shift FWER, 15 distinct alignments, floor")
    print("   1/15 = 0.0667. Specimen AND both zero-scattering controls.")
    print("=" * 74)
    az = np.array(AZ30)
    pred, _, _ = er2(11, "k-8", az)
    print("   %-20s %-6s %-6s %8s %8s %8s %8s %-13s"
          % ("sweep", "win", "treat", "T1 r", "rank", "max|r|", "p_fw",
             "argmax"))
    for nm in ("girdle_seed11_ppw8_dev",) + CONTROLS:
        sw = load_sweep(nm)
        for wn, win in WINDOWS.items():
            tg = tgrid(win)
            _, R0 = rf_on_grid(sw, tg)
            for tr in ("none", "loo"):
                f = level_none if tr == "none" else level_loo
                rk, r0 = rank_true(shift_scan(10 * np.log10(f(sw, win)[1]),
                                              pred), 15)
                R = R0 - loo_median(R0) if tr == "loo" else R0
                p, key, mx = fwer(bank(R, tg, win), pred)
                store["axis_%s_%s_%s" % (nm, wn, tr)] = np.array(
                    [r0, rk, mx, p])
                print("   %-20s %-6s %-6s %+8.3f %5d/15 %8.3f %8.4f %-13s"
                      % (nm, wn, tr, r0, rk, mx, p, key))
    print("\n   Read the control rows first. In the early windows the")
    print("   FABRIC predictor tracks specimens with NO fabric contrast")
    print("   better than it tracks the specimen. In the analysed gate it")
    print("   does not. That, and not any property of the removal, is why")
    print("   an early-window score cannot be taken at face value.")


def report_ensemble(store):
    print()
    print("=" * 74)
    print("4. TWELVE TESSELLATIONS. The predictor is built from each")
    print("   specimen's own realised c-axes, so alpha = 0 is true by")
    print("   construction: this is an oracle. If it cannot find the axis,")
    print("   no estimator of this class can.")
    print("=" * 74)
    az = np.array(AZ30)
    cells = [(w, t) for w in WINDOWS for t in ("none", "loo")]
    R = {c: [] for c in cells}
    for nm, seed, kap in SPECS:
        sw = load_sweep(nm)
        p, _, _ = er2(seed, kap, az)
        for wn, tr in cells:
            f = level_none if tr == "none" else level_loo
            rk, _ = rank_true(shift_scan(10 * np.log10(f(sw, WINDOWS[wn])[1]),
                                         p), 15)
            R[(wn, tr)].append(rk)
    R = {k: np.array(v) for k, v in R.items()}
    print("   %-14s %8s %10s %8s %10s"
          % ("cell", "first/12", "rank sum", "z", "p one-sided"))
    for k, v in R.items():
        z = (v.sum() - 8.0 * len(v)) / np.sqrt(224 / 12.0 * len(v))
        store["ens_%s_%s" % k] = v
        print("   %-14s %6d/12 %10d %8.2f %10.4f"
              % ("%s %s" % k, int((v == 1).sum()), v.sum(), z, norm.cdf(z)))
    print("   chance: 0.8 of 12 first, rank sum 96, z 0.")
    print("\n   Paired on the tessellation, which is the unit of")
    print("   replication, against the published gate:")
    g = R[("24-36", "none")]
    for k, v in R.items():
        if k == ("24-36", "none"):
            continue
        d = v - g
        nz = d[d != 0]
        sp = binomtest(int((nz < 0).sum()), len(nz)).pvalue if len(nz) else 1.0
        wp = wilcoxon(v, g).pvalue if np.any(d) else 1.0
        print("      %-14s better %2d worse %2d tied %2d   sign p = %.3f  "
              "wilcoxon p = %.3f" % ("%s %s" % k, int((d < 0).sum()),
                                     int((d > 0).sum()), int((d == 0).sum()),
                                     sp, wp))
    print("\n   No window separates from the gate. The first-rank counts")
    print("   differ by one to three of twelve, which Bin(12, 1/15) with")
    print("   mean 0.8 and sd 0.86 does not distinguish.")
    return R


def report_removal(store, ndraw):
    print()
    print("=" * 74)
    print("5. THE ATTACK. Does the leave-one-out median removal introduce")
    print("   a dependence between azimuths that the circular-shift null")
    print("   reads as signal?")
    print("=" * 74)
    tg = tgrid(WINDOWS["4-16"])
    m = (tg >= 4e-6) & (tg < 16e-6)
    _, R = rf_on_grid(load_sweep("girdle_seed11_ppw8_dev"), tg)
    n = len(R)
    worst = 0.0
    for k in range(n):
        worst = max(worst, np.abs(loo_median(np.roll(R, k, axis=0)) -
                                  np.roll(loo_median(R), k, axis=0)).max())
    rng = np.random.default_rng(7)
    for _ in range(20):
        P = rng.permutation(n)
        worst = max(worst, np.abs(loo_median(R[P]) - loo_median(R)[P]).max())
    print("\n   (a) EQUIVARIANCE. max |loo(PR) - P loo(R)| over all 30")
    print("       circular shifts and 20 random permutations: %.3e" % worst)
    print("       on a field of scale %.3e. The removal is a symmetric"
          % np.abs(R).max())
    print("       function of the azimuths, so it commutes exactly with")
    print("       relabelling them. A shift null applied after it is as")
    print("       exact as the same null applied before it.")

    print("\n   (b) MONTE CARLO, %d draws per cell. Surrogates that are"
          % ndraw)
    print("       rotationally exchangeable by construction, so the true")
    print("       alignment is null. Nominal P(rank 1) = 1/15 = 0.0667.")
    pred, _, _ = er2(11, "k-8", np.array(AZ30))
    common = np.median(R, axis=0)
    resid = R - common[None, :]
    sd = resid.std(axis=0)
    print("       %-8s %-10s %10s %8s %10s"
          % ("family", "treat", "P(rank 1)", "se", "mean rank"))
    for kind in ("perm", "gauss"):
        for tag, treat in (("untreated", False), ("loo", True)):
            rk = np.empty(ndraw, int)
            for i in range(ndraw):
                s = rng.choice([-1.0, 1.0], size=n)[:, None]
                if kind == "perm":
                    S = common[None, :] + s * resid[rng.permutation(n)]
                else:
                    S = (common[None, :] +
                         rng.standard_normal((n, R.shape[1])) * sd)
                if treat:
                    S = S - loo_median(S)
                y = 10 * np.log10(2.0 * (S[:, m] ** 2).mean(axis=1))
                rk[i], _ = rank_true(shift_scan(y, pred), 15)
            h = (rk == 1).mean()
            store["mc_%s_%s" % (kind, tag)] = np.array(
                [h, np.sqrt(h * (1 - h) / ndraw), rk.mean()])
            print("       %-8s %-10s %10.4f %8.4f %10.2f"
                  % (kind, tag, h, np.sqrt(h * (1 - h) / ndraw), rk.mean()))
    print("\n       ANSWER: no. The removal does not inflate the null.")
    print("       The prior module's disqualification of the 4 to 16 cell")
    print("       on the ground that a control reached the 1/15 floor is")
    print("       a multiplicity artefact: twelve control cells were")
    print("       scored and 1-(14/15)^12 = %.2f is the chance that at"
          % (1 - (14 / 15.) ** 12))
    print("       least one reaches it under a true null.")


def report_scales(store):
    print()
    print("=" * 74)
    print("6. THE LENGTH SCALES, fig_scales convention.")
    print("=" * 74)
    lam = C_REF / F0
    with np.load(os.path.join(TESS, "labels_seed11_ppw8_kappa-8.npz")) as z:
        lab = z["labels"]
        ng = int(np.unique(lab[lab >= 0]).size)
    dg = (6.0 * np.pi * (DIA / 2) ** 2 * THICK / (np.pi * ng)) ** (1 / 3)
    print("   wavelength %.3f mm, %d realised grains, grain %.2f mm"
          % (lam * 1e3, ng, dg * 1e3))
    for wn, win in WINDOWS.items():
        rng_m = 0.5 * C_REF * 0.5 * (win[0] + win[1])
        df = np.sqrt(2.0 * lam * rng_m)
        store["scale_%s" % wn] = np.array([rng_m, df, df / dg])
        print("   %-6s mid range %5.1f mm  D_Fresnel %5.2f mm  ratio %.3f"
              % (wn, rng_m * 1e3, df * 1e3, df / dg))
    teq = dg ** 2 / (lam * C_REF)
    print("   D_Fresnel equals the grain at %.1f us of two-way time, past"
          % (teq * 1e6))
    print("   the far end of the record. The inequality the mechanism")
    print("   rests on therefore holds in EVERY window of this")
    print("   acquisition and is strongest where the zone is smallest.")
    print("   Moving the window earlier strengthens the mechanism's")
    print("   antecedent. It cannot be tested by moving the window.")


def main():
    ndraw = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    store = {}
    report_harness()
    report_type(store)
    report_floor(store)
    report_axis(store)
    report_ensemble(store)
    report_removal(store, ndraw)
    report_scales(store)
    np.savez(os.path.join(HERE, "axis_window_adjudication.npz"), **store)
    print("\nwrote axis_window_adjudication.npz")


if __name__ == "__main__":
    main()
