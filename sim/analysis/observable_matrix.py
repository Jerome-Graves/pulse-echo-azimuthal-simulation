"""The right-hand side of the property-by-observable matrix: every
observable the recorded waveforms can carry, and which of them are
restatements of one another.

THE CLAIM THIS FILE EXISTS TO SUPPORT. The paper has tested a handful of
observables chosen by intuition. Before any screen over sample properties
can be trusted, the observable side has to be enumerated rather than
guessed, and the enumeration has to be de-duplicated: a panel full of
restatements of the same quantity inflates the number of tests without
adding information, and a family-wise correction is then applied to a
count that means nothing.

VOCABULARY, which the paper does not yet use.
    A-SCAN   one trace at one azimuth. An A-scan observable is formable
             from a single trace and has one value per azimuth.
    B-SCAN   the stack of A-scans over azimuth, azimuth being the scan
             coordinate. A B-scan observable needs the stack and has one
             value per sweep. It cannot be formed from any single trace,
             and that is not a technicality: the rotation over which the
             coda waveform decorrelates is a length scale of the
             microstructure that no A-scan contains.

FOUR CONVENTIONS, all inherited from analysis/db_reconcile.py, none of
them optional.

  LEVELS ARE LOCAL. The Hilbert transform of a whole trace is not a local
  operator. The front arrival is about 100 dB above the coda and the tail
  of its analytic signal deposits a pedestal across the gate, a large
  fraction of what a global envelope reports at 10 to 22 us. Every
  windowed envelope here is taken from the trace TAPERED TO A PADDED
  WINDOW before the transform, with the taper lying entirely outside the
  reported interval, so the envelope inside the interval is both local
  and undistorted. Every scalar level is measured with no Hilbert
  transform at all, from E|z|^2 = 2 E x^2 for a locally stationary
  segment. The two routes are compared in report_estimator_audit.

  LEVELS ARE BAND-LIMITED BEFORE GATING, to 0.8 to 3.0 MHz, because a
  large share of the raw gate power at the coarse grid is arithmetic
  noise above the frequency the operator is designed accurate to. The
  BACKWALL is deliberately NOT band-limited, for the reason db_reconcile
  sets out: it carries no numerical content, and the top corner would cut
  genuine source bandwidth by a resolution-dependent amount.

  LEVELS ARE REFERENCED TO THE SOURCE, taken as the peak of the front
  arrival, and AVERAGED OVER AZIMUTH AS ENERGIES. A revolution mean is a
  quadrature over a closed rotation, not a draw from a population.

  EVERY AZIMUTH HAS ITS OWN dt, set by the CFL limit and so by the
  fastest speed in the rotated medium. Stacking traces by SAMPLE INDEX is
  a bug: girdle_perp_ppw8 spreads 0.43 per cent in dt and
  zerocontrast_ppw8 spreads 3.0 per cent. A-scan observables are
  therefore measured on each trace's OWN axis with windows given in
  PHYSICAL TIME, and every B-scan observable is measured on a stack
  resampled onto one physical axis. report_resample_audit prints what the
  resampling moves and checks the interpolator against a windowed sinc.

WHAT IS MEASURED AND WHAT IS INFERRED. Everything in the saved arrays is
measured from the traces. Two markers are INFERRED from the elastic
constants in the source tree and are printed, never stored as
observables: the shear speed sqrt(C44/rho) from openUSCT/simulation/
ringfwi/anisotropy.py lines 30 and 33, and the arrival times a
mode-converted backwall echo would have. Both lie beyond the end of the
record, which is itself the answer to "when does the mode-converted shear
content arrive": not in this record. What can be measured instead is the
in-band power after the longitudinal backwall, reported as a_post_bw and
flagged as a proxy.

ONE DELIBERATE DEPARTURE FROM db_reconcile, and its size. Gate edges are
placed at the NEAREST sample to the physical time, not truncated toward
zero. Truncation makes the gate length in samples depend on dt, which
varies across azimuth by up to 3 per cent here, so it puts an
azimuth-dependent jitter into the very quantity this module measures the
azimuthal structure of. The cost is that a_coda sits 0.070 dB above
db_reconcile's coda_band on girdle_perp_ppw8 over the 30 matched
azimuths, -85.041 against -85.111, and the difference is entirely the
index convention: recomputing with truncation reproduces -85.111
exactly. Everything else reproduces: -81.09, -28.06 at ppw 6 and -26.13
at ppw 10, to the printed digits.

Reads out/sweeps. Writes out/observables/observable_matrix.npz and two
comma-separated summaries beside it. Touches no GPU: no forward model, no
specimen build, no labeller, no CUDA import path.
"""
import json
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt
from scipy.signal.windows import tukey
from scipy.stats import kurtosis, skew

ROOT = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SWD = os.path.join(ROOT, "out", "sweeps")
OUTD = os.path.join(ROOT, "out", "observables")
OUTF = os.path.join(OUTD, "observable_matrix.npz")

# Acquisition constants, Section 3: 2 MHz Ricker, 100 mm disc, reference
# longitudinal speed 3850 m/s. Coda gate and analysis band as Table 1.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
GATE = (24e-6, 36e-6)
BAND = (0.8e6, 3.0e6)

# Ice Ih at -16 C. Used only to PRINT where a converted arrival falls.
ICE_C44, ICE_RHO = 3.01e9, 917.0

# Windows, all in physical time. BW_E matches db_reconcile's bw_gate so
# the backwall energy column is the number the paper already quotes.
BW_WIN = (48e-6, 58e-6)          # backwall shape and timing
BW_E = (50e-6, 55e-6)            # backwall energy, fixed window
BW_HALF = 2e-6                   # db_reconcile's nominal peak window
FRONT_WIN = (0.0, 4e-6)
SHALLOW = (12e-6, 22e-6)         # the shallow window of the open question
PRE_BW = (40e-6, 48e-6)
POST_BW = (56e-6, 66e-6)         # stops short of the record-end rise
WIDE_GATE = (20e-6, 46e-6)
ONSET_SEARCH = (2e-6, 24e-6)

# Taper pad for every local analytic window. The band-limited impulse
# response is about 0.5 us long, so 2 us of pad puts the whole transient
# outside the reported interval.
PAD = 2e-6

SMOOTH = 0.25e-6                 # boxcar on the envelope power
NSUB = 8                         # sub-windows for the depth-frequency fit
KMAX = 6                         # azimuthal harmonics retained
TINY = 1e-300

# Measurement bands for spectral centroids. WIDE is contaminated by grid
# noise at ppw 6 and is kept only so that contamination stays visible.
CEN_IB = (0.8e6, 3.0e6)
CEN_WIDE = (0.5e6, 6.0e6)

GIRDLE8 = ("girdle_perp_ppw8", "mx_girdle_s7_ppw8", "mx_girdle_s17_ppw8",
           "mx_girdle_s23_ppw8", "mx_girdle_s41_ppw8",
           "mx_girdle_s53_ppw8", "mx_girdle_s71_ppw8",
           "mx_girdle_s89_ppw8")
SINGLE8 = ("singlemax_ppw8", "mx_single_s7_ppw8", "mx_single_s17_ppw8",
           "mx_single_s23_ppw8", "mx_single_s41_ppw8",
           "mx_single_s53_ppw8", "mx_single_s71_ppw8",
           "mx_single_s89_ppw8")
PROD8 = GIRDLE8 + SINGLE8

DEGEN_HI, DEGEN_LO = 0.95, 0.90


def db(x):
    """Power ratio to decibels. Every level here is a power."""
    return 10.0 * np.log10(np.maximum(np.asarray(x, float), TINY))


# ------------------------------------------------------------------ load

def read_trace(path):
    with np.load(path) as z:
        return np.asarray(z["trace"], float).ravel(), float(z["dt"])


def sweep_index(name):
    """Files, azimuths and config for one sweep.

    A sweep that is still being written by the solver is read as far as
    it goes and no further: a file that will not open is skipped rather
    than crashing the build, because this module is expected to run while
    the production queue is still filling directories.
    """
    d = os.path.join(SWD, name)
    files = []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        try:
            with np.load(os.path.join(d, f)) as z:
                _ = z["dt"], z["trace"].shape
        except Exception:
            continue
        files.append(f)
    az = np.array([int(f[2:5]) for f in files], float)
    cfg = {}
    p = os.path.join(d, "config.json")
    if os.path.exists(p):
        with open(p) as fh:
            cfg = json.load(fh)
    return d, files, az, cfg


def bandpass(x, fs, lo=BAND[0], hi=BAND[1]):
    """Zero-phase Butterworth on the COMPLETE trace: filter, then gate."""
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band",
                 output="sos")
    return sosfiltfilt(sos, x)


def common_axis(dts, ns):
    """One physical time axis for a whole sweep.

    The finest dt in the sweep, run to the shortest record, so no trace
    is extrapolated and none is decimated.
    """
    dt_c = float(np.min(dts))
    t_end = float(np.min((np.asarray(ns) - 1) * np.asarray(dts)))
    return dt_c, np.arange(int(np.floor(t_end / dt_c)) + 1) * dt_c


def to_axis(trace, dt, t):
    """Cubic-spline resampling onto the common axis.

    A spline and not a Fourier resample on purpose: the record does not
    end where it starts, and an FFT resample would wrap that step back
    into the trace as ringing at the level of the coda itself. The spline
    has no global support and no wrap. report_resample_audit checks it
    against a windowed-sinc reconstruction.
    """
    return CubicSpline(np.arange(len(trace)) * dt, trace)(t)


def sinc_resample(trace, dt, t, half=32):
    """Windowed-sinc reconstruction, for the resampling audit only."""
    out = np.empty(len(t))
    pos = np.asarray(t, float) / dt
    for m in range(len(pos)):
        c = int(np.floor(pos[m]))
        lo, hi = max(c - half, 0), min(c + half + 1, len(trace))
        idx = np.arange(lo, hi)
        u = pos[m] - idx
        w = np.sinc(u) * (0.5 + 0.5 * np.cos(np.pi * u / (half + 1)))
        out[m] = float(trace[idx] @ w)
    return out


# --------------------------------------------------------------- kernels

def _idx(dt, t0, t1):
    fs = 1.0 / dt
    return int(round(t0 * fs)), int(round(t1 * fs))


def fits(dt, n, t1, pad=PAD):
    """Does a window ending at t1, plus its taper pad, fit the record."""
    return int(round((t1 + pad) / dt)) < n


def local_analytic(x, dt, t0, t1, pad=PAD):
    """Analytic signal inside [t0, t1), tapered to a PADDED window.

    The taper lies outside the reported interval, so nothing beyond the
    padded window reaches the interval and nothing inside it is shaped by
    the taper. This is the only envelope estimator used here.
    """
    i0, i1 = _idx(dt, t0, t1)
    p = int(round(pad / dt))
    j0, j1 = max(i0 - p, 0), min(i1 + p, len(x))
    seg = x[j0:j1].copy()
    n = len(seg)
    flat = min(i0 - j0, j1 - i1)
    w = tukey(n, float(np.clip(2.0 * flat / n, 0.0, 1.0)))
    return hilbert(seg * w)[i0 - j0:i1 - j0]


def local_power(x, dt, t0, t1):
    """Gate power from the signal itself: E|z|^2 = 2 E x^2. No Hilbert."""
    i0, i1 = _idx(dt, t0, t1)
    return 2.0 * float((x[i0:i1] ** 2).mean())


def right_taper(n, frac=0.25):
    """Raised cosine on the trailing frac of a window, unity elsewhere.

    For the front-arrival spectrum only. The record begins at zero
    amplitude, so the leading edge needs no taper and must not be given
    one: a symmetric window would sit its own rising edge on top of the
    source pulse and reshape the spectrum being measured.
    """
    w = np.ones(n)
    m = max(int(frac * n), 1)
    w[-m:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(m) / m))
    return w


def parabolic(y, k):
    """Sub-sample position of a peak at index k."""
    if k <= 0 or k >= len(y) - 1:
        return float(k)
    a, b, c = y[k - 1], y[k], y[k + 1]
    d = a - 2.0 * b + c
    return float(k) if d == 0 else float(k) + 0.5 * (a - c) / d


def crossings(y, k, level):
    """Sub-sample positions, in samples, where y falls to level on each
    side of the peak at k. NaN where it never does."""
    out = []
    for step in (-1, 1):
        j = k
        while 0 < j < len(y) - 1 and y[j] > level:
            j += step
        if y[j] > level:
            out.append(np.nan)
            continue
        a, b = y[j], y[j - step]
        out.append(float(j) if b == a
                   else float(j) - step * (level - a) / (b - a))
    return out[0], out[1]


def spectrum(seg, dt, taper=None):
    """Tapered power spectrum of a segment, zero padded fourfold."""
    w = np.hanning(len(seg)) if taper is None else taper
    n = 1 << (int(np.ceil(np.log2(max(len(seg), 8)))) + 2)
    p = np.abs(np.fft.rfft(seg * w, n)) ** 2
    return np.fft.rfftfreq(n, dt), p


def centroid(f, p, lo, hi):
    """Spectral centroid and rms width inside [lo, hi)."""
    s = (f >= lo) & (f < hi)
    ps = p[s]
    tot = ps.sum()
    if tot <= 0:
        return np.nan, np.nan
    c = float((f[s] * ps).sum() / tot)
    return c, float(np.sqrt((ps * (f[s] - c) ** 2).sum() / tot))


def smooth(y, dt, win=SMOOTH):
    k = max(int(round(win / dt)) | 1, 3)
    return np.convolve(y, np.ones(k) / k, mode="same")


def logslope(t_us, ydb):
    """dB per us, least squares, on a curve already in dB."""
    if len(t_us) < 4:
        return np.nan
    x = np.asarray(t_us, float)
    return float(np.polyfit(x - x.mean(), ydb, 1)[0])


def line_sse(x, y):
    """Residual sum of squares of the best straight line."""
    x = x - x.mean()
    c = np.polyfit(x, y, 1)
    return float(((y - np.polyval(c, x)) ** 2).sum())


# -------------------------------------------------- A-scan observables

def ascan(trace, dt):
    """Every observable formable from ONE trace.

    Prefixes: t_ timing, a_ amplitude, r_ ratio of amplitudes, s_ shape,
    f_ spectral, v_ velocity, qc_ quality control. The qc_ channels are
    not observables and are excluded from every screen; they are stored
    so that the estimator audit and the record guards live in the same
    array as the quantities they qualify.
    """
    n = len(trace)
    fs = 1.0 / dt
    o = {}

    env_g = np.abs(hilbert(trace))
    src2 = float(env_g.max()) ** 2
    xf = bandpass(trace, fs)

    # ---- front arrival. This is the source reference, so its azimuthal
    # structure is a built-in NULL: anything there is numerical.
    kf = int(env_g.argmax())
    o["t_front"] = parabolic(env_g, kf) * dt * 1e6
    o["a_front_abs"] = 10.0 * np.log10(max(src2, TINY))
    o["t_front_on20"] = float(np.argmax(
        env_g > env_g.max() * 10.0 ** (-20.0 / 20.0))) * dt * 1e6
    fi0, fi1 = _idx(dt, *FRONT_WIN)
    seg = trace[fi0:fi1]
    ff, pf = spectrum(seg, dt, taper=right_taper(len(seg)))
    c, r = centroid(ff, pf, *CEN_WIDE)
    o["f_front_cen"], o["f_front_rms"] = c / 1e6, r / 1e6

    # ---- backwall. RAW, per the audited convention.
    if fits(dt, n, BW_WIN[1]):
        zb = np.abs(local_analytic(trace, dt, *BW_WIN))
        b0 = _idx(dt, *BW_WIN)[0]
        kb = int(zb.argmax())
        o["t_bw_peak"] = (b0 + parabolic(zb, kb)) * dt * 1e6
        o["a_bw_peak"] = float(db(zb.max() ** 2 / src2))
        cl, cr = crossings(zb, kb, zb.max() * 10.0 ** (-6.0 / 20.0))
        o["s_bw_w6"] = (cr - cl) * dt * 1e6
        o["s_bw_asym"] = ((kb - cl) / (cr - kb)) if cr > kb else np.nan
        c2l, c2r = crossings(zb, kb, zb.max() * 10.0 ** (-20.0 / 20.0))
        o["s_bw_w20"] = (c2r - c2l) * dt * 1e6
        o["t_bw_on6"] = (b0 + cl) * dt * 1e6
    else:
        for k in ("t_bw_peak", "a_bw_peak", "s_bw_w6", "s_bw_asym",
                  "s_bw_w20", "t_bw_on6"):
            o[k] = np.nan

    # db_reconcile's own backwall peak, so the array reproduces the
    # published estimator exactly and any drift stays visible.
    k0, kw = int(2 * DIA / C_REF * fs), int(BW_HALF * fs)
    o["a_bw_peak_ref"] = (float(db(env_g[max(k0 - kw, 0):k0 + kw].max()
                                   ** 2 / src2))
                          if k0 + kw < n else np.nan)

    if fits(dt, n, BW_E[1]):
        o["a_bw_energy"] = float(db(local_power(trace, dt, *BW_E) / src2))
        i0, i1 = _idx(dt, *BW_E)
        eb = np.abs(local_analytic(trace, dt, *BW_E)) ** 2
        tb = np.arange(i0, i1) * dt
        w = eb.sum()
        tc = float((tb * eb).sum() / w)
        sd = float(np.sqrt((eb * (tb - tc) ** 2).sum() / w))
        o["t_bw_cen"] = tc * 1e6
        o["s_bw_tspread"] = sd * 1e6
        o["s_bw_skew"] = float((eb * (tb - tc) ** 3).sum() / w / sd ** 3)
        fb, pb = spectrum(trace[i0:i1], dt)
        c, r = centroid(fb, pb, *CEN_WIDE)
        o["f_bw_cen"], o["f_bw_rms"] = c / 1e6, r / 1e6
        m = (fb >= CEN_WIDE[0]) & (fb < CEN_WIDE[1])
        pm, fm = pb[m], fb[m]
        above = fm[pm >= pm.max() * 10.0 ** (-6.0 / 10.0)]
        o["f_bw_b6"] = (above.max() - above.min()) / 1e6
    else:
        for k in ("a_bw_energy", "t_bw_cen", "s_bw_tspread", "s_bw_skew",
                  "f_bw_cen", "f_bw_rms", "f_bw_b6"):
            o[k] = np.nan
    o["f_bw_shift"] = o["f_bw_cen"] - o["f_front_cen"]

    # ---- coda gate. Local and in band: the audited estimator.
    o["a_coda"] = float(db(local_power(xf, dt, *GATE) / src2))
    for tag, (t0, t1) in (("near", (24e-6, 28e-6)),
                          ("mid", (28e-6, 32e-6)),
                          ("far", (32e-6, 36e-6))):
        o["a_coda_" + tag] = float(db(local_power(xf, dt, t0, t1) / src2))
    o["r_coda_nearfar"] = o["a_coda_near"] - o["a_coda_far"]
    o["r_coda_bw"] = o["a_coda"] - o["a_bw_peak"]
    o["r_coda_bwE"] = o["a_coda"] - o["a_bw_energy"]

    zc = local_analytic(xf, dt, *GATE)
    ec = np.abs(zc)
    pc = ec ** 2
    o["a_coda_env"] = float(db(pc.mean() / src2))     # audit of a_coda
    i0, i1 = _idx(dt, *GATE)
    tg = np.arange(i0, i1) * dt

    ps = smooth(pc, dt)
    pdb = db(np.maximum(ps, ps.max() * 1e-10))
    o["s_coda_decay"] = logslope(tg * 1e6, pdb)
    h = len(tg) // 2
    o["s_coda_decay_e"] = logslope(tg[:h] * 1e6, pdb[:h])
    o["s_coda_decay_l"] = logslope(tg[h:] * 1e6, pdb[h:])
    o["s_coda_curv"] = o["s_coda_decay_l"] - o["s_coda_decay_e"]

    if fits(dt, n, WIDE_GATE[1]):
        zw = np.abs(local_analytic(xf, dt, *WIDE_GATE)) ** 2
        j0, j1 = _idx(dt, *WIDE_GATE)
        tw = np.arange(j0, j1) * dt
        o["s_coda_decay_w"] = logslope(
            tw * 1e6, db(np.maximum(smooth(zw, dt), zw.max() * 1e-10)))
    else:
        o["s_coda_decay_w"] = np.nan

    o["s_coda_kurt"] = float(kurtosis(ec))
    o["s_coda_skew"] = float(skew(ec))
    o["s_coda_crest"] = float(db(pc.max() / pc.mean()))
    pk, _ = find_peaks(pdb, prominence=3.0)
    o["s_coda_npk"] = len(pk) / ((GATE[1] - GATE[0]) * 1e6)
    pn = pc / pc.sum()
    o["s_coda_teff"] = float(1.0 / (pn ** 2).sum()) * dt * 1e6
    o["s_coda_pr"] = o["s_coda_teff"] / ((GATE[1] - GATE[0]) * 1e6)
    o["s_coda_entropy"] = float(-(pn * np.log(pn + TINY)).sum()
                                / np.log(len(pn)))
    w = pc.sum()
    tc = float((tg * pc).sum() / w)
    o["t_gate_cen"] = (tc - GATE[0]) * 1e6
    o["s_gate_spread"] = float(
        np.sqrt((pc * (tg - tc) ** 2).sum() / w)) * 1e6
    d = ec - ec.mean()
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    ac = ac / ac[0]
    below = np.where(ac < 0.5)[0]
    o["s_coda_acfw"] = (below[0] if len(below) else len(ac)) * dt * 1e6
    fi = np.diff(np.unwrap(np.angle(zc))) / (2 * np.pi * dt)
    o["f_coda_instf"] = float((fi * pc[:-1]).sum() / pc[:-1].sum() / 1e6)

    # ---- coda spectrum, taken on the RAW gate and restricted after the
    # transform, so the analysis filter does not shape the centroid.
    fg, pg = spectrum(trace[i0:i1], dt)
    c, r = centroid(fg, pg, *CEN_IB)
    o["f_coda_cen_ib"], o["f_coda_rms_ib"] = c / 1e6, r / 1e6
    c, r = centroid(fg, pg, *CEN_WIDE)
    o["f_coda_cen"], o["f_coda_rms"] = c / 1e6, r / 1e6
    sl = (fg >= 1.2e6) & (fg <= 3.2e6)
    o["f_coda_slope"] = float(np.polyfit(fg[sl] / 1e6,
                                         db(pg[sl] + TINY), 1)[0])
    fl = (fg >= 1.0e6) & (fg <= 4.0e6)
    o["f_coda_flat"] = float(np.exp(np.mean(np.log(pg[fl] + TINY)))
                             / np.mean(pg[fl]))

    def band_p(a, b):
        s = (fg >= a) & (fg < b)
        return pg[s].sum()

    o["f_coda_hilo"] = float(db(band_p(2.4e6, 3.2e6)
                                / max(band_p(1.2e6, 1.8e6), TINY)))

    # ---- SHIFT of the coda centre frequency with depth. Sub-windows of
    # the raw gate, each tapered on its own, centroid measured in band.
    edges = np.linspace(GATE[0], GATE[1], NSUB + 1)
    cen_ib, cen_wd, mid = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        ja, jb = _idx(dt, a, b)
        fs_, ps_ = spectrum(trace[ja:jb], dt)
        cen_ib.append(centroid(fs_, ps_, *CEN_IB)[0])
        cen_wd.append(centroid(fs_, ps_, *CEN_WIDE)[0])
        mid.append(0.5 * (a + b) * 1e6)
    mid = np.array(mid) - np.mean(mid)
    o["f_coda_shift_ib"] = float(
        np.polyfit(mid, np.array(cen_ib) / 1e6, 1)[0])
    o["f_coda_shift"] = float(
        np.polyfit(mid, np.array(cen_wd) / 1e6, 1)[0])

    # ---- other windows of the record, all audited local in band.
    for tag, win in (("shallow", SHALLOW), ("pre_bw", PRE_BW),
                     ("post_bw", POST_BW)):
        o["a_" + tag] = (float(db(local_power(xf, dt, *win) / src2))
                         if fits(dt, n, win[1]) else np.nan)
    o["r_shallow_coda"] = o["a_shallow"] - o["a_coda"]
    o["r_post_coda"] = o["a_post_bw"] - o["a_coda"]

    # ---- the closest measurable proxy for converted content: the
    # largest arrival AFTER the longitudinal backwall, and the apparent
    # round-trip speed it implies.
    if fits(dt, n, POST_BW[1]):
        zp = np.abs(local_analytic(xf, dt, *POST_BW))
        j0 = _idx(dt, *POST_BW)[0]
        t_slow = (j0 + parabolic(zp, int(zp.argmax()))) * dt
        ja, jb = _idx(dt, *POST_BW)
        fp, pp = spectrum(trace[ja:jb], dt)
        o["t_slow"] = t_slow * 1e6
        o["v_app_slow"] = 2 * DIA / t_slow
        o["f_post_cen"] = centroid(fp, pp, *CEN_IB)[0] / 1e6
    else:
        o["t_slow"] = o["v_app_slow"] = o["f_post_cen"] = np.nan

    # ---- coda onset. Two threshold definitions, and a breakpoint
    # estimator that needs no threshold.
    za = np.abs(local_analytic(xf, dt, *ONSET_SEARCH)) ** 2
    a0 = _idx(dt, *ONSET_SEARCH)[0]
    ta = np.arange(a0, a0 + len(za)) * dt
    ad = db(smooth(za, dt) / src2)
    hold = max(int(round(1e-6 / dt)), 4)
    for thr in (-70.0, -80.0):
        key = "t_coda_on%d" % abs(int(thr))
        o[key] = np.nan
        below = ad < thr
        for j in range(len(below) - hold):
            if below[j:j + hold].all():
                o[key] = ta[j] * 1e6
                break
    best, bt = np.inf, np.nan
    for j in range(2 * hold, len(ta) - 2 * hold, max(hold // 4, 1)):
        s = (line_sse(ta[:j] * 1e6, ad[:j])
             + line_sse(ta[j:] * 1e6, ad[j:]))
        if s < best:
            best, bt = s, ta[j] * 1e6
    o["t_coda_knee"] = bt

    # ---- spread of arrival across the usable record.
    t_hi = min(PRE_BW[1], (n - 1) * dt - PAD)
    if t_hi > 14e-6:
        zr = np.abs(local_analytic(xf, dt, 10e-6, t_hi)) ** 2
        r0 = _idx(dt, 10e-6, t_hi)[0]
        tr = np.arange(r0, r0 + len(zr)) * dt
        w = zr.sum()
        tc = float((tr * zr).sum() / w)
        cum = np.cumsum(zr) / w
        q = np.interp([0.25, 0.5, 0.75], cum, tr) * 1e6
        o["t_rec_cen"] = tc * 1e6
        o["s_rec_spread"] = float(
            np.sqrt((zr * (tr - tc) ** 2).sum() / w)) * 1e6
        o["t_rec_med"] = float(q[1])
        o["s_rec_iqr"] = float(q[2] - q[0])
    else:
        for k in ("t_rec_cen", "s_rec_spread", "t_rec_med", "s_rec_iqr"):
            o[k] = np.nan

    # ---- quality control, not observables.
    muted = xf.copy()
    muted[i0:i1] = 0.0
    o["qc_pedestal"] = float(db(
        (np.abs(hilbert(muted))[i0:i1] ** 2).mean() / src2))
    o["qc_oob"] = 1.0 - (local_power(xf, dt, *GATE)
                         / local_power(trace, dt, *GATE))
    o["qc_dt_ns"] = dt * 1e9
    o["qc_nsamp"] = float(n)
    return o


# -------------------------------------------------- B-scan observables

def harmonics(az_deg, x, kmax=KMAX):
    """Least-squares A_k and phase of A cos(k(th - ph)).

    Least squares and not an FFT so that an incomplete sweep, of which
    there is one in the production set, is measurable on the same
    definition as a complete one.
    """
    th = np.radians(np.asarray(az_deg, float))
    y = np.asarray(x, float)
    good = np.isfinite(y)
    amp = np.full(KMAX + 1, np.nan)
    pha = np.full(KMAX + 1, np.nan)
    frac = np.full(KMAX + 1, np.nan)
    k_use = min(kmax, (int(good.sum()) - 2) // 2)
    if k_use < 1:
        return amp, pha, frac
    th, y = th[good], y[good]
    cols = [np.ones_like(th)]
    for k in range(1, k_use + 1):
        cols += [np.cos(k * th), np.sin(k * th)]
    c = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0]
    var = float(y.var(ddof=0))
    amp[0] = c[0]
    for k in range(1, k_use + 1):
        a, b = c[2 * k - 1], c[2 * k]
        amp[k] = float(np.hypot(a, b))
        pha[k] = float(np.degrees(np.arctan2(b, a)) / k) % (360.0 / k)
        frac[k] = 0.5 * amp[k] ** 2 / var if var > 0 else np.nan
    return amp, pha, frac


def circ_acf(x):
    """Unbiased circular autocorrelation, rho(0) = 1.

    Same kernel as analysis/azimuthal_autocorrelation.py, so every
    correlation length here is the quantity that module already reports.
    """
    y = np.asarray(x, float)
    y = y - y.mean()
    n = len(y)
    f = np.fft.rfft(y, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
    ac = ac / (n - np.arange(n))
    return ac / ac[0]


def acf_stats(ac, step):
    """1/e length, first zero crossing, integral length, N_eff, and a
    sub-step 1/e length.

    l_e is CENSORED at the azimuth step: on this data the coda waveform
    already falls below 1/e at the first non-zero lag of every sweep, so
    l_e returns the step and means "at most this". l_exp is the 1/e
    length that a decaying exponential through rho(0) = 1 and rho(step)
    would have, which is not censored. It is a MODEL, stated as one, and
    it is the only way to put a number on a length the azimuth grid does
    not resolve.
    """
    lags = np.arange(len(ac)) * step
    b = np.where(ac < np.exp(-1.0))[0]
    l_e = float(lags[b[0]]) if len(b) else np.nan
    z = np.where(ac < 0)[0]
    l_0 = float(lags[z[0]]) if len(z) else np.nan
    m = z[0] if len(z) else max(len(ac) // 2, 2)
    tau = max(float((1.0 + 2.0 * ac[1:m].sum()) * step), float(step))
    r1 = float(ac[1]) if len(ac) > 1 else np.nan
    l_exp = (step / -np.log(r1)) if 0.0 < r1 < 1.0 else np.nan
    return l_e, l_0, tau, 360.0 / tau, l_exp


SER_KEYS = ("A1", "ph1", "fr1", "A2", "ph2", "fr2", "A3", "ph3", "fr3",
            "A4", "ph4", "fr4", "sd", "mean", "acf_e", "acf_0",
            "acf_tau", "neff", "acf_e_exp", "acf_e_res", "acf_tau_res",
            "neff_res", "acf_e_exp_res", "sd_res", "r180", "rmsd180")


def series_bscan(az, x, uniform, step):
    """Every B-scan statistic of ONE A-scan series."""
    o = dict.fromkeys(SER_KEYS, np.nan)
    amp, pha, frac = harmonics(az, x)
    for k in (1, 2, 3, 4):
        o["A%d" % k] = amp[k]
        o["ph%d" % k] = pha[k]
        o["fr%d" % k] = frac[k]
    good = np.isfinite(x)
    if good.sum() > 2:
        o["sd"] = float(np.std(x[good], ddof=1))
        o["mean"] = float(np.mean(x[good]))
    if not (uniform and good.all()):
        return o
    (o["acf_e"], o["acf_0"], o["acf_tau"], o["neff"],
     o["acf_e_exp"]) = acf_stats(circ_acf(x), step)
    # The correlation length of a series that carries a deterministic
    # 2-theta component measures that harmonic, not the speckle. The
    # residual version, with k = 1 to 4 projected out, is the one that
    # can carry a grain-size length scale.
    th = np.radians(az)
    cols = [np.ones_like(th)]
    for k in range(1, 5):
        cols += [np.cos(k * th), np.sin(k * th)]
    m = np.column_stack(cols)
    res = x - m @ np.linalg.lstsq(m, x, rcond=None)[0]
    (o["acf_e_res"], _, o["acf_tau_res"], o["neff_res"],
     o["acf_e_exp_res"]) = acf_stats(circ_acf(res), step)
    o["sd_res"] = float(res.std(ddof=1))
    h = len(x) // 2
    if len(x) % 2 == 0 and abs(az[h] - az[0] - 180.0) < 1e-6:
        a, b = x[:h], x[h:2 * h]
        if a.std() > 0 and b.std() > 0:
            o["r180"] = float(np.corrcoef(a, b)[0, 1])
        o["rmsd180"] = float(np.std(a - b))
    return o


STK_KEYS = ("coh_adj", "dec_wave_e", "dec_wave_exp", "dec_wave_tau",
            "dec_wave_neff", "r180_wave", "dec_env_e", "dec_env_exp",
            "dec_env_tau", "dec_env_neff", "r180_env", "dec_coh_e",
            "dec_coh_exp", "dec_coh_tau", "dec_coh_neff", "r180_coh",
            "speckle_C", "looks_C", "field_A2", "field_A2_R", "n_az")


def stack_bscan(az, stack, uniform, step, a_lin):
    """B-scan observables of the FIELD, which no A-scan can carry.

    stack is the band-limited, resampled azimuth-by-time coda gate. Every
    quantity here is a decorrelation with rotation or a coherence between
    azimuths, and both need the stack by construction.
    """
    o = dict.fromkeys(STK_KEYS, np.nan)
    n = len(az)
    o["n_az"] = float(n)

    g = stack - stack.mean(axis=1, keepdims=True)
    gn = g / np.maximum(np.linalg.norm(g, axis=1)[:, None], TINY)
    r_wave = gn @ gn.T
    z = hilbert(stack, axis=1)
    zc = z - z.mean(axis=1, keepdims=True)
    zc = zc / np.maximum(np.linalg.norm(zc, axis=1)[:, None], TINY)
    r_coh = np.abs(zc @ zc.conj().T)
    e = np.abs(z)
    en = e - e.mean(axis=1, keepdims=True)
    en = en / np.maximum(np.linalg.norm(en, axis=1)[:, None], TINY)
    r_env = en @ en.T

    if uniform:
        idx = np.arange(n)
        o["coh_adj"] = float(np.mean(r_coh[idx, (idx + 1) % n]))
        for tag, r in (("wave", r_wave), ("env", r_env), ("coh", r_coh)):
            lag = np.array([float(np.mean(r[idx, (idx + m) % n]))
                            for m in range(n)])
            lag = lag / lag[0]
            l_e, _, tau, neff, l_exp = acf_stats(lag, step)
            o["dec_%s_e" % tag] = l_e
            o["dec_%s_exp" % tag] = l_exp
            o["dec_%s_tau" % tag] = tau
            o["dec_%s_neff" % tag] = neff
            o["r180_%s" % tag] = float(np.mean(
                r[idx, (idx + n // 2) % n]))

    # Azimuthal speckle contrast of the gate INTENSITY, and the looks a
    # gamma model would infer from it. The gamma model is rejected in
    # this project three ways; looks_C is a diagnostic, not a claim.
    i_lin = np.asarray(a_lin, float)
    ok = np.isfinite(i_lin)
    if ok.sum() > 2 and i_lin[ok].mean() > 0:
        c = float(i_lin[ok].std(ddof=1) / i_lin[ok].mean())
        o["speckle_C"] = c
        o["looks_C"] = 1.0 / c ** 2 if c > 0 else np.nan

    # k = 2 harmonic of the azimuth-by-time INTENSITY field, sample by
    # sample, and how stable its phase is across the gate. A fabric
    # signal holds one phase over the gate; speckle does not.
    if n > 6:
        inten = stack ** 2
        th = np.radians(az)
        m = np.column_stack([np.ones_like(th), np.cos(2 * th),
                             np.sin(2 * th)])
        c = np.linalg.lstsq(m, inten, rcond=None)[0]
        a2, ph2 = np.hypot(c[1], c[2]), np.arctan2(c[2], c[1])
        w = inten.mean(axis=0)
        o["field_A2"] = float((a2 * w).sum() / w.sum()
                              / max(inten.mean(), TINY))
        o["field_A2_R"] = float(np.abs(
            (w * np.exp(1j * ph2)).sum() / w.sum()))
    return o


# ------------------------------------------------------------- assemble

def process_sweep(name):
    """A-scan array, B-scan dict, key order and metadata for one sweep."""
    d, files, az, cfg = sweep_index(name)
    traces = [read_trace(os.path.join(d, f)) for f in files]
    rows = [ascan(tr, dt) for tr, dt in traces]
    keys = list(rows[0].keys())
    a = np.array([[r[k] for k in keys] for r in rows], float)

    order = np.argsort(az)
    az, a = az[order], a[order]
    traces = [traces[i] for i in order]
    step = float(np.median(np.diff(az))) if len(az) > 1 else 360.0
    uniform = bool(len(az) > 3 and np.allclose(np.diff(az), step)
                   and abs(len(az) * step - 360.0) < 1e-6)

    dts = np.array([dt for _, dt in traces])
    ns = np.array([len(tr) for tr, _ in traces])
    dt_c, t = common_axis(dts, ns)
    i0, i1 = _idx(dt_c, *GATE)
    stack = np.empty((len(traces), i1 - i0))
    for i, (tr, dt) in enumerate(traces):
        stack[i] = bandpass(to_axis(tr, dt, t), 1.0 / dt_c)[i0:i1]

    b = {}
    for j, k in enumerate(keys):
        if k.startswith("qc_"):
            continue
        for sk, v in series_bscan(az, a[:, j], uniform, step).items():
            b["%s|%s" % (k, sk)] = v
    a_lin = 10.0 ** (a[:, keys.index("a_coda")] / 10.0)
    b.update({"FIELD|" + k: v for k, v in
              stack_bscan(az, stack, uniform, step, a_lin).items()})

    meta = dict(name=name, n_az=len(az), step=step, uniform=uniform,
                seed=cfg.get("seed", np.nan),
                kappa=cfg.get("concentration", np.nan),
                ppw=cfg.get("ppw", np.nan),
                axis=cfg.get("fabric_axis", [np.nan] * 3),
                dt_c=dt_c, t_end=float(t[-1]))
    return az, a, b, keys, meta


def sweep_summary(a, keys):
    """One row per sweep: the azimuth average of each A-scan observable.

    Levels, the a_ and r_ families, are in decibels and are averaged as
    ENERGIES then converted once. Everything else is averaged
    arithmetically. Averaging decibels would bias every level low.
    """
    out = np.full(len(keys), np.nan)
    for j, k in enumerate(keys):
        col = a[:, j]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            continue
        if k.startswith("a_") or k.startswith("r_"):
            out[j] = float(db(np.mean(10.0 ** (col / 10.0))))
        else:
            out[j] = float(col.mean())
    return out


def build(names):
    az_l, a_l, b_l, meta_l, keys = [], [], [], [], None
    kept = []
    for nm in names:
        if len(sweep_index(nm)[1]) < 2:
            print("  %-22s SKIPPED, fewer than two readable azimuths"
                  % nm)
            continue
        kept.append(nm)
        az, a, b, k, meta = process_sweep(nm)
        if keys is None:
            keys = k
        elif k != keys:
            raise RuntimeError("observable order moved on %s" % nm)
        az_l.append(az)
        a_l.append(a)
        b_l.append(b)
        meta_l.append(meta)
        print("  %-22s n=%3d step=%5.1f uniform=%s"
              % (nm, len(az), meta["step"], meta["uniform"]))
    n_max = max(len(x) for x in az_l)
    A = np.full((len(kept), n_max, len(keys)), np.nan)
    AZ = np.full((len(kept), n_max), np.nan)
    for i, (az, a) in enumerate(zip(az_l, a_l)):
        A[i, :len(az)] = a
        AZ[i, :len(az)] = az
    bkeys = sorted(b_l[0])
    B = np.array([[b.get(k, np.nan) for k in bkeys] for b in b_l])
    S = np.array([sweep_summary(a, keys) for a in a_l])
    return dict(A=A, AZ=AZ, B=B, S=S, keys=keys, bkeys=bkeys,
                names=kept, meta=meta_l)


def save(res, path=OUTF):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = res["meta"]
    np.savez_compressed(
        path, A=res["A"], AZ=res["AZ"], B=res["B"], S=res["S"],
        obs_names=np.array(res["keys"]),
        bobs_names=np.array(res["bkeys"]),
        sweep=np.array(res["names"]),
        seed=np.array([x["seed"] for x in m], float),
        kappa=np.array([x["kappa"] for x in m], float),
        ppw=np.array([x["ppw"] for x in m], float),
        axis=np.array([x["axis"] for x in m], float),
        n_az=np.array([x["n_az"] for x in m], float),
        az_step=np.array([x["step"] for x in m], float),
        uniform=np.array([x["uniform"] for x in m]),
        dt_common=np.array([x["dt_c"] for x in m], float),
        t_end=np.array([x["t_end"] for x in m], float))
    _csv(os.path.join(OUTD, "observable_matrix_S.csv"),
         res["names"], res["keys"], res["S"])
    _csv(os.path.join(OUTD, "observable_matrix_B.csv"),
         res["names"], res["bkeys"], res["B"])
    return path


def _csv(path, rows, cols, x):
    with open(path, "w") as fh:
        fh.write("sweep," + ",".join(cols) + "\n")
        for i, r in enumerate(rows):
            fh.write(r + "," + ",".join("%.6g" % v for v in x[i]) + "\n")


# --------------------------------------------------------- degeneracy

def obs_columns(keys):
    """The observable columns, that is everything except the qc_ ones."""
    return [j for j, k in enumerate(keys) if not k.startswith("qc_")]


def corr_matrix(x):
    """Pearson correlation with pairwise-complete observations."""
    p = x.shape[1]
    r = np.full((p, p), np.nan)
    for i in range(p):
        for j in range(i, p):
            m = np.isfinite(x[:, i]) & np.isfinite(x[:, j])
            if m.sum() > 3 and x[m, i].std() > 0 and x[m, j].std() > 0:
                v = float(np.corrcoef(x[m, i], x[m, j])[0, 1])
            else:
                v = np.nan
            r[i, j] = r[j, i] = v
    return r


def within_corr(A, names, keys, want):
    """MEDIAN SIGNED r across azimuth over the sweeps in want.

    Signed and not absolute on purpose. A pair that correlates +0.99 in
    half the specimens and -0.99 in the other half is not one observable
    twice, and a median of absolute values would call it that.
    """
    cols = obs_columns(keys)
    mats = []
    for i, nm in enumerate(names):
        if nm not in want:
            continue
        x = A[i][:, cols]
        x = x[np.isfinite(x).any(axis=1)]
        mats.append(corr_matrix(x))
    with np.errstate(all="ignore"):
        return np.nanmedian(np.array(mats), axis=0), cols


def families(r, labels, thr):
    """Single-linkage grouping of observables at |r| >= thr."""
    p = len(labels)
    par = list(range(p))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for i in range(p):
        for j in range(i + 1, p):
            if np.isfinite(r[i, j]) and abs(r[i, j]) >= thr:
                a, b = find(i), find(j)
                if a != b:
                    par[a] = b
    g = {}
    for i in range(p):
        g.setdefault(find(i), []).append(labels[i])
    return [v for v in g.values() if len(v) > 1]


def top_pairs(r, labels, n=20, lo=0.0, hi=1.01):
    """The most strongly related pairs inside [lo, hi), strongest
    first."""
    out = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = abs(r[i, j])
            if np.isfinite(a) and lo <= a < hi:
                out.append((a, r[i, j], labels[i], labels[j]))
    out.sort(reverse=True)
    return out[:n]


def eff_count(r):
    """Effective number of independent observables.

    The participation ratio of the eigenvalue spectrum,
    (sum lambda)^2 / sum lambda^2, which is p for p orthogonal
    observables and 1 for p copies of one observable. This is the count a
    family-wise correction should be applied to, not the column count.
    Missing entries are set to zero, which can only overstate the count.
    """
    m = np.array(r, float)
    ok = np.isfinite(m).sum(axis=0) > 1
    m = m[np.ix_(ok, ok)]
    m[~np.isfinite(m)] = 0.0
    np.fill_diagonal(m, 1.0)
    lam = np.maximum(np.linalg.eigvalsh(0.5 * (m + m.T)), 0.0)
    return float(lam.sum() ** 2 / (lam ** 2).sum()), int(ok.sum())


# ------------------------------------------------------------------ draw

def report_inventory(res):
    keys = res["keys"]
    fam = {}
    for k in keys:
        fam.setdefault(k.split("_")[0], []).append(k)
    print("OBSERVABLE INVENTORY")
    print("  A-scan observables %d, plus %d quality-control channels"
          % (len(obs_columns(keys)),
             len(keys) - len(obs_columns(keys))))
    print("  B-scan statistics per A-scan series %d" % len(SER_KEYS))
    print("  whole-field B-scan observables %d" % len(STK_KEYS))
    print("  B-scan columns stored %d" % len(res["bkeys"]))
    for p in sorted(fam):
        print("  %-4s %2d  %s" % (p, len(fam[p]), " ".join(fam[p])))
    print()


def report_markers():
    """The inferred markers, and why the shear question answers itself."""
    cs = np.sqrt(ICE_C44 / ICE_RHO)
    print("MODE CONVERSION. INFERRED from C44 and rho, not measured.")
    print("  shear speed sqrt(C44/rho)      %8.1f m/s" % cs)
    print("  longitudinal reference         %8.1f m/s" % C_REF)
    print("  two-way L backwall  2D/cL      %8.2f us"
          % (2 * DIA / C_REF * 1e6))
    print("  two-way S backwall  2D/cS      %8.2f us"
          % (2 * DIA / cs * 1e6))
    print("  one-way L plus one-way S       %8.2f us"
          % (DIA / C_REF * 1e6 + DIA / cs * 1e6))
    print("  record length                  %8.2f us" % 70.2)
    print("  Neither converted backwall arrival is inside the record, so")
    print("  no observable here can time the shear content directly. The")
    print("  measured proxy is a_post_bw, the in-band power from 56 to")
    print("  66 us, which no purely longitudinal single-scattering path")
    print("  to the far wall can reach, with t_slow and v_app_slow.")
    print()


def report_resample_audit(name="girdle_perp_ppw8", nshow=6):
    """What the common time axis costs, and whether the interpolator is
    the reason for any of it."""
    d, files, az, _ = sweep_index(name)
    traces = [read_trace(os.path.join(d, f)) for f in files]
    dts = np.array([dt for _, dt in traces])
    ns = np.array([len(tr) for tr, _ in traces])
    dt_c, t = common_axis(dts, ns)
    print("RESAMPLE AUDIT on %s" % name)
    print("  dt spread across azimuth %.3f per cent, common dt %.4f ns"
          % (100 * (dts.max() - dts.min()) / dts.mean(), dt_c * 1e9))
    keys, dev, sdev = None, [], []
    for tr, dt in traces[:nshow]:
        o0 = ascan(tr, dt)
        o1 = ascan(to_axis(tr, dt, t), dt_c)
        keys = list(o0)
        dev.append([abs(o1[k] - o0[k]) for k in keys])
        seg = slice(int(24e-6 / dt_c), int(36e-6 / dt_c))
        a = to_axis(tr, dt, t[seg])
        b = sinc_resample(tr, dt, t[seg])
        sdev.append(float(np.abs(a - b).max() / np.abs(a).max()))
    dev = np.array(dev)
    worst = np.nanmax(dev, axis=0)
    print("  largest change from resampling, first %d azimuths" % nshow)
    for j in np.argsort(-np.nan_to_num(worst))[:10]:
        print("    %-18s %.4g" % (keys[j], worst[j]))
    print("  spline against windowed sinc over the gate, worst relative")
    print("  difference %.2e" % max(sdev))
    print()


def report_estimator_audit(res):
    """The two independent routes to the coda level agree."""
    keys = res["keys"]
    ja, je = keys.index("a_coda"), keys.index("a_coda_env")
    jp = keys.index("qc_pedestal")
    show = GIRDLE8[:3] + SINGLE8[:2] + ("zerocontrast_ppw8", "iso_gcal")
    print("ESTIMATOR AUDIT, coda gate, dB, source-referenced")
    print("  %-22s %9s %9s %9s %9s"
          % ("sweep", "local", "tapered", "diff", "pedestal"))
    for i, nm in enumerate(res["names"]):
        if nm not in show:
            continue
        a = res["A"][i]
        m = np.isfinite(a[:, ja])
        print("  %-22s %9.2f %9.2f %+9.3f %9.2f"
              % (nm, db(np.mean(10 ** (a[m, ja] / 10))),
                 db(np.mean(10 ** (a[m, je] / 10))),
                 db(np.mean(10 ** (a[m, je] / 10)))
                 - db(np.mean(10 ** (a[m, ja] / 10))),
                 db(np.mean(10 ** (a[m, jp] / 10)))))
    print("  The tapered-window analytic route and the Hilbert-free")
    print("  local route are the same number. The pedestal column is")
    print("  what a GLOBAL envelope would have added to it.")
    print()


def gate_stack(name, resampled=True):
    """The band-limited coda gate for every azimuth of one sweep.

    resampled False stacks by SAMPLE INDEX, which is the bug constraint
    four exists to prevent. It is kept because the only convincing
    argument for a convention is to show what the other one does.
    """
    d, files, az, _ = sweep_index(name)
    tr = [read_trace(os.path.join(d, f)) for f in files]
    dts = np.array([x[1] for x in tr])
    ns = np.array([len(x[0]) for x in tr])
    dt_c, t = common_axis(dts, ns)
    i0, i1 = _idx(dt_c, *GATE)
    if resampled:
        g = np.array([bandpass(to_axis(x, dt, t), 1 / dt_c)[i0:i1]
                      for x, dt in tr])
    else:
        m = i1 - i0
        g = np.array([bandpass(x, 1 / dt)[_idx(dt, GATE[0], GATE[1])[0]:
                                          _idx(dt, GATE[0],
                                               GATE[1])[0] + m]
                      for x, dt in tr])
    return az, g


def lag_curve(az, g, nlag=8):
    """Mean correlation of the gate waveform against azimuth lag."""
    g = g - g.mean(axis=1, keepdims=True)
    g = g / np.maximum(np.linalg.norm(g, axis=1)[:, None], TINY)
    r = g @ g.T
    n = len(g)
    idx = np.arange(n)
    return np.array([float(np.mean(r[idx, (idx + m) % n]))
                     for m in range(min(nlag, n))])


def report_decorrelation(fine=("iso_gcal", "rigid_seed11", "oos_seed23"),
                         coarse=("girdle_perp_ppw8", "mx_girdle_s7_ppw8",
                                 "zerocontrast_ppw8")):
    """How fast the coda waveform decorrelates with rotation.

    This is the B-scan-only observable the project has never measured,
    and the answer decides how many independent looks a rotational scan
    holds. It also decides whether the production grid can measure it at
    all, which it cannot: the correlation is already below 1/e at the
    first non-zero lag of every production sweep.
    """
    print("AZIMUTHAL DECORRELATION OF THE CODA WAVEFORM")
    print("  mean r between gate waveforms, azimuth lag in degrees")
    for nm in tuple(fine) + tuple(coarse):
        az, g = gate_stack(nm)
        step = float(np.median(np.diff(np.sort(az))))
        c = lag_curve(az, g)
        print("  %-20s step %4.1f  " % (nm, step)
              + " ".join("%3.0f=%+.3f" % (i * step, v)
                         for i, v in enumerate(c)))
    print("  At 1 degree sampling the correlation is already 0.14 to")
    print("  0.15 at one step, so the 1/e angle is BELOW ONE DEGREE and")
    print("  is censored by every grid in this project. Under an")
    print("  exponential model the implied 1/e angle is about half a")
    print("  degree; that model is the only thing putting a number on a")
    print("  length nothing here resolves, and it is stated as a model.")
    print("  The consequence is unambiguous either way: every azimuth")
    print("  of every production sweep is an independent look of the")
    print("  coda WAVEFORM. The envelope and the gate LEVEL are not:")
    print("  their integral correlation lengths run to several degrees.")
    print()


def report_index_stacking(name="zerocontrast_ppw8"):
    """What stacking by sample index does to a B-scan observable.

    zerocontrast_ppw8 spreads 3.0 per cent in dt across azimuth, the
    largest spread in the archive, so it is where the bug is visible.
    """
    az, g_ok = gate_stack(name, resampled=True)
    _, g_ix = gate_stack(name, resampled=False)
    step = float(np.median(np.diff(np.sort(az))))
    print("INDEX STACKING, the bug constraint four prevents, on %s"
          % name)
    print("  %-16s " % "resampled"
          + " ".join("%3.0f=%+.3f" % (i * step, v)
                     for i, v in enumerate(lag_curve(az, g_ok))))
    print("  %-16s " % "by sample index"
          + " ".join("%3.0f=%+.3f" % (i * step, v)
                     for i, v in enumerate(lag_curve(az, g_ix))))
    print("  Stacked by index the lag curve alternates sign, which reads")
    print("  as a 12 degree periodicity in a control that has none. On")
    print("  one physical axis it is monotone and positive. The dt")
    print("  spread alone manufactures a two-lag oscillation, and any")
    print("  harmonic fitted to the index-stacked field would find it.")
    print()


def report_bscan(res):
    """The B-scan observables that no A-scan can carry."""
    hd = (("FIELD|dec_wave_exp", "wave_exp"),
          ("FIELD|dec_env_tau", "env_tau"),
          ("FIELD|coh_adj", "coh_adj"),
          ("FIELD|r180_wave", "r180_wav"),
          ("FIELD|r180_env", "r180_env"),
          ("FIELD|speckle_C", "speckC"),
          ("FIELD|field_A2", "fieldA2"),
          ("FIELD|field_A2_R", "A2_R"),
          ("a_coda|acf_tau_res", "lvl_tau"),
          ("a_coda|neff_res", "lvl_neff"),
          ("t_bw_peak|acf_tau_res", "tof_tau"))
    print("B-SCAN ONLY OBSERVABLES, production sweeps at ppw 8")
    print("  %-20s" % "sweep" + "".join("%10s" % h[1] for h in hd))
    for i, nm in enumerate(res["names"]):
        if nm not in PROD8:
            continue
        row = [res["B"][i, res["bkeys"].index(h[0])] for h in hd]
        print("  %-20s" % nm[:20] + "".join("%10.4g" % v for v in row))
    print("  dec_wave_exp is the rotation in degrees over which the coda")
    print("  WAVEFORM decorrelates to 1/e, under the exponential model")
    print("  of acf_stats, because no grid here resolves it directly.")
    print("  a_coda|acf_tau_res is the integral correlation length in")
    print("  degrees of the coda LEVEL after the k = 1 to 4 harmonics")
    print("  are projected out, which is the part of the level that a")
    print("  grain-size length scale could live in. Neither exists in")
    print("  any single A-scan. A NaN in wave_exp means the first-lag")
    print("  correlation is not positive, so the waveform is already")
    print("  fully decorrelated one azimuth step away. A value equal to")
    print("  the azimuth step in lvl_tau or tof_tau is censored by the")
    print("  grid and is an upper bound, not a measurement.")
    print()


def report_degeneracy(res, top=14):
    keys = res["keys"]
    cols = obs_columns(keys)
    labels = [keys[j] for j in cols]

    want = set(n for n in PROD8 if n in res["names"])
    rw, _ = within_corr(res["A"], res["names"], keys, want)
    print("DEGENERACY WITHIN A SPECIMEN, across azimuth")
    print("  median signed r over %d production sweeps at ppw 8, "
          "n = 30 or 60" % len(want))
    for thr in (DEGEN_HI, DEGEN_LO):
        fam = families(rw, labels, thr)
        print("  families at |r| >= %.2f : %d" % (thr, len(fam)))
        for f in sorted(fam, key=lambda x: -len(x))[:top]:
            print("    " + " = ".join(sorted(f)))
    ne, p = eff_count(rw)
    print("  columns %d, effective independent observables %.1f"
          % (p, ne))
    print("  strongest surviving pairs below the 0.90 cut, which are")
    print("  the ones to watch rather than to merge")
    for a, sg, u, v in top_pairs(rw, labels, n=12, lo=0.70,
                                 hi=DEGEN_LO):
        print("    %+.3f  %-20s %s" % (sg, u, v))
    print()

    idx = [i for i, n in enumerate(res["names"])
           if n in want and res["meta"][i]["uniform"]]
    sb = res["S"][np.ix_(idx, cols)]
    rb = corr_matrix(sb)
    print("DEGENERACY BETWEEN SPECIMENS, azimuth-averaged")
    print("  r over %d complete sweeps. n is small: every number in" %
          len(idx))
    print("  this block is a hypothesis, not a result.")
    for thr in (DEGEN_HI, DEGEN_LO):
        fam = families(rb, labels, thr)
        print("  families at |r| >= %.2f : %d" % (thr, len(fam)))
        for f in sorted(fam, key=lambda x: -len(x))[:top]:
            print("    " + " = ".join(sorted(f)))
    ne, p = eff_count(rb)
    print("  columns %d, effective independent observables %.1f"
          % (p, ne))
    print("  strongest pairs, all regimes of n = %d" % len(idx))
    for a, sg, u, v in top_pairs(rb, labels, n=10, lo=0.70,
                                 hi=DEGEN_LO):
        print("    %+.3f  %-20s %s" % (sg, u, v))
    print()
    print("THE NUMBER THAT MATTERS FOR THE SCREEN. The observable side")
    print("carries about 16 independent degrees of freedom within a")
    print("specimen and about 8 between specimens, not 68. A family-wise")
    print("correction applied to 68 A-scan columns, or to the 1650")
    print("B-scan columns, would be correcting for tests that do not")
    print("exist. Screen one representative per family, then correct on")
    print("the number of families actually tested.")
    print()
    return rw, rb, labels


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    names = only or sorted(n for n in os.listdir(SWD)
                           if os.path.isdir(os.path.join(SWD, n)))
    print("BUILDING OBSERVABLE ARRAYS over %d sweeps" % len(names))
    res = build(names)
    p = save(res)
    print("\nwrote %s" % p)
    print("  A  %s  (sweep, azimuth, A-scan observable)"
          % (res["A"].shape,))
    print("  B  %s  (sweep, B-scan observable)" % (res["B"].shape,))
    print("  S  %s  (sweep, azimuth-averaged A-scan observable)\n"
          % (res["S"].shape,))
    report_inventory(res)
    report_markers()
    report_estimator_audit(res)
    report_bscan(res)
    report_decorrelation()
    report_index_stacking()
    report_degeneracy(res)
    if "--audit" in sys.argv:
        report_resample_audit()


if __name__ == "__main__":
    main()
