"""Are Sec. 5.2's three fabric-axis tests a property of the coda or of the
gate they were measured in?

THE QUESTION. Sec. 5.2 establishes its null with three tests, all in the
24 to 36 us gate, and explains the null by a coincidence of length scales:
the first Fresnel diameter there is 14.9 mm against a 17.2 mm
volume-equivalent grain, a ratio of 0.87, so the beam interrogates about
one boundary at a time. That ratio is 0.50 at 4 to 16 us. If the null
holds there too, the null is right and its stated mechanism is not.

WHAT IS SCORED. The manuscript's own three tests, re-measured in three
windows.

  T1  the azimuthal dependence of the coda level regressed against the
      predicted mean square of dv/2vbar over grain pairs, under the
      circular-shift null.
  T2  a panel of 22 waveform observables against the same predictor,
      family-wise error rate of the largest |r| under the same null.
  T3  the mean coda level between two fabrics carried on a bit-identical
      Laguerre tessellation, eight matched pairs, paired t.

Plus the question the three tests do not answer between them, which is
what the null is actually about: the AXIS ERROR IN DEGREES per specimen,
against that specimen's own volume-weighted sample axis, on the
convention of tab:axisrecovery, so that a coda number can be put beside
the time-of-flight channel's 5.9 degrees.

THE ADMISSIBILITY CONDITION. The 4 to 16 us window is unusable untreated:
analysis/median_trace_removal.py finds the zero-contrast specimen, which
cannot backscatter from a grain boundary at all, identifying its own
tessellation at rank 2 of 48 there. A leave-one-out azimuthal median trace
removal fixes it. That removal can also manufacture azimuthal structure,
and in four windows it LIFTS a control rather than suppressing it, so
every cell here carries girdle_seed11_ppw8_uniform_axis and girdle_seed11_ppw8_contrast_f000 beside the
specimen, in every window, under every treatment. A result that improves
in the early window while its zero-scattering control also improves is
reading the removal.

WINDOWS AND TREATMENTS, and why each pairing.

  24-36 us  the published gate. Reported untreated only. The removal is
            NOT applied here: median_trace_removal measures it lifting
            the zero-contrast control from 28 of 48 to 5 of 48 in this
            very gate, so applying it to the published gate would be
            applying a procedure already caught manufacturing structure
            in it.
  10-22 us  admissible untreated. Reported both ways, so that any early
            effect can be seen with and without the removal in the loop.
  4-16 us   inadmissible untreated. Reported both ways for the record,
            but only the treated row is evidence, and only if the two
            controls do not move with the specimen.

ESTIMATORS, and why there are two of them everywhere. |hilbert| of a
whole record is not a local operator and Sec. sec:window measures 53.7
per cent of the apparent envelope power at 10 to 22 us arriving from
outside that window. The two admissible estimators it names are used
here, both in every window:

  L_tf     the mean square of the band-passed trace over the window, with
           no transform at all. E|z|^2 = 2 E x^2 for a locally stationary
           segment, so this is an envelope power without an envelope.
  L_env|t  the trace tapered to the window with 2 us raised-cosine skirts
           BEFORE the Hilbert transform, then the mean square of the
           envelope. Cannot import energy from outside [lo-pad, hi+pad].

The published-gate validation additionally runs the published paths
themselves, untapered and at each azimuth's own sample rate, because a
reimplementation that does not return the published number is worthless.

WHAT IS MEASURED AND WHAT IS INFERRED.
  MEASURED   every number printed by report_validate, report_t1_t2,
             report_t3, report_axis and the Fresnel column of
             report_scales.
  INFERRED   nothing. The two remarks in report_scales that draw a
             conclusion from the measured ratios are labelled where they
             are made.

READS, all read-only
  out/sweeps/<name>/az*.npz            trace and dt, nothing else
  out/tesscache/tess_s<seed>_p8_k-8.npz  labels and axes, for the E[R^2]
                                       predictor of T1 and T2
  analysis/grain_axes_volumes_s<seed>_k-8_a1.000.npz  axes and grain volumes,
                                       for the sample axis of tab:axisrecovery
WRITES stdout, and fabric_axis_windows.npz beside this file.

TOUCHES NO CUDA. No forward model, no DiskSpecimen.build, no label
build: every tessellation quantity comes from a cache and the module
raises rather than build one.
"""
import os
import sys

import numpy as np
from scipy import stats
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SWD = os.path.join(ROOT, "out", "sweeps")
TESS = os.path.join(ROOT, "out", "tesscache")
sys.path.insert(0, os.path.dirname(HERE))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

# forward.py holds the quasi-longitudinal velocity surface of ice and
# imports numpy alone. Nothing else in this module reaches sim/.
import forward as FW                                       # noqa: E402

# Acquisition, Sec. 3 and Table 1. Nothing here is fitted.
C_REF, F0, DIA, THK = 3850.0, 2.0e6, 0.100, 0.035
BAND = (0.8e6, 3.0e6)
LAM = C_REF / F0

PUBLISHED = (24e-6, 36e-6)
EARLY = (4e-6, 16e-6)
MIDDLE = (10e-6, 22e-6)
WINDOWS = (EARLY, MIDDLE, PUBLISHED)

# Common resampling grid, _t2_common.DT_C, and the taper skirt width of
# analysis/median_trace_removal.py. Both copied rather than re-chosen so
# that a cell here is comparable with a cell there.
DT_C = 20e-9
TAPER_PAD = 2e-6
T_MAX = 70e-6

AZ_COMMON = tuple(range(0, 360, 12))

# The eight girdle tessellations and their single-maximum twins. Each
# pair shares a bit-identical Laguerre tessellation because DiskSpecimen
# draws the seed points and weights before it draws any c-axis.
SEEDS = (11, 7, 17, 23, 41, 53, 71, 89)
GIRDLE = {11: "girdle_seed11_ppw8_dev", 7: "girdle_seed7_ppw8_ensemble",
          17: "girdle_seed17_ppw8_ensemble", 23: "girdle_seed23_ppw8_ensemble",
          41: "girdle_seed41_ppw8_ensemble", 53: "girdle_seed53_ppw8_ensemble",
          71: "girdle_seed71_ppw8_ensemble", 89: "girdle_seed89_ppw8_ensemble"}
SINGLE = {11: "singlemax_seed11_ppw8_twin", 7: "singlemax_seed7_ppw8_ensemble",
          17: "singlemax_seed17_ppw8_ensemble", 23: "singlemax_seed23_ppw8_ensemble",
          41: "singlemax_seed41_ppw8_ensemble", 53: "singlemax_seed53_ppw8_ensemble",
          71: "singlemax_seed71_ppw8_ensemble", 89: "singlemax_seed89_ppw8_ensemble"}

# Both zero-scattering controls, on the seed-11 geometry, so the
# tessellation a fabric predictor could match is present and only the
# acoustic contrast across the boundaries is gone.
CONTROLS = (("girdle_seed11_ppw8_uniform_axis", "one c-axis in every grain"),
            ("girdle_seed11_ppw8_contrast_f000", "contrast f = 0.00"))
SPECIMEN = "girdle_seed11_ppw8_dev"

# The four sweeps of the Sec. 5.2 panel, in the manuscript's order.
PANEL_SWEEPS = (("girdle_seed11_ppw6_axis_perp", 11, -8.0, "ppw 6"),
                ("girdle_seed11_ppw8_dev", 11, -8.0, "ppw 8"),
                ("singlemax_seed11_ppw8_12az_check", 11, 3.93, "12-azimuth check"),
                ("isotropic_seed41_ppw6_calibration", 41, 0.001, "isotropic control"))

# Published values this module has to return before anything else it
# prints can be believed. Sec. 5.2 and Sec. sec:window.
REF_T1 = {"girdle_seed11_ppw6_axis_perp": (0.41, 10, 30), "girdle_seed11_ppw8_dev": (0.21, 11, 30)}
REF_T2_FWER = {"girdle_seed11_ppw6_axis_perp": 0.100, "girdle_seed11_ppw8_dev": 0.3667,
               "singlemax_seed11_ppw8_12az_check": 0.6667, "isotropic_seed41_ppw6_calibration": 0.3444}
REF_T2_MAXR = {"girdle_seed11_ppw6_axis_perp": 0.737, "girdle_seed11_ppw8_dev": 0.374}
REF_T3 = dict(delta=-2.86, t=-2.65, p=0.033)
REF_LADDER = {"girdle_seed11_ppw6_axis_perp": -81.09, "girdle_seed11_ppw8_dev": -85.11,
              "girdle_seed11_ppw10_licensing": -87.79}
REF_TOF_AXIS = 5.9          # deg, mean |error| mod 45 over the twelve

_CACHE = {}


# ─────────────────────────────── load ─────────────────────────────────
def _sos(fs):
    return butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                  btype="band", output="sos")


def loo_median(R):
    """Azimuth-common component of R, leaving each azimuth out.

    Row i is the median over the OTHER azimuths at every time sample, so
    no azimuth is ever compared against a statistic containing itself.
    """
    return np.array([np.median(np.delete(R, i, axis=0), axis=0)
                     for i in range(len(R))])


def raised_cosine(t, lo, hi, pad):
    """1 on [lo, hi], cosine skirts of width pad, 0 outside."""
    w = np.zeros_like(t)
    w[(t >= lo) & (t <= hi)] = 1.0
    a = (t > lo - pad) & (t < lo)
    w[a] = 0.5 * (1 - np.cos(np.pi * (t[a] - (lo - pad)) / pad))
    b = (t > hi) & (t < hi + pad)
    w[b] = 0.5 * (1 + np.cos(np.pi * (t[b] - hi) / pad))
    return w


def sweep(name, az_keep=None):
    """Everything one sweep contributes, loaded and resampled once.

    az        azimuths in degrees, ascending
    src2      per-azimuth squared peak of the unfiltered analytic
              envelope, the source reference of tab:reconcile
    e1        per-azimuth backwall envelope peak, the reference the
              22-observable panel uses
    native    list of (band-passed trace, raw trace, dt), each azimuth on
              its OWN sample rate, for the published-path validation
    tg        the common time axis, 0 to 70 us at DT_C
    R         band-passed RF resampled onto tg, one row per azimuth
    Rl        the same with the leave-one-out azimuthal median removed.
              The median is taken at each time sample and so does not
              depend on any window; forming it once on the full record
              and slicing is identical to forming it per window.
    """
    key = (name, az_keep)
    if key in _CACHE:
        return _CACHE[key]
    d = os.path.join(SWD, name)
    az, src2, e1, native, rows = [], [], [], [], []
    tg = np.arange(0.0, T_MAX, DT_C)
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        a = int(f[2:5])
        if az_keep is not None and a not in az_keep:
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        bp = sosfiltfilt(_sos(fs), tr)
        env = np.abs(hilbert(tr))
        k0, w = int(2 * DIA / C_REF * fs), int(2e-6 * fs)
        az.append(a)
        src2.append(env.max() ** 2)
        e1.append(env[max(k0 - w, 0):k0 + w].max())
        native.append((bp, tr, dt))
        rows.append(np.interp(tg, np.arange(len(tr)) * dt, bp))
    o = np.argsort(az)
    R = np.array(rows)[o]
    out = dict(name=name, az=np.array(az)[o], src2=np.array(src2)[o],
               e1=np.array(e1)[o], native=[native[i] for i in o],
               tg=tg, R=R, Rl=R - loo_median(R))
    _CACHE[key] = out
    return out


# ───────────────────────────── predictor ──────────────────────────────
def er2_curve(seed, kappa=-8.0, ppw=8, n=1441):
    """E[R^2] in dB against azimuth, from the CACHED label volume.

    The predictor of Sec. 5.2: the volume-weighted mean square of
    dv/2vbar over grain pairs for a beam along the azimuth, taken over
    every grain in the disc from the actual grain orientations. It sees
    the c-axes only through |axis . n| and so is exactly 180 degree
    periodic; the curve is returned on [0, 180) at 0.125 degree spacing
    and any azimuth is read off it by periodic interpolation.

    Reads out/tesscache and refuses to build: DiskSpecimen.build reaches
    CUDA on its label path.
    """
    key = ("er2", seed, kappa, ppw, n)
    if key in _CACHE:
        return _CACHE[key]
    p = os.path.join(TESS, f"labels_seed{seed}_ppw{ppw:g}_kappa{kappa:g}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"not cached, would reach CUDA: {p}")
    with np.load(p) as z:
        lab, axes = z["labels"], np.asarray(z["axes"], float)
    vol = np.bincount(lab[lab >= 0].ravel(),
                      minlength=len(axes)).astype(float)
    k = vol > 0
    axes, vol = axes[k], vol[k] / vol[k].sum()
    a = np.linspace(0.0, 180.0, n)
    out = np.empty(n)
    for i, deg in enumerate(a):
        t = np.radians(deg)
        nhat = np.array([np.cos(t), np.sin(t), 0.0])
        v = np.interp(np.arccos(np.clip(np.abs(axes @ nhat), 0.0, 1.0)),
                      FW._PSI, FW._VQP)
        vb = float(vol @ v)
        out[i] = 10 * np.log10(float(vol @ (v - vb) ** 2) / (2 * vb ** 2))
    _CACHE[key] = (a, out)
    return a, out


def predictor(seed, az_deg, kappa=-8.0, alpha=0.0, ppw=8):
    """E[R^2] dB at the given azimuths, for a specimen rotated by alpha.

    Rotating the specimen by alpha and looking along azimuth a is the
    same as looking along a - alpha at the unrotated specimen, so the
    trial axis enters as a continuous shift of one curve and no
    tessellation is ever rebuilt.
    """
    grid, curve = er2_curve(seed, kappa, ppw)
    return np.interp(np.mod(np.asarray(az_deg, float) - alpha, 180.0),
                     grid, curve, period=180.0)


# Sweeps whose specimen is not in out/tesscache. isotropic_seed41_ppw6_calibration is seed 41 at
# kappa = 0.001 on the ppw 6 grid, which was never cached as a label
# volume, so its predictor is read from the npz that
# analysis/observable_panel.fabric_pred wrote when the panel was first
# run. It is a stored copy of that build's E[R^2], not a new build:
# rebuilding it would reach CUDA, which this module will not do.
PRED_NPZ = {"isotropic_seed41_ppw6_calibration": "faxwin_fabric_pred_isotropic_seed41_ppw6_calibration.npz"}


def predictor_for(name, seed, kappa, az_deg, ppw=8):
    """The E[R^2] predictor of one sweep, from cache, never built."""
    if name in PRED_NPZ:
        p = os.path.join(HERE, PRED_NPZ[name])
        if not os.path.exists(p):
            raise SystemExit(f"missing stored predictor {p}")
        with np.load(p) as z:
            have = np.mod(np.asarray(z["rots"], float), 180.0)
            er2 = np.asarray(z["er2"], float)
        look = {round(float(a), 6): i for i, a in enumerate(have)}
        idx = [look.get(round(float(a) % 180.0, 6)) for a in az_deg]
        if any(i is None for i in idx):
            raise SystemExit(f"stored predictor for {name} lacks azimuths")
        return er2[np.asarray(idx, int)]
    return predictor(seed, az_deg, kappa, 0.0, ppw)


def sample_axis_deg(seed, kappa=-8.0, ax0=1.000):
    """Volume-weighted sample axis azimuth, tab:axisrecovery's convention.

    The Watson axis is the largest eigenvector of the volume-weighted
    orientation tensor for a single maximum and the smallest for a
    girdle, since the girdle axis is its normal. Read from the CPU
    rebuild cached beside tof_axis_recovery.py, which is the realisation
    tab:axisrecovery scores against.
    """
    p = os.path.join(HERE,
                     f"grain_axes_volumes_s{seed}_k{kappa:g}_a{ax0:.3f}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"not cached, would reach CUDA: {p}")
    with np.load(p) as z:
        axes, vol = np.asarray(z["axes"], float), np.asarray(z["vol"], float)
    w = vol / vol.sum()
    ev, vecs = np.linalg.eigh(np.einsum("g,gi,gj->ij", w, axes, axes))
    order = np.argsort(ev)[::-1]
    v = vecs[:, order][:, 0 if kappa > 0 else 2]
    return float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)


def fold_180(d):
    """Axial angular difference, folded onto [0, 90] degrees."""
    d = abs(float(d)) % 180.0
    return min(d, 180.0 - d)


def fold_45(d):
    """The same folded onto [0, 22.5], tab:axisrecovery's last column."""
    d = abs(float(d)) % 45.0
    return min(d, 45.0 - d)


def two_fold(y, az):
    """Amplitude in units of the series' own sd, and phase, of the 2-theta
    component. The phase is the azimuth at which the cosine peaks, folded
    onto [0, 180), which is an axis and not a direction."""
    a = np.radians(np.asarray(az, float))
    D = np.column_stack([np.ones_like(a), np.cos(2 * a), np.sin(2 * a)])
    c = np.linalg.pinv(D) @ zs(y)
    return (float(np.hypot(c[1], c[2])),
            float(np.degrees(0.5 * np.arctan2(c[2], c[1])) % 180.0))


# ──────────────────────────── measurement ─────────────────────────────
def gate_native(bp, dt, win):
    fs = 1.0 / dt
    return bp[int(win[0] * fs):int(win[1] * fs)]


def level_native(name, win=PUBLISHED, az_keep=AZ_COMMON, ref="src_az"):
    """Per-azimuth level in dB at each azimuth's own sample rate.

    ref 'src_az'   2 <x^2> over the window, referenced to that azimuth's
                   own squared source peak. This is db_reconcile's
                   audited coda_band exactly, and is what tab:reconcile
                   and the eight-pair test are quoted on.
    ref 'src_mean' the mean square of the band-passed ENVELOPE over the
                   window, referenced to the sweep-mean source peak.
                   This is analysis/fabric_channel/mean_level_clean_pair.py's
                   estimator, and is what the T1 regression is quoted on.
    """
    s = sweep(name, az_keep)
    out = []
    for (bp, tr, dt) in s["native"]:
        g = gate_native(bp, dt, win)
        if ref == "src_az":
            out.append(2.0 * float(np.mean(g ** 2)))
        else:
            fs = 1.0 / dt
            e = np.abs(hilbert(bp))[int(win[0] * fs):int(win[1] * fs)]
            out.append(float(np.mean(e ** 2)))
    out = np.array(out)
    if ref == "src_az":
        return 10 * np.log10(out / s["src2"])
    return 10 * np.log10(out / s["src2"].mean())


def window_rf(name, win, removal, az_keep=AZ_COMMON, taper=False):
    """Band-passed RF over one window on the common grid.

    removal None or 'none' subtracts nothing and 'loo' subtracts the
    leave-one-out azimuthal median. taper multiplies by a raised cosine
    with TAPER_PAD skirts BEFORE anything non-local is done to the trace,
    and is only meaningful for the envelope estimator.
    """
    s = sweep(name, az_keep)
    tg = s["tg"]
    R = s["Rl"] if removal == "loo" else s["R"]
    if taper:
        R = R * raised_cosine(tg, win[0], win[1], TAPER_PAD)[None, :]
    g = (tg >= win[0]) & (tg < win[1])
    return s, tg, R, g


def level_window(name, win, removal, az_keep=AZ_COMMON, est="tf"):
    """Per-azimuth level in dB over one window, leakage-safe.

    est 'tf'   mean square of the band-passed trace, no transform at all
    est 'env'  the trace tapered to the window before the Hilbert
               transform, then the mean square of the envelope
    Both are referenced to the per-azimuth squared source peak, which is
    always measured on the unremoved native-rate trace, so the removal
    can never move the normalisation.
    """
    if est == "tf":
        s, tg, R, g = window_rf(name, win, removal, az_keep, taper=False)
        p = 2.0 * np.mean(R[:, g] ** 2, axis=1)
    else:
        s, tg, R, g = window_rf(name, win, removal, az_keep, taper=True)
        p = np.mean(np.abs(hilbert(R, axis=1))[:, g] ** 2, axis=1)
    return 10 * np.log10(p / s["src2"])


# ───────────────────────── the 22 observables ─────────────────────────
def _spectrum(x, dt):
    x = x * np.hanning(len(x))
    n = 1 << (int(np.ceil(np.log2(len(x)))) + 2)
    return np.fft.rfftfreq(n, dt), np.abs(np.fft.rfft(x, n)) ** 2


def observables(an, xc, E1, dt, span):
    """The 22 coda observables of Sec. 5.2, on one analytic segment.

    an   analytic signal restricted to the window
    xc   the real trace restricted to the window
    E1   the backwall envelope peak, the level reference
    span window length in seconds, so per-microsecond rates transfer
    """
    fs = 1.0 / dt
    e = np.abs(an)
    p = e ** 2
    t = np.arange(len(e)) * dt
    o = {}
    rms = np.sqrt(p.mean())
    o["lvl_rms"] = 20 * np.log10(rms / E1)
    im = len(e) // 2
    r_e, r_l = np.sqrt(p[:im].mean()), np.sqrt(p[im:].mean())
    o["lvl_early"] = 20 * np.log10(r_e / E1)
    o["lvl_late"] = 20 * np.log10(r_l / E1)
    o["early_late"] = 20 * np.log10(r_e / r_l)
    k = max(int(0.25e-6 * fs) | 1, 3)
    sm = np.convolve(p, np.ones(k) / k, mode="same")
    ldb = 10 * np.log10(np.maximum(sm, sm.max() * 1e-8))
    o["decay"] = np.polyfit((t - t.mean()) * 1e6, ldb, 1)[0]
    wsum = p.sum()
    tc = (p * t).sum() / wsum
    o["t_centroid"] = tc * 1e6
    o["t_spread"] = np.sqrt((p * (t - tc) ** 2).sum() / wsum) * 1e6
    o["crest"] = 20 * np.log10(e.max() / rms)
    o["env_kurt"] = float(kurtosis(e))
    o["env_skew"] = float(skew(e))
    pn = p / wsum
    o["env_entropy"] = float(-(pn * np.log(pn + 1e-300)).sum()
                             / np.log(len(pn)))
    pk = ((e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:]) & (e[1:-1] > rms)).sum()
    o["n_peaks"] = pk / (span * 1e6)
    fi = np.diff(np.unwrap(np.angle(an))) / (2 * np.pi * dt)
    o["inst_freq"] = float((fi * p[:-1]).sum() / p[:-1].sum() / 1e6)
    f, P = _spectrum(xc, dt)
    m = (f > 0.5e6) & (f < 6e6)
    fm, Pm = f[m], P[m]
    cen = (fm * Pm).sum() / Pm.sum()
    o["f_centroid"] = cen / 1e6
    o["f_bandwidth"] = np.sqrt((Pm * (fm - cen) ** 2).sum() / Pm.sum()) / 1e6

    def band(a, b):
        return P[(f >= a * 1e6) & (f < b * 1e6)].sum()

    o["bandratio_lo"] = 10 * np.log10(band(1.2, 1.8) / band(1.8, 2.4))
    o["bandratio_hi"] = 10 * np.log10(band(2.4, 3.2) / band(1.8, 2.4))
    sl = (f >= 1.2e6) & (f <= 3.2e6)
    o["spec_slope"] = np.polyfit(f[sl] / 1e6,
                                 10 * np.log10(P[sl] + 1e-300), 1)[0]
    fl = (f >= 1.0e6) & (f <= 4.0e6)
    o["spec_flat"] = float(np.exp(np.mean(np.log(P[fl] + 1e-300)))
                           / np.mean(P[fl]))
    lg = np.log(P[fl] + 1e-300)
    lg -= lg.mean()
    C = np.abs(np.fft.rfft(lg * np.hanning(len(lg))))
    dq = 1.0 / (f[fl][-1] - f[fl][0])
    q = np.arange(len(C)) * dq
    qm = (q > 0.3e-6) & (q < 4e-6)
    if qm.sum() > 3:
        j = int(np.argmax(C[qm]))
        o["cep_peak"] = float(C[qm][j] / (C[qm].mean() + 1e-300))
        o["cep_quef"] = float(q[qm][j] * 1e6)
    else:
        o["cep_peak"] = o["cep_quef"] = np.nan
    d = e - e.mean()
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    ac /= ac[0]
    o["acf_width"] = (int(np.argmax(ac < 0.5)) if np.any(ac < 0.5)
                      else len(ac)) / fs * 1e6
    return o


def panel_native(name, win=PUBLISHED, az_keep=None):
    """The published panel path: whole-trace Hilbert of the UNFILTERED
    trace at each azimuth's own sample rate. Not local, and reported only
    at the published gate, where Sec. sec:window measures the imported
    share at 15.6 per cent."""
    s = sweep(name, az_keep)
    rows = []
    for (bp, tr, dt), E1 in zip(s["native"], s["e1"]):
        fs = 1.0 / dt
        an = hilbert(tr)
        i0, i1 = int(win[0] * fs), int(win[1] * fs)
        rows.append(observables(an[i0:i1], tr[i0:i1], E1, dt,
                                win[1] - win[0]))
    keys = sorted(rows[0])
    return s["az"], {k: np.array([r[k] for r in rows]) for k in keys}


def panel_window(name, win, removal, az_keep=AZ_COMMON):
    """The leakage-safe panel: band-passed, resampled, optionally median
    removed, tapered to the window before the transform."""
    s, tg, R, g = window_rf(name, win, removal, az_keep, taper=True)
    A = hilbert(R, axis=1)
    rows = [observables(A[i][g], R[i][g], s["e1"][i], DT_C,
                        win[1] - win[0]) for i in range(len(R))]
    keys = sorted(rows[0])
    return s["az"], {k: np.array([r[k] for r in rows]) for k in keys}


# ──────────────────────────── the nulls ───────────────────────────────
def zs(x):
    sd = np.std(x)
    return (x - np.mean(x)) / (sd if sd > 0 else 1.0)


def n_distinct(az):
    """Distinct alignments of a uniform full circle under np.roll.

    np.roll is a rigid rotation of the specimen only on a uniform full
    circle. The predictor's 180 degree period makes shifts s and s + n/2
    the same alignment, so only n/2 of the n shifts are distinct and n/2
    is the denominator of the p-value.
    """
    a = np.asarray(az, float)
    n = len(a)
    step = 360.0 / n
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    if n % 2 or not np.allclose(gaps, step, atol=1e-6):
        raise ValueError(f"not a uniform even full circle: n={n}")
    return n // 2


def shift_test(y, pred, az):
    """r at the true alignment, its rank, and the number of alignments."""
    nd = n_distinct(az)
    zy, zp = zs(y), zs(pred)
    rs = np.array([float(np.mean(np.roll(zy, s) * zp)) for s in range(nd)])
    return float(rs[0]), int(np.sum(np.abs(rs) >= abs(rs[0]) - 1e-12)), nd


def fwer(X, keys, pred, az):
    """Family-wise error rate of the largest |r| in the panel.

    The max statistic under the same circular-shift null: every
    observable is rolled by the same shift, so the panel's correlation
    structure is preserved and the true alignment s = 0 is one draw.
    """
    nd = n_distinct(az)
    Z = np.column_stack([zs(X[k]) for k in keys])
    zp = zs(pred)
    n = len(zp)
    obs = np.abs(Z.T @ zp / n)
    cyc = np.array([[float(np.mean(np.roll(Z[:, j], s) * zp))
                     for j in range(Z.shape[1])] for s in range(nd)])
    mx = np.abs(cyc).max(1)
    return (float(obs.max()), keys[int(np.argmax(obs))],
            float(np.mean(mx >= obs.max() - 1e-12)), nd)


def axis_fit(y, az, seed, kappa=-8.0, step=0.25):
    """Recovered fabric-axis offset, in degrees, from one channel.

    The trial axis enters as a continuous rotation of the predictor, so
    this is the same estimator the circular-shift null tests, evaluated
    off the 6 or 12 degree azimuth lattice. The correlation is maximised
    SIGNED: the forward model has the level rising with E[R^2], and a
    rotation that anticorrelates is not a candidate axis.

    Returns (alpha at the maximum, r there, r at alpha = 0). The axis
    error against the specimen's own sample axis is fold_180(alpha),
    because alpha = 0 is by construction the true registration of a
    predictor built from that specimen's realised c-axes.
    """
    zy = zs(y)
    al = np.arange(0.0, 180.0, step)
    r = np.array([float(np.mean(zy * zs(predictor(seed, az, kappa, a))))
                  for a in al])
    j = int(np.argmax(r))
    return float(al[j]), float(r[j]), float(np.mean(
        zy * zs(predictor(seed, az, kappa, 0.0))))


# ──────────────────────────── length scales ───────────────────────────
def fresnel(win):
    """First Fresnel diameter at the window mid-range, metres.

    Two-way: a scatterer at transverse offset r lengthens the path by
    2(sqrt(L^2 + r^2) - L), so the first zone ends at sqrt(lambda L / 2)
    and its diameter is sqrt(2 lambda L). Same convention as
    figures/fig_scales.py, including turning the window centre straight
    into a range without taking the source-wavelet delay off first, so
    the published-gate row is the 14.9 mm the paper prints.
    """
    L = 0.5 * C_REF * 0.5 * (win[0] + win[1])
    return float(np.sqrt(2.0 * LAM * L)), float(L)


def grain_diameter(seed=11):
    """Volume-equivalent mean grain diameter of the specimen, metres."""
    p = os.path.join(TESS, f"tess_s{seed}_p8_k-8.npz")
    with np.load(p) as z:
        n = len(z["seeds"])
    v = np.pi * (DIA / 2) ** 2 * THK
    return float((6.0 * v / (np.pi * n)) ** (1 / 3)), n


def win_tag(w):
    return f"{w[0] * 1e6:.0f}-{w[1] * 1e6:.0f}"


# ──────────────────────────── validation ──────────────────────────────
def report_validate():
    """Nothing below this is worth reading until these come back.

    A reimplementation that does not return the published number is
    worthless, so the published gate is scored here through this
    module's own loading, its own predictor, its own null and its own
    estimators, and printed beside Sec. 5.2.
    """
    ok = True
    print("=" * 79)
    print("0. HARNESS. The three tests in the published gate, through this")
    print("   module's own path, beside the values Sec. 5.2 states.")
    print("=" * 79)

    print("\n  T1  coda level against the E[R^2] predictor, circular-shift")
    print("      null. Estimator: mean square of the band-passed envelope")
    print("      in the gate over the sweep-mean source peak, which is")
    print("      analysis/fabric_channel/mean_level_clean_pair.py's.")
    print(f"  {'sweep':<20}{'r':>9}{'ref':>7}{'rank':>10}{'ref':>8}"
          f"{'p':>8}{'ref p':>8}")
    for name, ppw in (("girdle_seed11_ppw6_axis_perp", 6), ("girdle_seed11_ppw8_dev", 8)):
        s = sweep(name)
        y = level_native(name, PUBLISHED, None, ref="src_mean")
        pr = predictor(11, s["az"], -8.0, 0.0, ppw)
        r, rk, nd = shift_test(y, pr, s["az"])
        rr, rrk, rnd = REF_T1[name]
        good = abs(r - rr) < 0.006 and abs(rk - rrk) <= 1 and nd == rnd
        ok &= good
        print(f"  {name:<20}{r:>+9.3f}{rr:>+7.2f}{rk:>7d}/{nd:<2}"
              f"{rrk:>5d}/{rnd:<2}{rk / nd:>8.4f}{rrk / rnd:>8.4f}"
              f"   {'ok' if good else 'MISMATCH'}")
    print("      Sec. 5.2 quotes p = 0.33 and 0.35 for these two ranks;")
    print("      10/30 is 0.3333 and 11/30 is 0.3667, so 0.35 matches")
    print("      neither denominator. The rank and r are what is checked.")

    print("\n  T2  panel of 22 observables, family-wise error rate of the")
    print("      largest |r| under the same null. Published path: whole")
    print("      trace Hilbert, unfiltered, each azimuth's own dt.")
    print(f"  {'sweep':<20}{'n':>5}{'shifts':>7}{'max|r|':>9}{'ref':>7}"
          f"{'p_fw':>9}{'ref':>9}{'  argmax':<14}")
    for name, seed, kappa, tag in PANEL_SWEEPS:
        ppw = 6 if name in ("girdle_seed11_ppw6_axis_perp", "isotropic_seed41_ppw6_calibration") else 8
        az, X = panel_native(name, PUBLISHED, None)
        keys = sorted(X)
        pr = predictor_for(name, seed, kappa, az, ppw)
        mr, arg, pf, nd = fwer(X, keys, pr, az)
        rf = REF_T2_FWER[name]
        good = abs(pf - rf) < 0.002
        if name in REF_T2_MAXR:
            good &= abs(mr - REF_T2_MAXR[name]) < 0.004
        ok &= good
        print(f"  {name:<20}{len(az):>5}{nd:>7}{mr:>9.3f}"
              f"{REF_T2_MAXR.get(name, np.nan):>7.3f}{pf:>9.4f}{rf:>9.4f}"
              f"  {arg:<14}{'ok' if good else 'MISMATCH'}")

    print("\n  T3  eight matched pairs, girdle minus single maximum, on the")
    print("      audited level of tab:reconcile: 2<x^2> over the gate on")
    print("      the trace band-limited before gating, referenced to the")
    print("      source peak, averaged over the revolution as energies.")
    print(f"  {'ppw':<6}{'coda/source dB':>16}{'tab:reconcile':>15}")
    for nm, ppw in (("girdle_seed11_ppw6_axis_perp", 6), ("girdle_seed11_ppw8_dev", 8),
                    ("girdle_seed11_ppw10_licensing", 10)):
        v = revolution_level(nm, PUBLISHED, AZ_COMMON)
        good = abs(v - REF_LADDER[nm]) < 0.02
        ok &= good
        print(f"  {ppw:<6}{v:>16.2f}{REF_LADDER[nm]:>15.2f}"
              f"   {'ok' if good else 'MISMATCH'}")
    g, m = pair_levels(PUBLISHED, AZ_COMMON)
    d = g - m
    t, p = stats.ttest_rel(g, m)
    good = (abs(d.mean() - REF_T3["delta"]) < 0.02
            and abs(t - REF_T3["t"]) < 0.02 and abs(p - REF_T3["p"]) < 0.002)
    ok &= good
    print(f"  eight pairs   delta {d.mean():+.2f} dB (ref "
          f"{REF_T3['delta']:+.2f})   t {t:+.2f} (ref {REF_T3['t']:+.2f})"
          f"   p {p:.4f} (ref {REF_T3['p']:.3f})   "
          f"{'ok' if good else 'MISMATCH'}")
    print("\n   " + ("HARNESS REPRODUCES" if ok
                     else "HARNESS DOES NOT REPRODUCE"))
    return ok


def revolution_level(name, win, az_keep=AZ_COMMON, removal=None, est="tf"):
    """Revolution level in dB, averaged over azimuth as ENERGIES.

    A revolution mean is a quadrature over a closed rotation, not a draw
    from a population, so the average is taken over powers.
    """
    if removal is None:
        lv = level_native(name, win, az_keep, ref="src_az")
    else:
        lv = level_window(name, win, removal, az_keep, est)
    return float(10 * np.log10(np.mean(10.0 ** (lv / 10.0))))


def pair_levels(win, az_keep=AZ_COMMON, removal=None, est="tf"):
    """(girdle, single) revolution levels over the eight matched pairs."""
    g = np.array([revolution_level(GIRDLE[s], win, az_keep, removal, est)
                  for s in SEEDS])
    m = np.array([revolution_level(SINGLE[s], win, az_keep, removal, est)
                  for s in SEEDS])
    return g, m


# ─────────────────────── the three tests, by window ───────────────────
# Every cell is (window, treatment). The published gate is reported
# untreated only: median_trace_removal measures the removal LIFTING the
# zero-contrast control from 28 of 48 to 5 of 48 in that gate, so
# applying it there would be applying a procedure already caught
# manufacturing structure in the very window it would be judged in.
CELLS = ((EARLY, "none"), (EARLY, "loo"),
         (MIDDLE, "none"), (MIDDLE, "loo"),
         (PUBLISHED, "none"))

# Specimen first, then both zero-scattering controls. Every one of them
# carries the seed-11 geometry, so all three are scored against the SAME
# seed-11 girdle predictor: a control that tracks that predictor is
# tracking something that is not grain-boundary scattering, because it
# has none.
TRIO = ((SPECIMEN, "specimen, girdle k = -8"),
        (CONTROLS[0][0], "CONTROL, " + CONTROLS[0][1]),
        (CONTROLS[1][0], "CONTROL, " + CONTROLS[1][1]))


def report_t1_t2():
    """T1 and T2 in the three windows, with both controls in every cell."""
    print("\n" + "=" * 79)
    print("1. T1, the coda level against the E[R^2] predictor, and T2, the")
    print("   22-observable panel, in three windows. All three sweeps on the")
    print("   30 azimuths every ppw 8 sweep shares, so the shift null has 15")
    print("   distinct alignments everywhere and the floor is 1/15 = 0.067.")
    print("   Two leakage-safe estimators, quoted side by side in every")
    print("   window: 'tf' is the mean square of the band-passed trace with")
    print("   no transform at all, 'env' tapers to the window before the")
    print("   Hilbert transform. The panel is on the tapered envelope.")
    print("=" * 79)
    out = {}
    for name, note in TRIO:
        print(f"\n  {name}   {note}")
        print(f"  {'window':<9}{'treat':<7}"
              f"{'T1 r tf':>10}{'rank':>7}{'T1 r env':>10}{'rank':>7}"
              f"{'T2 max|r|':>11}{'p_fw':>8}  {'argmax':<13}")
        for win, treat in CELLS:
            s = sweep(name, AZ_COMMON)
            pr = predictor(11, s["az"], -8.0, 0.0, 8)
            row = [win_tag(win) + " us", treat]
            for est in ("tf", "env"):
                y = level_window(name, win, treat, AZ_COMMON, est)
                r, rk, nd = shift_test(y, pr, s["az"])
                row += [r, f"{rk}/{nd}"]
                out[(name, win_tag(win), treat, "T1" + est)] = (r, rk, nd)
            az, X = panel_window(name, win, treat, AZ_COMMON)
            keys = sorted(X)
            mr, arg, pf, nd = fwer(X, keys, pr, az)
            out[(name, win_tag(win), treat, "T2")] = (mr, arg, pf, nd)
            print(f"  {row[0]:<9}{row[1]:<7}{row[2]:>+10.3f}{row[3]:>7}"
                  f"{row[4]:>+10.3f}{row[5]:>7}{mr:>11.3f}{pf:>8.4f}"
                  f"  {arg:<13}")
    print("\n  Read the controls first. A cell in which a control's rank")
    print("  improves alongside the specimen's is reading the removal or")
    print("  the rendering, not the scattering, whatever the specimen does.")
    return out


def report_t3():
    """T3 in the three windows, with the zero-scattering floor beside it."""
    print("\n" + "=" * 79)
    print("2. T3, the fabric-type separation. Eight bit-identical")
    print("   tessellation pairs, girdle minus single maximum, paired t on")
    print("   7 degrees of freedom. This is the test with no model in it.")
    print("=" * 79)
    out = {}
    for est in ("tf", "env"):
        print(f"\n  estimator '{est}'")
        print(f"  {'window':<9}{'treat':<7}{'gir-sin dB':>12}{'sd':>8}"
              f"{'t':>8}{'p':>9}{'agree':>7}   {'95 % interval':<20}")
        for win, treat in CELLS:
            g, m = pair_levels(win, AZ_COMMON, treat, est)
            d = g - m
            t, p = stats.ttest_rel(g, m)
            se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            h = float(stats.t.ppf(0.975, len(d) - 1)) * se
            ag = int(np.sum(np.sign(d) == np.sign(d.mean())))
            out[(win_tag(win), treat, est)] = (float(d.mean()), float(t),
                                               float(p), d.copy())
            print(f"  {win_tag(win) + ' us':<9}{treat:<7}{d.mean():>+12.2f}"
                  f"{np.std(d, ddof=1):>8.2f}{t:>8.2f}{p:>9.4f}"
                  f"{ag:>5}/8   [{d.mean() - h:+6.2f},{d.mean() + h:+6.2f}]")
        print("  per-pair differences, dB   "
              + " ".join(f"{'s%d' % s:>7}" for s in SEEDS))
        for win, treat in CELLS:
            d = out[(win_tag(win), treat, est)][3]
            print(f"    {win_tag(win) + ' ' + treat:<21}"
                  + " ".join(f"{v:+7.2f}" for v in d))

    print("\n  THE ZERO-SCATTERING FLOOR, which decides whether the window")
    print("  can carry a level test at all. Both controls cannot backscatter")
    print("  from a grain boundary, so the margin below is how far the")
    print("  specimen's coda stands above a record with no coda in it. A")
    print("  window whose margin has collapsed is measuring the front")
    print("  arrival and the rendering, and a fabric contrast measured there")
    print("  is not a fabric contrast.")
    for est in ("tf", "env"):
        print(f"\n  estimator '{est}', revolution level dB re source")
        print(f"  {'window':<9}{'treat':<7}{'specimen':>10}"
              f"{'zerocontr':>11}{'margin':>8}{'cs f=0':>10}{'margin':>8}"
              f"{'null pair':>11}")
        for win, treat in CELLS:
            sp = revolution_level(SPECIMEN, win, AZ_COMMON, treat, est)
            zc = revolution_level(CONTROLS[0][0], win, AZ_COMMON, treat, est)
            cs = revolution_level(CONTROLS[1][0], win, AZ_COMMON, treat, est)
            out[("floor", win_tag(win), treat, est)] = (sp, zc, cs)
            print(f"  {win_tag(win) + ' us':<9}{treat:<7}{sp:>10.2f}"
                  f"{zc:>11.2f}{sp - zc:>8.2f}{cs:>10.2f}{sp - cs:>8.2f}"
                  f"{zc - cs:>11.2f}")
    print("\n  'null pair' is zerocontrast minus cs f = 0: two records with")
    print("  the same geometry and no scattering in either, so it is what")
    print("  this estimator returns for a difference that must be zero.")

    print("\n  THE TWO CONFIRMATORY SHAPE CHANNELS. Sec. sec:window reports")
    print("  the bandwidth falling from p = 0.0098 to 0.062 and the decay")
    print("  rate reversing sign between the gate and 10 to 22 us. They are")
    print("  taken here from the same panel as T2, so they are this module's")
    print("  estimator and not observable_matrix's, and only the direction")
    print("  and the p are comparable with the manuscript.")
    for key, unit in (("decay", "dB/us"), ("f_bandwidth", "MHz")):
        print(f"\n  {key}, {unit}")
        print(f"  {'window':<9}{'treat':<7}{'girdle':>10}{'single':>10}"
              f"{'gir-sin':>10}{'t':>8}{'p':>9}{'agree':>7}")
        for win, treat in CELLS:
            g = np.array([np.mean(panel_window(GIRDLE[s], win, treat,
                                               AZ_COMMON)[1][key])
                          for s in SEEDS])
            m = np.array([np.mean(panel_window(SINGLE[s], win, treat,
                                               AZ_COMMON)[1][key])
                          for s in SEEDS])
            d = g - m
            t, p = stats.ttest_rel(g, m)
            out[("shape", key, win_tag(win), treat)] = (float(d.mean()),
                                                        float(t), float(p))
            print(f"  {win_tag(win) + ' us':<9}{treat:<7}{g.mean():>10.3f}"
                  f"{m.mean():>10.3f}{d.mean():>+10.3f}{t:>8.2f}{p:>9.4f}"
                  f"{int(np.sum(np.sign(d) == np.sign(d.mean()))):>5}/8")
    return out


# ───────────────────── the axis, in degrees, per specimen ─────────────
AXIS_SPECIMENS = (
    [(GIRDLE[s], s, -8.0, 1.000, "girdle") for s in SEEDS]
    + [(SINGLE[s], s, 3.93, 0.866, "single") for s in (11, 17, 23, 41)])


def report_axis():
    """Axis error in degrees, tab:axisrecovery's convention, from the coda.

    The estimator is T1's, taken off the azimuth lattice: the trial axis
    rotates the predictor continuously and the signed correlation is
    maximised. It is an ORACLE estimator, and generously so. The
    predictor is built from that specimen's realised c-axes, so alpha = 0
    is the true registration by construction and the error below is the
    error of an estimator that already knows the specimen's own
    orientation distribution and has only to find its rotation.
    tab:axisrecovery's time-of-flight estimator knows a Watson template
    and nothing else. If the oracle cannot recover the axis, no estimator
    of this class can.
    """
    print("\n" + "=" * 79)
    print("3. THE AXIS ITSELF, in degrees, against each specimen's own")
    print("   volume-weighted sample axis, exactly as tab:axisrecovery")
    print("   scores the time-of-flight channel. Recovering the fabric AXIS")
    print("   is not the same as separating two fabric TYPES, and only this")
    print("   table is about the axis.")
    print("=" * 79)
    print("   chance for an axial angle folded onto [0, 90] is a mean of")
    print(f"   45.0 deg; the time-of-flight channel reaches "
          f"{REF_TOF_AXIS:.1f} deg over the same twelve specimens.")
    out = {}
    for win, treat in CELLS:
        print(f"\n  window {win_tag(win)} us, treatment '{treat}'")
        print(f"  {'sweep':<22}{'sample':>8}{'alpha':>8}{'err':>7}"
              f"{'mod45':>7}{'r':>7}{'err env':>9}{'shift rank':>12}"
              f"{'A2 meas':>9}{'2th err':>9}")
        errs, e45, first, phi = [], [], 0, []
        for name, seed, kappa, ax0, fam in AXIS_SPECIMENS:
            s = sweep(name, AZ_COMMON)
            sa = sample_axis_deg(seed, kappa, ax0)
            y = level_window(name, win, treat, AZ_COMMON, "tf")
            al, r, r0 = axis_fit(y, s["az"], seed, kappa)
            ye = level_window(name, win, treat, AZ_COMMON, "env")
            ale = axis_fit(ye, s["az"], seed, kappa)[0]
            pr = predictor(seed, s["az"], kappa, 0.0, 8)
            _, k_, nd = shift_test(y, pr, s["az"])
            first += int(k_ == 1)
            a2, ph = two_fold(y, s["az"])
            _, php = two_fold(pr, s["az"])
            d2 = fold_180(ph - php)
            errs.append(fold_180(al))
            e45.append(fold_45(al))
            phi.append(d2)
            out[(name, win_tag(win), treat)] = (sa, al, fold_180(al), r,
                                                k_, nd, a2, d2)
            print(f"  {name:<22}{sa:>8.2f}{al:>8.2f}{fold_180(al):>7.2f}"
                  f"{fold_45(al):>7.2f}{r:>+7.3f}{fold_180(ale):>9.2f}"
                  f"{('%d/%d' % (k_, nd)):>12}{a2:>9.3f}{d2:>9.2f}")
        e, e4, p2 = np.array(errs), np.array(e45), np.array(phi)
        print(f"    mean |error| over the twelve: {e.mean():6.2f} deg"
              f"   median {np.median(e):5.2f}   worst {e.max():5.2f}"
              f"   chance 45.0")
        print(f"    mean error mod 45           : {e4.mean():6.2f} deg"
              f"   chance 11.25   tab:axisrecovery {REF_TOF_AXIS:.1f}")
        print(f"    true alignment first of {nd} in {first} of 12 specimens"
              f"   (Sec. 5.2 reports one or two of twelve in this gate)")
        print(f"    2-theta phase of the level against the predictor's:"
              f" mean {p2.mean():6.2f} deg, which is what the alpha fit is")
        for cname, note in CONTROLS:
            s = sweep(cname, AZ_COMMON)
            y = level_window(cname, win, treat, AZ_COMMON, "tf")
            al, r, r0 = axis_fit(y, s["az"], 11, -8.0)
            a2, _ = two_fold(y, s["az"])
            print(f"    CONTROL {cname:<20} alpha {al:6.2f}"
                  f"  err {fold_180(al):5.2f} deg  r {r:+.3f}"
                  f"  A2 {a2:.3f}   ({note})")
    print("\n  A control's error is a draw from the chance distribution and")
    print("  is printed so that the specimens' errors can be read against")
    print("  something measured rather than against 45.0 asserted.")
    print("  The last two columns say what the estimator is doing: the")
    print("  predictor is dominated by its 2-theta term, so the alpha fit")
    print("  is a 2-theta phase match, and a 2-theta in the coda LEVEL is")
    print("  what a bulk velocity anisotropy produces whether or not any")
    print("  grain boundary scatters. That is the channel Sec. 5.2 already")
    print("  attributes to the phase, and the coherent arrival measures it")
    print("  directly. The column is here so the alpha errors are not read")
    print("  as grain-boundary axis recovery without argument.")
    return out


def report_scales():
    """The length-scale argument, in every window that was scored."""
    print("\n" + "=" * 79)
    print("4. LENGTH SCALES. First Fresnel diameter at the window mid-range")
    print("   against the volume-equivalent grain diameter.")
    print("=" * 79)
    dg, n = grain_diameter()
    print(f"   wavelength {LAM * 1e3:.3f} mm; {n} grains realised in the")
    print(f"   seed-11 tessellation; grain diameter {dg * 1e3:.1f} mm")
    print(f"  {'window':<10}{'mid range':>12}{'D_Fresnel':>12}"
          f"{'D_grain':>10}{'ratio':>8}")
    for w in WINDOWS:
        d, L = fresnel(w)
        print(f"  {win_tag(w) + ' us':<10}{L * 1e3:>10.1f}mm"
              f"{d * 1e3:>10.1f}mm{dg * 1e3:>8.1f}mm{d / dg:>8.2f}")
    print("\n   The ratio at the published gate is the coincidence the null's")
    print("   stated mechanism rests on. It is not a coincidence at 4 to 16")
    print("   us: the two lengths differ by a factor of two there. Whether")
    print("   that matters is decided by sections 1 to 3 above and not by")
    print("   this table.")


def main():
    if not report_validate():
        raise SystemExit("harness does not reproduce the published gate")
    t12 = report_t1_t2()
    t3 = report_t3()
    ax = report_axis()
    report_scales()
    np.savez(os.path.join(HERE, "fabric_axis_windows.npz"),
             cells=np.array([f"{win_tag(w)}|{t}" for w, t in CELLS]),
             t1t2=np.array(sorted(f"{k}={v}" for k, v in t12.items())),
             t3=np.array(sorted(f"{k}={v}" for k, v in t3.items())),
             axis=np.array(sorted(f"{k}={v}" for k, v in ax.items())))


if __name__ == "__main__":
    main()
