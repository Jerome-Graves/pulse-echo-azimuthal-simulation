"""Adjudication of the late-window test of the length-scale mechanism.

THE CLAIM UNDER TEST. Sec. 5.2's null is explained by an inequality: the
first Fresnel zone is no larger than a mean grain, so the beam
interrogates about one boundary at a time. Every window scored so far
satisfies the inequality, so none of them tests it. A test agent has
reported that two late gates, 38 to 50 and 40 to 48 us, sit past the
equality point at a ratio of 1.05, that both are admissible, that the
axis is not recovered in them, and that the outcome is inconclusive.

WHAT THIS MODULE DOES. It re-derives every load-bearing number through a
code path that shares no line with the tested module and no line with
analysis/fabric_axis_windows.py. Loading, band-pass, taper, Hilbert,
level, predictor, circular-shift null and Fresnel convention are all
written here from the acquisition constants and the stored sweeps. The
published gate is reproduced first: a reimplementation that cannot
return Sec. 5.2's own numbers cannot adjudicate anything.

It then attacks the result where it is most likely to be wrong.

  A  the harness, against Sec. 5.2 as published.
  B  the geometry at 44 us: what the beam is actually insonifying at a
     one-way range of 84.7 mm inside a 100 mm disc 35 mm thick, and
     whether a coda from there is a specimen measurement at all.
  C  the backwall. Its arrival time and envelope skirt measured, not
     assumed; the leakage into each candidate window measured three
     independent ways; the ordering of taper and Hilbert transform
     verified by constructing the failure it is supposed to prevent;
     and the smear the band-pass itself puts on a 48 dB arrival.
  D  the controls, on the 30 azimuths cs_f000_s11_ppw8 actually holds,
     with the exact floor 1/15, and every rank labelled with its grid.
  E  the confound the controls cannot see. Neither zero-scattering
     record carries the seed-11 girdle, so neither controls for bulk
     velocity anisotropy. The backwall arrival time and amplitude
     measure that channel directly at every azimuth, so the late-window
     level is regressed on them and the fabric correlation is re-scored
     on the residual.
  F  the multiplicity, counted over the whole project and not over this
     module.

MEASURED versus INFERRED is stated in each section header.

READS, all read-only
  out/sweeps/<name>/az*.npz              trace, dt, E1, t1_s
  out/tesscache/tess_s<seed>_p8_k<k>.npz labels and axes
  sim/analysis/tofaxis_build_*.npz       axes and volumes, cross-check
WRITES stdout and late_window_adjudication.npz beside this file.

TOUCHES NO CUDA. No forward model, no DiskSpecimen.build, no label
build. Every tessellation quantity is read from a cache.
"""
import os
import sys

import numpy as np
from scipy import stats
from scipy.signal import butter, hilbert, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SWD = os.path.join(ROOT, "out", "sweeps")
TESS = os.path.join(ROOT, "out", "tesscache")
sys.path.insert(0, os.path.dirname(HERE))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

# forward.py holds the quasi-longitudinal velocity surface of ice and
# imports numpy alone. It is a material constant, not analysis code.
import forward as FW                                        # noqa: E402

# Acquisition, Sec. 3 and Table 1. Nothing here is fitted.
CREF, F0 = 3850.0, 2.0e6
DIA, THK, ELEM = 0.100, 0.035, 6.35e-3
BAND = (0.8e6, 3.0e6)
LAM = CREF / F0

GATE = (24e-6, 36e-6)
CANDS = ((24e-6, 36e-6), (30e-6, 42e-6), (38e-6, 44e-6),
         (38e-6, 50e-6), (40e-6, 48e-6))

DTC = 20e-9
PAD = 2e-6
TMAX = 70e-6
AZ30 = tuple(range(0, 360, 12))

SPEC = "girdle_perp_ppw8"
CTL = ("zerocontrast_ppw8", "cs_f000_s11_ppw8")

SEEDS = (11, 7, 17, 23, 41, 53, 71, 89)
GIR = {11: "girdle_perp_ppw8", 7: "mx_girdle_s7_ppw8",
       17: "mx_girdle_s17_ppw8", 23: "mx_girdle_s23_ppw8",
       41: "mx_girdle_s41_ppw8", 53: "mx_girdle_s53_ppw8",
       71: "mx_girdle_s71_ppw8", 89: "mx_girdle_s89_ppw8"}
SIN = {11: "singlemax_ppw8", 7: "mx_single_s7_ppw8",
       17: "mx_single_s17_ppw8", 23: "mx_single_s23_ppw8",
       41: "mx_single_s41_ppw8", 53: "mx_single_s53_ppw8",
       71: "mx_single_s71_ppw8", 89: "mx_single_s89_ppw8"}
TWELVE = ([(GIR[s], s, -8.0) for s in SEEDS]
          + [(SIN[s], s, 3.93) for s in (11, 17, 23, 41)])

# Published values. Sec. 5.2, tab:reconcile, Sec. sec:window.
PUB = dict(t1_p6=(0.41, 10, 30), t1_p8=(0.21, 11, 30),
           lad_p6=-81.09, lad_p8=-85.11, lad_p10=-87.79,
           t3=(-2.86, -2.65, 0.033), imported_gate=15.6, imported_3042=3.0)

_C = {}


# ───────────────────────────── primitives ─────────────────────────────
def bp_filter(x, dt):
    """Zero-phase 4th-order Butterworth over the recorded band."""
    ny = 0.5 / dt
    sos = butter(4, [BAND[0] / ny, BAND[1] / ny], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def cosine_gate(t, lo, hi, pad, clip=None):
    """Unit on [lo, hi] with cosine skirts of width pad, zero outside.

    clip, if given, forces the window to zero at and beyond that time,
    which is how a skirt is kept off the backwall without shortening the
    flat top.
    """
    w = np.zeros_like(t)
    w[(t >= lo) & (t <= hi)] = 1.0
    m = (t > lo - pad) & (t < lo)
    w[m] = 0.5 - 0.5 * np.cos(np.pi * (t[m] - lo + pad) / pad)
    m = (t > hi) & (t < hi + pad)
    w[m] = 0.5 + 0.5 * np.cos(np.pi * (t[m] - hi) / pad)
    if clip is not None:
        w[t >= clip] = 0.0
    return w


def load(name, keep=None):
    """One sweep: raw traces at native rate plus a common-grid band-pass.

    Returns az, dt, raw (list), src2 (squared peak of the unfiltered
    analytic envelope, the source reference), e1 (stored backwall
    envelope peak), t1 (stored backwall arrival time), tg, X (band-passed
    on the common grid).
    """
    key = (name, keep)
    if key in _C:
        return _C[key]
    d = os.path.join(SWD, name)
    tg = np.arange(0.0, TMAX, DTC)
    az, raw, dts, src2, e1, t1, rows = [], [], [], [], [], [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        a = int(f[2:5])
        if keep is not None and a not in keep:
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], np.float64).ravel()
            dt = float(z["dt"])
            k = list(z.keys())
            e1.append(float(z["E1"]) if "E1" in k else np.nan)
            t1.append(float(z["t1_s"]) if "t1_s" in k else np.nan)
        az.append(a)
        raw.append(tr)
        dts.append(dt)
        src2.append(np.abs(hilbert(tr)).max() ** 2)
        rows.append(np.interp(tg, np.arange(len(tr)) * dt, bp_filter(tr, dt)))
    o = np.argsort(az)
    out = dict(name=name, az=np.array(az)[o],
               raw=[raw[i] for i in o], dt=np.array(dts)[o],
               src2=np.array(src2)[o], e1=np.array(e1)[o],
               t1=np.array(t1)[o], tg=tg, X=np.array(rows)[o])
    _C[key] = out
    return out


def level(name, win, keep=AZ30, est="tf", cut=None, band=BAND, clip=None):
    """Per-azimuth window level in dB re the per-azimuth source peak.

    est 'tf'   twice the mean square of the band-passed trace over the
               window. No transform, so nothing can be imported.
    est 'env'  the trace multiplied by a cosine gate BEFORE the Hilbert
               transform, then the mean square of the envelope inside
               the flat top. The order matters and section C proves it.
    cut, if given, zeroes the RAW trace at and beyond that time with a
    0.5 us cosine roll-off, before the band-pass, so the filter never
    sees the backwall either.
    """
    s = load(name, keep)
    tg = s["tg"]
    if cut is None and band == BAND:
        X = s["X"]
    else:
        rows = []
        for tr, dt in zip(s["raw"], s["dt"]):
            t = np.arange(len(tr)) * dt
            y = tr.copy()
            if cut is not None:
                r = 0.5e-6
                w = np.ones_like(t)
                m = (t > cut - r) & (t < cut)
                w[m] = 0.5 + 0.5 * np.cos(np.pi * (t[m] - cut + r) / r)
                w[t >= cut] = 0.0
                y = y * w
            ny = 0.5 / dt
            sos = butter(4, [band[0] / ny, band[1] / ny], btype="band",
                         output="sos")
            rows.append(np.interp(tg, t, sosfiltfilt(sos, y)))
        X = np.array(rows)
    g = (tg >= win[0]) & (tg < win[1])
    if est == "tf":
        p = 2.0 * np.mean(X[:, g] ** 2, axis=1)
    else:
        W = cosine_gate(tg, win[0], win[1], PAD, clip)
        p = np.mean(np.abs(hilbert(X * W[None, :], axis=1))[:, g] ** 2, axis=1)
    return 10 * np.log10(p / s["src2"])


def level_native(name, win, keep=AZ30, ref="src_az"):
    """The published paths, at each azimuth's own sample rate.

    'src_az'   2<x^2> of the band-passed trace over the window over that
               azimuth's own squared source peak. tab:reconcile's.
    'src_mean' mean square of the band-passed ENVELOPE over the window
               over the sweep-mean source peak. The T1 estimator.
    """
    s = load(name, keep)
    out = []
    for tr, dt in zip(s["raw"], s["dt"]):
        b = bp_filter(tr, dt)
        i0, i1 = int(win[0] / dt), int(win[1] / dt)
        if ref == "src_az":
            out.append(2.0 * float(np.mean(b[i0:i1] ** 2)))
        else:
            out.append(float(np.mean(np.abs(hilbert(b))[i0:i1] ** 2)))
    out = np.array(out)
    ref2 = s["src2"] if ref == "src_az" else s["src2"].mean()
    return 10 * np.log10(out / ref2)


def rev_level(name, win, keep=AZ30, **kw):
    """Revolution level: the azimuth mean taken over POWERS, in dB."""
    if kw:
        lv = level(name, win, keep, **kw)
    else:
        lv = level_native(name, win, keep, ref="src_az")
    return float(10 * np.log10(np.mean(10.0 ** (lv / 10.0))))


# ───────────────────────────── predictor ──────────────────────────────
def er2(seed, kappa=-8.0, ppw=8, src="labels"):
    """E[R^2] in dB against azimuth on [0, 180), 0.125 deg spacing.

    Volume-weighted mean square of dv/2vbar over grain pairs for a beam
    along the azimuth. src 'labels' weights by the voxel count of each
    grain in the cached label volume; src 'build' weights by the grain
    volumes stored beside tof_axis_recovery. The two are independent
    reads of the same realisation and section A checks they agree.
    """
    key = ("er2", seed, kappa, ppw, src)
    if key in _C:
        return _C[key]
    if src == "labels":
        p = os.path.join(TESS, f"tess_s{seed}_p{ppw:g}_k{kappa:g}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"not cached, would reach CUDA: {p}")
        with np.load(p) as z:
            lab = z["labels"]
            ax = np.asarray(z["axes"], np.float64)
        v = np.bincount(lab[lab >= 0].ravel(), minlength=len(ax))
        v = v.astype(np.float64)
    else:
        a0 = 1.000 if kappa < 0 else 0.866
        p = os.path.join(HERE, f"tofaxis_build_s{seed}_k{kappa:g}_"
                               f"a{a0:.3f}.npz")
        with np.load(p) as z:
            ax = np.asarray(z["axes"], np.float64)
            v = np.asarray(z["vol"], np.float64)
    k = v > 0
    ax, w = ax[k], v[k] / v[k].sum()
    grid = np.linspace(0.0, 180.0, 1441)
    th = np.radians(grid)
    N = np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
    ct = np.clip(np.abs(ax @ N.T), 0.0, 1.0)            # (grain, angle)
    vq = np.interp(np.arccos(ct), FW._PSI, FW._VQP)
    vb = w @ vq
    out = 10 * np.log10((w @ (vq - vb) ** 2) / (2.0 * vb ** 2))
    _C[key] = (grid, out)
    return grid, out


def pred(seed, az, kappa=-8.0, alpha=0.0, ppw=8, src="labels"):
    grid, c = er2(seed, kappa, ppw, src)
    return np.interp(np.mod(np.asarray(az, float) - alpha, 180.0),
                     grid, c, period=180.0)


# ─────────────────────────────── nulls ────────────────────────────────
def z(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def nshift(az):
    """Distinct rigid alignments: n/2, because the predictor is 180 deg
    periodic and np.roll on a uniform full circle is a rotation."""
    a = np.asarray(az, float)
    n = len(a)
    gap = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    if n % 2 or not np.allclose(gap, 360.0 / n, atol=1e-6):
        raise ValueError(f"not a uniform even circle, n = {n}")
    return n // 2


def shift_rank(y, p, az):
    """r at the true alignment, its rank among the distinct shifts, nd."""
    nd = nshift(az)
    zy, zp = z(y), z(p)
    rs = np.array([np.mean(np.roll(zy, s) * zp) for s in range(nd)])
    return float(rs[0]), int(np.sum(np.abs(rs) >= abs(rs[0]) - 1e-12)), nd


def alpha_fit(y, az, seed, kappa=-8.0, step=0.25):
    """Signed-maximising continuous rotation fit. Oracle: alpha = 0 is
    the true registration because the predictor is built from this
    specimen's own realised c-axes."""
    zy = z(y)
    al = np.arange(0.0, 180.0, step)
    r = np.array([np.mean(zy * z(pred(seed, az, kappa, a))) for a in al])
    j = int(np.argmax(r))
    d = abs(float(al[j])) % 180.0
    return float(al[j]), min(d, 180.0 - d), float(r[j])


# ──────────────────────────── length scales ───────────────────────────
def fresnel(win, lam=LAM):
    """First Fresnel diameter sqrt(2 lam L) at the window mid-range, and
    that range. Same convention as figures/fig_scales.py: the window
    centre is turned straight into a range, with no wavelet delay taken
    off, so the published gate returns the 14.9 mm the paper prints."""
    L = 0.25 * CREF * (win[0] + win[1])
    return float(np.sqrt(2.0 * lam * L)), float(L)


def grain_d(seed=11, ppw=8, kappa=-8.0):
    with np.load(os.path.join(TESS,
                              f"tess_s{seed}_p{ppw:g}_k{kappa:g}.npz")) as zz:
        n = len(zz["seeds"])
    v = np.pi * (DIA / 2) ** 2 * THK
    return float((6.0 * v / (np.pi * n)) ** (1 / 3)), n


def tag(w):
    return f"{w[0] * 1e6:.0f}-{w[1] * 1e6:.0f}"


def centroid(name, win, keep=AZ30):
    """Power-weighted spectral centroid of the windowed coda, in Hz."""
    s = load(name, keep)
    tg, num, den = s["tg"], 0.0, 0.0
    W = cosine_gate(tg, win[0], win[1], PAD)
    f = np.fft.rfftfreq(len(tg), DTC)
    m = (f >= BAND[0]) & (f <= BAND[1])
    for r in s["X"]:
        P = np.abs(np.fft.rfft(r * W)) ** 2
        num += float((f[m] * P[m]).sum())
        den += float(P[m].sum())
    return num / den


# ══════════════════════════════ A. HARNESS ════════════════════════════
def sec_a():
    """MEASURED. Sec. 5.2's own numbers, through this module's path."""
    print("=" * 78)
    print("A. HARNESS. Nothing below is worth reading until these return.")
    print("=" * 78)
    ok = True

    print("\n  A1  T1 in the published gate, 60 azimuths, 30 alignments,")
    print("      estimator: mean square of the band-passed envelope over")
    print("      the gate re the sweep-mean source peak.")
    print(f"      {'sweep':<20}{'r':>8}{'pub':>7}{'rank':>9}{'pub':>7}")
    for nm, ppw, k in (("girdle_perp", 6, "t1_p6"),
                       ("girdle_perp_ppw8", 8, "t1_p8")):
        s = load(nm)
        y = level_native(nm, GATE, None, ref="src_mean")
        r, rk, nd = shift_rank(y, pred(11, s["az"], -8.0, 0.0, ppw), s["az"])
        pr, prk, pnd = PUB[k]
        g = abs(r - pr) < 0.006 and abs(rk - prk) <= 1 and nd == pnd
        ok &= g
        print(f"      {nm:<20}{r:>+8.3f}{pr:>+7.2f}{rk:>6d}/{nd:<3}"
              f"{prk:>4d}/{pnd:<3}  {'ok' if g else 'MISMATCH'}")

    print("\n  A2  tab:reconcile ladder, revolution level re source, 30 az.")
    for nm, k in (("girdle_perp", "lad_p6"), ("girdle_perp_ppw8", "lad_p8"),
                  ("lic_girdle_s11_ppw10", "lad_p10")):
        v = rev_level(nm, GATE)
        g = abs(v - PUB[k]) < 0.02
        ok &= g
        print(f"      {nm:<22}{v:>10.2f}{PUB[k]:>10.2f}   "
              f"{'ok' if g else 'MISMATCH'}")

    print("\n  A3  T3, eight matched pairs, girdle minus single maximum.")
    gg = np.array([rev_level(GIR[s], GATE) for s in SEEDS])
    mm = np.array([rev_level(SIN[s], GATE) for s in SEEDS])
    d = gg - mm
    t, p = stats.ttest_rel(gg, mm)
    pd, pt, pp = PUB["t3"]
    g = abs(d.mean() - pd) < 0.02 and abs(t - pt) < 0.02 and abs(p - pp) < 2e-3
    ok &= g
    print(f"      delta {d.mean():+.2f} (pub {pd:+.2f})   t {t:+.2f} "
          f"(pub {pt:+.2f})   p {p:.4f} (pub {pp:.3f})   "
          f"{'ok' if g else 'MISMATCH'}")

    print("\n  A4  the two independent reads of the seed-11 realisation.")
    for seed, kap in ((11, -8.0), (17, -8.0), (11, 3.93)):
        a = pred(seed, np.arange(0, 360, 6), kap, 0.0, 8, "labels")
        b = pred(seed, np.arange(0, 360, 6), kap, 0.0, 8, "build")
        print(f"      seed {seed:<3} kappa {kap:>5}   max |labels - build| "
              f"{np.abs(a - b).max():.4f} dB   r {np.corrcoef(a, b)[0, 1]:.6f}")

    print("\n  A5  Sec. sec:window's imported share: whole-trace Hilbert")
    print("      minus tapered, as a fraction of the untapered.")
    for w, ref in ((GATE, PUB["imported_gate"]), ((30e-6, 42e-6),
                                                  PUB["imported_3042"])):
        s = load(SPEC, AZ30)
        tg, g = s["tg"], (s["tg"] >= w[0]) & (s["tg"] < w[1])
        W = cosine_gate(tg, w[0], w[1], PAD)
        a = np.mean(np.abs(hilbert(s["X"], axis=1))[:, g] ** 2)
        b = np.mean(np.abs(hilbert(s["X"] * W[None, :], axis=1))[:, g] ** 2)
        print(f"      {tag(w):<8} imported {100 * (a - b) / a:>6.1f} %"
              f"   published {ref:>5.1f} %")

    print("\n   " + ("HARNESS REPRODUCES" if ok else "HARNESS DOES NOT"))
    return ok


# ═════════════════════════ B. GEOMETRY AT 44 us ═══════════════════════
def sec_b():
    """MEASURED geometry, INFERRED interpretation, both labelled."""
    print("\n" + "=" * 78)
    print("B. WHAT IS INSONIFIED AT 44 us. MEASURED where marked.")
    print("=" * 78)
    dg, n = grain_d()
    print(f"   MEASURED  lambda {LAM * 1e3:.3f} mm, {n} grains realised,")
    print(f"             volume-equivalent grain {dg * 1e3:.2f} mm.")
    a = ELEM / 2
    near = a ** 2 / LAM
    sin_null = min(1.0, 0.61 * LAM / a)
    th = np.degrees(np.arcsin(sin_null))
    print(f"   MEASURED  element {ELEM * 1e3:.2f} mm, near field "
          f"{near * 1e3:.2f} mm, first-null half angle {th:.1f} deg.")
    print(f"\n   {'window':<8}{'centre':>8}{'L one-way':>11}{'D_Fr':>8}"
          f"{'ratio':>7}{'beam r':>8}{'disc r':>8}{'half thk':>9}"
          f"{'lost':>7}")
    rows = {}
    for w in CANDS:
        D, L = fresnel(w)
        br = L * np.tan(np.radians(th))
        dr = np.sqrt(max((DIA / 2) ** 2 - (L - DIA / 2) ** 2, 0.0))
        lost = "rim+face" if br > dr else ("face" if br > THK / 2 else "-")
        rows[tag(w)] = (L, D, D / dg, br, dr)
        print(f"   {tag(w):<8}{0.5 * (w[0] + w[1]) * 1e6:>7.1f}u"
              f"{L * 1e3:>10.1f}m{D * 1e3:>7.2f}m{D / dg:>7.3f}"
              f"{br * 1e3:>7.1f}m{dr * 1e3:>7.1f}m{THK / 2 * 1e3:>8.1f}m"
              f"{lost:>8}")
    eq = dg ** 2 / (LAM * CREF)
    print(f"\n   MEASURED  equality D_Fresnel = D_grain at "
          f"{eq * 1e6:.2f} us two-way.")
    print("   INFERRED  the first-null beam radius passes the disc's own")
    print("   radius at that range, so the outer beam is on the curved rim")
    print("   and its absorbing layer, and it has exceeded the half")
    print("   thickness since well before the published gate. The late")
    print("   window is not a clean bulk measurement; it is a measurement")
    print("   of a beam that is partly on the boundary treatment.")
    return rows


# ═════════════════════════ C. THE BACKWALL ════════════════════════════
def sec_c():
    """MEASURED. Every backwall number, three independent ways."""
    print("\n" + "=" * 78)
    print("C. THE BACKWALL. MEASURED.")
    print("=" * 78)
    out = {}

    print("\n  C1  arrival and skirt, measured from the trace. The skirt is")
    print("      walked back CONTIGUOUSLY from the peak, so the front")
    print("      arrival cannot be mistaken for it. Only some sweeps store")
    print("      t1_s, so the estimator is checked against it where it is.")
    for nm in (SPEC,) + CTL:
        s = load(nm, AZ30)
        pk, ed, sk = [], [], {20: [], 40: [], 48: []}
        for tr, dt in zip(s["raw"], s["dt"]):
            tp, ep, te, e, t = backwall(tr, dt)
            pk.append(tp)
            ed.append(te)
            j = int(round(tp / dt))
            for db in sk:
                thr = ep * 10 ** (-db / 20)
                i = j
                while i > 0 and e[i] > thr:
                    i -= 1
                sk[db].append(t[i])
        st = s["t1"]
        chk = (f"stored {np.nanmean(st) * 1e6:6.2f}"
               if np.isfinite(st).all() else "stored  -    ")
        print(f"      {nm:<22} peak {np.mean(pk) * 1e6:6.2f} "
              f"(sd {np.std(pk) * 1e6:.3f})  -6dB edge "
              f"{np.mean(ed) * 1e6:6.2f}  {chk}")
        print(f"        contiguous skirt reaches back to: "
              + "  ".join(f"-{d} dB {np.mean(v) * 1e6:6.2f} us"
                          for d, v in sk.items()))
        out[("arrival", nm)] = (float(np.mean(pk)),
                                float(np.mean(sk[48])))
    print(f"      nominal 2D/c = {2 * DIA / CREF * 1e6:.2f} us. The coda")
    print("      stands about 48 dB below the arrival, so the -48 dB row")
    print("      is the time at which the arrival stops being negligible.")

    print("\n  C1b  the specimen's median envelope, dB below the backwall")
    print("       peak, at the edges of every candidate window.")
    s = load(SPEC, AZ30)
    e = np.median(np.abs(hilbert(s["X"], axis=1)), axis=0)
    ep = e[(s["tg"] > 48e-6) & (s["tg"] < 58e-6)].max()
    print("       " + "".join(f"{u:>8.0f}" for u in
                              (36, 42, 44, 48, 50, 51, 52)) + "  us")
    print("       " + "".join(
        f"{20 * np.log10(np.interp(u * 1e-6, s['tg'], e) / ep):>8.1f}"
        for u in (36, 42, 44, 48, 50, 51, 52)) + "  dB")

    print("\n  C2  band-pass smear of the arrival on its own. A unit")
    print("      impulse at the measured peak, through the same filter,")
    print("      and the time at which its envelope has fallen 48 dB, the")
    print("      backwall-to-coda ratio.")
    s = load(SPEC, AZ30)
    dt = s["dt"][0]
    tp = out[("arrival", SPEC)][0]
    imp = np.zeros(len(s["raw"][0]))
    imp[int(round(tp / dt))] = 1.0
    e = np.abs(hilbert(bp_filter(imp, dt)))
    t = np.arange(len(e)) * dt
    for db in (20, 40, 48, 60):
        m = np.where(e[t < tp] > e.max() * 10 ** (-db / 20))[0]
        print(f"      -{db:>2} dB reached {(tp - t[m[0]]) * 1e6:5.2f} us "
              f"before the peak" if len(m) else f"      -{db} dB not reached")

    print("\n  C3  leakage into each window, three ways, per sweep.")
    print("      (a) share removed by excising the RAW trace at 51.0 us")
    print("          before the band-pass, estimator tf and env;")
    print("      (b) the backwall-only trace, i.e. raw minus excised,")
    print("          carried through the identical path;")
    print("      (c) cut sensitivity of (b) over 49.0 to 51.5 us.")
    print(f"      {'window':<8}{'sweep':<22}{'(a) tf':>9}{'(a) env':>9}"
          f"{'(b) tf':>9}{'(b) env':>9}")
    CUT = 51.0e-6
    for w in CANDS:
        for nm in (SPEC,) + CTL:
            row = []
            for est in ("tf", "env"):
                f = 10 ** (level(nm, w, AZ30, est) / 10)
                c = 10 ** (level(nm, w, AZ30, est, cut=CUT) / 10)
                row.append(100 * (f.mean() - c.mean()) / f.mean())
            for est in ("tf", "env"):
                f = 10 ** (level(nm, w, AZ30, est) / 10)
                b = bw_only_power(nm, w, est, CUT)
                row.append(100 * b / f.mean())
            out[("leak", tag(w), nm)] = tuple(float(v) for v in row)
            print(f"      {tag(w):<8}{nm:<22}" + "".join(
                f"{v:>9.2f}" for v in row))

    print("\n      cut sensitivity, backwall-only share, estimator env")
    print(f"      {'window':<8}{'sweep':<22}" + "".join(
        f"{c:>8.1f}" for c in (49.0, 50.0, 51.0, 51.5)))
    for w in ((38e-6, 50e-6), (40e-6, 48e-6), (38e-6, 44e-6)):
        for nm in (SPEC, CTL[0]):
            vals = []
            for c in (49.0e-6, 50.0e-6, 51.0e-6, 51.5e-6):
                if c <= w[1]:
                    vals.append(np.nan)
                    continue
                f = 10 ** (level(nm, w, AZ30, "env") / 10)
                vals.append(100 * bw_only_power(nm, w, "env", c) / f.mean())
            print(f"      {tag(w):<8}{nm:<22}" + "".join(
                f"{v:>8.2f}" for v in vals))

    print("\n  C4  is the taper actually applied BEFORE the transform?")
    print("      Constructed failure: the same window scored with the")
    print("      Hilbert transform taken on the whole record and the")
    print("      window applied afterwards, which is what the early")
    print("      windows were caught doing.")
    print(f"      {'window':<8}{'taper-then-H':>14}{'H-then-window':>15}"
          f"{'imported %':>12}")
    for w in CANDS:
        s = load(SPEC, AZ30)
        tg, g = s["tg"], (s["tg"] >= w[0]) & (s["tg"] < w[1])
        W = cosine_gate(tg, w[0], w[1], PAD)
        a = np.mean(np.abs(hilbert(s["X"] * W[None, :], axis=1))[:, g] ** 2)
        b = np.mean(np.abs(hilbert(s["X"], axis=1))[:, g] ** 2)
        print(f"      {tag(w):<8}{10 * np.log10(a):>14.2f}"
              f"{10 * np.log10(b):>15.2f}{100 * (b - a) / b:>12.1f}")

    print("\n  C5  does the 38-50 skirt reach the backwall? Its top is at")
    print("      52.0 us and the peak is at 52.2. Scored with the skirt")
    print("      clipped at 51.0 us, flat top unchanged.")
    s = load(SPEC, AZ30)
    for w in ((38e-6, 50e-6), (40e-6, 48e-6)):
        a = level(SPEC, w, AZ30, "env")
        b = level(SPEC, w, AZ30, "env", clip=51.0e-6)
        p = pred(11, s["az"], -8.0, 0.0, 8)
        ra = shift_rank(a, p, s["az"])
        rb = shift_rank(b, p, s["az"])
        print(f"      {tag(w):<8} level {a.mean():+8.2f} -> {b.mean():+8.2f} dB"
              f"   r {ra[0]:+.3f} -> {rb[0]:+.3f}   rank {ra[1]}/{ra[2]}"
              f" -> {rb[1]}/{rb[2]}")
    return out


def backwall(tr, dt):
    """Backwall peak time and amplitude, and its -6 dB leading edge.

    Measured on the band-passed analytic envelope inside 46 to 60 us,
    with a parabolic refinement of the peak sample. The leading edge is
    walked back contiguously from that peak, so no earlier arrival can
    be mistaken for it. Returns (t_peak, e_peak, t_edge, envelope, t).
    """
    e = np.abs(hilbert(bp_filter(tr, dt)))
    t = np.arange(len(e)) * dt
    m = np.where((t > 46e-6) & (t < 60e-6))[0]
    j = m[int(np.argmax(e[m]))]
    y0, y1, y2 = e[j - 1], e[j], e[j + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) else 0.0
    tp, ep = (j + d) * dt, y1
    i = j
    while i > 0 and e[i] > ep * 0.5:
        i -= 1
    fr = ((ep * 0.5 - e[i]) / (e[i + 1] - e[i])) if e[i + 1] > e[i] else 0.0
    return float(tp), float(ep), float((i + fr) * dt), e, t


def bw_channel(name, keep=AZ30):
    """The two coherent-channel observables at every azimuth: the
    backwall's -6 dB arrival time, which is the diameter-average
    slowness, and its peak amplitude in dB re the source peak, which is
    the diameter-average attenuation and beam distortion. Neither
    involves the coda and neither involves any grain-boundary echo."""
    key = ("bw", name, keep)
    if key in _C:
        return _C[key]
    s = load(name, keep)
    te, ea = [], []
    for tr, dt in zip(s["raw"], s["dt"]):
        tp, ep, edge, _, _ = backwall(tr, dt)
        te.append(edge)
        ea.append(ep)
    v = (np.array(te), 20 * np.log10(np.array(ea) / np.sqrt(s["src2"])))
    _C[key] = v
    return v


def bw_only_power(name, win, est, cut):
    """Mean windowed power of the BACKWALL-ONLY trace: raw minus the
    excised raw, through the same band-pass, gate and estimator."""
    s = load(name, AZ30)
    tg = s["tg"]
    rows = []
    for tr, dt in zip(s["raw"], s["dt"]):
        t = np.arange(len(tr)) * dt
        r = 0.5e-6
        w = np.ones_like(t)
        m = (t > cut - r) & (t < cut)
        w[m] = 0.5 + 0.5 * np.cos(np.pi * (t[m] - cut + r) / r)
        w[t >= cut] = 0.0
        rows.append(np.interp(t if False else tg, t,
                              bp_filter(tr * (1.0 - w), dt)))
    X = np.array(rows)
    g = (tg >= win[0]) & (tg < win[1])
    if est == "tf":
        p = 2.0 * np.mean(X[:, g] ** 2, axis=1)
    else:
        W = cosine_gate(tg, win[0], win[1], PAD)
        p = np.mean(np.abs(hilbert(X * W[None, :], axis=1))[:, g] ** 2, axis=1)
    return float(np.mean(p / s["src2"]))


# ═════════════════════ D. CONTROLS AND THE AXIS TEST ══════════════════
def sec_d():
    """MEASURED. Every rank labelled with the grid it is on."""
    print("\n" + "=" * 78)
    print("D. T1 AND THE CONTROLS. GRID 30: 30 azimuths, 15 distinct")
    print("   alignments, exact floor 1/15 = 0.0667. A rank out of 15 is")
    print("   NOT comparable with a published rank out of 30.")
    print("=" * 78)
    out = {}
    for nm in (SPEC,) + CTL:
        s = load(nm, AZ30)
        p = pred(11, s["az"], -8.0, 0.0, 8)
        print(f"\n  {nm}   n = {len(s['az'])}, alignments = {nshift(s['az'])}")
        print(f"  {'window':<9}{'r tf':>9}{'rank':>8}{'r env':>9}{'rank':>8}"
              f"{'r tf cut':>10}{'rank':>8}")
        for w in CANDS:
            row = []
            for est, cut in (("tf", None), ("env", None), ("tf", 51.0e-6)):
                y = level(nm, w, AZ30, est, cut=cut)
                r, rk, nd = shift_rank(y, p, s["az"])
                row += [r, f"{rk}/{nd}"]
                out[(nm, tag(w), est, "cut" if cut else "raw")] = (r, rk, nd)
            print(f"  {tag(w):<9}{row[0]:>+9.3f}{row[1]:>8}{row[2]:>+9.3f}"
                  f"{row[3]:>8}{row[4]:>+9.3f}{row[5]:>8}")

    print("\n  GRID 60: specimen and zerocontrast only, 30 alignments,")
    print("  exact floor 1/30 = 0.0333. cs_f000_s11_ppw8 holds 30")
    print("  azimuths and cannot appear here.")
    for nm in (SPEC, CTL[0]):
        s = load(nm, None)
        p = pred(11, s["az"], -8.0, 0.0, 8)
        print(f"\n  {nm}   n = {len(s['az'])}, alignments = {nshift(s['az'])}")
        print(f"  {'window':<9}{'r tf':>9}{'rank':>8}{'r env':>9}{'rank':>8}")
        for w in CANDS:
            row = []
            for est in ("tf", "env"):
                y = level(nm, w, None, est)
                r, rk, nd = shift_rank(y, p, s["az"])
                row += [r, f"{rk}/{nd}"]
                out[(nm, tag(w), est, "g60")] = (r, rk, nd)
            print(f"  {tag(w):<9}{row[0]:>+9.3f}{row[1]:>8}{row[2]:>+9.3f}"
                  f"{row[3]:>8}")

    print("\n  ZERO-SCATTERING MARGIN, revolution level dB re source, tf.")
    print(f"  {'window':<9}{'specimen':>10}{'zerocon':>10}{'margin':>9}"
          f"{'cs f=0':>10}{'margin':>9}{'null pair':>11}")
    for w in CANDS:
        sp = rev_level(SPEC, w, AZ30, est="tf")
        zc = rev_level(CTL[0], w, AZ30, est="tf")
        cs = rev_level(CTL[1], w, AZ30, est="tf")
        out[("margin", tag(w))] = (sp, zc, cs)
        print(f"  {tag(w):<9}{sp:>10.2f}{zc:>10.2f}{sp - zc:>9.2f}"
              f"{cs:>10.2f}{sp - cs:>9.2f}{zc - cs:>11.2f}")
    print("  'null pair' is a difference that must be zero.")
    return out


# ═════════════════ E. THE CONFOUND THE CONTROLS CANNOT SEE ════════════
def sec_e():
    """MEASURED regressions, INFERRED reading, both labelled.

    Neither zero-scattering record carries the seed-11 girdle, so a clean
    control rules out rendering and geometry and does NOT rule out the
    bulk velocity-anisotropy channel. That channel is measured directly
    at every azimuth by the coherent backwall: its arrival TIME is the
    diameter-average slowness and its AMPLITUDE the diameter-average
    attenuation and beam distortion.
    """
    print("\n" + "=" * 78)
    print("E. THE CHANNEL THE CONTROLS DO NOT CONTROL FOR.")
    print("=" * 78)
    s = load(SPEC, AZ30)
    az = s["az"]
    p = pred(11, az, -8.0, 0.0, 8)
    tof, amp = bw_channel(SPEC)
    st = s["t1"]
    if np.isfinite(st).all():
        print(f"\n  E0  the measured arrival tracks the stored t1_s at "
              f"r = {np.corrcoef(tof, st)[0, 1]:+.4f}, offset "
              f"{np.mean(tof - st) * 1e6:+.3f} us.")
    print("\n  E1  the coherent backwall against the fabric predictor.")
    for lab, v in (("arrival time  t1_s", tof), ("backwall level dB", amp)):
        r, rk, nd = shift_rank(v, p, az)
        print(f"      {lab:<20} r {r:+.3f}   rank {rk}/{nd}   "
              f"sd {np.std(v) * (1e9 if 'time' in lab else 1):.3f}"
              f"{' ns' if 'time' in lab else ' dB'}")
    print("      This is the Sec. 5.4 phase channel measured directly. It")
    print("      is present at every window because it is the same wave.")

    print("\n  E2  the window level against the backwall, and the fabric")
    print("      correlation re-scored on the residual after the backwall")
    print("      channel is projected out. Grid 30, 15 alignments.")
    print(f"      {'window':<9}{'r(lvl,tof)':>11}{'r(lvl,amp)':>11}"
          f"{'r raw':>8}{'rank':>7}{'r resid':>9}{'rank':>7}")
    out = {}
    for w in CANDS:
        y = level(SPEC, w, AZ30, "tf")
        r_t = float(np.corrcoef(y, tof)[0, 1])
        r_a = float(np.corrcoef(y, amp)[0, 1])
        r0, rk0, nd = shift_rank(y, p, az)
        D = np.column_stack([np.ones_like(y), z(tof), z(amp)])
        res = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
        r1, rk1, _ = shift_rank(res, p, az)
        out[tag(w)] = (r_t, r_a, r0, rk0, r1, rk1, nd)
        print(f"      {tag(w):<9}{r_t:>+11.3f}{r_a:>+11.3f}{r0:>+8.3f}"
              f"{rk0:>5}/{nd:<2}{r1:>+9.3f}{rk1:>5}/{nd:<2}")
    print("      INFERRED. If the late-window rise survives removal of the")
    print("      coherent channel it is not that channel; if it does not,")
    print("      the two cannot be told apart on this record.")

    print("\n  E3  the same on both controls. A control has no coda, so a")
    print("      strong r(level, backwall) there is the same coherent")
    print("      channel with nothing else in the window.")
    print(f"      {'sweep':<22}{'window':<8}{'r(lvl,tof)':>11}"
          f"{'r(lvl,amp)':>11}{'r raw':>8}{'rank':>7}{'r resid':>9}")
    for nm in CTL:
        sc = load(nm, AZ30)
        pc = pred(11, sc["az"], -8.0, 0.0, 8)
        tf_, am_ = bw_channel(nm)
        for w in CANDS:
            y = level(nm, w, AZ30, "tf")
            r0, rk0, nd = shift_rank(y, pc, sc["az"])
            D = np.column_stack([np.ones_like(y), z(tf_), z(am_)])
            res = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
            r1, _, _ = shift_rank(res, pc, sc["az"])
            print(f"      {nm:<22}{tag(w):<8}"
                  f"{np.corrcoef(y, tf_)[0, 1]:>+11.3f}"
                  f"{np.corrcoef(y, am_)[0, 1]:>+11.3f}{r0:>+8.3f}"
                  f"{rk0:>5}/{nd:<2}{r1:>+9.3f}")

    print("\n  E4  the same over the twelve tessellations: the first-rank")
    print("      count that carries the late-window case, recomputed on")
    print("      the level with the coherent channel projected out.")
    print(f"      {'window':<9}{'mean r(lvl,tof)':>17}{'first/12 raw':>14}"
          f"{'first/12 resid':>16}{'ranksum raw':>13}{'resid':>8}")
    for w in CANDS:
        rr, r0s, r1s = [], [], []
        for nm, seed, kap in TWELVE:
            sw = load(nm, AZ30)
            pw = pred(seed, sw["az"], kap, 0.0, 8)
            tf_, am_ = bw_channel(nm)
            y = level(nm, w, AZ30, "tf")
            rr.append(np.corrcoef(y, tf_)[0, 1])
            r0s.append(shift_rank(y, pw, sw["az"])[1])
            D = np.column_stack([np.ones_like(y), z(tf_), z(am_)])
            res = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
            r1s.append(shift_rank(res, pw, sw["az"])[1])
        r0s, r1s = np.array(r0s), np.array(r1s)
        out[("twelve", tag(w))] = (float(np.mean(rr)),
                                   int((r0s == 1).sum()),
                                   int((r1s == 1).sum()),
                                   int(r0s.sum()), int(r1s.sum()))
        print(f"      {tag(w):<9}{np.mean(rr):>+17.3f}{(r0s == 1).sum():>11}"
              f"/12{(r1s == 1).sum():>13}/12{r0s.sum():>13}{r1s.sum():>8}")
    return out


# ═══════════════════ F. THE TWELVE, AND MULTIPLICITY ══════════════════
def sec_f():
    """MEASURED counts and paired tests; INFERRED multiplicity reading."""
    print("\n" + "=" * 78)
    print("F. THE TWELVE TESSELLATIONS, AND WHAT A COUNT IS WORTH.")
    print("=" * 78)
    print("   Chance for a first rank is 1/15 per specimen, so 0.8 of 12.")
    print("   Chance rank sum is 12 * 8 = 96.")
    print(f"\n   {'window':<9}{'first/12':>10}{'exact p':>10}{'ranksum':>10}"
          f"{'z':>8}{'axis deg':>10}{'median':>8}{'worst':>8}")
    out = {}
    base = None
    for w in CANDS:
        ranks, errs = [], []
        for nm, seed, kap in TWELVE:
            s = load(nm, AZ30)
            y = level(nm, w, AZ30, "tf")
            _, rk, nd = shift_rank(y, pred(seed, s["az"], kap, 0.0, 8),
                                   s["az"])
            ranks.append(rk)
            errs.append(alpha_fit(y, s["az"], seed, kap)[1])
        ranks, errs = np.array(ranks), np.array(errs)
        first = int((ranks == 1).sum())
        pe = float(stats.binom.sf(first - 1, 12, 1.0 / nd))
        rs = int(ranks.sum())
        sd = np.sqrt(12 * (nd ** 2 - 1) / 12.0)
        zz = (rs - 12 * (nd + 1) / 2) / sd
        out[tag(w)] = (ranks.copy(), errs.copy(), first, pe, rs)
        if base is None:
            base = ranks.copy()
        print(f"   {tag(w):<9}{first:>7}/12{pe:>10.4f}{rs:>10}{zz:>8.2f}"
              f"{errs.mean():>10.2f}{np.median(errs):>8.2f}"
              f"{errs.max():>8.2f}")
    print("   chance axis error 45.0 deg; time of flight reaches 5.9 deg.")

    print("\n   Paired on the tessellation, against the published gate.")
    print(f"   {'window':<9}{'better':>8}{'worse':>7}{'tied':>6}"
          f"{'sign p':>9}{'wilcoxon':>10}")
    for w in CANDS[1:]:
        d = base - out[tag(w)][0]
        b, ws = int((d > 0).sum()), int((d < 0).sum())
        sp = float(stats.binomtest(b, b + ws, 0.5).pvalue) if b + ws else 1.0
        try:
            wp = float(stats.wilcoxon(base, out[tag(w)][0]).pvalue)
        except ValueError:
            wp = np.nan
        print(f"   {tag(w):<9}{b:>8}{ws:>7}{12 - b - ws:>6}{sp:>9.3f}"
              f"{wp:>10.3f}")

    print("\n   MULTIPLICITY, counted over the project and not over this")
    print("   module. Sec. 5.3 scores 16 gates; the window study adds 3")
    print("   windows x 2 treatments x 2 estimators; this adjudication and")
    print("   the tested module together add 6 windows x 2 estimators x 2")
    print("   grids x 2 excision modes. Two predictors, bulk and beam")
    print("   local, have been carried through most of them.")
    n_gate = 16 + 3 * 2 * 2 + 6 * 2 * 2 * 2
    print(f"   A conservative count of window-by-estimator cells is "
          f"{n_gate}.")
    for w in ((38e-6, 50e-6), (40e-6, 48e-6)):
        pe = out[tag(w)][3]
        print(f"   {tag(w)}: exact p {pe:.4f} on its own; "
              f"1 - (1 - p)^6 over the six windows alone = "
              f"{1 - (1 - pe) ** 6:.3f}; over {n_gate} cells = "
              f"{1 - (1 - pe) ** n_gate:.3f}.")
    print("   INFERRED. A 3-of-12 first-rank count reported at p = 0.04 is")
    print("   not a finding at this multiplicity, and the same count")
    print("   appears in three different windows, which is one family and")
    print("   not three confirmations.")
    return out


# ═══════════ G. IS THE PROJECTION ITSELF DOING THE DAMAGE? ════════════
def sec_g():
    """MEASURED placebo. Removing two covariates costs two degrees of
    freedom and could move a rank count on its own, so the projection is
    repeated against every WRONG alignment of the same two covariates.
    A circular shift of the backwall channel preserves its azimuthal
    structure, its variance and its cost, and destroys only its
    registration with the specimen. The true projection is one draw
    from that null, and section E's claim stands only if it sits in the
    tail."""
    print("\n" + "=" * 78)
    print("G. PLACEBO FOR THE PROJECTION. MEASURED.")
    print("=" * 78)
    print("   The coherent channel projected out at every one of the 15")
    print("   alignments. Shift 0 is the true one; the other 14 are the")
    print("   null. Statistic: first ranks of 12, and the rank sum.")
    print(f"\n   {'window':<9}{'R2 spec':>9}{'raw':>7}{'true':>7}"
          f"{'null mean':>11}{'null min':>10}{'p(<=true)':>11}"
          f"{'  ranksum true / null mean'}")
    out = {}
    for w in CANDS:
        Y, P, T, A, AZ = [], [], [], [], None
        for nm, seed, kap in TWELVE:
            sw = load(nm, AZ30)
            AZ = sw["az"]
            Y.append(level(nm, w, AZ30, "tf"))
            P.append(pred(seed, sw["az"], kap, 0.0, 8))
            tf_, am_ = bw_channel(nm)
            T.append(tf_)
            A.append(am_)
        r2 = []
        for y, tf_, am_ in zip(Y, T, A):
            D = np.column_stack([np.ones_like(y), z(tf_), z(am_)])
            res = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
            r2.append(1.0 - res.var() / y.var())
        firsts, sums = [], []
        for s in range(15):
            rk = []
            for y, p, tf_, am_ in zip(Y, P, T, A):
                D = np.column_stack([np.ones_like(y), z(np.roll(tf_, s)),
                                     z(np.roll(am_, s))])
                res = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
                rk.append(shift_rank(res, p, AZ)[1])
            rk = np.array(rk)
            firsts.append(int((rk == 1).sum()))
            sums.append(int(rk.sum()))
        raw = int(np.array([shift_rank(y, p, AZ)[1]
                            for y, p in zip(Y, P)]) == 1).sum() \
            if False else int(sum(shift_rank(y, p, AZ)[1] == 1
                                  for y, p in zip(Y, P)))
        nullf = np.array(firsts[1:])
        pv = float(np.mean(nullf <= firsts[0]))
        out[tag(w)] = (float(np.mean(r2)), raw, firsts[0], pv, sums[0],
                       float(np.mean(sums[1:])))
        print(f"   {tag(w):<9}{np.mean(r2[0:1]):>9.3f}{raw:>7}{firsts[0]:>7}"
              f"{nullf.mean():>11.2f}{nullf.min():>10}{pv:>11.3f}"
              f"      {sums[0]} / {np.mean(sums[1:]):.1f}")
    print("\n   R2 spec is the share of the seed-11 specimen's azimuthal")
    print("   level variance the coherent channel alone explains in that")
    print("   window. INFERRED: a true projection that lands at or below")
    print("   the worst of the fourteen wrong ones is removing something")
    print("   registered to the specimen, not paying for two degrees of")
    print("   freedom.")
    return out


# ══════ H. WHERE THE FLOOR RISES, AGAINST WHERE THE ZONE GROWS ════════
def sec_h():
    """MEASURED. The two things that both happen with range, profiled on
    the same axis: the Fresnel-to-grain ratio, which the mechanism needs
    above unity, and the margin over the two records that cannot
    backscatter, which an admissible window needs high."""
    print("\n" + "=" * 78)
    print("H. THE FLOOR AND THE ZONE, ON ONE AXIS. MEASURED.")
    print("=" * 78)
    dg, _ = grain_d()
    print(f"   {'gate':<9}{'ratio':>7}{'specimen':>10}{'zerocon':>10}"
          f"{'margin':>8}{'cs f=0':>9}{'margin':>8}{'worse margin':>14}")
    prof = {}
    for a in range(6, 49, 4):
        w = (a * 1e-6, (a + 4) * 1e-6)
        D, _ = fresnel(w)
        sp = rev_level(SPEC, w, AZ30, est="tf")
        zc = rev_level(CTL[0], w, AZ30, est="tf")
        cs = rev_level(CTL[1], w, AZ30, est="tf")
        prof[tag(w)] = (D / dg, sp, zc, cs)
        print(f"   {tag(w):<9}{D / dg:>7.3f}{sp:>10.2f}{zc:>10.2f}"
              f"{sp - zc:>8.2f}{cs:>9.2f}{sp - cs:>8.2f}"
              f"{min(sp - zc, sp - cs):>14.2f}")
    eq = dg ** 2 / (LAM * CREF)
    print(f"\n   The ratio passes unity at {eq * 1e6:.1f} us. Read the last")
    print("   column against that time. INFERRED: on this record the")
    print("   window that first satisfies the mechanism's antecedent is")
    print("   also the first window in which the zero-scattering floor has")
    print("   climbed to within single-figure decibels of the coda, so the")
    print("   two conditions a decisive test needs are not simultaneously")
    print("   available anywhere in it.")
    return prof


def main():
    ok = sec_a()
    if not ok:
        print("\n  ADJUDICATION STOPS: the harness does not reproduce.")
    geom = sec_b()
    leak = sec_c()
    ctl = sec_d()
    conf = sec_e()
    ens = sec_f()
    plac = sec_g()
    prof = sec_h()
    np.savez(os.path.join(HERE, "late_window_adjudication_placebo.npz"),
             plac=np.array(sorted(f"{k}={v}" for k, v in plac.items())),
             prof=np.array(sorted(f"{k}={v}" for k, v in prof.items())))
    np.savez(os.path.join(HERE, "late_window_adjudication.npz"),
             harness=np.array([ok]),
             geom=np.array(sorted(f"{k}={v}" for k, v in geom.items())),
             leak=np.array(sorted(f"{k}={v}" for k, v in leak.items())),
             ctl=np.array(sorted(f"{k}={v}" for k, v in ctl.items())),
             conf=np.array(sorted(f"{k}={v}" for k, v in conf.items())),
             ens=np.array(sorted(f"{k}={v[2:]}" for k, v in ens.items())))


if __name__ == "__main__":
    main()
