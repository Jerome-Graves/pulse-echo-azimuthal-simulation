"""Adjudication of the beam-local predictor of Section 5.2.

The claim under attack is the last paragraph of the subsection: that a
weak crystallographic orientation channel is visible in an earlier
window that the paper does not analyse in.  The bulk predictor is
already known to be inadmissible there, because two specimens that
cannot backscatter from a grain boundary at all are tracked better than
the real one.  This module asks whether the beam-local predictor shares
that defect, and it asks three further questions that the admissibility
test alone does not settle:

  A  is the predictor separable from the geometry of its own column at
     all, or does regressing the orientation-blind column descriptors
     out of it destroy whatever correlation it has;
  B  how collinear are the predictor and that geometry, azimuth by
     azimuth, in each window;
  C  does the azimuthal level of a zero-scattering record carried on the
     SAME tessellation already predict the specimen's level, which would
     make the whole correlation a rendering artefact rather than a
     measurement.

Nothing here builds a specimen.  Every tessellation quantity is read
from out/tesscache and every level from the stored traces in
out/sweeps.  No CUDA import is on the path.

The predictor.  Established from analysis/beam_descriptors.py, whose
v_var is the statistic Section 5.2 describes verbatim: a rectangular
column of the element width marched at h/2 over the depth range the
window listens to, lateral offsets at the grid pitch, through-thickness
slices every 2 mm, and the variance of the qP speed for that beam
direction taken over the SAMPLED POINTS, so each grain enters weighted
by the volume of it the column holds.  That reading returns the
published r to three decimals at both resolutions; the two competing
readings of the same English sentence do not.

Run:  python beam_local_break.py [n_perm]
"""
import os
import sys
import time

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import chi2, wilcoxon

ROOT = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SWD = os.path.join(ROOT, "out", "sweeps")
TESS = os.path.join(ROOT, "out", "tesscache")
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import forward as F                                        # noqa: E402

C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
ELEM, BAND = 6.35e-3, (0.8e6, 3.0e6)
WINDOWS = {"4-16": (4e-6, 16e-6), "10-22": (10e-6, 22e-6),
           "24-36": (24e-6, 36e-6)}
AZ30 = tuple(range(0, 360, 12))

# The twelve tessellations with a cached label volume at ppw 8, as in
# axis_window_adjudication.SPECS.
SPECS = ([("girdle_seed11_ppw8_dev", 11, "k-8")] +
         [("girdle_seed%d_ppw8_ensemble" % s, s, "k-8")
          for s in (7, 17, 23, 41, 53, 71, 89)] +
         [("singlemax_seed11_ppw8_twin", 11, "k3.93")] +
         [("singlemax_seed%d_ppw8_ensemble" % s, s, "k3.93") for s in (17, 23, 41)])

CONTROLS = ("girdle_seed11_ppw8_uniform_axis", "girdle_seed11_ppw8_contrast_f000")

# Orientation-blind column descriptors.  Not one of these can be
# computed differently if every c-axis in the disc is replaced.
GEOM = ("n_grain", "n_cross", "vol_mean", "vol_cv", "chord_mm", "d_first",
        "sep_mean", "vol_max", "facet_align", "facet_area", "geom_dir",
        "n_pair", "eff_grain")
LAM, DG = C_REF / F0, 17.4e-3
KWAVE = 2 * np.pi / LAM


# ───────────────────────────────── levels ────────────────────────────────
def _bp(x, fs):
    sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype="band",
                 output="sos")
    return sosfiltfilt(sos, x)


_LV = {}


def levels(name, win, kind="env", az_keep=None):
    """Azimuthal level in dB.

    kind 'env' is beam_descriptors.measure: rms of the band-passed
    envelope over the window, referenced to the sweep-mean backwall
    peak.  This is the estimator the published r is on.  kind 'tf' is
    the audited tab:reconcile level, 2<x^2> in band per-azimuth
    source-referenced.  kind 'abs' is the same in-band power with NO
    reference at all, which is what a cross-sweep dB margin has to be
    read on.
    """
    key = (name, win, kind, az_keep)
    if key in _LV:
        return _LV[key]
    d = os.path.join(SWD, name)
    rots, val, e1 = [], [], []
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
        k0, w = int(2 * DIA / C_REF * fs), int(2e-6 * fs)
        if k0 + w >= len(tr):
            continue
        i0, i1 = int(win[0] * fs), int(win[1] * fs)
        xb = _bp(tr, fs)
        if kind == "env":
            val.append((np.abs(hilbert(xb))[i0:i1] ** 2).mean())
            e1.append(np.abs(hilbert(tr))[max(k0 - w, 0):k0 + w].max())
        elif kind == "tf":
            val.append(2.0 * (xb[i0:i1] ** 2).mean()
                       / np.abs(hilbert(tr)).max() ** 2)
            e1.append(1.0)
        else:
            val.append(2.0 * (xb[i0:i1] ** 2).mean())
            e1.append(1.0)
        rots.append(a)
    val, e1 = np.array(val), np.array(e1)
    ref = e1.mean() ** 2 if kind == "env" else 1.0
    out = (np.array(rots), 10 * np.log10(val / ref))
    _LV[key] = out
    return out


# ──────────────────────────── the beam column ────────────────────────────
_TC, _COL = {}, {}


def tess(seed, kappa):
    return tess_path("tess_s%d_p8_%s.npz" % (seed, kappa))


def tess_path(path):
    if path not in _TC:
        with np.load(os.path.join(TESS, path)) as z:
            _TC[path] = (np.asarray(z["labels"]),
                         np.asarray(z["axes"], float),
                         np.asarray(z["seeds"], float), float(z["h"]))
    return _TC[path]


def column(tkey, rots, win):
    """Per-azimuth grain weights and orientation-blind geometry.

    W[i]  normalised count of sampled points falling in each grain, so
          that the sample variance of the qP speed over the column is
          exactly  W@v^2 - (W@v)^2.
    """
    key = (tkey, tuple(rots), win)
    if key in _COL:
        return _COL[key]
    lab, ax, sd, h = tess_path(tkey) if isinstance(tkey, str) \
        else tess(*tkey)
    ng = len(ax)
    nx, ny, nz = lab.shape
    d0, d1 = win[0] * C_REF / 2, win[1] * C_REF / 2
    s = np.arange(d0, d1, h / 2)
    off = np.arange(-ELEM / 2, ELEM / 2 + h, h)
    zc = (np.arange(nz) + 0.5) * h - nz * h / 2
    zk = np.arange(0, nz, max(1, int(round(2e-3 / h))))
    zoff = np.stack([np.zeros_like(zc[zk]), np.zeros_like(zc[zk]),
                     zc[zk]], 1)[None, None, :, :]
    W, G, CR = [], {k: [] for k in GEOM}, []
    for r in rots:
        a = np.radians(float(r))
        n = np.array([np.cos(a), np.sin(a), 0.0])
        t = np.array([-np.sin(a), np.cos(a), 0.0])
        P = ((DIA / 2 * n - s[:, None] * n)[:, None, None, :]
             + off[None, :, None, None] * t + zoff)
        gi = np.rint((P[..., 0] + nx * h / 2) / h - 0.5).astype(int)
        gj = np.rint((P[..., 1] + ny * h / 2) / h - 0.5).astype(int)
        gk = np.rint((P[..., 2] + nz * h / 2) / h - 0.5).astype(int)
        ok = ((gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
              & (gk >= 0) & (gk < nz))
        L = np.where(ok, lab[np.clip(gi, 0, nx - 1), np.clip(gj, 0, ny - 1),
                             np.clip(gk, 0, nz - 1)], -1)
        good = L >= 0
        ids = L[good]
        cnt = np.bincount(ids, minlength=ng).astype(float)
        W.append(cnt / cnt.sum())
        nz_cnt = cnt[cnt > 0]
        A, B = L[:-1], L[1:]
        cross = (A != B) & (A >= 0) & (B >= 0)
        ia, ib = A[cross].astype(int), B[cross].astype(int)
        CR.append((ia, ib))
        # facet geometry: the boundary normal is the Laguerre seed
        # difference.  No c-axis enters any of these.
        nmv = sd[ib] - sd[ia]
        sep = np.linalg.norm(nmv, axis=1)
        nmv = nmv / (sep[:, None] + 1e-30)
        al = np.abs(nmv @ n)
        st = np.sqrt(np.clip(1.0 - al ** 2, 0.0, 1.0))
        x = KWAVE * DG * st
        from scipy.special import j1
        dd = np.where(x < 1e-9, 1.0, 2 * j1(np.maximum(x, 1e-9))
                      / np.maximum(x, 1e-9)) ** 2
        G["facet_align"].append(float(al.mean()) if al.size else 0.0)
        G["facet_area"].append(float((1.0 / np.maximum(al, 0.05)).sum())
                               if al.size else 0.0)
        G["geom_dir"].append(float(dd.sum()) if al.size else 0.0)
        G["n_pair"].append(float(len(np.unique(np.minimum(ia, ib) * (ng + 1)
                                               + np.maximum(ia, ib)))))
        w = cnt[cnt > 0]
        G["eff_grain"].append(float(w.sum() ** 2 / (w ** 2).sum()))
        G["n_grain"].append(len(nz_cnt))
        G["vol_mean"].append(nz_cnt.mean())
        G["vol_cv"].append(nz_cnt.std() / nz_cnt.mean())
        G["vol_max"].append(nz_cnt.max() / nz_cnt.sum())
        G["n_cross"].append(int(cross.sum()))
        G["chord_mm"].append(float(good.sum(0).mean()) * (h / 2) * 1e3)
        idx = np.argmax(good, axis=0).astype(float)
        G["d_first"].append(float(idx[good.any(0)].mean()) * (h / 2) * 1e3)
        G["sep_mean"].append(float(good.sum()) / max(1, int(cross.sum())))
    out = (np.array(W), {k: np.array(v, float) for k, v in G.items()}, CR)
    _COL[key] = out
    return out


def vmat(ax, rots):
    """v[i, g] = qP speed of grain g for beam direction rots[i]."""
    out = np.empty((len(rots), len(ax)))
    for i, r in enumerate(rots):
        a = np.radians(float(r))
        n = np.array([np.cos(a), np.sin(a), 0.0])
        out[i] = np.interp(np.arccos(np.clip(np.abs(ax @ n), 0, 1)),
                           F._PSI, F._VQP)
    return out


def v_var(W, V, perm=None):
    Vp = V if perm is None else V[:, perm]
    m1 = (W * Vp).sum(1)
    return (W * Vp ** 2).sum(1) - m1 ** 2


# ──────────────────────────────── the null ───────────────────────────────
def shift(x, y):
    """r at every circular shift of y.  rs[0] is the true alignment."""
    xc = x - x.mean()
    n = len(y)
    out = np.empty(n)
    for k in range(n):
        yc = np.roll(y, k)
        yc = yc - yc.mean()
        d = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
        out[k] = (xc * yc).sum() / d if d > 0 else 0.0
    return out


def rank_of(rs):
    k = int((np.abs(rs) >= abs(rs[0])).sum())
    return float(rs[0]), k, k / len(rs)


def resid(x, G, keys):
    cols = [G[k] - G[k].mean() for k in keys if G[k].std() > 1e-12]
    A = np.column_stack([np.ones(len(x))] + cols)
    b, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ b


def partial(x, y, z):
    """Partial correlation of x and y given z."""
    rx = resid(x, {"z": z}, ["z"])
    ry = resid(y, {"z": z}, ["z"])
    return float(np.corrcoef(rx, ry)[0, 1])


def fisher(ps):
    ps = np.clip(np.asarray(ps, float), 1e-12, 1.0)
    x2 = -2.0 * np.log(ps).sum()
    return x2, float(chi2.sf(x2, 2 * len(ps)))


def sign_p(a, b):
    d = np.sign(np.asarray(a) - np.asarray(b))
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    from math import comb
    k = min(pos, neg)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return pos, neg, min(1.0, p)


# ═════════════════════════════════ report ════════════════════════════════
def main(nperm=2000):
    t0 = time.time()
    store = {}
    rng = np.random.default_rng(20260803)

    print("=" * 72)
    print("1.  VALIDATION.  Is this the published statistic?")
    print("=" * 72)
    print("    Gate 24-36 us, seed 11, level = beam_descriptors.measure.")
    print("    %-6s %-8s %8s %10s %10s" % ("res", "stat", "r", "rank/60",
                                           "rank/30 fold"))
    val = {}
    for sw, tk, tag in (("girdle_seed11_ppw6_axis_perp", "labels_seed11_ppw6_kappa-8.npz", "ppw6"),
                        ("girdle_seed11_ppw8_dev", "labels_seed11_ppw8_kappa-8.npz", "ppw8")):
        rots, lev = levels(sw, WINDOWS["24-36"], "env")
        W, G, CR = column(tk, rots, WINDOWS["24-36"])
        _, ax, _, _ = tess_path(tk)
        V = vmat(ax, rots)
        vv = v_var(W, V)
        rs = shift(vv, lev)
        n = len(rots)
        fold = np.maximum(np.abs(rs[:n // 2]), np.abs(rs[n // 2:]))
        print("    %-6s %-8s %+8.4f %10s %10s"
              % (tag, "v_var", rs[0], "%d/%d" % (rank_of(rs)[1], n),
                 "%d/%d" % (int((fold >= abs(rs[0])).sum()), n // 2)))
        # competing readings of the same English sentence
        Wd = (W > 0).astype(float)
        Wd /= Wd.sum(1, keepdims=True)
        rd = np.corrcoef(v_var(Wd, V), lev)[0, 1]
        print("    %-6s %-8s %+8.4f  (one vote per distinct grain)"
              % ("", "v_var_d", rd))
        val[tag] = (rs, vv, lev, rots, W, G, V, ax)
        store["shift_%s" % tag] = rs
    print()
    print("    published: r = 0.57 (ppw 6) and 0.36 (ppw 8), rank first of")
    print("    thirty at both.  The r reproduces to three decimals.  The")
    print("    rank does not: see part 2.")

    print()
    print("=" * 72)
    print("2.  THE SHIFT NULL IS NOT THIRTY INDEPENDENT ALIGNMENTS")
    print("=" * 72)
    for tag in ("ppw6", "ppw8"):
        rs, vv, lev, rots, W, G, V, ax = val[tag]
        n = len(rs)
        o = np.argsort(-np.abs(rs))[:4]
        print("    %s  top shifts:  %s" % (
            tag, "   ".join("%d deg %+0.4f" % (6 * k, rs[k]) for k in o)))
        # periodicity of the predictor and of the level
        print("       r(pred, pred+180) = %+0.3f    "
              "azimuthal autocorrelation of pred at 6/12 deg = %+0.3f/%+0.3f"
              % (np.corrcoef(vv, np.roll(vv, n // 2))[0, 1],
                 np.corrcoef(vv, np.roll(vv, 1))[0, 1],
                 np.corrcoef(vv, np.roll(vv, 2))[0, 1]))
    print("    The bulk predictor is a function of |c.n| and is exactly")
    print("    180-periodic, so 60 azimuths give 30 alignments.  This one")
    print("    is not: its column covers a fixed depth range from the rim,")
    print("    so az+180 holds the other half of the chord.  Every shift is")
    print("    a distinct alignment and the floor is 1/60, not 1/30.")

    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("3.  CONTROLS.  Specimens that cannot backscatter at all.")
    print("=" * 72)
    print("    GRID 60 = the 60 production azimuths (specimen and")
    print("    zerocontrast only).  GRID 30 = the 30 azimuths at 12 deg")
    print("    that cs_f000 also holds.  Ranks are never compared across")
    print("    grids.")
    tk8 = "labels_seed11_ppw8_kappa-8.npz"
    _, ax8, _, _ = tess_path(tk8)
    rows = []
    for grid, keep in (("60", None), ("30", AZ30)):
        sweeps = (("girdle_seed11_ppw8_dev", "specimen"),
                  ("girdle_seed11_ppw8_uniform_axis", "control"))
        if grid == "30":
            sweeps = sweeps + (("girdle_seed11_ppw8_contrast_f000", "control"),)
        for sw, role in sweeps:
            for wn, wv in WINDOWS.items():
                for kind in ("env", "tf"):
                    rots, lev = levels(sw, wv, kind, keep)
                    W, G, CR = column(tk8, rots, wv)
                    V = vmat(ax8, rots)
                    rs = shift(v_var(W, V), lev)
                    r, k, p = rank_of(rs)
                    # gate-fixed column, window-varying level
                    Wg, Gg, _ = column(tk8, rots, WINDOWS["24-36"])
                    rg = np.corrcoef(v_var(Wg, V), lev)[0, 1]
                    rows.append((grid, sw, role, wn, kind, r, k, len(rots),
                                 rg))
    print("    %-5s %-19s %-6s %-5s %7s %9s %8s" %
          ("grid", "sweep", "window", "lvl", "r", "rank", "r gate-col"))
    for g, sw, role, wn, kind, r, k, n, rg in rows:
        print("    %-5s %-19s %-6s %-5s %+7.3f %9s %+8.3f"
              % (g, sw, wn, kind, r, "%d/%d" % (k, n), rg))
    store["cells"] = np.array([(g, sw, wn, kind, r, k, n, rg)
                               for g, sw, _, wn, kind, r, k, n, rg in rows],
                              dtype=object)

    print()
    print("    MARGIN, control against specimen at the true alignment,")
    print("    like for like inside a grid (same window, same estimator,")
    print("    same window-matched column):")
    print("    %-6s %-5s %8s %8s %8s %8s" %
          ("window", "lvl", "spec", "zeroc", "cs_f000", "verdict"))
    nwin = {w: [0, 0] for w in WINDOWS}
    for wn in WINDOWS:
        for kind in ("env", "tf"):
            got = {}
            for g, sw, role, w2, k2, r, k, n, rg in rows:
                if w2 == wn and k2 == kind:
                    got.setdefault(sw, {})[g] = abs(r)
            sp = got["girdle_seed11_ppw8_dev"]
            zc = got["girdle_seed11_ppw8_uniform_axis"]
            cs = got["girdle_seed11_ppw8_contrast_f000"]
            for g in ("60", "30"):
                if g not in sp:
                    continue
                cands = [zc[g]] + ([cs[g]] if g in cs else [])
                win = sum(1 for c in cands if c >= sp[g])
                nwin[wn][0] += win
                nwin[wn][1] += len(cands)
                print("    %-6s %-5s grid%-3s %7.3f %8.3f %8s   %s"
                      % (wn, kind, g, sp[g], zc[g],
                         "%.3f" % cs[g] if g in cs else "  -",
                         "CONTROL WINS x%d" % win if win else "clean"))
    print()
    for wn in WINDOWS:
        print("    %-6s controls beating the specimen: %d of %d cells"
              % (wn, nwin[wn][0], nwin[wn][1]))

    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("4.  IS THERE ANY FABRIC-BORNE LEVEL TO CORRELATE WITH?")
    print("=" * 72)
    print("    Absolute in-band power, no reference of any kind, on the")
    print("    thirty common azimuths.  dB re 1.")
    print("    %-6s %9s %9s %9s %9s %9s" %
          ("window", "spec", "zeroc", "cs_f000", "marg zc", "marg cs"))
    for wn, wv in WINDOWS.items():
        a = [10 * np.log10(np.mean(10 ** (levels(s, wv, "abs", AZ30)[1] / 10)))
             for s in ("girdle_seed11_ppw8_dev",) + CONTROLS]
        print("    %-6s %9.2f %9.2f %9.2f %9.2f %9.2f"
              % (wn, a[0], a[1], a[2], a[0] - a[1], a[0] - a[2]))
    print("    (An 'env' dB margin is referenced to each sweep's own")
    print("    backwall and is not a cross-sweep margin; this one is.)")

    print()
    print("    THE RENDERING TEST.  A zero-scattering record on the SAME")
    print("    tessellation has an azimuthal level that is pure rendering.")
    print("    If it already predicts the specimen's level, the specimen's")
    print("    azimuthal variation is not scattering.")
    print("    %-6s %-5s %10s %10s %12s %12s" %
          ("window", "lvl", "r spec~zc", "r spec~cs", "r pred~lvl",
           "partial|zc"))
    for wn, wv in WINDOWS.items():
        for kind in ("env", "tf"):
            _, ls = levels("girdle_seed11_ppw8_dev", wv, kind, AZ30)
            _, lz = levels("girdle_seed11_ppw8_uniform_axis", wv, kind, AZ30)
            _, lc = levels("girdle_seed11_ppw8_contrast_f000", wv, kind, AZ30)
            rots = levels("girdle_seed11_ppw8_dev", wv, kind, AZ30)[0]
            W, G, CR = column(tk8, rots, wv)
            vv = v_var(W, vmat(ax8, rots))
            print("    %-6s %-5s %+10.3f %+10.3f %+12.3f %+12.3f"
                  % (wn, kind, np.corrcoef(ls, lz)[0, 1],
                     np.corrcoef(ls, lc)[0, 1], np.corrcoef(vv, ls)[0, 1],
                     partial(vv, ls, lz)))

    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("5.  ATTACK B.  Collinearity of the predictor with the")
    print("    orientation-blind geometry of its own column.")
    print("=" * 72)
    print("    R2 of v_var regressed on the eight column descriptors, and")
    print("    the single strongest of them, GRID 60, seed 11 ppw 8.")
    print("    %-6s %8s %-11s %8s" % ("window", "R2", "worst one", "r"))
    for wn, wv in WINDOWS.items():
        rots = levels("girdle_seed11_ppw8_dev", wv, "env")[0]
        W, G, CR = column(tk8, rots, wv)
        vv = v_var(W, vmat(ax8, rots))
        rr = resid(vv, G, GEOM)
        r2 = 1.0 - rr.var() / vv.var()
        cc = {k: np.corrcoef(vv, G[k])[0, 1] for k in GEOM if G[k].std() > 0}
        b = max(cc, key=lambda k: abs(cc[k]))
        print("    %-6s %8.3f %-11s %+8.3f" % (wn, r2, b, cc[b]))
        print("         " + "  ".join("%s %+.2f" % (k, v)
                                      for k, v in sorted(
                                          cc.items(),
                                          key=lambda t: -abs(t[1]))))

    print()
    print("=" * 72)
    print("6.  ATTACK A.  Does anything orientation-dependent survive")
    print("    projecting that geometry out?")
    print("=" * 72)
    print("    GRID 60, specimen, both estimators.  'best geom' is the")
    print("    largest |r| over the eight blind descriptors with its own")
    print("    exact circular-shift family-wise p over the eight.")
    print("    %-6s %-5s %8s %8s %10s %8s %8s %10s" %
          ("window", "lvl", "r pred", "rank", "r resid", "rank",
           "best geom", "p_fw geom"))
    geo_cells = 0
    for wn, wv in WINDOWS.items():
        for kind in ("env", "tf"):
            rots, lev = levels("girdle_seed11_ppw8_dev", wv, kind)
            W, G, CR = column(tk8, rots, wv)
            vv = v_var(W, vmat(ax8, rots))
            rs = shift(vv, lev)
            rr = shift(resid(vv, G, GEOM), lev)
            live = [k for k in GEOM if G[k].std() > 1e-12]
            geo_cells += len(live)
            SS = np.array([shift(G[k], lev) for k in live])
            mx = np.abs(SS).max(0)
            bi = int(np.argmax(np.abs(SS[:, 0])))
            pfw = float((mx >= np.abs(SS[:, 0]).max()).sum()) / len(rots)
            print("    %-6s %-5s %+8.3f %8s %+10.3f %8s %8s %10.3f"
                  % (wn, kind, rs[0], "%d/%d" % (rank_of(rs)[1], len(rots)),
                     rr[0], "%d/%d" % (rank_of(rr)[1], len(rots)),
                     "%s %+.2f" % (live[bi][:6], SS[bi, 0]), pfw))
    print()
    print("    live orientation-blind cells scored: %d" % geo_cells)

    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("7.  THE ORIENTATION-PERMUTATION CONTROL, AND WHAT IT CANNOT DO")
    print("=" * 72)
    print("    Permutes which grain carries which c-axis with the")
    print("    tessellation, the beam sampling and the orientation")
    print("    multiset held fixed.  %d draws." % nperm)
    print("    %-6s %-5s %-6s %8s %10s %12s" %
          ("res", "lvl", "window", "|r|", "p perm", "p perm resid"))
    perm_gate = {}
    for sw, tk, tag in (("girdle_seed11_ppw6_axis_perp", "labels_seed11_ppw6_kappa-8.npz", "ppw6"),
                        ("girdle_seed11_ppw8_dev", "labels_seed11_ppw8_kappa-8.npz", "ppw8")):
        _, axl, _, _ = tess_path(tk)
        for wn, wv in WINDOWS.items():
            rots, lev = levels(sw, wv, "env")
            W, G, CR = column(tk, rots, wv)
            V = vmat(axl, rots)
            r0 = abs(np.corrcoef(v_var(W, V), lev)[0, 1])
            rr0 = abs(np.corrcoef(resid(v_var(W, V), G, GEOM), lev)[0, 1])
            hit = hitr = 0
            ng = V.shape[1]
            for _ in range(nperm):
                pm = rng.permutation(ng)
                vp = v_var(W, V, pm)
                if abs(np.corrcoef(vp, lev)[0, 1]) >= r0:
                    hit += 1
                if abs(np.corrcoef(resid(vp, G, GEOM), lev)[0, 1]) >= rr0:
                    hitr += 1
            p, pr = (hit + 1) / (nperm + 1), (hitr + 1) / (nperm + 1)
            print("    %-6s %-5s %-6s %8.3f %10.4f %12.4f"
                  % (tag, "env", wn, r0, p, pr))
            if wn == "24-36":
                perm_gate[tag] = p
    print("    published, this gate: p = 0.005 (ppw 6) and 0.070 (ppw 8).")
    print("    The control moves the c-axes but never moves the specimen")
    print("    relative to the beam, so it cannot tell an orientation")
    print("    channel from a geometry channel that orientation modulates.")

    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("8.  THE ENSEMBLE.  Twelve tessellations, GRID 30.")
    print("=" * 72)
    ens = {}
    for wn, wv in WINDOWS.items():
        for kind in ("env", "tf"):
            ranks, rvals, ps = [], [], []
            for nm, seed, kap in SPECS:
                rots, lev = levels(nm, wv, kind, AZ30)
                W, G, CR = column((seed, kap), rots, wv)
                _, axl, _, _ = tess(seed, kap)
                rs = shift(v_var(W, vmat(axl, rots)), lev)
                r, k, p = rank_of(rs)
                ranks.append(k)
                rvals.append(abs(r))
                ps.append(p)
            ens[(wn, kind)] = (np.array(ranks), np.array(rvals), np.array(ps))
    print("    %-6s %-5s %8s %10s %10s %12s" %
          ("window", "lvl", "first/12", "rank sum", "Fisher p", "Binom p"))
    from math import comb
    for k, (ranks, rv, ps) in ens.items():
        n1 = int((ranks == 1).sum())
        pb = sum(comb(12, i) * (1 / 30) ** i * (29 / 30) ** (12 - i)
                 for i in range(n1, 13))
        print("    %-6s %-5s %8d %10d %10.4f %12.4f"
              % (k[0], k[1], n1, int(ranks.sum()), fisher(ps)[1], pb))
    print("    chance first count = 0.4 of 12; chance rank sum = 186.")
    print()
    print("    PAIRED ON THE TESSELLATION, the unit of replication.")
    print("    %-6s %-5s %8s %8s %8s %10s %10s" %
          ("window", "lvl", "better", "worse", "tied", "sign p", "wilcoxon"))
    for kind in ("env", "tf"):
        gr = ens[("24-36", kind)][0]
        for wn in ("4-16", "10-22"):
            er = ens[(wn, kind)][0]
            pos, neg, sp = sign_p(gr, er)          # better = lower rank
            d = gr.astype(float) - er
            try:
                wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except Exception:
                wp = 1.0
            print("    %-6s %-5s %8d %8d %8d %10.3f %10.3f"
                  % (wn, kind, pos, neg, 12 - pos - neg, sp, wp))

    print()
    print("    ENSEMBLE AFTER PROJECTING THE BLIND GEOMETRY OUT:")
    print("    %-6s %-5s %8s %10s %10s" %
          ("window", "lvl", "first/12", "rank sum", "Fisher p"))
    ensr = {}
    for wn, wv in WINDOWS.items():
        for kind in ("env", "tf"):
            ranks, ps = [], []
            for nm, seed, kap in SPECS:
                rots, lev = levels(nm, wv, kind, AZ30)
                W, G, CR = column((seed, kap), rots, wv)
                _, axl, _, _ = tess(seed, kap)
                vv = v_var(W, vmat(axl, rots))
                rs = shift(resid(vv, G, GEOM), lev)
                r, k, p = rank_of(rs)
                ranks.append(k)
                ps.append(p)
            ranks = np.array(ranks)
            ensr[(wn, kind)] = ranks
            print("    %-6s %-5s %8d %10d %10.4f"
                  % (wn, kind, int((ranks == 1).sum()), int(ranks.sum()),
                     fisher(ps)[1]))

    print()
    print("=" * 72)
    print("9.  MULTIPLICITY, PAID.")
    print("=" * 72)
    print("    control cells scored here: %d.  Under a true null with a"
          % sum(n for n in (v[1] for v in nwin.values())))
    print("    1/30 floor, P(at least one rank 1) = %.2f, so no conclusion"
          % (1 - (29 / 30) ** 12))
    print("    rests on a rank-1 control; the statistic used is the paired")
    print("    |r| at the TRUE alignment, one comparison per cell.")
    print("    ensemble cells scored: 2 predictors x 2 estimators x 3")
    print("    windows = 12 before the permutation control, of which the")
    print("    manuscript admits 8.  Sidak on 12: a nominal 0.0065 is")
    print("    %.3f, and a nominal 0.0005 is %.4f."
          % (1 - (1 - 0.0065) ** 12, 1 - (1 - 0.0005) ** 12))

    np.savez_compressed(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "beam_local_break.npz"),
        **{("ens_%s_%s" % k): v[0] for k, v in ens.items()},
        **{("ensr_%s_%s" % k): v for k, v in ensr.items()})
    print()
    print("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
