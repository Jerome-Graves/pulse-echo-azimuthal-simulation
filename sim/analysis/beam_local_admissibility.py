"""Is the BEAM-LOCAL predictor of Section 5.2 admissible in the earlier
windows, where the bulk predictor is not?

THE QUESTION. Section 5.2 closes with a sentence that rests on one
statistic: "A weak orientation channel is therefore visible in a window
this paper does not analyse in." The statistic is the beam-local
predictor, the variance of the quasi-longitudinal speed over the grains
the beam column intersects inside the window, scored against the
measured level under a circular-shift null. The bulk predictor, the
predicted mean square of dv/2vbar over every grain pair in the disc, was
shown by analysis/axis_window_adjudication.py to be INADMISSIBLE at 4 to
16 and 10 to 22 us: there it tracks specimens that cannot backscatter
from a grain boundary at all better than it tracks the specimen. This
module asks whether the beam-local predictor shares that defect, since
an inadmissible window cannot support a positive claim any more than it
can confirm a null.

WHAT THE PREDICTOR IS, ESTABLISHED BY MEASUREMENT AND NOT BY ASSUMPTION.
The predictor was rebuilt here from the description alone: march the
beam column through the cached label volume over the depth range the
window listens to, collect the grains the column intersects, and take
the variance of their qP speed for that beam direction. Two free choices
in that description are settled by which one returns the published
number, and both are printed in report_harness:

  COLUMN     a 6.35 mm cylinder of the element width, sampled every h
             laterally and every 2 mm in z, or the 8.9 degree diverging
             cone of analysis/length_scales.py. The published r is the
             CYLINDER. The cone, which is the physically correct far
             field of this aperture, returns r = 0.50 and 0.28, ranks
             5th and 4th, and would not support the sentence at all.
  WEIGHTING  variance over the sample points, so each grain enters
             weighted by the volume of it the column intersects, or
             variance over the distinct grains. The published r is the
             SAMPLE-WEIGHTED one. Unweighted returns 0.32 and 0.09.

So the published statistic is one of four readings of its own sentence,
and it is the strongest of the four. That is recorded here and not
argued about; the tests below are run on the published reading.

THE PERIODICITY, which decides how many alignments a sweep offers. A
predictor that sees the c-axes only through |c . n| is exactly invariant
under a beam reversal, so a 60-azimuth sweep offers 30 distinct
alignments of it and a 30-azimuth sweep offers 15. The beam-local
predictor is NOT such a quantity: at az + 180 the column enters the disc
on the opposite side and the window covers a different part of the
chord. report_periodicity measures r(x_az, x_az+180) = 0.59 to 0.60, so
the halving is not available to it, every shift is a distinct alignment,
and the honest floor is 1/60 on a 60-azimuth sweep and 1/30 on a
30-azimuth one. Both counts are printed everywhere: rank/n is the honest
one and rank/(n/2) is the convention the published sentence used.

WHAT IS MEASURED, AND WHAT WAS FOUND.

  0. HARNESS. The published beam-local numbers come back through this
     path, r to two decimals at both resolutions. The rank does not: at
     ppw 6 the true alignment is beaten by its 6-degree neighbour,
     0.592 against 0.573, so this rebuild returns 2nd of 30 where the
     manuscript reports 1st. The bulk-predictor control numbers of
     axis_window_adjudication are also reproduced exactly, which is what
     licenses comparing the two predictors cell by cell.

  1. THE DECISIVE TEST, AND THE ANSWER IS THE SAME AS FOR THE BULK
     PREDICTOR. On the thirty azimuths the specimen and both
     zero-scattering controls share, untreated, |r| at the true
     alignment runs specimen 0.17, zerocontrast 0.44, cs_f000 0.49 at
     4 to 16 us; specimen 0.01, 0.36, 0.42 at 10 to 22 us; and specimen
     0.21, 0.16, 0.14 in the gate. Both controls beat the specimen in
     both earlier windows, under both treatments, and neither beats it
     in the gate untreated. On the native 60-azimuth grid the gate is
     unambiguous: the specimen reaches 0.354 at first rank of sixty
     where zerocontrast reaches 0.001 at sixtieth. The beam-local
     predictor is inadmissible exactly where the bulk one is.

     A second fact settles 4 to 16 us on its own. There the specimen's
     revolution level is -70.67 dB against -71.18 and -70.69 dB for two
     specimens that cannot backscatter at all, so 99 per cent of what is
     being correlated there is a quantity all three share and no fabric
     channel is present to be read.

  2. PURE GEOMETRY. The manuscript's "do not track the level at all"
     does not survive re-measurement in any window. In the gate the
     summed facet directivity, computed with every velocity jump set to
     one and therefore with no c-axis in it, reaches |r| = 0.41 at first
     rank of sixty on the cylinder column and 0.58 of thirty on the
     30-azimuth grid, ABOVE the beam-local predictor's 0.35 and 0.21. In
     the earlier windows pure geometry also outscores the beam-local
     predictor on the specimen, 0.21 against 0.13 at 4 to 16 and 0.31
     against 0.00 at 10 to 22. Nothing in the geometry family survives
     its own family-wise null earlier, p_fw = 0.32 to 0.90, but neither
     does the c-axis predictor, and the orientation reading of the
     earlier window is therefore not identified.

  3. THE ORIENTATION-PERMUTATION CONTROL reproduces, p = 0.0050 at
     ppw 6 against a published 0.005 and 0.060 at ppw 8 against 0.070,
     and on the ensemble it passes at Fisher p = 0.0002 at 10 to 22 us
     against a published 0.0010, and at 0.029 in the gate against a
     published 0.078. It cannot rescue the reading. It is evaluated at
     the true registration and holds the tessellation fixed, so it
     establishes that the statistic depends on which grain carries which
     c-axis. It cannot separate an orientation channel from the
     rendering of the tessellation, which is what the controls of test 1
     show the earlier windows are reading.

  4. THE ENSEMBLE. First ranks of twelve are 2 at 4 to 16 us, 3 at
     10 to 22 us and 1 in the gate, close to the published 3 or 4
     against 1 or 2. Paired on the tessellation, which is the unit of
     replication, no window separates from the gate: sign p = 1.00 and
     Wilcoxon p = 0.88 to 0.97. A cross-tessellation control finds the
     10 to 22 us ensemble hits are specimen-specific, 3 of 12 matched
     against 1 of 132 mismatched, Fisher p = 0.0017, but the matched
     cell shares the TESSELLATION as well as the c-axes with the
     measured level, so that control separates specimen from specimen
     and not orientation from geometry.

TOUCHES NO CUDA. Nothing here calls fdtd.forward_*, DiskSpecimen.build,
or any solver path. Every tessellation quantity is read from
out/tesscache/tess_s<seed>_p<ppw>_k<kappa>.npz and the module raises
SystemExit rather than build a missing one. sim/model/forward.py is imported
for its tabulated qP velocity surface alone and imports numpy only.

READS   out/sweeps/<name>/az*.npz          trace and dt
        out/tesscache/tess_s*_p*_k*.npz    labels, axes, seeds
WRITES  stdout, and beam_local_admissibility.npz beside this file.

Run with `python beam_local_admissibility.py`; `--quick` cuts the
permutation draw counts by ten for a smoke test.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import j1
from scipy.stats import binom, binomtest, chi2, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SWD = os.path.join(ROOT, "out", "sweeps")
TESS = os.path.join(ROOT, "out", "tesscache")
sys.path.insert(0, os.path.dirname(HERE))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

import forward as FW                                      # noqa: E402

# Acquisition, Section 3 and Table 1. Nothing here is fitted.
C_REF, F0, DIA, THK = 3850.0, 2.0e6, 0.100, 0.035
BAND = (0.8e6, 3.0e6)
ELEM = 6.35e-3
LAM = C_REF / F0
KWAVE = 2 * np.pi / LAM
DG = 17.4e-3                       # volume-equivalent grain, length_scales
HALF = np.radians(8.9)             # far-field half angle of the element
ZSTEP = 2e-3                       # z planes of the sampled column

GATE = (24e-6, 36e-6)
WINDOWS = {"4-16": (4e-6, 16e-6), "10-22": (10e-6, 22e-6), "24-36": GATE}
AZ30 = tuple(range(0, 360, 12))

# The two specimens that cannot backscatter from a grain boundary. Both
# carry the seed-11 tessellation the predictor is built from.
CONTROLS = ("girdle_seed11_ppw8_uniform_axis", "girdle_seed11_ppw8_contrast_f000")

# The twelve tessellations with a cached label volume at ppw 8, exactly
# the set of analysis/axis_window_adjudication.py.
SPECS = ([("girdle_seed11_ppw8_dev", 11, "-8")] +
         [("girdle_seed%d_ppw8_ensemble" % s, s, "-8")
          for s in (7, 17, 23, 41, 53, 71, 89)] +
         [("singlemax_seed11_ppw8_twin", 11, "3.93")] +
         [("singlemax_seed%d_ppw8_ensemble" % s, s, "3.93") for s in (17, 23, 41)])

GEOM_KEYS = ("n_cross", "n_grain", "n_pair", "eff_grain", "vol_cv",
             "geom_dir", "d_first", "sep_mean")

PAD, DTG = 4e-6, 2e-8              # common time grid of the loo path

_SW, _COL = {}, {}


# ───────────────────────────────── loading ───────────────────────────────
def bandpass(x, fs):
    sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype="band",
                 output="sos")
    return sosfiltfilt(sos, x)


def load_sweep(name, az_keep=None):
    """Traces of one sweep, keyed by azimuth in degrees."""
    key = (name, az_keep)
    if key in _SW:
        return _SW[key]
    d = os.path.join(SWD, name)
    if not os.path.isdir(d):
        raise SystemExit("missing sweep %s" % d)
    out = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        a = int(f[2:5])
        if az_keep is not None and a not in az_keep:
            continue
        with np.load(os.path.join(d, f)) as z:
            out[a] = (np.asarray(z["trace"], float).ravel(), float(z["dt"]))
    _SW[key] = out
    return out


def load_tess(seed, ppw, kap):
    p = os.path.join(TESS, "labels_seed%d_ppw%d_kappa%s.npz" % (seed, ppw, kap))
    if not os.path.exists(p):
        raise SystemExit("missing %s; refusing to build it (CUDA)" % p)
    with np.load(p) as z:
        return (z["labels"], np.asarray(z["axes"], float),
                np.asarray(z["seeds"], float), float(z["h"]))


# ──────────────────────────────── the level ──────────────────────────────
def level_none(sw, win):
    """Audited local in-band level, dB, at each azimuth's own dt."""
    az = sorted(sw)
    out = []
    for a in az:
        tr, dt = sw[a]
        fs = 1.0 / dt
        i0, i1 = int(win[0] * fs), int(win[1] * fs)
        src2 = np.abs(hilbert(tr)).max() ** 2
        out.append(10 * np.log10(2.0 * (bandpass(tr, fs)[i0:i1] ** 2).mean()
                                 / src2))
    return np.array(az), np.array(out)


def rf_on_grid(sw, tg):
    az = sorted(sw)
    rows = []
    for a in az:
        tr, dt = sw[a]
        s = np.abs(hilbert(tr)).max()
        rows.append(np.interp(tg, np.arange(len(tr)) * dt,
                              bandpass(tr, 1.0 / dt) / s))
    return np.array(az), np.array(rows)


def loo_median(R):
    return np.array([np.median(np.delete(R, i, axis=0), axis=0)
                     for i in range(len(R))])


def level_loo(sw, win):
    """Level after the leave-one-out azimuthal median trace removal."""
    tg = np.arange(win[0] - PAD, win[1] + PAD, DTG)
    az, R = rf_on_grid(sw, tg)
    R = R - loo_median(R)
    m = (tg >= win[0]) & (tg < win[1])
    return az, 10 * np.log10(2.0 * (R[:, m] ** 2).mean(axis=1))


def level(sw, win, treat):
    return level_none(sw, win) if treat == "none" else level_loo(sw, win)


# ───────────────────────────── the beam column ───────────────────────────
def march(lab, seeds, h, rots, win, geom="cyl"):
    """Everything the beam column intersects, per azimuth.

    Returns counts[naz, ngrain], the number of column sample points
    falling in each grain, and a dict of c-axis-free geometric
    descriptors. The counts are all the fabric predictor needs, and they
    hold the beam sampling fixed under any permutation of the c-axes.
    """
    nx, ny, nz = lab.shape
    ng = int(lab.max()) + 1
    d0, d1 = win[0] * C_REF / 2, win[1] * C_REF / 2
    s = np.arange(d0, d1, h)
    if geom == "cyl":
        off = np.arange(-ELEM / 2, ELEM / 2 + h, h)
        zc = (np.arange(nz) + 0.5) * h - nz * h / 2
        zk = zc[np.arange(0, nz, max(1, int(round(ZSTEP / h))))]
    else:
        uu = np.linspace(-1, 1, 7)
        U, W = np.meshgrid(uu, uu, indexing="ij")
        keep = U ** 2 + W ** 2 <= 1.0
        U, W = U[keep], W[keep]
    counts = np.zeros((len(rots), ng))
    G = {k: [] for k in GEOM_KEYS}
    for i, r in enumerate(rots):
        a = np.radians(float(r))
        n = np.array([np.cos(a), np.sin(a), 0.0])
        t1 = np.array([-np.sin(a), np.cos(a), 0.0])
        t2 = np.array([0.0, 0.0, 1.0])
        if geom == "cyl":
            P = (DIA / 2 * n - s[:, None] * n)[:, None, None, :] \
                + off[None, :, None, None] * t1 \
                + np.stack([np.zeros_like(zk), np.zeros_like(zk), zk],
                           1)[None, None, :, :]
            P = P.reshape(len(s), -1, 3)
        else:
            rad = (s * np.tan(HALF))[:, None]
            P = (DIA / 2 * n - s[:, None] * n)[:, None, :] \
                + (rad * U[None, :])[:, :, None] * t1 \
                + (rad * W[None, :])[:, :, None] * t2
        gi = np.rint((P[..., 0] + nx * h / 2) / h - 0.5).astype(int)
        gj = np.rint((P[..., 1] + ny * h / 2) / h - 0.5).astype(int)
        gk = np.rint((P[..., 2] + nz * h / 2) / h - 0.5).astype(int)
        ok = ((gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
              & (gk >= 0) & (gk < nz))
        L = np.where(ok, lab[np.clip(gi, 0, nx - 1), np.clip(gj, 0, ny - 1),
                             np.clip(gk, 0, nz - 1)], -1)
        ids = L[L >= 0]
        c = np.bincount(ids, minlength=ng).astype(float)
        counts[i] = c
        w = c[c > 0]
        A, B = L[:-1], L[1:]
        cr = (A != B) & (A >= 0) & (B >= 0)
        ia, ib = A[cr], B[cr]
        nmv = seeds[ia] - seeds[ib]
        sep = np.linalg.norm(nmv, axis=1)
        nmv = nmv / (sep[:, None] + 1e-30)
        st = np.sqrt(np.clip(1.0 - (nmv @ n) ** 2, 0.0, 1.0))
        x = KWAVE * DG * st
        d = np.where(x < 1e-9, 1.0,
                     2 * j1(np.where(x < 1e-9, 1.0, x))
                     / np.where(x < 1e-9, 1.0, x)) ** 2
        pair = np.unique(np.minimum(ia, ib) * (ng + 1) + np.maximum(ia, ib))
        si = np.nonzero(cr.any(axis=tuple(range(1, cr.ndim))))[0]
        G["n_cross"].append(float(cr.sum()))
        G["n_grain"].append(float(len(w)))
        G["n_pair"].append(float(len(pair)))
        G["eff_grain"].append(float(w.sum() ** 2 / (w ** 2).sum()))
        G["vol_cv"].append(float(w.std() / w.mean()))
        G["geom_dir"].append(float(d.sum()))
        G["d_first"].append(float(s[si[0]]) if len(si) else float(s[-1]))
        G["sep_mean"].append(float(sep.mean()) if len(sep) else 0.0)
    return counts, {k: np.array(v, float) for k, v in G.items()}


def column(seed, ppw, kap, rots, win, geom="cyl"):
    """Cached march. Returns counts, geometry, axes."""
    key = (seed, ppw, kap, tuple(rots), win, geom)
    if key in _COL:
        return _COL[key]
    lab, axes, seeds, h = load_tess(seed, ppw, kap)
    counts, G = march(lab, seeds, h, rots, win, geom)
    _COL[key] = (counts, G, axes)
    return _COL[key]


def vmatrix(axes, rots):
    """qP speed of every grain for every beam direction."""
    out = np.empty((len(rots), len(axes)))
    for i, r in enumerate(rots):
        a = np.radians(float(r))
        n = np.array([np.cos(a), np.sin(a), 0.0])
        out[i] = np.interp(np.arccos(np.clip(np.abs(axes @ n), 0.0, 1.0)),
                           FW._PSI, FW._VQP)
    return out


def vvar(counts, V, weighted=True):
    """The beam-local predictor: speed variance over the column grains."""
    if weighted:
        w = counts
    else:
        w = (counts > 0).astype(float)
    W = w.sum(1)
    m = (w * V).sum(1) / W
    return (w * V ** 2).sum(1) / W - m ** 2


def beam_local(seed, ppw, kap, rots, win, geom="cyl", weighted=True):
    counts, _, axes = column(seed, ppw, kap, rots, win, geom)
    return vvar(counts, vmatrix(axes, rots), weighted)


# ─────────────────────────── the bulk predictor ──────────────────────────
def er2(seed, ppw, kap, rots):
    """E[R^2] in dB per azimuth, the predictor of T1, from the cache."""
    lab, axes, _, _ = load_tess(seed, ppw, kap)
    vol = np.bincount(lab[lab >= 0].ravel(),
                      minlength=len(axes)).astype(float)
    keep = vol > 0
    axes, vol = axes[keep], vol[keep] / vol[keep].sum()
    V = vmatrix(axes, rots)
    vb = (V * vol).sum(1)
    return 10 * np.log10(((V - vb[:, None]) ** 2 * vol).sum(1)
                         / (2 * vb ** 2))


# ──────────────────────────────── the null ───────────────────────────────
def corr(x, y):
    """Pearson r, zero for a degenerate series rather than a warning."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


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


def rank_true(rs, m=None):
    """Rank of the true alignment by |r| among the first m shifts."""
    v = rs[:(m or len(rs))]
    return int((np.abs(v) >= abs(v[0])).sum()), float(v[0])


def score(y, pred):
    """r, honest rank of n, and the rank under the halving convention."""
    rs = shift_scan(y, pred)
    n = len(rs)
    rk, r0 = rank_true(rs, n)
    rh, _ = rank_true(rs, n // 2)
    return r0, rk, n, rh, n // 2


def fwer_family(y, preds):
    """Exact circular-shift FWER of the largest |r| over a family."""
    keys = sorted(preds)
    M = np.array([shift_scan(y, preds[k]) for k in keys])
    stat = np.abs(M).max(0)
    j = int(np.argmax(np.abs(M[:, 0])))
    return (float((stat >= stat[0]).sum()) / len(stat), keys[j],
            float(np.abs(M[j, 0])))


# ─────────────────────────────── the reports ─────────────────────────────
def report_harness(store):
    print("=" * 74)
    print("0. HARNESS. The published beam-local numbers, this path.")
    print("=" * 74)
    print("\n   Section 5.2, analysed gate, native azimuth grid of each")
    print("   sweep. Published: rank 1 of 30 distinct alignments at both")
    print("   resolutions, r = 0.57 at ppw 6 and 0.36 at ppw 8.")
    print("   %-6s %-6s %-9s %8s %9s %9s"
          % ("ppw", "column", "weight", "r", "rank/n", "rank/(n/2)"))
    ok_r = True
    for tag, nm, ppw, ref in (("ppw6", "girdle_seed11_ppw6_axis_perp", 6, 0.57),
                              ("ppw8", "girdle_seed11_ppw8_dev", 8, 0.36)):
        sw = load_sweep(nm)
        az, y = level_none(sw, GATE)
        for geom in ("cyl", "cone"):
            for wt in (True, False):
                p = beam_local(11, ppw, "-8", az, GATE, geom, wt)
                r0, rk, n, rh, m = score(y, p)
                flag = ""
                if geom == "cyl" and wt:
                    ok_r &= abs(abs(r0) - ref) < 0.01
                    flag = "   <- published reading, published r %.2f" % ref
                    store["harness_%s" % tag] = np.array([r0, rk, n, rh, m])
                print("   %-6s %-6s %-9s %+8.3f %6d/%-3d %6d/%-3d%s"
                      % (tag, geom, "sample" if wt else "distinct", r0,
                         rk, n, rh, m, flag))
    print("\n   The r values reproduce to two decimals on the cylinder")
    print("   column with sample weighting, and on no other reading of")
    print("   the sentence. The RANK does not reproduce at ppw 6: the")
    print("   6-degree neighbour scores 0.592 against the true 0.573, so")
    print("   this rebuild returns 2nd of 30 where the manuscript reports")
    print("   1st. At ppw 8 the true alignment is 1st of 30 under the")
    print("   halving convention and 2nd of 60 without it.")

    print("\n   Cross-check against analysis/axis_window_adjudication.py:")
    print("   the BULK predictor, 30 common azimuths, untreated levels.")
    print("   %-22s %-7s %8s %9s   %s"
          % ("sweep", "window", "r", "rank/15", "published r"))
    pub = {("girdle_seed11_ppw8_dev", "4-16"): 0.284,
           ("girdle_seed11_ppw8_dev", "10-22"): 0.250,
           ("girdle_seed11_ppw8_dev", "24-36"): 0.273,
           ("girdle_seed11_ppw8_uniform_axis", "4-16"): 0.734,
           ("girdle_seed11_ppw8_uniform_axis", "10-22"): 0.559,
           ("girdle_seed11_ppw8_uniform_axis", "24-36"): 0.153,
           ("girdle_seed11_ppw8_contrast_f000", "4-16"): 0.904,
           ("girdle_seed11_ppw8_contrast_f000", "10-22"): 0.701,
           ("girdle_seed11_ppw8_contrast_f000", "24-36"): 0.219}
    ok_b = True
    for nm in ("girdle_seed11_ppw8_dev",) + CONTROLS:
        sw = load_sweep(nm, AZ30)
        for wn, win in WINDOWS.items():
            az, y = level_none(sw, win)
            p = er2(11, 8, "-8", az)
            rs = shift_scan(y, p)
            rh, r0 = rank_true(rs, 15)
            ref = pub[(nm, wn)]
            ok_b &= abs(abs(r0) - ref) < 0.01
            print("   %-22s %-7s %+8.3f %6d/15    %.3f"
                  % (nm, wn, r0, rh, ref))
    print("\n   HARNESS: beam-local r reproduces %s; beam-local rank at"
          % ("YES" if ok_r else "NO"))
    print("   ppw 6 does NOT; bulk-predictor cross-check reproduces %s."
          % ("YES" if ok_b else "NO"))
    store["harness_ok"] = np.array([ok_r, ok_b], float)
    return ok_r and ok_b


def report_periodicity(store):
    print()
    print("=" * 74)
    print("1. PERIODICITY. How many distinct alignments does a sweep")
    print("   offer this predictor?")
    print("=" * 74)
    print("   %-22s %-7s %10s %10s"
          % ("sweep", "window", "r(az,+180)", "n azimuths"))
    for nm, ppw in (("girdle_seed11_ppw6_axis_perp", 6), ("girdle_seed11_ppw8_dev", 8)):
        sw = load_sweep(nm)
        az = np.array(sorted(sw))
        for wn, win in WINDOWS.items():
            p = beam_local(11, ppw, "-8", az, win)
            h = len(az) // 2
            rr = float(np.corrcoef(p, np.roll(p, h))[0, 1])
            store["per_%s_%s" % (nm, wn)] = np.array([rr, len(az)])
            print("   %-22s %-7s %10.3f %10d" % (nm, wn, rr, len(az)))
    print("\n   The bulk predictor is a function of |c . n| alone and is")
    print("   exactly invariant under a beam reversal, r = 1 by")
    print("   construction. The beam-local predictor is not: 0.59 to 0.60")
    print("   in the gate and NEGATIVE, -0.10 and -0.29, in the two")
    print("   earlier windows, because at az + 180 the column enters the")
    print("   disc on the far side and the window covers a different part")
    print("   of the chord. Every shift is therefore a distinct alignment")
    print("   of it and the honest floor is 1/n, not 2/n. Both counts are")
    print("   printed below, and the difference matters: the published")
    print("   'first of thirty' at ppw 8 is first of thirty only after")
    print("   halving a grid this predictor does not permit halving.")


def report_decisive(store):
    print()
    print("=" * 74)
    print("2. THE DECISIVE TEST. The beam-local predictor of the seed-11")
    print("   girdle, applied to the specimen and to two specimens that")
    print("   cannot backscatter from a grain boundary, in three windows.")
    print("=" * 74)
    print("\n   GRID A: 30 common azimuths, all three sweeps, 30 distinct")
    print("   alignments (floor 1/30) or 15 under the halving convention.")
    print("   %-22s %-7s %-6s %8s %8s %8s"
          % ("sweep", "window", "treat", "r", "rank/30", "rank/15"))
    tabA = {}
    for nm in ("girdle_seed11_ppw8_dev",) + CONTROLS:
        sw = load_sweep(nm, AZ30)
        for wn, win in WINDOWS.items():
            for tr in ("none", "loo"):
                az, y = level(sw, win, tr)
                p = beam_local(11, 8, "-8", az, win)
                r0, rk, n, rh, m = score(y, p)
                tabA[(nm, wn, tr)] = (r0, rk, rh)
                store["decA_%s_%s_%s" % (nm, wn, tr)] = np.array([r0, rk, rh])
                print("   %-22s %-7s %-6s %+8.3f %5d/%-3d %5d/%-3d"
                      % (nm, wn, tr, r0, rk, n, rh, m))
    print("\n   GRID B: 60 azimuths, specimen and zerocontrast only,")
    print("   60 distinct alignments (floor 1/60), 30 under halving.")
    print("   %-22s %-7s %-6s %8s %8s %8s"
          % ("sweep", "window", "treat", "r", "rank/60", "rank/30"))
    for nm in ("girdle_seed11_ppw8_dev", "girdle_seed11_ppw8_uniform_axis"):
        sw = load_sweep(nm)
        for wn, win in WINDOWS.items():
            for tr in ("none", "loo"):
                az, y = level(sw, win, tr)
                p = beam_local(11, 8, "-8", az, win)
                r0, rk, n, rh, m = score(y, p)
                store["decB_%s_%s_%s" % (nm, wn, tr)] = np.array([r0, rk, rh])
                print("   %-22s %-7s %-6s %+8.3f %5d/%-3d %5d/%-3d"
                      % (nm, wn, tr, r0, rk, n, rh, m))
    print("\n   IS THERE ANYTHING TO READ? Revolution level, dB, and the")
    print("   margin of the specimen over specimens that cannot")
    print("   backscatter from a grain boundary, same 30 azimuths.")
    print("   %-7s %-6s %10s %10s %8s %10s %8s"
          % ("window", "treat", "specimen", "zerocont", "margin", "cs f=0",
             "margin"))
    for wn, win in WINDOWS.items():
        for tr in ("none", "loo"):
            v = []
            for nm in ("girdle_seed11_ppw8_dev",) + CONTROLS:
                _, y = level(load_sweep(nm, AZ30), win, tr)
                v.append(10 * np.log10(np.mean(10 ** (y / 10.0))))
            store["lvl_%s_%s" % (wn, tr)] = np.array(v)
            print("   %-7s %-6s %10.2f %10.2f %8.2f %10.2f %8.2f"
                  % (wn, tr, v[0], v[1], v[0] - v[1], v[2], v[0] - v[2]))
    print("   At 4 to 16 us untreated the specimen is within half a")
    print("   decibel of both controls, so what is being correlated there")
    print("   is almost entirely a quantity all three share.")

    print("\n   THE COMPARISON THE TEST IS FOR, grid A, |r| at the true")
    print("   alignment, specimen against the better of the two controls:")
    print("   %-7s %-6s %10s %10s %10s %9s"
          % ("window", "treat", "specimen", "zerocont", "cs f=0", "verdict"))
    for wn in WINDOWS:
        for tr in ("none", "loo"):
            v = [abs(tabA[(nm, wn, tr)][0]) for nm in
                 ("girdle_seed11_ppw8_dev",) + CONTROLS]
            bad = max(v[1], v[2]) >= v[0]
            store["dec_margin_%s_%s" % (wn, tr)] = np.array(v)
            print("   %-7s %-6s %10.3f %10.3f %10.3f %9s"
                  % (wn, tr, v[0], v[1], v[2],
                     "FAILS" if bad else "clean"))
    ncell = 2 * len(WINDOWS) * 2
    print("\n   MULTIPLICITY. %d control cells are scored above, two"
          % ncell)
    print("   controls by three windows by two treatments. Under a true")
    print("   null with a 1/30 floor the chance that at least one of them")
    print("   reaches first rank is 1 - (29/30)^%d = %.2f, so a single"
          % (ncell, 1 - (29 / 30.) ** ncell))
    print("   rank-1 control would be no evidence of anything. The test")
    print("   used here is not a rank-1 count: it is whether a control's")
    print("   |r| at the TRUE alignment exceeds the specimen's in the")
    print("   same window, which is one pre-specified comparison per")
    print("   window and carries no hidden multiplicity.")
    return tabA


def report_geometry(store):
    print()
    print("=" * 74)
    print("3. PURE GEOMETRY, IN EVERY WINDOW. Eight column descriptors")
    print("   that contain NO c-axis information whatever, scored the")
    print("   same way against the specimen's own level.")
    print("=" * 74)
    print("   n_cross crossings along the column, n_grain distinct grains,")
    print("   n_pair distinct grain pairs crossed, eff_grain the")
    print("   Herfindahl effective grain count, vol_cv the spread of the")
    print("   intersected volumes, geom_dir the summed facet directivity")
    print("   of length_scales with every velocity jump set to one,")
    print("   d_first the depth of the first crossing in the window,")
    print("   sep_mean the mean seed separation across the crossings.")
    print("   The insonified path length is EXACTLY constant across")
    print("   azimuth, because the disc is circular and the column is")
    print("   rigid, so it carries no azimuthal information and is not in")
    print("   the family.")
    print("\n   Three panels: the cylinder column the published")
    print("   beam-local predictor uses, on the native 60-azimuth grid")
    print("   and on the 30 common azimuths of the control table, and the")
    print("   diverging cone of analysis/geometry_vs_fabric.py, which is")
    print("   the column the manuscript's own geometry-only statement was")
    print("   made with.")
    panels = (("60 az", None, "cyl", ("none", "loo")),
              ("30 az", AZ30, "cyl", ("none", "loo")),
              ("60 az", None, "cone", ("none",)))
    for grid, keep, geom, treats in panels:
        sw = load_sweep("girdle_seed11_ppw8_dev", keep)
        az = np.array(sorted(sw))
        for tr in treats:
            print("\n   grid %s, column %s, treatment %s. Family-wise p is"
                  % (grid, geom, tr))
            print("   the exact circular-shift FWER of the largest |r| over")
            print("   all eight, floor 1/%d." % len(az))
            print("   %-7s %8s %10s %9s %9s %10s"
                  % ("window", "best |r|", "descriptor", "rank", "p_fw",
                     "beam-local"))
            for wn, win in WINDOWS.items():
                _, y = level(sw, win, tr)
                _, G, _ = column(11, 8, "-8", az, win, geom)
                pfw, key, mx = fwer_family(y, G)
                rk, _ = rank_true(shift_scan(y, G[key]), len(az))
                bl = beam_local(11, 8, "-8", az, win, geom)
                store["geom_%s_%s_%s_%s" % (grid[:2], geom, wn, tr)] = \
                    np.array([mx, rk, pfw, corr(bl, y)])
                print("   %-7s %8.3f %10s %5d/%-3d %9.4f %10.3f"
                      % (wn, mx, key, rk, len(az), pfw, corr(bl, y)))
            print("   per-descriptor r at the true alignment")
            print("   %-7s" % "window"
                  + "".join("%10s" % k for k in GEOM_KEYS))
            for wn, win in WINDOWS.items():
                _, y = level(sw, win, tr)
                _, G, _ = column(11, 8, "-8", az, win, geom)
                print("   %-7s" % wn + "".join(
                    "%+10.3f" % corr(G[k], y) for k in GEOM_KEYS))
    print("\n   READ THE GATE ROWS FIRST. The manuscript's statement that")
    print("   pure-geometry column descriptors 'do not track the level at")
    print("   all' does not hold on the cylinder column even in the gate:")
    print("   the summed facet directivity, which contains no c-axis")
    print("   information whatever, reaches |r| = 0.41 on the native grid")
    print("   and 0.58 on the 30 common azimuths, above the beam-local")
    print("   predictor's 0.35 and 0.21, at first rank. That is the facet")
    print("   geometry of Eq. eq:facetmodel and it is not news; what is")
    print("   news is that it outscores the predictor the sentence rests")
    print("   on, in the sentence's own window.")
    print("\n   In the two earlier windows nothing in the family survives")
    print("   its own family-wise null, p_fw = 0.32 to 0.90. Neither does")
    print("   the beam-local predictor on this specimen: 0.13 and -0.00")
    print("   against 0.35 in the gate. What decides the orientation")
    print("   reading is the ORDER, and the order goes the wrong way: a")
    print("   descriptor with no c-axis in it reaches 0.21 at 4 to 16 and")
    print("   0.31 at 10 to 22 where the c-axis predictor reaches 0.13")
    print("   and 0.00. On the cone column, which is the one the")
    print("   manuscript's own geometry-only statement was made with, the")
    print("   same holds in every window, best geometry 0.38, 0.42 and")
    print("   0.45 against the beam-local 0.14, 0.20 and 0.29.")
    print("   The earlier-window claim therefore does not rest on this")
    print("   specimen at all; it rests on the twelve-tessellation")
    print("   first-rank count, which the next section examines.")


def report_permutation(store, ndraw, nens):
    print()
    print("=" * 74)
    print("4. THE ORIENTATION-PERMUTATION CONTROL. Permute which grain")
    print("   carries which c-axis; hold the tessellation, the beam")
    print("   sampling and the orientation multiset fixed.")
    print("=" * 74)
    rng = np.random.default_rng(11)
    print("\n   Seed 11, analysed gate, native grids, %d draws." % ndraw)
    print("   Published p = 0.005 at ppw 6 and 0.070 at ppw 8.")
    print("   %-6s %8s %10s %10s" % ("ppw", "r true", "p perm", "published"))
    for tag, nm, ppw, ref in (("ppw6", "girdle_seed11_ppw6_axis_perp", 6, 0.005),
                              ("ppw8", "girdle_seed11_ppw8_dev", 8, 0.070)):
        sw = load_sweep(nm)
        az, y = level_none(sw, GATE)
        counts, _, axes = column(11, ppw, "-8", az, GATE)
        V = vmatrix(axes, az)
        r0 = abs(float(np.corrcoef(vvar(counts, V), y)[0, 1]))
        hit = 0
        for _ in range(ndraw):
            pv = vvar(counts, V[:, rng.permutation(V.shape[1])])
            if abs(float(np.corrcoef(pv, y)[0, 1])) >= r0:
                hit += 1
        p = (1.0 + hit) / (1.0 + ndraw)
        store["perm_%s" % tag] = np.array([r0, p])
        print("   %-6s %8.3f %10.4f %10.3f" % (tag, r0, p, ref))

    print("\n   The ensemble, twelve tessellations, 30 common azimuths,")
    print("   %d draws each, Fisher combination over the twelve."
          % nens)
    print("   Published Fisher p = 0.0010 at 4 to 16 us and 0.078 in the")
    print("   gate.")
    az = np.array(AZ30)
    print("   %-7s %-6s %10s %12s %10s"
          % ("window", "treat", "Fisher X2", "Fisher p", "median p"))
    for wn, win in WINDOWS.items():
        for tr in ("none", "loo"):
            ps = []
            for nm, seed, kap in SPECS:
                sw = load_sweep(nm, AZ30)
                a, y = level(sw, win, tr)
                counts, _, axes = column(seed, 8, kap, a, win)
                V = vmatrix(axes, a)
                r0 = abs(float(np.corrcoef(vvar(counts, V), y)[0, 1]))
                hit = 0
                for _ in range(nens):
                    pv = vvar(counts, V[:, rng.permutation(V.shape[1])])
                    if abs(float(np.corrcoef(pv, y)[0, 1])) >= r0:
                        hit += 1
                ps.append((1.0 + hit) / (1.0 + nens))
            ps = np.array(ps)
            x2 = float(-2.0 * np.log(ps).sum())
            pf = float(chi2.sf(x2, 2 * len(ps)))
            store["permens_%s_%s" % (wn, tr)] = np.concatenate(
                [[x2, pf], ps])
            print("   %-7s %-6s %10.2f %12.6f %10.4f"
                  % (wn, tr, x2, pf, float(np.median(ps))))
    print("\n   WHAT THIS CONTROL DOES AND DOES NOT ESTABLISH. It is")
    print("   evaluated at the TRUE registration, so it tests whether the")
    print("   statistic depends on which grain carries which c-axis. It")
    print("   does. It does not test whether the true rotation outranks")
    print("   the wrong ones, which is the shift null, and it cannot")
    print("   separate an orientation channel from the rendering of the")
    print("   tessellation, because the permutation never moves the")
    print("   specimen relative to the beam and every geometric quantity")
    print("   the column reads is invariant under it. A pure-geometry")
    print("   descriptor would return p = 1 here by construction, so")
    print("   passing it distinguishes the predictor from geometry as a")
    print("   FUNCTION and not as an EXPLANATION of the correlation.")


def report_ensemble(store):
    print()
    print("=" * 74)
    print("5. THE ENSEMBLE. Twelve tessellations, 30 common azimuths,")
    print("   each specimen scored against its OWN realised c-axes.")
    print("=" * 74)
    az = np.array(AZ30)
    cells = [(w, t) for w in WINDOWS for t in ("none", "loo")]
    R = {c: [] for c in cells}
    for nm, seed, kap in SPECS:
        sw = load_sweep(nm, AZ30)
        for wn, tr in cells:
            a, y = level(sw, WINDOWS[wn], tr)
            p = beam_local(seed, 8, kap, a, WINDOWS[wn])
            rk, _ = rank_true(shift_scan(y, p), len(a))
            R[(wn, tr)].append(rk)
    R = {k: np.array(v) for k, v in R.items()}
    print("   %-14s %10s %10s %12s"
          % ("cell", "first/12", "rank sum", "p, Bin(12,1/30)"))
    for k, v in R.items():
        nf = int((v == 1).sum())
        p = float(binom.sf(nf - 1, 12, 1 / 30.))
        store["ens_%s_%s" % k] = v
        print("   %-14s %8d/12 %10d %12.4f"
              % ("%s %s" % k, nf, v.sum(), p))
    print("   chance 0.4 of 12 first, rank sum 186.")
    print("\n   Paired on the tessellation, which is the unit of")
    print("   replication, against the published gate:")
    g = R[("24-36", "none")]
    for k, v in R.items():
        if k == ("24-36", "none"):
            continue
        d = v - g
        nz = d[d != 0]
        sp = binomtest(int((nz < 0).sum()), len(nz)).pvalue if len(nz) else 1.0
        wp = float(wilcoxon(v, g).pvalue) if np.any(d) else 1.0
        store["enspair_%s_%s" % k] = np.array([sp, wp])
        print("      %-14s better %2d worse %2d tied %2d  sign p = %.3f  "
              "wilcoxon p = %.3f" % ("%s %s" % k, int((d < 0).sum()),
                                     int((d > 0).sum()), int((d == 0).sum()),
                                     sp, wp))
    print("\n   THE CROSS-TESSELLATION CONTROL, which the ensemble can")
    print("   afford where the zero-scattering controls exist for seed 11")
    print("   alone. Score every specimen's level against every OTHER")
    print("   specimen's beam-local predictor. A predictor that reads the")
    print("   c-axes of the specimen in front of it cannot work on a")
    print("   different specimen; one that reads a generic property of a")
    print("   Laguerre column can. Chance is 1/30 either way, 0.4 of 12")
    print("   matched and 4.4 of 132 mismatched.")
    P = {}
    for nm, seed, kap in SPECS:
        for wn, win in WINDOWS.items():
            sw = load_sweep(nm, AZ30)
            a, _ = level_none(sw, WINDOWS[wn])
            P[(nm, wn)] = beam_local(seed, 8, kap, a, win)
    print("   %-7s %-6s %12s %14s %14s"
          % ("window", "treat", "matched", "mismatched", "same tess"))
    for wn, win in WINDOWS.items():
        for tr in ("none", "loo"):
            mt, mm, st = [], [], []
            for nm, seed, kap in SPECS:
                sw = load_sweep(nm, AZ30)
                a, y = level(sw, win, tr)
                for nm2, seed2, kap2 in SPECS:
                    rk, _ = rank_true(shift_scan(y, P[(nm2, wn)]), len(a))
                    if nm2 == nm:
                        mt.append(rk)
                    else:
                        mm.append(rk)
                        if seed2 == seed:
                            st.append(rk)
            store["cross_%s_%s" % (wn, tr)] = np.array(
                [int((np.array(mt) == 1).sum()), len(mt),
                 int((np.array(mm) == 1).sum()), len(mm),
                 int((np.array(st) == 1).sum()), len(st)])
            print("   %-7s %-6s %8d/%-3d %10d/%-3d %10d/%-3d"
                  % (wn, tr, int((np.array(mt) == 1).sum()), len(mt),
                     int((np.array(mm) == 1).sum()), len(mm),
                     int((np.array(st) == 1).sum()), len(st)))
    print("   The mismatched rate is the honest null for this ensemble,")
    print("   because it holds the acquisition, the column and the class")
    print("   of predictor fixed and only breaks the registration between")
    print("   the c-axes and the specimen that was insonified.")

    print("\n   A count of first ranks is a coarse statistic on twelve")
    print("   specimens: Bin(12, 1/30) has mean 0.4 and sd 0.62, so 1 and")
    print("   4 are two and six standard deviations apart only if the")
    print("   twelve are independent tests of the same question, which")
    print("   they are not, since four of them share a tessellation with")
    print("   another and all twelve share one acquisition geometry. The")
    print("   paired test on the ranks themselves is the honest one.")
    return R


def main():
    quick = "--quick" in sys.argv
    ndraw, nens = (20, 200) if quick else (200, 2000)
    store = {}
    report_harness(store)
    report_periodicity(store)
    report_decisive(store)
    report_geometry(store)
    report_ensemble(store)
    report_permutation(store, ndraw, nens)
    np.savez(os.path.join(HERE, "beam_local_admissibility.npz"), **store)
    print("\nwrote beam_local_admissibility.npz")


if __name__ == "__main__":
    main()
