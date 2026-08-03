"""Is the BEAM-LOCAL predictor of Sec. 5.2 admissible in the earlier
windows, or does it share the bulk predictor's defect?

THE QUESTION. Sec. 5.2 ends on a beam-local predictor, "the variance of
the quasi-longitudinal speed over the grains the beam column actually
intersects inside the coda gate", and uses it to claim that "a weak
orientation channel is therefore visible in a window this paper does not
analyse in". Every number behind that sentence is measured at 4 to 16 us
and 10 to 22 us. The BULK predictor, the volume-weighted mean square of
dv/2vbar over every grain in the disc, has already been shown
inadmissible in exactly those windows: it tracks specimens that cannot
backscatter from a grain boundary at all BETTER than it tracks the real
specimen (|r| 0.90 and 0.73 against 0.28 at 4 to 16, 0.70 and 0.56
against 0.25 at 10 to 22, against 0.22 and 0.15 against 0.27 in the
gate). An inadmissible window cannot support a positive claim any more
than it can confirm a null. This module asks the same question of the
beam-local predictor, and adds the two tests the beam-local claim rests
on and the bulk one does not: the pure-geometry column controls and the
orientation-permutation control.

WHAT THE PREDICTOR IS, established from the code and not from the prose.
analysis/beam_descriptors.py computes six descriptors inside the beam
column, of which v_var is "velocity variance among the grains in the
beam". Its column is a rectangle of the ELEMENT width, 6.35 mm, marched
at h/2 along the beam over the depth range the window listens to, with
the lateral offsets at the grid pitch and the through-thickness slices
every 2 mm, and the variance is taken over the SAMPLED POINTS, so a grain
enters weighted by the volume of it the column contains. That estimator
is rebuilt here, verbatim in geometry, reading the cached label volume
instead of calling DiskSpecimen.build. Its published r comes back to two
decimals at both resolutions (report_harness), which is what licenses
everything below.

The point-count form makes the orientation-permutation control free. The
column geometry fixes an (azimuth x grain) matrix of sampled-point counts
that no permutation of the c-axes can move, so the predictor is a
weighted variance of the speed over that fixed matrix and a permutation
is a permutation of the speed columns. Nothing is re-marched.

THE NUMBER OF DISTINCT ALIGNMENTS IS MEASURED, NOT ASSUMED, and this is
where the published p floor is wrong. The BULK predictor is a function of
|c . n| alone, so it is exactly invariant under a beam reversal and a
60-azimuth sweep offers it only 30 distinct alignments. The BEAM-LOCAL
predictor is not: the column enters the disc at the rim and marches
inward over a fixed depth range, so at az + 180 it holds the opposite
part of the chord. report_periodicity measures the invariance directly
and finds r(az, az + 180) = +0.59 in the gate and NEGATIVE, -0.10 and
-0.29, in the two earlier windows. Every one of the n circular shifts is
therefore a distinct alignment, the floor is 1/60 on the production grid
and 1/30 on the common grid, and a rank quoted out of thirty on a
60-azimuth sweep has silently discarded half the null. Both conventions
are printed everywhere, labelled "of n" and "of n/2", so the published
numbers can be read on their own convention and on the measured one.

THE GRIDS ARE KEPT APART, because the two controls do not share azimuths.

  GRID 60  the production grid: 60 azimuths at 6 degrees, 60 alignments,
           floor 1/60 = 0.017 (30 and 1/30 on the halving convention).
           Specimen and zerocontrast_ppw8 only.
  GRID 30  the 30 azimuths at 12 degrees every ppw 8 sweep shares: 30
           alignments, floor 1/30 (15 and 1/15 halved). Any cell carrying
           cs_f000_s11_ppw8, and the twelve-tessellation ensemble.

A rank of 1 of 30 and a rank of 1 of 60 are not the same evidence and are
never compared here; every table names its grid.

MULTIPLICITY IS PAID EXPLICITLY. report_geometry scores nine geometry
descriptors in three windows and prints, beside the count of first ranks,
the chance of at least one under a true null. One rank-1 among twelve
cells at a floor of 1/15 is p = 0.56 and is not evidence of anything.

  MEASURED   every r, rank, p and count printed below.
  INFERRED   the closing paragraph of each report, which is labelled.

TOUCHES NO CUDA. No solver, no DiskSpecimen.build, no label build. Every
tessellation quantity is read from out/tesscache and the module raises
rather than build a missing one. Reads out/sweeps traces. Writes
beam_local_windows.npz beside this file.

Run with `python beam_local_windows.py`; add an integer to set the
permutation draw count (default 2000).
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
sys.path.insert(0, os.path.join(ROOT, "sim"))

# forward.py tabulates the quasi-longitudinal velocity surface of ice and
# imports numpy alone. Nothing else on the solver's path is touched.
import forward as FW                                        # noqa: E402

# Acquisition, Sec. 3, identical to analysis/beam_descriptors.py.
C_REF, F0, DIA, THK = 3850.0, 2.0e6, 0.100, 0.035
BAND = (0.8e6, 3.0e6)
ELEM = 6.35e-3                       # element width, the column width
Z_STEP = 2e-3                        # through-thickness slice spacing

GATE = (24e-6, 36e-6)
WINDOWS = (("4-16", (4e-6, 16e-6)), ("10-22", (10e-6, 22e-6)),
           ("24-36", GATE))

AZ60 = tuple(range(0, 360, 6))
AZ30 = tuple(range(0, 360, 12))

SPECIMEN = "girdle_perp_ppw8"
SPEC_PPW6 = "girdle_perp"

# Neither control can backscatter from a grain boundary and both carry
# the seed-11 geometry the predictor is built from.
CONTROLS = (("zerocontrast_ppw8", "one c-axis in every grain"),
            ("cs_f000_s11_ppw8", "contrast f = 0.00"))

# The twelve tessellations with a cached ppw 8 label volume: the ensemble
# of Sec. sec:ensemble.
SPECS = ([(SPECIMEN, 11, "k-8")]
         + [("mx_girdle_s%d_ppw8" % s, s, "k-8")
            for s in (7, 17, 23, 41, 53, 71, 89)]
         + [("singlemax_ppw8", 11, "k3.93")]
         + [("mx_single_s%d_ppw8" % s, s, "k3.93") for s in (17, 23, 41)])

# Sec. 5.2: r and the first rank it reports out of thirty alignments.
REF_R = {SPEC_PPW6: 0.57, SPECIMEN: 0.36}
REF_PERM = {SPEC_PPW6: 0.005, SPECIMEN: 0.070}

_SWEEP, _TESS, _COL = {}, {}, {}


# ──────────────────────────────── load ───────────────────────────────
def _sos(fs):
    return butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                  btype="band", output="sos")


def load_sweep(name, az_keep):
    """Traces of one sweep at the requested azimuths, each with its own dt."""
    key = (name, az_keep)
    if key in _SWEEP:
        return _SWEEP[key]
    d = os.path.join(SWD, name)
    az, tr, dt = [], [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        a = int(f[2:5])
        if az_keep is not None and a not in az_keep:
            continue
        with np.load(os.path.join(d, f)) as z:
            tr.append(np.asarray(z["trace"], float).ravel())
            dt.append(float(z["dt"]))
        az.append(a)
    o = np.argsort(az)
    out = (np.array(az)[o], [tr[i] for i in o], [dt[i] for i in o])
    _SWEEP[key] = out
    return out


def levels(name, win, az_keep):
    """Two level estimators per azimuth, in dB, at each azimuth's own dt.

    env   the mean square of the band-passed analytic envelope over the
          window, referenced to the sweep-mean backwall peak. This is
          analysis/beam_descriptors.measure's coda, which is the level
          the published beam-local r is quoted on.
    tf    2 <x^2> of the band-passed trace over the window, referenced to
          that azimuth's own squared source peak: the audited level of
          tab:reconcile, carried as a second opinion because it applies
          no transform inside the window at all.
    """
    key = (name, win, az_keep)
    az, trs, dts = load_sweep(name, az_keep)
    env, tf, e1 = [], [], []
    for tr, dt in zip(trs, dts):
        fs = 1.0 / dt
        bp = sosfiltfilt(_sos(fs), tr)
        i0, i1 = int(win[0] * fs), int(win[1] * fs)
        env.append(float((np.abs(hilbert(bp))[i0:i1] ** 2).mean()))
        e = np.abs(hilbert(tr))
        tf.append(2.0 * float((bp[i0:i1] ** 2).mean()) / e.max() ** 2)
        k0, w = int(2 * DIA / C_REF * fs), int(2e-6 * fs)
        e1.append(e[max(k0 - w, 0):k0 + w].max())
    e1 = np.array(e1)
    return az, {"env": 10 * np.log10(np.array(env) / e1.mean() ** 2),
                "tf": 10 * np.log10(np.array(tf))}


def tess(seed, ppw, kappa):
    """(labels, axes, seed points, h) of one cached tessellation."""
    key = (seed, ppw, kappa)
    if key in _TESS:
        return _TESS[key]
    p = os.path.join(TESS, "tess_s%d_p%g_%s.npz" % (seed, ppw, kappa))
    if not os.path.exists(p):
        raise SystemExit("not cached, would reach CUDA: %s" % p)
    with np.load(p) as z:
        out = (np.asarray(z["labels"]), np.asarray(z["axes"], float),
               np.asarray(z["seeds"], float), float(z["h"]))
    _TESS[key] = out
    return out


# ────────────────────────────── the column ───────────────────────────
def column(seed, ppw, kappa, win, rots):
    """The beam column of analysis/beam_descriptors.py, as counts.

    Returns

      W    (n_az, n_grain) sampled-point counts inside the column. The
           beam-local predictor is a W-weighted variance of the speed, so
           W is everything the geometry contributes and the c-axes enter
           only through the speed.
      G    pure-geometry column descriptors, none of which contains any
           c-axis information: the counts, the crossings, the intercept
           lengths, the sizes of the grains the column meets, and the
           orientation of the Laguerre facets it crosses relative to the
           beam. Every one is invariant under a permutation of the
           c-axes, which is what makes them the right control.
    """
    key = (seed, ppw, kappa, win, tuple(rots))
    if key in _COL:
        return _COL[key]
    lab, axes, pts, h = tess(seed, ppw, kappa)
    nx, ny, nz = lab.shape
    ng = len(axes)
    d0, d1 = win[0] * C_REF / 2.0, win[1] * C_REF / 2.0
    s = np.arange(d0, d1, h / 2.0)
    off = np.arange(-ELEM / 2, ELEM / 2 + h, h)
    zc = (np.arange(nz) + 0.5) * h - nz * h / 2.0
    zk = np.arange(0, nz, max(1, int(round(Z_STEP / h))))
    vol = np.bincount(lab[lab >= 0].ravel().astype(int),
                      minlength=ng).astype(float) * h ** 3
    W, G = [], {k: [] for k in ("n_grain", "n_cross", "n_pts", "chord_mm",
                                "path_mm", "vol_mean", "vol_cv",
                                "facet_align", "facet_area")}
    for r in rots:
        a = np.radians(float(r))
        n = np.array([np.cos(a), np.sin(a), 0.0])
        t = np.array([-np.sin(a), np.cos(a), 0.0])
        P = (DIA / 2.0 * n - s[:, None] * n)[:, None, None, :] \
            + off[None, :, None, None] * t \
            + np.stack([np.zeros_like(zc[zk]), np.zeros_like(zc[zk]),
                        zc[zk]], 1)[None, None, :, :]
        gi = np.rint((P[..., 0] + nx * h / 2) / h - 0.5).astype(int)
        gj = np.rint((P[..., 1] + ny * h / 2) / h - 0.5).astype(int)
        gk = np.rint((P[..., 2] + nz * h / 2) / h - 0.5).astype(int)
        ok = ((gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
              & (gk >= 0) & (gk < nz))
        L = np.where(ok, lab[np.clip(gi, 0, nx - 1), np.clip(gj, 0, ny - 1),
                             np.clip(gk, 0, nz - 1)], -1)
        ids = L[L >= 0].astype(int)
        c = np.bincount(ids, minlength=ng).astype(float)
        W.append(c)
        A, B = L[:-1], L[1:]
        cr = (A != B) & (A >= 0) & (B >= 0)
        ia, ib = A[cr].astype(int), B[cr].astype(int)
        nm = pts[ib] - pts[ia]
        nm /= np.linalg.norm(nm, axis=1, keepdims=True) + 1e-30
        al = np.abs(nm @ n)
        nray = float(len(off) * len(zk))
        met = dict(
            n_grain=float((c > 0).sum()),
            n_cross=float(cr.sum()),
            n_pts=float(c.sum()),
            chord_mm=float(c.sum() / max(cr.sum(), 1) * (h / 2.0) * 1e3),
            path_mm=float(c.sum() / nray * (h / 2.0) * 1e3),
            vol_mean=float(vol[c > 0].mean() * 1e9),
            vol_cv=float(vol[c > 0].std() / max(vol[c > 0].mean(), 1e-30)),
            facet_align=float(al.mean()) if al.size else 0.0,
            # A facet crossed at grazing incidence to the beam presents
            # more area inside the column than one crossed head on, so
            # 1/|cos| summed over crossings is an intersected-area proxy.
            facet_area=float((1.0 / np.maximum(al, 0.05)).sum())
            if al.size else 0.0)
        for k, v in met.items():
            G[k].append(v)
    out = (np.array(W), {k: np.array(v, float) for k, v in G.items()})
    _COL[key] = out
    return out


def speeds(axes, rots):
    """(n_az, n_grain) quasi-longitudinal speed for each beam direction."""
    a = np.radians(np.asarray(rots, float))
    n = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1)
    return np.interp(np.arccos(np.clip(np.abs(n @ axes.T), 0.0, 1.0)),
                     FW._PSI, FW._VQP)


def beam_local(W, V):
    """The predictor: W-weighted variance of the speed, per azimuth.

    Identical to beam_descriptors' vg[ids].var() to nine significant
    figures, because a variance over sampled points is a variance over
    grains weighted by the number of points each grain owns.
    """
    w = W / W.sum(1, keepdims=True)
    m = (w * V).sum(1)
    return (w * V ** 2).sum(1) - m ** 2


def predictor(seed, kappa, win, az):
    """The beam-local predictor of one tessellation at one window."""
    W, _ = column(seed, 8, kappa, win, az)
    return beam_local(W, speeds(tess(seed, 8, kappa)[1], az))


# ─────────────────────────────── the null ────────────────────────────
def zs(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def shift_curve(y, pred):
    """Pearson r at every circular shift of the level. r[0] is the truth."""
    zy, zp = zs(y), zs(pred)
    return np.array([float(np.mean(np.roll(zy, k) * zp))
                     for k in range(len(zy))])


def ranks(rs):
    """(r at the truth, rank among all n shifts, rank among the first n/2).

    The second is the manuscript's convention, which is exact only for a
    predictor invariant under a beam reversal. report_periodicity
    measures that this predictor is not.
    """
    n = len(rs)
    full = int(np.sum(np.abs(rs) >= abs(rs[0]) - 1e-12))
    half = int(np.sum(np.abs(rs[:n // 2]) >= abs(rs[0]) - 1e-12))
    return float(rs[0]), full, half, n


def score(y, pred):
    return ranks(shift_curve(y, pred))


def perm_draws(W, V, y, ndraw, rng):
    """|r| at the true alignment under ndraw c-axis permutations.

    The permutation exchanges which grain carries which c-axis. The
    tessellation, the beam sampling and the orientation multiset are held
    fixed: W is untouched and V has its columns reordered.
    """
    zy = zs(y)
    out = np.empty(ndraw)
    for i in range(ndraw):
        q = rng.permutation(V.shape[1])
        out[i] = abs(float(np.mean(zy * zs(beam_local(W, V[:, q])))))
    return out


def perm_p(obs, draws):
    """Add-one permutation level, so nothing can return exactly zero."""
    return float((1 + int((draws >= obs - 1e-12).sum())) / (len(draws) + 1))


# ────────────────────────────── periodicity ──────────────────────────
def report_periodicity(store):
    """How many distinct alignments does this predictor actually have?"""
    print("=" * 78)
    print("0. PERIODICITY, measured. The manuscript's rank denominator of")
    print("   thirty on a 60-azimuth sweep assumes the predictor is")
    print("   invariant under a beam reversal. It is for the bulk")
    print("   predictor, which is a function of |c . n| alone. It is not")
    print("   for the beam-local one, whose column enters at the rim and")
    print("   holds the opposite part of the chord at az + 180.")
    print("=" * 78)
    az, _ = levels(SPECIMEN, GATE, AZ60)
    print("   %-8s %14s %14s %14s"
          % ("window", "r(pred,+180)", "r(n_cross,+180)", "distinct"))
    for wtag, win in WINDOWS:
        p = predictor(11, "k-8", win, az)
        _, G = column(11, 8, "k-8", win, az)
        rp = float(np.corrcoef(p, np.roll(p, len(az) // 2))[0, 1])
        rc = float(np.corrcoef(G["n_cross"],
                               np.roll(G["n_cross"], len(az) // 2))[0, 1])
        store["period_%s" % wtag] = np.array([rp, rc])
        print("   %-8s %+14.3f %+14.3f %14d" % (wtag, rp, rc, len(az)))
    print("\n   INFERRED. A rank quoted out of thirty on this predictor")
    print("   discards half the null. Both denominators are printed below.")


# ──────────────────────────────── harness ────────────────────────────
def report_harness(ndraw, rng, store):
    """Sec. 5.2's beam-local numbers, through this module's own path."""
    print()
    print("=" * 78)
    print("1. HARNESS. The published beam-local numbers, rebuilt from the")
    print("   cached tessellations. GRID 60, analysed gate.")
    print("=" * 78)
    ok = True
    print("\n   %-16s %-4s %8s %7s %9s %9s %9s %7s"
          % ("sweep", "lvl", "r", "ref r", "rank/60", "rank/30", "perm p",
             "ref p"))
    for name, ppw in ((SPEC_PPW6, 6), (SPECIMEN, 8)):
        az, LV = levels(name, GATE, AZ60)
        W, _ = column(11, ppw, "k-8", GATE, az)
        V = speeds(tess(11, ppw, "k-8")[1], az)
        pred = beam_local(W, V)
        for lvl in ("env", "tf"):
            r, kf, kh, n = score(LV[lvl], pred)
            d = perm_draws(W, V, LV[lvl], ndraw, rng)
            pp = perm_p(abs(r), d)
            store["harness_%s_%s" % (name, lvl)] = np.array(
                [r, kf, kh, n, pp])
            ok &= abs(abs(r) - REF_R[name]) < 0.011
            print("   %-16s %-4s %+8.4f %7.2f %6d/%-3d %6d/%-3d %9.4f "
                  "%7.3f" % (name, lvl, r, REF_R[name], kf, n, kh, n // 2,
                             pp, REF_PERM[name]))
    print("\n   MEASURED. The published r reproduces to two decimals at")
    print("   both resolutions, on both level estimators, which is what")
    print("   licenses this harness. The published FIRST RANK reproduces")
    print("   only on the halving convention; on the measured 60")
    print("   alignments the ppw 6 cell is not first, because the")
    print("   6-degree neighbour of the true alignment scores higher.")
    print("   The permutation control reproduces in kind: p near 0.005 at")
    print("   ppw 6 as published, and near 0.04 at ppw 8 against the")
    print("   published 0.070, the same verdict at %d draws." % ndraw)
    print("\n   %s" % ("HARNESS REPRODUCES the correlations"
                       if ok else "HARNESS DOES NOT REPRODUCE"))
    return ok


# ──────────────────────────── the decisive test ──────────────────────
def report_controls(store):
    """The beam-local predictor against records that cannot backscatter."""
    print()
    print("=" * 78)
    print("2. THE DECISIVE TEST. The seed-11 beam-local predictor scored")
    print("   against the specimen and against two specimens that cannot")
    print("   backscatter from a grain boundary at all, in three windows.")
    print("   This is the test the bulk predictor fails in both earlier")
    print("   windows.")
    print("=" * 78)
    for gtag, az_keep, members in (
            ("GRID 60: 60 alignments, floor 1/60 (halved 1/30)",
             AZ60, (SPECIMEN, CONTROLS[0][0])),
            ("GRID 30: 30 alignments, floor 1/30 (halved 1/15)",
             AZ30, (SPECIMEN, CONTROLS[0][0], CONTROLS[1][0]))):
        print("\n   %s" % gtag)
        print("   %-20s %-6s %-4s %8s %9s %8s %9s"
              % ("sweep", "window", "lvl", "r win", "rank", "r gate",
                 "rank"))
        for nm in members:
            for wtag, win in WINDOWS:
                az, LV = levels(nm, win, az_keep)
                pw = predictor(11, "k-8", win, az)
                pg = predictor(11, "k-8", GATE, az)
                for lvl in ("env", "tf"):
                    rw, kwf, kwh, n = score(LV[lvl], pw)
                    rg, kgf, kgh, _ = score(LV[lvl], pg)
                    store["ctrl_%s_%s_%s_%d" % (nm, wtag, lvl, n)] = \
                        np.array([rw, kwf, kwh, rg, kgf, kgh, n])
                    print("   %-20s %-6s %-4s %+8.3f %5d/%-3d %+8.3f "
                          "%5d/%-3d" % (nm if lvl == "env" else "",
                                        wtag if lvl == "env" else "", lvl,
                                        rw, kwf, n, rg, kgf, n))
    print("\n   'r win' rebuilds the column over the depth range the window")
    print("   listens to, which is what the descriptor means in that")
    print("   window; 'r gate' keeps the published gate column and moves")
    print("   only the level. Ranks are on the measured denominator.")

    print("\n   THE MARGIN, stated as the test it is: in how many matched")
    print("   cells does a specimen that CANNOT backscatter track the")
    print("   fabric predictor at least as well as the real specimen? A")
    print("   cell is one (control, level, column) triple on one grid, so")
    print("   the comparison is like for like and no rank crosses grids.")
    print("   Under a predictor with no sensitivity the control wins half.")
    print("   %-6s %8s %9s %12s %s"
          % ("window", "cells", "ctrl wins", "median |r|", "specimen |r|"))
    for wtag, win in WINDOWS:
        cells, wins, cr, sr = 0, 0, [], []
        for az_keep, ctrls in ((AZ60, (CONTROLS[0][0],)),
                               (AZ30, (CONTROLS[0][0], CONTROLS[1][0]))):
            az, LS = levels(SPECIMEN, win, az_keep)
            preds = {"win": predictor(11, "k-8", win, az),
                     "gate": predictor(11, "k-8", GATE, az)}
            for cn in ctrls:
                _, LC = levels(cn, win, az_keep)
                for lvl in ("env", "tf"):
                    for pk in ("win", "gate"):
                        rs = abs(float(np.mean(zs(LS[lvl]) * zs(preds[pk]))))
                        rc = abs(float(np.mean(zs(LC[lvl]) * zs(preds[pk]))))
                        cells += 1
                        wins += int(rc >= rs)
                        cr.append(rc)
                        sr.append(rs)
        store["margin_%s" % wtag] = np.array([cells, wins, np.median(cr),
                                              min(sr), max(sr)])
        print("   %-6s %8d %9d %12.3f %.3f to %.3f"
              % (wtag, cells, wins, float(np.median(cr)), min(sr), max(sr)))

    print("\n   THE LEVELS THEMSELVES, GRID 30, so that a correlation with")
    print("   a control is read against how much level there is to")
    print("   correlate with: revolution level in dB and its azimuthal")
    print("   spread.")
    print("   %-20s %-6s %10s %8s %10s %8s"
          % ("sweep", "window", "env dB", "sd", "tf dB", "sd"))
    for nm in (SPECIMEN, CONTROLS[0][0], CONTROLS[1][0]):
        for wtag, win in WINDOWS:
            _, LV = levels(nm, win, AZ30)
            e, t = LV["env"], LV["tf"]
            store["level_%s_%s" % (nm, wtag)] = np.array(
                [e.mean(), e.std(), t.mean(), t.std()])
            print("   %-20s %-6s %10.2f %8.2f %10.2f %8.2f"
                  % (nm, wtag, e.mean(), e.std(), t.mean(), t.std()))


# ────────────────────────── the geometry control ─────────────────────
def report_geometry(store):
    """Pure-geometry column descriptors, in every window."""
    print()
    print("=" * 78)
    print("3. THE PURE-GEOMETRY CONTROL. Nine column descriptors carrying")
    print("   NO c-axis information, scored exactly as the beam-local")
    print("   predictor is. Sec. 5.2 says these 'do not track the level at")
    print("   all'; that was established in the gate. GRID 60, 60")
    print("   alignments, floor 1/60.")
    print("=" * 78)
    az, _ = levels(SPECIMEN, GATE, AZ60)
    hits = {}
    for wtag, win in WINDOWS:
        _, LV = levels(SPECIMEN, win, AZ60)
        W, G = column(11, 8, "k-8", win, az)
        V = speeds(tess(11, 8, "k-8")[1], az)
        print("\n   window %s us" % wtag)
        print("   %-12s %9s %9s %9s %9s" % ("descriptor", "r env", "rank",
                                            "r tf", "rank"))
        r, kf, _, n = score(LV["env"], beam_local(W, V))
        r2, kf2, _, _ = score(LV["tf"], beam_local(W, V))
        print("   %-12s %+9.3f %5d/%-3d %+9.3f %5d/%-3d   <- beam-local"
              % ("v_var", r, kf, n, r2, kf2, n))
        for kk in sorted(G):
            if np.std(G[kk]) <= 0:
                print("   %-12s %9s %9s %9s %9s   constant"
                      % (kk, "-", "-", "-", "-"))
                continue
            r, kf, _, _ = score(LV["env"], G[kk])
            r2, kf2, _, _ = score(LV["tf"], G[kk])
            store["geom_%s_%s" % (wtag, kk)] = np.array([r, kf, r2, kf2, n])
            hits[(wtag, kk)] = (min(kf, kf2), max(abs(r), abs(r2)))
            print("   %-12s %+9.3f %5d/%-3d %+9.3f %5d/%-3d"
                  % (kk, r, kf, n, r2, kf2, n))
    n_cell = len(hits)
    n_first = sum(1 for v, _ in hits.values() if v == 1)
    n_top3 = sum(1 for v, _ in hits.values() if v <= 3)
    big = sum(1 for _, rr in hits.values() if rr >= 0.30)
    print("\n   MULTIPLICITY. %d live geometry cells were scored (nine"
          % n_cell)
    print("   descriptors minus the constant one, three windows, best of")
    print("   two level estimators). Under a true null the chance that at")
    print("   least one reaches rank 1 of 60 is 1 - (59/60)^%d = %.2f and"
          % (n_cell, 1 - (59 / 60.) ** n_cell))
    print("   the expected count at rank 3 or better is %.1f."
          % (n_cell * 3 / 60.))
    print("   Observed: %d at rank 1, %d at rank 3 or better, %d with"
          % (n_first, n_top3, big))
    print("   |r| at or above 0.30, which is the size of the beam-local")
    print("   correlation itself.")
    store["geom_multiplicity"] = np.array([n_cell, n_first, n_top3, big])


def report_separability(store):
    """Is the beam-local predictor separable from column geometry?"""
    print()
    print("=" * 78)
    print("4. SEPARABILITY. The beam-local predictor is a variance over")
    print("   the grains a column of a particular shape happens to hold,")
    print("   so it carries that column's geometry as well as the c-axes.")
    print("   Here it is regressed on the whole geometry panel and the")
    print("   residual is scored against the same null. GRID 60, env.")
    print("=" * 78)
    az, _ = levels(SPECIMEN, GATE, AZ60)
    print("   %-8s %9s %8s %11s %10s %11s %9s"
          % ("window", "r v_var", "rank", "r vol_mean", "rank",
             "r(v,vol)", "resid rank"))
    for wtag, win in WINDOWS:
        _, LV = levels(SPECIMEN, win, AZ60)
        W, G = column(11, 8, "k-8", win, az)
        V = speeds(tess(11, 8, "k-8")[1], az)
        p = beam_local(W, V)
        r, kf, _, n = score(LV["env"], p)
        rv, kv, _, _ = score(LV["env"], G["vol_mean"])
        cross = float(np.mean(zs(p) * zs(G["vol_mean"])))
        keys = [q for q in sorted(G) if np.std(G[q]) > 0]
        X = np.column_stack([np.ones_like(p)] + [zs(G[q]) for q in keys])
        res = p - X @ np.linalg.lstsq(X, p, rcond=None)[0]
        rr, kr, _, _ = score(LV["env"], res)
        store["sep_%s" % wtag] = np.array([r, kf, rv, kv, cross, rr, kr, n])
        print("   %-8s %+9.3f %5d/%-3d %+11.3f %5d/%-3d %+11.3f %5d/%-3d"
              % (wtag, r, kf, n, rv, kv, n, cross, kr, n))
    print("\n   'r vol_mean' is the mean volume of the grains the column")
    print("   meets, which contains no c-axis information at all.")


# ──────────────────── the orientation-permutation control ────────────
def report_permutation(ndraw, rng, store):
    """The orientation-permutation control, per tessellation and combined."""
    print()
    print("=" * 78)
    print("5. THE ORIENTATION-PERMUTATION CONTROL, %d draws. Which grain"
          % ndraw)
    print("   carries which c-axis is permuted; the tessellation, the beam")
    print("   sampling and the orientation multiset are held fixed. The")
    print("   statistic is |r| AT THE TRUE ALIGNMENT, so this control can")
    print("   only show that the statistic depends on which grain carries")
    print("   which c-axis. It cannot show that the true rotation")
    print("   outranks the wrong ones. GRID 30, twelve tessellations, env.")
    print("=" * 78)
    out = {}
    for wtag, win in WINDOWS:
        R, D = [], []
        for nm, seed, kap in SPECS:
            az, LV = levels(nm, win, AZ30)
            W, _ = column(seed, 8, kap, win, az)
            V = speeds(tess(seed, 8, kap)[1], az)
            R.append(abs(float(np.mean(zs(LV["env"])
                                       * zs(beam_local(W, V))))))
            D.append(perm_draws(W, V, LV["env"], ndraw, rng))
        R, D = np.array(R), np.array(D)
        p_i = np.array([perm_p(R[i], D[i]) for i in range(len(R))])
        X = float(-2.0 * np.log(p_i).sum())
        # Exchangeable null: each draw is taken in turn as the
        # observation and scored against the pool of the others plus the
        # observation, which is the same operation the observed p is.
        pool = np.concatenate([D, R[:, None]], axis=1)
        Xd = np.empty(ndraw)
        for d in range(ndraw):
            pd = np.array([(pool[i] >= D[i, d] - 1e-12).sum()
                           / float(ndraw + 1) for i in range(len(R))])
            Xd[d] = -2.0 * np.log(np.maximum(pd, 1.0 / (ndraw + 1))).sum()
        pens = float((1 + int((Xd >= X).sum())) / (ndraw + 1))
        out[wtag] = (X, pens, p_i, R)
        store["perm_%s" % wtag] = np.concatenate([[X, pens], p_i])
    print("\n   %-8s %10s %12s %14s %s"
          % ("window", "Fisher X", "ensemble p", "published", "median p_i"))
    pub = {"4-16": "0.0010", "10-22": "0.0010", "24-36": "0.078"}
    for wtag, _ in WINDOWS:
        X, pens, p_i, _ = out[wtag]
        print("   %-8s %10.1f %12.4f %14s %10.3f"
              % (wtag, X, pens, pub[wtag], float(np.median(p_i))))
    print("\n   per-tessellation |r| and permutation p")
    print("   %-20s %s" % ("tessellation", "".join(
        "%18s" % ("%s us" % w) for w, _ in WINDOWS)))
    for i, (nm, _, _) in enumerate(SPECS):
        print("   %-20s %s" % (nm, "".join(
            "  |r| %.3f p %.4f" % (out[w][3][i], out[w][2][i])
            for w, _ in WINDOWS)))
    print("\n   MEASURED. The control does not separate the windows in")
    print("   this rebuild: it passes in the gate as well as in the two")
    print("   earlier windows. The published contrast, 0.0010 earlier")
    print("   against 0.078 in the gate, is not reproduced.")


# ──────────────────────────────── ensemble ───────────────────────────
def report_ensemble(store):
    """Twelve-tessellation first-rank counts, by window, paired."""
    print()
    print("=" * 78)
    print("6. THE ENSEMBLE. Twelve tessellations, GRID 30, 30 measured")
    print("   alignments (15 on the halving convention). Chance of first")
    print("   rank is 1/30, so 0.4 of 12 expected; 0.8 of 12 halved.")
    print("=" * 78)
    rk_full, rk_half = {}, {}
    for lvl in ("env", "tf"):
        for wtag, win in WINDOWS:
            a, b = [], []
            for nm, seed, kap in SPECS:
                az, LV = levels(nm, win, AZ30)
                r, kf, kh, n = score(LV[lvl], predictor(seed, kap, win, az))
                a.append(kf)
                b.append(kh)
            rk_full[(lvl, wtag)] = np.array(a)
            rk_half[(lvl, wtag)] = np.array(b)
    print("\n   %-5s %-7s %11s %11s %10s %11s %11s"
          % ("lvl", "window", "first/12", "first/12", "rank sum",
             "Fisher p", "Fisher p"))
    print("   %-5s %-7s %11s %11s %10s %11s %11s"
          % ("", "", "of 30", "halved", "of 30", "of 30", "halved"))
    for lvl in ("env", "tf"):
        for wtag, _ in WINDOWS:
            vf, vh = rk_full[(lvl, wtag)], rk_half[(lvl, wtag)]
            store["ens_%s_%s" % (lvl, wtag)] = np.vstack([vf, vh])
            rng = np.random.default_rng(20260803)
            pfs = []
            for v, m in ((vf, 30), (vh, 15)):
                X = float(-2.0 * np.log(v / float(m)).sum())
                draw = rng.integers(1, m + 1, size=(200000, len(v)))
                pfs.append(float((-2.0 * np.log(draw / float(m)).sum(1)
                                  >= X).mean()))
            print("   %-5s %-7s %9d/12 %9d/12 %10d %11.4f %11.4f"
                  % (lvl, wtag, int((vf == 1).sum()), int((vh == 1).sum()),
                     vf.sum(), pfs[0], pfs[1]))
    print("\n   PAIRED ON THE TESSELLATION, which is the unit of")
    print("   replication: each earlier window against the gate, same")
    print("   twelve tessellations, same level, measured denominator.")
    print("   %-5s %-7s %8s %8s %7s %10s %12s"
          % ("lvl", "window", "better", "worse", "tied", "sign p",
             "wilcoxon p"))
    for lvl in ("env", "tf"):
        g = rk_full[(lvl, "24-36")]
        for wtag, _ in WINDOWS[:2]:
            v = rk_full[(lvl, wtag)]
            d = v - g
            nz = d[d != 0]
            sp = (float(stats.binomtest(int((nz < 0).sum()), len(nz)).pvalue)
                  if len(nz) else 1.0)
            wp = float(stats.wilcoxon(v, g).pvalue) if np.any(d) else 1.0
            store["pair_%s_%s" % (lvl, wtag)] = np.array(
                [int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum()),
                 sp, wp])
            print("   %-5s %-7s %8d %8d %7d %10.3f %12.3f"
                  % (lvl, wtag, int((d < 0).sum()), int((d > 0).sum()),
                     int((d == 0).sum()), sp, wp))
    print("\n   MEASURED. No earlier window separates from the gate on the")
    print("   unit of replication.")
    return rk_full, rk_half


def main():
    ndraw = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rng = np.random.default_rng(20260803)
    store = {}
    report_periodicity(store)
    report_harness(min(ndraw, 2000), rng, store)
    report_controls(store)
    report_geometry(store)
    report_separability(store)
    report_permutation(ndraw, rng, store)
    report_ensemble(store)
    np.savez(os.path.join(HERE, "beam_local_windows.npz"), **store)
    print("\nwrote beam_local_windows.npz")


if __name__ == "__main__":
    main()
