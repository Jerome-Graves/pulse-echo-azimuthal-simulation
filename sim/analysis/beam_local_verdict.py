"""Adjudication of the beam-local predictor of Section 5.2, and of the
earlier-window sentence that rests on it.

WHAT THE PREDICTOR IS, established from code and not from the prose.
The Section 5.2 sentence describes beam_descriptors.v_var verbatim: a
rectangular column of the ELEMENT width, 6.35 mm, marched at h/2 over
the depth range the window listens to, lateral offsets at the grid
pitch, through-thickness slices every 2 mm, and the variance of the qP
speed for that beam direction taken over the SAMPLED POINTS, so each
grain enters weighted by the volume of it the column holds.  Rebuilt
here from out/tesscache (no DiskSpecimen.build, no solver, no CUDA) it
returns r = +0.5730 at ppw 6 and +0.3597 at ppw 8 against the published
0.57 and 0.36.  One vote per distinct grain does not reproduce.  The
estimator is therefore identified; no module in either repository
computes the published beam-local figures end to end.

WHAT THIS MODULE DECIDES.  Section 5.2 claims a weak orientation
channel is visible in an earlier window this paper does not analyse in.
Work on the BULK predictor showed the earlier windows are inadmissible
for it: the predictor tracks specimens that cannot backscatter from a
grain boundary at all better than it tracks the specimen.  This module
runs the same test on the BEAM-LOCAL predictor, and it fails it the
same way.  On the thirty azimuths shared by the specimen and both
zero-contrast controls, all three carried on the identical seed-11
tessellation, and on the transform-free in-band level that Section 5.6
requires outside the published gate, |r| at the true alignment is

  window        specimen    zerocontrast   cs_f000
  4-16 us         0.26          0.42         0.52
  10-22 us        0.02          0.38         0.42
  24-36 us        0.21          0.12         0.13

so both controls beat the specimen in both earlier windows and neither
beats it in the gate.  Three further measurements say the same thing.
At 4-16 us the specimen's ABSOLUTE in-band power stands 0.45 dB above
one control and 0.97 dB above the other, against 20 dB or more in the
other two windows, so that window is front arrival and rendering and
has almost no coda in it.  At 10-22 us the specimen is at chance under
both estimators, rank 30 of 30 and 27 of 30, while a record that
cannot scatter is at 0.38 and 0.42.  And the specimen's azimuthal level
is itself predicted by a zero-scattering record on the same
tessellation, r = 0.48 at 4-16 us and 0.40 at 10-22 us, against 0.08 at
most in the gate: the earlier-window azimuthal variation is
substantially the rendering of the tessellation.

WHAT REPRODUCES AND WHAT DOES NOT.  The gate correlations reproduce to
three decimals.  The ensemble first-rank counts reproduce: 3 of 12 at
10-22 us, p = 0.0065 on a 1/30 floor, against 1 or 2 of 12 in the gate.
Two published statements do not.  First, "ranks first of the thirty
distinct alignments" halves a grid this predictor does not permit
halving: r(pred(az), pred(az+180)) is +0.585 in the gate and NEGATIVE,
-0.101 and -0.294, in the two earlier windows, where the bulk
predictor is a function of |c.n| and is exactly 1.  All sixty shifts
are distinct alignments; the measured rank is 2 of 60 at ppw 6 and 1 or
2 of 60 at ppw 8 depending on the level estimator, and the exact
shift-null p is unchanged at 0.033.  Second, the orientation-
permutation control does not show the published window contrast: on the
ensemble over 2000 draws it gives Fisher p = 0.028, 0.005 and 0.013 for
the three windows, so it passes in the gate too and 0.0010 against
0.078 is not recoverable.

WHAT ALSO FAILS, AND CUTS AGAINST THE GATE RESULT THE PAPER KEEPS.
"Pure-geometry column descriptors do not track the level at all" is not
reproducible.  Five descriptors computed from the label volume and the
Laguerre seeds alone, and so bit-identical under any reassignment of
c-axes, reach |r| = 0.31 in the gate (summed facet directivity, rank 8
of 60) and 0.32 at 10-22 us (boundary-crossing count, rank 5 of 60,
where the predictor itself is at rank 46 of 60).  Blind geometry
outranks the c-axis predictor in both earlier windows.

Run: python beam_local_verdict.py
"""
import os
import sys
import time

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import binomtest, chi2, wilcoxon

SIM = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.insert(0, os.path.join(SIM, "sim"))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import forward as F                                        # noqa: E402

SWD = os.path.join(SIM, "out", "sweeps")
TESS = os.path.join(SIM, "out", "tesscache")
C_REF, F0 = 3850.0, 2.0e6
BAND = (0.8e6, 3.0e6)
ELEM, DIA = 6.35e-3, 0.100
WINS = {"4-16": (4e-6, 16e-6), "10-22": (10e-6, 22e-6),
        "24-36": (24e-6, 36e-6)}
NDRAW = 2000

# The twelve tessellations with a cached label volume at ppw 8, as in
# axis_window_adjudication.py.
SPECS = ([("girdle_perp_ppw8", 11, "k-8")]
         + [("mx_girdle_s%d_ppw8" % s, s, "k-8")
            for s in (7, 17, 23, 41, 53, 71, 89)]
         + [("singlemax_ppw8", 11, "k3.93")]
         + [("mx_single_s%d_ppw8" % s, s, "k3.93") for s in (17, 23, 41)])

# Neither can backscatter from a grain boundary, and both carry the
# seed-11 geometry the predictor is built from.  cs_f000 holds 30
# azimuths where the production sweeps hold 60, so any cell carrying
# both controls runs on the 30 common azimuths and its floor is 1/30.
CONTROLS = ("zerocontrast_ppw8", "cs_f000_s11_ppw8")


# ───────────────────────────────── levels ────────────────────────────────
def levels(name, win):
    """env  whole-trace band-passed envelope rms, sweep-mean backwall ref
    (beam_descriptors.measure, the estimator the published r is on)
    tf   absolute in-band mean square, no reference of any kind, dB re 1
    (transform-free, and the only form in which a CROSS-SWEEP margin can
    be read: an env dB is referenced to each sweep's own backwall)"""
    d = os.path.join(SWD, name)
    rots, cd, e1, ab = [], [], [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        k0, w = int(2 * DIA / C_REF * fs), int(2e-6 * fs)
        if k0 + w >= len(tr):
            continue
        sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                     btype="band", output="sos")
        bp = sosfiltfilt(sos, tr)
        a, b = int(win[0] * fs), int(win[1] * fs)
        e1.append(np.abs(hilbert(tr))[max(k0 - w, 0):k0 + w].max())
        cd.append(np.sqrt((np.abs(hilbert(bp))[a:b] ** 2).mean()))
        ab.append((bp[a:b] ** 2).mean())
        rots.append(int(f[2:5]))
    e1 = np.array(e1)
    return (np.array(rots), 20 * np.log10(np.array(cd) / e1.mean()),
            10 * np.log10(np.array(ab)))


# ─────────────────────────────── the column ──────────────────────────────
def column(tess, rots, win, blind=False):
    """beam_descriptors' column, from the cache.  Returns the per-grain
    sample counts W[az, grain] and speeds V[az, grain], from which v_var
    is a weighted variance, plus the orientation-blind descriptors."""
    with np.load(os.path.join(TESS, tess)) as z:
        lab = z["labels"]
        ax = np.asarray(z["axes"], float)
        sd = np.asarray(z["seeds"], float)
        h = float(z["h"])
    ng, (nx, ny, nz) = len(ax), lab.shape
    d0, d1 = win[0] * C_REF / 2, win[1] * C_REF / 2
    s = np.arange(d0, d1, h / 2)
    off = np.arange(-ELEM / 2, ELEM / 2 + h, h)
    zc = (np.arange(nz) + 0.5) * h - nz * h / 2
    zk = np.arange(0, nz, max(1, int(round(2e-3 / h))))
    W, V = np.zeros((len(rots), ng)), np.zeros((len(rots), ng))
    G = {k: [] for k in ["n_grain", "n_cross", "vol_mean", "vol_cv",
                         "geom_dir"]}
    for i, r in enumerate(rots):
        a = np.radians(r)
        n = np.array([np.cos(a), np.sin(a), 0.0])
        t = np.array([-np.sin(a), np.cos(a), 0.0])
        V[i] = np.interp(np.arccos(np.clip(np.abs(ax @ n), 0, 1)),
                         F._PSI, F._VQP)
        P = (DIA / 2 * n - s[:, None] * n)[:, None, None, :] \
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
        W[i] = np.bincount(L[L >= 0], minlength=ng)[:ng]
        if blind:
            c = W[i][W[i] > 0]
            A, B = L[:-1], L[1:]
            cr = (A != B) & (A >= 0) & (B >= 0)
            d = sd[B[cr]] - sd[A[cr]]
            d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-30
            G["n_grain"].append(len(c))
            G["vol_mean"].append(c.mean())
            G["vol_cv"].append(c.std() / c.mean())
            G["n_cross"].append(float(cr.sum()))
            G["geom_dir"].append(float(np.abs(d @ n).sum()))
    return W, V, {k: np.array(v, float) for k, v in G.items()}


def v_var(W, V):
    n = W.sum(1)
    m = (W * V).sum(1) / n
    return (W * V ** 2).sum(1) / n - m ** 2


def shifts(x, y):
    return np.array([np.corrcoef(x, np.roll(y, k))[0, 1]
                     for k in range(len(y))])


def rank_of(x, y):
    rs = shifts(x, y)
    return rs[0], int(np.sum(np.abs(rs) >= abs(rs[0]))), len(y), rs


# ──────────────────────────────── the runs ───────────────────────────────
def gate_reproduction():
    print("\n=== 1. GATE REPRODUCTION AND PERIODICITY, GRID 60 ===")
    for sw, tc, tag in [("girdle_perp", "tess_s11_p6_k-8.npz", "ppw 6"),
                        ("girdle_perp_ppw8", "tess_s11_p8_k-8.npz",
                         "ppw 8")]:
        rots, env, ab = levels(sw, WINS["24-36"])
        W, V, _ = column(tc, rots, WINS["24-36"])
        x = v_var(W, V)
        for nm, y in [("env", env), ("tf", ab)]:
            r0, k, m, rs = rank_of(x, y)
            j = int(np.argsort(-np.abs(rs))[0])
            print(f"  {tag} {nm:4s} r={r0:+.4f}  rank {k}/{m}   "
                  f"best shift {int(rots[j] - rots[0]):3d} deg "
                  f"{rs[j]:+.4f}   first-30-shift rank "
                  f"{int(np.sum(np.abs(rs[:30]) >= abs(r0)))}/30")
        print(f"  {tag}      r(pred, pred+180) = "
              f"{np.corrcoef(x, np.roll(x, len(x) // 2))[0, 1]:+.4f}")
    rots, _, _ = levels("girdle_perp_ppw8", WINS["24-36"])
    for w, ww in WINS.items():
        W, V, _ = column("tess_s11_p8_k-8.npz", rots, ww)
        x = v_var(W, V)
        print(f"  ppw 8 {w:6s} r(pred, pred+180) = "
              f"{np.corrcoef(x, np.roll(x, len(x) // 2))[0, 1]:+.4f}")


def admissibility():
    print("\n=== 2. THE DECISIVE TEST ===")
    rots30 = np.array(sorted(int(f[2:5]) for f in
                             os.listdir(os.path.join(SWD, CONTROLS[1]))
                             if f.startswith("az") and f.endswith(".npz")))
    L, sel = {}, {}
    for sw in ("girdle_perp_ppw8",) + CONTROLS:
        for w, ww in WINS.items():
            rots, env, ab = levels(sw, ww)
            L[(sw, w)] = (env, ab)
            sel[sw] = np.array([i for i, r in enumerate(rots)
                                if r in set(rots30)])
    X = {}
    for grid, rr in [("60", np.arange(0, 360, 6)), ("30", rots30)]:
        for w, ww in WINS.items():
            W, V, _ = column("tess_s11_p8_k-8.npz", rr, ww)
            X[(grid, w)] = v_var(W, V)

    print("  ABSOLUTE in-band power, dB re 1, GRID 30 (no reference)")
    for w in WINS:
        v = [10 * np.log10(np.mean(10 ** (L[(sw, w)][1][sel[sw]] / 10)))
             for sw in ("girdle_perp_ppw8",) + CONTROLS]
        print(f"    {w:7s} specimen {v[0]:8.2f}   zerocontrast {v[1]:8.2f}"
              f"   cs_f000 {v[2]:8.2f}   margins {v[0] - v[1]:5.2f} "
              f"{v[0] - v[2]:5.2f} dB")

    for grid, sws in [("60", ("girdle_perp_ppw8", CONTROLS[0])),
                      ("30", ("girdle_perp_ppw8",) + CONTROLS)]:
        print(f"  |r| AT THE TRUE ALIGNMENT, GRID {grid} "
              f"({X[(grid, '4-16')].size} alignments, floor "
              f"1/{X[(grid, '4-16')].size})")
        for sw in sws:
            for w in WINS:
                out = []
                for j, nm in enumerate(["env", "tf"]):
                    y = L[(sw, w)][j]
                    if grid == "30" and len(y) == 60:
                        y = y[sel[sw]]
                    r0, k, m, _ = rank_of(X[(grid, w)], y)
                    out.append(f"{nm} {r0:+.4f} {k:>2}/{m}")
                print(f"    {sw:20s}{w:7s}" + "   ".join(out))

    print("  r(specimen level, zero-scattering level), GRID 30")
    for w in WINS:
        o = []
        for j, nm in enumerate(["env", "tf"]):
            a = L[("girdle_perp_ppw8", w)][j][sel["girdle_perp_ppw8"]]
            o += [f"{nm} vs {c.split('_')[0]} "
                  f"{np.corrcoef(a, L[(c, w)][j][sel[c]])[0, 1]:+.3f}"
                  for c in CONTROLS]
        print(f"    {w:7s}" + "  ".join(o))


def blind_panel():
    print("\n=== 3. ORIENTATION-BLIND COLUMN DESCRIPTORS, GRID 60 ===")
    rots, _, _ = levels("girdle_perp_ppw8", WINS["24-36"])
    for w, ww in WINS.items():
        _, env, ab = levels("girdle_perp_ppw8", ww)
        W, V, G = column("tess_s11_p8_k-8.npz", rots, ww, blind=True)
        G["v_var (c-axis)"] = v_var(W, V)
        print(f"  -- {w} us")
        for k, x in G.items():
            r1, k1, m, _ = rank_of(x, env)
            r2, k2, _, _ = rank_of(x, ab)
            print(f"     {k:16s} env {r1:+.4f} {k1:>3}/{m}    "
                  f"tf {r2:+.4f} {k2:>3}/{m}")


def ensemble():
    print("\n=== 4. TWELVE TESSELLATIONS AND THE PERMUTATION CONTROL ===")
    rng = np.random.default_rng(20260803)
    rots30 = np.array(sorted(int(f[2:5]) for f in
                             os.listdir(os.path.join(SWD, CONTROLS[1]))
                             if f.startswith("az") and f.endswith(".npz")))
    res, t0 = {}, time.time()
    for sw, seed, kap in SPECS:
        rots, _, _ = levels(sw, WINS["24-36"])
        s = np.array([i for i, r in enumerate(rots) if r in set(rots30)])
        for w, ww in WINS.items():
            _, env, ab = levels(sw, ww)
            W, V, _ = column("tess_s%d_p8_%s.npz" % (seed, kap), rots30, ww)
            x = v_var(W, V)
            for j, nm in enumerate(["env", "tf"]):
                res[(sw, w, nm)] = rank_of(x, [env, ab][j][s])[1]
            y, r_true = env[s], abs(np.corrcoef(x, env[s])[0, 1])
            hits = sum(abs(np.corrcoef(v_var(W, V[:, rng.permutation(
                V.shape[1])]), y)[0, 1]) >= r_true for _ in range(NDRAW))
            res[(sw, w, "perm")] = (hits + 1) / (NDRAW + 1)
        print(f"    {sw:22s} {time.time() - t0:6.1f}s", flush=True)

    for nm in ["env", "tf"]:
        for w in WINS:
            r = [res[(s[0], w, nm)] for s in SPECS]
            print(f"  {nm:4s}{w:7s} first {sum(k == 1 for k in r):2d}/12  "
                  f"ranksum {sum(r):4d} (chance 186)  p(Bin,1/30) "
                  f"{binomtest(sum(k == 1 for k in r), 12, 1 / 30, 'greater').pvalue:.4f}")
    for nm in ["env", "tf"]:
        g = np.array([res[(s[0], "24-36", nm)] for s in SPECS], float)
        for w in ["4-16", "10-22"]:
            e = np.array([res[(s[0], w, nm)] for s in SPECS], float)
            nb, nw = int((g > e).sum()), int((g < e).sum())
            sp = binomtest(min(nb, nw), nb + nw).pvalue if nb + nw else 1.0
            print(f"  {nm:4s}{w:7s} paired vs gate: earlier better {nb:2d} "
                  f"worse {nw:2d} tied {12 - nb - nw:2d}  sign p={sp:.3f} "
                  f" wilcoxon p={wilcoxon(e, g).pvalue:.3f}")
    for w in WINS:
        p = np.array([res[(s[0], w, "perm")] for s in SPECS])
        X = -2 * np.log(p).sum()
        print(f"  perm {w:7s} Fisher X2={X:6.2f}  p={chi2.sf(X, 24):.4f}")


if __name__ == "__main__":
    gate_reproduction()
    admissibility()
    blind_panel()
    ensemble()
