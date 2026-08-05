"""Second pass on the beam-local predictor: the attacks that need more
than the standard cell table.

  10  the paper's own estimator rule.  Section 5.6 measures that 53.7
      per cent of the apparent envelope power at 10-22 us is imported
      from outside the window by the whole-trace Hilbert transform, and
      rules that every quantity for a window other than the gate be
      measured window-tapered or transform-free.  The beam-local
      earlier-window paragraph is scored on the whole-trace envelope.
      This part remeasures the leakage and rescores the predictor under
      the rule the same paper imposes.
  11  the scattered part of the level.  Subtracting the zero-scattering
      records leaves only what a grain boundary put there.  The
      predictor is rescored against that.
  12  which tessellations carry the earlier-window first ranks, and
      whether they are independent of each other.
  13  how many effectively independent alignments the shift null has,
      and what a rank of 1 against a rank of 2 is worth.
  14  residual on ONE blind descriptor, chosen before the fact as the
      most collinear with the predictor, so that the projection cannot
      be accused of dredging thirteen regressors through sixty points.

Reads out/tesscache and out/sweeps only.  No CUDA, no solver.
"""
import os
import sys

import numpy as np
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from beam_local_break import (AZ30, GEOM, SPECS, WINDOWS, _bp, column,  # noqa
                             fisher, levels, rank_of, resid, shift, SWD,
                             tess, tess_path, v_var, vmat)


def leak(name, win, az_keep=AZ30):
    """Fraction of in-window envelope power imported from outside it.

    Section 5.6's measurement: the loss of in-window envelope power when
    the band-passed trace is tapered to its window BEFORE the transform.
    """
    d = os.path.join(SWD, name)
    a, b = [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        if az_keep is not None and int(f[2:5]) not in az_keep:
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        i0, i1 = int(win[0] * fs), int(win[1] * fs)
        xb = _bp(tr, fs)
        a.append((np.abs(hilbert(xb))[i0:i1] ** 2).mean())
        xt = np.zeros_like(xb)
        xt[i0:i1] = xb[i0:i1]
        b.append((np.abs(hilbert(xt))[i0:i1] ** 2).mean())
    a, b = np.array(a), np.array(b)
    return float(np.mean(1.0 - b / a))


def lvl_taper(name, win, az_keep=AZ30):
    """Envelope power of the trace tapered to its window first."""
    d = os.path.join(SWD, name)
    rots, val = [], []
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
        i0, i1 = int(win[0] * fs), int(win[1] * fs)
        xb = _bp(tr, fs)
        xt = np.zeros_like(xb)
        xt[i0:i1] = xb[i0:i1]
        val.append((np.abs(hilbert(xt))[i0:i1] ** 2).mean())
        rots.append(a)
    return np.array(rots), 10 * np.log10(np.array(val))


def main():
    tk8 = "labels_seed11_ppw8_kappa-8.npz"
    _, ax8, _, _ = tess_path(tk8)

    print("=" * 72)
    print("10.  THE PAPER'S OWN ESTIMATOR RULE, APPLIED TO ITS OWN CLAIM")
    print("=" * 72)
    print("    Section 5.6: 'Every quantity below for a window other than")
    print("    the published gate is therefore measured either on a trace")
    print("    tapered to its window before the transform or on the mean")
    print("    square of the band-passed trace with no transform at all.'")
    print("    The Section 5.2 earlier-window figures are on the")
    print("    whole-trace envelope, which is the estimator that rule")
    print("    excludes.  Leakage remeasured here, seed 11 ppw 8, GRID 30:")
    for wn, wv in WINDOWS.items():
        print("      %-6s imported from outside the window: %5.1f %%"
              % (wn, 100 * leak("girdle_seed11_ppw8_dev", wv)))
    print("    (Section 5.6 reports 53.7 %% at 10-22 and 15.6 %% in the")
    print("    gate.)")
    print()
    print("    THE PREDICTOR UNDER ALL THREE ESTIMATORS, seed 11 ppw 8,")
    print("    GRID 60 where available, window-matched column:")
    print("    %-6s %-22s %8s %9s" % ("window", "estimator", "r", "rank"))
    for wn, wv in WINDOWS.items():
        rots, _ = levels("girdle_seed11_ppw8_dev", wv, "env")
        W, G, CR = column(tk8, rots, wv)
        vv = v_var(W, vmat(ax8, rots))
        for tag, lev in (("whole-trace envelope",
                          levels("girdle_seed11_ppw8_dev", wv, "env")[1]),
                         ("window-tapered envelope",
                          lvl_taper("girdle_seed11_ppw8_dev", wv, None)[1]),
                         ("transform-free 2<x^2>",
                          levels("girdle_seed11_ppw8_dev", wv, "tf")[1])):
            rs = shift(vv, lev)
            r, k, p = rank_of(rs)
            print("    %-6s %-22s %+8.3f %9s"
                  % (wn, tag, r, "%d/%d" % (k, len(rots))))
    print()
    print("    ENSEMBLE ON THE WINDOW-TAPERED ENVELOPE, twelve")
    print("    tessellations, GRID 30, first ranks of 30 alignments:")
    print("    %-6s %10s %10s %10s" % ("window", "first/12", "rank sum",
                                       "Fisher p"))
    tap = {}
    for wn, wv in WINDOWS.items():
        ranks, ps = [], []
        for nm, seed, kap in SPECS:
            rots, lev = lvl_taper(nm, wv)
            W, G, CR = column((seed, kap), rots, wv)
            _, axl, _, _ = tess(seed, kap)
            rs = shift(v_var(W, vmat(axl, rots)), lev)
            r, k, p = rank_of(rs)
            ranks.append(k)
            ps.append(p)
        ranks = np.array(ranks)
        tap[wn] = ranks
        print("    %-6s %10d %10d %10.4f"
              % (wn, int((ranks == 1).sum()), int(ranks.sum()),
                 fisher(ps)[1]))
    print("    chance: 0.4 first of 12, rank sum 186.")

    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("11.  THE SCATTERED PART OF THE LEVEL ONLY")
    print("=" * 72)
    print("    Both controls carry the seed-11 tessellation and cannot")
    print("    backscatter from a grain boundary, so their azimuthal")
    print("    level is the rendering of that tessellation and nothing")
    print("    else.  Subtracting it in power leaves what a grain")
    print("    boundary put there.  GRID 30, absolute in-band power.")
    print("    %-6s %9s %9s %9s %9s" %
          ("window", "scatt frac", "r pred~raw", "r pred~scat", "rank"))
    for wn, wv in WINDOWS.items():
        rots, ls = levels("girdle_seed11_ppw8_dev", wv, "abs", AZ30)
        _, lz = levels("girdle_seed11_ppw8_uniform_axis", wv, "abs", AZ30)
        _, lc = levels("girdle_seed11_ppw8_contrast_f000", wv, "abs", AZ30)
        ps = 10 ** (ls / 10)
        pn = 0.5 * (10 ** (lz / 10) + 10 ** (lc / 10))
        sc = np.maximum(ps - pn, 1e-30 * ps.max())
        W, G, CR = column(tk8, rots, wv)
        vv = v_var(W, vmat(ax8, rots))
        rs = shift(vv, 10 * np.log10(sc))
        r, k, p = rank_of(rs)
        print("    %-6s %9.3f %+9.3f %+9.3f %9s"
              % (wn, float(sc.sum() / ps.sum()),
                 np.corrcoef(vv, ls)[0, 1], r, "%d/%d" % (k, len(rots))))
    print("    'scatt frac' is the fraction of the window's power that a")
    print("    zero-scattering record does NOT already contain.")

    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("12.  WHICH TESSELLATIONS CARRY THE EARLIER-WINDOW HITS")
    print("=" * 72)
    print("    Four seeds carry a girdle AND a single maximum on a")
    print("    bit-identical tessellation (11, 17, 23, 41), so the column")
    print("    geometry of those pairs is the same volume.  The twelve")
    print("    are not twelve independent tests of anything geometric.")
    print("    rank of the true alignment, of 30, GRID 30, env level:")
    print("    %-20s %8s %8s %8s" % ("sweep", "4-16", "10-22", "24-36"))
    hits = {w: [] for w in WINDOWS}
    for nm, seed, kap in SPECS:
        row = []
        for wn, wv in WINDOWS.items():
            rots, lev = levels(nm, wv, "env", AZ30)
            W, G, CR = column((seed, kap), rots, wv)
            _, axl, _, _ = tess(seed, kap)
            rs = shift(v_var(W, vmat(axl, rots)), lev)
            r, k, p = rank_of(rs)
            row.append(k)
            if k == 1:
                hits[wn].append(nm)
        print("    %-20s %8d %8d %8d" % (nm, row[0], row[1], row[2]))
    for wn in WINDOWS:
        print("    first ranks at %-6s: %s" % (wn, ", ".join(hits[wn])
                                               or "none"))

    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("13.  HOW MANY DISTINCT ALIGNMENTS DOES THE NULL REALLY HAVE")
    print("=" * 72)
    print("    The shift p is exact whatever the autocorrelation, but the")
    print("    RANK is not a robust statistic when neighbouring shifts")
    print("    are near-copies of each other.")
    for tk, sw, tag in (("labels_seed11_ppw6_kappa-8.npz", "girdle_seed11_ppw6_axis_perp", "ppw6"),
                        ("labels_seed11_ppw8_kappa-8.npz", "girdle_seed11_ppw8_dev", "ppw8")):
        rots, lev = levels(sw, WINDOWS["24-36"], "env")
        W, G, CR = column(tk, rots, WINDOWS["24-36"])
        _, axl, _, _ = tess_path(tk)
        vv = v_var(W, vmat(axl, rots))
        rs = shift(vv, lev)
        n = len(rots)
        ac = np.array([np.corrcoef(vv, np.roll(vv, k))[0, 1]
                       for k in range(n)])
        neff = n / (1.0 + 2.0 * np.abs(ac[1:n // 2]).sum())
        o = np.argsort(-np.abs(rs))
        print("    %-5s n=%d  effective independent alignments ~ %.1f"
              % (tag, n, neff))
        print("          best shift %d deg |r| %.4f; true alignment |r| "
              "%.4f; margin %.4f"
              % (6 * o[0], abs(rs[o[0]]), abs(rs[0]),
                 abs(rs[o[0]]) - abs(rs[0])))

    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("14.  RESIDUAL ON ONE PRE-CHOSEN BLIND DESCRIPTOR")
    print("=" * 72)
    print("    The descriptor is chosen as the one most collinear with")
    print("    the PREDICTOR, which is a choice made without looking at")
    print("    the level, so the projection cannot be dredging.")
    print("    %-6s %-5s %-11s %8s %8s %9s %9s" %
          ("window", "lvl", "regressor", "r pred", "r res", "rank pred",
           "rank res"))
    for wn, wv in WINDOWS.items():
        for kind in ("env", "tf"):
            rots, lev = levels("girdle_seed11_ppw8_dev", wv, kind)
            W, G, CR = column("labels_seed11_ppw8_kappa-8.npz", rots, wv)
            vv = v_var(W, vmat(ax8, rots))
            cc = {k: abs(np.corrcoef(vv, G[k])[0, 1]) for k in GEOM
                  if G[k].std() > 1e-12}
            b = max(cc, key=cc.get)
            rs = shift(vv, lev)
            rr = shift(resid(vv, G, [b]), lev)
            print("    %-6s %-5s %-11s %+8.3f %+8.3f %9s %9s"
                  % (wn, kind, b, rs[0], rr[0],
                     "%d/%d" % (rank_of(rs)[1], len(rots)),
                     "%d/%d" % (rank_of(rr)[1], len(rots))))


if __name__ == "__main__":
    main()
