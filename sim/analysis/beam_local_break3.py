"""Third pass: the tests that decide the earlier-window claim itself.

  15  partial correlation with the orientation-blind column geometry
      projected out of BOTH the predictor and the level, which is the
      standard form of the separability question.
  16  the orientation-permutation control on the ensemble, in every
      window, raw and with the blind geometry projected out.
  17  what the ensemble first-rank counts are worth once the
      multiplicity the manuscript itself declares is paid, and whether
      the earlier window separates from the gate by a paired
      permutation on the twelve tessellations.

Reads out/tesscache and out/sweeps only.  No CUDA, no solver.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from beam_local_break import (AZ30, GEOM, SPECS, WINDOWS, column,      # noqa
                             fisher, levels, rank_of, resid, shift, tess,
                             tess_path, v_var, vmat)


def main(nperm=2000):
    rng = np.random.default_rng(31415)
    tk8 = "tess_s11_p8_k-8.npz"
    _, ax8, _, _ = tess_path(tk8)

    print("=" * 72)
    print("15.  PARTIAL CORRELATION, GEOMETRY OUT OF BOTH SIDES")
    print("=" * 72)
    print("    seed 11 ppw 8, GRID 60.  'partial' projects the thirteen")
    print("    orientation-blind column descriptors out of the predictor")
    print("    AND out of the level, then correlates.  Rank is against")
    print("    the same projection applied at every circular shift.")
    print("    %-6s %-5s %8s %9s %9s %9s" %
          ("window", "lvl", "r", "rank", "partial", "rank"))
    for wn, wv in WINDOWS.items():
        for kind in ("env", "tf"):
            rots, lev = levels("girdle_perp_ppw8", wv, kind)
            W, G, CR = column(tk8, rots, wv)
            vv = v_var(W, vmat(ax8, rots))
            rs = shift(vv, lev)
            vr = resid(vv, G, GEOM)
            n = len(rots)
            pr = np.empty(n)
            for k in range(n):
                pr[k] = np.corrcoef(vr, resid(np.roll(lev, k), G, GEOM))[0, 1]
            print("    %-6s %-5s %+8.3f %9s %+9.3f %9s"
                  % (wn, kind, rs[0], "%d/%d" % (rank_of(rs)[1], n), pr[0],
                     "%d/%d" % (int((np.abs(pr) >= abs(pr[0])).sum()), n)))

    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("16.  ORIENTATION-PERMUTATION CONTROL ON THE ENSEMBLE")
    print("=" * 72)
    print("    %d draws per tessellation, twelve tessellations, GRID 30,"
          % nperm)
    print("    env level.  'resid' is the same control applied to the")
    print("    predictor with the blind column geometry projected out.")
    print("    %-6s %10s %10s %10s %10s" %
          ("window", "Fisher X2", "p", "X2 resid", "p resid"))
    for wn, wv in WINDOWS.items():
        ps, prs = [], []
        for nm, seed, kap in SPECS:
            rots, lev = levels(nm, wv, "env", AZ30)
            W, G, CR = column((seed, kap), rots, wv)
            _, axl, _, _ = tess(seed, kap)
            V = vmat(axl, rots)
            vv = v_var(W, V)
            r0 = abs(np.corrcoef(vv, lev)[0, 1])
            rr0 = abs(np.corrcoef(resid(vv, G, GEOM), lev)[0, 1])
            hit = hitr = 0
            for _ in range(nperm):
                pm = rng.permutation(V.shape[1])
                vp = v_var(W, V, pm)
                if abs(np.corrcoef(vp, lev)[0, 1]) >= r0:
                    hit += 1
                if abs(np.corrcoef(resid(vp, G, GEOM), lev)[0, 1]) >= rr0:
                    hitr += 1
            ps.append((hit + 1) / (nperm + 1))
            prs.append((hitr + 1) / (nperm + 1))
        x2, p = fisher(ps)
        x2r, pr = fisher(prs)
        print("    %-6s %10.1f %10.5f %10.1f %10.5f" % (wn, x2, p, x2r, pr))
    print("    published: Fisher p = 0.0010 earlier against 0.078 in the")
    print("    gate, 2000 draws.")

    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("17.  WHAT THE FIRST-RANK COUNTS ARE WORTH")
    print("=" * 72)
    R = {}
    for wn, wv in WINDOWS.items():
        for kind in ("env", "tf"):
            ranks = []
            for nm, seed, kap in SPECS:
                rots, lev = levels(nm, wv, kind, AZ30)
                W, G, CR = column((seed, kap), rots, wv)
                _, axl, _, _ = tess(seed, kap)
                rs = shift(v_var(W, vmat(axl, rots)), lev)
                ranks.append(rank_of(rs)[1])
            R[(wn, kind)] = np.array(ranks)
    from math import comb
    print("    Binomial p of the first-rank count against a 1/30 floor,")
    print("    then Sidak-corrected for the twelve cells scored (two")
    print("    predictors x two estimators x three windows), and for the")
    print("    eight the manuscript itself admits.")
    print("    %-6s %-5s %8s %10s %10s %10s" %
          ("window", "lvl", "first", "raw p", "Sidak 12", "Sidak 8"))
    for k, ranks in R.items():
        n1 = int((ranks == 1).sum())
        p = sum(comb(12, i) * (1 / 30) ** i * (29 / 30) ** (12 - i)
                for i in range(n1, 13))
        print("    %-6s %-5s %8d %10.4f %10.3f %10.3f"
              % (k[0], k[1], n1, p, 1 - (1 - p) ** 12, 1 - (1 - p) ** 8))
    print()
    print("    PAIRED PERMUTATION, earlier window against the gate, on")
    print("    the twelve tessellations.  Statistic: the difference in")
    print("    mean rank; null: the window label is exchangeable within")
    print("    each tessellation.  Exact over all 4096 sign flips.")
    print("    %-6s %-5s %10s %10s %10s" %
          ("window", "lvl", "mean gate", "mean early", "p"))
    for kind in ("env", "tf"):
        g = R[("24-36", kind)].astype(float)
        for wn in ("4-16", "10-22"):
            e = R[(wn, kind)].astype(float)
            d = g - e
            obs = d.mean()
            cnt = 0
            for m in range(4096):
                s = np.array([1 if (m >> i) & 1 else -1 for i in range(12)])
                if abs((d * s).mean()) >= abs(obs) - 1e-12:
                    cnt += 1
            print("    %-6s %-5s %10.2f %10.2f %10.4f"
                  % (wn, kind, g.mean(), e.mean(), cnt / 4096))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
