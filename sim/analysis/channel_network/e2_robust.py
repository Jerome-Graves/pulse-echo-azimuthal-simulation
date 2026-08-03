"""Edge 2 robustness: is any of it more than one specimen?

  F  jackknife every edge-2 correlation over the eight tessellations
  G  seed 53 and seed 71 examined against what the chain predicts
  H  combined shift-null test for the beam-local contrast predictor
  I  Edge 4: does the chain survive substituting RECOVERED for realised
"""
import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0,
                r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")

E = np.load(os.path.join(HERE, "edge23.npz"))
Z = np.load(os.path.join(HERE, "edge1.npz"))
SEEDS = E["seeds"]


def sec(t):
    print()
    print("=" * 70)
    print(t)
    print("=" * 70)


def jack(x, y, tag):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r0 = float(stats.pearsonr(x, y)[0])
    out = []
    for i in range(len(x)):
        k = np.arange(len(x)) != i
        out.append(float(stats.pearsonr(x[k], y[k])[0]))
    out = np.array(out)
    worst = int(np.argmin(np.abs(out)))
    print("  %-28s r %+.3f | drop-one range %+.3f to %+.3f | "
          "weakest without seed %d (r %+.3f)"
          % (tag, r0, out.min(), out.max(), SEEDS[worst], out[worst]))
    return r0, out


sec("F. JACKKNIFE OF EVERY EDGE-2 AND EDGE-3 LINK (n = 8)")
targets = {"level": E["level"], "ident": E["ident"], "field": E["field"]}
preds = {"a1_vol": E["a1"], "a1_unw": E["a1u"], "a3_vol": E["a3"],
         "mis_adj": E["mis_mean"], "dv_ray": E["dv_ray"],
         "dv_disc": E["dv_disc"], "crossings": E["cross"],
         "area": E["area"]}
for tn, tv in targets.items():
    print(" target: %s" % tn)
    for pn, pv in preds.items():
        jack(pv, tv, pn)

sec("G. THE TWO NAMED SPECIMENS AGAINST WHAT THE CHAIN PREDICTS")
hdr = ("seed  a1_unw  a1_vol   a3    dv_ray   dv_disc  n_cross  level   "
       "ident   i_rank  field  f_rank  scalar")
print("  " + hdr)
with np.load(r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim"
             r"\analysis\tessellation_replication.npz") as z:
    irank = z["ident_own"][:8, 1]
    frank = z["field"][:8, 1]
    scal = z["scalar_full"][:8, 0]
    srank = z["scalar_full"][:8, 1]
for i, s in enumerate(SEEDS):
    print("  %4d %7.4f %7.4f %7.4f %8.5f %8.5f %7.2f %7.2f %+7.4f %5d "
          "%+7.4f %5d %+7.4f"
          % (s, E["a1u"][i], E["a1"][i], E["a3"][i], E["dv_ray"][i],
             E["dv_disc"][i], E["cross"][i], E["level"][i],
             E["ident"][i], irank[i], E["field"][i], frank[i], scal[i]))


def rankof(v, i, big_is_good=True):
    o = np.argsort(-v if big_is_good else v)
    return int(np.where(o == i)[0][0]) + 1


for s in (53, 71, 41):
    i = int(np.where(SEEDS == s)[0][0])
    print()
    print("  seed %d:" % s)
    print("    fabric strength a1_unw  rank %d of 8 (1 = strongest)"
          % rankof(E["a1u"], i))
    print("    fabric strength a1_vol  rank %d of 8" % rankof(E["a1"], i))
    print("    ray contrast dv_ray     rank %d of 8 (1 = most contrast)"
          % rankof(E["dv_ray"], i))
    print("    boundary crossings      rank %d of 8 (1 = most)"
          % rankof(E["cross"], i))
    print("    coda level              rank %d of 8 (1 = loudest)"
          % rankof(E["level"], i))
    print("    identification score    rank %d of 8; rank among the 48 "
          "candidates %d" % (rankof(E["ident"], i), irank[i]))
    print("    field correlation       rank %d of 8; shift-null rank %d "
          "of 30" % (rankof(E["field"], i), frank[i]))
    print("    gate scalar             rank %d of 8; shift-null rank %d "
          "of 30" % (rankof(scal, i), srank[i]))

sec("H. COMBINED SHIFT-NULL TEST, BEAM-LOCAL CONTRAST PREDICTOR")
ranks = np.array([2, 2, 5, 5, 2, 2, 2, 1])       # from e14_extras section D
print("  per-seed ranks of the true alignment among 15 distinct "
      "alignments:")
print("   ", dict(zip(SEEDS.tolist(), ranks.tolist())))
p_i = ranks / 15.0
X = -2.0 * np.log(p_i).sum()
pf = float(stats.chi2.sf(X, 2 * len(p_i)))
print("  Fisher combination X = %.2f on %d df, p = %.4g"
      % (X, 2 * len(p_i), pf))
rng = np.random.default_rng(11)
draw = rng.integers(1, 16, size=(200000, 8))
stat = -2.0 * np.log(draw / 15.0).sum(axis=1)
pmc = float((stat >= X).mean())
print("  exact Monte-Carlo null (200000 draws of 8 uniform ranks): "
      "p = %.4g" % pmc)
print("  mean rank %.2f against 8.0 expected under the null"
      % ranks.mean())
pmc2 = float((draw.mean(axis=1) <= ranks.mean()).mean())
print("  mean-rank test p = %.4g" % pmc2)
print("  first of fifteen in %d of 8, which alone is p = %.3f"
      % (int((ranks == 1).sum()),
         1 - (14 / 15.) ** 8 if (ranks == 1).sum() == 1 else float("nan")))

sec("I. EDGE 4: RECOVERED SUBSTITUTED FOR REALISED")
import specimen as SP                                     # noqa: E402

names = [str(n) for n in Z["names"]]
ng = int(Z["n_girdle"])
kr, km = Z["kappa_real"], Z["kappa_amp"]
a1_real = np.array([SP.watson_eigenvalue(k) for k in kr])
a1_meas = np.array([SP.watson_eigenvalue(k) for k in km])
print("  girdle arm, the arm the coda ensemble is drawn from:")
print("  %-22s %8s %8s %9s %9s" % ("sweep", "k_real", "k_rec",
                                   "a3_real", "a3_rec"))
for i in range(ng):
    print("  %-22s %8.2f %8.2f %9.4f %9.4f"
          % (names[i], kr[i], km[i], a1_real[i], a1_meas[i]))
r1 = stats.pearsonr(kr[:ng], km[:ng])
print("  recovered vs realised kappa: r %+.3f p %.3f (n = %d)"
      % (r1[0], r1[1], ng))
r2 = stats.pearsonr(a1_real[:ng], a1_meas[:ng])
print("  recovered vs realised axial eigenvalue: r %+.3f p %.3f"
      % (r2[0], r2[1]))
print()
print("  the coda prediction the network would ask an operator to make,")
print("  run on the realised quantity and then on the recovered one:")
for tag, v in (("realised kappa", kr[:ng]), ("recovered kappa", km[:ng]),
               ("realised a3", a1_real[:ng]),
               ("recovered a3", a1_meas[:ng])):
    for yn, yv in (("coda level", E["level"]), ("ident", E["ident"])):
        r, p = stats.pearsonr(v, yv)
        print("    %-18s -> %-11s r %+.3f  p %.3f" % (tag, yn, r, p))
print()
print("  and the descriptor that DOES predict the level, n_cross, is")
print("  not recoverable from this acquisition at all: no time-of-flight")
print("  statistic in the repository estimates it.")
