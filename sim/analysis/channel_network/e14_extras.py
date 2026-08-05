"""Follow-ups the first two passes raised.

  A  branch-failure count, the right test for a bimodal error
  B  why the new fifth single maximum rails the concentration fit
  C  the fabric-type classifier, run on today's thirteen labelled sweeps
  D  circular-shift null for the within-sweep beam contrast link
  E  what ensemble would make grain size recoverable from the coda
All CPU, all from stored sweeps and cached builds.
"""
import math
import os
import sys

import numpy as np
from scipy import stats

ANA = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim", "analysis")))
sys.path.insert(0, ANA)
HERE = os.path.dirname(os.path.abspath(__file__))

import tof_axis_recovery as TOF                          # noqa: E402


def sec(t):
    print()
    print("=" * 70)
    print(t)
    print("=" * 70)


# ------------------------------------------------------------------ A
sec("A. BRANCH FAILURE COUNT")
Z = np.load(os.path.join(HERE, "edge1.npz"))
names = Z["names"]
e = Z["e_tpl"]
ng = int(Z["n_girdle"])
for thr in (30.0, 45.0):
    g = int((e[:ng] > thr).sum())
    s = int((e[ng:] > thr).sum())
    tab = [[g, ng - g], [s, len(e) - ng - s]]
    odds, p = stats.fisher_exact(tab, alternative="greater")
    print("  error > %4.0f deg: girdle %d/%d, single max %d/%d, "
          "Fisher exact one sided p = %.4f"
          % (thr, g, ng, s, len(e) - ng, p))
print("  the two-sided rank test on the raw errors is NOT significant")
print("  (reported in the edge-1 run) because three girdles land on the")
print("  right branch by chance; the count of branch failures is the")
print("  statistic that matches the mechanism.")
print()
print("  mod-45 error, which is what the template shape does supply:")
print("    girdle     median %.2f deg, %d of %d within 10 deg"
      % (np.median(Z["e_tpl45"][:ng]),
         int((Z["e_tpl45"][:ng] < 10).sum()), ng))
print("    single max median %.2f deg, %d of %d within 10 deg"
      % (np.median(Z["e_tpl45"][ng:]),
         int((Z["e_tpl45"][ng:] < 10).sum()), len(e) - ng))
print()
print("  axis swing when the assumed kappa is halved and doubled:")
for n, sw in zip(names, Z["swing"]):
    print("    %-22s %6.2f deg" % (n, sw))

# ------------------------------------------------------------------ B
sec("B. THE FIFTH SINGLE MAXIMUM AND THE CONCENTRATION FIT")
print("  %-22s %8s %8s %8s %8s %8s"
      % ("sweep", "A2_meas", "A2_orc", "k_real", "k_amp", "k_shape"))
for i, n in enumerate(names):
    print("  %-22s %8.4f %8.4f %8.2f %8.2f %8.2f"
          % (n, Z["amp2"][i], Z["amp2_oracle"][i], Z["kappa_real"][i],
             Z["kappa_amp"][i], Z["kappa_shape"][i]))
az = np.arange(0.0, 360.0, 1.0)
print()
print("  model two-fold amplitude the depth estimator has to match:")
for k in (2.0, 3.0, 3.93, 5.0, 7.62, 10.0, 20.0, 40.0):
    print("    kappa %+6.2f  A2_model %.4f us"
          % (k, TOF.harmonic(az, TOF.model_tof(az, 0.0, k), 2)[0]))
sel = [i for i, n in enumerate(names) if "single" in str(n)
       or str(n) == "singlemax_ppw8"]
k4 = [i for i in sel if str(names[i]) != "mx_single_s7_ppw8"]
er4 = Z["kappa_amp"][k4] - Z["kappa_real"][k4]
er5 = Z["kappa_amp"][sel] - Z["kappa_real"][sel]
print()
print("  single-maximum kappa error, four tessellations (the published "
      "set): %+.2f +- %.2f" % (er4.mean(), er4.std(ddof=1)))
print("  single-maximum kappa error, five  tessellations (with seed 7): "
      "%+.2f +- %.2f" % (er5.mean(), er5.std(ddof=1)))

# ------------------------------------------------------------------ C
sec("C. CAN THE FABRIC TYPE BE CALLED BEFORE THE AXIS IS ATTEMPTED?")
import fabric_type as FT                                  # noqa: E402

rows, missing = FT.compute_sweeps()
print("  sweeps used: %d girdle, %d single maximum"
      % (sum(r["label"] == "G" for r in rows),
         sum(r["label"] == "S" for r in rows)))
if missing:
    print("  skipped (incomplete on the 30-azimuth common grid): %s"
          % ", ".join(missing))
ens = FT.compute_ensemble()
for kap, tag in ((FT.K_SINGLE, "single maximum"), (FT.K_GIRDLE, "girdle")):
    a2, a4, r = ens[kap]
    print("  ensemble %-15s kappa %+5.2f  A2 %.5f  A4 %.5f  A4/A2 %7.3f"
          % (tag, kap, a2, a4, r))
print("  predicted separation in the ratio: a factor of %.1f"
      % (ens[FT.K_GIRDLE][2] / ens[FT.K_SINGLE][2]))
print()
print("  %-22s %2s %8s %8s %9s %9s"
      % ("sweep", "l", "A2", "A4/A2", "orc A4/A2", "select"))
for r in rows:
    print("  %-22s %2s %8.4f %8.3f %9.3f %9.3f"
          % (r["name"], r["label"], r["a2"], r["r"], r["o_r"],
             r["select"]))
gm = [r["r"] for r in rows if r["label"] == "G"]
sm = [r["r"] for r in rows if r["label"] == "S"]
go = [r["o_r"] for r in rows if r["label"] == "G"]
so = [r["o_r"] for r in rows if r["label"] == "S"]
print("  measured A4/A2 ranges overlap between %.3f and %.3f"
      % (max(min(gm), min(sm)), min(max(gm), max(sm))))
print("  noise-free oracle ranges overlap between %.3f and %.3f"
      % (max(min(go), min(so)), min(max(go), max(so))))
print()
scores = FT.compute_scores(rows)
print("  majority-class baseline %d/%d" % (scores[0]["base"],
                                           scores[0]["n"]))
print("  %-10s %8s %8s %7s %10s" % ("statistic", "resub", "LOO", "AUC",
                                    "p vs base"))
for s in scores:
    print("  %-10s %5d/%2d %5d/%2d %7.3f %10.3f"
          % (s["key"], s["resub"], s["n"], s["loo"], s["n"], s["auc"],
             s["p_vs_base"]))
pairs, pair_scores, orc_pairs = FT.compute_pairs(rows)
print()
print("  MATCHED PAIRS (same tessellation, both fabrics), n = %d"
      % len(pairs))
print("  seeds: %s" % ", ".join(str(s) for s, _, _ in pairs))
for s in pair_scores:
    print("    %-10s %d/%d correct, sign-test p = %.4f"
          % (s["key"], s["wins"], s["n"], s["p"]))

# ------------------------------------------------------------------ D
sec("D. WITHIN-SWEEP BEAM CONTRAST AGAINST CODA LEVEL, SHIFT NULL")
E = np.load(os.path.join(HERE, "edge23.npz"))
with np.load((os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim")) +
             r"\results\ensemble.npz")) as z:
    gaz = z["girdle_azimuth"]
    seeds = z["girdle_seeds"]
import e23_links as L                                    # noqa: E402

print("  predictor: boundary-area-weighted mean square of dv/2vbar over")
print("  FACE-ADJACENT grain pairs, for each beam direction. It is 180")
print("  deg periodic, so the 30 circular shifts realise 15 distinct")
print("  alignments and the p floor is 1/15 = 0.067.")
ranks = []
for sd, lv in zip(seeds, gaz):
    row = L.per_seed(int(sd))
    x = 10 * np.log10(row["dv_ray_az"])
    y = np.asarray(lv, float)
    x = x - x.mean()
    y = y - y.mean()
    rr = np.array([float(np.corrcoef(np.roll(x, s), y)[0, 1])
                   for s in range(15)])
    rank = int(1 + (rr > rr[0]).sum())
    ranks.append(rank)
    print("  seed %3d  r %+.3f  rank %2d of 15  p %.3f"
          % (sd, rr[0], rank, rank / 15.0))
ranks = np.array(ranks)
k = int((ranks == 1).sum())
p = float(sum(math.comb(8, i) * (1 / 15.) ** i * (14 / 15.) ** (8 - i)
              for i in range(k, 9)))
print("  first of fifteen in %d of 8; Bin(8, 1/15) upper tail p = %.4g"
      % (k, p))

# ------------------------------------------------------------------ E
sec("E. WHAT ENSEMBLE WOULD MAKE GRAIN SIZE RECOVERABLE?")
cross = E["cross"]
level = E["level"]
b, a = np.polyfit(cross, level, 1)
pred = a + b * cross
res = level - pred
r, pv = stats.pearsonr(cross, level)
print("  level = %+.4f dB per crossing x n_cross %+.2f dB" % (b, a))
print("  r = %+.3f (p = %.3f, n = 8); residual sd %.3f dB"
      % (r, pv, res.std(ddof=1)))
print("  n_cross over the ensemble: %.2f +- %.2f (%.1f %% of the mean)"
      % (cross.mean(), cross.std(ddof=1),
         100 * cross.std(ddof=1) / cross.mean()))
print("  level spread explained by that: %.2f dB of the %.2f dB measured"
      % (abs(b) * cross.std(ddof=1), level.std(ddof=1)))
print()
print("  A beam column of fixed length crosses boundaries at a rate set")
print("  by the mean intercept length, so n_cross scales as 1/D. To move")
print("  the coda level by a given amount the grain size must move by:")
for want in (1.0, 2.0, 3.0, 6.0):
    dn = want / abs(b)
    fD = cross.mean() / (cross.mean() + dn)
    print("    %.0f dB  needs d(n_cross) = %5.1f, i.e. D x %.2f "
          "(a factor %.2f in grain diameter)" % (want, dn, fD, 1 / fD))
print()
print("  With the present ensemble the level is measured to a residual")
print("  of %.2f dB about the n_cross fit, so a grain-size estimate from"
      % res.std(ddof=1))
print("  the coda alone would carry sigma(n_cross) = %.1f, i.e."
      % (res.std(ddof=1) / abs(b)))
print("  sigma(D)/D = %.0f per cent, on an ensemble whose D varies by"
      % (100 * res.std(ddof=1) / abs(b) / cross.mean()))
print("  far less than that.")
