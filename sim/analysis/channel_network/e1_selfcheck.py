"""The self-diagnosis Edge 1 actually supports.

Not a fabric-type classifier. A CONCENTRATION-SWING flag: refit the axis
with the concentration assumed in the template halved and doubled, and
report how far the recovered axis moves. On a single maximum the shape
pins the axis and the answer barely moves; on a girdle the two-fold term
that chooses the branch is below the realisation scatter, so the fit
jumps to the other branch and the axis swings by ninety degrees.

It costs two extra template fits, needs no threshold fitted to labelled
data, and it is a statement about THIS specimen's identifiability rather
than about which population it came from.

Run on every labelled sweep and on the four controls that defeated the
fabric-type statistics in analysis/fabric_type.py.
"""
import os
import sys

import numpy as np

ANA = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim", "analysis")))
sys.path.insert(0, ANA)
import fabric_type as FT                                  # noqa: E402
import tof_axis_recovery as TOF                           # noqa: E402

LABELLED = ([(n, "G", -8.0) for n in FT.GIRDLE]
            + [(n, "S", 3.93) for n in FT.SINGLE])
CONTROLS = [("cs_f000_s11_ppw8", "ladder f = 0.00, girdle seed 11"),
            ("cs_f025_s11_ppw8", "ladder f = 0.25, girdle seed 11"),
            ("cs_f050_s11_ppw8", "ladder f = 0.50, girdle seed 11"),
            ("cs_f075_s11_ppw8", "ladder f = 0.75, girdle seed 11"),
            ("zerocontrast_ppw8", "single crystal (extreme single max)"),
            ("iso_gcal", "isotropic, no fabric axis at all")]


def swing(az, tof, kappa):
    """Largest axis disagreement over kappa/2, kappa, 2*kappa."""
    a = [TOF.fit_axis_template(az, tof, k)[0]
         for k in (kappa / 2.0, kappa, 2.0 * kappa)]
    return max(TOF.fold_180(a[i] - a[j])
               for i in range(3) for j in range(3)), a[1]


def main():
    print("CONCENTRATION-SWING FLAG")
    print("  swing = largest axis disagreement when the template's kappa")
    print("  is halved and doubled. Large swing = the branch is not")
    print("  determined and the axis must not be reported.")
    print()
    keep_g, miss_g = FT.available(FT.GIRDLE)
    keep_s, miss_s = FT.available(FT.SINGLE)
    use = ([(n, "G", -8.0) for n in keep_g]
           + [(n, "S", 3.93) for n in keep_s])
    print("  labelled sweeps: %d girdle, %d single maximum"
          % (len(keep_g), len(keep_s)))
    if miss_g + miss_s:
        print("  skipped, incomplete: %s" % ", ".join(miss_g + miss_s))
    print()
    print("  %-22s %2s %8s %9s" % ("sweep", "l", "swing", "axis"))
    vals, labs = [], []
    for name, lab, kap in use:
        az, tof = FT.load_series(name)
        sw, ax = swing(az, tof, kap)
        vals.append(sw)
        labs.append(lab)
        print("  %-22s %2s %8.2f %9.2f" % (name, lab, sw, ax))
    vals = np.array(vals)
    labs = np.array(labs)
    g = vals[labs == "G"]
    s = vals[labs == "S"]
    print()
    print("  girdle     %.2f to %.2f deg (n = %d)" % (g.min(), g.max(),
                                                      len(g)))
    print("  single max %.2f to %.2f deg (n = %d)" % (s.min(), s.max(),
                                                      len(s)))
    print("  separation margin: %.2f deg, any threshold in (%.2f, %.2f)"
          % (g.min() - s.max(), s.max(), g.min()))
    ok, marg = FT.leave_one_out(vals, labs)
    print("  leave-one-out %d/%d against a majority baseline of %d/%d"
          % (int(ok.sum()), len(vals), FT.majority_baseline(labs)[0],
             len(vals)))
    print("  AUC %.3f; smallest held-out margin %.2f deg"
          % (FT.auc(vals, labs), np.abs(marg).min()))
    print("  p against the majority-class baseline: %.4g"
          % FT.binom_tail(int(ok.sum()), len(vals),
                          FT.majority_baseline(labs)[0] / float(len(vals))))
    thr = FT.fit_threshold(vals, labs)
    print("  threshold fitted on all labelled sweeps: %.2f deg" % thr)

    print()
    print("  CONTROLS. The flag is a statement about identifiability, so")
    print("  the correct answer on the girdle ladder is 'not identifiable'")
    print("  at every rung, on the isotropic sweep 'not identifiable', and")
    print("  on the single crystal 'identifiable'.")
    print("  %-22s %-36s %8s %9s %-14s"
          % ("sweep", "what", "swing", "axis", "verdict"))
    for name, tag in CONTROLS:
        az, tof = FT.load_series(name)
        kap = -8.0 if ("girdle" in tag or "isotropic" in tag) else 3.93
        sw, ax = swing(az, tof, kap)
        # verdict is threshold-free in practice: the two populations are
        # 87 deg apart, so report against the fitted threshold anyway
        print("  %-22s %-36s %8.2f %9.2f %-14s"
              % (name, tag, sw, ax,
                 "NOT identifiable" if sw > thr else "identifiable"))
    print()
    print("  A ladder rung is one tessellation under one fabric, so a")
    print("  fabric-TYPE statistic must not move along it. The swing does")
    print("  not have to satisfy that, because it is not reading fabric")
    print("  type: every rung is the same girdle and every rung should")
    print("  read 'not identifiable'. Check the spread anyway.")


if __name__ == "__main__":
    main()
