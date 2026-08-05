"""EDGE 1: does fabric type gate the axis channel?

Runs Section 5.1's own estimator, from sim/analysis/tof_axis_recovery.py,
so the pick and the template are the manuscript's and not a re-write.
That module forces the CPU labeller at import and every tessellation it
needs is already cached beside it, so nothing here builds or touches CUDA.

Adds mx_single_s7_ppw8, which completed after Table 'axisrecovery' was
written, taking the single-maximum arm from four tessellations to five.

Four questions:
  1. how large is the axis error, by fabric type
  2. is the girdle failure the vanishing two-fold component or noise
  3. over what interval of kappa is the axis channel unusable
  4. can the fabric type be called BEFORE the axis is attempted
"""
import os
import sys

import numpy as np
from scipy import stats

ANA = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim", "analysis")))
sys.path.insert(0, ANA)
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

import tof_axis_recovery as TOF                          # noqa: E402

GIRDLE = TOF.GIRDLE
SINGLE = ("singlemax_ppw8", "mx_single_s7_ppw8", "mx_single_s17_ppw8",
          "mx_single_s23_ppw8", "mx_single_s41_ppw8")
AZ_COMMON = np.arange(0, 360, 12)


def on_common(s):
    """Decimate a 60-azimuth sweep onto the 12 deg grid every sweep has."""
    keep = np.isin(s["az"].astype(int), AZ_COMMON)
    return s["az"][keep], s["tof"][keep], s["oracle"][keep]


def main():
    print("EDGE 1  FABRIC TYPE GATES THE AXIS CHANNEL")
    print()

    spec = TOF.compute_specimens(GIRDLE + SINGLE)
    TOF.compute_axes(spec)
    TOF.compute_kappa(spec)
    ng = len(GIRDLE)

    print("PER SPECIMEN, error against the volume-weighted sample axis")
    print("%-22s %4s %7s %7s %7s %7s %7s %8s %7s"
          % ("sweep", "naz", "floor", "e_tpl", "e_2fold", "e_oracle",
             "mod45", "dk_swing", "rms_us"))
    for s in spec:
        print("%-22s %4d %7.2f %7.2f %7.2f %7.2f %7.2f %8.2f %7.3f"
              % (s["name"], len(s["az"]),
                 TOF.fold_180(s["sample"] - s["nominal"]),
                 s["e_tpl"], s["e_two"], s["e_oracle"], s["e_tpl45"],
                 s["tpl_sensitivity"], s["rms"]))

    print()
    for tag, sel in (("girdle", spec[:ng]), ("single max", spec[ng:])):
        for key, lab in (("e_tpl", "template"), ("e_two", "two-fold"),
                         ("e_oracle", "oracle two-fold"),
                         ("e_tpl45", "template mod 45")):
            e = np.array([s[key] for s in sel])
            print("  %-11s n=%d %-17s median %6.2f  mean %6.2f  "
                  "worst %6.2f  <10 deg %d/%d"
                  % (tag, len(sel), lab, np.median(e), e.mean(), e.max(),
                     int((e < 10).sum()), len(e)))
        sw = np.array([s["tpl_sensitivity"] for s in sel])
        print("  %-11s      %-17s %.1f to %.1f deg"
              % (tag, "axis swing on k/2..2k", sw.min(), sw.max()))
        print()

    ge = np.array([s["e_tpl"] for s in spec[:ng]])
    se = np.array([s["e_tpl"] for s in spec[ng:]])
    u, pu = stats.mannwhitneyu(ge, se, alternative="greater")
    print("  girdle vs single-max template error, Mann-Whitney one sided:"
          " U=%.0f p=%.4f (n=%d,%d)" % (u, pu, len(ge), len(se)))
    print("  the same comparison on the NOISE-FREE oracle:")
    go = np.array([s["e_oracle"] for s in spec[:ng]])
    so = np.array([s["e_oracle"] for s in spec[ng:]])
    u2, pu2 = stats.mannwhitneyu(go, so, alternative="greater")
    print("    girdle median %.2f, single median %.2f, U=%.0f p=%.4f"
          % (np.median(go), np.median(so), u2, pu2))

    # ---------------------------------------------------------------
    print()
    print("IS IT NOISE?  measurement scatter against the signal it must "
          "resolve")
    for tag, sel in (("girdle", spec[:ng]), ("single max", spec[ng:])):
        rms = np.array([s["rms"] for s in sel])
        pick = np.array([s["pick_sd"] for s in sel])
        a2 = np.array([s["amp2"] for s in sel])
        a2o = np.array([s["amp2_oracle"] for s in sel])
        print("  %-11s residual rms %.3f-%.3f us, pick sd %.3f-%.3f us"
              % (tag, rms.min(), rms.max(), pick.min(), pick.max()))
        print("  %-11s measured A2 %.3f+-%.3f us, realised (oracle) A2 "
              "%.3f+-%.3f us" % ("", a2.mean(), a2.std(ddof=1),
                                 a2o.mean(), a2o.std(ddof=1)))

    # ---------------------------------------------------------------
    print()
    print("THE TEMPLATE'S OWN HARMONICS AGAINST KAPPA")
    az = np.arange(0.0, 360.0, 1.0)
    k0 = TOF.two_fold_null_kappa()
    print("  two-fold null at kappa = %.4f" % k0)
    grid = np.r_[np.linspace(-20.0, -0.5, 40), [k0], [-8.0], [3.93],
                 np.linspace(0.5, 20.0, 20)]
    grid = np.unique(np.round(grid, 4))
    rows = []
    for k in grid:
        m = TOF.model_tof(az, 0.0, float(k))
        a2 = TOF.harmonic(az, m, 2)[0]
        a4 = TOF.harmonic(az, m, 4)[0]
        # signed two-fold: + when the pattern is slow along the axis
        sgn = np.sign(m[0] - m[90])
        rows.append((float(k), sgn * a2, a4))
    rows = np.array(rows)
    print("  %8s %10s %10s %8s" % ("kappa", "A2_signed", "A4", "A4/|A2|"))
    for k, a2s, a4 in rows:
        if k in (-20.0, -12.0, -10.0, -8.0, -6.5, k0, -5.0, -3.0, -1.0,
                 3.93, 10.0, 20.0) or abs(k - k0) < 0.6 or k in (-8.0,):
            print("  %8.3f %+10.5f %10.5f %8.1f"
                  % (k, a2s, a4, a4 / max(abs(a2s), 1e-9)))
    m8 = TOF.model_tof(az, 0.0, -8.0)
    print("  at the simulated kappa = -8: A2 %.4f us, A4 %.4f us, "
          "ratio 1 : %.1f"
          % (TOF.harmonic(az, m8, 2)[0], TOF.harmonic(az, m8, 4)[0],
             TOF.harmonic(az, m8, 4)[0] / TOF.harmonic(az, m8, 2)[0]))

    # unusable interval: |A2(kappa)| below the scatter that a single
    # specimen brings to A2, measured two ways
    a2_meas_sd = float(np.std([s["amp2"] for s in spec[:ng]], ddof=1))
    a2_orc_sd = float(np.std([s["amp2_oracle"] for s in spec[:ng]],
                             ddof=1))
    pick_sd = float(np.mean([s["pick_sd"] for s in spec]))
    a2_pick = pick_sd * np.sqrt(2.0 / 30.0)     # sd of A2 from pick noise
    print()
    print("  scatter a single specimen brings to the measured A2:")
    print("    realisation scatter of the exact A2 over eight girdles "
          "%.4f us" % a2_orc_sd)
    print("    total measured scatter of A2 over the same eight     "
          "%.4f us" % a2_meas_sd)
    print("    pick noise alone, sd(pick) %.4f us over 30 azimuths  "
          "%.4f us" % (pick_sd, a2_pick))
    fine = np.linspace(-20.0, -0.2, 4000)
    a2f = np.array([TOF.harmonic(az, TOF.model_tof(az, 0.0, float(k)),
                                 2)[0] for k in fine])
    for tag, thr in (("realisation scatter", a2_orc_sd),
                     ("measured scatter", a2_meas_sd),
                     ("pick noise", a2_pick)):
        bad = fine[a2f < thr]
        if bad.size:
            print("    |A2| < %-20s (%.4f us):  kappa in [%.2f, %.2f]"
                  % (tag, thr, bad.min(), bad.max()))
        else:
            print("    |A2| < %-20s (%.4f us):  empty" % (tag, thr))
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "edge1_a2k.npz"), kappa=fine, a2=a2f,
             null_kappa=k0)

    # ---------------------------------------------------------------
    print()
    print("EDGE 4 INPUT: what the instrument actually recovers")
    print("%-22s %7s %7s %8s %8s %8s"
          % ("sweep", "k_real", "k_amp", "k_shape", "e_tpl", "a1_from_k"))
    import specimen as SP
    for s in spec:
        a1r = SP.watson_eigenvalue(s["kappa_real"])
        a1m = SP.watson_eigenvalue(s["kappa_amp"])
        print("%-22s %7.2f %7.2f %8.2f %8.2f  real %.4f meas %.4f"
              % (s["name"], s["kappa_real"], s["kappa_amp"],
                 s["kappa_shape"], s["e_tpl"], a1r, a1m))
    for tag, sel in (("girdle", spec[:ng]), ("single max", spec[ng:])):
        kr = np.array([s["kappa_real"] for s in sel])
        km = np.array([s["kappa_amp"] for s in sel])
        e = km - kr
        r, p = stats.pearsonr(kr, km)
        print("  %-11s kappa error %+.2f +- %.2f, recovered-vs-realised "
              "r %+.3f p %.3f" % (tag, e.mean(), e.std(ddof=1), r, p))

    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "edge1.npz"),
             names=np.array([s["name"] for s in spec]),
             e_tpl=np.array([s["e_tpl"] for s in spec]),
             e_oracle=np.array([s["e_oracle"] for s in spec]),
             e_tpl45=np.array([s["e_tpl45"] for s in spec]),
             kappa_real=np.array([s["kappa_real"] for s in spec]),
             kappa_amp=np.array([s["kappa_amp"] for s in spec]),
             kappa_shape=np.array([s["kappa_shape"] for s in spec]),
             amp2=np.array([s["amp2"] for s in spec]),
             amp2_oracle=np.array([s["amp2_oracle"] for s in spec]),
             swing=np.array([s["tpl_sensitivity"] for s in spec]),
             n_girdle=ng)


if __name__ == "__main__":
    main()
