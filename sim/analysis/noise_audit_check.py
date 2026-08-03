"""Re-derive the noise-sweep thresholds from coda_noise_floor.npz.

The noise module reported its thresholds as grid points with an
interpolated value in brackets. This recomputes both from the stored
curves, so the numbers quoted in the adjudication come from the archive
and not from the report that wrote it. It also converts each threshold
into the number of coherent averages a stated single-shot receiver needs,
which is the only form in which the noise result is actionable.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def cross(x, y, floor, rising=True):
    """First abscissa at which a monotone-ish curve falls below a floor.

    The SNR axis is descending in difficulty, so the sweep is walked from
    the high-SNR end and the crossing is linearly interpolated in dB.
    """
    o = np.argsort(-np.asarray(x))
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    for q in range(1, len(x)):
        if y[q] < floor <= y[q - 1]:
            if y[q - 1] == y[q]:
                return float(x[q])
            return float(x[q - 1] + (y[q - 1] - floor) * (x[q] - x[q - 1])
                         / (y[q - 1] - y[q]))
    return None


def main():
    z = np.load(os.path.join(HERE, "coda_noise_floor.npz"), allow_pickle=True)
    snr = z["snr"]
    print("=" * 79)
    print("NOISE SWEEP, thresholds recomputed from the archive")
    print("=" * 79)
    print("  coda below backwall: envelope rms %.1f dB, signal rms %.1f dB"
          % (float(z["coda_env"]), float(z["coda_sig"])))
    print("  wideband-to-in-band conversion %.1f dB"
          % float(z["wideband_gain"]))
    print("  baseline, noiseless: r = %.4f +- %.4f, registration %d of 8, "
          "identification %d of 8"
          % (z["base_r"].mean(), z["base_r"].std(ddof=1),
             int((z["base_reg"] == 1).sum()), int((z["base_id"] == 1).sum())))
    # Two readings of "the count still holds". The counts are means over
    # 20 realisations, so they are never exactly an integer; the reported
    # thresholds use the rounding reading, in which a mean of 3.65 is
    # still four of eight. The strict reading demands no loss at all. Both
    # are printed because the difference is 20 dB and the practical
    # statement depends on which is meant.
    print("\n  %-40s %8s %8s %8s %8s" % ("criterion", "rnd grid", "rnd int",
                                         "strict", "N avg"))
    r0 = float(z["base_r"].mean())
    tests = (("field r at 90 per cent of noiseless", z["r_mean"],
              0.9 * r0, 0.9 * r0),
             ("field r at half noiseless", z["r_mean"], 0.5 * r0, 0.5 * r0),
             ("registration recovers 6 of 8", z["n_reg"], 5.5, 6.0),
             ("registration significant, 2 of 8", z["n_reg"], 1.5, 2.0),
             ("identification recovers 4 of 8", z["n_id"], 3.5, 4.0),
             ("identification significant, 2 of 8", z["n_id"], 1.5, 2.0))
    for lab, y, floor, strict in tests:
        c = cross(snr, y, floor)
        cs = cross(snr, y, strict)
        grid = None
        for q in np.argsort(-snr):
            if y[q] >= floor:
                grid = snr[q]
        n = 10 ** ((c - 40.0) / 10.0) if c is not None else np.nan
        print("  %-40s %8.1f %8.1f %8.1f %8.0f"
              % (lab, grid if grid is not None else np.nan,
                 c if c is not None else np.nan,
                 cs if cs is not None else np.nan, n))
    print("  N avg is the coherent averages a receiver with a 40 dB")
    print("  single-shot backwall-to-noise ratio needs to reach the")
    print("  rounded interpolated threshold, N = 10^((SNR - 40)/10).")
    print("\n  the coda sits %.1f dB below the backwall in the same rms"
          % float(z["coda_sig"]))
    print("  convention the SNR axis uses, so the coda-to-noise ratio at")
    print("  each threshold is the interpolated SNR minus %.1f dB:"
          % -float(z["coda_sig"]))
    for lab, y, floor, _ in tests[2:]:
        c = cross(snr, y, floor)
        if c is not None:
            print("    %-40s %+6.1f dB" % (lab, c + float(z["coda_sig"])))
    print("\n  ring contamination, identification count against ring level")
    for q, v in enumerate(z["ring_var"]):
        print("    %-24s %s" % (str(v), " ".join("%.1f" % x
                                                 for x in z["ring_id"][q])))
    print("    ring levels: %s" % " ".join("%.0f" % x for x in z["ring_level"]))


if __name__ == "__main__":
    main()
