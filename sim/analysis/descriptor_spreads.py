"""Spread of the low-dimensional descriptors across the eight tessellations.

WHY THIS EXISTS. The conclusions state that the present design very
nearly holds the low-dimensional descriptors fixed, which is what makes
"estimating grain size from the coda remains open" a scope statement
about the design rather than a failure of the method. That sentence
originally quoted a single figure, "at most 2.4 per cent", for three
descriptors at once, and the appendix it cited reported only one of the
three. A referee following the citation would have found a third of the
claim supported.

The numbers were never wrong. They were unpublished. This module prints
all four so the appendix can carry them.

WHAT IT DOES NOT DO. It computes nothing new. tessellation_geometry in
ensemble_stats.py already builds the four descriptors, and the spread
convention already used there is the standard deviation over the mean,
with ddof = 1. Both are imported rather than restated, so this module
cannot drift from the analysis that produced the published figures.

The descriptors, from that function's own docstring: the realised grain
count, the mean equivalent grain diameter, the total interior
grain-boundary area of the disc, and the number of grain-boundary
crossings inside the beam column over the coda gate. The tessellation is
the production one, since the Laguerre seeds and weights do not depend
on the grid; only the rasterised area is coarser, and face counting on a
cubic grid overestimates a smooth area by a fixed factor that cancels in
a comparison across seeds.

READS  nothing on disk. The tessellations are rebuilt from their seeds.
WRITES nothing. Prints a table.

CPU only. tessellation_geometry stubs out the GPU labelling path for
exactly this reason, so no CUDA device is required or used.

Run:  python descriptor_spreads.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ensemble_stats import tessellation_geometry            # noqa: E402

# The eight girdle tessellations of the ensemble, in the order Section 4
# reports them: the development specimen first, then the seven simulated
# after the model was fixed.
SEEDS = [11, 7, 17, 23, 41, 53, 71, 89]

LABELS = [("n_grains", "realised grain count", "%8.0f"),
          ("d_mean_mm", "mean equivalent diameter (mm)", "%8.2f"),
          ("area_cm2", "interior boundary area (cm2)", "%8.1f"),
          ("crossings", "beam-column crossings", "%8.1f")]


def main():
    rows = tessellation_geometry(SEEDS)

    print("\nPER TESSELLATION")
    print("  %-6s %8s %10s %12s %11s"
          % ("seed", "grains", "d (mm)", "area (cm2)", "crossings"))
    for row in rows:
        print("  %-6d %8d %10.2f %12.1f %11.1f"
              % (row["seed"], row["n_grains"], row["d_mean_mm"],
                 row["area_cm2"], row["crossings"]))

    print("\nSPREAD ACROSS THE EIGHT, standard deviation over the mean")
    print("  %-32s %10s %10s %9s" % ("descriptor", "mean", "sd", "spread"))
    out = {}
    for key, name, fmt in LABELS:
        x = np.array([row[key] for row in rows], float)
        spread = 100.0 * x.std(ddof=1) / x.mean()
        out[key] = spread
        print(("  %-32s " + fmt + " %10.3f %8.1f%%")
              % (name, x.mean(), x.std(ddof=1), spread))

    largest = max((v, k) for k, v in out.items()
                  if k != "crossings")[0]
    print("\n  Of the three bulk descriptors the largest spread is "
          "%.1f per cent." % largest)
    print("  The crossing count is the one that varies appreciably, at "
          "%.1f per cent," % out["crossings"])
    print("  which is why it is the only descriptor that tracks the "
          "measured level.")


if __name__ == "__main__":
    main()
