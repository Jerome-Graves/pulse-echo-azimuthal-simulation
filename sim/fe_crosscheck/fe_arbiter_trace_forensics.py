"""Forensics on saved A-B cross-check traces (fe_arbiter_round1_baseline*.npz).

Prints, for each solver: per-0.2 us envelope levels of the uniform-B
trace (absolute box-artifact content) and of the A-B difference, plus
window RMS verdicts. This is the analysis that exposed the round-1 FE
box reverberation (-19.7 dB B-content vs FDTD's -52.3) and should show
the quartic pad shell curing it in round 2.

Usage: python fe_arbiter_trace_forensics.py [fe_arbiter_round2_padded_box_traces.npz]
"""
import sys

import numpy as np
from scipy.signal import hilbert

F0 = 1e6
T_DIR = 0.009 / 4046.0 + 1.2 / F0
WINDOWS = ((4.9e-6, 5.6e-6), (4.9e-6, 6.2e-6))


def profile(a, b, dt, name):
    fs = 1.0 / dt
    k = int(T_DIR * fs)
    envb = np.abs(hilbert(b))
    direct = envb[k - int(1e-6 * fs):k + int(1e-6 * fs)].max()
    envd = np.abs(hilbert(a - b))
    print(f"{name}: per-0.2us envelope re direct (B abs | A-B)")
    for t0 in np.arange(4.3e-6, 6.8e-6, 0.2e-6):
        s0, s1 = int(t0 * fs), int((t0 + 0.2e-6) * fs)
        lb = 20 * np.log10(np.sqrt((envb[s0:s1] ** 2).mean())
                           / direct + 1e-30)
        ld = 20 * np.log10(np.sqrt((envd[s0:s1] ** 2).mean())
                           / direct + 1e-30)
        print(f"  {t0*1e6:.1f}-{(t0+0.2e-6)*1e6:.1f} us: "
              f"B {lb:6.1f} | A-B {ld:6.1f}")
    for w0, w1 in WINDOWS:
        s0, s1 = int(w0 * fs), int(w1 * fs)
        ld = 20 * np.log10(np.sqrt((envd[s0:s1] ** 2).mean())
                           / direct + 1e-30)
        print(f"  A-B RMS {w0*1e6:.1f}-{w1*1e6:.1f} us: {ld:.1f} dB")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "fe_arbiter_round2_padded_box_traces.npz"
    d = np.load(path)
    profile(d["fa"], d["fb"], float(d["dtf"]), "FDTD")
    profile(d["ea"], d["eb"], float(d["dte"]), "P2+ FE")


if __name__ == "__main__":
    main()
