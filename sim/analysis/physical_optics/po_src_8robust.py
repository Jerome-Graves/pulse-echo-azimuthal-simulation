"""Step 8: how much of the agreement is carried by single glints, and
which way is the MEASURED level still moving with grid resolution.
"""
import os

import numpy as np
from scipy.signal import hilbert

import po_src_3po as PO
from po_src_3po import DT, F0, GATE, LAM, NT, ricker_spec

HERE = os.path.dirname(os.path.abspath(__file__))
SWD = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
Wspec, _ = ricker_spec(NT, DT, F0)
REAL = {6: "girdle_perp", 8: "girdle_perp_ppw8", 10: "lic_girdle_s11_ppw10"}
COMMON = list(range(0, 360, 12))


def meas(name):
    out = []
    for az in COMMON:
        with np.load(os.path.join(SWD, name, f"az{az:03d}.npz")) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        e = np.abs(hilbert(tr))
        a, b = int(GATE[0] / dt), int(GATE[1] / dt)
        out.append(np.sqrt((e[a:b] ** 2).mean()) / e.max())
    return np.array(out)


print("=== measured coda/source vs grid resolution (30 matched azimuths) ===")
print(f"{'ppw':>4}{'mean of dB':>12}{'dB mean power':>15}{'median dB':>12}"
      f"{'sd dB':>8}")
MP = {}
for p, n in REAL.items():
    r = meas(n)
    MP[p] = r
    print(f"{p:>4}{20*np.log10(r).mean():>12.2f}"
          f"{10*np.log10((r**2).mean()):>15.2f}"
          f"{np.median(20*np.log10(r)):>12.2f}"
          f"{20*np.log10(r).std(ddof=1):>8.2f}")
print("  steps in mean power: "
      f"ppw6->8 {10*np.log10((MP[8]**2).mean()/(MP[6]**2).mean()):+.2f} dB, "
      f"ppw8->10 {10*np.log10((MP[10]**2).mean()/(MP[8]**2).mean()):+.2f} dB")
print("  the measured coda is STILL FALLING with resolution, so the ppw 10")
print("  value is an upper bound on the converged physical level.")

print("\n=== robustness to the tail, 30 matched azimuths ===")
P = np.load(os.path.join(HERE, "po_src_pred.npz"))["lev"]
r = MP[10]


def stats(x):
    d = 20 * np.log10(x)
    s = np.sort(x ** 2)[::-1]
    return (10 * np.log10((x ** 2).mean()),
            10 * np.log10(s[1:].mean()),
            10 * np.log10(s[3:].mean()),
            np.median(d), d.mean())


print(f"{'':18}{'mean pwr':>10}{'drop top1':>11}{'drop top3':>11}"
      f"{'median dB':>11}{'mean dB':>10}")
for tag, x in (("measured ppw10", r), ("physical optics", P)):
    a = stats(x)
    print(f"{tag:18}" + "".join(f"{v:>11.2f}" for v in a))
a, b = stats(r), stats(P)
print(f"{'measured - PO':18}" + "".join(f"{u-v:>+11.2f}" for u, v in zip(a, b)))

print("\n=== dense-grid gate / speed sensitivity (180 azimuths) ===")
AZD = np.arange(0, 360, 2.0)
base_c, base_g = PO.C, PO.GATE


def dense():
    lev = np.array([PO.env_rms_gate(PO.trace_from(
        PO.azimuth_response(a, LAM / 8)[0], Wspec)) for a in AZD])
    s = np.sort(lev ** 2)[::-1]
    return 10 * np.log10((lev ** 2).mean()), 10 * np.log10(s[1:].mean())


ref, ref1 = dense()
print(f"  base                 {ref:7.2f} dB   (drop top azimuth "
      f"{ref1:7.2f})")
for tag, c, g in (("speed -3 %", 3735.0, base_g),
                  ("gate 24.5-35.5 us", base_c, (24.5e-6, 35.5e-6))):
    PO.C, PO.GATE = c, g
    v, v1 = dense()
    print(f"  {tag:20} {v:7.2f} dB ({v-ref:+.2f})   (drop top azimuth "
          f"{v1:7.2f}, {v1-ref1:+.2f})")
    PO.C, PO.GATE = base_c, base_g
