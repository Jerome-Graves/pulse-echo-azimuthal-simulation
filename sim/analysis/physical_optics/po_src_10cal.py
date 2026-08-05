"""Step 10: the delay mapping range -> time was assuming a nominal
3850 m/s.  The simulation measures its own diameter-average speed at
every azimuth (the backwall arrival time is stored in each npz), so use
that instead.  Then re-do the anchor and the gate sensitivity.
"""
import os

import numpy as np

import po_src_3po as PO
from po_src_3po import DT, F0, LAM, NT, ricker_spec

HERE = os.path.dirname(os.path.abspath(__file__))
SWD = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "out", "sweeps")))
Wspec, _ = ricker_spec(NT, DT, F0)

from scipy.signal import hilbert                       # noqa: E402

az_m, t1 = [], []
for f in sorted(os.listdir(os.path.join(SWD, "lic_girdle_s11_ppw10"))):
    if not (f.startswith("az") and f.endswith(".npz")):
        continue
    with np.load(os.path.join(SWD, "lic_girdle_s11_ppw10", f)) as z:
        tr = np.asarray(z["trace"], float).ravel()
        dt = float(z["dt"])
        az_m.append(float(z["az"]))
    e = np.abs(hilbert(tr))
    k0, w = int(2 * 0.100 / 3850.0 / dt), int(2e-6 / dt)
    a = max(k0 - w, 0)
    k = a + int(np.argmax(e[a:k0 + w]))
    y0, y1, y2 = e[k - 1], e[k], e[k + 1]
    den = y0 - 2 * y1 + y2
    t1.append((k + (0.5 * (y0 - y2) / den if abs(den) > 1e-30 else 0.0)) * dt)
az_m, t1 = np.array(az_m), np.array(t1)
c_az = 2 * 0.100 / t1
print("measured diameter-average speed from the backwall arrival")
print(f"  mean {c_az.mean():.1f} m/s  ({100*(c_az.mean()/3850-1):+.2f} % vs "
      f"the nominal 3850), range {c_az.min():.0f}-{c_az.max():.0f} m/s "
      f"(+/-{100*(c_az.max()-c_az.min())/2/c_az.mean():.2f} %)")
print(f"  delay error at 34 us if 3850 is assumed: "
      f"{34*(3850/c_az.mean()-1):+.3f} us")

BASE_C = PO.C
AZD = np.arange(0, 360, 2.0)
cd = np.interp(AZD, np.concatenate([az_m, az_m + 360]),
               np.concatenate([c_az, c_az]))


def dense(use_meas_c, gate=None):
    g0 = PO.GATE
    if gate:
        PO.GATE = gate
    lev = []
    for a, c in zip(AZD, cd):
        PO.C = c if use_meas_c else BASE_C
        lev.append(PO.env_rms_gate(PO.trace_from(
            PO.azimuth_response(a, LAM / 8)[0], Wspec)))
    PO.C, PO.GATE = BASE_C, g0
    lev = np.array(lev)
    return 10 * np.log10((lev ** 2).mean()), lev


print("\nPO ensemble mean over 180 azimuths, dB re source")
a, la = dense(False)
b, lb = dense(True)
print(f"  nominal 3850 m/s                : {a:7.2f}")
print(f"  per-azimuth measured speed      : {b:7.2f}   ({b-a:+.2f})")
for g in ((24.5e-6, 35.5e-6), (23.5e-6, 36.5e-6), (25e-6, 35e-6)):
    v, _ = dense(True, g)
    print(f"  gate {g[0]*1e6:.1f}-{g[1]*1e6:.1f} us              : "
          f"{v:7.2f}   ({v-b:+.2f})")
rng = np.random.default_rng(0)
bs = (lb ** 2)[rng.integers(0, len(lb), (20000, len(lb)))].mean(1)
ci = 10 * np.log10(np.percentile(bs, [2.5, 97.5]))
print(f"  bootstrap 95% CI                : [{ci[0]:.2f}, {ci[1]:.2f}]")
s = np.sort(lb ** 2)[::-1]
print(f"  drop the single top azimuth     : "
      f"{10*np.log10(s[1:].mean()):7.2f}")
print(f"  mean of dB {20*np.log10(lb).mean():.2f}, "
      f"median {np.median(20*np.log10(lb)):.2f}")
np.savez(os.path.join(HERE, "po_src_final.npz"), az=AZD, lev=lb, c=cd)
