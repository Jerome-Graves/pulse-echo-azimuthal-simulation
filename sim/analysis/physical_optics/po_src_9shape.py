"""Step 9: azimuth-averaged coda envelope vs delay, predicted and
measured.  The gate sensitivity says the predicted energy is not spread
the way the closed form says it should be; find out where it sits.
"""
import os

import numpy as np
from scipy.signal import hilbert

import po_src_3po as PO
from po_src_3po import DT, F0, LAM, NT, ricker_spec

HERE = os.path.dirname(os.path.abspath(__file__))
SWD = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
Wspec, _ = ricker_spec(NT, DT, F0)

t = np.arange(NT) * DT
acc = np.zeros(NT)
AZ = np.arange(0, 360, 4.0)
for a in AZ:
    tr = PO.trace_from(PO.azimuth_response(a, LAM / 8)[0], Wspec)
    acc += np.abs(hilbert(tr)) ** 2
acc /= len(AZ)

accm = None
tm = None
for az in range(0, 360, 12):
    with np.load(os.path.join(SWD, "lic_girdle_s11_ppw10",
                              f"az{az:03d}.npz")) as z:
        tr = np.asarray(z["trace"], float).ravel()
        dt = float(z["dt"])
    e = (np.abs(hilbert(tr)) / np.abs(hilbert(tr)).max()) ** 2
    e = e[:8400]
    accm = e if accm is None else accm + e
    tm = np.arange(len(e)) * dt
accm /= 30

print("azimuth-averaged coda power vs delay (1 us bins), dB re source")
print(f"{'t us':>7}{'range mm':>10}{'measured':>11}{'PO':>10}{'diff':>8}")
for lo in np.arange(20, 40, 1.0):
    a0, a1 = int(lo * 1e-6 / DT), int((lo + 1) * 1e-6 / DT)
    b0, b1 = int(lo * 1e-6 / dt), int((lo + 1) * 1e-6 / dt)
    p = 10 * np.log10(acc[a0:a1].mean() / 2 + 1e-300)
    m = 10 * np.log10(accm[b0:b1].mean() / 2 + 1e-300)
    mark = "  <- gate" if 24 <= lo < 36 else ""
    print(f"{lo:>7.0f}{(lo+0.5)*1e-6*PO.C/2*1e3:>10.1f}{m:>11.2f}{p:>10.2f}"
          f"{m-p:>8.2f}{mark}")

# where does the predicted in-gate energy come from, in range?
g0, g1 = int(24e-6 / DT), int(36e-6 / DT)
w = acc[g0:g1]
tt = t[g0:g1]
cum = np.cumsum(w) / w.sum()
print(f"\nPO in-gate energy: 25/50/75 % arrive by "
      f"{np.interp([0.25,0.5,0.75], cum, tt)*1e6} us")
wm = accm[int(24e-6/dt):int(36e-6/dt)]
ttm = np.arange(len(wm)) * dt + 24e-6
cm = np.cumsum(wm) / wm.sum()
print(f"measured        : 25/50/75 % arrive by "
      f"{np.interp([0.25,0.5,0.75], cm, ttm)*1e6} us")
np.savez(os.path.join(HERE, "po_src_shape.npz"), t=t, po=acc, tm=tm, me=accm)
