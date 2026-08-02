"""Step 5: the coda level is glint-dominated, so the ENSEMBLE MEAN is the
only stable statistic.  Quantify the sampling error on both sides, and
close out the bandwidth factor in the flat-mirror check.
"""
import os
import time

import numpy as np
from scipy.signal import hilbert

import po_src_3po as PO
from po_src_3po import DT, F0, LAM, NT, ricker_spec

HERE = os.path.dirname(os.path.abspath(__file__))
Wspec, _ = ricker_spec(NT, DT, F0)
rng = np.random.default_rng(0)


def boot_power(x, nb=20000):
    n = len(x)
    m = (x ** 2)[rng.integers(0, n, (nb, n))].mean(1)
    return 10 * np.log10(np.percentile(m, [2.5, 50, 97.5]))


print("=== bandwidth factor of the round-trip response ===")
fr = np.fft.rfftfreq(NT, DT)
for tag, p in (("mirror  H ~ f  ", 1), ("facet   H ~ f^2", 2)):
    e = np.abs(hilbert(np.fft.irfft(Wspec * (fr / F0) ** p, NT)))
    print(f"  {tag}: peak envelope {e.max():.4f} "
          f"({20*np.log10(e.max()):+.2f} dB vs the narrowband f0 value)")

print("\n=== PO ensemble mean on a DENSE azimuth grid ===")
t0 = time.time()
AZD = np.arange(0, 360, 2.0)
lev = np.array([PO.env_rms_gate(PO.trace_from(
    PO.azimuth_response(a, LAM / 8)[0], Wspec)) for a in AZD])
print(f"  {len(AZD)} azimuths at 2 deg ({time.time()-t0:.0f} s)")
np.savez(os.path.join(HERE, "po_src_dense.npz"), az=AZD, lev=lev)
db = 20 * np.log10(lev)
print(f"  dB of mean power {10*np.log10((lev**2).mean()):7.2f}   "
      f"mean of dB {db.mean():7.2f}   median {np.median(db):7.2f}")
ci = boot_power(lev)
print(f"  bootstrap 95% CI on the mean power: "
      f"[{ci[0]:.2f}, {ci[2]:.2f}] dB")
s = np.sort(lev ** 2)[::-1]
cum = np.cumsum(s) / s.sum()
print(f"  top azimuth carries {100*cum[0]:.1f} % of the total power, "
      f"top 5 {100*cum[4]:.1f} %, top 10 {100*cum[9]:.1f} %")

P30 = np.load(os.path.join(HERE, "po_src_pred.npz"))["lev"]
c30 = boot_power(P30)
print(f"  the 30 measured azimuths only: mean power "
      f"{10*np.log10((P30**2).mean()):.2f} dB, CI [{c30[0]:.2f}, {c30[2]:.2f}]")

print("\n=== measured, same statistics ===")
M = np.load(os.path.join(HERE, "po_src_meas.npz"))
r = M["coda10"] / M["bang10"]
cm = boot_power(r)
print(f"  dB of mean power {10*np.log10((r**2).mean()):7.2f}   "
      f"mean of dB {20*np.log10(r).mean():7.2f}   "
      f"median {np.median(20*np.log10(r)):7.2f}")
print(f"  bootstrap 95% CI on the mean power: [{cm[0]:.2f}, {cm[2]:.2f}] dB")
sm = np.sort(r ** 2)[::-1]
cmu = np.cumsum(sm) / sm.sum()
print(f"  top azimuth carries {100*cmu[0]:.1f} % of the total power, "
      f"top 5 {100*cmu[4]:.1f} %")

print("\n=== per-azimuth correlation (30 matched azimuths) ===")
mz = 20 * np.log10(r)
pz = 20 * np.log10(P30)
print(f"  Pearson r on dB = {np.corrcoef(mz, pz)[0,1]:+.3f}")
print(f"  Spearman        = "
      f"{np.corrcoef(np.argsort(np.argsort(mz)), np.argsort(np.argsort(pz)))[0,1]:+.3f}")
print(f"  measured sd across azimuths {mz.std(ddof=1):.2f} dB, "
      f"PO sd {pz.std(ddof=1):.2f} dB")
for a, m_, p_ in zip(M["common"], mz, pz):
    print(f"    az {int(a):3d}   measured {m_:7.2f}   PO {p_:8.2f}")
