"""Does 5 MHz make the coda stronger?

fittest  = 2 MHz, ppw 6, seed 7, kappa +3.93, axis [1,0,0], 360 az
prod5mhz = 5 MHz, ppw 6, seed 7, kappa +3.93, axis [1,0,0],  40 az
Same specimen, same fabric, same ppw, same solver.  Only f0 differs.

Because ppw is matched, the staircase artefact bears the SAME ratio to the
wavelength in both, so this is a fair like-for-like frequency comparison.

Three levels are reported so the answer cannot hide in the normalisation:
  bang   peak of the whole trace  (proxy for the transmitted amplitude)
  E1     backwall echo            (attenuated over the full 200 mm round trip)
  coda   24-36 us envelope rms    (backscatter from ~46-69 mm depth)
The scattering REGIME is read off coda/bang: in the Rayleigh regime it rises
as f^4, stochastic as f^2, geometric (grain >> wavelength) it is flat.
"""
import os
import sys

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")

OUT = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
C_REF = 3850.0
CODA_W = (24e-6, 36e-6)


def levels(d, rots=None):
    out = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        r = int(f[2:5])
        if rots is not None and r not in rots:
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        e = np.abs(hilbert(tr))
        k0 = int(2 * 0.100 / C_REF * fs)
        w = int(2e-6 * fs)
        if k0 + w >= len(e):
            continue
        bang = e.max()
        E1 = e[max(k0 - w, 0):k0 + w].max()
        coda = np.sqrt((e[int(CODA_W[0] * fs):int(CODA_W[1] * fs)] ** 2).mean())
        out[r] = (bang, E1, coda, dt, len(tr))
    return out


A = levels(os.path.join(OUT, "fittest"))
B = levels(os.path.join(OUT, "prod5mhz"))
common = sorted(set(A) & set(B))
print(f"fittest(2MHz) {len(A)} az, prod5mhz(5MHz) {len(B)} az, "
      f"{len(common)} common azimuths\n")


def db(x):
    return 20 * np.log10(x)


print(f"{'':<12}{'coda/E1':>10}{'coda/bang':>12}{'E1/bang':>10}"
      f"{'dt (ns)':>10}{'nt':>8}")
rows = {}
for tag, S in (("2 MHz", A), ("5 MHz", B)):
    v = np.array([S[r][:3] for r in common])
    dt = np.mean([S[r][3] for r in common])
    nt = int(np.mean([S[r][4] for r in common]))
    ce = db(v[:, 2] / v[:, 1])
    cb = db(v[:, 2] / v[:, 0])
    eb = db(v[:, 1] / v[:, 0])
    rows[tag] = (ce, cb, eb)
    print(f"{tag:<12}{ce.mean():>9.1f}d{cb.mean():>11.1f}d{eb.mean():>9.1f}d"
          f"{dt*1e9:>10.1f}{nt:>8}")

for k, nm in ((0, "coda/E1"), (1, "coda/bang"), (2, "E1/bang")):
    d = rows["5 MHz"][k] - rows["2 MHz"][k]
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\n  {nm:<10} 5MHz - 2MHz = {d.mean():+6.2f} +- {se:.2f} dB")
    if k == 1:
        r = 10 ** (d.mean() / 20)
        p = np.log(r) / np.log(5.0 / 2.0)
        print(f"             backscatter amplitude ratio {r:.2f}x  ->  "
              f"f^{p:+.2f}")
        print("             Rayleigh f^4 | stochastic f^2 | geometric f^0")

print("\nper-azimuth scatter (rms about the mean, the speckle level):")
for tag in ("2 MHz", "5 MHz"):
    print(f"  {tag:<8} coda/E1 {rows[tag][0].std(ddof=1):5.2f} dB   "
          f"coda/bang {rows[tag][1].std(ddof=1):5.2f} dB")
np.savez(os.path.join(os.path.dirname(__file__), "freq_scaling.npz"),
         common=common, **{f"{k.replace(' ','')}": np.array(v)
                           for k, v in rows.items()})
