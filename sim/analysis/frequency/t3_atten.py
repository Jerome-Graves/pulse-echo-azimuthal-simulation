"""TEST 3, part 1b: how much of the 2 -> 5 MHz level change is attenuation?

The coda/E1 metric divides by the BACKWALL, whose two-way path (200 mm) is
LONGER than the coda gate centre's (115.5 mm).  So extra attenuation at
5 MHz biases coda/E1 UPWARD and would MASK a fall.  Size that bias.

env(t) = G(d) * exp(-alpha c t),  d = ct/2.  G is a power of d with the
SAME exponent at both frequencies (beam radius is proportional to d at
both), so

    d/dt [ ln env_5(t) - ln env_2(t) ] = -c (alpha_5 - alpha_2)

with G cancelling.  Fit the DIFFERENCE directly, bootstrap over azimuths.
"""
import numpy as np
from scipy.signal import hilbert, sosfiltfilt

import t3_common as T

C = T.C_REF
FIT = (16e-6, 44e-6)      # after source ring-down, before the backwall rise
AZ5 = T.az_list("prod5mhz")
TG = np.arange(FIT[0], FIT[1], 0.25e-6)     # common time grid


def env_grid(name, azs):
    """per-azimuth band-passed envelope resampled onto TG"""
    c = T.cfg_of(name)
    f0 = c["f0_mhz"] * 1e6
    E = []
    for a in azs:
        tr, dt = T.load_trace(name, a)
        fs = 1 / dt
        sos, _, _ = T.sos_of(fs, f0)
        e = np.abs(hilbert(sosfiltfilt(sos, tr)))
        t = np.arange(len(e)) * dt
        E.append(np.interp(TG, t, e))
    return np.array(E)


E2 = env_grid("fittest", AZ5)
E5 = env_grid("prod5mhz", AZ5)
print(f"envelopes: 2 MHz {E2.shape}, 5 MHz {E5.shape}, "
      f"t = {FIT[0]*1e6:.0f}-{FIT[1]*1e6:.0f} us")


def slope(e2, e5):
    d = np.log(np.sqrt((e5 ** 2).mean(0))) - np.log(np.sqrt((e2 ** 2).mean(0)))
    return np.polyfit(TG, d, 1)[0], d


s0, d0 = slope(E2, E5)
rng = np.random.default_rng(1)
bs = []
for _ in range(3000):
    i = rng.integers(0, len(AZ5), len(AZ5))
    bs.append(slope(E2[i], E5[i])[0])
bs = np.array(bs)
lo, hi = np.percentile(bs, [2.5, 97.5])


def to_alpha(s):
    return -s / C


print(f"\nd/dt[ln env5 - ln env2] = {s0:.4g} Np/s   "
      f"95% CI [{lo:.4g}, {hi:.4g}]")
print(f"alpha_5 - alpha_2 = {to_alpha(s0):+.2f} Np/m   "
      f"CI [{to_alpha(hi):+.2f}, {to_alpha(lo):+.2f}] Np/m")
print(f"                  = {8.686*to_alpha(s0):+.1f} dB/m   "
      f"CI [{8.686*to_alpha(hi):+.1f}, {8.686*to_alpha(lo):+.1f}] dB/m (one way)")

dL = 0.200 - C * 30e-6
corr = 8.686 * to_alpha(s0) * dL
clo = 8.686 * to_alpha(hi) * dL
chi = 8.686 * to_alpha(lo) * dL
print(f"\nbackwall path exceeds gate-centre path by {dL*1e3:.1f} mm")
print(f"=> coda/E1 at 5 MHz is biased by {corr:+.2f} dB "
      f"(CI [{min(clo,chi):+.2f}, {max(clo,chi):+.2f}]) relative to 2 MHz")

meas = -2.10
print(f"\nmeasured  coda/E1 (5 - 2 MHz)  = {meas:+.2f} dB")
print(f"attenuation-corrected          = {meas - corr:+.2f} dB "
      f"(CI on the correction only: "
      f"[{meas - max(clo,chi):+.2f}, {meas - min(clo,chi):+.2f}])")
print(f"implied exponent q, coda ~ f^q = "
      f"{(meas - corr)/(20*np.log10(2.5)):+.3f}")

# how flat is the coda envelope in each sweep separately?
for nm, E in (("2 MHz", E2), ("5 MHz", E5)):
    m = np.log(np.sqrt((E ** 2).mean(0)))
    p = np.polyfit(TG, m, 1)
    print(f"  {nm} mean-env slope {p[0]:+.4g} Np/s "
          f"({8.686*(-p[0]/C):+.1f} dB/m equivalent) -- the coda envelope "
          f"is nearly FLAT,\n        so a single-exponential decay model "
          f"does not describe it; only the DIFFERENCE is used.")

np.savez("t3_atten.npz", corr=corr, corr_lo=min(clo, chi),
         corr_hi=max(clo, chi), dalpha=to_alpha(s0))
