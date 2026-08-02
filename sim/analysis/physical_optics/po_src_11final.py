"""Step 11: the closed-form anchor done with correct spectral bookkeeping.

The orientation-averaged backscatter form factor is FREQUENCY FLAT:
    E|f_bs|^2 = (1/4pi) INT R^2 (A/lam)^2 |Lam(2 theta_n)|^2 dOmega_n
              = (1/4pi) R^2 (A^2/lam^2) (lam^2 / 4A)  =  R^2 A /(16 pi)
(the specular lobe narrows as f^-2 exactly as fast as its peak grows as
f^2), so the only surviving frequency dependence of the round trip is the
single factor S_t/lam from the transmit leg.  Doing this with an f^2
pulse-shape shortcut, as an earlier draft did, over-weights the top of
the band.  Everything below is integrated over the true Ricker spectrum.

  <env^2>_gate = (4/T_g) INT_0^inf |W(f)|^2 SSQ(f) df
  SSQ(f) = (S_t/lam)^2 . (R_rms^2 S_v g /(16 pi)) . INT B(f)^4 dOmega
                       . INT_gate dd/d^2
"""
import os

import numpy as np
from scipy.special import j1

from po_src_3po import (A_EL, C, DT, F0, GATE, LAM, NT, R_DISC, S_T, T_G,
                        THK, ricker_spec)

HERE = os.path.dirname(os.path.abspath(__file__))
G = np.load(os.path.join(HERE, "po_src_geom.npz"))
D = np.load(os.path.join(HERE, "po_src_diag.npz"))
S_V = G["farea"].sum() / (np.pi * R_DISC ** 2 * THK)
R_RMS = float(np.sqrt((D["rr"] ** 2 * D["ain"]).sum() / D["ain"].sum()))
G_ORI = 1.123

Wspec, w = ricker_spec(NT, DT, F0)
fr = np.fft.rfftfreq(NT, DT)
dfr = fr[1]
Wc = Wspec * DT                                   # continuous-FT estimate

ph = np.linspace(1e-9, np.pi / 2, 20001)
sp = np.sin(ph)
wq = 2 * np.pi * sp * np.gradient(ph)


def I_B4(f):
    ka = 2 * np.pi * A_EL * f / C
    x = np.maximum(ka * sp, 1e-9)
    B = 2 * j1(x) / x
    return (B ** 4 * wq).sum()


# Diameter-average quasi-longitudinal speed of the girdle specimen,
# computed from its realised c-axes (Sec. 3). An earlier value of
# 3807.5 m/s was inferred by dividing twice the diameter by the backwall
# ENVELOPE PEAK time without removing the 1.2/f0 = 0.6 us envelope-peak
# delay of the Ricker source, and was low by about 1 per cent.
C_MEAS = 3843.0
out = {}
for tag, cc in (("nominal 3850 m/s", 3850.0),
                ("superseded 3807.5 m/s", 3807.5),
                ("realised 3843 m/s", C_MEAS)):
    d0, d1 = GATE[0] * cc / 2, GATE[1] * cc / 2
    I_d = 1 / d0 - 1 / d1
    ssq = np.array([(S_T * f / C) ** 2 for f in fr])
    ssq *= (R_RMS ** 2 * S_V * G_ORI / (16 * np.pi)) * I_d
    ib4 = np.array([I_B4(f) for f in fr])
    ssq *= ib4
    env2 = 4.0 / T_G * (np.abs(Wc) ** 2 * ssq).sum() * dfr
    out[tag] = 10 * np.log10(env2)
    print(f"  {tag:22}: gate {d0*1e3:.1f}-{d1*1e3:.1f} mm, I_d {I_d:.3f} 1/m"
          f"  ->  {out[tag]:7.2f} dB re source")

print(f"\n  inputs: S_t {S_T*1e6:.2f} mm^2, lam(f0) {LAM*1e3:.3f} mm, "
      f"S_v {S_V:.1f} 1/m, |R|_rms {R_RMS*100:.3f} %, g {G_ORI:.3f}")
print(f"          INT B^4 dOmega at f0 = {I_B4(F0):.5f} sr")
print(f"  narrowband cross-check at f0 only:")
d0, d1 = GATE[0] * C_MEAS / 2, GATE[1] * C_MEAS / 2
ssq0 = ((S_T / LAM) ** 2 * (R_RMS ** 2 * S_V * G_ORI / (16 * np.pi))
        * I_B4(F0) * (1 / d0 - 1 / d1))
print(f"     SUM (a_f/P0)^2 at f0 = {ssq0:.4g}  -> "
      f"{10*np.log10(ssq0):.2f} dB peak-amplitude-squared sum")

print("\n" + "=" * 70)
print("SUMMARY, all levels dB re source amplitude, ppw 10, seed 11 girdle")
print("=" * 70)
bf = np.load(os.path.join(HERE, "po_src_final.npz"))["lev"]
bfn = np.load(os.path.join(HERE, "po_src_dense.npz"))["lev"]
M = np.load(os.path.join(HERE, "po_src_meas.npz"))
r = M["coda10"] / M["bang10"]
print(f"  PREDICTED, closed form, measured speed   : "
      f"{out['realised 3843 m/s']:7.2f}")
print(f"  PREDICTED, closed form, nominal speed    : "
      f"{out['nominal 3850 m/s']:7.2f}")
print(f"  PREDICTED, brute force 180 az, meas speed: "
      f"{10*np.log10((bf**2).mean()):7.2f}")
print(f"  PREDICTED, brute force 180 az, nominal   : "
      f"{10*np.log10((bfn**2).mean()):7.2f}")
print(f"  MEASURED   FDTD ppw10, mean power        : "
      f"{10*np.log10((r**2).mean()):7.2f}")
print(f"  MEASURED   FDTD ppw10, mean of dB        : "
      f"{20*np.log10(r).mean():7.2f}   <- the number in the paper")
print(f"  PREDICTED  brute force, mean of dB       : "
      f"{20*np.log10(bf).mean():7.2f}")
