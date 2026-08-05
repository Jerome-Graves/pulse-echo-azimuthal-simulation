"""Step 6: close the loop.

(a) the closed form was missing the peak-envelope factor of the f^2
    round-trip response; redo it and reconcile with the brute force
(b) facet normals in a 100 x 35 mm disc are NOT isotropic - the domain is
    oblate, so seed-difference vectors (= facet normals) lie preferentially
    in the horizontal plane, which is exactly where the beam looks
(c) spectral content of the measured coda vs the predicted coda: if they
    do not overlap, the level agreement is a coincidence
"""
import os

import numpy as np
from scipy.signal import hilbert
from scipy.special import j1

import po_src_03_kirchhoff_pred as PO
from po_src_03_kirchhoff_pred import (A_EL, C, DT, F0, GATE, LAM, NT, R_DISC, S_T, T_G,
                        THK, bess, ricker_spec)

HERE = os.path.dirname(os.path.abspath(__file__))
Wspec, _ = ricker_spec(NT, DT, F0)
G = np.load(os.path.join(HERE, "po_src_geom.npz"))
FN, FA = G["fn"], G["farea"]
V_SPEC = np.pi * R_DISC ** 2 * THK
S_V = FA.sum() / V_SPEC

print("=== (b) facet-normal orientation statistics (area weighted) ===")
nz = np.abs(FN[:, 2])
print(f"  <|n_z|> = {(nz*FA).sum()/FA.sum():.4f}   (0.500 = isotropic)")
print(f"  <n_z^2> = {(nz**2*FA).sum()/FA.sum():.4f}   (0.3333 = isotropic)")
# density of normals per steradian near the horizontal ring the beam probes
delt = np.radians(10.0)
gs = []
for az in np.arange(0, 360, 2.0):
    u = np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), 0.0])
    c = np.abs(FN @ u)
    sel = c > np.cos(delt)
    # solid angle of the two caps of half-angle delt on the unit sphere
    om = 2 * 2 * np.pi * (1 - np.cos(delt))
    gs.append(FA[sel].sum() / FA.sum() / (om / (4 * np.pi)))
gs = np.array(gs)
print(f"  area fraction of facets with normal within {np.degrees(delt):.0f} "
      f"deg of the beam axis, relative to isotropic:")
print(f"     g = {gs.mean():.3f} +/- {gs.std():.3f}  "
      f"(1.0 = isotropic) -> {10*np.log10(gs.mean()):+.2f} dB on the "
      f"backscattered power")

print("\n=== (a) closed form, corrected ===")
ph = np.linspace(1e-9, np.pi / 2, 400001)
ka = 2 * np.pi * A_EL / LAM
B = bess(ka * np.sin(ph))
I_B4 = float(np.trapezoid(B ** 4 * 2 * np.pi * np.sin(ph), ph))
d0, d1 = GATE[0] * C / 2, GATE[1] * C / 2
I_d = 1 / d0 - 1 / d1
D = np.load(os.path.join(HERE, "po_src_diag.npz"))
R_RMS = float(np.sqrt((D["rr"] ** 2 * D["ain"]).sum() / D["ain"].sum()))
fr = np.fft.rfftfreq(NT, DT)
pulse = np.fft.irfft(Wspec * (fr / F0) ** 2, NT)
ev = np.abs(hilbert(pulse))
PK = ev.max()
T_env = np.trapezoid(ev ** 2, dx=DT) / PK ** 2
ssq = (S_T / LAM) ** 2 * (R_RMS ** 2 * S_V / (16 * np.pi)) * I_B4 * I_d
print(f"  SUM (a_f/P0)^2 = (S_t/lam)^2 . R^2 S_v/(16 pi) . INT B^4 dOmega "
      f". INT dd/d^2")
print(f"                 = {(S_T/LAM)**2:.5g} x "
      f"{R_RMS**2*S_V/(16*np.pi):.5g} x {I_B4:.5g} x {I_d:.5g} = {ssq:.4g}")
print(f"  peak envelope of the f^2 round-trip pulse = {PK:.4f}, "
      f"T_env = {T_env*1e6:.3f} us, T_g = {T_G*1e6:.0f} us")
base = 10 * np.log10(ssq * PK ** 2 * T_env / T_G)
print(f"  isotropic normals      : {base:7.2f} dB re source")
print(f"  x g = {gs.mean():.3f} (oblate domain) : "
      f"{base + 10*np.log10(gs.mean()):7.2f} dB re source")
dense = np.load(os.path.join(HERE, "po_src_dense.npz"))["lev"]
print(f"  brute-force PO, 180 az : "
      f"{10*np.log10((dense**2).mean()):7.2f} dB re source")

print("\n=== (c) where the coda energy sits in frequency ===")
SWD = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "out", "sweeps")))
acc_m = None
for az in range(0, 360, 12):
    with np.load(os.path.join(SWD, "girdle_seed11_ppw10_licensing", f"az{az:03d}.npz")) as z:
        tr = np.asarray(z["trace"], float).ravel()
        dt = float(z["dt"])
    a, b = int(GATE[0] / dt), int(GATE[1] / dt)
    S = np.abs(np.fft.rfft(tr[a:b] * np.hanning(b - a), 8192)) ** 2
    acc_m = S if acc_m is None else acc_m + S
fm = np.fft.rfftfreq(8192, dt)
acc_p = None
for az in np.arange(0, 360, 12.0):
    t = PO.trace_from(PO.azimuth_response(az, LAM / 8)[0], Wspec)
    a, b = int(GATE[0] / DT), int(GATE[1] / DT)
    S = np.abs(np.fft.rfft(t[a:b] * np.hanning(b - a), 8192)) ** 2
    acc_p = S if acc_p is None else acc_p + S
fp = np.fft.rfftfreq(8192, DT)


def cent(f, S, lo=0.3e6, hi=6e6):
    m = (f > lo) & (f < hi)
    f, S = f[m], S[m]
    c = (f * S).sum() / S.sum()
    cum = np.cumsum(S) / S.sum()
    return c, f[np.searchsorted(cum, 0.1)], f[np.searchsorted(cum, 0.9)]


cm = cent(fm, acc_m)
cp = cent(fp, acc_p)
print(f"  measured coda : centroid {cm[0]/1e6:.2f} MHz, 10-90% "
      f"{cm[1]/1e6:.2f}-{cm[2]/1e6:.2f} MHz")
print(f"  predicted coda: centroid {cp[0]/1e6:.2f} MHz, 10-90% "
      f"{cp[1]/1e6:.2f}-{cp[2]/1e6:.2f} MHz")
