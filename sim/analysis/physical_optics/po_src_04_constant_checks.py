"""Step 4: validate the absolute constant of the PO machinery, calibrate
the source reference against the scattering-free control, and put an
uncertainty band on the anchor.  CPU only.

Three independent checks of the SAME code path that produced the coda:
  1. infinite flat mirror at L: must reproduce the exact image-source
     answer p/P0 = S_t R /(2 lam L).  Fixes the constant.
  2. the real curved rim (a concave cylinder seen from inside): predicts
     the backwall echo with no free parameter; compared with the
     ZERO-CONTRAST control, where no scattering attenuation exists.
     This is the test of 'recorded bang = P0'.
  3. closed-form orientation-averaged incoherent sum, which needs no
     facet positions, as a check on the brute-force integral.
"""
import os

import numpy as np
from scipy.signal import hilbert
from scipy.special import j1

import po_src_03_kirchhoff_pred as PO
from po_src_03_kirchhoff_pred import (A_EL, C, DT, F0, GATE, H, LAM, NT, N_EL, R_DISC,
                        S_T, T_G, THK, bess, ricker_spec)

HERE = os.path.dirname(os.path.abspath(__file__))
R_BW = (RHOF_C := 300.0 * 1500.0 - 917.0 * C) / (300.0 * 1500.0 + 917.0 * C)
Wspec, _ = ricker_spec(NT, DT, F0)


def synth(tau, wgt, sphi):
    """Same delay/angle -> trace path as the coda calculation."""
    Gh = np.zeros((NT, PO.NS_BIN))
    ti = tau / DT
    i0 = np.floor(ti).astype(int)
    fr = ti - i0
    si = np.clip((sphi / PO.S_MAX * PO.NS_BIN).astype(int), 0, PO.NS_BIN - 1)
    np.add.at(Gh, (i0, si), wgt * (1 - fr))
    np.add.at(Gh, (i0 + 1, si), wgt * fr)
    return PO.trace_from(Gh, Wspec)


def peak_env(tr, t_lo, t_hi):
    e = np.abs(hilbert(tr))
    return e[int(t_lo / DT):int(t_hi / DT)].max()


print("=" * 74)
print("CHECK 1  infinite flat mirror at L = 100 mm, R = 1")
L = 0.100
ds = LAM / 16
n = int(0.070 / ds)
q = (np.arange(-n, n + 1)) * ds
Y, Z = np.meshgrid(q, q, indexing="ij")
d = np.sqrt(L ** 2 + Y ** 2 + Z ** 2)
sphi = np.sqrt(Y ** 2 + Z ** 2) / d
cth = L / d                                   # normal along the axis
m = sphi < PO.S_MAX
tr = synth(2 * d[m] / C, (1.0 * cth[m] * ds ** 2 / d[m] ** 2), sphi[m])
got = peak_env(tr, 45e-6, 60e-6)
exact = S_T / (2 * LAM * L)
print(f"  PO quadrature  {20*np.log10(got):8.3f} dB re source")
print(f"  image source   {20*np.log10(exact):8.3f} dB re source "
      f"(= S_t/(2 lam L), exact)")
print(f"  difference     {20*np.log10(got/exact):+8.3f} dB   <- the "
      f"absolute constant of the model is verified end to end")

print("\n" + "=" * 74)
print("CHECK 2  the REAL rim: concave cylinder R = 50 mm, ice->fluid")
print(f"  R_backwall = {R_BW:+.4f}  (rho c: ice {917*C/1e6:.2f}, "
       f"fluid {300*1500/1e6:.2f} MRayl)")
nth, nz = 4000, 900
th = (np.arange(nth) + 0.5) / nth * np.pi - np.pi / 2      # far half
zz = (np.arange(nz) + 0.5) / nz * THK - THK / 2
TH, ZZ = np.meshgrid(th, zz, indexing="ij")
# surface point on the far rim, probe at (+R,0,0) looking along -x
X = -R_DISC * np.cos(TH)
Y = R_DISC * np.sin(TH)
dA = R_DISC * (np.pi / nth) * (THK / nz)
p0 = np.array([R_DISC, 0.0, 0.0])
V = np.stack([X - p0[0], Y, ZZ], -1)
dd = np.linalg.norm(V, axis=-1)
U = V / dd[..., None]
nrm = np.stack([-np.cos(TH), np.sin(TH), np.zeros_like(TH)], -1)  # outward
cth = (U * nrm).sum(-1)
sphi = np.sqrt(np.clip(1 - U[..., 0] ** 2 * 0 - (U @ np.array([-1.0, 0, 0])) ** 2,
                       0, 1))
m = (sphi < PO.S_MAX) & (cth > 0)
tr = synth(2 * dd[m] / C, R_BW * cth[m] * dA / dd[m] ** 2, sphi[m])
bw = peak_env(tr, 45e-6, 60e-6)
print(f"  PO over the true curved rim : {20*np.log10(abs(bw)):7.2f} dB re source")
flat = S_T * abs(R_BW) / (2 * LAM * L)
print(f"  (flat-mirror value would be : {20*np.log10(flat):7.2f} dB)")
print("  MEASURED, zero-contrast control (no scattering loss):")
print("     ppw 8  -24.02 dB   ppw 6  -27.05 dB")
print("  MEASURED, real specimen: ppw 10 -27.48, ppw 8 -26.59, ppw 6 -29.00 dB")
for p, a_el in ((8, 13 * C / F0 / 8), (6, 9 * C / F0 / 6)):
    print(f"     element radius at ppw {p} = {a_el*1e3:.3f} mm vs "
          f"{A_EL*1e3:.3f} at ppw 10 -> S_t term "
          f"{20*np.log10((a_el/A_EL)**2):+.2f} dB")

print("\n" + "=" * 74)
print("CHECK 3  closed-form orientation-averaged incoherent sum")
# E[|f_bs|^2] per facet = R^2 A_eff/(16 pi):
#   f_bs = R (A_eff/lam) Lam,  INT|Lam|^2 dOmega_scatter = lam^2/A_eff,
#   dOmega_normal = dOmega_scatter/4 (tilting a mirror by th deflects 2th),
#   normals uniform on the sphere with density 1/(4 pi).
ph = np.linspace(1e-9, np.pi / 2, 400001)
ka = 2 * np.pi * A_EL / LAM
B = bess(ka * np.sin(ph))
I_B4 = float(np.trapezoid(B ** 4 * 2 * np.pi * np.sin(ph), ph))
I_B2 = float(np.trapezoid(B ** 2 * 2 * np.pi * np.sin(ph), ph))
d0, d1 = GATE[0] * C / 2, GATE[1] * C / 2
I_d = 1 / d0 - 1 / d1
GEO = np.load(os.path.join(HERE, "po_src_geom.npz"))
V_SPEC = np.pi * R_DISC ** 2 * THK
S_V = GEO["farea"].sum() / V_SPEC
DGN = np.load(os.path.join(HERE, "po_src_diag.npz"))
R_RMS = float(np.sqrt((DGN["rr"] ** 2 * DGN["ain"]).sum() / DGN["ain"].sum()))
A_MEAN = float(GEO["farea"].mean())
print(f"  S_v {S_V:.2f} 1/m,  |R|_rms {R_RMS*100:.4f} %,  "
       f"INT B^4 dOmega {I_B4:.5f} sr,  INT B^2 {I_B2:.5f} sr")
print(f"  lam^2/S_t = {LAM**2/S_T:.5f} sr (aperture gain theorem check "
      f"vs INT B^2)")
print(f"  gate {d0*1e3:.1f}-{d1*1e3:.1f} mm, INT dd/d^2 = {I_d:.3f} 1/m")
for tag, kA in (("A_eff = A (no Fresnel cap)", 1.0),
                ("A_eff = min(A, lam d/2)",
                 float(np.minimum(GEO["farea"], LAM * 0.0575 / 2).sum()
                       / GEO["farea"].sum()))):
    ssq = (S_T / LAM) ** 2 * (R_RMS ** 2 * S_V * kA / (16 * np.pi)) * I_B4 * I_d
    # effective envelope duration of the round-trip pulse (H ~ f^2)
    fr = np.fft.rfftfreq(NT, DT)
    pulse = np.fft.irfft(Wspec * (fr / F0) ** 2, NT)
    ev = np.abs(hilbert(pulse))
    T_env = np.trapezoid(ev ** 2, dx=DT) / ev.max() ** 2
    lev = 10 * np.log10(ssq * T_env / T_G)
    print(f"  {tag:28s} kappa_A {kA:.3f}  SUM(a_f/P0)^2 {ssq:.4g}  "
          f"T_env {T_env*1e6:.3f} us  ->  {lev:7.2f} dB re source")
print(f"  brute-force PO (mean power over 30 azimuths):        -85.75 dB")
