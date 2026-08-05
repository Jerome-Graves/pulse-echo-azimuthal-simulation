"""Step 3: PHYSICAL-OPTICS backscatter anchor, referenced to the SOURCE
amplitude.  CPU / numpy only.

MODEL (all of it, no fitted constants)
--------------------------------------
Let P0(t) be the pressure on the element face; the recorded 'bang' (peak
of the envelope of the excitation as recorded) is taken to BE P0's peak.

  transmit, far field      p_i(x) = P0 S_t B(phi,f) e^{ikd}/(lam d)
  Kirchhoff / physical optics backscatter from a surface S, evaluated at
  the element and weighted by the receive directivity (reciprocity):

     p_s = -i (P0 S_t / lam^2) INT_S R(x) B(phi)^2 (n.u) e^{2ikd}/d^2 dS

  which reproduces the exact image-source result for an infinite plane
  mirror (INT -> lam/2L, giving p = P0 S_t R/(lam 2L)), so the constant
  is fixed, not chosen.  R(x) is the normal-incidence pressure reflection
  coefficient of the grain boundary, (v_far - v_near)/(v_far + v_near),
  with v the qP speed of each grain ALONG the line of sight (density is
  uniform), and the sign of (n.u) carries the near/far ordering, so
  R*(n.u) = (n.u)*(v_i - v_j)/(v_i + v_j) with n = (s_i - s_j)/|s_i - s_j|.

  The e^{2ikd} phase is NOT applied explicitly: every surface element is
  deposited into a delay histogram at tau = 2d/c and the whole synthetic
  A-scan is formed by one FFT, so the facet's specular lobe, its
  sidelobes, the Fresnel-zone limit, the beam footprint and the finite
  bandwidth all emerge from the same integral instead of being modelled
  by separate ad-hoc 'effective aperture' factors.

  Facets are then summed with their true relative delays, i.e. the sum is
  incoherent between facets separated by more than a pulse length and
  coherent within one - again automatic, not assumed.
"""
import os
import sys
import time

import numpy as np
from scipy.signal import hilbert
from scipy.special import j1

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vendor"))))
import forward as FW                                     # noqa: E402  (no CUDA)

HERE = os.path.dirname(os.path.abspath(__file__))
C = 3850.0
F0 = 2.0e6
LAM = C / F0
DIA, THK = 0.100, 0.035
R_DISC = DIA / 2
GATE = (24e-6, 36e-6)
T_G = GATE[1] - GATE[0]
RHO_F, C_F = 300.0, 1500.0
RHO_I = 917.0

PPW = 10.0
H = C / F0 / PPW
ER = max(int(6.35e-3 / 2 / H), 1)
_dy, _dz = np.meshgrid(np.arange(-ER, ER + 1), np.arange(-ER, ER + 1))
N_EL = int(((_dy ** 2 + _dz ** 2) <= ER ** 2).sum())
S_T = N_EL * H ** 2
A_EL = np.sqrt(S_T / np.pi)

NT, DT = 8192, 8.29e-9
NS_BIN, S_MAX = 64, 0.40


def bess(x):
    x = np.asarray(x, float)
    out = np.ones_like(x)
    m = x > 1e-9
    out[m] = 2 * j1(x[m]) / x[m]
    return out


def ricker_spec(nt, dt, f0):
    t = np.fft.fftfreq(nt, d=1.0 / (nt * dt))
    a = (np.pi * f0 * t) ** 2
    w = (1 - 2 * a) * np.exp(-a)
    return np.fft.rfft(w), w


G = np.load(os.path.join(HERE, "po_src_geom.npz"))
SEEDS, AXES = G["seeds"], G["axes"]
FI, FJ, FN, FU, FV, FX0 = G["fi"], G["fj"], G["fn"], G["fu"], G["fv"], G["fx0"]
FAREA, FCEN, NPOLY = G["farea"], G["fcen"], G["npoly"]
POFF = np.concatenate([[0], np.cumsum(NPOLY)])
POLYS = G["polys"]
NF = len(FI)


def poly_of(k):
    return POLYS[POFF[k]:POFF[k + 1]]


def inside(poly, P):
    """point-in-convex-polygon, vectorised over P (n,2)."""
    n = len(poly)
    s = np.ones(len(P), bool)
    for a in range(n):
        e = poly[(a + 1) % n] - poly[a]
        cr = e[0] * (P[:, 1] - poly[a][1]) - e[1] * (P[:, 0] - poly[a][0])
        s &= cr >= -1e-15 if _ccw(poly) else cr <= 1e-15
    return s


def _ccw(poly):
    x, y = poly[:, 0], poly[:, 1]
    return (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() > 0


def quad_points(k, ds):
    """Regular lattice quadrature of facet k: 3-D points and cell area."""
    p = poly_of(k)
    lo, hi = p.min(0), p.max(0)
    na = max(int(np.ceil((hi[0] - lo[0]) / ds)), 1)
    nb = max(int(np.ceil((hi[1] - lo[1]) / ds)), 1)
    da, db = (hi[0] - lo[0]) / na, (hi[1] - lo[1]) / nb
    aa = lo[0] + (np.arange(na) + 0.5) * da
    bb = lo[1] + (np.arange(nb) + 0.5) * db
    A, B = np.meshgrid(aa, bb, indexing="ij")
    P2 = np.stack([A.ravel(), B.ravel()], 1)
    m = inside(p, P2)
    P2 = P2[m]
    if len(P2) == 0:
        return None, 0.0
    X = FX0[k][None, :] + P2[:, 0:1] * FU[k][None, :] + P2[:, 1:2] * FV[k][None, :]
    return X, da * db


def azimuth_response(az_deg, ds, want_diag=False):
    """Delay/angle histogram of the PO integrand for one probe azimuth."""
    a = np.radians(az_deg)
    nb = np.array([np.cos(a), np.sin(a), 0.0])
    p0 = R_DISC * nb
    axis = -nb
    Gh = np.zeros((NT, NS_BIN))
    diag = []
    dlo, dhi = (GATE[0] - 2e-6) * C / 2, (GATE[1] + 2e-6) * C / 2
    for k in range(NF):
        rc = np.linalg.norm(poly_of(k)[:, 0:1] * FU[k][None, :]
                            + poly_of(k)[:, 1:2] * FV[k][None, :]
                            + FX0[k][None, :] - FCEN[k][None, :], axis=1).max()
        vc = FCEN[k] - p0
        dc = np.linalg.norm(vc)
        if dc + rc < dlo or dc - rc > dhi:
            continue
        cphi_c = (vc / dc) @ axis
        ang = np.arccos(np.clip(cphi_c, -1, 1)) - np.arcsin(min(1.0, rc / dc))
        if ang > np.arcsin(S_MAX):
            continue
        X, dA = quad_points(k, ds)
        if X is None:
            continue
        V = X - p0[None, :]
        d = np.linalg.norm(V, axis=1)
        U = V / d[:, None]
        cphi = U @ axis
        sphi = np.sqrt(np.clip(1 - cphi ** 2, 0, 1))
        cth = U @ FN[k]                      # signed: carries near/far order
        vi = np.interp(np.arccos(np.clip(np.abs(U @ AXES[FI[k]]), 0, 1)),
                       FW._PSI, FW._VQP)
        vj = np.interp(np.arccos(np.clip(np.abs(U @ AXES[FJ[k]]), 0, 1)),
                       FW._PSI, FW._VQP)
        Rg = (vi - vj) / (vi + vj)
        tau = 2 * d / C
        m = ((tau > GATE[0] - 2e-6) & (tau < GATE[1] + 2e-6)
             & (sphi < S_MAX) & (cphi > 0))
        if not m.any():
            continue
        wgt = (Rg * cth * dA / d ** 2)[m]
        ti = tau[m] / DT
        i0 = np.floor(ti).astype(int)
        fr = ti - i0
        si = np.clip((sphi[m] / S_MAX * NS_BIN).astype(int), 0, NS_BIN - 1)
        np.add.at(Gh, (i0, si), wgt * (1 - fr))
        np.add.at(Gh, (i0 + 1, si), wgt * fr)
        if want_diag:
            g = m & (tau > GATE[0]) & (tau < GATE[1])
            if g.any():
                b6 = bess(2 * np.pi * A_EL / LAM * sphi[g]) ** 2 >= 0.5
                diag.append((k, dA * g.sum(), dA * int(b6.sum()),
                             float(np.sqrt((Rg[g] ** 2).mean())),
                             float(np.abs(cth[g]).mean()), float(d[g].mean())))
    return Gh, diag


def trace_from(Gh, Wspec):
    fr = np.fft.rfftfreq(NT, DT)
    lam = C / np.maximum(fr, 1e-9)
    s_mid = (np.arange(NS_BIN) + 0.5) / NS_BIN * S_MAX
    Hs = np.fft.rfft(Gh, axis=0)                 # (nf, ns)
    ka = 2 * np.pi * A_EL / lam
    B2 = bess(ka[:, None] * s_mid[None, :]) ** 2
    Hf = (Hs * B2).sum(1) * (S_T / lam ** 2)
    return np.fft.irfft(Hf * Wspec, NT)


def env_rms_gate(tr):
    e = np.abs(hilbert(tr))
    a, b = int(GATE[0] / DT), int(GATE[1] / DT)
    return np.sqrt((e[a:b] ** 2).mean())


# ─────────────────────────────── run ────────────────────────────────────
if __name__ == "__main__":
    Wspec, wt = ricker_spec(NT, DT, F0)
    AZ = list(range(0, 360, 12))
    print("PHYSICAL-OPTICS ANCHOR, source-referenced")
    print(f"  element: ppw {PPW:.0f}, h {H*1e3:.4f} mm, radius {ER}h = "
          f"{A_EL*1e3:.3f} mm equiv ({N_EL} source points), "
          f"S_t = {S_T*1e6:.2f} mm^2")
    print(f"  lambda {LAM*1e3:.3f} mm, ka {2*np.pi*A_EL/LAM:.2f}, "
          f"far-field (Rayleigh) distance S_t/lam = {S_T/LAM*1e3:.2f} mm")
    print(f"  gate {GATE[0]*1e6:.0f}-{GATE[1]*1e6:.0f} us = "
          f"{GATE[0]*C/2*1e3:.1f}-{GATE[1]*C/2*1e3:.1f} mm range\n")

    for ds in (LAM / 8, LAM / 16, LAM / 24):
        t0 = time.time()
        lev, diags = [], []
        for az in AZ:
            Gh, dg = azimuth_response(az, ds, want_diag=(ds == LAM / 16))
            lev.append(env_rms_gate(trace_from(Gh, Wspec)))
            diags.append(dg)
        lev = np.array(lev)
        m = 20 * np.log10(lev)
        print(f"  quadrature ds = lam/{LAM/ds:.0f} = {ds*1e3:.3f} mm : "
              f"mean(dB) {m.mean():7.2f}   dB(mean power) "
              f"{10*np.log10((lev**2).mean()):7.2f}   "
              f"spread {m.min():.1f}..{m.max():.1f}  ({time.time()-t0:.0f} s)")
        if ds == LAM / 16:
            np.savez(os.path.join(HERE, "po_src_pred.npz"),
                     az=np.array(AZ), lev=lev)
            DG = diags

    # ---- diagnostics at the reference quadrature
    print("\nGEOMETRY ACTUALLY INSONIFIED (per azimuth, in the gate)")
    nfac = np.array([len(d) for d in DG])
    n6 = np.array([sum(1 for r in d if r[2] > 0) for d in DG])
    ain = np.concatenate([[r[1] for r in d] for d in DG])
    a6 = np.concatenate([[r[2] for r in d] for d in DG])
    rr = np.concatenate([[r[3] for r in d] for d in DG])
    aw = np.concatenate([[r[1] for r in d] for d in DG])
    ct = np.concatenate([[r[4] for r in d] for d in DG])
    print(f"  boundaries with any area in the gate      : "
          f"{nfac.mean():.1f} per azimuth (range {nfac.min()}-{nfac.max()})")
    print(f"  boundaries inside the -6 dB beam AND gate  : "
          f"{n6.mean():.1f} per azimuth (range {n6.min()}-{n6.max()})")
    print(f"  facet area intercepted, in gate           : mean "
          f"{ain.mean()*1e6:.1f} mm^2, total per azimuth "
          f"{ain.sum()/len(AZ)*1e6:.0f} mm^2")
    print(f"  facet area intercepted, gate & -6 dB beam : mean "
          f"{a6[a6>0].mean()*1e6:.1f} mm^2, total per azimuth "
          f"{a6.sum()/len(AZ)*1e6:.0f} mm^2")
    print(f"  |R| rms over insonified boundaries        : "
          f"{np.sqrt((rr**2*aw).sum()/aw.sum())*100:.4f} %  "
          f"(unweighted {np.sqrt((rr**2).mean())*100:.4f} %)")
    print(f"  mean |n.u| (obliquity) of those facets    : {ct.mean():.3f}")

    # ---- like-for-like against the measurement
    M = np.load(os.path.join(HERE, "po_src_meas.npz"))
    r = M["coda10"] / M["bang10"]
    P = np.load(os.path.join(HERE, "po_src_pred.npz"))["lev"]
    print("\nCOMPARISON (30 azimuths, source-referenced, ppw 10)")
    print(f"{'':22}{'mean of dB':>12}{'dB of mean power':>18}{'az spread':>12}")
    print(f"{'  measured (FDTD)':22}{20*np.log10(r).mean():>12.2f}"
          f"{10*np.log10((r**2).mean()):>18.2f}"
          f"{20*np.log10(r).max()-20*np.log10(r).min():>11.1f}")
    print(f"{'  physical optics':22}{20*np.log10(P).mean():>12.2f}"
          f"{10*np.log10((P**2).mean()):>18.2f}"
          f"{20*np.log10(P).max()-20*np.log10(P).min():>11.1f}")
    print(f"{'  measured - PO':22}{20*np.log10(r).mean()-20*np.log10(P).mean():>+12.2f}"
          f"{10*np.log10((r**2).mean())-10*np.log10((P**2).mean()):>+18.2f}")
    np.savez(os.path.join(HERE, "po_src_diag.npz"),
             nfac=nfac, n6=n6, ain=ain, a6=a6, rr=rr, ct=ct)
