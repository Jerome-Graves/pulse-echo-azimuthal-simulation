"""ITERATIVE FABRIC INVERSION: fit COF distribution parameters to scan data.

This is the inversion the project is actually for. No per-grain picking:
guess the orientation-distribution parameters, predict the scan observables
with the validated fast model, compare, iterate. What comes back is the
fabric DISTRIBUTION (mean c-axis direction + concentration/eigenvalue), the
same statistics a thin section gives.

OBSERVABLES per azimuth (all from the trace, no picking):
  * coda RMS in the clean window 24-36 us  - grain-to-grain disorder seen
    by the whole beam volume (deflection + off-axis scattering included)
  * E1 amplitude                            - beam walk-off fade, encodes
    the mean-axis direction through the group-velocity skew
  * E1 time of flight                       - mean chord slowness

FORWARD MODEL, realisation-noise-free by factorisation: because grain axes
are iid, the expected rung-3 envelope factorises into a fabric-independent
GEOMETRIC envelope (boundary area x beam x spreading; computed once with
unit_contrast=True) times sqrt(E[R^2]), where E[R^2] = Var(v)/(2 v_mean^2)
is an exact expectation over the Watson ODF, computed by fixed spherical
quadrature. The optimiser therefore sees a smooth, deterministic objective.

OBJECTIVE DESIGN - three failures before the one that works (all kept in
the code comments so nobody retries them):
  1. Candidate-dependent walk-off masking + E1-amplitude fade channel:
     the optimiser silenced whichever azimuths disagreed with a wrong axis
     (recovered 150 deg), and the mean-axis fade model misfit by +/-18 dB
     because E1 crosses ~9 individual grains.
  2. Masking with a fixed penalty + concentration-scaled skew: still
     exploitable (recovered kappa -> 0 by flattening the model and
     masking the witnesses).
  3. Predicting the contamination as walk-off multipath: falsified by the
     data itself - E1 is equally walk-off-suppressed at az 31/47, yet
     only az 76 shows window excess (mechanism specific to psi ~ 76,
     likely a group caustic), so truth was punished for predicting excess
     where there is none.
  FINAL: simple physics (ODF coda + ToF), TWO parameters (alpha, kappa),
  robust Huber loss so the one unmodelled outlier cannot drag the fit and
  there is nothing candidate-dependent to exploit.

RECOVERY TEST RESULT (2026-07-24): measurement = the six saved full-wave
sweep traces (ref/fw_az*.npz); truth alpha = 0 (axis headless, so
170.6 == -9.4), kappa = 3.93, a1 = 0.688. Recovered:
    alpha 170.6 deg (9.4 deg from truth), kappa 4.63, a1 = 0.744
    chi2 24.3 vs 25.0 at truth - statistically indistinguishable.
az 76 sits isolated at +13 dB in the Huber tail; the five clean azimuths
carry the fit. Six azimuths of one realisation -> ~9 deg axis error and
a1 to 0.06; a full 180-angle scan tightens both substantially.
"""
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
from scipy.optimize import minimize            # noqa: E402
from scipy.signal import hilbert               # noqa: E402

import born                                    # noqa: E402
import forward as F                            # noqa: E402
import ladder                                  # noqa: E402
import specimen as S                           # noqa: E402

REF = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ref")
W0, W1 = ladder.W_CLEAN
C_REF = 3850.0
D = 0.100
ELEMENT = 6.35e-3

# measurement noise scales (dB / dB / us), from the validation campaign.
# SIG_AMP is huge = the E1-amplitude channel is DIAGNOSTIC ONLY for now:
# first fit attempt showed +/-18 dB residuals because E1 crosses ~9
# individual grains and the mean-axis skew proxy mispredicts the fade at
# mid angles; a grain-resolved walk-off model is needed before this
# channel can constrain the fit. SIG_TOF 0.5 us covers per-chord
# realisation scatter plus pick error on walk-off-weakened E1 peaks.
SIG_CODA, SIG_AMP, SIG_TOF = 2.5, 1e9, 0.8

# a downweighted coda point still costs this many sigma^2: without the
# penalty the optimiser EXPLOITS the candidate-dependent walk-off mask,
# silencing whichever azimuths disagree with a wrong axis (measured: it
# parked the axis at 150 deg and masked 4 of 6 points).
MASK_PENALTY = 4.0


# ── ODF machinery ────────────────────────────────────────────────────────
def fib_sphere(n=1500):
    """Deterministic quasi-uniform directions on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    th = np.arccos(np.clip(z, -1, 1))
    ph = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), z]


_DIRS = fib_sphere()
_DPSI = np.gradient(F._VQP, F._PSI)


def odf_weights(axis, kappa):
    # subtract the max exponent before exp: identical after the
    # normalisation, and immune to overflow when an unbounded
    # optimiser wanders kappa large (seen on an 8-azimuth 5 MHz fit)
    ex = kappa * (_DIRS @ axis) ** 2
    w = np.exp(ex - ex.max())
    return w / w.sum()


def odf_moments(axis, kappa, beam):
    """E[R^2] and E[1/v] for grains ~ Watson(axis, kappa), beam direction."""
    w = odf_weights(axis, kappa)
    cos = np.clip(np.abs(_DIRS @ beam), 0, 1)
    v = np.interp(np.arccos(cos), F._PSI, F._VQP)
    vm = np.sum(w * v)
    var = np.sum(w * (v - vm) ** 2)
    return var / (2.0 * vm * vm), np.sum(w / v)


def skew_deg(axis, kappa, beam):
    """Systematic beam walk-off angle (deg) for the candidate fabric.

    The mean-axis skew only steers the beam SYSTEMATICALLY to the extent
    the grains agree: at kappa=0 individual-grain skews point every which
    way and cancel, so the proxy is scaled by the fabric order parameter
    (a1 - 1/3)/(2/3), 0 for isotropic, 1 for a delta fabric. Without this
    scaling an isotropic candidate kept full walk-off masking rights and
    the optimiser exploited it (recovered kappa -> 0, axis 115 deg).
    """
    psi = np.arccos(np.clip(np.abs(np.dot(axis, beam)), 0, 1))
    dv = np.interp(psi, F._PSI, _DPSI)
    v = np.interp(psi, F._PSI, F._VQP)
    order = np.clip((S.watson_eigenvalue(kappa) - 1.0 / 3.0) / (2.0 / 3.0),
                    0.0, 1.0)
    return order * np.degrees(np.arctan(np.abs(dv) / v))


# ── data extraction ──────────────────────────────────────────────────────
def load_sweep():
    """Per-azimuth (coda_rms_abs, E1_amp, E1_tof_us) from the saved traces."""
    rows = []
    for rot in (17, 31, 47, 62, 76, 90):
        d = np.load(os.path.join(REF, f"fw_az{rot:02d}.npz"))
        tr, dt = d["trace"].ravel(), float(d["dt"])
        fs = 1.0 / dt
        e = np.abs(hilbert(tr))
        t1 = 2 * D / C_REF
        k0 = int(t1 * fs)
        a0 = max(k0 - int(2e-6 * fs), 0)
        seg = e[a0:k0 + int(2e-6 * fs)]
        E1 = seg.max()
        # LEADING-EDGE ToF pick (25% of E1, sub-sample interpolated),
        # minus the Ricker envelope-peak source delay 1.2/f0. The old
        # envelope-PEAK pick carried the 0.6 us wavelet delay PLUS a
        # scattering-correlated peak drift (up to ~0.8 us at mid-psi) -
        # a >1% velocity bias vs the offset-free tof_p comparison in
        # objective(); fit_sweep learned this on 2026-07-28, backported
        # here 2026-07-29.
        kp = int(np.argmax(seg))
        thr = 0.25 * E1
        j = kp
        while j > 0 and seg[j] > thr:
            j -= 1
        frac = ((thr - seg[j]) / max(seg[j + 1] - seg[j], 1e-30)
                if seg[j + 1] > seg[j] else 0.0)
        tof = (a0 + j + frac) / fs - 1.2 / 2.0e6
        coda = np.sqrt((e[int(W0 * fs):int(W1 * fs)] ** 2).mean())
        rows.append(dict(rot=rot, coda=coda, e1=E1, tof_us=tof * 1e6))
    return rows


# ── forward prediction ───────────────────────────────────────────────────
def geometric_envelopes(build, cfg, rots):
    """Fabric-independent geometric coda RMS per azimuth (computed once)."""
    dt = ladder.DT
    lo, hi = int(W0 / dt), int(W1 / dt)
    out = []
    for rot in rots:
        _, env = born.boundary_scatter(dict(build), cfg,
                                       azimuth_rad=np.radians(rot),
                                       unit_contrast=True)
        out.append(np.sqrt((np.asarray(env, float)[lo:hi] ** 2).mean()))
    return np.array(out)


def predict_basis(alpha_deg, kappa, rots, geo):
    """Shape vectors for one candidate: grain-scatter coda, walk-off
    multipath coda, and ToF. Gains are applied by the caller."""
    axis = np.array([np.cos(np.radians(alpha_deg)),
                     np.sin(np.radians(alpha_deg)), 0.0])
    c_odf, m_walk, tof, skews = [], [], [], []
    for rot, g in zip(rots, geo):
        beam = -np.array([np.cos(np.radians(rot)),
                          np.sin(np.radians(rot)), 0.0])
        er2, inv_v = odf_moments(axis, kappa, beam)
        c_odf.append(g * np.sqrt(er2))
        sk = skew_deg(axis, kappa, beam)
        skews.append(sk)
        walk = 2 * D * np.tan(np.radians(sk))
        # multipath strength ~ the E1 energy the walk-off removed from the
        # retroreflection path; it rattles on and lands in the coda window
        m_walk.append(1.0 - np.exp(-(walk / ELEMENT) ** 2))
        tof.append(2 * D * inv_v * 1e6)
    return (np.array(c_odf), np.array(m_walk), np.array(tof),
            np.array(skews))


def huber(x, k=2.0):
    """Quadratic core, linear tails: one unmodelled outlier cannot drag
    the fit, and unlike candidate-dependent masking there is nothing for
    the optimiser to exploit (that design failed twice; and predicting the
    excess as walk-off multipath failed too, because E1 is equally
    walk-off-suppressed at az 31/47 yet only az 76 shows window excess -
    the mechanism is specific to psi ~ 76, likely a group caustic, and is
    left to the robust loss rather than hand-crafted)."""
    a = np.abs(x)
    return np.where(a <= k, x * x, 2 * k * a - k * k)


def objective(alpha_deg, kappa, data, geo, detail=False):
    rots = [d["rot"] for d in data]
    c_odf, m_walk, tof_p, skews = predict_basis(alpha_deg, kappa, rots, geo)
    coda_m = np.array([d["coda"] for d in data])
    tof_m = np.array([d["tof_us"] for d in data])
    r0 = 20 * (np.log10(coda_m) - np.log10(c_odf + 1e-30))
    # robust gain: minimise the Huber cost over the single offset
    from scipy.optimize import minimize_scalar
    gfit = minimize_scalar(
        lambda g: np.sum(huber((r0 - g) / SIG_CODA)),
        bracket=(r0.min(), np.median(r0), r0.max()))
    r_c = r0 - gfit.x
    # free common ToF offset (robust), mirroring the coda gain: any
    # residual pick/calibration bias is a nuisance, the azimuthal
    # SHAPE carries the fabric (2026-07-29, with the leading-edge pick)
    rt0 = tof_m - tof_p
    tfit = minimize_scalar(
        lambda o: np.sum(huber((rt0 - o) / SIG_TOF)),
        bracket=(rt0.min(), np.median(rt0), rt0.max()))
    r_t = rt0 - tfit.x
    chi2 = (np.sum(huber(r_c / SIG_CODA))
            + np.sum(huber(r_t / SIG_TOF)))
    if not detail:
        return chi2
    return chi2, dict(coda_res=r_c, tof_res=r_t, skews=skews)


def main():
    print("loading measurement (6 full-wave sweep traces)...")
    data = load_sweep()
    build = ladder.standard_build()
    cfg = ladder.standard_cfg()
    rots = [d["rot"] for d in data]
    print("computing geometric envelopes (once)...", flush=True)
    geo = geometric_envelopes(build, cfg, rots)

    print(f"objective at TRUTH (alpha 0, kappa 3.93): "
          f"{objective(0.0, 3.93, data, geo):.1f}", flush=True)

    print("grid search...", flush=True)
    best = None
    for a in range(0, 180, 5):
        for k in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
            c = objective(a, k, data, geo)
            if best is None or c < best[0]:
                best = (c, a, k)
    print(f"   grid best: alpha {best[1]} deg, kappa {best[2]}, "
          f"chi2 {best[0]:.1f}")

    res = minimize(lambda p: objective(p[0], np.exp(p[1]), data, geo),
                   x0=[best[1], np.log(best[2])], method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-4, maxiter=2000))
    a_fit = res.x[0] % 180.0
    k_fit = float(np.exp(res.x[1]))
    a1 = S.watson_eigenvalue(k_fit)
    chi2, det = objective(res.x[0], k_fit, data, geo, detail=True)

    print("\n=== RECOVERED FABRIC ===")
    print(f"  mean c-axis azimuth : {a_fit:6.1f} deg   (truth   0.0)")
    print(f"  Watson kappa        : {k_fit:6.2f}       (truth  3.93)")
    print(f"  eigenvalue a1       : {a1:6.3f}      (truth 0.688)")
    print(f"  chi2 = {chi2:.1f} over {2*len(data)} points, 2 parameters")
    print("\nper-azimuth residuals (coda dB / ToF us / candidate skew):")
    for d, rc, rt, sk in zip(data, det["coda_res"], det["tof_res"],
                             det["skews"]):
        print(f"  az {d['rot']:>3}: {rc:+6.1f}  {rt:+6.2f}   "
              f"skew {sk:4.1f} deg")


if __name__ == "__main__":
    main()
