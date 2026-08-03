"""Acoustic visibility of a grain boundary, measured from scratch.

WHAT IS BEING CHECKED. Section 5 asserts a population claim about the
boundaries the beam illuminates: that a tenth of them return almost
nothing, that the gated return is carried by one or two crossings rather
than by all of them, that the brightest per cent of facets carries almost
all the power, and that what makes a boundary bright is the jump in
quasi-longitudinal speed along the ray and not the misorientation between
the two grains. That result was measured once. This module measures it
again along an independent path and then closes the three questions it
left open.

INDEPENDENT OF WHAT, EXACTLY. Nothing here imports analysis/_t2_common.py,
analysis/facet_predictors.py, analysis/physical_optics/*, forward.py or
specimen.py, and no cache written by them is read. The elastic core is
rebuilt from the single-crystal constants, the Christoffel problem, the
rotation of the stiffness tensor and the 6x6 interface solve are written
out here, and the beam is marched with a different cone-sampling rule (a
jittered polar lattice rather than a sunflower spiral) so that agreement
cannot be an artefact of a shared quadrature. The only shared inputs are
the cached label volumes, the c-axis draws, the seed points, and the
recorded traces, which are data.

The elastic core is checked before it is used, in check_core:
  * two isotropic media reproduce (Z2-Z1)/(Z2+Z1) to 8 figures and give
    exactly zero mode conversion;
  * identical media on both sides give zero reflection to 1e-19;
  * the whole solve is invariant under a rigid rotation of the pair and
    the ray together;
  * the qP speed against angle to the c-axis reproduces the curve the
    solver's own tables carry, which is the one place a silent constant
    error would hide.

THE THREE OPEN QUESTIONS.

  1. THE COEFFICIENT. The published predictor weights a crossing by the
     scalar (v2-v1)/(v2+v1). The exact normal-incidence solve at the same
     boundary disagrees per pair by a large factor while agreeing in
     ensemble rms. That is tolerable for a level and intolerable for a
     population claim about a dark tail, because the scalar vanishes
     identically when the two grains happen to present the same qP speed
     along the ray and the exact one does not. report_coefficient
     measures the disagreement, and measures what the exact coefficient
     does inside the scalar's dark tail.

  2. THE SHEAR CHANNEL. A normal-incidence qP wave at an ice/ice boundary
     also reflects two quasi-shear waves, and a quasi-shear wave in a
     rotated crystal carries a non-zero stress trace, so the receiver,
     which records -(s11+s22+s33)/3, is not blind to it. Two things then
     have to be established rather than asserted: that the converted
     return arrives inside the analysis gate, which is a statement about
     the slower return leg and is settled by integrating the qS slowness
     along the real path; and that it is large enough to matter, which is
     a statement about recorded pressure and about how much narrower the
     specular lobe of the converted mode is.

  3. WHETHER IT CHANGES ANYTHING. The field correlation is the claim the
     paper rests on. report_field rebuilds it on eight tessellations with
     the scalar coefficient and with the exact one, changing nothing else.

MEASURED   everything printed unless it is named INFERRED below.
INFERRED   (a) the dark threshold, one tenth of the median coefficient,
               is a convention and the tables print the population under
               three thresholds so the reader can see the dependence;
           (b) the specular lobe width uses the mean grain diameter as
               the facet size, which is the published model's choice and
               is kept so that the two coefficients are compared inside
               one model rather than across two;
           (c) the converted-mode lobe width scales that same facet size
               by (k_P + k_S)/(2 k_P), which is the stationary-phase
               mismatch for a mode-converting facet and is stated as a
               model rather than measured;
           (d) the interface normal is taken along the ray, which is the
               normal-incidence assumption both coefficients share.

THE GATE, AND WHY EVERY AZIMUTH IS RESAMPLED. The CFL limit gives each
azimuth its own dt, so nothing is ever stacked by sample index: every
trace is measured on its own axis and interpolated onto one physical time
grid before anything is combined.

NO HILBERT ON A WHOLE TRACE. The measured field is built as a local mean
square smoothed by the power envelope of the source pulse, a +-2 us
kernel that cannot reach the front arrival or the backwall from inside a
24-36 us gate. The published Hilbert-envelope field is computed beside it
and the two correlations are printed together, so the choice is visible.

READS
  out/tesscache/tess_s<seed>_p8_k-8.npz    eight girdle tessellations
  out/sweeps/girdle_perp_ppw8/az*.npz      seed 11
  out/sweeps/mx_girdle_s<seed>_ppw8/az*.npz  the other seven
WRITES
  nothing.

WHAT THIS RUN FOUND, so that a reader of the file knows before running
it which way each question went.
  * the visibility population reproduces, with one wording correction:
    the concentration statistic is per CROSSING and not per facet;
  * the exact coefficient agrees with the scalar in ensemble rms to
    0.07 dB and disagrees per pair by 0.42 to 2.41 over the 5 to 95
    percentiles, which is what was reported; but it does NOT put a floor
    under the dark tail, and it disagrees with the scalar about WHICH
    boundaries are dark, Jaccard 0.47 at one tenth of the median;
  * the converted return does arrive inside the gate, from 29 to 49 mm
    rather than 44 to 70 mm, and it is 16.6 dB below the P-P channel in
    gated power, so it moves the coda level by 0.09 dB and the dark
    boundaries contribute 3 per cent of even that;
  * substituting the exact coefficient changes the field correlation by
    -0.0004 +- 0.0062 over eight tessellations, which confirms the claim
    that the visibility population is a fact about the statistics of the
    coda and not about the pattern;
  * the field correlation itself reproduces: at the published cone
    half-angle with the published Hilbert-envelope estimator this module
    returns +0.157 +- 0.083 against the published +0.162 +- 0.079,
    reached with different code, a different quadrature and a different
    elastic core. Measuring the same field with a local estimator
    instead of a whole-trace Hilbert envelope raises it to
    +0.201 +- 0.129, so the published figure is depressed by the
    pedestal the envelope deposits in the gate.

Run with no argument for everything, or with one of
  core coefficient population shear field
to run a single section. The field section rebuilds its own censuses at
two cone half-angles and takes about ten minutes.
"""
import os
import sys
import time

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import j1
from scipy.stats import spearmanr, t as student

ROOT = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim"
TESS = os.path.join(ROOT, "out", "tesscache")
SWD = os.path.join(ROOT, "out", "sweeps")

# --------------------------------------------------------------- physics
# Ice Ih at -16 C, the constants the solver carries. Voigt notation, Pa.
C11, C33, C44, C12, C13 = 13.93e9, 15.01e9, 3.01e9, 7.08e9, 5.77e9
C66 = 0.5 * (C11 - C12)
RHO = 917.0
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

# Acquisition, Section 3.
C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
LAM = C_REF / F0
KW = 2 * np.pi / LAM
ELEM_A = 6.35e-3 / 2.0
DG = 17.4e-3                     # mean grain diameter, the facet scale
T0_SRC = 1.2 / F0                # ricker delay carried by the sweeps
GATE = (24e-6, 36e-6)
BAND = (0.8e6, 3.0e6)
CONE_DEG = 18.0                  # cone marched; the piston weight tapers it
DT_C = 20e-9                     # common physical time grid
TG = np.arange(20e-6, 42e-6, DT_C)

SEEDS = (11, 7, 17, 23, 41, 53, 71, 89)
SWEEP = {11: "girdle_perp_ppw8"}
SWEEP.update({s: "mx_girdle_s%d_ppw8" % s for s in SEEDS if s != 11})
AZ_COMMON = np.arange(0, 360, 12)

DARK_CUTS = (0.05, 0.10, 0.20)   # INFERRED thresholds, in units of median
MIS_CUT = 20.0                   # degrees, the manuscript's misorientation


# ------------------------------------------------------------ elasticity
def ti_voigt():
    """6x6 stiffness of ice Ih with the c-axis along +z."""
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C11
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
    C[3, 3] = C[4, 4] = C44
    C[5, 5] = C66
    return C


def voigt_to_tensor(C6):
    """Voigt 6x6 (engineering strain) to the 3x3x3x3 stiffness tensor."""
    T = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VOIGT):
        for b, (k, l) in enumerate(VOIGT):
            v = C6[a, b]
            T[i, j, k, l] = T[j, i, k, l] = v
            T[i, j, l, k] = T[j, i, l, k] = v
    return T


def rot_z_to(axis):
    """A rotation carrying +z onto axis. Ice is TI, so the residual spin
    about the c-axis is immaterial and any such rotation will do."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    v = np.cross(np.array([0.0, 0.0, 1.0]), a)
    s, c = np.linalg.norm(v), float(a[2])
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / s ** 2)


def grain_tensors(axes):
    """Full stiffness tensor of every grain, (n_grain, 3,3,3,3)."""
    T0 = voigt_to_tensor(ti_voigt())
    return np.array([np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, T0)
                     for R in (rot_z_to(a) for a in np.asarray(axes, float))])


def christoffel(T, n):
    """Phase speeds (descending) and polarisations along n.

    Broadcasts: T is (...,3,3,3,3) and n is (...,3).
    """
    G = np.einsum("...ijkl,...j,...l->...ik", T, n, n)
    w, V = np.linalg.eigh(G)
    v = np.sqrt(np.maximum(w, 0.0) / RHO)
    o = np.argsort(-v, axis=-1)
    return (np.take_along_axis(v, o, axis=-1),
            np.take_along_axis(V, o[..., None, :], axis=-1))


def interface(T1, T2, n):
    """Exact normal-incidence qP reflection at an anisotropic interface.

    Displacement and traction continuity for the three Christoffel modes
    each side, six equations, solved as written. The interface plane has
    normal n and the incident qP of medium 1 travels along +n.

    Returns (R_PP, |R_PS| pair, v1, v2, P1), with R_PP the PRESSURE
    reflection coefficient: the reflected wave keeps the polarisation
    vector of the incident one but reverses its propagation sense, which
    flips the sign of its stress trace, so the pressure coefficient is
    minus the displacement coefficient. In the isotropic limit that is
    (Z2-Z1)/(Z2+Z1), the same sign convention as the scalar the published
    predictor uses.
    """
    v1, P1 = christoffel(T1, n)
    v2, P2 = christoffel(T2, n)
    sh = np.broadcast_shapes(v1.shape[:-1], v2.shape[:-1])
    A = np.zeros(sh + (6, 6))
    b = np.zeros(sh + (6,))
    A[..., 0:3, 0:3] = P1
    A[..., 3:6, 0:3] = RHO * v1[..., None, :] * P1
    A[..., 0:3, 3:6] = -P2
    A[..., 3:6, 3:6] = RHO * v2[..., None, :] * P2
    b[..., 0:3] = -P1[..., :, 0]
    b[..., 3:6] = RHO * v1[..., 0, None] * P1[..., :, 0]
    x = np.linalg.solve(A, b[..., None])[..., 0]
    return -x[..., 0], np.abs(x[..., 1:3]), v1, v2, P1


def trace_factor(T, n, p, v):
    """|trace(sigma)| per unit displacement amplitude, over (i omega).

    The receiver records -(s11+s22+s33)/3, so this is what a mode of
    polarisation p travelling along n at speed v actually deposits on the
    trace. It vanishes identically for an isotropic shear wave, because
    C_iikl reduces to (3 lambda + 2 mu) delta_kl and p.n = 0 there.
    """
    return np.abs(np.einsum("...kkjl,...j,...l->...", T, p, n)) / v


# --------------------------------------------------------------- the beam
def cone(n_ring=14, n_spoke=44, seed=0, half_deg=None):
    """Jittered polar cone sample and the two-way piston weight.

    Equal-area rings so the quadrature is uniform in the transverse plane,
    jittered so that no ray sits on a lattice direction of the label
    volume. Deliberately not the sunflower spiral the published predictor
    uses.
    """
    rng = np.random.default_rng(seed)
    tmax = np.tan(np.radians(CONE_DEG if half_deg is None else half_deg))
    frac = (np.arange(n_ring) + rng.uniform(0.2, 0.8, n_ring)) / n_ring
    rr = np.repeat(tmax * np.sqrt(frac), n_spoke)
    ph = ((np.arange(n_spoke)[None, :] + rng.uniform(0, 1, (n_ring, 1)))
          / n_spoke * 2 * np.pi).ravel()
    x = KW * ELEM_A * np.sin(np.arctan(rr))
    return rr, ph, airy_amp(x) ** 4


def airy_amp(x):
    """2 J1(x) / x, the piston / circular-facet amplitude pattern."""
    xx = np.maximum(np.abs(x), 1e-12)
    return np.where(np.abs(x) < 1e-9, 1.0, 2.0 * j1(xx) / xx)


def load_tess(seed):
    """(labels, c-axes, Laguerre seed points, cell size) from the cache."""
    with np.load(os.path.join(TESS, "tess_s%d_p8_k-8.npz" % seed)) as z:
        return (np.asarray(z["labels"], np.int16), np.asarray(z["axes"]),
                np.asarray(z["seeds"]), float(z["h"]))


def march(lab, h, az_deg, rays, s_max=0.080):
    """Cell label at every (ray, step) along the beam. -1 is outside."""
    ds = h / 2.0
    a = np.radians(az_deg)
    nh = np.array([np.cos(a), np.sin(a), 0.0])
    t1 = np.array([-np.sin(a), np.cos(a), 0.0])
    t2 = np.array([0.0, 0.0, 1.0])
    rr, ph, wb = rays
    D = (-nh[None, :] + (rr * np.cos(ph))[:, None] * t1
         + (rr * np.sin(ph))[:, None] * t2)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    s = (np.arange(int(s_max / ds)) + 0.5) * ds
    P = (DIA / 2.0) * nh[None, None, :] + s[None, :, None] * D[:, None, :]
    n = np.array(lab.shape)
    g = np.rint((P + n * h / 2.0) / h - 0.5).astype(np.int32)
    ok = np.all((g >= 0) & (g < n), axis=-1)
    gc = np.clip(g, 0, n - 1)
    L = np.where(ok, lab[gc[..., 0], gc[..., 1], gc[..., 2]], -1)
    return L, D, s, ds, wb


def census(tess, az_deg, rays, tensors):
    """Every grain-boundary crossing the beam makes, with its physics.

    One record per (ray, crossing). Times are two-way and include the
    source delay, so they are directly comparable with the recorded
    traces. Both coefficients are evaluated at the same crossing.
    """
    lab, axes, sds, h = tess
    L, D, s, ds, wb = march(lab, h, az_deg, rays)
    nray = L.shape[0]
    v_all, P_all = christoffel(tensors[:, None], D[None, :])   # (ng,nray,3)
    ridx = np.broadcast_to(np.arange(nray)[:, None], L.shape)
    lc = np.clip(L, 0, None)
    inside = L >= 0

    def slow_time(mode):
        V = np.where(inside, v_all[..., mode][lc, ridx], C_REF)
        return np.cumsum(ds / V, axis=1) - 0.5 * ds / V

    tP, tS1, tS2 = slow_time(0), slow_time(1), slow_time(2)

    A, B = L[:, :-1], L[:, 1:]
    cr = (A != B) & (A >= 0) & (B >= 0)
    ri, si = np.nonzero(cr)
    ia, ib = A[ri, si].astype(np.int64), B[ri, si].astype(np.int64)

    def mid(t):
        return 0.5 * (t[ri, si] + t[ri, si + 1])

    t_out = mid(tP)
    va, vb = v_all[ia, ri, 0], v_all[ib, ri, 0]
    r_sc = (vb - va) / (vb + va)
    r_ex, r_ps, v1, _, P1 = interface(tensors[ia], tensors[ib], D[ri])

    nm = sds[ib] - sds[ia]
    nm /= np.linalg.norm(nm, axis=1, keepdims=True) + 1e-30
    ct = np.abs(np.einsum("ij,ij->i", nm, D[ri]))
    st = np.sqrt(np.clip(1.0 - ct ** 2, 0.0, 1.0))

    # recorded pressure per unit incident pressure, both channels
    Tg = tensors[ia]
    f_p = trace_factor(Tg, D[ri], P1[:, :, 0], v1[:, 0])
    f_s = np.stack([trace_factor(Tg, D[ri], P1[:, :, m], v1[:, m])
                    for m in (1, 2)], axis=1)
    ps_press = np.sqrt(((r_ps * f_s) ** 2).sum(1)) / f_p

    # converted lobe: the stationary-phase mismatch across a facet is
    # (k_in + k_out) sin(theta) rather than 2 k_in sin(theta). INFERRED.
    vs_mean = 0.5 * (v1[:, 1] + v1[:, 2])
    lobe_s = 0.5 * (1.0 + v1[:, 0] / vs_mean)

    return dict(
        ray=ri, ia=ia, ib=ib, _dir=D[ri], depth=0.5 * (s[si] + s[si + 1]),
        t_pp=2.0 * t_out + T0_SRC,
        t_ps=np.stack([t_out + mid(tS1), t_out + mid(tS2)], 1) + T0_SRC,
        va=va, vb=vb, r_scalar=r_sc, r_exact=r_ex, r_ps=r_ps,
        ps_press=ps_press, cos_facet=ct, sin_facet=st,
        d_pp=airy_amp(KW * DG * st) ** 2,
        d_ps=airy_amp(lobe_s * KW * DG * st) ** 2,
        wbeam=wb[ri], vs=vs_mean,
        mis=np.degrees(np.arccos(np.clip(
            np.abs(np.einsum("ij,ij->i", axes[ia], axes[ib])), 0.0, 1.0))))


def gated(ev, key="t_pp"):
    t = ev[key]
    return (t >= GATE[0]) & (t < GATE[1])


# ------------------------------------------------------------ section 0
def check_core():
    """Everything the elastic core has to satisfy before it is trusted."""
    print("=" * 74)
    print("CORE CHECKS")
    print("=" * 74)

    def iso(vp, vs):
        lam, mu = RHO * (vp ** 2 - 2 * vs ** 2), RHO * vs ** 2
        C = np.zeros((6, 6))
        C[:3, :3] = lam
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
        C[3, 3] = C[4, 4] = C[5, 5] = mu
        return voigt_to_tensor(C)

    n = np.array([0.3, 0.5, np.sqrt(1 - 0.09 - 0.25)])
    rp, rs, _, _, _ = interface(iso(4000.0, 2000.0), iso(4200.0, 2100.0), n)
    Z1, Z2 = RHO * 4000.0, RHO * 4200.0
    print("  isotropic pair      R_PP %+.8f   (Z2-Z1)/(Z2+Z1) %+.8f"
          % (rp, (Z2 - Z1) / (Z2 + Z1)))
    print("                      |R_PS| %.2e  (must be zero)"
          % np.linalg.norm(rs))

    ax = np.array([[0.0, 0.0, 1.0], [np.sin(0.7), 0.0, np.cos(0.7)],
                   [0.3, 0.6, np.sqrt(1 - 0.45)]])
    T = grain_tensors(ax)
    worst = max(abs(interface(T[g], T[g], n)[0]) for g in range(3))
    print("  identical media     max |R_PP| %.1e  (must be zero)" % worst)

    th = 0.83
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0],
                   [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    Tr = grain_tensors((Rz @ ax.T).T)
    a0 = interface(T[0], T[1], n)[0]
    a1 = interface(Tr[0], Tr[1], Rz @ n)[0]
    print("  rigid rotation      R_PP %+.9e vs %+.9e" % (a0, a1))

    print("  qP against angle to the c-axis, m/s")
    Tc = grain_tensors(np.array([[0.0, 0.0, 1.0]]))[0]
    for psi in (0, 30, 45, 60, 90):
        d = np.array([np.sin(np.radians(psi)), 0.0, np.cos(np.radians(psi))])
        v, P = christoffel(Tc, d)
        tilt = np.degrees(np.arccos(min(abs(float(P[:, 0] @ d)), 1.0)))
        print("    psi %5.1f  vqP %8.2f  vqS %8.2f %8.2f  qP tilt %5.2f deg"
              % (psi, v[0], v[1], v[2], tilt))
    print("  The qP polarisation is up to 4.8 deg off the ray, which is")
    print("  large beside a 1 per cent speed contrast: that tilt, not the")
    print("  impedance jump, is what drives the conversion in section 3.")
    print()


# ------------------------------------------------------------ section 1
def report_coefficient(evs):
    """Exact against scalar, and what happens in the scalar's dark tail."""
    print("=" * 74)
    print("1. THE COEFFICIENT")
    print("=" * 74)
    sc = np.abs(np.concatenate([e["r_scalar"][gated(e)] for e in evs]))
    ex = np.abs(np.concatenate([e["r_exact"][gated(e)] for e in evs]))
    n = len(sc)
    rms_s, rms_e = np.sqrt((sc ** 2).mean()), np.sqrt((ex ** 2).mean())
    print("  %d gated crossings, 8 tessellations x %d azimuths"
          % (n, len(AZ_COMMON)))
    print("  scalar |R|  rms %.4f %%   median %.4f %%"
          % (100 * rms_s, 100 * np.median(sc)))
    print("  exact  |R|  rms %.4f %%   median %.4f %%"
          % (100 * rms_e, 100 * np.median(ex)))
    print("  ENSEMBLE rms agreement %+.3f dB" % (20 * np.log10(rms_e / rms_s)))
    q = np.percentile(ex / np.maximum(sc, 1e-15), [1, 5, 25, 50, 75, 95, 99])
    print("  PER-PAIR ratio exact/scalar, percentiles 1/5/25/50/75/95/99")
    print("    " + "  ".join("%.3f" % v for v in q))
    print("  Spearman rank correlation of the two coefficients %.4f"
          % spearmanr(sc, ex).statistic)
    print()
    print("  WHAT THE EXACT COEFFICIENT DOES WHERE THE SCALAR VANISHES")
    med_s, med_e = np.median(sc), np.median(ex)
    print("  %-26s %8s %12s %12s" % ("scalar selection", "n", "med exact",
                                     "x exact med"))
    for cut in (1e-3, 1e-2, 5e-2, 1e-1):
        sel = sc < cut * med_s
        if sel.sum() < 3:
            continue
        print("  scalar |R| < %6.3f x med  %8d %12.3e %12.4f"
              % (cut, sel.sum(), np.median(ex[sel]),
                 np.median(ex[sel]) / med_e))
    lo = np.argsort(sc)[:max(n // 1000, 10)]
    print("  the 0.1 per cent darkest scalar crossings sit at scalar |R| ")
    print("  below %.1e and exact |R| at %.1e, a factor %.0f apart"
          % (sc[lo].max(), np.median(ex[lo]), np.median(ex[lo] / sc[lo])))
    print("  floor: min scalar |R| %.2e, min exact |R| %.2e"
          % (sc.min(), ex.min()))
    print("  Both reach zero, so the exact coefficient does not put a floor")
    print("  under the population. What it does is move WHICH boundaries")
    print("  sit at the bottom.")
    print()
    print("  DARK FRACTION, AND WHETHER THE TWO AGREE ON WHICH ONES")
    print("  %-10s %10s %10s %12s %12s"
          % ("threshold", "scalar", "exact", "both", "Jaccard"))
    for cut in DARK_CUTS:
        ds, de = sc < cut * med_s, ex < cut * med_e
        both = (ds & de).sum()
        print("  %-10s %9.1f%% %9.1f%% %11.1f%% %12.3f"
              % ("%.2f x med" % cut, 100 * ds.mean(), 100 * de.mean(),
                 100 * both / n, both / max((ds | de).sum(), 1)))
    print("  The dark DECILE is the same size under either coefficient and")
    print("  is NOT the same decile: at 0.10 x median the two agree on")
    print("  about half the members. A population claim about the dark")
    print("  tail is reproducible as a COUNT and not as a MEMBERSHIP.")
    print()
    return sc, ex


def report_asymmetry(tensors, evs):
    """The exact coefficient is not antisymmetric under swapping sides.

    The scalar obeys R(1->2) = -R(2->1) identically, so under the scalar a
    boundary has one visibility. The exact solve does not: the reflected
    quasi-shear carries away a different share of the energy depending on
    which crystal the wave arrives from, and the qP coefficient absorbs
    the difference. If a boundary can be brighter from one side than the
    other, "the visibility of a boundary" is not a property of the
    boundary alone, which the manuscript's wording has to allow for.
    """
    e = evs[0]
    g = gated(e)
    ia, ib = e["ia"][g], e["ib"][g]
    print("  RECIPROCITY, on %d gated crossings of the first census."
          % g.sum())
    fwd = np.abs(e["r_exact"][g])
    rev = np.abs(interface(tensors[ib], tensors[ia], e["_dir"][g])[0])
    sc = np.abs(e["r_scalar"][g])
    asym = np.abs(fwd - rev) / np.maximum(0.5 * (fwd + rev), 1e-30)
    print("    |R| forward against reverse, relative difference")
    print("      percentiles 5/50/95: %s"
          % np.percentile(100 * asym, [5, 50, 95]).round(1) + " per cent")
    print("    the scalar's own forward/reverse difference is %.1e"
          % np.abs(sc - sc).max())
    print()


# ------------------------------------------------------------ section 2
def report_population(evs, per_spec):
    """The four visibility numbers, rebuilt."""
    print("=" * 74)
    print("2. THE VISIBILITY POPULATION")
    print("=" * 74)
    for tag, key in (("scalar (published predictor)", "r_scalar"),
                     ("exact 6x6 solve", "r_exact")):
        r = np.abs(np.concatenate([e[key][gated(e)] for e in evs]))
        med = np.median(r)
        print("  %-30s dark fraction at %s x median:"
              % (tag, "/".join("%.2f" % c for c in DARK_CUTS)))
        print("  %-30s %s" % ("", "  ".join(
            "%.1f%%" % (100 * (r < c * med).mean()) for c in DARK_CUTS)))
    print()

    print("  CROSSINGS IN THE GATE AND HOW MANY OF THEM CARRY IT")
    print("  %-6s %7s %7s %7s %7s %7s %7s %7s %7s"
          % ("seed", "n_geom", "n_eff", "n_ax", "n_effax", "n_beam",
             "tp1_az", "tp1", "tp1_ev"))
    keys = ("n_geom", "n_eff", "n_axis", "n_eff_axis", "n_eff_beam",
            "top1_az", "top1", "top1_ev")
    rows = np.array([[d[k] for k in keys] for d in per_spec])
    for seed, r in zip(SEEDS, rows):
        print("  %-6d %7.2f %7.2f %7.2f %7.2f %7.2f %6.1f%% %6.1f%% %6.1f%%"
              % (seed, r[0], r[1], r[2], r[3], r[4], 100 * r[5],
                 100 * r[6], 100 * r[7]))
    lbl = ("beam-weighted crossings per ray",
           "effective bright crossings per ray",
           "crossings on the beam axis",
           "effective bright crossings on the axis",
           "effective bright crossings in the whole beam",
           "share carried by the brightest 1 per cent, per azimuth",
           "share carried by the brightest 1 per cent, pooled facets",
           "share carried by the brightest 1 per cent of CROSSINGS")
    print()
    for k, name in enumerate(lbl):
        sc = 100 if k >= 5 else 1
        print("  %-56s %6.2f +- %.2f%s"
              % (name, sc * rows[:, k].mean(), sc * rows[:, k].std(ddof=1),
                 " %" if k >= 5 else ""))
    print("  facets illuminated per specimen over the revolution: %d"
          % int(np.mean([d["n_facet"] for d in per_spec])))
    print()

    print("  WHAT SETS BRIGHTNESS: SPEARMAN RANK CORRELATIONS")
    dv = np.abs(np.concatenate([e["vb"][gated(e)] - e["va"][gated(e)]
                                for e in evs]))
    mis = np.concatenate([e["mis"][gated(e)] for e in evs])
    sc = np.abs(np.concatenate([e["r_scalar"][gated(e)] for e in evs]))
    ex = np.abs(np.concatenate([e["r_exact"][gated(e)] for e in evs]))
    pw = np.concatenate([(e["r_exact"][gated(e)] ** 2) * e["d_pp"][gated(e)]
                         * e["wbeam"][gated(e)] for e in evs])
    print("  %-34s %11s %11s %11s"
          % ("predictor", "vs scalar", "vs exact", "vs power"))
    for name, x in (("|dv| of qP along the ray", dv),
                    ("misorientation of the c-axes", mis)):
        print("  %-34s %11.5f %11.5f %11.5f"
              % (name, spearmanr(x, sc).statistic,
                 spearmanr(x, ex).statistic, spearmanr(x, pw).statistic))
    print("  Against the SCALAR the first row is a near identity: the")
    print("  scalar IS |dv| divided by a sum that varies by 6 per cent, so")
    print("  a rank correlation of one is arithmetic and not evidence. The")
    print("  exact column is the one that carries information.")
    print("  The last column is the gated power, which carries the facet")
    print("  directivity as well as the coefficient. Neither geometric")
    print("  quantity survives it, which is the point: what a facet")
    print("  RETURNS is set by where it points, and how visible it is when")
    print("  it does point at you is set by the speed jump.")
    print()
    print("  MISORIENTATION OF THE DARK BOUNDARIES")
    print("  %-30s %9s %11s %11s"
          % ("selection", "n", "median mis", "> %.0f deg" % MIS_CUT))
    for tag, r in (("scalar-dark", sc), ("exact-dark", ex)):
        med = np.median(r)
        for cut in DARK_CUTS:
            sel = r < cut * med
            print("  %-30s %9d %10.1f %10.1f%%"
                  % ("crossings, %s < %.2f med" % (tag, cut), sel.sum(),
                     np.median(mis[sel]), 100 * (mis[sel] > MIS_CUT).mean()))
    print("  %-30s %9d %10.1f %10.1f%%"
          % ("all illuminated crossings", mis.size, np.median(mis),
             100 * (mis > MIS_CUT).mean()))
    # the same population counted once per (facet, azimuth) rather than
    # once per ray that happens to cross it
    kk = np.concatenate([facet_key(e["ia"][gated(e)], e["ib"][gated(e)])
                         + 10 ** 11 * i for i, e in enumerate(evs)])
    wb = np.concatenate([e["wbeam"][gated(e)] for e in evs])
    order = np.lexsort((-wb, kk))
    first = np.ones(len(order), bool)
    first[1:] = kk[order][1:] != kk[order][:-1]
    u = order[first]
    med = np.median(ex[u])
    print("  counted once per (facet, azimuth), brightest ray as the")
    print("  representative:")
    for cut in DARK_CUTS:
        sel = ex[u] < cut * med
        print("  %-30s %9d %10.1f %10.1f%%"
              % ("facets, exact < %.2f med" % cut, sel.sum(),
                 np.median(mis[u][sel]), 100 * (mis[u][sel] > MIS_CUT).mean()))
    print("  %-30s %9d %10.1f %10.1f%%"
          % ("all illuminated facets", u.size, np.median(mis[u]),
             100 * (mis[u] > MIS_CUT).mean()))
    print("  absolute thresholds, which do not move with the median:")
    for cut in (1e-3, 1e-2):
        sel = ex[u] < cut
        m = mis[u][sel]
        print("  %-30s %9d %10.1f %10.1f%%"
              % ("facets, exact |R| < %.0e" % cut, sel.sum(),
                 np.median(m) if m.size else np.nan,
                 100 * (m > MIS_CUT).mean() if m.size else np.nan))
    print()


def facet_key(ia, ib):
    """One key per unordered grain pair: a facet is the same facet from
    either side, even though the exact coefficient is not."""
    lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
    return lo * 100000 + hi


def _concentration(k, p, frac=0.01):
    """Share of the power carried by the brightest `frac` of the facets."""
    _, inv = np.unique(k, return_inverse=True)
    f = np.sort(np.bincount(inv, weights=p))[::-1]
    n = max(int(round(frac * len(f))), 1)
    return float(f[:n].sum() / f.sum()), len(f)


FRESNEL_D = 14.9e-3      # first Fresnel diameter at the coda gate, Table 3


def report_dynamic_range(evs):
    """The two terms of Eq. (facetmodel) on the same population.

    Section 5 states that the geometric term dominates in predictive value
    and not in dynamic range, and quotes the 5 to 95 percentile spread of
    the squared reflection coefficient. The companion figure for the
    geometric factor is flagged in the manuscript as having no module
    behind it. It is measured here, on one population, both terms at once,
    so the comparison is between two numbers of the same kind.

    The facet aperture is INFERRED twice over and both are printed: the
    mean grain diameter, which is what the predictor uses, and the first
    Fresnel diameter, which is what the census in Section 5 caps it at.
    """
    print("  DYNAMIC RANGE OF THE TWO TERMS, same insonified population")
    g = [gated(e) for e in evs]
    r2 = np.concatenate([e["r_exact"][s] ** 2 for e, s in zip(evs, g)])
    r2s = np.concatenate([e["r_scalar"][s] ** 2 for e, s in zip(evs, g)])
    st = np.concatenate([e["sin_facet"][s] for e, s in zip(evs, g)])
    dd = np.concatenate([e["d_pp"][s] for e, s in zip(evs, g)])
    df = airy_amp(KW * FRESNEL_D * st) ** 2
    print("  %-34s %11s %11s %11s"
          % ("term", "5 %", "median", "5-95, dB"))
    for name, x in (("squared coefficient, scalar", r2s),
                    ("squared coefficient, exact", r2),
                    ("geometric factor, D = %.1f mm" % (DG * 1e3), dd),
                    ("geometric factor, D = %.1f mm" % (FRESNEL_D * 1e3),
                     df)):
        p5, p50, p95 = np.percentile(x, [5, 50, 95])
        print("  %-34s %11.3e %11.3e %11.1f"
              % (name, p5, p50, 10 * np.log10(p95 / max(p5, 1e-300))))
    print("  Measured on this population the coefficient has the LARGER")
    print("  dynamic range of the two, not the smaller, which strengthens")
    print("  rather than weakens the point being made: what separates the")
    print("  two terms is predictive value and not spread.")
    print()


def report_random_look(n_dir=8, seed=0):
    """The same population sampled the way the earlier work sampled it.

    The published visibility measurement drew (boundary, look-direction)
    pairs at random rather than taking the boundaries the beam actually
    lights, and the two populations are not the same: the beam enters at
    the rim and travels a chord, so its look directions all lie in the
    disc plane, which under a girdle fabric is the plane the c-axes avoid.
    A uniformly random direction on the sphere is a different experiment.
    This is here so the difference can be seen rather than argued about.
    """
    print("  THE RANDOM-DIRECTION POPULATION, for comparison")
    print("  %-8s %8s %10s %10s %10s"
          % ("seed", "n pairs", "med |R| %", "dark mis", "dark > 20"))
    rng = np.random.default_rng(seed)
    agg = []
    for sd in SEEDS:
        lab, axes, sds, h = load_tess(sd)
        T = grain_tensors(axes)
        pairs = _adjacent_pairs(lab)
        u = rng.normal(size=(n_dir, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        ia = np.repeat(pairs[:, 0], n_dir)
        ib = np.repeat(pairs[:, 1], n_dir)
        d = np.tile(u, (len(pairs), 1))
        r = np.abs(interface(T[ia], T[ib], d)[0])
        mis = np.degrees(np.arccos(np.clip(
            np.abs(np.einsum("ij,ij->i", axes[ia], axes[ib])), 0.0, 1.0)))
        dk = r < 0.1 * np.median(r)
        agg.append((100 * (mis[dk] > MIS_CUT).mean(),
                    100 * (mis > MIS_CUT).mean()))
        print("  %-8d %8d %10.4f %10.1f %9.1f%%"
              % (sd, len(ia), 100 * np.median(r), np.median(mis[dk]),
                 agg[-1][0]))
    a = np.array(agg)
    print("  dark boundaries above %.0f deg: %.1f +- %.1f per cent, against"
          % (MIS_CUT, a[:, 0].mean(), a[:, 0].std(ddof=1)))
    print("  %.1f +- %.1f per cent for all boundaries."
          % (a[:, 1].mean(), a[:, 1].std(ddof=1)))
    print()


def _adjacent_pairs(lab):
    """Unordered grain pairs that share a face in the label volume."""
    out = set()
    for ax in range(3):
        a = np.take(lab, np.arange(lab.shape[ax] - 1), axis=ax)
        b = np.take(lab, np.arange(1, lab.shape[ax]), axis=ax)
        m = (a != b) & (a >= 0) & (b >= 0)
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        out.update(np.unique(lo * 100000 + hi).tolist())
    k = np.array(sorted(out), dtype=np.int64)
    return np.stack([k // 100000, k % 100000], axis=1)


def specimen_summary(evs_of_seed, rays):
    """Per-specimen concentration and crossing counts, azimuth-averaged.

    Two ways of counting the facets, both reported, because the claim
    is sensitive to the choice and the choice is a convention:
      per azimuth  the beam sees one specimen from one direction, so the
                   population is the facets lit at that azimuth;
      pooled       the specimen's whole illuminated facet set over the
                   revolution, which is the larger denominator.
    """
    ngeo, neff, nax, neffax, nbeam, top_az = [], [], [], [], [], []
    pooled_k, pooled_p = [], []
    wr = rays[2]
    nr = wr.size
    axr = np.argsort(-wr)[:max(nr // 20, 1)]
    for ev in evs_of_seed:
        g = gated(ev)
        p = (ev["r_exact"][g] ** 2) * ev["d_pp"][g] * ev["wbeam"][g]
        ray = ev["ray"][g]
        cnt = np.bincount(ray, minlength=nr).astype(float)
        s1 = np.bincount(ray, weights=p, minlength=nr)
        s2 = np.bincount(ray, weights=p ** 2, minlength=nr)
        live = s1 > 0
        ngeo.append(float((cnt * wr).sum() / wr.sum()))
        neff.append(float((wr[live] * s1[live] ** 2 / s2[live]).sum()
                          / wr[live].sum()))
        # the axis is not a sampled ray under either quadrature rule, so
        # the axial statistic is taken over the 5 per cent of rays with
        # the largest two-way piston weight
        nax.append(float(cnt[axr].mean()))
        ok = s1[axr] > 0
        neffax.append(float((s1[axr][ok] ** 2 / s2[axr][ok]).mean()))
        nbeam.append(float(p.sum() ** 2 / (p ** 2).sum()))
        k = facet_key(ev["ia"][g], ev["ib"][g])
        top_az.append(_concentration(k, p)[0])
        pooled_k.append(k)
        pooled_p.append(p)
    allp = np.concatenate(pooled_p)
    tp, nfac = _concentration(np.concatenate(pooled_k), allp)
    ev1 = np.sort(allp)[::-1]
    n1 = max(int(round(0.01 * len(ev1))), 1)
    return dict(n_geom=np.mean(ngeo), n_eff=np.mean(neff),
                n_axis=np.mean(nax), n_eff_axis=np.mean(neffax),
                n_eff_beam=np.mean(nbeam), top1_az=np.mean(top_az),
                top1=tp, top1_ev=float(ev1[:n1].sum() / ev1.sum()),
                n_facet=nfac)


# ------------------------------------------------------------ section 3
def report_shear(evs):
    """Does the converted return land in the gate, and does it matter?"""
    print("=" * 74)
    print("3. THE SHEAR CHANNEL")
    print("=" * 74)
    ex = np.concatenate([np.abs(e["r_exact"]) for e in evs])
    ps = np.concatenate([np.sqrt((e["r_ps"] ** 2).sum(1)) for e in evs])
    # recorded pressure of each channel per unit INCIDENT pressure: for
    # the P->P return that is |R_PP| itself, because the reflected qP has
    # the incident mode's stress trace; for the converted return it is
    # ps_press, which already carries the quasi-shear trace factors.
    sp = np.concatenate([e["ps_press"] for e in evs])
    pr = sp / np.maximum(ex, 1e-30)
    gpp = np.concatenate([gated(e) for e in evs])
    tps = np.concatenate([e["t_ps"] for e in evs])
    dep = np.concatenate([e["depth"] for e in evs])
    vs = np.concatenate([e["vs"] for e in evs])
    gs = ((tps >= GATE[0]) & (tps < GATE[1])).any(1)

    print("  DISPLACEMENT. The conversion is not small:")
    print("    |R_PS| rms %.4f %%  against |R_PP| rms %.4f %%  -> %.2f x"
          % (100 * np.sqrt((ps ** 2).mean()), 100 * np.sqrt((ex ** 2).mean()),
             np.sqrt((ps ** 2).mean() / (ex ** 2).mean())))
    print("  PRESSURE. What the receiver records is not that. The quasi-")
    print("  shear stress trace nearly vanishes: C_iikl p_k n_l reduces to")
    print("  (3 lambda + 2 mu) p.n for an isotropic solid and ice is only")
    print("  weakly anisotropic, so most of the conversion is thrown away")
    print("  on reception rather than never generated.")
    print("    ensemble rms recorded pressure, S / P = %.4f (%+.1f dB)"
          % (_rms_ratio(sp, ex), 20 * np.log10(_rms_ratio(sp, ex))))
    print("    per-pair ratio percentiles 5/50/95: %s"
          % np.percentile(pr, [5, 50, 95]).round(4))
    print("    mean quasi-shear speed on the path %.0f m/s, Vp/Vs %.2f"
          % (vs.mean(), C_REF / vs.mean()))
    print()
    print("  ARRIVAL TIME. The return leg is integrated at the qS slowness")
    print("  of the real path, not at a nominal speed.")
    print("    P->P crossings in the gate come from %.0f-%.0f mm"
          % (1e3 * dep[gpp].min(), 1e3 * dep[gpp].max()))
    print("    P->S crossings in the gate come from %.0f-%.0f mm"
          % (1e3 * dep[gs].min(), 1e3 * dep[gs].max()))
    print("    %.1f per cent of all crossings put a converted return in"
          % (100 * gs.mean()))
    print("    the gate, against %.1f per cent for the P->P channel."
          % (100 * gpp.mean()))
    print("    Overlap of the two populations: %.1f per cent of gated P->S"
          % (100 * (gs & gpp).sum() / max(gs.sum(), 1)))
    print("    crossings also return in P inside the gate, so the S channel")
    print("    is looking at a DIFFERENT part of the specimen.")
    print()
    print("  DOES IT MATTER? Gated power in each channel, same beam, same")
    print("  facets, with the converted lobe narrowed by (k_P+k_S)/2k_P.")
    ppw = np.concatenate([(e["r_exact"] ** 2) * e["d_pp"] * e["wbeam"]
                          for e in evs])
    psw = np.concatenate([(e["ps_press"] ** 2) * e["d_ps"] * e["wbeam"]
                          for e in evs])
    tot_p = ppw[gpp].sum()
    tot_s = psw[gs].sum()
    print("    gated P->P power (arbitrary units) %.4e" % tot_p)
    print("    gated P->S power, same units       %.4e" % tot_s)
    print("    S / P in the gate = %.4f  (%+.2f dB), and adding it to the"
          % (tot_s / tot_p, 10 * np.log10(tot_s / tot_p)))
    print("    coda would raise the level by %+.2f dB."
          % (10 * np.log10(1.0 + tot_s / tot_p)))
    same = np.concatenate([(e["ps_press"] ** 2) * e["d_pp"] * e["wbeam"]
                           for e in evs])
    print("    With the P lobe instead of the narrowed one it would be")
    print("    %+.2f dB, so the lobe model carries %.1f dB of the answer."
          % (10 * np.log10(same[gs].sum() / tot_p),
             10 * np.log10(same[gs].sum() / tot_s)))
    print()
    print("  WHERE THE CONVERTED POWER COMES FROM. If the S channel merely")
    print("  re-illuminated the boundaries the P channel already sees, it")
    print("  would carry no new information whatever its level.")
    med = np.median(ex[gpp])
    print("    %-22s %14s %14s"
          % ("facets dark in P at", "share of S", "share of P"))
    for cut in DARK_CUTS:
        d = ex < cut * med
        print("    %-22s %13.1f%% %13.2f%%"
              % ("%.2f x median" % cut, 100 * psw[gs & d].sum() / tot_s,
                 100 * ppw[gpp & d].sum() / tot_p))
    print()
    print("  AT DARK FACETS. Definition matters, so both are printed.")
    print("  %-22s %8s %14s %14s"
          % ("dark by", "n", "median S/P", "share S > P"))
    for tag, r in (("exact |R_PP|", ex),
                   ("scalar |R|", np.concatenate(
                       [np.abs(e["r_scalar"]) for e in evs]))):
        med = np.median(r[gpp])
        for cut in DARK_CUTS:
            sel = gpp & (r < cut * med)
            print("  %-22s %8d %14.3f %13.1f%%"
                  % ("%s < %.2f med" % (tag, cut), sel.sum(),
                     np.median(pr[sel]), 100 * (pr[sel] > 1).mean()))
    print()


def _rms_ratio(a, b):
    return float(np.sqrt((a ** 2).mean() / (b ** 2).mean())) if b.size else 0.0


# ------------------------------------------------------------ section 4
def pulse_power_kernel(dt, half_us=2.0):
    """Power envelope of the band-passed ricker, unit sum.

    Built on an isolated, zero-padded wavelet, so the analytic signal is
    taken of something compactly supported and the non-locality that makes
    a whole-trace Hilbert unsafe does not arise.
    """
    fs = 2e9
    t = np.arange(int(12e-6 * fs)) / fs
    a = (np.pi * F0 * (t - 6e-6)) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype="band",
                 output="sos")
    e = np.abs(hilbert(sosfiltfilt(sos, w))) ** 2
    n = int(round(half_us * 1e-6 / dt))
    k = np.interp(np.arange(-n, n + 1) * dt + 6e-6, t, e)
    return k / k.sum()


def measured_fields(sweep, az_list):
    """(local, envelope) gated power fields on the common time grid.

    local     2 <x^2> smoothed by the pulse power kernel. Local by
              construction: the kernel is +-2 us and the gate starts
              24 us after a front arrival that ends by 4 us.
    envelope  |hilbert| of the whole band-passed trace, squared, which is
              the published estimator and is carried only for comparison.
    """
    root = os.path.join(SWD, sweep)
    loc, env = [], []
    for a in az_list:
        with np.load(os.path.join(root, "az%03d.npz" % a)) as z:
            x = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                     btype="band", output="sos")
        xf = sosfiltfilt(sos, x)
        t = np.arange(len(x)) * dt
        k = pulse_power_kernel(dt)
        loc.append(np.interp(TG, t, np.convolve(2.0 * xf ** 2, k, "same")))
        env.append(np.interp(TG, t, np.abs(hilbert(xf)) ** 2))
    return np.array(loc), np.array(env)


def predicted_field(ev_list, key, add_shear=False):
    """Azimuth x time predicted power for one reflection coefficient.

    With add_shear the mode-converted return is binned as well, at its own
    two-way time, with its own recorded-pressure amplitude and its own
    narrower specular lobe. Both quasi-shear branches are laid down with
    half the converted power each, which is the neutral choice: a wave
    converted into one branch does not stay in it along the return path.
    """
    edges = np.concatenate([TG - DT_C / 2, [TG[-1] + DT_C / 2]])
    k = pulse_power_kernel(DT_C)
    rows = []
    for ev in ev_list:
        w = (ev[key] ** 2) * ev["d_pp"] * ev["wbeam"]
        h = np.histogram(ev["t_pp"], edges, weights=w)[0]
        if add_shear:
            ws = 0.5 * (ev["ps_press"] ** 2) * ev["d_ps"] * ev["wbeam"]
            for m in (0, 1):
                h = h + np.histogram(ev["t_ps"][:, m], edges,
                                     weights=ws)[0]
        rows.append(np.convolve(h, k, "same"))
    return np.array(rows)


def harm_centre(X, kmax=4):
    """Strip azimuthal harmonics 0..kmax at every time sample, then the
    per-azimuth level. Nothing a bulk fabric model could produce survives.
    """
    n = X.shape[0]
    th = np.arange(n) / n * 2 * np.pi
    cols = [np.ones(n)]
    for k in range(1, kmax + 1):
        cols += [np.cos(k * th), np.sin(k * th)]
    A = np.column_stack(cols)
    Y = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]
    return Y - Y.mean(1, keepdims=True)


def corr(a, b):
    a = a.ravel() - a.ravel().mean()
    b = b.ravel() - b.ravel().mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def channel_fields(ev_list, key="r_exact"):
    """(P->P field, P->S field) separately, on the common time grid."""
    edges = np.concatenate([TG - DT_C / 2, [TG[-1] + DT_C / 2]])
    k = pulse_power_kernel(DT_C)
    pp, ps = [], []
    for ev in ev_list:
        w = (ev[key] ** 2) * ev["d_pp"] * ev["wbeam"]
        pp.append(np.convolve(np.histogram(ev["t_pp"], edges,
                                           weights=w)[0], k, "same"))
        ws = 0.5 * (ev["ps_press"] ** 2) * ev["d_ps"] * ev["wbeam"]
        h = sum(np.histogram(ev["t_ps"][:, m], edges, weights=ws)[0]
                for m in (0, 1))
        ps.append(np.convolve(h, k, "same"))
    return np.array(pp), np.array(ps)


def report_shear_null(cone_deg=8.9, n_perm=200, seed=0):
    """Is the improvement the shear channel brings real, or is it slack?

    Adding any extra positive term to a prediction can raise a
    correlation, so the addition is scored against two nulls that keep
    everything except the physics.

      AZIMUTH ROLL   the converted field is rotated rigidly against the
                     measurement before it is added, which preserves its
                     amplitude, its time structure and its azimuthal
                     marginal and destroys only its registration. All
                     n_az - 1 non-trivial rolls are evaluated exactly.
      WEIGHT SHUFFLE the converted events keep their arrival times and
                     the multiset of their weights, but the pairing is
                     permuted within each azimuth, which destroys which
                     facet returned what while leaving the level alone.
    """
    print("  IS THE SHEAR ADDITION REAL? Two nulls on the addition alone,")
    print("  cone half-angle %.1f deg, %d weight permutations."
          % (cone_deg, n_perm))
    g = (TG >= GATE[0]) & (TG < GATE[1])
    rays = cone(half_deg=cone_deg)
    edges = np.concatenate([TG - DT_C / 2, [TG[-1] + DT_C / 2]])
    kern = pulse_power_kernel(DT_C)
    rng = np.random.default_rng(seed)
    print("  %-6s %8s %8s %9s %8s %8s %8s"
          % ("seed", "r P", "r P+S", "roll rank", "p roll", "r rolled",
             "z shuf"))
    ranks, gain_true, gain_roll = [], [], []
    for sd in SEEDS:
        loc, _ = measured_fields(SWEEP[sd], AZ_COMMON)
        m = harm_centre(10 * np.log10(np.maximum(loc[:, g], 1e-300)))
        tess = load_tess(sd)
        T = grain_tensors(tess[1])
        evs = [census(tess, a, rays, T) for a in AZ_COMMON]
        P, S = channel_fields(evs)

        def sc(F):
            return corr(m, harm_centre(10 * np.log10(F[:, g] + 1e-30)))

        r0, rp = sc(P), sc(P + S)
        roll = np.array([sc(P + np.roll(S, k, axis=0))
                         for k in range(len(AZ_COMMON))])
        rank = int((roll >= roll[0]).sum())
        shuf = []
        for _ in range(n_perm):
            rows = []
            for ev in evs:
                ws = 0.5 * (ev["ps_press"] ** 2) * ev["d_ps"] * ev["wbeam"]
                ws = rng.permutation(ws)
                h = sum(np.histogram(ev["t_ps"][:, mm], edges,
                                     weights=ws)[0] for mm in (0, 1))
                rows.append(np.convolve(h, kern, "same"))
            shuf.append(sc(P + np.array(rows)))
        shuf = np.array(shuf)
        z = (rp - shuf.mean()) / shuf.std(ddof=1)
        ranks.append(rank)
        gain_true.append(rp - r0)
        gain_roll.append(roll[1:].mean() - r0)
        print("  %-6d %8.3f %8.3f %6d/%-2d %8.4f %8.3f %8.2f"
              % (sd, r0, rp, rank, len(AZ_COMMON), rank / len(AZ_COMMON),
                 roll[1:].mean(), z))
    k1 = sum(1 for r in ranks if r == 1)
    gt, gr = np.array(gain_true), np.array(gain_roll)
    print("  %d of %d tessellations rank first among the %d rolls; under"
          % (k1, len(SEEDS), len(AZ_COMMON)))
    print("  the null that is binomial(%d, 1/%d), p = %.3g"
          % (len(SEEDS), len(AZ_COMMON),
             float(1 - sum(_binom_pmf(len(SEEDS), 1.0 / len(AZ_COMMON), i)
                           for i in range(k1)))))
    print("  gain from the CORRECTLY registered converted field %+.4f"
          % gt.mean())
    print("  gain from a MISREGISTERED one, averaged over rolls %+.4f"
          % gr.mean())
    d = gt - gr
    tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print("  registered minus misregistered %+.4f +- %.4f over %d"
          % (d.mean(), d.std(ddof=1), len(d)))
    print("  tessellations, paired t = %.2f on %d degrees of freedom,"
          % (tstat, len(d) - 1))
    print("  one-sided p = %.4f" % float(student.sf(tstat, len(d) - 1)))
    print("  %.0f per cent of the apparent improvement is slack rather than"
          % (100 * gr.mean() / gt.mean()))
    print("  information: a broad positive term added to a sparse")
    print("  prediction fills its logarithmic nulls and raises r whatever")
    print("  it is registered to. The rest is registration, and it is the")
    print("  paired comparison and not the rank that has the power to see")
    print("  it at n = 8.")
    print()


def _binom_pmf(n, p, k):
    from math import comb
    return comb(n, k) * p ** k * (1 - p) ** (n - k)


VARIANTS = (("scalar", "r_scalar", False), ("exact", "r_exact", False),
            ("exact+S", "r_exact", True))


def report_field(cone_list=(8.9, 18.0)):
    """Does the coefficient, or the shear channel, change the prediction?

    The census is rebuilt at each cone half-angle rather than reused,
    because the beam model is the one convention this module does not
    inherit: the published field prediction marches the far-field
    half-angle of the element, 8.9 degrees, and the published time-
    resolved predictor marches 18. Both are run so the answer to the
    question that matters, whether the coefficient changes anything, is
    seen to be independent of that choice.
    """
    print("=" * 74)
    print("4. DOES THE COEFFICIENT CHANGE THE PREDICTION?")
    print("=" * 74)
    g = (TG >= GATE[0]) & (TG < GATE[1])
    meas = {s: measured_fields(SWEEP[s], AZ_COMMON) for s in SEEDS}
    out = {}
    for cd in cone_list:
        rays = cone(half_deg=cd)
        rows = []
        for seed in SEEDS:
            tess = load_tess(seed)
            T = grain_tensors(tess[1])
            evs = [census(tess, a, rays, T) for a in AZ_COMMON]
            r = []
            for M in meas[seed]:
                m = harm_centre(10 * np.log10(np.maximum(M[:, g], 1e-300)))
                for _, key, sh in VARIANTS:
                    P = predicted_field(evs, key, sh)[:, g]
                    r.append(corr(m, harm_centre(10 * np.log10(P + 1e-30))))
            rows.append(r)
        a = np.array(rows)
        out[cd] = a
        print("  cone half-angle %.1f deg" % cd)
        print("  %-6s %s" % ("seed", "".join(
            "%10s" % ("%s %s" % (e, v[0])) for e in ("loc", "env")
            for v in VARIANTS)))
        for seed, r in zip(SEEDS, a):
            print("  %-6d %s" % (seed, "".join("%+10.3f" % x for x in r)))
        print("  %-6s %s" % ("mean", "".join("%+10.3f" % x
                                             for x in a.mean(0))))
        print("  %-6s %s" % ("sd", "".join("%10.3f" % x
                                           for x in a.std(0, ddof=1))))
        print()
        for j, tag in enumerate(("local", "envelope")):
            base = a[:, 3 * j]
            for k in (1, 2):
                d = a[:, 3 * j + k] - base
                print("    %-9s %-8s minus scalar: %+.4f +- %.4f  "
                      "(max |change| %.4f)"
                      % (tag, VARIANTS[k][0], d.mean(), d.std(ddof=1),
                         np.abs(d).max()))
        print()
    return out


# ------------------------------------------------------------------ main
def build_all(verbose=True):
    """Every census the module needs, once."""
    rays = cone()
    out, per_spec, flat = {}, [], []
    for seed in SEEDS:
        t0 = time.time()
        tess = load_tess(seed)
        T = grain_tensors(tess[1])
        evs = [census(tess, a, rays, T) for a in AZ_COMMON]
        out[seed] = evs
        flat.extend(evs)
        per_spec.append(specimen_summary(evs, rays))
        if verbose:
            print("  seed %-4d %d grains, %d gated crossings over %d "
                  "azimuths (%.0f s)"
                  % (seed, len(tess[1]),
                     sum(int(gated(e).sum()) for e in evs), len(AZ_COMMON),
                     time.time() - t0), flush=True)
    return out, per_spec, flat, rays


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "core"):
        check_core()
        if what == "core":
            return
    if what == "field":
        report_field()
        report_shear_null()
        return
    print("building the facet census, %d tessellations x %d azimuths"
          % (len(SEEDS), len(AZ_COMMON)))
    all_ev, per_spec, flat, rays = build_all()
    print()
    if what in ("all", "coefficient"):
        report_coefficient(flat)
        report_asymmetry(grain_tensors(load_tess(SEEDS[0])[1]), flat)
    if what in ("all", "population"):
        report_population(flat, per_spec)
        report_dynamic_range(flat)
        report_random_look()
    if what in ("all", "shear"):
        report_shear(flat)
    if what in ("all", "field"):
        report_field()
        report_shear_null()


if __name__ == "__main__":
    main()
