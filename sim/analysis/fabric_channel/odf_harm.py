"""General spherical-harmonic ODF description of the project's two
observables, reusing fit_fabric's quadrature and forward's qP table.

Conventions (deliberately identical to the repo):
  * quadrature  = fit_fabric.fib_sphere(1500), EQUAL-AREA weights
  * qP velocity = forward._PSI / forward._VQP, interpolated at
                  arccos(|n.b|)  exactly as fit_fabric.odf_moments does
  * E[R^2]      = Var(v) / (2 vbar^2)   (fit_fabric.odf_moments[0])
  * E[1/v]      = sum w/v              (fit_fabric.odf_moments[1])

ODF parameterisation.  f(n) = f_base(n) + sum_k c_k B_k(n) with
B_k = sqrt(4pi) * real Y_lm, so B_00 == 1 and (1/4pi) int B_k B_j = delta.
f_base is normalised to unit MEAN over the quadrature, so a coefficient
c_k is on the same footing as the isotropic "1".  Only EVEN l matter
(c-axes are headless); l = 2 and 4 are carried, 14 free coefficients.
"""
import sys

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")

import fit_fabric as FF      # noqa: E402
import forward as F          # noqa: E402

DIRS = FF._DIRS              # (N,3) quasi-uniform, equal area
N = len(DIRS)

# ── real spherical-harmonic basis, 4pi-normalised (B_00 = 1) ────────────
_TH = np.arccos(np.clip(DIRS[:, 2], -1, 1))
_PH = np.arctan2(DIRS[:, 1], DIRS[:, 0])

LM = [(l, m) for l in (2, 4) for m in range(-l, l + 1)]
LABEL = [f"c{l}{'m' if m < 0 else 'p'}{abs(m)}" for l, m in LM]
NC = len(LM)


def _realY(l, m, th, ph):
    if m == 0:
        return np.real(sph_harm_y(l, 0, th, ph))
    a = abs(m)
    Y = sph_harm_y(l, a, th, ph)
    s = np.sqrt(2.0) * (-1.0) ** a
    return s * (np.real(Y) if m > 0 else np.imag(Y))


def basis(th, ph):
    """(n_k, n_pts) matrix of B_k = sqrt(4pi) Y^R_lm."""
    return np.array([np.sqrt(4 * np.pi) * _realY(l, m, th, ph)
                     for l, m in LM])


B = basis(_TH, _PH)                       # (NC, N) on the quadrature


def basis_at(dirs):
    d = np.atleast_2d(np.asarray(dirs, float))
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    th = np.arccos(np.clip(d[:, 2], -1, 1))
    ph = np.arctan2(d[:, 1], d[:, 0])
    return basis(th, ph)


# ── base fabrics ────────────────────────────────────────────────────────
def watson_f(axis, kappa):
    """Watson density on the quadrature, normalised to unit mean.
    kappa > 0 single maximum, kappa < 0 girdle (specimen.sample_watson)."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    ex = kappa * (DIRS @ axis) ** 2
    f = np.exp(ex - ex.max())
    return f * (N / f.sum())


def f_from_axes(axes, sigma_deg=8.0):
    """Kernel-smoothed empirical ODF of a finite set of sampled c-axes."""
    a = np.asarray(axes, float)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    k = 1.0 / np.radians(sigma_deg) ** 2
    f = np.zeros(N)
    c = np.abs(DIRS @ a.T)                       # (N, n_axes)
    f = np.exp(k * (c ** 2 - 1.0)).sum(axis=1)
    return f * (N / f.sum())


def coeffs_of(f):
    """Harmonic coefficients of a density (unit-mean) on the quadrature."""
    return (B * f).mean(axis=1)


ISO = np.ones(N)


# ── observables ─────────────────────────────────────────────────────────
def _v_of_beam(beam):
    """qP speed of every quadrature orientation for one beam direction -
    fit_fabric.odf_moments' exact expression."""
    b = np.asarray(beam, float)
    b = b / np.linalg.norm(b)
    cos = np.clip(np.abs(DIRS @ b), 0, 1)
    return np.interp(np.arccos(cos), F._PSI, F._VQP)


def observables(f, beams):
    """(tof_slowness E[1/v], coda_dB 10log10 E[R^2]) per beam."""
    w = f / f.sum()
    inv, er2 = [], []
    for b in beams:
        v = _v_of_beam(b)
        Lp = w @ v
        Lq = w @ (v * v)
        er2.append(Lq / (2 * Lp * Lp) - 0.5)
        inv.append(w @ (1.0 / v))
    er2 = np.asarray(er2)
    return np.asarray(inv), 10 * np.log10(np.maximum(er2, 1e-300))


def jacobians(f, beams):
    """Analytic d(observable)/d(c_k) at the base density f.

    f is unit-mean on the quadrature so S = sum f = N for every fabric,
    which puts coefficients on a common footing across base fabrics.
    Returns (J_tof (n_beam, NC) in s/m, J_coda (n_beam, NC) in dB).
    """
    S = f.sum()
    w = f / S
    T = B.sum(axis=1)                                  # (NC,)
    Jt = np.empty((len(beams), NC))
    Jc = np.empty((len(beams), NC))
    for i, b in enumerate(beams):
        v = _v_of_beam(b)
        u = 1.0 / v
        Lp, Lq, Lu = w @ v, w @ (v * v), w @ u
        # dL[g]/dc_k = (sum_i g_i B_ki - L[g] * T_k) / S
        dLp = (B @ v - Lp * T) / S
        dLq = (B @ (v * v) - Lq * T) / S
        dLu = (B @ u - Lu * T) / S
        er2 = Lq / (2 * Lp * Lp) - 0.5
        der2 = dLq / (2 * Lp * Lp) - Lq * dLp / (Lp ** 3)
        Jt[i] = dLu
        Jc[i] = (10.0 / np.log(10.0)) * der2 / er2
    return Jt, Jc


def inplane_beams(n_az=60, start=0.0):
    az = np.radians(start + np.arange(n_az) * 360.0 / n_az)
    return np.c_[np.cos(az), np.sin(az), np.zeros(n_az)], np.degrees(az)


def set_quadrature(n):
    """Refine the spherical quadrature. The repo uses 1500; the RANK
    analysis needs the quadrature's own residual (int Y_lm dOmega != 0 on
    a finite Fibonacci set) pushed well below the smallest PHYSICAL
    singular value, otherwise numerical rank counts quadrature noise."""
    global DIRS, N, _TH, _PH, B, ISO
    DIRS = FF.fib_sphere(n)
    N = len(DIRS)
    _TH = np.arccos(np.clip(DIRS[:, 2], -1, 1))
    _PH = np.arctan2(DIRS[:, 1], DIRS[:, 0])
    B = basis(_TH, _PH)
    ISO = np.ones(N)
