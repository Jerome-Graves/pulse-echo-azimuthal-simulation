"""Statistics for the time-resolved azimuth x time coda test.

Everything is scored with 2-D circular-shift nulls (azimuth AND time), and
the effective sample size is estimated from the 2-D autocorrelations of the
two fields rather than assumed.
"""
import os

import numpy as np

import _t2_common as C

EPS = 1e-30


def load(sweep, tag):
    z = np.load(os.path.join(C.HERE, f"t2p_{sweep}__{tag}.npz"))
    TG = z["tgrid"]
    g = (TG >= C.CODA_W[0]) & (TG < C.CODA_W[1])
    M = (z["meas"][:, g] / float(z["e1"])) ** 2
    P = {v: z[f"p_{v}"][:, g] for v in ("full", "geom", "nodirec", "flat")}
    return z["az"], TG[g], M, P


# ───────────────────────────── centring ───────────────────────────────
def _harm_basis(n):
    t = np.arange(n) / n * 2 * np.pi
    return np.column_stack([np.ones(n), np.cos(t), np.sin(t),
                            np.cos(2 * t), np.sin(2 * t),
                            np.cos(3 * t), np.sin(3 * t),
                            np.cos(4 * t), np.sin(4 * t)])


def centre(X, mode):
    if mode == "raw":
        return X - X.mean()
    if mode == "col":                       # remove the depth profile
        return X - X.mean(0, keepdims=True)
    if mode == "double":                    # + remove per-azimuth level
        Y = X - X.mean(0, keepdims=True)
        return Y - Y.mean(1, keepdims=True)
    if mode == "harm":
        # strip mean + azimuthal harmonics k=1..4 at EVERY time sample,
        # then remove the per-azimuth level: kills any texture template
        A = _harm_basis(X.shape[0])
        Y = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]
        return Y - Y.mean(1, keepdims=True)
    raise ValueError(mode)


def corr(a, b):
    a, b = a.ravel(), b.ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def score(M, P, mode, log=True):
    m = 10 * np.log10(M + EPS) if log else M
    p = 10 * np.log10(P + EPS) if log else P
    return corr(centre(m, mode), centre(p, mode))


# ─────────────────────────── shift nulls ──────────────────────────────
def shift_null(M, P, mode, log=True, kind="2d", max_null=4000, rng=None):
    """p = fraction of circular shifts of the MEASUREMENT scoring >= |r0|."""
    na, nt = M.shape
    r0 = score(M, P, mode, log)
    if kind == "az":
        shifts = [(k, 0) for k in range(na)]
    elif kind == "time":
        shifts = [(0, j) for j in range(nt)]
    else:
        rng = rng or np.random.default_rng(0)
        shifts = [(0, 0)]
        tot = na * nt
        if tot <= max_null:
            shifts = [(k, j) for k in range(na) for j in range(nt)]
        else:
            ks = rng.integers(0, na, max_null)
            js = rng.integers(0, nt, max_null)
            shifts = [(0, 0)] + list(zip(ks, js))
    rs = np.array([score(np.roll(np.roll(M, k, 0), j, 1), P, mode, log)
                   for k, j in shifts])
    return r0, float(np.mean(np.abs(rs) >= abs(r0))), len(shifts)


# ────────────────────── effective degrees of freedom ──────────────────
def acf2(X):
    X = X - X.mean()
    F = np.fft.rfft2(X, s=(2 * X.shape[0], 2 * X.shape[1]))
    A = np.fft.irfft2(np.abs(F) ** 2, s=(2 * X.shape[0], 2 * X.shape[1]))
    A = A[:X.shape[0], :X.shape[1]]
    return A / A[0, 0]


def n_eff(A, B):
    """Bartlett/Bretherton: N_eff = N / sum_lags rho_A rho_B."""
    ra, rb = acf2(A), acf2(B)
    na, nt = A.shape
    wa = 1.0 - np.arange(na) / na
    wt = 1.0 - np.arange(nt) / nt
    w = np.outer(wa, wt)
    # two-sided: lags in +-az and +-t; the rfft acf is one quadrant, and the
    # field is real so rho(-l) = rho(l) in each axis separately for the
    # separable estimate.  Use 4x the quadrant minus the double counted axes.
    s = 4 * np.sum(w * ra * rb) - 2 * np.sum(w[0] * ra[0] * rb[0]) \
        - 2 * np.sum(w[:, 0] * ra[:, 0] * rb[:, 0]) + ra[0, 0] * rb[0, 0]
    return float(na * nt / max(s, 1.0))


def t_p(r, n):
    from scipy import stats
    if n <= 3:
        return 1.0
    t = r * np.sqrt((n - 2) / max(1 - r ** 2, 1e-15))
    return float(2 * stats.t.sf(abs(t), n - 2))
