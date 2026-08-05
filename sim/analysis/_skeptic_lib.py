"""Independent re-implementation of the range-domain boundary predictor.

Differences from range_domain_test.py (deliberate, so this is a genuine second route):
  * NO rasterised 0.5 mm label volume and NO nearest-voxel lookup.  The
    Laguerre power metric is evaluated EXACTLY at every sample point.
  * 0.1 mm along-ray sampling (5x finer than their voxel pitch).
  * 19-ray annular fan (2 rings + axis) instead of a 5x5 square grid.
  * crossings placed at the exact bracketing midpoint.
Same physics: two-way time 2L/c_az with c_az = 2D/t1 measured, two-way
Gaussian beam weight, optional exact facet directivity |2J1(x)/x|^2.
"""
import os as _os
import sys as _sys
# helper modules shared with sibling directories
for _d in ('facet_model',):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import os, sys
import numpy as np
from scipy.special import j1
import range_domain_common as T

SCR = T.SCR
DS = 0.1e-3
S0, S1 = 6e-3, 96e-3
TBIN = 0.1e-6
TSIG = 0.20e-6
R = T.DIA / 2

def fan(nring=2, nphi=9):
    """19-ray annular fan: axis + 2 rings."""
    us, ws = [(0.0, 0.0)], [1.0]
    for k in range(1, nring + 1):
        rr = k / nring
        for m in range(nphi):
            ph = 2 * np.pi * m / nphi + 0.3 * k
            us.append((rr * np.cos(ph), rr * np.sin(ph)))
            ws.append(np.exp(-2.0 * rr ** 2))
    U = np.array([u[0] for u in us]); W = np.array([u[1] for u in us])
    return U, W, np.array(ws)

U, W, WRAY = fan()
RF = np.hypot(U, W)
SS = np.arange(S0, S1, DS)
SEC = np.sqrt(1.0 + (RF * np.tan(T.HALF)) ** 2)
LPATH = SS[None, :] * SEC[:, None]          # (n_ray, n_s)

def ray_points(a_deg, sign=+1.0):
    th = np.radians(float(a_deg))
    n = np.array([np.cos(th), np.sin(th), 0.0])
    e1 = np.array([-np.sin(th), np.cos(th), 0.0])
    e2 = np.array([0.0, 0.0, 1.0])
    rad = (SS * np.tan(T.HALF))[None, :]
    P = (R * n - SS[:, None] * n)[None, :, :] \
        + (rad * U[:, None])[:, :, None] * e1 \
        + (rad * W[:, None])[:, :, None] * e2
    q = R * n
    u = q[None, None, :] - P
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    return P, u

def power_labels(P, sd, wt):
    """Exact Laguerre label at each point; -1 outside the disc/slab."""
    sh = P.shape[:-1]
    Q = P.reshape(-1, 3)
    ins = (Q[:, 0] ** 2 + Q[:, 1] ** 2 <= R ** 2) & (np.abs(Q[:, 2]) <= T.THK / 2)
    lab = np.full(len(Q), -1, np.int32)
    Qi = Q[ins]
    out = np.empty(len(Qi), np.int32)
    CH = 60000
    for i in range(0, len(Qi), CH):
        A = Qi[i:i + CH]
        d2 = ((A[:, None, :] - sd[None, :, :]) ** 2).sum(-1) - wt[None, :]
        out[i:i + CH] = d2.argmin(1)
    lab[ins] = out
    return lab.reshape(sh)

def predict_az(a_deg, sd, wt, c_az, tgrid, spec=True, sign=+1.0):
    P, u = ray_points(a_deg * sign)
    if sign < 0:   # mirror control: geometry from -a, times unchanged
        pass
    lab = power_labels(P, sd, wt)
    A, B = lab[:, :-1], lab[:, 1:]
    cr = (A != B) & (A >= 0) & (B >= 0)
    ir, isamp = np.nonzero(cr)
    h = np.zeros(len(tgrid))
    if ir.size:
        ia, ib = A[cr], B[cr]
        t2 = 2.0 * 0.5 * (LPATH[ir, isamp] + LPATH[ir, isamp + 1]) / c_az
        w = WRAY[ir]
        if spec:
            nm = sd[ia] - sd[ib]
            nm /= np.linalg.norm(nm, axis=1, keepdims=True) + 1e-30
            uu = u[ir, isamp]
            ct = np.abs(np.einsum("ij,ij->i", nm, uu))
            st = np.sqrt(np.clip(1 - ct ** 2, 0, 1))
            x = T.K * T.DG * st
            d = np.where(x < 1e-9, 1.0, 2 * j1(np.maximum(x, 1e-9)) / np.maximum(x, 1e-9))
            w = w * d ** 2
        h, _ = np.histogram(t2, bins=len(tgrid),
                            range=(tgrid[0] - TBIN / 2, tgrid[-1] + TBIN / 2),
                            weights=w)
    return h

def smooth_t(M):
    k = np.exp(-0.5 * (np.arange(-4 * TSIG / TBIN, 4 * TSIG / TBIN + 1) * TBIN / TSIG) ** 2)
    k /= k.sum()
    return np.array([np.convolve(r, k, mode="same") for r in M])

def predict(az, sd, wt, c_az, tgrid, spec=True, sign=+1.0):
    M = np.array([predict_az(a, sd, wt, c_az[q], tgrid, spec, sign)
                  for q, a in enumerate(az)])
    return smooth_t(M)

def measured(nm, tgrid, sub=1):
    az, ana, dt, t1 = T.load_sweep(nm)
    if sub > 1:
        az, ana, t1 = az[::sub], ana[::sub], t1[::sub]
    env = np.abs(ana)
    m = env.mean(0)
    kk = max(int(3e-6 / dt) | 1, 3)
    hn = np.hanning(kk); hn /= hn.sum()
    ms = np.maximum(np.convolve(m, hn, mode="same"), 1e-3 * m.max())
    E = np.array([np.interp(tgrid, np.arange(len(r)) * dt, r / ms) for r in env])
    return az, E, t1, dt

def dcent(M):
    M = M - M.mean(axis=0, keepdims=True)
    return M - M.mean(axis=1, keepdims=True)

def corr(A, B):
    a, b = dcent(A).ravel(), dcent(B).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

def cand_sw(seed):
    b = T.build_seeds(8.0, -8.0, (1.0, 0.0, 0.0), seed)
    return b["seeds"], b["weights"]
