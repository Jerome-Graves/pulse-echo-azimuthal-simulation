"""Shared machinery for the time-resolved (azimuth x time) coda test."""
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import j1

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
import forward as F                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "sweeps")))

C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
BAND = (0.8e6, 3.0e6)
CODA_W = (24e-6, 36e-6)
T0_SRC = 1.2 / F0                     # ricker delay used by azimuth_sweep
LAM = C_REF / F0
K = 2 * np.pi / LAM
DG = 17.4e-3                          # mean grain diameter
A_EL = 6.35e-3 / 2                    # transducer radius
DT_C = 20e-9                          # common resampling grid


# ───────────────────────────── measurement ────────────────────────────
def _sos(fs):
    return butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                  btype="band", output="sos")


def load_sweep(name, tgrid):
    """(az_deg, E[n_az, n_t] linear envelope on tgrid, e1_mean, t_backwall)."""
    d = os.path.join(OUT, name)
    az, rows, e1, tb = [], [], [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        env = np.abs(hilbert(sosfiltfilt(_sos(fs), tr)))
        t = np.arange(len(tr)) * dt
        k0, w = int(2 * DIA / C_REF * fs), int(3e-6 * fs)
        if k0 + w >= len(tr):
            continue
        seg = env[max(k0 - w, 0):k0 + w]
        e1.append(seg.max())
        tb.append((max(k0 - w, 0) + int(np.argmax(seg))) * dt)
        rows.append(np.interp(tgrid, t, env))
        az.append(int(f[2:5]))
    return np.array(az), np.array(rows), float(np.mean(e1)), np.array(tb)


# ─────────────────────────── pulse power kernel ───────────────────────
def pulse_power_kernel(dt=DT_C, half_us=2.0):
    """Power envelope of the band-passed ricker, normalised to unit sum."""
    fs_hi = 1e9
    nt = int(6e-6 * fs_hi)
    tc = np.arange(nt) / fs_hi
    a = (np.pi * F0 * (tc - 3e-6)) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    e = np.abs(hilbert(sosfiltfilt(_sos(fs_hi), w))) ** 2
    n = int(round(half_us * 1e-6 / dt))
    tk = np.arange(-n, n + 1) * dt + 3e-6
    k = np.interp(tk, tc, e)
    return k / k.sum()


# ───────────────────────────── ray geometry ───────────────────────────
def cone_rays(n_ray=600, th_max_deg=18.0, seed=0):
    """Ray offsets uniform in the transverse plane + two-way piston weight."""
    rng = np.random.default_rng(seed)
    tmax = np.tan(np.radians(th_max_deg))
    i = np.arange(n_ray) + 0.5
    rr = tmax * np.sqrt(i / n_ray)                     # uniform in area
    ph = i * np.pi * (3.0 - np.sqrt(5.0)) + rng.uniform(0, 2 * np.pi)
    th = np.arctan(rr)
    x = K * A_EL * np.sin(th)
    B = np.where(x < 1e-9, 1.0, 2 * j1(np.where(x < 1e-9, 1.0, x))
                 / np.where(x < 1e-9, 1.0, x))
    return rr, ph, B ** 4


def _march(lab, h, p_t, D, s):
    nx, ny, nz = lab.shape
    P = p_t[None, None, :] + s[None, :, None] * D[:, None, :]
    gi = np.rint((P[..., 0] + nx * h / 2) / h - 0.5).astype(np.int32)
    gj = np.rint((P[..., 1] + ny * h / 2) / h - 0.5).astype(np.int32)
    gk = np.rint((P[..., 2] + nz * h / 2) / h - 0.5).astype(np.int32)
    ok = ((gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
          & (gk >= 0) & (gk < nz))
    return np.where(ok, lab[np.clip(gi, 0, nx - 1), np.clip(gj, 0, ny - 1),
                            np.clip(gk, 0, nz - 1)], -1)


def beam_frame(az_deg):
    a = np.radians(az_deg)
    n = np.array([np.cos(a), np.sin(a), 0.0])
    t1 = np.array([-np.sin(a), np.cos(a), 0.0])
    t2 = np.array([0.0, 0.0, 1.0])
    return n, t1, t2


def facet_events(lab, axes, seeds, h, az_deg, s_max=0.088, ds=None,
                 n_ray=600, th_max_deg=18.0, use_dv=True, use_direc=True,
                 rayseed=0, const_v=False):
    """All grain-boundary crossings inside the diverging beam.

    Returns (t_two_way_incl_source_delay, incoherent power weight, s).
    """
    ds = ds if ds is not None else h / 2.0
    n, t1, t2 = beam_frame(az_deg)
    p_t = (DIA / 2.0) * n
    rr, ph, wb = cone_rays(n_ray, th_max_deg, rayseed)
    D = (-n[None, :] + (rr * np.cos(ph))[:, None] * t1
         + (rr * np.sin(ph))[:, None] * t2)
    D /= np.linalg.norm(D, axis=1, keepdims=True)

    s = (np.arange(int(s_max / ds)) + 0.5) * ds
    L = _march(lab, h, p_t, D, s)                       # (n_ray, n_s)

    cosang = np.abs(axes @ D.T)                         # (n_g, n_ray)
    vg = np.interp(np.arccos(np.clip(cosang, 0, 1)), F._PSI, F._VQP).T
    V = np.take_along_axis(vg, np.clip(L, 0, None), axis=1)
    V = np.where(L >= 0, V, C_REF)
    if const_v:
        tw = np.broadcast_to(s / C_REF, V.shape).copy()
    else:
        tw = np.cumsum(ds / V, axis=1) - 0.5 * ds / V   # cell-centre times

    A, Bb = L[:, :-1], L[:, 1:]
    cr = (A != Bb) & (A >= 0) & (Bb >= 0)
    ri, si = np.nonzero(cr)
    if ri.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    ia, ib = A[ri, si], Bb[ri, si]
    t_one = 0.5 * (tw[ri, si] + tw[ri, si + 1])
    va, vb = V[ri, si], V[ri, si + 1]
    R = (vb - va) / (vb + va) if use_dv else np.full(ia.shape, 0.02)
    nm = seeds[ib] - seeds[ia]
    nm /= np.linalg.norm(nm, axis=1, keepdims=True) + 1e-30
    if use_direc:
        ct = np.abs(np.einsum("ij,ij->i", nm, D[ri]))
        st = np.sqrt(np.clip(1 - ct ** 2, 0, 1))
        x = K * DG * st
        Dd = np.where(x < 1e-9, 1.0, 2 * j1(np.where(x < 1e-9, 1.0, x))
                      / np.where(x < 1e-9, 1.0, x)) ** 2
    else:
        Dd = np.ones(ia.shape)
    pw = (R ** 2) * Dd * wb[ri]
    return 2.0 * t_one + T0_SRC, pw, 0.5 * (s[si] + s[si + 1])


def axial_time(lab, axes, h, az_deg, ds=None):
    """One-way qP travel time along the beam axis across the whole disc."""
    ds = ds if ds is not None else h / 2.0
    n, _, _ = beam_frame(az_deg)
    p_t = (DIA / 2.0) * n
    s = (np.arange(int(DIA / ds)) + 0.5) * ds
    L = _march(lab, h, p_t, (-n)[None, :], s)[0]
    vg = np.interp(np.arccos(np.clip(np.abs(axes @ (-n)), 0, 1)),
                   F._PSI, F._VQP)
    inside = L >= 0
    return float(np.sum(ds / vg[np.clip(L, 0, None)][inside])), int(inside.sum())
