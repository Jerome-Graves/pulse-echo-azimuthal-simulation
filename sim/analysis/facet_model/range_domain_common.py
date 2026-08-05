"""TEST 4 shared machinery: builds, coda traces, boundary maps.

Frame convention (derived, not assumed):
  rotation_test.rotated_grid puts solver seed = seeds @ Rm = R(-rot) * seed,
  i.e. the map specimen->solver is p -> R(-rot) p.  The probe is fixed on
  the solver +x rim, so in the SPECIMEN frame the probe sits at
  (R cos rot, R sin rot, 0) and fires along -(cos rot, sin rot, 0).
  Matches cone_specular / decisive (n = [cos a, sin a, 0]) and the
  post-2026-07-31 arc_backproject (+a).
"""
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

SIM = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim")))
sys.path.insert(0, SIM)
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vendor"))))

OUT = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "out", "sweeps")))
SCR = os.path.dirname(os.path.abspath(__file__))

C_REF, F0, DIA, THK = 3850.0, 2.0e6, 0.100, 0.035
LAM = C_REF / F0                       # 1.925 mm
BAND = (0.8e6, 3.0e6)
CODA_W = (24e-6, 36e-6)
HALF = np.radians(8.9)                 # far-field divergence half-angle
A_EL = 6.35e-3 / 2.0
DG = 17.4e-3
K = 2 * np.pi / LAM


# ───────────────────────────── specimen builds ────────────────────────────
def build_seeds(ppw, kappa, axis, seed):
    """Seeds/weights/axes only (no label volume kept) - for null specimens."""
    p = os.path.join(SCR, f"t4sw_s{seed}_p{ppw:g}_k{kappa:g}.npz")
    if os.path.exists(p):
        with np.load(p) as z:
            return dict(seeds=z["seeds"], weights=z["weights"],
                        axes=z["axes"], h=float(z["h"]))
    from specimen import DiskSpecimen
    h = C_REF / F0 / ppw
    b = DiskSpecimen(diameter_m=DIA, thickness_m=THK, n_grains=100,
                     size_cv=0.35, concentration=kappa, spatial_corr=0.0,
                     fabric_axis=tuple(axis), seed=seed).build(h)
    d = dict(seeds=np.asarray(b["seeds"], float),
             weights=np.asarray(b["weights"], float),
             axes=np.asarray(b["axes"], float), h=h)
    np.savez_compressed(p, **d)
    return d


def build_spec(ppw, kappa, axis, seed, cache=True):
    """DiskSpecimen build with an on-disk cache in the scratchpad."""
    from specimen import DiskSpecimen
    h = C_REF / F0 / ppw
    tag = f"t4build_s{seed}_p{ppw:g}_k{kappa:g}.npz"
    p = os.path.join(SCR, tag)
    if cache and os.path.exists(p):
        with np.load(p) as z:
            return dict(labels=z["labels"], axes=z["axes"], seeds=z["seeds"],
                        weights=z["weights"], h=float(z["h"]))
    b = DiskSpecimen(diameter_m=DIA, thickness_m=THK, n_grains=100,
                     size_cv=0.35, concentration=kappa, spatial_corr=0.0,
                     fabric_axis=tuple(axis), seed=seed).build(h)
    d = dict(labels=b["labels"], axes=np.asarray(b["axes"], float),
             seeds=np.asarray(b["seeds"], float),
             weights=np.asarray(b["weights"], float), h=h)
    if cache:
        np.savez_compressed(p, **d)
    return d


# ───────────────────────────── trace handling ─────────────────────────────
def load_sweep(name):
    """az (deg), filtered analytic traces on a COMMON time base, dt, E1 time.

    dt varies by ~0.4 % between azimuths (the CFL step follows the rotated
    stiffness field), so the traces are band-passed at their own fs and then
    linearly resampled onto one grid.  At fs ~ 96 MHz for a <=3 MHz band that
    is a ~1e-4 amplitude operation.
    """
    d = os.path.join(OUT, name)
    files = sorted(f for f in os.listdir(d)
                   if f.startswith("az") and f.endswith(".npz"))
    raw, dts, t1, az = [], [], [], []
    for f in files:
        with np.load(os.path.join(d, f)) as z:
            raw.append(np.asarray(z["trace"], float).ravel())
            dts.append(float(z["dt"]))
            t1.append(float(z["t1_s"]))
        az.append(int(f[2:5]))
    dt0 = float(np.median(dts))
    nt = int(min(len(r) * dd for r, dd in zip(raw, dts)) / dt0)
    tc = np.arange(nt) * dt0
    ana = np.empty((len(raw), nt), complex)
    for i, (tr, dd) in enumerate(zip(raw, dts)):
        fs = 1.0 / dd
        sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                     btype="band", output="sos")
        # filter the WHOLE trace, then gate (never the other way round)
        y = hilbert(sosfiltfilt(sos, tr))
        t = np.arange(len(tr)) * dd
        ana[i] = (np.interp(tc, t, y.real)
                  + 1j * np.interp(tc, t, y.imag))
    return np.array(az), ana, dt0, np.array(t1)


# ────────────────────── Laguerre evaluation at arbitrary points ───────────
def laguerre_labels(P, seeds, weights, chunk=200000):
    """Exact all-seeds power-diagram argmin at points P (n,3)."""
    out = np.empty(len(P), np.int32)
    for i in range(0, len(P), chunk):
        Q = P[i:i + chunk]
        d2 = (((Q[:, None, :] - seeds[None, :, :]) ** 2).sum(-1)
              - weights[None, :])
        out[i:i + chunk] = d2.argmin(1)
    return out


def boundary_map_2d(seeds, weights, pix_mm, z_list, R=DIA / 2,
                    z_weight=None):
    """2-D boundary-density map: fraction of z-planes in which the pixel is
    within one pixel of a Laguerre boundary, weighted by z_weight.

    Evaluating the tessellation analytically (seeds+weights) rather than
    resampling a rasterised volume means every specimen, true or wrong, is
    treated identically on the SAME pixel grid.
    """
    n = int(round(2 * R / (pix_mm * 1e-3)))
    ax = (np.arange(n) + 0.5) * (2 * R / n) - R
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    inside = (X ** 2 + Y ** 2) <= R ** 2
    acc = np.zeros((n, n))
    wsum = 0.0
    zw = np.ones(len(z_list)) if z_weight is None else np.asarray(z_weight)
    for zi, z in enumerate(z_list):
        P = np.stack([X[inside], Y[inside],
                      np.full(inside.sum(), z)], axis=1)
        lab = np.full((n, n), -1, np.int32)
        lab[inside] = laguerre_labels(P, seeds, weights)
        b = np.zeros((n, n), bool)
        b[:-1, :] |= (lab[:-1, :] != lab[1:, :]) & (lab[:-1, :] >= 0) \
            & (lab[1:, :] >= 0)
        b[1:, :] |= (lab[:-1, :] != lab[1:, :]) & (lab[:-1, :] >= 0) \
            & (lab[1:, :] >= 0)
        b[:, :-1] |= (lab[:, :-1] != lab[:, 1:]) & (lab[:, :-1] >= 0) \
            & (lab[:, 1:] >= 0)
        b[:, 1:] |= (lab[:, :-1] != lab[:, 1:]) & (lab[:, :-1] >= 0) \
            & (lab[:, 1:] >= 0)
        acc += zw[zi] * b
        wsum += zw[zi]
    return ax, acc / max(wsum, 1e-30), inside


# ───────────────────────────── facet extraction ───────────────────────────
def facets_from_labels(labels, seeds, h):
    """Every grain-boundary voxel FACE: position, exact normal, pair.

    Normal of the i|j Laguerre boundary is exactly (seed_i - seed_j).
    """
    nx, ny, nz = labels.shape
    pos, pair = [], []
    ax = ((np.arange(nx) - (nx - 1) / 2) * h,
          (np.arange(ny) - (ny - 1) / 2) * h,
          (np.arange(nz) - (nz - 1) / 2) * h)
    for d in range(3):
        A = np.take(labels, np.arange(labels.shape[d] - 1), axis=d)
        B = np.take(labels, np.arange(1, labels.shape[d]), axis=d)
        m = (A != B) & (A >= 0) & (B >= 0)
        idx = np.nonzero(m)
        c = []
        for k in range(3):
            v = ax[k][idx[k]]
            if k == d:
                v = v + 0.5 * h
            c.append(v)
        pos.append(np.stack(c, 1))
        pair.append(np.stack([A[m], B[m]], 1))
    pos = np.concatenate(pos, 0)
    pair = np.concatenate(pair, 0).astype(np.int64)
    lo = np.minimum(pair[:, 0], pair[:, 1])
    hi = np.maximum(pair[:, 0], pair[:, 1])
    nrm = seeds[lo] - seeds[hi]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    return pos, np.stack([lo, hi], 1), nrm


def beam_radius(d_par):
    """1/e-ish beam radius: element radius plus far-field divergence."""
    return A_EL + np.maximum(d_par, 0.0) * np.tan(HALF)
