"""TEST 4, the physically achievable form of the inverse problem.

The 2-D image asks for something the geometry cannot give: with a 6.35 mm
element on a 100 mm rim the cross-range resolution is the beam width
(14-22 mm in the coda gate), i.e. one grain diameter, while the range
resolution is c/2BW ~ 0.9 mm.  So test the axis the instrument actually
resolves: for each azimuth, does the coda envelope rise at the two-way
TIMES at which the beam crosses grain boundaries?

Prediction, per azimuth: shoot a 21-ray fan filling the 8.9 deg cone,
find every Laguerre boundary crossing, convert its range to two-way time
with the SAME per-azimuth path-average velocity used on the measurement
(c_az = 2D/t1, measured, no ground truth), weight by the two-way beam
profile and optionally by the exact specular directivity of the facet
normal (seed_i - seed_j), bin, and convolve with the pulse envelope.

Everything a candidate tessellation contributes is seeds + weights, so the
true specimen and the 43 wrong ones go through identical code.

Scoring: double-centred (per-azimuth and per-time means removed from BOTH
matrices) Pearson correlation over the gate.
Nulls: (1) 43 wrong tessellations, (2) circular shift in azimuth,
(3) circular shift in time.
"""
import os
import sys

import numpy as np
from scipy.special import j1

import t4_common as T

SCR = T.SCR
HVOX = 0.5e-3                 # label volume for ray marching
NLAT = 5                      # -> 21 rays in the cone
DS = 0.5e-3                   # along-beam step
S0, S1 = 6e-3, 96e-3          # along-beam range covered
TBIN = 0.1e-6
TSIG = 0.20e-6                # pulse envelope sigma
GATE = (24e-6, 36e-6)
GATE_FULL = (12e-6, 48e-6)
NULL_SEEDS = [17, 23, 41] + list(range(100, 140))

SWEEPS = [("girdle_perp_ppw8", 11, 1), ("singlemax_ppw8", 11, 1),
          ("girdle_par", 11, 1), ("kappa8_seed17", 17, 1),
          ("oos_seed23", 23, 4), ("iso_gcal", 41, 4),
          ("rigid_seed11", 11, 4)]


# ───────────────────────────── label volume ───────────────────────────────
def label_volume(seed):
    p = os.path.join(SCR, f"t4vol_s{seed}.npz")
    if os.path.exists(p):
        with np.load(p) as z:
            return z["lab"], z["seeds"]
    b = T.build_seeds(8.0, -8.0, (1.0, 0.0, 0.0), seed)
    sd, wt = b["seeds"], b["weights"]
    R = T.DIA / 2
    nx = int(round(T.DIA / HVOX))
    nz = int(round(T.THK / HVOX))
    x = (np.arange(nx) - (nx - 1) / 2) * HVOX
    z = (np.arange(nz) - (nz - 1) / 2) * HVOX
    X, Y = np.meshgrid(x, x, indexing="ij")
    ins = (X ** 2 + Y ** 2) <= R ** 2
    lab = np.full((nx, nx, nz), -1, np.int16)
    P = np.stack([X[ins], Y[ins]], 1)
    d_xy = ((P[:, None, :] - sd[None, :, :2]) ** 2).sum(-1) - wt[None, :]
    for k in range(nz):
        d2 = d_xy + (z[k] - sd[:, 2])[None, :] ** 2
        pl = np.full((nx, nx), -1, np.int16)
        pl[ins] = d2.argmin(1).astype(np.int16)
        lab[:, :, k] = pl
    np.savez_compressed(p, lab=lab, seeds=sd)
    return lab, sd


def ray_geometry(az_deg):
    """Ray sample points, back-directions and path lengths for each azimuth."""
    uu = np.linspace(-1, 1, NLAT)
    U, W = np.meshgrid(uu, uu, indexing="ij")
    keep = U ** 2 + W ** 2 <= 1.0
    U, W = U[keep], W[keep]
    rf = np.hypot(U, W)
    wray = np.exp(-2.0 * rf ** 2)          # two-way Gaussian beam profile
    s = np.arange(S0, S1, DS)
    sec = np.sqrt(1.0 + (rf * np.tan(T.HALF)) ** 2)
    L = s[None, :] * sec[:, None]          # ray path length (n_ray, n_s)
    R = T.DIA / 2
    out = []
    for a in az_deg:
        th = np.radians(float(a))
        n = np.array([np.cos(th), np.sin(th), 0.0])
        e1 = np.array([-np.sin(th), np.cos(th), 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
        rad = (s * np.tan(T.HALF))[None, :]
        P = (R * n - s[:, None] * n)[None, :, :] \
            + (rad * U[:, None])[:, :, None] * e1 \
            + (rad * W[:, None])[:, :, None] * e2
        q = R * n
        u = q[None, None, :] - P
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        out.append((P, u))
    return out, L, wray


def predict(lab, sd, geo, L, wray, c_az, tgrid, spec=True):
    """(n_az, n_t) predicted boundary-echo expectation."""
    nx, ny, nz = lab.shape
    n_t = len(tgrid)
    out = np.zeros((len(geo), n_t))
    for q, (P, u) in enumerate(geo):
        gi = np.rint(P[..., 0] / HVOX + (nx - 1) / 2).astype(np.int32)
        gj = np.rint(P[..., 1] / HVOX + (ny - 1) / 2).astype(np.int32)
        gk = np.rint(P[..., 2] / HVOX + (nz - 1) / 2).astype(np.int32)
        ok = ((gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
              & (gk >= 0) & (gk < nz))
        Lb = np.where(ok, lab[np.clip(gi, 0, nx - 1), np.clip(gj, 0, ny - 1),
                              np.clip(gk, 0, nz - 1)], -1)
        A, B = Lb[:, :-1], Lb[:, 1:]
        cr = (A != B) & (A >= 0) & (B >= 0)
        ir, isamp = np.nonzero(cr)
        if ir.size == 0:
            continue
        ia, ib = A[cr], B[cr]
        t2 = 2.0 * 0.5 * (L[ir, isamp] + L[ir, isamp + 1]) / c_az[q]
        w = wray[ir]
        if spec:
            nm = sd[ia] - sd[ib]
            nm /= np.linalg.norm(nm, axis=1, keepdims=True) + 1e-30
            uu = u[ir, isamp]
            ct = np.abs(np.einsum("ij,ij->i", nm, uu))
            st = np.sqrt(np.clip(1 - ct ** 2, 0, 1))
            x = T.K * T.DG * st
            d = np.where(x < 1e-9, 1.0,
                         2 * j1(np.where(x < 1e-9, 1.0, x))
                         / np.where(x < 1e-9, 1.0, x))
            w = w * d ** 2
        h, _ = np.histogram(t2, bins=len(tgrid),
                            range=(tgrid[0] - TBIN / 2,
                                   tgrid[-1] + TBIN / 2), weights=w)
        out[q] = h
    # pulse-envelope smoothing
    k = np.exp(-0.5 * (np.arange(-4 * TSIG / TBIN, 4 * TSIG / TBIN + 1)
                       * TBIN / TSIG) ** 2)
    k /= k.sum()
    return np.array([np.convolve(r, k, mode="same") for r in out])


def dcent(M):
    M = M - M.mean(axis=0, keepdims=True)
    return M - M.mean(axis=1, keepdims=True)


def corr(A, B):
    a, b = dcent(A).ravel(), dcent(B).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)
                                 + 1e-30))


def main():
    tgrid = np.arange(GATE_FULL[0], GATE_FULL[1], TBIN)
    inb = {g: (tgrid >= g[0]) & (tgrid <= g[1])
           for g in (GATE, GATE_FULL)}
    cands = [11] + NULL_SEEDS
    vols = {}
    for s in cands:
        vols[s] = label_volume(s)
        print(f"  volume seed {s} ready", flush=True)

    print(f"\n{'sweep':<18}{'own':>4}{'gate':>7}{'spec':>6}"
          f"{'r_own':>8}{'rank/N':>9}{'p_wrong':>9}"
          f"{'p_azshift':>11}{'p_tshift':>10}")
    summary = []
    for nm, own, sub in SWEEPS:
        az, ana, dt, t1 = T.load_sweep(nm)
        if sub > 1:
            az, ana, t1 = az[::sub], ana[::sub], t1[::sub]
        env = np.abs(ana)
        m = env.mean(0)
        kk = max(int(3e-6 / dt) | 1, 3)
        hn = np.hanning(kk)
        hn /= hn.sum()
        ms = np.maximum(np.convolve(m, hn, mode="same"), 1e-3 * m.max())
        E = np.array([np.interp(tgrid, np.arange(len(r)) * dt, r / ms)
                      for r in env])
        c_az = 2.0 * T.DIA / t1
        geo, L, wray = ray_geometry(az)
        for spec in (True, False):
            P = {s: predict(vols[s][0], vols[s][1], geo, L, wray, c_az,
                            tgrid, spec) for s in cands}
            for g in (GATE, GATE_FULL):
                sel = inb[g]
                Em = E[:, sel]
                r = {s: corr(Em, P[s][:, sel]) for s in cands}
                key = own if own in r else 11
                vals = np.array([r[s] for s in cands])
                rank = int(np.sum(vals >= r[key]))
                pw = (1 + np.sum(np.array([r[s] for s in cands
                                           if s != key]) >= r[key])) \
                    / len(vals)
                # azimuth circular shift
                rs = [corr(np.roll(Em, k, axis=0), P[key][:, sel])
                      for k in range(len(az))]
                paz = float(np.mean(np.array(rs) >= r[key]))
                # time circular shift (steps of 1 us, excluding 0)
                st = int(1e-6 / TBIN)
                nsh = Em.shape[1] // st
                rt = [corr(np.roll(Em, k * st, axis=1), P[key][:, sel])
                      for k in range(1, nsh)]
                pt = (1 + np.sum(np.array(rt) >= r[key])) / (len(rt) + 1)
                gl = "coda" if g == GATE else "wide"
                print(f"{nm:<18}{own:>4}{gl:>7}{str(spec):>6}"
                      f"{r[key]:>8.3f}{rank:>5}/{len(vals):<3}{pw:>9.3f}"
                      f"{paz:>11.3f}{pt:>10.3f}", flush=True)
                summary.append((nm, own, gl, spec, r[key], rank, pw,
                                paz, pt, vals))
    np.save(os.path.join(SCR, "t4_range.npy"),
            np.array(summary, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
