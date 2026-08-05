"""GRAIN-PARAMETERIZED CONTINUOUS INVERSION (v5) - Jerome's design
arc, final form of the 2026-07-29 session:

  "our grid has to represent the texture and odd shaped grains"
+ "there could be general assumptions about the ice fabric we use"
+ literature verdict: invert only what the data supports
= THE STATE IS ~100 GRAINS, NOT 1600 CELLS:
      seeds  (N_g, 2)  Voronoi seed positions (the partition)
      angs   (N_g,)    one in-plane c-axis angle per grain
Cells (2.5 mm) exist only as the rasterization for the forward model;
their angle field is angs[nearest-seed]. Pie-wedge null-space
structure is unrepresentable by construction; walls are grain walls.

PRIORS (replace v4's TV/edge/anchor - the parametrization is the
prior): angles ~ Watson ODF measured by the validated global fit
(penalty kappa_hat * sin^2(ang - axis_hat) per grain); seeds repel
below ~half the mean grain spacing and stay in the disk; grain count
fixed at the specimen's n_grains.

DATA CHANNELS (all doctrine-masked, all nuisance-fitted per iteration):
  1. ToF(az)          - leading-edge picks, huberised +-0.6 us
  2. clean-window coda- 24-36 us, 1 us bins, axial+lateral boundary
                        R^2 model, frozen floor, gain+trend nuisance
  3. E1 FADE (NEW)    - per-az E1 amplitude from e1_fade.npz
                        (trace_lab.py); model = c_f + b_f * X(az),
                        X = full-path sum |R| (misorientation contrast
                        the beam crossed - the validated strong
                        disorder observable, twin-calibrated through
                        the free b_f); walk-off band excluded.

OPTIMISATION per loop: damped Gauss-Newton on the grain ANGLES
(cell Jacobian chained through the partition indicator); every
SEED_EVERY-th iteration, greedy trial displacement of a random batch
of seeds accepted on the full objective. All forward evaluation is
the vectorised (CUDA when available) _Pack batch engine.

Outputs keep the GUI contract: alpha_grid (rasterized), theta_grid /
theta_gauge vs truth, overlays for ToF + coda + fade (dots vs line),
misfit history (per obs), plus seeds_xy / grain_angles.

Process discipline (review-hardened, unchanged): persistent heartbeat
thread; pid-claim single flight; GRID_STOP at startup = exit
(launcher clears it); guarded fit_result reads; resume discards
stale-n state.

CLI:  python grid_daemon.py <sweep_name>
Stop: create <sweep>/GRID_STOP (the GUI button does this).
"""
import json
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys
import threading
import time

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

from scipy.signal import hilbert               # noqa: E402
from scipy import sparse                       # noqa: E402
from scipy.sparse.linalg import spsolve        # noqa: E402

import fit_fabric as FF                        # noqa: E402
import fit_sweep as FS                         # noqa: E402
import sweep_runner as SW                      # noqa: E402

try:
    import cupy as _cp
    _cp.zeros(4).sum()
    XP = _cp
    GPU = True
except Exception:
    XP = np
    GPU = False


def _scatter_add(a, idx, val):
    if GPU:
        import cupyx
        cupyx.scatter_add(a, idx, val)
    else:
        np.add.at(a, idx, val)


def _asnp(a):
    return XP.asnumpy(a) if GPU else a


C_REF = 3850.0
CELL_MM = 2.5
# CONTINUOUS ENVELOPE MATCHING (Jerome: "I don't think we should use
# binned values"): the clean window is sampled at 0.25 us and model
# boundaries deposit through a Gaussian pulse-envelope kernel instead
# of into hard 1 us bins. Binning quantized boundary depth to 1.9 mm
# and split edge-straddling echoes; envelope PEAK timing localizes a
# strong boundary to ~0.6 mm and is single-trace information (needs
# no cross-azimuth phase coherence). Samples are ~4x oversampled vs
# the envelope correlation time, so the channel weight is de-rated.
T_BINS = np.arange(24e-6, 36e-6 + 1e-9, 0.25e-6)   # sample grid
PULSE_SIG = 0.5e-6       # envelope kernel sigma (~pulse length)
KERN_HALF = 4            # kernel support: +-4 samples (+-1 us)
DAMP = 0.5
SLEEP_S = 0.5
W_TOF = 1.0 / 0.2        # 1/sigma (us)
W_CODA = 0.5 / 3.0       # 1/sigma (dB), /2 for sample correlation
W_FADE = 1.0 / 1.5       # 1/sigma (dB) - E1 fade channel
E_FLOOR_REL = 0.30       # frozen at first solve
LAT_W = 0.4              # lateral boundary weight
LAM_ODF = 1.0            # Watson prior strength on grain angles
LAM_REP = 5.0            # seed repulsion strength
SEED_EVERY = 5           # seed trial-move cadence (iterations)
SEED_BATCH = 12          # seeds tried per move round
SEED_SIG = 2.0e-3        # proposal std (m)
ALIVE_S = 30.0
INIT_NOISE_DEG = 10.0

_PSI_F = np.linspace(0.0, np.pi / 2, 2001)
_V_F = np.interp(_PSI_F, FF.F._PSI, FF.F._VQP)
_DV_F = np.gradient(_V_F, _PSI_F)


def _cell_of(x, y, D_m, n):
    ix = int((x + D_m / 2) / D_m * n)
    iy = int((y + D_m / 2) / D_m * n)
    if 0 <= ix < n and 0 <= iy < n:
        return iy * n + ix
    return -1


def _geometry_one(cfg, rt, n):
    """Ordered cell chain + per-bin lateral pairs (see v4)."""
    D_m = cfg["diameter_mm"] * 1e-3
    R = D_m / 2
    # specimen-frame probe at +rot (2026-07-29 fix, see grid_inversion
    # FRAME note: the old -rt mirrored attribution about the x-axis)
    th = np.radians(rt)
    probe = np.array([R * np.cos(th), R * np.sin(th)])
    bdir = -np.array([np.cos(th), np.sin(th)])
    phi = float(np.arctan2(bdir[1], bdir[0]))
    n_seg = 1600
    seg = D_m / n_seg
    s_mid = (np.arange(n_seg) + 0.5) * seg
    pts = probe[None, :] + s_mid[:, None] * bdir[None, :]
    cs = np.fromiter((_cell_of(x, y, D_m, n) for x, y in pts),
                     int, n_seg)
    inside = cs >= 0
    cells, L, s_bound = [], [], []
    prev = -2
    for j in range(n_seg):
        if not inside[j]:
            prev = -2
            continue
        c = cs[j]
        if c != prev:
            if prev >= 0:
                s_bound.append(s_mid[j] - seg / 2)
            cells.append(int(c))
            L.append(0.0)
            prev = c
        L[-1] += seg
    perp = np.array([-bdir[1], bdir[0]])
    off = CELL_MM * 1e-3
    lat_bins = []
    # Ricker envelope-peak source delay: subtract before time->depth
    t0_src = 1.2 / (cfg["f0_mhz"] * 1e6)
    for k in range(len(T_BINS) - 1):
        s0_ = C_REF * max(T_BINS[k] - t0_src, 0.0) / 2.0
        s1_ = C_REF * max(T_BINS[k + 1] - t0_src, 0.0) / 2.0
        m = inside & (s_mid >= s0_) & (s_mid < s1_)
        pairs = {}
        idxs = np.nonzero(m)[0]
        if idxs.size:
            wseg = 1.0 / idxs.size
            for j in idxs:
                c = int(cs[j])
                for sgn in (1.0, -1.0):
                    q = pts[j] + sgn * off * perp
                    cl = _cell_of(q[0], q[1], D_m, n)
                    if cl >= 0 and cl != c:
                        key = (min(c, cl), max(c, cl))
                        pairs[key] = pairs.get(key, 0.0) + wseg
        lat_bins.append([(a, b, w) for (a, b), w in pairs.items()])
    return (np.asarray(cells, int), np.asarray(L),
            np.asarray(s_bound), phi, lat_bins)


def _measured_row(d, rt):
    """Envelope (dB) at the fine sample grid - continuous matching,
    no energy binning."""
    with np.load(os.path.join(d, f"az{int(rt):03d}.npz")) as z:
        tr = np.asarray(z["trace"], float).ravel()
        fs = 1.0 / float(z["dt"])
    e = np.abs(hilbert(tr))
    row = np.full(len(T_BINS) - 1, np.nan)
    for k in range(len(T_BINS) - 1):
        s0_, s1_ = int(T_BINS[k] * fs), int(T_BINS[k + 1] * fs)
        if s1_ <= e.size and s1_ > s0_:
            v = np.sqrt((e[s0_:s1_] ** 2).mean())
            if v > 0:
                row[k] = 20 * np.log10(v)
    return row


def _status(d, phase, it=None):
    tmp = os.path.join(d, f"grid_status.tmp{os.getpid()}.json")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"phase": phase, "iteration": it,
                       "pid": os.getpid(), "t": time.time()}, fh)
        SW._atomic_replace(tmp, os.path.join(d, "grid_status.json"),
                           critical=False)
    except OSError:
        pass


def _pid_alive(pid):
    """NEVER os.kill(pid, 0) on Windows - it TERMINATES the target."""
    if not pid:
        return False
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    except Exception:
        return True


class _Beat(threading.Thread):
    def __init__(self, d):
        super().__init__(daemon=True)
        self.d = d
        self.phase = "starting"
        self.it = None
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(5.0):
            _status(self.d, self.phase, self.it)

    def set(self, phase, it=None):
        self.phase = phase
        if it is not None:
            self.it = it
        _status(self.d, phase, self.it)

    def stop(self):
        self._stop.set()


def _truth_axes_per_cell(d, n, D_m):
    """The tensor cache's squares are EXACTLY 5.0 mm, anchored at
    -D_cache/2, with ncell = ceil(D_cache/5) (the label grid slightly
    exceeds the disk, so the last row/col can be empty). Mapping by
    D/ncell drifted registration ~1 cell across the disk and shifted
    the truth mask off the circle (Jerome caught the misalignment) -
    replicate the builder's anchoring exactly."""
    p = os.path.join(d, "cof_cell_tensors.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        T = np.asarray(z["T"])
        tot = np.asarray(z["tot"])
        ncell = int(z["ncell"])
        D_c = float(z["D_m"])
    cell_t = 5.0e-3                            # cof_truth_map default
    ax_sq = np.full((ncell * ncell, 3), np.nan)
    for j in range(ncell * ncell):
        if tot[j] > 0:
            _, vecs = np.linalg.eigh(T[j])
            ax_sq[j] = vecs[:, -1]
    cell_d = D_m / n
    out = np.full((n * n, 3), np.nan)
    for iy in range(n):
        for ix in range(n):
            xc = (ix + 0.5) * cell_d - D_m / 2   # world coords
            yc = (iy + 0.5) * cell_d - D_m / 2
            jx = min(max(int((xc + D_c / 2) / cell_t), 0), ncell - 1)
            jy = min(max(int((yc + D_c / 2) / cell_t), 0), ncell - 1)
            out[iy * n + ix] = ax_sq[jx * ncell + jy]
    return out


def _read_fit(d, prev):
    p = os.path.join(d, "fit_result.json")
    for _ in range(3):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            time.sleep(0.1)
    return prev


class _Pack:
    """Padded batch arrays over all azimuths; vectorised forward +
    Jacobian pieces (CUDA when available). v5 adds the full-path
    contrast integral X(az) = sum |R| for the E1 fade channel."""

    def __init__(self, rays, coda_m):
        n_az = len(rays)
        nb = np.array([len(r[0]) for r in rays])
        Lmax = int(nb.max())
        self.n_az, self.Lmax = n_az, Lmax
        CELLS = np.zeros((n_az, Lmax), int)
        LEN = np.zeros((n_az, Lmax))
        PHI = np.zeros(n_az)
        MASK = np.zeros((n_az, Lmax), bool)
        nS = coda_m.shape[1]
        t_mid = 0.5 * (T_BINS[:-1] + T_BINS[1:])
        dts = float(T_BINS[1] - T_BINS[0])
        # discrete Gaussian pulse-envelope kernel (unit sum)
        dk = np.arange(-KERN_HALF, KERN_HALF + 1)
        KW = np.exp(-(dk * dts) ** 2 / (2 * PULSE_SIG ** 2))
        KW /= KW.sum()
        # observations = EVERY finite envelope sample (continuous
        # matching - quiet samples constrain silence)
        OBS = np.full((n_az, nS), -1, int)
        cnt = 0
        for i in range(n_az):
            for k in range(nS):
                if np.isfinite(coda_m[i, k]):
                    OBS[i, k] = cnt
                    cnt += 1
        self.n_obs = cnt
        # axial boundary -> kernel-spread deposit entries
        ax_obs, ax_flat, ax_w = [], [], []
        for i, (cells, L, s_bound, phi, _lat) in enumerate(rays):
            m = len(cells)
            CELLS[i, :m] = cells
            LEN[i, :m] = L
            MASK[i, :m] = True
            PHI[i] = phi
            for j, sb in enumerate(s_bound):
                t_b = 2.0 * sb / C_REF
                k0 = int(round((t_b - t_mid[0]) / dts))
                for kk, w_k in zip(dk, KW):
                    k = k0 + kk
                    if 0 <= k < nS and OBS[i, k] >= 0:
                        ax_obs.append(OBS[i, k])
                        ax_flat.append(i * (Lmax - 1) + j)
                        ax_w.append(w_k)
        # lateral pairs (unique per az) + kernel-spread deposits
        p_az, p_c1, p_c2 = [], [], []
        ld_pair, ld_obs, ld_w = [], [], []
        for i in range(n_az):
            pair_idx = {}
            for k in range(nS):
                for (a, b, w) in rays[i][4][k]:
                    key = (a, b)
                    if key not in pair_idx:
                        pair_idx[key] = len(p_az)
                        p_az.append(i)
                        p_c1.append(a)
                        p_c2.append(b)
                    pi = pair_idx[key]
                    for kk, w_k in zip(dk, KW):
                        k2 = k + kk
                        if 0 <= k2 < nS and OBS[i, k2] >= 0:
                            ld_pair.append(pi)
                            ld_obs.append(OBS[i, k2])
                            ld_w.append(w * LAT_W * w_k)
        self.n_lat = len(p_az)
        self.meas = np.array(
            [coda_m[i, k] for i in range(n_az) for k in range(nS)
             if OBS[i, k] >= 0])
        self.tt = np.array(
            [t_mid[k] * 1e6 for i in range(n_az) for k in range(nS)
             if OBS[i, k] >= 0])
        self.i_obs = np.array(
            [i for i in range(n_az) for k in range(nS)
             if OBS[i, k] >= 0])
        self.e_floor = None
        self.CELLS_np = CELLS
        self.MASK_np = MASK
        self.AX_OBS = XP.asarray(np.asarray(ax_obs, int))
        self.AX_FLAT = XP.asarray(np.asarray(ax_flat, int))
        self.AX_W = XP.asarray(np.asarray(ax_w))
        self.P_AZ = XP.asarray(np.asarray(p_az, int))
        self.P_C1 = XP.asarray(np.asarray(p_c1, int))
        self.P_C2 = XP.asarray(np.asarray(p_c2, int))
        self.LD_PAIR = XP.asarray(np.asarray(ld_pair, int))
        self.LD_OBS = XP.asarray(np.asarray(ld_obs, int))
        self.LD_W = XP.asarray(np.asarray(ld_w))
        self.CELLS = XP.asarray(CELLS)
        self.LEN = XP.asarray(LEN)
        self.PHI = XP.asarray(PHI)
        self.MASK = XP.asarray(MASK)
        self.BMF = XP.asarray(MASK[:, :-1] & MASK[:, 1:])
        self.PSI_F = XP.asarray(_PSI_F)
        self.V_F = XP.asarray(_V_F)
        self.DV_F = XP.asarray(_DV_F)

    def forward(self, alpha_np, need_jac=True):
        al = XP.asarray(alpha_np)
        A = al[self.CELLS]
        u = self.PHI[:, None] - A
        cu = XP.cos(u)
        psi = XP.arccos(XP.clip(XP.abs(cu), 0.0, 1.0))
        sp = XP.sin(psi)
        dpsi = -XP.where(sp > 1e-6,
                         XP.sign(cu) * XP.sin(u)
                         / XP.maximum(sp, 1e-6), 0.0)
        v = XP.interp(psi.ravel(), self.PSI_F,
                      self.V_F).reshape(psi.shape)
        dv = XP.interp(psi.ravel(), self.PSI_F,
                       self.DV_F).reshape(psi.shape) * dpsi
        s = 1e3 / v
        tof_raw = 2e3 * (self.LEN * s * self.MASK).sum(axis=1)
        v1, v2 = v[:, :-1], v[:, 1:]
        Rf = XP.where(self.BMF, (v2 - v1) / (v2 + v1), 0.0)
        Xfade = XP.abs(Rf).sum(axis=1)          # full-path contrast
        # axial deposits: kernel-spread through the pulse envelope
        E_obs = XP.zeros(self.n_obs)
        Rfl = Rf.ravel()
        R_ax = Rfl[self.AX_FLAT]
        _scatter_add(E_obs, self.AX_OBS, self.AX_W * R_ax ** 2)
        if self.n_lat:
            ph = self.PHI[self.P_AZ]

            def vd(Ai):
                ui = ph - Ai
                cui = XP.cos(ui)
                ps = XP.arccos(XP.clip(XP.abs(cui), 0.0, 1.0))
                spi = XP.sin(ps)
                dpi = -XP.where(spi > 1e-6,
                                XP.sign(cui) * XP.sin(ui)
                                / XP.maximum(spi, 1e-6), 0.0)
                vv = XP.interp(ps, self.PSI_F, self.V_F)
                dd = XP.interp(ps, self.PSI_F, self.DV_F) * dpi
                return vv, dd
            vl1, dl1 = vd(al[self.P_C1])
            vl2, dl2 = vd(al[self.P_C2])
            Rl = (vl2 - vl1) / (vl2 + vl1)
            _scatter_add(E_obs, self.LD_OBS,
                         self.LD_W * (Rl ** 2)[self.LD_PAIR])
        out = {"tof_raw": _asnp(tof_raw), "E_obs": _asnp(E_obs),
               "Xfade": _asnp(Xfade)}
        if not need_jac:
            return out
        dtof = XP.where(self.MASK,
                        -2e6 * self.LEN * dv / v ** 2, 0.0)
        dR1 = -2 * v2 / (v2 + v1) ** 2 * dv[:, :-1]
        dR2 = 2 * v1 / (v2 + v1) ** 2 * dv[:, 1:]
        # axial Jacobian entries via the same flat gathers
        C1f = self.CELLS[:, :-1].ravel()[self.AX_FLAT]
        C2f = self.CELLS[:, 1:].ravel()[self.AX_FLAT]
        dR1f = dR1.ravel()[self.AX_FLAT]
        dR2f = dR2.ravel()[self.AX_FLAT]
        rows_parts = [self.AX_OBS, self.AX_OBS]
        cols_parts = [C1f, C2f]
        dE_parts = [self.AX_W * 2 * R_ax * dR1f,
                    self.AX_W * 2 * R_ax * dR2f]
        if self.n_lat:
            dRl1 = -2 * vl2 / (vl2 + vl1) ** 2 * dl1
            dRl2 = 2 * vl1 / (vl2 + vl1) ** 2 * dl2
            rows_parts += [self.LD_OBS, self.LD_OBS]
            cols_parts += [self.P_C1[self.LD_PAIR],
                           self.P_C2[self.LD_PAIR]]
            dE_parts += [self.LD_W * (2 * Rl * dRl1)[self.LD_PAIR],
                         self.LD_W * (2 * Rl * dRl2)[self.LD_PAIR]]
        # fade Jacobian: dX/dalpha via sign(R) on full-path boundaries
        bf = self.BMF
        sgn = XP.sign(Rf)
        fx_rows = XP.concatenate(
            [XP.nonzero(bf)[0], XP.nonzero(bf)[0]])
        fx_cols = XP.concatenate(
            [self.CELLS[:, :-1][bf], self.CELLS[:, 1:][bf]])
        fx_vals = XP.concatenate(
            [(sgn * dR1)[bf], (sgn * dR2)[bf]])
        out.update(dtof=_asnp(dtof),
                   rows_b=_asnp(XP.concatenate(rows_parts)),
                   cols_b=_asnp(XP.concatenate(cols_parts)),
                   dE_b=_asnp(XP.concatenate(dE_parts)),
                   fx_rows=_asnp(fx_rows), fx_cols=_asnp(fx_cols),
                   fx_vals=_asnp(fx_vals))
        return out


def _init_seeds(rng, Ng, R):
    """Poisson-disc-ish: jittered grid clipped to the disk."""
    g = int(np.ceil(np.sqrt(Ng * 4 / np.pi)))
    xs = (np.arange(g) + 0.5) / g * 2 * R - R
    XXs, YYs = np.meshgrid(xs, xs)
    pts = np.stack([XXs.ravel(), YYs.ravel()], axis=1)
    pts += rng.normal(0, R / g * 0.5, pts.shape)
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < R * 0.97]
    idx = rng.permutation(len(pts))[:Ng]
    return pts[idx].copy()


def _residuals(fwd, tof_m, keep_az, keep_fade, fade_m, pack, i_az):
    """Channel residuals + nuisance fits at a given forward state.
    Returns (rhs pieces, diagnostics, nuisance dict)."""
    tof_raw = fwd["tof_raw"]
    t_off = float(np.mean(tof_m - tof_raw))
    r_tof = tof_m - (tof_raw + t_off)
    r_tof_h = np.clip(r_tof, -0.6, 0.6)
    E_obs = fwd["E_obs"]
    ok_obs = keep_az[pack.i_obs]
    if pack.e_floor is None and (E_obs > 0).any():
        pack.e_floor = (E_FLOOR_REL
                        * float(np.median(E_obs[E_obs > 0])))
    e_floor = pack.e_floor if pack.e_floor else 1e-12
    raw_db = 10 * np.log10(E_obs + e_floor)
    tt = pack.tt
    Adm = np.stack([np.ones_like(tt), tt - tt.mean()], axis=1)
    gg, *_ = np.linalg.lstsq(Adm[ok_obs],
                             (pack.meas - raw_db)[ok_obs],
                             rcond=None)
    pred_db = raw_db + Adm @ gg
    r_cod = pack.meas - pred_db
    r_cod_h = np.clip(r_cod, -6.0, 6.0) * ok_obs
    X = fwd["Xfade"]
    kf = keep_fade & np.isfinite(fade_m)
    if kf.sum() > 10 and np.std(X[kf]) > 1e-9:
        Af = np.stack([np.ones(kf.sum()), X[kf]], axis=1)
        cf, *_ = np.linalg.lstsq(Af, fade_m[kf], rcond=None)
    else:
        cf = np.array([np.nanmean(fade_m[kf]) if kf.any() else 0.0,
                       0.0])
    # PHYSICAL SIGN CONSTRAINT: more path contrast must mean MORE
    # fade (lower amplitude), so b_f <= 0. A positive LS fit means
    # the naive sum|R| kernel is riding a non-causal correlation
    # (the validated record says fade is DECOHERENCE physics, not
    # mean-field contrast) - disable the channel rather than let it
    # drag angles the wrong way. First live run fitted b_f = +50.
    if cf[1] > 0:
        kf = np.zeros_like(kf)
        cf = np.array([float(np.nanmean(fade_m)) if
                       np.isfinite(fade_m).any() else 0.0, 0.0])
    fade_pred = cf[0] + cf[1] * X
    r_fad = np.where(kf, fade_m - fade_pred, 0.0)
    r_fad_h = np.clip(r_fad, -3.0, 3.0)
    return (r_tof_h, r_cod_h, r_fad_h,
            {"r_tof": r_tof, "r_cod": r_cod, "r_fad": r_fad,
             "ok_obs": ok_obs, "kf": kf, "e_floor": e_floor,
             "t_off": t_off, "b_f": float(cf[1]),
             "pred_db": pred_db, "fade_pred": fade_pred})


def _objective(rhs_t, rhs_c, rhs_f, n_eff):
    return float((W_TOF ** 2 * (rhs_t ** 2).sum()
                  + W_CODA ** 2 * (rhs_c ** 2).sum()
                  + W_FADE ** 2 * (rhs_f ** 2).sum())) / max(n_eff, 1)


def run(cfg):
    name = cfg["name"]
    d = SW.sweep_dir(name)
    stop_f = os.path.join(d, "GRID_STOP")
    if os.path.exists(stop_f):
        print("[daemon] GRID_STOP present at startup - exiting "
              "(launcher clears the flag)", flush=True)
        return
    st_p = os.path.join(d, "grid_status.json")
    try:
        with open(st_p, encoding="utf-8") as fh:
            other = json.load(fh)
        if (time.time() - float(other.get("t", 0)) < ALIVE_S
                and int(other.get("pid") or 0) != os.getpid()
                and _pid_alive(other.get("pid"))):
            print("[daemon] another grid daemon is live (pid "
                  f"{other.get('pid')}) - exiting", flush=True)
            return
    except (OSError, ValueError):
        pass
    _status(d, "claiming")
    time.sleep(1.0)
    try:
        with open(st_p, encoding="utf-8") as fh:
            claim = json.load(fh)
        if int(claim.get("pid") or 0) != os.getpid():
            print("[daemon] lost startup race - exiting", flush=True)
            return
    except (OSError, ValueError):
        pass
    beat = _Beat(d)
    beat.start()
    state_p = os.path.join(d, "grid_state.npz")

    D_m = cfg["diameter_mm"] * 1e-3
    R = D_m / 2
    n = int(np.ceil(D_m / (CELL_MM * 1e-3)))
    N = n * n
    Ng = int(cfg.get("n_grains", 100))
    rng = np.random.default_rng(2)
    # cell centres for Voronoi assignment + gauge angles
    _ctr = (np.arange(n) + 0.5) * (D_m / n) - R
    _yy, _xx = np.meshgrid(_ctr, _ctr, indexing="ij")
    cell_xy = np.stack([_xx.ravel(), _yy.ravel()], axis=1)
    phi_cell = np.arctan2(_yy, _xx).ravel() % np.pi

    seeds = None
    angs = None
    hist = []
    it = 0
    if os.path.exists(state_p):
        with np.load(state_p) as z:
            if (int(z["n"]) == n and "seeds_xy" in z
                    and len(z["grain_angles"]) == Ng):
                seeds = np.asarray(z["seeds_xy"], float).copy()
                angs = np.radians(np.asarray(z["grain_angles"],
                                             float)).copy()
                hist = list(z["misfit"])
                it = int(z["iteration"])
                print(f"[daemon] resuming at iteration {it}",
                      flush=True)
            else:
                print("[daemon] stale/foreign state - fresh start",
                      flush=True)

    truth_ax = _truth_axes_per_cell(d, n, D_m)
    # E1 fade channel data (trace_lab.py output)
    fade_by_rot = {}
    fp = os.path.join(d, "e1_fade.npz")
    if os.path.exists(fp):
        with np.load(fp) as z:
            for r_, a_ in zip(z["rots"], z["amp_db"]):
                fade_by_rot[int(r_)] = float(a_)
    else:
        print("[daemon] no e1_fade.npz - fade channel disabled "
              "(run trace_lab.py once)", flush=True)

    FS.AZ_SMOOTH_DEG = 0.0
    ray_cache, coda_cache = {}, {}
    n_rots = -1
    fit = None
    pack = None
    while not os.path.exists(stop_f):
        it += 1
        fit = _read_fit(d, fit)
        if fit is None:
            beat.set("waiting for first fit_result.json", it)
            time.sleep(5)
            continue
        a_bar = np.radians(float(fit["alpha_deg"]) % 180.0)
        kap_bar = min(float(fit.get("kappa", 2.5)), 10.0)
        if seeds is None:
            seeds = _init_seeds(rng, Ng, R)
            Ng = len(seeds)
            angs = (a_bar + np.radians(INIT_NOISE_DEG)
                    * rng.standard_normal(Ng)) % np.pi
            print(f"[daemon] initialised {Ng} grains at axis "
                  f"{np.degrees(a_bar):.1f} deg", flush=True)
        beat.set("loading observables", it)
        data = FS.sweep_data(cfg)
        rots = np.array([r_["rot"] for r_ in data], float)
        tof_m = np.array([r_["tof_us"] for r_ in data])
        alpha_p = float(fit.get("alpha_probe_deg",
                                -fit["alpha_deg"])) % 180
        psi_doc = FS._psi_from_axis(rots, alpha_p)
        keep_az = (psi_doc >= 15) & ~((psi_doc > 65) & (psi_doc < 85))
        keep_fade = ~((psi_doc > 65) & (psi_doc < 85))
        fade_m = np.array([fade_by_rot.get(int(r_), np.nan)
                           for r_ in rots])
        new_r = [rt for rt in rots if int(rt) not in ray_cache]
        if new_r:
            beat.set(f"indexing {len(new_r)} azimuths", it)
            for rt in new_r:
                ri = int(rt)
                ray_cache[ri] = _geometry_one(cfg, rt, n)
                coda_cache[ri] = _measured_row(d, rt)
        rays = [ray_cache[int(rt)] for rt in rots]
        coda_m = np.stack([coda_cache[int(rt)] for rt in rots])
        if len(rots) != n_rots:
            beat.set("packing batch arrays", it)
            pack = _Pack(rays, coda_m)
            n_rots = len(rots)
            print(f"[daemon] {n_rots} az; {pack.n_obs} coda obs; "
                  f"fade {int(np.isfinite(fade_m).sum())} az "
                  f"({'CUDA' if GPU else 'CPU'})", flush=True)

        beat.set("solving", it)
        # ── rasterize partition, forward + Jacobian ─────────────────
        d2 = ((cell_xy[:, None, :] - seeds[None, :, :]) ** 2).sum(2)
        assign = np.argmin(d2, axis=1)
        alpha = angs[assign]
        fwd = pack.forward(alpha, need_jac=True)
        (rhs_t, rhs_c, rhs_f, diag) = _residuals(
            fwd, tof_m, keep_az, keep_fade, fade_m, pack, rots)
        n_eff = (n_rots + int(diag["ok_obs"].sum())
                 + int(diag["kf"].sum()))
        mis = _objective(rhs_t, rhs_c, rhs_f, n_eff)
        hist = (hist + [mis])[-300:]

        # cell-level Jacobian rows -> grain columns via assignment
        f_obs = 10.0 / np.log(10.0) / (fwd["E_obs"] + diag["e_floor"])
        mtx, mcl = np.nonzero(fwd["dtof"])
        b_f = diag["b_f"]
        rows = np.concatenate([
            mtx,
            n_rots + fwd["rows_b"],
            n_rots + pack.n_obs + fwd["fx_rows"]])
        cols_cells = np.concatenate([
            pack.CELLS_np[mtx, mcl], fwd["cols_b"], fwd["fx_cols"]])
        vals = np.concatenate([
            W_TOF * fwd["dtof"][mtx, mcl],
            W_CODA * f_obs[fwd["rows_b"]] * fwd["dE_b"]
            * diag["ok_obs"][fwd["rows_b"]],
            W_FADE * b_f * fwd["fx_vals"]
            * diag["kf"][fwd["fx_rows"]]])
        cols_g = assign[cols_cells]
        n_rows = n_rots + pack.n_obs + n_rots
        Jg = sparse.coo_matrix((vals, (rows, cols_g)),
                               shape=(n_rows, Ng)).tocsr()
        rhs = np.concatenate([W_TOF * rhs_t, W_CODA * rhs_c,
                              W_FADE * rhs_f])
        # Watson ODF prior on angles (gradient + diagonal Hessian)
        dA = (angs - a_bar + np.pi / 2) % np.pi - np.pi / 2
        g_odf = LAM_ODF * kap_bar * np.sin(2 * dA)
        H_odf = LAM_ODF * 2.0 * kap_bar
        M = (Jg.T @ Jg) + (H_odf + 1e-2) * sparse.identity(Ng)
        step = spsolve(M.tocsc(), Jg.T @ rhs - g_odf)
        angs = (angs + DAMP * step) % np.pi

        # ── periodic greedy seed moves ──────────────────────────────
        if it % SEED_EVERY == 0:
            beat.set("seed moves", it)
            base = mis + _rep_penalty(seeds, R, Ng)
            order = rng.permutation(Ng)[:SEED_BATCH]
            for gidx in order:
                prop = seeds.copy()
                prop[gidx] += rng.normal(0, SEED_SIG, 2)
                if np.hypot(*prop[gidx]) > R * 0.97:
                    continue
                d2p = ((cell_xy[:, None, :]
                        - prop[None, :, :]) ** 2).sum(2)
                al_p = angs[np.argmin(d2p, axis=1)]
                fw_p = pack.forward(al_p, need_jac=False)
                (pt, pc, pf, _dg) = _residuals(
                    fw_p, tof_m, keep_az, keep_fade, fade_m,
                    pack, rots)
                cand = (_objective(pt, pc, pf, n_eff)
                        + _rep_penalty(prop, R, Ng))
                if cand < base:
                    seeds = prop
                    base = cand

        # ── outputs ─────────────────────────────────────────────────
        theta = np.full(N, np.nan)
        theta_g = np.full(N, np.nan)
        if truth_ax is not None:
            ag = np.stack([np.cos(alpha), np.sin(alpha),
                           np.zeros(N)], axis=1)
            dot = np.abs(np.einsum("ci,ci->c", truth_ax, ag))
            th_all = np.degrees(np.arccos(np.clip(dot, 0, 1)))
            a_mir = (2.0 * phi_cell - alpha) % np.pi
            am = np.stack([np.cos(a_mir), np.sin(a_mir),
                           np.zeros(N)], axis=1)
            dot_m = np.abs(np.einsum("ci,ci->c", truth_ax, am))
            th_mir = np.degrees(np.arccos(np.clip(dot_m, 0, 1)))
            m_ok = np.isfinite(truth_ax[:, 0])
            theta[m_ok] = th_all[m_ok]
            theta_g[m_ok] = np.minimum(th_all, th_mir)[m_ok]

        tof_pred = fwd["tof_raw"] + diag["t_off"]
        coda_az_meas = np.full(n_rots, np.nan)
        coda_az_pred = np.full(n_rots, np.nan)
        if pack.n_obs:
            s_m = np.zeros(n_rots)
            s_p = np.zeros(n_rots)
            cnt = np.zeros(n_rots)
            np.add.at(s_m, pack.i_obs, pack.meas)
            np.add.at(s_p, pack.i_obs, diag["pred_db"])
            np.add.at(cnt, pack.i_obs, 1.0)
            okz = cnt > 0
            coda_az_meas[okz] = s_m[okz] / cnt[okz]
            coda_az_pred[okz] = s_p[okz] / cnt[okz]

        tmp = state_p[:-4] + f".tmp{os.getpid()}.npz"
        try:
            np.savez(
                tmp, n=n, cell_mm=CELL_MM, iteration=it,
                alpha_grid=np.degrees(alpha).reshape(n, n),
                theta_grid=theta.reshape(n, n),
                theta_gauge=theta_g.reshape(n, n),
                seeds_xy=seeds, grain_angles=np.degrees(angs),
                misfit=np.asarray(hist), n_az=n_rots,
                n_az_coda=n_rots, texture=True, grains=True,
                coda_rms=float(diag["r_cod"][diag["ok_obs"]].std())
                if diag["ok_obs"].any() else np.nan,
                tof_rms=float(diag["r_tof"].std()),
                fade_rms=float(diag["r_fad"][diag["kf"]].std())
                if diag["kf"].any() else np.nan,
                wall_frac=float((assign.reshape(n, n)[:, :-1]
                                 != assign.reshape(n, n)[:, 1:])
                                .mean()),
                rots=rots, tof_meas=tof_m, tof_pred=tof_pred,
                coda_meas=coda_az_meas, coda_pred=coda_az_pred,
                fade_meas=fade_m, fade_pred=diag["fade_pred"])
            if not SW._atomic_replace(tmp, state_p, critical=False):
                print("[daemon] state save BLOCKED - retrying next "
                      "loop", flush=True)
        except OSError as e:
            print(f"[daemon] state save failed: {e}", flush=True)

        if it % 20 == 0:
            th_med = (float(np.nanmedian(theta_g))
                      if np.isfinite(theta_g).any() else float("nan"))
            print(f"[daemon] it {it}: misfit/obs {mis:.2f}, tof "
                  f"{diag['r_tof'].std():.3f} us, coda "
                  f"{diag['r_cod'][diag['ok_obs']].std():.2f} dB, "
                  f"fade "
                  f"{diag['r_fad'][diag['kf']].std():.2f} dB, "
                  f"theta-gauge med {th_med:.1f}, b_f "
                  f"{diag['b_f']:+.1f}", flush=True)
        time.sleep(SLEEP_S)
    beat.stop()
    print("[daemon] GRID_STOP found - exiting; state saved.",
          flush=True)


def _rep_penalty(seeds, R, Ng):
    """Seed repulsion below ~half the mean grain spacing."""
    d_mean = 2 * R / np.sqrt(Ng)
    dmin = 0.5 * d_mean
    dx = seeds[:, None, :] - seeds[None, :, :]
    dd = np.sqrt((dx ** 2).sum(2))
    np.fill_diagonal(dd, 1e9)
    v = np.clip(dmin - dd, 0.0, None)
    return LAM_REP * float((v ** 2).sum()) / max(Ng, 1)


if __name__ == "__main__":
    run(SW.load(sys.argv[1]))
