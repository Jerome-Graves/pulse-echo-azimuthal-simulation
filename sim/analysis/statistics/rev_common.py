"""Shared loader for the adversarial review. READ-ONLY on the repo:
observables / geo envelopes for sweeps that lack a repo cache are
computed here and cached in the SCRATCHPAD."""
import json
import os
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", ".."))
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "sim"))
sys.path.insert(0, os.path.join(REPO, "vendor"))
sys.path[:0] = [os.path.join(REPO, "sim", _d)
                for _d in ('core', 'model', 'pipeline',
                           'fe_crosscheck', 'fw_checks')]

import fit_sweep as FS            # noqa: E402
import fit_fabric as FF           # noqa: E402
import sweep_runner as SW         # noqa: E402
import ladder                     # noqa: E402
from scipy.signal import hilbert  # noqa: E402

C_REF = FS.C_REF
D = FS.D
W0, W1 = FS.W0, FS.W1


def cfg_of(name):
    with open(os.path.join(SW.sweep_dir(name), "config.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def fitres(name):
    p = os.path.join(SW.sweep_dir(name), "fit_result.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# ── observables: exact replica of FS.sweep_data's per-azimuth block,
# but cached in the scratchpad so nothing is written to the repo ──────
def _obs_one(d, az, cfg, nb):
    with np.load(os.path.join(d, f"az{az:03d}.npz")) as z:
        tr = np.array(z["trace"], float).ravel()
        dt = float(z["dt"])
        f0v = float(z["f0"]) if "f0" in z.files else cfg["f0_mhz"] * 1e6
    fs = 1.0 / dt
    e = np.abs(hilbert(tr))
    k0 = int(2 * D / C_REF * fs)
    a = max(k0 - int(2e-6 * fs), 0)
    seg = e[a:k0 + int(2e-6 * fs)]
    E1 = seg.max()
    ipk = int(np.argmax(seg))
    thr = 0.25 * E1
    above = np.where(seg[:ipk + 1] >= thr)[0]
    j = int(above[0]) if len(above) else ipk
    if j > 0 and seg[j] > seg[j - 1]:
        frac = (thr - seg[j - 1]) / (seg[j] - seg[j - 1])
    else:
        frac = 0.0
    tof = (a + j - 1 + frac) / fs
    if nb > 1:
        F = np.fft.rfft(tr)
        fr = np.fft.rfftfreq(len(tr), dt)
        edges = np.linspace(0.5 * f0v, 1.5 * f0v, nb + 1)
        band_e = []
        for bi in range(nb):
            mask = (fr >= edges[bi]) & (fr < edges[bi + 1])
            trb = np.fft.irfft(F * mask, len(tr))
            eb = np.abs(hilbert(trb))
            band_e.append((eb[int(W0 * fs):int(W1 * fs)] ** 2).mean())
        coda = np.sqrt(np.mean(band_e))
    else:
        coda = np.sqrt((e[int(W0 * fs):int(W1 * fs)] ** 2).mean())
    lo_t = int(FS.TAIL_W[0] * fs)
    hi_t = min(int(FS.TAIL_W[1] * fs), len(tr))
    if hi_t - lo_t >= int(2e-6 * fs):
        tail_db = float(20 * np.log10(
            np.sqrt((e[lo_t:hi_t] ** 2).mean()) / max(E1, 1e-30)))
    else:
        tail_db = float("nan")
    kpk = a + ipk
    wn = int(2e-6 * fs)
    s2 = tr[max(kpk - wn, 0):min(kpk + wn, len(tr))]
    mag = np.abs(np.fft.rfft(s2))
    frq = np.fft.rfftfreq(len(s2), dt)
    bm = (frq >= 0.25 * f0v) & (frq <= 2.0 * f0v)
    cent = float((frq[bm] * mag[bm]).sum()
                 / max(mag[bm].sum(), 1e-30) / 1e6)
    inten = e[int(W0 * fs):int(W1 * fs)] ** 2
    spk = float(inten.std() / max(inten.mean(), 1e-30))
    return (float(coda), float(E1), float(tof * 1e6), tail_db, cent, spk)


def sweep_rows(name, smooth=True):
    """FS.sweep_data-equivalent rows, scratchpad-cached."""
    cfg = cfg_of(name)
    d = SW.sweep_dir(name)
    nb = FS._fd_bands(cfg["f0_mhz"])
    cp = os.path.join(SCRATCH, f"revobs_{name}.npz")
    cached = {}
    if os.path.exists(cp):
        with np.load(cp) as z:
            for k in z["az"]:
                pass
            A = z["az"]
            M = z["vals"]
        for i, k in enumerate(A):
            cached[int(k)] = tuple(M[i])
    azs = SW.done_azimuths(cfg)
    todo = [a for a in azs if a not in cached]
    if todo:
        print(f"[obs] {name}: computing {len(todo)} azimuths", flush=True)
        for a in todo:
            cached[a] = _obs_one(d, a, cfg, nb)
        ks = sorted(cached)
        np.savez(cp, az=np.asarray(ks),
                 vals=np.asarray([cached[k] for k in ks], float))
    rows = []
    for az in azs:
        c_, e_, t_, td_, cm_, sc_ = cached[az]
        rows.append(dict(rot=az, coda=c_, e1=e_, tof_us=t_, tail_db=td_,
                         e1_db=20 * np.log10(max(e_, 1e-30)),
                         cent_mhz=cm_, speckle_contrast=sc_))
    rows.sort(key=lambda r_: r_["rot"])
    if smooth and FS.AZ_SMOOTH_DEG > 0 and len(rows) > 4:
        rr = np.array([r["rot"] for r in rows], float)
        cd = 20 * np.log10([r["coda"] for r in rows])
        tf = np.array([r["tof_us"] for r in rows])
        rr180 = rr % 180.0
        for i, r in enumerate(rows):
            dd = np.abs(rr180 - rr180[i])
            sel = np.minimum(dd, 180.0 - dd) <= FS.AZ_SMOOTH_DEG
            r["coda"] = 10 ** (cd[sel].mean() / 20)
            r["tof_us"] = float(tf[sel].mean())
    return rows


def sweep_geo(name, rots):
    """geo, geo_tail: repo cache if valid, else scratchpad cache."""
    cfg = cfg_of(name)
    rp = os.path.join(SW.sweep_dir(name), "fit_geo_cache.npz")
    have = {}
    if os.path.exists(rp):
        with np.load(rp) as z:
            if "ver" in z.files and str(z["ver"]) == FS.GEO_VER:
                have = {int(r): (float(g), float(gt)) for r, g, gt in
                        zip(z["rots"], z["geo"], z["geo_tail"])}
    sp_ = os.path.join(SCRATCH, f"revgeo_{name}.npz")
    if os.path.exists(sp_):
        with np.load(sp_) as z:
            for r, g, gt in zip(z["rots"], z["geo"], z["geo_tail"]):
                have.setdefault(int(r), (float(g), float(gt)))
    missing = [int(r) for r in rots if int(r) not in have]
    if missing:
        import born
        from specimen import DiskSpecimen
        print(f"[geo] {name}: computing {len(missing)}", flush=True)
        f0 = cfg["f0_mhz"] * 1e6
        h = C_REF / f0 / cfg["ppw"]
        sp = DiskSpecimen(diameter_m=cfg["diameter_mm"] * 1e-3,
                          thickness_m=cfg["thickness_mm"] * 1e-3,
                          n_grains=cfg["n_grains"], size_cv=cfg["size_cv"],
                          concentration=cfg["concentration"],
                          spatial_corr=cfg["spatial_corr"],
                          fabric_axis=tuple(cfg["fabric_axis"]),
                          seed=cfg["seed"])
        build = sp.build(h)
        mcfg = ladder.standard_cfg()
        mcfg.probe.f0 = f0
        dt_m = ladder.DT
        lo, hi = int(W0 / dt_m), int(W1 / dt_m)
        lo_t, hi_t = int(FS.TAIL_W[0] / dt_m), int(FS.TAIL_W[1] / dt_m)
        for rot in missing:
            _, env = born.boundary_scatter(dict(build), mcfg,
                                           azimuth_rad=np.radians(rot),
                                           unit_contrast=True)
            env = np.asarray(env, float)
            have[int(rot)] = (
                float(np.sqrt((env[lo:hi] ** 2).mean())),
                float(np.sqrt((env[lo_t:min(hi_t, len(env))] ** 2).mean())))
        ks = sorted(have)
        np.savez(sp_, rots=np.asarray(ks),
                 geo=np.asarray([have[k][0] for k in ks]),
                 geo_tail=np.asarray([have[k][1] for k in ks]))
    return (np.asarray([have[int(r)][0] for r in rots], float),
            np.asarray([have[int(r)][1] for r in rots], float))


def truth_alpha(cfg):
    ax = np.asarray(cfg["fabric_axis"], float)
    return float(np.degrees(np.arctan2(ax[1], ax[0])) % 180.0)


def harm_fit(az_deg, y, orders=(2, 4), w=None):
    """LSQ fit of const + sum_k (Ak cos k*az + Bk sin k*az).
    Returns dict order -> (amp, phase_deg) with phase = the azimuth of
    the maximum, folded to [0, 360/k)."""
    th = np.radians(np.asarray(az_deg, float))
    cols = [np.ones_like(th)]
    for k in orders:
        cols += [np.cos(k * th), np.sin(k * th)]
    X = np.column_stack(cols)
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if w is None:
        c, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    else:
        sw = np.sqrt(np.asarray(w, float))[m]
        c, *_ = np.linalg.lstsq(X[m] * sw[:, None], y[m] * sw, rcond=None)
    out = {"const": float(c[0]), "cond": float(np.linalg.cond(X[m]))}
    for i, k in enumerate(orders):
        a, b = c[1 + 2 * i], c[2 + 2 * i]
        out[k] = (float(np.hypot(a, b)),
                  float(np.degrees(np.arctan2(b, a)) / k % (360.0 / k)))
    return out


def naive_2t(r0, rots_keep):
    """The estimator currently in FS._lab_2theta (no cap)."""
    rk = np.asarray(rots_keep, float)
    ph = np.radians(2.0 * rk)
    d = np.asarray(r0, float) - np.median(r0)
    c = 2.0 * np.mean(d * np.exp(-1j * ph))
    return float(abs(c)), float(np.degrees(np.angle(c)) / 2.0 % 180.0)
