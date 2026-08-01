"""STRAND 4: is the facet-based Born model a better coda predictor than E[R^2]?

Builds the specimen, extracts every grain-boundary voxel face ONCE, then for
each azimuth evaluates a family of facet backscatter predictors that differ
only in how they weight FACET ORIENTATION (the thing E[R^2] cannot see).

CPU only.  The GPU label path is monkeypatched out (a sweep is running).
"""
import json
import os
import sys

import numpy as np
from scipy.signal import hilbert
from scipy.special import j1

SIM = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim"
sys.path.insert(0, SIM)
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")

OUT = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
SCR = (r"C:\Users\Jerome\AppData\Local\Temp\claude\C--Users-Jerome"
       r"\72aa31c0-c0c1-48de-881e-9470fe03e8ba\scratchpad")

C_REF = 3850.0
DT = 2.164e-8                 # ladder.DT, the model record step
REC_END = 67.5e-6
CODA_W = (24e-6, 36e-6)
FRAC_BW = 0.6
A_EL = 6.35e-3 / 2.0
RHO = 917.0


# ───────────── CPU replacement for the GPU power-diagram labeller ─────────
def _label_grid_cpu(x, z, R, seeds, weights):
    """Exact all-seeds power argmin, identical semantics to _label_grid_gpu."""
    nx, nz = len(x), len(z)
    X, Y = np.meshgrid(x, x, indexing="ij")
    inside = (X ** 2 + Y ** 2) <= R ** 2
    Pxy = np.stack([X[inside], Y[inside]], axis=1)
    d_xy = (((Pxy[:, None, :] - seeds[None, :, :2]) ** 2).sum(-1)
            - weights[None, :])
    raw = np.full((nx, nx, nz), -1, np.int32)
    for iz in range(nz):
        d2 = d_xy + (z[iz] - seeds[:, 2])[None, :] ** 2
        plane = np.full((nx, nx), -1, np.int32)
        plane[inside] = d2.argmin(axis=1).astype(np.int32)
        raw[:, :, iz] = plane
    return raw


import specimen as SP                                        # noqa: E402
SP.DiskSpecimen._label_grid_gpu = staticmethod(_label_grid_cpu)
from specimen import DiskSpecimen                            # noqa: E402
import forward as F                                          # noqa: E402


# ───────────────────────────── facet extraction ───────────────────────────
def facets(cfgd, ppw, tag):
    """Every internal voxel face where the grain label changes.

    Returns dict of flat arrays: centre coords (m), grain pair (ga, gb),
    the mesh axis the face is normal to, and the EXACT Laguerre wall normal
    (unit).  Laguerre/power-diagram walls are planes whose normal is the
    seed difference, so no voxel-gradient estimate (and no staircase normal
    bias) enters anywhere.  Cached on disk.
    """
    cache = os.path.join(SCR, f"facets_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    h = C_REF / (cfgd["f0_mhz"] * 1e6) / ppw
    sp = DiskSpecimen(diameter_m=cfgd["diameter_mm"] * 1e-3,
                      thickness_m=cfgd["thickness_mm"] * 1e-3,
                      n_grains=cfgd["n_grains"], size_cv=cfgd["size_cv"],
                      concentration=cfgd["concentration"],
                      spatial_corr=cfgd["spatial_corr"],
                      fabric_axis=tuple(cfgd["fabric_axis"]),
                      seed=cfgd["seed"])
    print(f"  building {tag} (h = {h*1e3:.4f} mm) ...", flush=True)
    b = sp.build(h)
    lab = b["labels"].astype(np.int32)
    axes = np.asarray(b["axes"], float)
    seeds = np.asarray(b["seeds"], float)
    nx, ny, nz = lab.shape

    Xc, Yc, Zc, GA, GB, AX = [], [], [], [], [], []
    for ax in range(3):
        s1 = [slice(None)] * 3
        s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1)
        s2[ax] = slice(1, None)
        la, lb = lab[tuple(s1)], lab[tuple(s2)]
        m = (la != lb) & (la >= 0) & (lb >= 0)
        if not m.any():
            continue
        idx = np.argwhere(m).astype(np.float64)
        idx[:, ax] += 0.5
        Xc.append((idx[:, 0] - (nx - 1) / 2.0) * h)
        Yc.append((idx[:, 1] - (ny - 1) / 2.0) * h)
        Zc.append((idx[:, 2] - (nz - 1) / 2.0) * h)
        GA.append(la[m].astype(np.int32))
        GB.append(lb[m].astype(np.int32))
        AX.append(np.full(m.sum(), ax, np.int8))
    d = dict(x=np.concatenate(Xc), y=np.concatenate(Yc), z=np.concatenate(Zc),
             ga=np.concatenate(GA), gb=np.concatenate(GB),
             ax=np.concatenate(AX), axes=axes, seeds=seeds,
             h=np.array([h]))
    n = seeds[d["gb"]] - seeds[d["ga"]]
    nn = np.linalg.norm(n, axis=1)
    nn[nn < 1e-12] = 1.0
    d["nrm"] = (n / nn[:, None]).astype(np.float32)
    # volume fractions, for the E[R^2] baseline
    ok = lab >= 0
    vol = np.bincount(lab[ok].ravel(), minlength=len(axes)).astype(float)
    d["vol"] = vol / vol.sum()
    np.savez_compressed(cache, **d)
    print(f"  {tag}: {len(d['x'])} boundary faces, {len(axes)} grains",
          flush=True)
    return d


# ─────────────────────────── predictor family ─────────────────────────────
LOBES = (5.0, 10.0, 20.0, 40.0)
VARIANTS = (["rung3_flat", "unit_contrast", "cos1_rhat", "cos2_rhat",
             "cos1_dhat"]
            + [f"lobe{w:g}" for w in LOBES]
            + [f"lobe{w:g}_unit" for w in LOBES]
            + [f"lobe{w:g}_cononly" for w in LOBES]
            + ["ER2_volume"])


def predict(fac, rot_deg, f0):
    """All facet-model coda levels (dB, arbitrary common gain) at one azimuth.

    Frame: rotated_grid resamples specimen point R(+rot)x at solver point x
    and the probe sits on the solver +x rim, so in the SPECIMEN frame the
    probe is at azimuth +rot looking along -(cos rot, sin rot, 0).  This is
    the same convention fit_sweep.sweep_geo uses (azimuth_rad = radians(rot)).
    """
    h = float(fac["h"][0])
    axes, seeds = fac["axes"], fac["seeds"]
    a = np.radians(rot_deg)
    ca, sa = np.cos(a), np.sin(a)
    Rd = 0.050
    origin = np.array([Rd * ca, Rd * sa, 0.0])
    dhat = np.array([-ca, -sa, 0.0])
    k = 2 * np.pi * f0 / C_REF

    # per-grain qP impedance for this beam direction
    Zg = RHO * F.grain_speeds(axes, dhat)
    Za, Zb = Zg[fac["ga"]], Zg[fac["gb"]]
    Rc = (Zb - Za) / (Zb + Za)

    rx = fac["x"] - origin[0]
    ry = fac["y"] - origin[1]
    rz = fac["z"] - origin[2]
    r = np.sqrt(rx * rx + ry * ry + rz * rz)
    np.maximum(r, h, out=r)
    inv = 1.0 / r
    ux, uy, uz = rx * inv, ry * inv, rz * inv          # unit probe->facet
    cosang = ux * dhat[0] + uy * dhat[1]               # dhat_z = 0
    ang = np.arccos(np.clip(cosang, -1.0, 1.0))
    xa = k * A_EL * np.sin(ang)
    B = np.ones_like(xa)
    m = np.abs(xa) > 1e-9
    B[m] = 2.0 * j1(xa[m]) / xa[m]
    B[cosang <= 0] = 0.0

    nrm = fac["nrm"].astype(np.float64)
    # EXACT AREA weight: a wall of unit normal n crosses mesh axis `ax` with
    # face density |n.e_ax|, so weighting each face by |n.e_ax| makes the
    # three axis passes sum to n1^2+n2^2+n3^2 = 1 times the true wall area
    # (born.boundary_scatter face_weight='normal').  This removes the
    # voxelisation (L1) area inflation, which is orientation dependent.
    wA = np.abs(nrm[np.arange(len(nrm)), fac["ax"].astype(np.int64)])

    base = (h * h) * wA * (B ** 4) * inv ** 4          # geometry x area
    # monostatic specular deviation: perfect retro when n || probe->facet
    cdev = np.abs(nrm[:, 0] * ux + nrm[:, 1] * uy + nrm[:, 2] * uz)
    cdh = np.abs(nrm @ dhat)
    theta = np.arccos(np.clip(cdev, 0.0, 1.0))

    tb = np.rint((2.0 * r / C_REF) / DT).astype(np.int64)
    n_t = int(REC_END / DT)
    ok = (tb >= 0) & (tb < n_t)
    lo, hi = int(CODA_W[0] / DT), int(CODA_W[1] / DT)
    npul = max(int((1.0 / DT) / (FRAC_BW * f0)), 1)
    box = np.ones(npul) / npul

    tbo = tb[ok]

    def level(pw):
        acc = np.bincount(tbo, weights=pw[ok], minlength=n_t)[:n_t]
        env = np.sqrt(acc)
        if npul > 1:
            env = np.convolve(env, box, mode="same")
        return 20.0 * np.log10(np.sqrt((env[lo:hi] ** 2).mean()) + 1e-300)

    R2 = Rc ** 2
    out = {}
    out["rung3_flat"] = level(R2 * base)
    out["unit_contrast"] = level(base)
    out["cos1_rhat"] = level(R2 * base * cdev)
    out["cos2_rhat"] = level(R2 * base * cdev ** 2)
    out["cos1_dhat"] = level(R2 * base * cdh)
    for w in LOBES:
        g = np.exp(-(theta / np.radians(w)) ** 2)
        out[f"lobe{w:g}"] = level(R2 * base * g)
        # SAME specular geometry, contrast switched off: whatever this
        # explains is grain SHAPE/POSITION realisation, not fabric.
        out[f"lobe{w:g}_unit"] = level(base * g)
        # the fabric-only part of the specular predictor
        out[f"lobe{w:g}_cononly"] = (out[f"lobe{w:g}"]
                                     - out[f"lobe{w:g}_unit"])
    # volume E[R^2] baseline (the incumbent predictor)
    psi = np.arccos(np.clip(np.abs(axes @ (-dhat)), 0.0, 1.0))
    v = np.interp(psi, F._PSI, F._VQP)
    vol = fac["vol"]
    vb = float(vol @ v)
    var = float(vol @ (v - vb) ** 2)
    out["ER2_volume"] = 10.0 * np.log10(var / (2.0 * vb ** 2))
    return out


# ───────────────────────────── measurement ────────────────────────────────
def measured(d, cache=True):
    cf = os.path.join(SCR, "meas_" + os.path.basename(d) + ".npz")
    import time
    now = time.time()
    # skip anything the running sweep may still be writing
    files = sorted(f for f in os.listdir(d)
                   if f.startswith("az") and f.endswith(".npz")
                   and now - os.path.getmtime(os.path.join(d, f)) > 60)
    if cache and os.path.exists(cf):
        z = np.load(cf)
        if len(z["rots"]) == len(files):
            return z["rots"], z["lv"], z["raw"], z["e1"]
    rots, lv, raw, e1s = [], [], [], []
    for f in files:
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        e = np.abs(hilbert(tr))
        k0 = int(2 * 0.100 / C_REF * fs)
        w = int(2e-6 * fs)
        E1 = e[max(k0 - w, 0):k0 + w].max()
        c = np.sqrt((e[int(CODA_W[0] * fs):int(CODA_W[1] * fs)] ** 2).mean())
        rots.append(int(f[2:5]))
        lv.append(20.0 * np.log10(c / E1))
        raw.append(20.0 * np.log10(c))
        e1s.append(20.0 * np.log10(E1))
    out = (np.array(rots), np.array(lv), np.array(raw), np.array(e1s))
    if cache:
        np.savez(cf, rots=out[0], lv=out[1], raw=out[2], e1=out[3])
    return out


# ──────────────────────────────── stats ───────────────────────────────────
def stats(P, meas, ndraw=20000, seed=0):
    y = meas - meas.mean()
    ys = y / (np.linalg.norm(y) + 1e-300)
    Xs = P - P.mean(0)
    nrm = np.linalg.norm(Xs, axis=0)
    nrm[nrm < 1e-12] = 1.0
    Xs = Xs / nrm
    r_obs = Xs.T @ ys
    rng = np.random.default_rng(seed)
    n = len(meas)
    perm = np.empty((ndraw, n))
    for i in range(ndraw):
        perm[i] = ys[rng.permutation(n)]
    Rn = np.abs(perm @ Xs)
    p_ind = ((Rn >= np.abs(r_obs)[None, :]).sum(0) + 1) / (ndraw + 1)
    mx = Rn.max(1)
    p_fw = np.array([((mx >= abs(ro)).sum() + 1) / (ndraw + 1)
                     for ro in r_obs])
    rows = []
    for j, v in enumerate(VARIANTS):
        sl = np.polyfit(P[:, j], meas, 1)[0]
        fab = abs(sl) * P[:, j].std()
        res = np.sqrt(max(meas.var() - fab ** 2, 0.0))
        r95 = np.quantile(Rn[:, j], 0.95)
        rows.append((v, r_obs[j], p_ind[j], p_fw[j], fab, res,
                     r95 * meas.std()))
    return rows


def show(tag, rows):
    print(f"    -- observable: {tag}")
    print(f"    {'predictor':<15}{'r':>8}{'p':>9}{'p_fw':>9}"
          f"{'fab_dB':>9}{'res_dB':>9}{'null95':>9}")
    for v, ro, pi, pf, fab, res, n95 in rows:
        print(f"    {v:<15}{ro:>8.3f}{pi:>9.4f}{pf:>9.4f}"
              f"{fab:>9.2f}{res:>9.2f}{n95:>9.2f}")


def run(name, ppw_key="ppw", ndraw=20000, seed=0, step=1):
    d = os.path.join(OUT, name)
    with open(os.path.join(d, "config.json")) as f:
        cfgd = json.load(f)
    rot, meas, raw, e1 = measured(d)
    if step > 1:
        sel = (rot % step) == 0
        rot, meas, raw, e1 = rot[sel], meas[sel], raw[sel], e1[sel]
    fac = facets(cfgd, cfgd[ppw_key], name)
    f0 = cfgd["f0_mhz"] * 1e6
    P = np.array([[predict(fac, r, f0)[v] for v in VARIANTS] for r in rot])
    print(f"\n### {name}   n={len(rot)}  ppw={cfgd[ppw_key]}  "
          f"kappa={cfgd['concentration']}  seed={cfgd['seed']}")
    print(f"    coda/E1 mean {meas.mean():.2f} dB sd {meas.std(ddof=1):.2f}"
          f" | raw coda sd {raw.std(ddof=1):.2f} | E1 sd {e1.std(ddof=1):.2f}")
    ra = stats(P, meas, ndraw, seed)
    show("coda/E1 (dB)", ra)
    rb = stats(P, raw, ndraw, seed)
    show("raw coda (dB, no E1 normalisation)", rb)
    return dict(name=name, n=len(rot), rot=rot, meas=meas, raw=raw, e1=e1,
                P=P, rows=ra, rows_raw=rb)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["girdle_perp", "girdle_perp_ppw8", "iso_gcal",
                            "rigid_seed11", "gcheck_ppw8"]
    res = {}
    for nm in todo:
        res[nm] = run(nm)
    np.save(os.path.join(SCR, "facet_pred_results.npy"), res,
            allow_pickle=True)
