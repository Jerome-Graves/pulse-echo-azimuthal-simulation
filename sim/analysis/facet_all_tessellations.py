"""TEST 1: does the facet-geometry model predict the coda on ALL FOUR
tessellations, or only on seed 11?

Predictors (exactly as in cone_specular.py / decisive.py):
  geom_only  SUM over beam-crossed facets of the specular directivity
             |2J1(k Dg sin th)/x|^2 -- pure GEOMETRY, no fabric at all
  born_spec  same, weighted by (dv/2vbar)^2 -- geometry x fabric
  born_iso   SUM of (dv/2vbar)^2, no orientation weighting -- fabric only
  n_cross    raw count of facet crossings (naive geometry baseline)

Scoring: circular-shift null (floor 1/n), plus harmonic stripping
(mean, 2-theta, 4-theta removed from BOTH series).
"""
import json
import os
import sys
import time

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import j1

SIM = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim"
sys.path.insert(0, SIM)
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
import forward as F                                   # noqa: E402
from specimen import DiskSpecimen                     # noqa: E402

OUT = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
SCR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCR, "t1cache")
os.makedirs(CACHE, exist_ok=True)

C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
CODA_W, BAND = (24e-6, 36e-6), (0.8e6, 3.0e6)
LAM = C_REF / F0
HALF = np.radians(8.9)          # far-field divergence half-angle
DG = 17.4e-3                    # mean grain diameter -> specular lobe width
K = 2 * np.pi / LAM
KEYS = ("geom_only", "born_spec", "born_iso", "n_cross")

# name, ppw, kappa, axis, seed, is_isotropic, raster
SWEEPS = [
    ("rigid_seed11",     6.0,  3.93, (0.866, 0.5, 0.0),    11, False, "double"),
    ("girdle_perp",      6.0, -8.00, (1.0, 0.0, 0.0),      11, False, "single"),
    ("girdle_par",       6.0, -8.00, (0.0, 0.0, 1.0),      11, False, "single"),
    ("kappa8_seed17",    6.0,  8.00, (0.5, 0.866, 0.0),    17, False, "double"),
    ("oos_seed23",       6.0,  3.93, (-0.342, 0.94, 0.0),  23, False, "double"),
    ("iso_gcal",         6.0, 0.001, (1.0, 0.0, 0.0),      41, True,  "double"),
    ("girdle_perp_ppw8", 8.0, -8.00, (1.0, 0.0, 0.0),      11, False, "single"),
    ("singlemax_ppw8",   8.0,  3.93, (0.866, 0.5, 0.0),    11, False, "single"),
]
SMALL = [
    ("girdle_20",        6.0, -8.00, (0.342, 0.0, 0.94),   11, False, "single"),
    ("gcheck_ppw8",      8.0,  3.93, (0.866, 0.5, 0.0),    11, False, "double"),
]
MAXN = 120


def measure(d):
    rots, cd, e1 = [], [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        k0, w = int(2 * DIA / C_REF * fs), int(2e-6 * fs)
        if k0 + w >= len(tr):
            continue
        sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                     btype="band", output="sos")
        ef = np.abs(hilbert(sosfiltfilt(sos, tr)))        # filter WHOLE trace
        e1.append(np.abs(hilbert(tr))[max(k0 - w, 0):k0 + w].max())
        cd.append(np.sqrt((ef[int(CODA_W[0] * fs):int(CODA_W[1] * fs)] ** 2)
                          .mean()))
        rots.append(int(f[2:5]))
    return np.array(rots), 20 * np.log10(np.array(cd) / np.mean(e1))


def build_cached(ppw, kappa, axis, seed):
    h = C_REF / F0 / ppw
    tag = f"s{seed}_p{ppw:g}_k{kappa:g}_a{axis[0]:g}_{axis[1]:g}_{axis[2]:g}"
    p = os.path.join(CACHE, tag + ".npz")
    if os.path.exists(p):
        with np.load(p) as z:
            return z["labels"], z["axes"], z["seeds"], h
    t = time.time()
    b = DiskSpecimen(diameter_m=DIA, thickness_m=0.035, n_grains=100,
                     size_cv=0.35, concentration=kappa, spatial_corr=0.0,
                     fabric_axis=axis, seed=seed).build(h)
    lab = np.asarray(b["labels"])
    ax = np.asarray(b["axes"], float)
    sd = np.asarray(b["seeds"], float)
    np.savez_compressed(p, labels=lab, axes=ax, seeds=sd)
    print(f"    built {tag} in {time.time()-t:.0f}s  "
          f"grid{lab.shape} ngrain={len(sd)}", flush=True)
    return lab, ax, sd, h


def preds(lab, ax, seeds, h, rots, nlat=7, sign=1.0):
    """Diverging-cone sampling; exact Laguerre facet normals."""
    nx, ny, nz = lab.shape
    d0, d1 = CODA_W[0] * C_REF / 2, CODA_W[1] * C_REF / 2
    s = np.arange(d0, d1, h)
    uu = np.linspace(-1, 1, nlat)
    U, W = np.meshgrid(uu, uu, indexing="ij")
    keep = U ** 2 + W ** 2 <= 1.0
    U, W = U[keep], W[keep]
    out = {k: [] for k in KEYS}
    for r in rots:
        a = sign * np.radians(r)
        n = np.array([np.cos(a), np.sin(a), 0.0])
        t1 = np.array([-np.sin(a), np.cos(a), 0.0])
        t2 = np.array([0.0, 0.0, 1.0])
        vg = np.interp(np.arccos(np.clip(np.abs(ax @ n), 0, 1)),
                       F._PSI, F._VQP)
        vbar = vg.mean()
        rad = (s * np.tan(HALF))[:, None]
        P = (DIA / 2 * n - s[:, None] * n)[:, None, :] \
            + (rad * U[None, :])[:, :, None] * t1 \
            + (rad * W[None, :])[:, :, None] * t2
        gi = np.rint((P[..., 0] + nx * h / 2) / h - 0.5).astype(int)
        gj = np.rint((P[..., 1] + ny * h / 2) / h - 0.5).astype(int)
        gk = np.rint((P[..., 2] + nz * h / 2) / h - 0.5).astype(int)
        ok = ((gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
              & (gk >= 0) & (gk < nz))
        L = np.where(ok, lab[np.clip(gi, 0, nx - 1), np.clip(gj, 0, ny - 1),
                             np.clip(gk, 0, nz - 1)], -1)
        A, B = L[:-1], L[1:]
        cr = (A != B) & (A >= 0) & (B >= 0)
        ia, ib = A[cr], B[cr]
        nm = seeds[ia] - seeds[ib]
        nm /= np.linalg.norm(nm, axis=1, keepdims=True) + 1e-30
        st = np.sqrt(np.clip(1 - (nm @ n) ** 2, 0, 1))
        x = K * DG * st
        D = np.where(x < 1e-9, 1.0,
                     2 * j1(np.where(x < 1e-9, 1.0, x))
                     / np.where(x < 1e-9, 1.0, x)) ** 2
        R2 = ((vg[ia] - vg[ib]) / (2 * vbar)) ** 2
        out["geom_only"].append(float(D.sum()))
        out["born_spec"].append(float((R2 * D).sum()))
        out["born_iso"].append(float(R2.sum()))
        out["n_cross"].append(float(cr.sum()))
    return {k: np.array(v, float) for k, v in out.items()}


def shift_p(x, y):
    """Circular-shift null.  Two-sided on |r|.  Floor 1/n."""
    n = len(y)
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return np.nan, np.nan
    r0 = np.corrcoef(x, y)[0, 1]
    rs = np.array([np.corrcoef(x, np.roll(y, k))[0, 1] for k in range(n)])
    return float(r0), float(np.sum(np.abs(rs) >= abs(r0)) / n)


def strip(y, az):
    t = np.radians(az)
    A = np.column_stack([np.ones_like(t), np.cos(2 * t), np.sin(2 * t),
                         np.cos(4 * t), np.sin(4 * t)])
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def neff(y):
    """effective n from the circular autocorrelation integral length"""
    n = len(y)
    z = y - y.mean()
    ac = np.array([np.dot(z, np.roll(z, k)) for k in range(n)]) / np.dot(z, z)
    k = 1
    while k < n // 2 and ac[k] > 0:
        k += 1
    tau = 1 + 2 * ac[1:k].sum()
    return max(2.0, n / max(tau, 1.0)), tau


def fisher_ci(r, ne):
    if ne <= 3 or not np.isfinite(r):
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    s = 1.96 / np.sqrt(ne - 3)
    return (float(np.tanh(z - s)), float(np.tanh(z + s)))


def holm(pv):
    pv = np.asarray(pv, float)
    m = len(pv)
    o = np.argsort(pv)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (m - i) * pv[idx])
        adj[idx] = min(1.0, run)
    return adj


# ---------------------------------------------------------------- load
DATA = {}
for name, ppw, kap, axis, seed, iso, ras in SWEEPS + SMALL:
    d = os.path.join(OUT, name)
    rots, coda = measure(d)
    if len(rots) > MAXN:
        sel = np.arange(0, len(rots), len(rots) // MAXN)
        rots, coda = rots[sel], coda[sel]
    lab, ax, sd, h = build_cached(ppw, kap, axis, seed)
    DATA[name] = dict(rots=rots, coda=coda, lab=lab, ax=ax, sd=sd, h=h,
                      ppw=ppw, seed=seed, iso=iso, ras=ras, kap=kap)
    print(f"loaded {name:<18} n={len(rots):>3} coda {coda.mean():7.2f} "
          f"+- {coda.std():.2f} dB", flush=True)

# --------------------------------------------- 0. rotation-sign calibration
print("\n=== 0. ROTATION-SIGN CALIBRATION on the reference sweep ===")
ref = DATA["girdle_perp_ppw8"]
for sg in (+1.0, -1.0):
    P = preds(ref["lab"], ref["ax"], ref["sd"], ref["h"], ref["rots"], sign=sg)
    r, p = shift_p(strip(P["geom_only"], ref["rots"]),
                   strip(ref["coda"], ref["rots"]))
    rb, pb = shift_p(strip(P["born_spec"], ref["rots"]),
                     strip(ref["coda"], ref["rots"]))
    print(f"  sign {sg:+.0f}: geom_only resid r={r:+.3f} p={p:.3f}   "
          f"born_spec resid r={rb:+.3f} p={pb:.3f}")

# ---------------------------------------------------------------- 1. main
print("\n=== 1. PER-SWEEP PREDICTOR PANEL (sign +1) ===")
print(f"{'sweep':<18}{'seed':>5}{'ppw':>4}{'ras':>7}{'n':>5}  "
      f"{'predictor':<10}{'r':>7}{'p':>7}{'r_res':>8}{'p_res':>7}"
      f"{'  95%CI(res)':>18}{'  n_eff':>7}")
ROWS = []
PRED = {}
for name, ppw, kap, axis, seed, iso, ras in SWEEPS + SMALL:
    D = DATA[name]
    P = preds(D["lab"], D["ax"], D["sd"], D["h"], D["rots"])
    PRED[name] = P
    ne, tau = neff(strip(D["coda"], D["rots"]))
    for k in KEYS:
        r, p = shift_p(P[k], D["coda"])
        rr, pr = shift_p(strip(P[k], D["rots"]), strip(D["coda"], D["rots"]))
        lo, hi = fisher_ci(rr, ne)
        ROWS.append(dict(sweep=name, seed=seed, ppw=ppw, ras=ras, iso=iso,
                         n=len(D["rots"]), pred=k, r=r, p=p, rr=rr, pr=pr,
                         lo=lo, hi=hi, ne=ne, small=(name in
                                                     [s[0] for s in SMALL])))
        print(f"{name:<18}{seed:>5}{ppw:>4.0f}{ras:>7}{len(D['rots']):>5}  "
              f"{k:<10}{r:>7.3f}{p:>7.3f}{rr:>8.3f}{pr:>7.3f}"
              f"   [{lo:+.2f},{hi:+.2f}]{ne:>7.1f}"
              f"{'  ISO' if iso else ''}", flush=True)
    print()

# ------------------------------------------- 2. cross-tessellation control
print("=== 2. CROSS-TESSELLATION WRONG-PREDICTOR CONTROL (geom_only) ===")
print("   rows: geometry taken from   cols: coda measured on")
main = [s[0] for s in SWEEPS]
XT = np.full((len(main), len(main)), np.nan)
for i, pn in enumerate(main):
    Dp = DATA[pn]
    for j, cn in enumerate(main):
        Dc = DATA[cn]
        # geometry predictor from specimen pn evaluated at cn's azimuths
        Pi = preds(Dp["lab"], Dp["ax"], Dp["sd"], Dp["h"], Dc["rots"])
        rr, pr = shift_p(strip(Pi["geom_only"], Dc["rots"]),
                         strip(Dc["coda"], Dc["rots"]))
        XT[i, j] = rr
hdr = "".join(f"{m[:9]:>10}" for m in main)
print(f"{'':<19}{hdr}")
for i, pn in enumerate(main):
    print(f"{pn:<19}" + "".join(f"{XT[i,j]:>10.3f}" for j in range(len(main))))

np.savez(os.path.join(SCR, "t1_results.npz"),
         rows=json.dumps(ROWS, default=float), XT=XT, main=main)

# ---------------------------------------------------------------- 3. Holm
print("\n=== 3. MULTIPLE COMPARISONS (primary family) ===")
prim = [r for r in ROWS if not r["small"] and r["pred"] in
        ("geom_only", "born_spec", "born_iso")]
adj = holm([r["pr"] for r in prim])
print(f"  family size m = {len(prim)}  (3 predictors x {len(SWEEPS)} sweeps,"
      f" residual-space circular-shift p)")
print(f"  {'sweep':<18}{'predictor':<11}{'p_res':>8}{'Holm':>8}")
for r, a in sorted(zip(prim, adj), key=lambda t: t[1]):
    print(f"  {r['sweep']:<18}{r['pred']:<11}{r['pr']:>8.3f}{a:>8.3f}"
          f"{' *' if a < 0.05 else ''}")

# ---------------------------------------------------------- 4. var split
print("\n=== 4. VARIANCE SPLIT (residual space, r^2) ===")
print(f"  {'sweep':<18}{'geom%':>8}{'full%':>8}{'fabric+%':>10}{'unmod%':>8}")
for name, *_ in SWEEPS:
    g = [r for r in ROWS if r["sweep"] == name and r["pred"] == "geom_only"][0]
    b = [r for r in ROWS if r["sweep"] == name and r["pred"] == "born_spec"][0]
    gg, bb = 100 * g["rr"] ** 2, 100 * b["rr"] ** 2
    print(f"  {name:<18}{gg:>8.1f}{bb:>8.1f}{bb-gg:>10.1f}{100-bb:>8.1f}")
