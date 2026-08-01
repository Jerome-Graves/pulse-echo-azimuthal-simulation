"""A display that actually shows the grain boundaries in the coda.

Three fixes over the four-trace version:

1. SMOOTH TO THE MODEL'S RESOLUTION.  The fast wiggle in the envelope is
   interference between echoes that overlap because the pulse is ~0.5 us
   long.  It is real signal, not carrier, so the Hilbert envelope does not
   remove it; only averaging over a pulse length does.  1.0 us was measured
   to be optimal.
2. SHOW ALL 60 AZIMUTHS AT ONCE as an image, instead of four traces.  The
   agreement is a weak effect spread over the whole field, so four slices
   can never show it.
3. PUT A WRONG GRAIN ARRANGEMENT NEXT TO IT.  Without that control the
   picture proves nothing, because any two blobby images look alike.

Both fields are centred exactly as the statistics centre them: the depth
profile is removed at every time, and each azimuth's own level is removed,
so what remains is only WHERE IN TIME the backscatter sits.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
from scipy.ndimage import uniform_filter1d             # noqa: E402
from scipy.signal import butter, hilbert, sosfiltfilt  # noqa: E402
from scipy.special import j1                           # noqa: E402

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
import forward as F                                    # noqa: E402
from specimen import DiskSpecimen                      # noqa: E402

SWD = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
C, F0, DIA = 3850.0, 2.0e6, 0.100
GATE, BAND = (24e-6, 36e-6), (0.8e6, 3.0e6)
DG, HALF = 17.4e-3, np.radians(8.9)
KP = 2 * np.pi * F0 / C
NT = 240
TG = np.linspace(GATE[0], GATE[1], NT)
h = C / F0 / 8.0


def spec(seed, kappa=-8.0, axis=(1.0, 0.0, 0.0)):
    b = DiskSpecimen(diameter_m=DIA, thickness_m=0.035, n_grains=100,
                     size_cv=0.35, concentration=kappa, spatial_corr=0.0,
                     fabric_axis=axis, seed=seed).build(h)
    return (b["labels"], np.asarray(b["axes"], float),
            np.asarray(b["seeds"], float))


def train(S, rot, nlat=7):
    lab, axg, seeds = S
    nx, ny, nz = lab.shape
    a = np.radians(rot)
    n = np.array([np.cos(a), np.sin(a), 0.0])
    t1 = np.array([-np.sin(a), np.cos(a), 0.0])
    t2 = np.array([0.0, 0.0, 1.0])
    vg = np.interp(np.arccos(np.clip(np.abs(axg @ n), 0, 1)), F._PSI, F._VQP)
    s = np.arange(GATE[0] * C / 2, GATE[1] * C / 2, h)
    uu = np.linspace(-1, 1, nlat)
    U, W = np.meshgrid(uu, uu, indexing="ij")
    k = U ** 2 + W ** 2 <= 1.0
    U, W = U[k], W[k]
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
    si = np.broadcast_to(s[:-1, None], A.shape)[cr]
    ia, ib = A[cr], B[cr]
    nm = seeds[ia] - seeds[ib]
    nm /= np.linalg.norm(nm, axis=1, keepdims=True) + 1e-30
    st = np.sqrt(np.clip(1 - (nm @ n) ** 2, 0, 1))
    x = KP * DG * st
    D = np.where(x < 1e-9, 1.0, 2 * j1(np.where(x < 1e-9, 1.0, x))
                 / np.where(x < 1e-9, 1.0, x)) ** 2
    v = 0.5 * (vg[ia] + vg[ib])
    R2 = ((vg[ia] - vg[ib]) / (2 * vg.mean())) ** 2
    y = np.zeros(NT)
    # + the SOURCE DELAY: fdtd.ricker uses t0 = 1.2/f0, so every arrival is
    # late by that against the geometric time. Omitting it put the whole
    # prediction 0.6 us early, which showed up as the lag-curve peak sitting
    # at +0.45 us instead of 0. A documented constant, not a fitted one.
    t = 2 * si / v + 1.2 / F0
    m = (t >= GATE[0]) & (t < GATE[1])
    np.add.at(y, np.clip(np.searchsorted(TG, t[m]), 0, NT - 1),
              (R2 * D)[m])
    return uniform_filter1d(y, max(int(NT / (GATE[1] - GATE[0]) / F0), 3))


d = os.path.join(SWD, "girdle_perp_ppw8")
rots = sorted(int(f[2:5]) for f in os.listdir(d)
              if f.startswith("az") and f.endswith(".npz"))
M = []
for r in rots:
    with np.load(os.path.join(d, f"az{r:03d}.npz")) as z:
        tr, dt = np.asarray(z["trace"], float).ravel(), float(z["dt"])
    fs = 1.0 / dt
    sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                 btype="band", output="sos")
    e = uniform_filter1d(np.abs(hilbert(sosfiltfilt(sos, tr))),
                         int(1.0e-6 * fs))
    M.append(np.interp(TG, np.arange(len(e)) / fs, e))
M = np.array(M)
print("measured field", M.shape)

TRUE = spec(11)
WRONG = [spec(s) for s in (77, 91, 123)]
PT = np.array([train(TRUE, r) for r in rots])
PW = [np.array([train(S, r) for r in rots]) for S in WRONG]
print("predictions built")


def cen(X):
    X = X - X.mean(0, keepdims=True)
    return X - X.mean(1, keepdims=True)


def nrm(X):
    X = cen(X)
    return X / X.std()


Mc, Pc = nrm(M), nrm(PT)
Wc = [nrm(w) for w in PW]
rt = np.corrcoef(Mc.ravel(), Pc.ravel())[0, 1]
rw = [np.corrcoef(Mc.ravel(), w.ravel())[0, 1] for w in Wc]

ext = [TG[0] * 1e6, TG[-1] * 1e6, rots[0], rots[-1]]
fig = plt.figure(figsize=(12, 7.2))
gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.0], hspace=0.38,
                      wspace=0.22)
for j, (X, ttl) in enumerate([
        (Mc, "MEASURED backscatter\n(simulated traces)"),
        (Pc, f"PREDICTED from the TRUE grain\narrangement    r = {rt:+.3f}"),
        (Wc[0], f"PREDICTED from a WRONG grain\narrangement    r = {rw[0]:+.3f}")]):
    a = fig.add_subplot(gs[0, j])
    a.imshow(X, aspect="auto", origin="lower", extent=ext,
             cmap="RdBu_r", vmin=-2.2, vmax=2.2, interpolation="bilinear")
    a.set_title(ttl, fontsize=10)
    a.set_xlabel("time (µs)")
    if j == 0:
        a.set_ylabel("azimuth (degrees)")

a = fig.add_subplot(gs[1, :])
lags = np.arange(-60, 61)
a.plot(lags * (TG[1] - TG[0]) * 1e6,
       [np.corrcoef(Mc.ravel(), np.roll(Pc, L, axis=1).ravel())[0, 1]
        for L in lags], lw=2.0, color="tab:red",
       label="true grain arrangement")
for i, w in enumerate(Wc):
    a.plot(lags * (TG[1] - TG[0]) * 1e6,
           [np.corrcoef(Mc.ravel(), np.roll(w, L, axis=1).ravel())[0, 1]
            for L in lags], lw=1.0, color="0.6",
           label="wrong arrangements" if i == 0 else None)
a.axvline(0, color="0.3", lw=0.7, ls=":")
a.axhline(0, color="0.3", lw=0.7)
a.set_xlabel("depth registration error (µs)      "
             "[peak at zero = correct depth registration]")
a.set_ylabel("correlation")
a.legend(fontsize=9, loc="upper right")
fig.suptitle("Are the grain boundaries visible in the coda?   "
             "girdle specimen, ppw 8 converged, 60 azimuths, nothing fitted",
             fontsize=11)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "coda_boundaries.png")
fig.savefig(out, dpi=145, bbox_inches="tight")
print(f"true r = {rt:+.4f}   wrong r = "
      + ", ".join(f"{x:+.4f}" for x in rw))
print("wrote", out)
