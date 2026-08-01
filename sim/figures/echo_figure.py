"""Show what 'predicting when the echoes come back' actually means.

For a few azimuths of the converged ppw8 girdle sweep, plot the MEASURED
backscatter envelope against the echo train PREDICTED from the grain
boundaries alone: each boundary the beam crosses at distance s returns at
t = 2s/c, with a weight set by how square-on that facet is (its normal is
exactly perpendicular to the line joining the two Laguerre seeds).

Nothing is fitted.  The predicted times and weights come entirely from
the tessellation geometry.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
from scipy.signal import butter, hilbert, sosfiltfilt  # noqa: E402
from scipy.special import j1                           # noqa: E402

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward as F                                    # noqa: E402
from specimen import DiskSpecimen                      # noqa: E402

SW = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
C, F0, DIA = 3850.0, 2.0e6, 0.100
GATE, BAND = (24e-6, 36e-6), (0.8e6, 3.0e6)
LAM = C / F0
K = 2 * np.pi / LAM
DG, HALF = 17.4e-3, np.radians(8.9)
AZ = [0, 90, 180, 270]

h = C / F0 / 8.0
b = DiskSpecimen(diameter_m=DIA, thickness_m=0.035, n_grains=100,
                 size_cv=0.35, concentration=-8.0, spatial_corr=0.0,
                 fabric_axis=(1.0, 0.0, 0.0), seed=11).build(h)
lab, ax, seeds = b["labels"], np.asarray(b["axes"], float), \
    np.asarray(b["seeds"], float)
nx, ny, nz = lab.shape
print(f"specimen {lab.shape}, {len(ax)} grains, h={h*1e3:.3f} mm")


def predicted(rot, nlat=7):
    a = np.radians(rot)
    n = np.array([np.cos(a), np.sin(a), 0.0])
    t1 = np.array([-np.sin(a), np.cos(a), 0.0])
    t2 = np.array([0.0, 0.0, 1.0])
    vg = np.interp(np.arccos(np.clip(np.abs(ax @ n), 0, 1)), F._PSI, F._VQP)
    vbar = vg.mean()
    s = np.arange(GATE[0] * C / 2, GATE[1] * C / 2, h)
    uu = np.linspace(-1, 1, nlat)
    U, W = np.meshgrid(uu, uu, indexing="ij")
    keep = U ** 2 + W ** 2 <= 1.0
    U, W = U[keep], W[keep]
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
    x = K * DG * st
    D = np.where(x < 1e-9, 1.0,
                 2 * j1(np.where(x < 1e-9, 1.0, x))
                 / np.where(x < 1e-9, 1.0, x)) ** 2
    R2 = ((vg[ia] - vg[ib]) / (2 * vbar)) ** 2
    return 2 * si / C, R2 * D          # arrival time, power weight


fig, axs = plt.subplots(len(AZ), 1, figsize=(9, 9), sharex=True)
for k, rot in enumerate(AZ):
    with np.load(os.path.join(SW, "girdle_perp_ppw8", f"az{rot:03d}.npz")) as z:
        tr = np.asarray(z["trace"], float).ravel()
        dt = float(z["dt"])
    fs = 1.0 / dt
    sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                 btype="band", output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, tr)))
    i0, i1 = int(GATE[0] * fs), int(GATE[1] * fs)
    tt = np.arange(i0, i1) / fs * 1e6
    e = env[i0:i1] / env[i0:i1].max()
    ta, w = predicted(rot)
    # bin the predicted facets into pulse-length cells, then smooth
    edges = np.arange(GATE[0], GATE[1] + 1 / F0, 1 / F0)
    hgt, _ = np.histogram(ta, bins=edges, weights=w)
    ctr = (edges[:-1] + edges[1:]) / 2 * 1e6
    hgt = hgt / (hgt.max() + 1e-30)
    a = axs[k]
    a.plot(tt, e, lw=1.0, color="0.25", label="measured backscatter envelope")
    a.bar(ctr, hgt, width=0.45, color="tab:red", alpha=0.55,
          label="predicted echoes from grain boundaries")
    a.set_ylabel(f"az {rot}°")
    a.set_ylim(0, 1.05)
    r = np.corrcoef(np.interp(tt, ctr, hgt), e)[0, 1]
    a.text(0.985, 0.86, f"r = {r:+.2f}", ha="right", transform=a.transAxes)
    if k == 0:
        a.legend(loc="upper left", fontsize=8, framealpha=0.9)
axs[-1].set_xlabel("time (µs)      [depth along the beam = 1.925 mm per µs]")
axs[0].set_title("Predicted echo train from grain-boundary geometry alone, "
                 "vs measured backscatter\n"
                 "girdle specimen, ppw 8 converged, nothing fitted",
                 fontsize=10)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "echo_prediction.png")
fig.savefig(out, dpi=140)
print("wrote", out)
