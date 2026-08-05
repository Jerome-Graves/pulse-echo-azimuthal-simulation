"""The ppw8 sweep, tested with a null that is actually valid.

An i.i.d. permutation destroys the azimuthal autocorrelation, so it
under-states the chance of a spurious correlation.  Two honest nulls:

  CIRCULAR SHIFT  rotate the measured series against the predictor.  This
      preserves the measurement's own autocorrelation exactly, and asks
      the real question: is the agreement at the TRUE relative angle
      better than at a wrong one?

  PHASE null       the predictor turns out to be nearly a pure 2-theta
      cosine, so correlating with it just asks "is there a 2-theta
      component at this phase".  Quantified here directly.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
import forward as F                              # noqa: E402
from specimen import DiskSpecimen                # noqa: E402

OUT = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "sweeps")))
C_REF, F0 = 3850.0, 2.0e6
CODA_W, BAND = (24e-6, 36e-6), (0.8e6, 3.0e6)

SW = [("girdle_seed11_ppw6_axis_perp", 6.0, -8.0, (1.0, 0.0, 0.0), 11),
      ("girdle_seed11_ppw8_dev", 8.0, -8.0, (1.0, 0.0, 0.0), 11),
      ("girdle_seed11_ppw6_axis_par", 6.0, -8.0, (0.0, 0.0, 1.0), 11),
      ("isotropic_seed41_ppw6_calibration", 6.0, 0.001, (1.0, 0.0, 0.0), 41)]


def load(d):
    rots, cd, e1 = [], [], []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        k0, w = int(2 * 0.100 / C_REF * fs), int(2e-6 * fs)
        if k0 + w >= len(tr):
            continue
        sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                     btype="band", output="sos")
        ef = np.abs(hilbert(sosfiltfilt(sos, tr)))
        e1.append(np.abs(hilbert(tr))[max(k0 - w, 0):k0 + w].max())
        cd.append(np.sqrt((ef[int(CODA_W[0] * fs):int(CODA_W[1] * fs)] ** 2)
                          .mean()))
        rots.append(int(f[2:5]))
    e1 = np.array(e1)
    return np.array(rots), 20 * np.log10(np.array(cd) / e1.mean())


def predict(ppw, kappa, axis, seed, rots):
    h = C_REF / F0 / ppw
    b = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                     size_cv=0.35, concentration=kappa, spatial_corr=0.0,
                     fabric_axis=axis, seed=seed).build(h)
    lab, ax = b["labels"], np.asarray(b["axes"], float)
    vol = np.bincount(lab[lab >= 0].ravel(), minlength=len(ax)).astype(float)
    k = vol > 0
    ax, vol = ax[k], vol[k] / vol[k].sum()
    out = []
    for r in rots:
        a = np.radians(r)
        n = np.array([np.cos(a), np.sin(a), 0.0])
        v = np.interp(np.arccos(np.clip(np.abs(ax @ n), 0, 1)), F._PSI, F._VQP)
        vb = float(vol @ v)
        out.append(10 * np.log10(float(vol @ (v - vb) ** 2) / (2 * vb ** 2)))
    return np.array(out)


def harm_frac(y, az):
    """what fraction of the predictor's variance is a single 2-theta term"""
    t = np.radians(az)
    A = np.c_[np.ones_like(t), np.cos(2 * t), np.sin(2 * t)]
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    fit = A @ c
    return 1 - ((y - fit).var() / y.var())


print(f"{'sweep':<19}{'n':>4}{'r':>8}{'2th frac':>10}"
      f"{'shift p':>9}{'rank':>10}")
for name, ppw, kap, axis, seed in SW:
    d = os.path.join(OUT, name)
    if not os.path.isdir(d):
        continue
    rots, meas = load(d)
    if len(rots) < 8:
        continue
    pred = predict(ppw, kap, axis, seed, rots)
    n = len(rots)
    r0 = np.corrcoef(pred, meas)[0, 1]
    # circular shift null: only valid on a uniformly sampled full circle
    rs = np.array([np.corrcoef(pred, np.roll(meas, k))[0, 1]
                   for k in range(n)])
    p = (np.sum(np.abs(rs) >= abs(r0))) / n
    rank = int(np.sum(np.abs(rs) > abs(r0))) + 1
    print(f"{name:<19}{n:>4}{r0:>8.3f}{harm_frac(pred, rots):>9.2f}"
          f"{p:>9.3f}{rank:>4}/{n:<5}")

print("\n2th frac = fraction of the predictor that is a single 2-theta cosine")
print("shift p  = fraction of circular shifts scoring at least as well")
print("rank     = where the TRUE alignment ranks among all n alignments")
print(f"floor on p with n azimuths is 1/n, so {1/60:.3f} for a 60-az sweep")
