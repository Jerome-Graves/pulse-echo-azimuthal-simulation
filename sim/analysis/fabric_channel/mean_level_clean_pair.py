"""Does the MEAN coda level respond to fabric?  The one clean pair.

girdle_seed11_ppw6_axis_perp  girdle normal IN-PLANE (+x)   seed 11, ppw6, single raster
girdle_seed11_ppw6_axis_par   girdle normal AXIAL   (+z)    seed 11, ppw6, single raster

Same tessellation (same seed => identical grain shapes and positions),
same fabric strength (kappa = -8), same solver settings, same code path.
ONLY the orientation of the c-axis distribution differs.  So any
difference in mean coda level is fabric, not geometry and not numerics.

Normalised by the SOURCE peak, not the backwall: the backwall is itself
attenuated by scattering, so dividing by it would double-count the very
effect being measured.

Error bars use the EFFECTIVE number of independent azimuths (from the
measured azimuthal autocorrelation), not the raw count of 60.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vendor"))))
import forward as F                              # noqa: E402
from specimen import DiskSpecimen                # noqa: E402

OUT = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "out", "sweeps")))
C_REF, F0 = 3850.0, 2.0e6
CODA_W, BAND = (24e-6, 36e-6), (0.8e6, 3.0e6)

PAIR = [("girdle_seed11_ppw6_axis_perp", (1.0, 0.0, 0.0), "girdle normal in-plane"),
        ("girdle_seed11_ppw6_axis_par", (0.0, 0.0, 1.0), "girdle normal axial")]


def load(d):
    rots, cd, bang, e1 = [], [], [], []
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
        ef, er = np.abs(hilbert(sosfiltfilt(sos, tr))), np.abs(hilbert(tr))
        rots.append(int(f[2:5]))
        bang.append(er.max())
        e1.append(er[max(k0 - w, 0):k0 + w].max())
        cd.append(np.sqrt((ef[int(CODA_W[0] * fs):int(CODA_W[1] * fs)] ** 2)
                          .mean()))
    return (np.array(rots), np.array(cd), np.array(bang), np.array(e1))


def neff(x):
    """independent samples from the integral azimuthal correlation length"""
    y = x - x.mean()
    n = len(y)
    ac = np.fft.irfft(np.abs(np.fft.rfft(y, 2 * n)) ** 2)[:n]
    ac /= ac[0]
    cut = np.argmax(ac < 0) if (ac < 0).any() else n // 4
    tau = 1 + 2 * ac[1:cut].sum()
    return max(n / max(tau, 1.0), 1.0), tau


def predict(kappa, axis, seed, rots, ppw=6.0):
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


res = {}
print(f"{'sweep':<14}{'n':>4}{'Neff':>6}{'coda/source':>13}{'coda/E1':>10}"
      f"{'predicted':>11}")
for name, axis, tag in PAIR:
    rots, cd, bang, e1 = load(os.path.join(OUT, name))
    ls = 20 * np.log10(cd / bang.mean())
    le = 20 * np.log10(cd / e1.mean())
    ne, tau = neff(ls)
    pr = predict(-8.0, axis, 11, rots)
    res[name] = (ls, le, ne, pr, tag)
    print(f"{name:<14}{len(rots):>4}{ne:>6.1f}{ls.mean():>12.2f}d"
          f"{le.mean():>9.2f}d{pr.mean():>10.2f}d")

a, b = res["girdle_seed11_ppw6_axis_perp"], res["girdle_seed11_ppw6_axis_par"]
print("\nIN-PLANE minus AXIAL girdle (same grains, only the fabric axis moved):")
for lbl, i in (("coda / source", 0), ("coda / backwall", 1)):
    dm = a[i].mean() - b[i].mean()
    se = np.hypot(a[i].std(ddof=1) / np.sqrt(a[2]),
                  b[i].std(ddof=1) / np.sqrt(b[2]))
    print(f"  measured {lbl:<16} {dm:+6.2f} +- {se:.2f} dB   "
          f"({abs(dm)/se:.1f} sigma)")
dp = a[3].mean() - b[3].mean()
print(f"  PREDICTED from the grains  {dp:+6.2f} dB")
print(f"\n  agreement (coda/source vs predicted): "
      f"{a[0].mean()-b[0].mean()-dp:+.2f} dB")
