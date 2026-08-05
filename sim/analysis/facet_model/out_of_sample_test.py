"""Out-of-sample test of born_spec, plus the clean converged fabric contrast.

singlemax_seed11_ppw8_twin finished after the born_spec result was obtained, so it is
a genuine prediction test: nothing was tuned on it.

It also completes the pair the mean-level question needed all along:
  girdle_seed11_ppw8_dev   seed 11, kappa -8.0  (girdle, normal in-plane)
  singlemax_seed11_ppw8_twin     seed 11, kappa +3.93 (single max, in-plane axis)
same tessellation, same ppw 8, same single-raster code path, same solver.
Identical grain GEOMETRY, only the c-axis orientations differ.  So the
facet layout is common and any level difference is fabric.
"""
import os as _os
import sys as _sys
# helper modules shared with sibling directories
for _d in ('..',):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vendor"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward as F                                   # noqa: E402
from length_scales import build, cone, shift_p, strip  # noqa: E402
from beam_descriptors import OUT                           # noqa: E402

C_REF, F0, DIA = 3850.0, 2.0e6, 0.100
CODA_W, BAND = (24e-6, 36e-6), (0.8e6, 3.0e6)

PAIR = [("singlemax_seed11_ppw8_twin", 3.93, (0.866, 0.5, 0.0), "single max k=+3.93"),
        ("girdle_seed11_ppw8_dev", -8.0, (1.0, 0.0, 0.0), "girdle k=-8 in-plane")]


def measure2(d):
    """coda (band-limited), and both references"""
    rots, cd, bang, e1 = [], [], [], []
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
        er, ef = np.abs(hilbert(tr)), np.abs(hilbert(sosfiltfilt(sos, tr)))
        rots.append(int(f[2:5]))
        bang.append(er.max())
        e1.append(er[max(k0 - w, 0):k0 + w].max())
        cd.append(np.sqrt((ef[int(CODA_W[0] * fs):int(CODA_W[1] * fs)] ** 2)
                          .mean()))
    return (np.array(rots), np.array(cd), np.array(bang), np.array(e1))


def neff(x):
    y = x - x.mean()
    n = len(y)
    ac = np.fft.irfft(np.abs(np.fft.rfft(y, 2 * n)) ** 2)[:n]
    ac /= ac[0]
    cut = np.argmax(ac < 0) if (ac < 0).any() else n // 4
    return max(n / max(1 + 2 * ac[1:cut].sum(), 1.0), 1.0)


print("=== 1. OUT-OF-SAMPLE: does born_spec predict the new sweep? ===")
print(f"{'specimen':<19}{'n':>4}  {'descriptor':<11}{'r':>8}{'p':>7}"
      f"{'  r resid':>10}{'p':>7}")
store = {}
for name, kap, axis, tag in PAIR:
    d = os.path.join(OUT, name)
    rots, cd, bang, e1 = measure2(d)
    coda = 20 * np.log10(cd / e1.mean())
    lab, ax, seeds, h = build(8.0, kap, axis, 11)
    D = cone(lab, ax, seeds, h, rots)
    store[name] = (rots, cd, bang, e1, coda, ax, lab, tag, kap, axis)
    for k in ("born_spec", "born_iso"):
        rf, pf = shift_p(D[k], coda)
        rr, pr = shift_p(strip(D[k], rots), strip(coda, rots))
        print(f"{name:<19}{len(rots):>4}  {k:<11}{rf:>8.3f}{pf:>7.3f}"
              f"{rr:>10.3f}{pr:>7.3f}{' *' if pf < 0.05 else ''}")

print("\n=== 2. CLEAN CONVERGED FABRIC CONTRAST ===")
print("same seed 11 tessellation, same ppw 8, only the c-axes differ\n")


def predE(ax, lab, rots):
    vol = np.bincount(lab[lab >= 0].ravel(), minlength=len(ax)).astype(float)
    k = vol > 0
    a2, v2 = ax[k], vol[k] / vol[k].sum()
    o = []
    for r in rots:
        t = np.radians(r)
        n = np.array([np.cos(t), np.sin(t), 0.0])
        v = np.interp(np.arccos(np.clip(np.abs(a2 @ n), 0, 1)),
                      F._PSI, F._VQP)
        vb = float(v2 @ v)
        o.append(10 * np.log10(float(v2 @ (v - vb) ** 2) / (2 * vb ** 2)))
    return np.array(o)


res = {}
print(f"{'specimen':<19}{'n':>4}{'Neff':>6}{'coda/source':>13}"
      f"{'coda/E1':>10}{'predicted':>11}")
for name, kap, axis, tag in PAIR:
    rots, cd, bang, e1, coda, ax, lab, tg, kp, axs = store[name]
    ls = 20 * np.log10(cd / bang.mean())
    le = 20 * np.log10(cd / e1.mean())
    pe = predE(ax, lab, rots)
    res[name] = (ls, le, pe, neff(ls))
    print(f"{name:<19}{len(rots):>4}{res[name][3]:>6.1f}{ls.mean():>12.2f}d"
          f"{le.mean():>9.2f}d{pe.mean():>10.2f}d")

a, b = res["singlemax_seed11_ppw8_twin"], res["girdle_seed11_ppw8_dev"]
print("\nSINGLE MAX minus GIRDLE (identical grains, only the fabric moved):")
for lbl, i in (("coda / source", 0), ("coda / backwall", 1)):
    dm = a[i].mean() - b[i].mean()
    se = np.hypot(a[i].std(ddof=1) / np.sqrt(a[3]),
                  b[i].std(ddof=1) / np.sqrt(b[3]))
    print(f"  measured {lbl:<16}{dm:+7.2f} +- {se:.2f} dB  "
          f"({abs(dm)/se:.1f} sigma)")
dp = a[2].mean() - b[2].mean()
print(f"  PREDICTED from the grains {dp:+7.2f} dB")
print(f"  agreement (source-referenced): "
      f"{a[0].mean()-b[0].mean()-dp:+.2f} dB")
