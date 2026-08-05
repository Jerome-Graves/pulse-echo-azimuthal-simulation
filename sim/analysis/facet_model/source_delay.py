"""ITEM 4e - tighten the detector: apply the SOURCE-WAVELET delay
t0 = 1.2/f0 (fdtd.ricker default) to the ray-model boundary times, which the
historical blip test never did, and narrow the peak-search half-window."""
import os, sys
import numpy as np
from scipy.signal import hilbert
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vendor"))))
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import adv_common as A
import acquisition as AQ
from config import Config
import item4b_blips as B

for name in ("singlemax_seed11_ppw6_rigid2", "singlemax_seed7_ppw6_5mhz_production"):
    build = B.get_build(name); cfg = Config(); cfg.err.enabled = False
    az, X, dt, f0, dts = A.load_sweep(name); fs = 1/dt
    w = B.ricker(f0, dt, int(round(3.0/f0/dt)) | 1); w /= np.linalg.norm(w)
    E = np.abs(hilbert(np.stack([np.convolve(x, w[::-1], "same") for x in X]), axis=1))
    T = {int(a): AQ.truth_along_diameter(build, cfg, np.radians(a)) for a in az}
    T = {k: 2*np.cumsum(v["length_m"]/v["speed"])[:-1] for k, v in T.items()}
    t0 = 1.2 / f0
    print(f"\n== {name} f0 {f0/1e6:.0f} MHz, source t0 = {t0*1e6:.2f} us")
    for half_us in (1.0, 0.5, 0.3):
        half, ring = int(half_us*1e-6*fs), int(4e-6*fs)
        for shift, lab in ((0.0, "no t0 correction (historical)"),
                           (t0, "t0-corrected")):
            hit, nul = [], []
            for i, a in enumerate(az):
                ts = T[int(a)]; ts = ts[(ts > 8e-6) & (ts < 46e-6)] + shift
                for t in ts:
                    v = B.excess(E[i], int(t*fs), half, ring)
                    if np.isfinite(v): hit.append(v)
                    for off in (-3e-6, 3e-6):
                        v2 = B.excess(E[i], int((t+off)*fs), half, ring)
                        if np.isfinite(v2): nul.append(v2)
            hit, nul = np.array(hit), np.array(nul)
            s = (hit.mean()-nul.mean())/np.hypot(hit.std()/np.sqrt(len(hit)),
                                                 nul.std()/np.sqrt(len(nul)))
            print(f"   half {half_us:.1f} us | {lab:28s} excess "
                  f"{hit.mean()-nul.mean():+.3f} dB  ({s:.1f} sigma, n={len(hit)})")
