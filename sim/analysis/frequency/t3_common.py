"""Shared plumbing for TEST 3 (frequency scaling of the facet model)."""
import json
import os
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vendor"))))

OUT = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "out", "sweeps")))
C_REF, DIA = 3850.0, 0.100
CODA_W = (24e-6, 36e-6)
BAND_FRAC = (0.4, 1.5)          # 0.8-3.0 MHz at f0 = 2 MHz


def cfg_of(name):
    with open(os.path.join(OUT, name, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def az_list(name):
    d = os.path.join(OUT, name)
    return sorted(int(f[2:5]) for f in os.listdir(d)
                  if f.startswith("az") and f.endswith(".npz")
                  and "tmp" not in f)


def load_trace(name, az):
    with np.load(os.path.join(OUT, name, f"az{az:03d}.npz")) as z:
        return np.asarray(z["trace"], float).ravel(), float(z["dt"])


def sos_of(fs, f0):
    ny = fs / 2
    lo, hi = BAND_FRAC[0] * f0, min(BAND_FRAC[1] * f0, 0.95 * ny)
    return butter(4, [lo / ny, hi / ny], btype="band", output="sos"), lo, hi


def measure(name, azs=None, gate=CODA_W, filt_e1=False):
    """Coda RMS (dB re sweep-mean backwall peak) per azimuth.

    Filter the WHOLE trace, then gate.  Returns dict with rot, coda_db,
    coda_lin (linear, same normalisation), e1 (per-az backwall peak).
    """
    c = cfg_of(name)
    f0 = c["f0_mhz"] * 1e6
    azs = az_list(name) if azs is None else azs
    rots, cd, e1 = [], [], []
    for a in azs:
        tr, dt = load_trace(name, a)
        fs = 1.0 / dt
        sos, _, _ = sos_of(fs, f0)
        trf = sosfiltfilt(sos, tr)
        ef = np.abs(hilbert(trf))
        k0 = int(2 * DIA / C_REF * fs)
        w = int(2e-6 * fs)
        if k0 + w >= len(tr):
            raise RuntimeError(f"{name} az{a}: record too short")
        src = ef if filt_e1 else np.abs(hilbert(tr))
        e1.append(src[max(k0 - w, 0):k0 + w].max())
        g = ef[int(gate[0] * fs):int(gate[1] * fs)]
        cd.append(np.sqrt((g ** 2).mean()))
        rots.append(a)
    e1 = np.array(e1)
    cd = np.array(cd)
    return dict(rot=np.array(rots), e1=e1, coda_raw=cd,
                coda_lin=cd / e1.mean(),
                coda_db=20 * np.log10(cd / e1.mean()), f0=f0)


def envelopes(name, azs=None, gate=CODA_W):
    """Band-passed envelope inside the gate for every azimuth (2-D array)."""
    c = cfg_of(name)
    f0 = c["f0_mhz"] * 1e6
    azs = az_list(name) if azs is None else azs
    E, ts = [], None
    for a in azs:
        tr, dt = load_trace(name, a)
        fs = 1.0 / dt
        sos, _, _ = sos_of(fs, f0)
        ef = np.abs(hilbert(sosfiltfilt(sos, tr)))
        i0, i1 = int(gate[0] * fs), int(gate[1] * fs)
        E.append(ef[i0:i1])
        ts = (dt, i0, i1)
    n = min(len(e) for e in E)
    return np.array([e[:n] for e in E]), ts


def shift_p(x, y):
    """Circular-shift null.  Returns (r, p, null distribution)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(y)
    r0 = np.corrcoef(x, y)[0, 1]
    rs = np.array([np.corrcoef(x, np.roll(y, k))[0, 1] for k in range(n)])
    return r0, float(np.sum(np.abs(rs) >= abs(r0)) / n), rs


def strip(y, az):
    """Remove mean, 2-theta and 4-theta."""
    t = np.radians(np.asarray(az, float))
    A = np.column_stack([np.ones_like(t), np.cos(2 * t), np.sin(2 * t),
                         np.cos(4 * t), np.sin(4 * t)])
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
