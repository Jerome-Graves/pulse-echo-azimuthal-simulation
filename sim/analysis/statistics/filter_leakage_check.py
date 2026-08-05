"""Confirm the FD-band mechanism with an independent implementation
(time-domain Butterworth) and check what the leakage source is."""
import os

import numpy as np
import review_common as RC
import fit_sweep as FS
import sweep_runner as SW
from scipy.signal import hilbert, butter, sosfiltfilt

W0, W1 = 24e-6, 36e-6


def variants(name):
    cp = os.path.join(RC.SCRATCH, f"revfd2_{name}.npz")
    if os.path.exists(cp):
        with np.load(cp) as z:
            return {k: np.array(z[k]) for k in z.files}
    cfg = RC.cfg_of(name)
    d = SW.sweep_dir(name)
    keys = ["fd1", "fd3_rect", "fd3_butter", "fd3_rect_seg",
            "fd3_rect_noE1_noearly", "early_db"]
    R = {k: [] for k in keys}
    az = []
    for a_ in SW.done_azimuths(cfg):
        with np.load(os.path.join(d, f"az{a_:03d}.npz")) as z:
            tr = np.array(z["trace"], float).ravel()
            dt = float(z["dt"])
            f0v = float(z["f0"]) if "f0" in z.files else cfg["f0_mhz"] * 1e6
        fs = 1.0 / dt
        i0, i1 = int(W0 * fs), int(W1 * fs)
        e = np.abs(hilbert(tr))
        R["fd1"].append(20 * np.log10(np.sqrt((e[i0:i1] ** 2).mean())))
        R["early_db"].append(20 * np.log10(
            np.sqrt((e[int(8e-6 * fs):int(20e-6 * fs)] ** 2).mean())))
        fr = np.fft.rfftfreq(len(tr), dt)
        edges = np.linspace(0.5 * f0v, 1.5 * f0v, 4)
        F = np.fft.rfft(tr)

        def rect(sig):
            Fs = np.fft.rfft(sig)
            out = []
            for bi in range(3):
                m = ((fr >= edges[bi]) & (fr < edges[bi + 1])).astype(float)
                eb = np.abs(hilbert(np.fft.irfft(Fs * m, len(sig))))
                out.append((eb[i0:i1] ** 2).mean())
            return 20 * np.log10(np.sqrt(np.mean(out)))
        R["fd3_rect"].append(rect(tr))
        # segment-only (24-36 us isolated, zero elsewhere)
        seg = np.zeros_like(tr)
        seg[i0:i1] = tr[i0:i1]
        R["fd3_rect_seg"].append(rect(seg))
        # keep only 20-46 us (kills E1 AND the loud early coda)
        s2 = np.zeros_like(tr)
        s2[int(20e-6 * fs):int(46e-6 * fs)] = \
            tr[int(20e-6 * fs):int(46e-6 * fs)]
        R["fd3_rect_noE1_noearly"].append(rect(s2))
        out = []
        for bi in range(3):
            sos = butter(4, [edges[bi], edges[bi + 1]], btype="band",
                         fs=fs, output="sos")
            eb = np.abs(hilbert(sosfiltfilt(sos, tr)))
            out.append((eb[i0:i1] ** 2).mean())
        R["fd3_butter"].append(20 * np.log10(np.sqrt(np.mean(out))))
        az.append(float(a_))
    o = {k: np.asarray(v) for k, v in R.items()}
    o["az"] = np.asarray(az)
    np.savez(cp, **o)
    return o


for nm in ["singlemax_seed11_ppw6_rigid2", "isotropic_seed41_ppw6_calibration"]:
    Z = variants(nm)
    az = Z["az"]
    cfg = RC.cfg_of(nm)
    fr_ = RC.fitres(nm)
    alpha = (fr_["alpha_probe_deg"] if fr_ and fr_.get("alpha_probe_deg")
             else RC.truth_alpha(cfg))
    print(f"\n=== {nm} alpha_probe {alpha:.1f}")
    base = Z["fd1"]
    for k in ["fd1", "fd3_rect", "fd3_butter", "fd3_rect_seg",
              "fd3_rect_noE1_noearly", "early_db"]:
        v = Z[k]
        h = RC.harm_fit(az, v, orders=(2, 4, 6))
        print(f"  {k:24s} std {v.std():5.3f}  A2 {h[2][0]:5.3f}"
              f"@{h[2][1]/2%180:6.1f} (d_alpha "
              f"{(h[2][1]/2-alpha+90)%180-90:+6.1f})  corr(fd1) "
              f"{np.corrcoef(v, base)[0,1]:+.3f}")
