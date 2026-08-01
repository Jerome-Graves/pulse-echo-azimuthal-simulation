"""Why does FD_BANDS=3 annihilate the fabric 2-theta?
Hypothesis: the brick-wall sub-band mask is applied to the WHOLE record,
so the 45 dB-louder E1 arrival at ~52 us leaks (sinc kernel) into the
24-36 us coda window and swamps it."""
import os

import numpy as np
import rev_common as RC
import fit_sweep as FS
import sweep_runner as SW
from scipy.signal import hilbert

W0, W1 = 24e-6, 36e-6


def variants(name):
    cp = os.path.join(RC.SCRATCH, f"revfd_{name}.npz")
    if os.path.exists(cp):
        with np.load(cp) as z:
            return {k: np.array(z[k]) for k in z.files}
    cfg = RC.cfg_of(name)
    d = SW.sweep_dir(name)
    nb = 3
    keys = ["fd1", "fd3_full", "fd3_trunc", "fd3_taper", "fd3_trunc_taper",
            "fd8_trunc_taper", "e1_db"]
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
        k0 = int(2 * 0.1 / 3850.0 * fs)
        R["e1_db"].append(20 * np.log10(
            e[max(k0 - int(2e-6 * fs), 0):k0 + int(2e-6 * fs)].max()))
        # pre-E1 truncated copy: zero after 46 us with a 2 us cosine ramp
        tro = tr.copy()
        j0, j1 = int(44e-6 * fs), int(46e-6 * fs)
        ramp = 0.5 * (1 + np.cos(np.linspace(0, np.pi, j1 - j0)))
        tro[j0:j1] *= ramp
        tro[j1:] = 0.0
        fr = np.fft.rfftfreq(len(tr), dt)
        edges = np.linspace(0.5 * f0v, 1.5 * f0v, nb + 1)

        def band_rms(sig, edg, taper):
            F = np.fft.rfft(sig)
            out = []
            for bi in range(len(edg) - 1):
                if taper:
                    lo_, hi_ = edg[bi], edg[bi + 1]
                    w = np.clip((fr - lo_) / (hi_ - lo_), 0, 1)
                    m = np.where((fr >= lo_) & (fr < hi_),
                                 0.5 * (1 - np.cos(2 * np.pi * w)), 0.0)
                else:
                    m = ((fr >= edg[bi]) & (fr < edg[bi + 1])).astype(float)
                eb = np.abs(hilbert(np.fft.irfft(F * m, len(sig))))
                out.append((eb[i0:i1] ** 2).mean())
            return 20 * np.log10(np.sqrt(np.mean(out)))
        R["fd3_full"].append(band_rms(tr, edges, False))
        R["fd3_trunc"].append(band_rms(tro, edges, False))
        R["fd3_taper"].append(band_rms(tr, edges, True))
        R["fd3_trunc_taper"].append(band_rms(tro, edges, True))
        R["fd8_trunc_taper"].append(
            band_rms(tro, np.linspace(0.5 * f0v, 1.5 * f0v, 9), True))
        az.append(float(a_))
    out = {k: np.asarray(v) for k, v in R.items()}
    out["az"] = np.asarray(az)
    np.savez(cp, **out)
    return out


for nm in ["rigid_seed11", "oos_seed23", "iso_gcal"]:
    Z = variants(nm)
    az = Z["az"]
    cfg = RC.cfg_of(nm)
    fr = RC.fitres(nm)
    alpha = (fr["alpha_probe_deg"] if fr and fr.get("alpha_probe_deg")
             else RC.truth_alpha(cfg))
    print(f"\n=== {nm} (alpha_probe {alpha:.1f}, truth "
          f"{RC.truth_alpha(cfg):.1f}, kappa {cfg['concentration']})")
    psi = FS._psi_from_axis(az, alpha)
    keep = (psi >= 15.0) & ~((psi > 65.0) & (psi < 85.0))
    for k in ["fd1", "fd3_full", "fd3_trunc", "fd3_taper",
              "fd3_trunc_taper", "fd8_trunc_taper", "e1_db"]:
        v = Z[k]
        ha = RC.harm_fit(az, v, orders=(2, 4, 6))
        hk = RC.harm_fit(az[keep], v[keep], orders=(2, 4))
        print(f"  {k:16s} std {v.std():5.3f} | ALL-az A2 {ha[2][0]:5.3f}"
              f"@{ha[2][1]/2%180:6.1f} (d_alpha "
              f"{(ha[2][1]/2-alpha+90)%180-90:+6.1f}) A4 {ha[4][0]:5.3f}"
              f" | keep A2 {hk[2][0]:5.3f}@{hk[2][1]/2%180:6.1f}")
