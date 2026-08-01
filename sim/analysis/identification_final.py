"""TEST 4 final: full-azimuth range test, mirror control, and a
PSF-CONSISTENT 2-D image test.

The step-1 image test compared the back-projected image against a blurred
boundary MAP.  That is unfair to the image: the imaging operator has a
1 mm x 20 mm PSF, so a boundary map blurred isotropically is not what the
image should look like even if the physics were perfect.  Here the SAME
back-projection operator is applied to the PREDICTED echo matrix of each
candidate tessellation, and the two images are compared.  Any candidate
that is right must now win on equal terms.

Controls
  wrong tessellation   43 alternatives (p floor 1/44 = 0.023)
  azimuth mirror       predict with the probe at -a instead of +a; the
                       frame is then wrong and the score must collapse
  azimuth shift        roll the measurement against the prediction
                       (p floor 1/n_az, 1/360 on the 360-az sweeps)
"""
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.special import j1

import t4_common as T
import t4_range as RG

SCR = T.SCR
PIX_MM = 0.5
R_MAX = 42.0
GATE = (24e-6, 36e-6)
GATE_W = (12e-6, 48e-6)
SWEEPS = [("girdle_perp_ppw8", 11), ("singlemax_ppw8", 11),
          ("girdle_par", 11), ("kappa8_seed17", 17),
          ("oos_seed23", 23), ("iso_gcal", 41), ("rigid_seed11", 11)]
CANDS = [11] + RG.NULL_SEEDS


def backproject(az, M, tgrid, t1, gate, pix_mm=PIX_MM):
    """Incoherent compound of an (n_az, n_t) non-negative echo matrix."""
    R = T.DIA / 2
    n = int(round(2 * R / (pix_mm * 1e-3)))
    axm = (np.arange(n) + 0.5) * (2 * R / n) - R
    X, Y = np.meshgrid(axm, axm, indexing="ij")
    acc = np.zeros((n, n))
    wsum = np.zeros((n, n))
    dt = tgrid[1] - tgrid[0]
    for q, a in enumerate(az):
        th = np.radians(float(a))
        px, py = R * np.cos(th), R * np.sin(th)
        bx, by = -np.cos(th), -np.sin(th)
        dx, dy = X - px, Y - py
        r = np.hypot(dx, dy)
        dpar = dx * bx + dy * by
        dperp = np.abs(dx * by - dy * bx)
        t2 = 2.0 * r / (2.0 * T.DIA / t1[q])
        idx = (t2 - tgrid[0]) / dt
        i0 = np.clip(idx.astype(np.int64), 0, len(tgrid) - 2)
        fr = np.clip(idx - i0, 0, 1)
        v = M[q][i0] * (1 - fr) + M[q][i0 + 1] * fr
        w = (np.exp(-(dperp / T.beam_radius(dpar)) ** 2) * (dpar > 0)
             * (t2 >= gate[0]) * (t2 <= gate[1]))
        acc += w * v
        wsum += w
    ok = wsum > 0.15 * np.median(wsum[wsum > 1e-6])
    inside = (X ** 2 + Y ** 2) <= (R - 2.5e-3) ** 2
    m = ok & inside & ((X ** 2 + Y ** 2) <= (R_MAX * 1e-3) ** 2)
    return np.where(m, acc / np.maximum(wsum, 1e-12), 0.0), m, axm * 1e3


def main():
    tgrid = np.arange(GATE_W[0], GATE_W[1], RG.TBIN)
    vols = {s: RG.label_volume(s) for s in CANDS}

    print("=== A. RANGE-DOMAIN TEST, full azimuth sets, specular predictor,"
          " wide gate 12-48 us ===")
    print(f"{'sweep':<18}{'own':>4}{'n_az':>6}{'r_own':>8}{'rank/N':>9}"
          f"{'p_wrong':>9}{'p_shift':>9}{'z_vs_null':>11}"
          f"{'r_mirror':>10}")
    store = {}
    for nm, own in SWEEPS:
        az, ana, dt, t1 = T.load_sweep(nm)
        env = np.abs(ana)
        m = env.mean(0)
        kk = max(int(3e-6 / dt) | 1, 3)
        hn = np.hanning(kk)
        hn /= hn.sum()
        ms = np.maximum(np.convolve(m, hn, mode="same"), 1e-3 * m.max())
        E = np.array([np.interp(tgrid, np.arange(len(r)) * dt, r / ms)
                      for r in env])
        c_az = 2.0 * T.DIA / t1
        geo, L, wray = RG.ray_geometry(az)
        geo_m, _, _ = RG.ray_geometry(-az)          # mirrored frame
        P = {s: RG.predict(vols[s][0], vols[s][1], geo, L, wray, c_az,
                           tgrid, True) for s in CANDS}
        Pm = RG.predict(vols[own][0], vols[own][1], geo_m, L, wray, c_az,
                        tgrid, True)
        r = {s: RG.corr(E, P[s]) for s in CANDS}
        vals = np.array([r[s] for s in CANDS])
        oth = np.array([r[s] for s in CANDS if s != own])
        rank = int(np.sum(vals >= r[own]))
        pw = (1 + np.sum(oth >= r[own])) / len(vals)
        rs = np.array([RG.corr(np.roll(E, k, axis=0), P[own])
                       for k in range(1, len(az))])
        ps = (1 + np.sum(rs >= r[own])) / (len(rs) + 1)
        z = (r[own] - oth.mean()) / oth.std()
        rm = RG.corr(E, Pm)
        store[nm] = dict(az=az, E=E, P=P, t1=t1, r=r, own=own)
        print(f"{nm:<18}{own:>4}{len(az):>6}{r[own]:>8.3f}"
              f"{rank:>5}/{len(vals):<3}{pw:>9.4f}{ps:>9.4f}{z:>11.1f}"
              f"{rm:>10.3f}", flush=True)

    print("\n=== B. PSF-CONSISTENT 2-D IMAGE TEST (coda gate 24-36 us) ===")
    print("measured image vs the image the SAME operator makes from each"
          " candidate's predicted echoes")
    print(f"{'sweep':<18}{'own':>4}{'r_own':>8}{'rank/N':>9}{'p_wrong':>9}"
          f"{'z_vs_null':>11}{'d_peak_mm':>11}{'p_dpk':>8}")
    for nm, own in SWEEPS:
        S = store[nm]
        az, t1 = S["az"], S["t1"]
        Emeas, mm, xmm = backproject(az, S["E"], tgrid, t1, GATE)
        imgs = {s: backproject(az, S["P"][s], tgrid, t1, GATE)[0]
                for s in CANDS}
        a = Emeas[mm]
        r = {s: float(np.corrcoef(a, imgs[s][mm])[0, 1]) for s in CANDS}
        vals = np.array([r[s] for s in CANDS])
        oth = np.array([r[s] for s in CANDS if s != own])
        rank = int(np.sum(vals >= r[own]))
        pw = (1 + np.sum(oth >= r[own])) / len(vals)
        z = (r[own] - oth.mean()) / oth.std()
        # peak-to-nearest-predicted-bright-region distance
        v = np.where(mm, Emeas, -np.inf)
        mx = ndimage.maximum_filter(v, size=13)
        cand = np.argwhere((v == mx) & mm)
        val = v[cand[:, 0], cand[:, 1]]
        pk = cand[np.argsort(-val)[:30]]

        def dmed(im):
            hi = im >= np.percentile(im[mm], 90)
            dt_ = ndimage.distance_transform_edt(~hi) * PIX_MM
            return float(np.median(dt_[pk[:, 0], pk[:, 1]]))
        d_own = dmed(imgs[own])
        d_oth = np.array([dmed(imgs[s]) for s in CANDS if s != own])
        p_d = (1 + np.sum(d_oth <= d_own)) / (len(d_oth) + 1)
        print(f"{nm:<18}{own:>4}{r[own]:>8.3f}{rank:>5}/{len(vals):<3}"
              f"{pw:>9.4f}{z:>11.1f}{d_own:>11.2f}{p_d:>8.3f}", flush=True)


if __name__ == "__main__":
    main()
