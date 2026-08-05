"""TRACE ENHANCEMENT LAB - the full pre-inversion treatment chain
(Jerome: "make the data we need pronounced" / "let's do all of it").
Software only, existing traces only, no apparatus change.

Chain (each step feeds the next):
  1. COMMON-MODE SUBTRACTION - the azimuth-median trace (source
     ring-down, rim systematics) carries zero texture info; remove it.
  2. EMPIRICAL WAVELET - align the E1 echoes of all azimuths on their
     picked ToF and stack: the as-propagated system wavelet, no
     parameter guessing (works identically on the rig).
  3. WIENER DECONVOLUTION (pulse compression) - whiten with the
     estimated wavelet: every boundary echo compresses toward a spike,
     sharpening the migration PSF and separating close walls.
  4. REFINED ToF - after compression E1 is a band-limited spike: peak
     of the analytic signal near the stored pick, parabolic
     sub-sample. Feeds migration velocity + E1 gating.
  5. optional AZIMUTHAL SVD tail-denoise on the (az x t) gather -
     coherent moveout lives in the leading ranks, incoherent noise in
     the tail. Mild default (keep half); the fabric signal itself is
     azimuth-smooth, so never truncate aggressively.
  6. ARC BACKPROJECTION with PHASE-WEIGHTED STACKING - pixel value =
     linear coherent stack x (phase coherence)^NU: real reflectors are
     phase-consistent across views, speckle is not.
  7. SUB-BAND COMPOUNDING - migrate three zero-phase Butterworth
     sub-bands separately (each samples a different 2k shell) and
     average the normalised magnitudes: thicker k-space annulus,
     speckle averaged, phase kept within band.
  8. E1 FADE METRICS per azimuth - peak analytic amplitude (dB) and
     spectral CENTROID downshift (the robust scattering-loss
     estimator, gain-insensitive) -> e1_fade.npz, the new inversion
     input channel.

Outputs in the sweep dir:
  arc_enhanced.npz  (img_pws full-band, img_compound sub-band, axes)
  arc_enhanced.png  (raw-chain vs enhanced comparison)
  e1_fade.npz       (rots, amp_db, centroid_mhz, tof_refined_us)

CLI: python trace_lab.py <sweep_name> [pixel_mm]
"""
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]

from scipy.signal import hilbert, butter, sosfiltfilt   # noqa: E402

import fit_sweep as FS                                  # noqa: E402
import sweep_runner as SW                               # noqa: E402

C_REF = 3850.0
T_MIN = 8e-6
E1_GUARD = 3e-6
BEAM_W = 6e-3
PWS_NU = 2.0
SVD_KEEP = 180          # of 360 - mild tail denoise only
WIENER_EPS = 0.02
SUBBANDS = [(1.0e6, 2.0e6), (1.7e6, 2.7e6), (2.4e6, 3.4e6)]

try:
    import cupy as _cp
    _cp.zeros(4).sum()
    XP = _cp
    GPU = True
except Exception:
    XP = np
    GPU = False


def _asnp(a):
    return XP.asnumpy(a) if GPU else a


def load_gather(name):
    cfg = SW.load(name)
    d = SW.sweep_dir(name)
    done = SW.done_azimuths(cfg)
    # leading-edge ToF picks live in the FIT pipeline's observable
    # cache, NOT in the az files (az files: trace/dt/coda_db/E1)
    FS.AZ_SMOOTH_DEG = 0.0
    tof_by_rot = {int(r_["rot"]): float(r_["tof_us"]) * 1e-6
                  for r_ in FS.sweep_data(cfg)}
    trs, tofs = [], []
    dt = None
    for a in done:
        with np.load(os.path.join(d, f"az{a:03d}.npz")) as z:
            trs.append(np.asarray(z["trace"], float).ravel())
            dt = float(z["dt"])
        tofs.append(tof_by_rot.get(int(a), np.nan))
    nt = min(len(t) for t in trs)
    G = np.stack([t[:nt] for t in trs])
    tofs = np.asarray(tofs)
    if not np.isfinite(tofs).any():
        raise RuntimeError("no ToF picks available - run the fit once")
    return cfg, d, np.asarray(done, float), G, dt, tofs


def common_mode(G):
    return G - np.median(G, axis=0, keepdims=True)


def estimate_wavelet(G, dt, tofs, half_us=2.5):
    """Align E1 on the picked ToF and stack -> as-propagated wavelet."""
    h = int(half_us * 1e-6 / dt)
    t_ax = np.arange(G.shape[1]) * dt
    acc = np.zeros(2 * h)
    n = 0
    for i in range(G.shape[0]):
        if not np.isfinite(tofs[i]):
            continue
        # window centred a touch after the leading edge
        seg = np.interp(tofs[i] + (np.arange(2 * h) - h + h // 3) * dt,
                        t_ax, G[i])
        p = np.abs(hilbert(seg)).max()
        if p > 0:
            acc += seg / p
            n += 1
    w = acc / max(n, 1)
    w *= np.hanning(len(w))                    # taper
    return w


def wiener_deconv(G, w, dt):
    nt = G.shape[1]
    nf = int(2 ** np.ceil(np.log2(nt + len(w))))
    W = np.fft.rfft(w, nf)
    lam = WIENER_EPS * float(np.max(np.abs(W)) ** 2)
    inv = np.conj(W) / (np.abs(W) ** 2 + lam)
    D = np.fft.irfft(np.fft.rfft(G, nf, axis=1) * inv[None, :],
                     nf, axis=1)[:, :nt]
    # deconv makes E1 a spike at the wavelet-window ONSET; the group
    # delay removed equals the stack window's internal lead
    return D


def refine_tof(G, dt, tofs, half_us=1.5):
    """Peak of the analytic signal near the stored pick, parabolic
    sub-sample. On the deconvolved gather E1 is a compact spike."""
    t_ax = np.arange(G.shape[1]) * dt
    out = np.copy(tofs)
    for i in range(G.shape[0]):
        if not np.isfinite(tofs[i]):
            continue
        m = (t_ax > tofs[i] - half_us * 1e-6) \
            & (t_ax < tofs[i] + half_us * 1e-6)
        if m.sum() < 5:
            continue
        e = np.abs(hilbert(G[i]))[m]
        j = int(np.argmax(e))
        if 0 < j < len(e) - 1:
            dd = (e[j - 1] - e[j + 1]) / (2 * (e[j - 1] - 2 * e[j]
                                               + e[j + 1]) + 1e-30)
            out[i] = t_ax[m][j] + dd * dt
        else:
            out[i] = t_ax[m][j]
    return out


def svd_denoise(G, keep=SVD_KEEP):
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    s[keep:] = 0.0
    return (U * s) @ Vt


def bandpass(G, dt, f1, f2):
    sos = butter(4, [f1, f2], btype="band", fs=1.0 / dt, output="sos")
    return sosfiltfilt(sos, G, axis=1)


def backproject(G, dt, rots, tofs, D_m, n_px, pws=True):
    """Coherent DAS with per-azimuth velocity + phase-weighted stack."""
    R = D_m / 2
    ax = (np.arange(n_px) + 0.5) * (D_m / n_px) - R
    XXc, YYc = np.meshgrid(ax, ax)
    PX, PY = XP.asarray(XXc), XP.asarray(YYc)
    acc = XP.zeros(PX.shape, dtype=XP.complex128)
    phs = XP.zeros(PX.shape, dtype=XP.complex128)
    wsum = XP.zeros(PX.shape)
    for i, a in enumerate(rots):
        tr = G[i]
        ana = XP.asarray(hilbert(tr))
        th = np.radians(-a)
        px_, py_ = R * np.cos(th), R * np.sin(th)
        bx, by = -np.cos(th), -np.sin(th)
        dx, dy = PX - px_, PY - py_
        r = XP.sqrt(dx * dx + dy * dy)
        d_par = dx * bx + dy * by
        d_perp = XP.abs(dx * by - dy * bx)
        t_e1 = tofs[i] if np.isfinite(tofs[i]) else 51e-6
        c_az = 2.0 * D_m / t_e1
        idx = (2.0 * r / c_az) / dt
        i0 = XP.clip(idx.astype(XP.int64), 0, len(tr) - 2)
        fr = idx - i0
        s_c = ana[i0] * (1 - fr) + ana[i0 + 1] * fr
        t2 = 2.0 * r / c_az
        w = (XP.exp(-(d_perp / BEAM_W) ** 2) * (d_par > 0)
             * (t2 >= T_MIN) * (t2 <= t_e1 - E1_GUARD))
        w = w * XP.sqrt(XP.maximum(r, 1e-3) / R)
        acc = acc + w * s_c
        mag = XP.abs(s_c)
        phs = phs + w * s_c / XP.maximum(mag, 1e-30)
        wsum = wsum + w
    ok = wsum > 0.15 * float(XP.median(wsum[wsum > 1e-6]))
    lin = XP.where(ok, XP.abs(acc) / XP.maximum(wsum, 1e-6), XP.nan)
    if pws:
        coh = XP.where(ok, XP.abs(phs) / XP.maximum(wsum, 1e-6), 0.0)
        lin = lin * coh ** PWS_NU
    inside = ((XP.asarray(XXc) ** 2 + XP.asarray(YYc) ** 2)
              <= (R - 2.5e-3) ** 2)
    return _asnp(XP.where(inside, lin, XP.nan)), ax * 1e3


def e1_metrics(G_raw, dt, tofs):
    """Per-azimuth E1 amplitude (dB) + spectral centroid (MHz)."""
    t_ax = np.arange(G_raw.shape[1]) * dt
    amp = np.full(G_raw.shape[0], np.nan)
    cen = np.full(G_raw.shape[0], np.nan)
    for i in range(G_raw.shape[0]):
        if not np.isfinite(tofs[i]):
            continue
        m = (t_ax > tofs[i] - 1e-6) & (t_ax < tofs[i] + 4e-6)
        seg = G_raw[i][m] * np.hanning(m.sum())
        amp[i] = 20 * np.log10(np.abs(hilbert(G_raw[i]))[m].max()
                               + 1e-30)
        S = np.abs(np.fft.rfft(seg))
        f = np.fft.rfftfreq(m.sum(), dt)
        band = (f > 0.5e6) & (f < 4e6)
        if S[band].sum() > 0:
            cen[i] = float((f[band] * S[band]).sum()
                           / S[band].sum()) / 1e6
    return amp, cen


def main(name, pixel_mm=1.0):
    cfg, d, rots, G_raw, dt, tofs0 = load_gather(name)
    D_m = cfg["diameter_mm"] * 1e-3
    n_px = int(np.ceil(D_m / (pixel_mm * 1e-3)))
    print(f"[lab] {len(rots)} az, nt {G_raw.shape[1]}, "
          f"fs {1e-6 / dt:.1f} MHz ({'CUDA' if GPU else 'CPU'})",
          flush=True)

    # 1-3: common mode, wavelet, deconvolution
    G = common_mode(G_raw)
    w = estimate_wavelet(G, dt, tofs0)
    Gd = wiener_deconv(G, w, dt)
    print(f"[lab] wavelet {len(w) * dt * 1e6:.1f} us; deconv done",
          flush=True)
    # 4: refined ToF on the compressed gather
    tofs = refine_tof(Gd, dt, tofs0)
    dtof = (tofs - tofs0) * 1e6
    print(f"[lab] ToF refinement: mean {np.nanmean(dtof):+.3f} us, "
          f"rms dev {np.nanstd(dtof):.3f} us", flush=True)
    # 5: mild azimuthal SVD tail denoise
    Gs = svd_denoise(Gd)

    # 6: full-band PWS image
    img_pws, ax_mm = backproject(Gs, dt, rots, tofs, D_m, n_px)
    # 7: sub-band compound
    comp = None
    for (f1, f2) in SUBBANDS:
        Gb = bandpass(Gs, dt, f1, f2)
        im, _ = backproject(Gb, dt, rots, tofs, D_m, n_px)
        im = im / np.nanmax(im)
        comp = im if comp is None else comp + im
    img_comp = comp / len(SUBBANDS)
    # 8: E1 fade channel (on RAW traces - physical amplitudes)
    amp, cen = e1_metrics(G_raw, dt, tofs0)

    tmp = os.path.join(d, f"arc_enhanced.tmp{os.getpid()}.npz")
    np.savez(tmp, img_pws=img_pws, img_compound=img_comp,
             x_mm=ax_mm, y_mm=ax_mm, pixel_mm=pixel_mm)
    SW._atomic_replace(tmp, os.path.join(d, "arc_enhanced.npz"),
                       critical=True)
    tmp = os.path.join(d, f"e1_fade.tmp{os.getpid()}.npz")
    np.savez(tmp, rots=rots, amp_db=amp, centroid_mhz=cen,
             tof_refined_us=tofs * 1e6)
    SW._atomic_replace(tmp, os.path.join(d, "e1_fade.npz"),
                       critical=True)
    print(f"[lab] E1 fade: amp spread {np.nanstd(amp):.1f} dB, "
          f"centroid {np.nanmean(cen):.2f} +- {np.nanstd(cen):.3f} "
          "MHz", flush=True)

    # comparison render vs the plain-chain image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(img_pws, "deconv + common-mode + SVD +\n"
                        "PHASE-WEIGHTED stack (full band)"),
              (img_comp, "sub-band compound (3 bands,\n"
                         "PWS each, averaged)")]
    old = os.path.join(d, "arc_image.npz")
    if os.path.exists(old):
        with np.load(old) as z:
            panels.insert(0, (np.asarray(z["img_coh"]),
                              "before: plain coherent stack"))
    fig, axs = plt.subplots(1, len(panels),
                            figsize=(6.2 * len(panels), 6),
                            facecolor="#111")
    axs = np.atleast_1d(axs)
    for ax_, (im, ttl) in zip(axs, panels):
        db = 20 * np.log10(im / np.nanmax(im) + 1e-6)
        h = ax_.imshow(db, extent=[ax_mm[0], ax_mm[-1],
                                   ax_mm[0], ax_mm[-1]],
                       origin="lower", cmap="magma",
                       vmin=-25, vmax=0)
        ax_.set_title(ttl, color="w", fontsize=10)
        ax_.set_xlabel("x (mm)", color="w")
        ax_.set_ylabel("y (mm)", color="w")
        ax_.tick_params(colors="w")
        fig.colorbar(h, ax=ax_, label="dB re max")
    fig.tight_layout()
    p = os.path.join(d, "arc_enhanced.png")
    fig.savefig(p, dpi=130, facecolor="#111")
    print(f"[lab] rendered {p}", flush=True)


if __name__ == "__main__":
    nm = sys.argv[1]
    px = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    main(nm, px)
