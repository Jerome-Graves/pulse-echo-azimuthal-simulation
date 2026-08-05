"""COHERENT ARC BACKPROJECTION (Norton-Linzer reflectivity imaging)
of an existing monostatic sweep - no new data, no apparatus change.

The texture-inversion literature review's one legitimately-supported
spatially-resolved product for this geometry: each RF sample at time t
backprojects onto the arc r = c*t/2 about that azimuth's probe
position; summing ANALYTIC signals coherently over all 360 azimuths
lets boundary echoes add in phase where a real reflector sits and
cancel elsewhere. This uses the PHASE the envelope pipeline discards -
the information Jerome pointed at in the coda-to-E1 window - and the
full usable record (early coherent-facet region included: coherent
processing is exactly the treatment it needs).

Output: <sweep>/arc_image.npz with
    img_coh   |sum of analytic signals|      - the boundary image
    img_inc   sum of envelopes (incoherent)  - control/compound
    x_mm, y_mm pixel centres
plus a PNG rendering.

Gates: t >= T_MIN (source ring-down); per-azimuth t <= tof_us - E1_GUARD
(keeps E1 and its skirt out); per-pixel beam-corridor weight
exp(-(d_perp/BEAM_W)^2) because the physical source is directional.

CLI: python arc_backproject.py <sweep_name> [pixel_mm]
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

from scipy.signal import hilbert               # noqa: E402

import sweep_runner as SW                      # noqa: E402

C_REF = 3850.0
T_MIN = 8e-6          # source ring-down guard
E1_GUARD = 3e-6       # keep E1's leading skirt out
BEAM_W = 6e-3         # beam corridor 1/e half-width (m), ~Fresnel

try:
    import cupy as _cp
    _cp.zeros(4).sum()
    XP = _cp
    GPU = True
except Exception:
    XP = np
    GPU = False


def image(name, pixel_mm=1.0):
    cfg = SW.load(name)
    d = SW.sweep_dir(name)
    D_m = cfg["diameter_mm"] * 1e-3
    R = D_m / 2
    n_px = int(np.ceil(D_m / (pixel_mm * 1e-3)))
    ax = (np.arange(n_px) + 0.5) * (D_m / n_px) - R
    XXc, YYc = np.meshgrid(ax, ax)             # (y, x) grids
    PX = XP.asarray(XXc)
    PY = XP.asarray(YYc)
    acc_c = XP.zeros(PX.shape, dtype=XP.complex128)
    acc_i = XP.zeros(PX.shape)
    wsum = XP.zeros(PX.shape)

    done = SW.done_azimuths(cfg)
    print(f"[arc] {len(done)} azimuths, {n_px}x{n_px} px "
          f"({'CUDA' if GPU else 'CPU'})", flush=True)
    for a in done:
        with np.load(os.path.join(d, f"az{a:03d}.npz")) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
            t_e1 = (float(z["tof_us"]) * 1e-6
                    if "tof_us" in z.files else 51e-6)
        ana = XP.asarray(hilbert(tr))
        env = XP.abs(ana)
        # SPECIMEN frame at +a (fixed 2026-07-31). rotated_grid reads
        # specimen point R(+rot)x at solver point x with the probe fixed
        # on the solver +x rim, so the probe sits at specimen azimuth
        # +rot - the same convention as acquisition.truth_along_diameter.
        # The old -a mirrored every arc image about the specimen x-axis.
        # Settled empirically, not by derivation: the pooled boundary-
        # echo excess on rigid_seed11 reads 12.7 sigma with +a and only
        # 1.3 sigma with -a (same trace set, same detector).
        th = np.radians(a)                     # specimen frame
        px_, py_ = R * np.cos(th), R * np.sin(th)
        bx, by = -np.cos(th), -np.sin(th)      # beam unit vector
        dx = PX - px_
        dy = PY - py_
        r = XP.sqrt(dx * dx + dy * dy)
        d_par = dx * bx + dy * by              # along-beam distance
        d_perp = XP.abs(dx * by - dy * bx)     # cross-beam distance
        # per-azimuth effective velocity from THIS path's measured
        # E1 ToF: anisotropy shifts arrivals by up to ~1.5 lambda at
        # 2 MHz - a single global velocity de-coheres real
        # reflectors (first run: interior summed to cancellation)
        c_az = 2.0 * D_m / t_e1 if t_e1 > 0 else C_REF
        t2 = 2.0 * r / c_az
        idx = t2 / dt
        i0 = XP.clip(idx.astype(XP.int64), 0, len(tr) - 2)
        fr = idx - i0
        s_c = ana[i0] * (1 - fr) + ana[i0 + 1] * fr
        s_i = env[i0] * (1 - fr) + env[i0 + 1] * fr
        w = (XP.exp(-(d_perp / BEAM_W) ** 2)
             * (d_par > 0)
             * (t2 >= T_MIN) * (t2 <= t_e1 - E1_GUARD))
        # r-compensation: undo cylindrical spreading so deep
        # boundaries are not under-counted
        w = w * XP.sqrt(XP.maximum(r, 1e-3) / R)
        acc_c = acc_c + w * s_c
        acc_i = acc_i + w * s_i
        wsum = wsum + w
    # require real multi-azimuth coverage: low-wsum rim pixels are
    # normalisation-amplified noise (the bright rim ring, run 1)
    ok = wsum > 0.15 * float(XP.median(wsum[wsum > 1e-6]))
    img_c = XP.where(ok, XP.abs(acc_c) / XP.maximum(wsum, 1e-6), 0.0)
    img_i = XP.where(ok, acc_i / XP.maximum(wsum, 1e-6), 0.0)
    inside = ((XP.asarray(XXc) ** 2 + XP.asarray(YYc) ** 2)
              <= (R - 2.5e-3) ** 2)
    img_c = XP.where(inside, img_c, XP.nan)
    img_i = XP.where(inside, img_i, XP.nan)
    if GPU:
        img_c, img_i = XP.asnumpy(img_c), XP.asnumpy(img_i)
    out = os.path.join(d, "arc_image.npz")
    tmp = out[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, img_coh=img_c, img_inc=img_i,
             x_mm=ax * 1e3, y_mm=ax * 1e3, pixel_mm=pixel_mm)
    SW._atomic_replace(tmp, out, critical=True)
    print(f"[arc] saved {out}", flush=True)
    return img_c, img_i, ax * 1e3


def render(name):
    d = SW.sweep_dir(name)
    with np.load(os.path.join(d, "arc_image.npz")) as z:
        img_c = z["img_coh"]
        img_i = z["img_inc"]
        x = z["x_mm"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), facecolor="#111")
    for ax_, im, ttl in ((axs[0], img_c,
                          "COHERENT arc backprojection (phase used)\n"
                          "bright = in-phase reflectors = boundaries"),
                         (axs[1], img_i,
                          "incoherent compound (envelope only) - "
                          "control")):
        db = 20 * np.log10(im / np.nanmax(im) + 1e-6)
        h = ax_.imshow(db, extent=[x[0], x[-1], x[0], x[-1]],
                       origin="lower", cmap="magma", vmin=-30,
                       vmax=0)
        ax_.set_title(ttl, color="w", fontsize=10)
        ax_.set_xlabel("x (mm)", color="w")
        ax_.set_ylabel("y (mm)", color="w")
        ax_.tick_params(colors="w")
        fig.colorbar(h, ax=ax_, label="dB re max")
    fig.tight_layout()
    p = os.path.join(d, "arc_image.png")
    fig.savefig(p, dpi=130, facecolor="#111")
    print(f"[arc] rendered {p}", flush=True)
    return p


if __name__ == "__main__":
    nm = sys.argv[1]
    px = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    image(nm, px)
    render(nm)
