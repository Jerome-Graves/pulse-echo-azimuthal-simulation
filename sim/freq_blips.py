"""Does 5 MHz make the boundary blips pickable? Model prediction + fw check.

At 2 MHz the full wave showed on-axis boundary echoes standing only +1 to
+5.5 dB above the off-axis speckle background - not pickable. Higher
frequency helps twice: the pulse is shorter (less background per range cell)
and the beam is narrower (fewer off-axis contributors). Full-wave 5 MHz on
the whole disk needs 243M cells (~31 GB) and does not fit the 4070, so:

  * PREDICT blip-above-background at 2 / 3 / 5 MHz with the validated
    models: ray model for the specular blip amplitudes, rung 3
    boundary_scatter for the speckle background envelope.
  * VALIDATE the trend with one full-wave trace at 3 MHz (46M cells, fits).
  * If the 2 -> 3 MHz trend is right, trust the 5 MHz prediction.

Run with --fw3 to generate the 3 MHz full-wave trace (GPU, ~18 min); without
it, prints the model prediction table and, if ref/fw_3mhz.npz exists, the
measured 3 MHz blip table for comparison.

MEASURED OUTCOME (2026-07-24, ref/fw_3mhz.npz):
  * Speckle null at 3 MHz (200 random non-boundary times, peak-above-local-
    median in +/-1 us): median +1.6 dB, 99th percentile +3.6 dB. Detection
    threshold is therefore ~+4 dB, and the 2 MHz "blips" (+1..+5.5) were
    almost all speckle.
  * At 3 MHz, boundaries 6 (42.1 us, +5.2 dB) and 7 (44.3 us, +6.7 dB)
    emerge ABOVE the null - exactly the two the model ranked strongest.
    Boundary 3 (model +4.1) stayed buried (+1.4): the model's RANKING and
    frequency TREND are right, its absolute visibility is optimistic by
    ~3-8 dB, consistent with deep-facet phase decoherence it doesn't model.
  * Extrapolation to 5 MHz (model gain +3..+4 dB over 3 MHz, minus the
    optimism): expect roughly half the boundaries per trace to be pickable
    at azimuths where their wall faces the probe, and the tilted-wall
    boundaries never.

FALSIFIED BY THE 5 MHz MEASUREMENT (2026-07-25, ref/fw_5mhz_fabric00.npz):
  * At 5 MHz ZERO of 7 boundaries clear the speckle null (+0.3..+3.7 vs
    99th pct +3.6) - the two boundaries that were clearly detected at
    3 MHz VANISH. The monotonic model ignores propagation decoherence:
    the phase variance a facet's specular flash accumulates through the
    grains in front of it scales as f^2, so resolution wins 2 -> 3 MHz and
    decoherence wins 3 -> 5 MHz. BLIP VISIBILITY HAS AN OPTIMUM NEAR
    ~3 MHz for this grain size and geometry.
  * The 5 MHz coda also sits ~17 dB lower re E1 than at 2 MHz (clean-
    window RMS -53.4 vs -36.2 dB): narrower beam (~-8), shorter pulse
    (~-4), f^2 scattering attenuation of the illumination (rest). At
    -53 dB re E1 the coda is sub-LSB on the 8-bit scope if E1 sets full
    scale: recoverable through the 0.5% FS receiver-noise dither with
    1024 averages, but gating the rim echo out of the record (so gain can
    be raised) goes from nice-to-have to essentially mandatory at 5 MHz.
  * Experiment-design consequence: the 5 MHz probe is NOT optimal for
    interval-timing blips or coda SNR; a ~2.25-3.5 MHz probe would be.
    5 MHz remains fine for E1 ToF precision and the E1(az) walk-off fade.
    All from ONE azimuth / one realisation - worth one confirming azimuth
    before hardware decisions.
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
from scipy.signal import hilbert                # noqa: E402

import born                                     # noqa: E402
import forward as F                             # noqa: E402
import ladder                                   # noqa: E402
from config import Config                       # noqa: E402
from specimen import DiskSpecimen               # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")
C_REF = 3850.0


def spec_build(h):
    sp = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                      size_cv=0.35, concentration=3.93, spatial_corr=0.0,
                      fabric_axis=(1, 0, 0), seed=7)
    return sp.build(h)


def blip_prediction(f0, build, fs=1.0 / ladder.DT):
    """Per-boundary (t_us, blip/background dB) from ray blips vs rung 3."""
    import acquisition as A
    cfg = Config()
    cfg.probe.f0 = f0
    cfg.acq.fs = fs
    cfg.acq.record_end_s = 60e-6
    cfg.acq.tilts_deg = (0.0,)
    cfg.err.enabled = False
    cfg.err.rim_reflection = -0.776
    tr = A.truth_along_diameter(build, cfg, 0.0)
    L, V = tr["length_m"], tr["speed"]
    t_bnd = 2.0 * np.cumsum(L / V)[:-1]

    _, x_ray = F.simulate_angle(build, cfg, 0.0)
    e_ray = np.abs(hilbert(np.asarray(x_ray, float)))
    _, bg = born.boundary_scatter(dict(build), cfg)
    bg = np.asarray(bg, float)

    out = []
    half = int(0.5e-6 * fs)
    for tb in t_bnd:
        k = int(tb * fs)
        if k - half < 0 or k + half >= min(len(e_ray), len(bg)):
            continue
        blip = e_ray[k - half:k + half].max()
        back = bg[k - half:k + half].mean()
        out.append((tb * 1e6, 20 * np.log10(blip / back + 1e-30)))
    return out


def _sym_ricker(f0, dt):
    """Zero-phase Ricker for matched filtering (peak at the centre tap)."""
    n = int(round(3.0 / f0 / dt)) | 1
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f0 * t) ** 2
    w = (1 - 2 * a) * np.exp(-a)
    return w / np.linalg.norm(w)


def fw_blip_table(npz_path, build, f0=None):
    """Measured blip-above-local-background from a full-wave trace.

    Two corrections landed 2026-07-31 (adversarial review of the
    'blips are dead' verdict), measured on rigid_seed11 pooled over
    360 azimuths / 3238 boundary tests:
      (1) SOURCE DELAY - fdtd.ricker peaks at t0 = 1.2/f0 (0.60 us at
          2 MHz, more than one period) while t_bnd starts at t = 0, so
          the peak search sat a full pulse off the echo.
      (2) MATCHED FILTER - correlating with the source wavelet before
          the envelope roughly doubles boundary-vs-speckle separation
          (+0.40 dB / 5.9 sigma envelope -> +0.67 dB / 7.5 sigma
          matched filter; +0.98 dB / 12.4 sigma with both fixes).
    The historical single-trace verdicts in the module docstring asked
    'does ONE boundary clear a per-trace threshold' - answered no, and
    correctly. Pooled over a full sweep the boundary excess is highly
    significant and scales with the ray-model velocity contrast, and
    5 MHz shows the LARGEST per-boundary excess of the four sweeps
    tested - so the 'optimum near 3 MHz' claim is a statement about a
    per-trace detection threshold, not about the observable."""
    import acquisition as A
    cfg = Config(); cfg.err.enabled = False
    tr = A.truth_along_diameter(build, cfg, 0.0)
    L, V = tr["length_m"], tr["speed"]
    t_bnd = 2.0 * np.cumsum(L / V)[:-1]
    d = np.load(npz_path)
    x, dt = d["trace"].ravel(), float(d["dt"])
    if f0 is None:
        f0 = float(d["f0"]) if "f0" in d.files else 2e6
    fs = 1.0 / dt
    e = np.abs(hilbert(np.convolve(x, _sym_ricker(f0, dt)[::-1], "same")))
    t_bnd = t_bnd + 1.2 / f0          # source-wavelet delay
    out = []
    half = int(1.0e-6 * fs)
    k4 = int(4e-6 * fs)
    for tb in t_bnd:
        k = int(tb * fs)
        if k - half < 0 or k + half >= len(e):
            continue
        pk = e[k - half:k + half].max()
        ring = np.r_[e[max(k - k4, 0):k - half], e[k + half:min(k + k4, len(e))]]
        bg = np.median(ring) if len(ring) else 1e-30
        out.append((tb * 1e6, 20 * np.log10(pk / bg + 1e-30)))
    return out


def run_fw3(f0=3.0e6):
    from scipy import ndimage
    import time
    import coda_convergence as CC
    import fdtd
    h = C_REF / f0 / 6.0
    build = spec_build(h)
    lab, nd, m = CC.build_grid(build, h, 10)
    nz, nx = lab.shape[0], lab.shape[1]
    co = fdtd.optimised_coeffs(8)
    mats = (lab + 1).astype(np.uint8)
    Ct, rho_t = fdtd.material_tables(build["axes"])
    dt = fdtd.safe_dt_labels(mats, Ct, rho_t, h, co, safety=0.5)
    D = nd * h
    nt = int(2.2 * D / C_REF / dt)
    wav = fdtd.ricker(f0, dt, nt)
    dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    dm = np.exp(-0.02 * dist).astype(np.float32)
    cz, cy, cx = nz // 2, nx // 2, nx // 2
    Rc = nd // 2
    ixp = next(cx + k - 1 for k in range(Rc - 1, 0, -1)
               if lab[cz, cy, cx + k] >= 0)
    er = max(int(6.35e-3 / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
           for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er and lab[cz + dz, cy + dy, ixp] >= 0]
    w = 1.0 / len(pts)
    print(f"3 MHz grid {lab.shape} = {lab.size/1e6:.0f}M cells, {nt} steps, "
          f"dt {dt*1e9:.2f} ns", flush=True)
    t0 = time.time()
    trc = np.asarray(fdtd.forward_fused_labels(
        lab, build["axes"], h, dt, nt, [(p, w) for p in pts], wav,
        [(pts, np.full(len(pts), w))], order=8, coeffs=co,
        sponge_width=10, damp_mask=dm), float).ravel()
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    np.savez(os.path.join(REF, "fw_3mhz.npz"), trace=trc, dt=dt, f0=f0,
             h=h, order=8, ppw=6, seed_specimen=7)


def main():
    if "--fw3" in sys.argv:
        run_fw3()
        return
    print("MODEL PREDICTION: specular blip / speckle background (dB)")
    print(f"{'boundary t us':>14} {'2 MHz':>7} {'3 MHz':>7} {'5 MHz':>7}")
    preds = {}
    for f0 in (2e6, 3e6, 5e6):
        b = spec_build(C_REF / f0 / 6.0) if f0 != 2e6 else ladder.standard_build()
        preds[f0] = blip_prediction(f0, b)
    n = min(len(v) for v in preds.values())
    for i in range(n):
        t = preds[2e6][i][0]
        print(f"{t:>13.1f} {preds[2e6][i][1]:>+7.1f} {preds[3e6][i][1]:>+7.1f} "
              f"{preds[5e6][i][1]:>+7.1f}")
    fw3 = os.path.join(REF, "fw_3mhz.npz")
    if os.path.exists(fw3):
        print("\nFULL-WAVE 3 MHz, measured blip above local background:")
        b3 = spec_build(C_REF / 3e6 / 6.0)
        for t, db in fw_blip_table(fw3, b3):
            print(f"{t:>13.1f} {db:>+7.1f}")
        print("\n(2 MHz measured, for the trend: +1.2 +2.3 +1.4 +4.7 +5.5 "
              "+1.8 +2.4)")


if __name__ == "__main__":
    main()
