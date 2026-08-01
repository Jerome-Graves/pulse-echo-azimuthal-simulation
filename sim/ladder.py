"""The simplified-model LADDER: score every candidate against the full wave.

This file freezes the scoring protocol so every rung is judged identically.
The full-wave reference numbers were measured once with the validated FDTD
solver (order 8, ppw 6, safety 0.5, fluid damping, Ricker source) at 275 s a
trace; every simplified model is then scored in under a second against them.

PROTOCOL (do not change any of it without re-running the full wave):
  * specimen: 100 mm x 35 mm disk, 100 grains, size_cv 0.35, Watson
    concentration 3.93 about (1,0,0), seed 7, built at h = 3850/2e6/6
  * fabrics: progressively randomise 0 / 25 / 50 / 100 % of the c-axes,
    drawn SEQUENTIALLY from default_rng(11) - the stream is consumed across
    the loop, so the fabrics only reproduce if generated in this exact order
  * probe: 2 MHz, fs = 1/2.164e-8, record to 67.5 us, rim reflection -0.776
  * metric: 20*log10( max envelope in 22-42 us / 0.776 )
    (0.776 is the measured full-wave E1 amplitude, so levels are dB re E1)
  * FW reference levels: -28.1 / -32.9 / -31.6 / -43.4 dB

SCORES:
  * level  = model level at fabric 0, minus FW's -28.1 (0 is perfect)
  * sens   = RMS over the three perturbed fabrics of
             (model change from its own baseline) - (FW change from -28.1);
    this is the number that decides inversion suitability, because a global
    level offset calibrates away but a wrong RESPONSE cannot.
"""
import sys
import time

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")

import born                                    # noqa: E402
from config import Config                      # noqa: E402
from specimen import DiskSpecimen              # noqa: E402

DT = 2.164e-8
# CONTAMINATED-ERA constants (fluid_damp 0.01, unstable tail; see
# fw_reference.py) - kept ONLY for the historical comparison print in
# fw_reference.py. Decontaminated 2 MHz max-metric levels are
# -41.1/-32.2/-31.2/-43.2 (only fabric00 moved - it was floor-
# dominated). NEVER score against these: fw_reference_levels() now
# raises instead of falling back here (2026-07-29).
FW = np.array([-28.1, -32.9, -31.6, -43.4])    # full-wave dB re E1
FRACS = (0.0, 0.25, 0.5, 1.0)
E1 = 0.776
LO, HI = int(22e-6 / DT), int(42e-6 / DT)

# CLEAN WINDOW, found by the time-resolved shape diagnostic on the saved
# reference traces: below ~24 us the shallow coda is in the coherent-facet
# regime (Fresnel zone ~ beam width) where incoherent sums under-read by
# 10-14 dB, and above ~36 us a pre-E1 buildup (rim-guided precursor energy)
# inflates the full-wave envelope. Both contaminations sit inside the
# historical 22-42 us window and drove the apparent model-fw disagreement.
# In 24-36 us (depths 47-70 mm) with the RMS metric, rung 3 matches the
# full wave to -3.0 dB absolute and 3.1 dB sensitivity RMS.
W_CLEAN = (24e-6, 36e-6)


def standard_build():
    sp = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                      size_cv=0.35, concentration=3.93, spatial_corr=0.0,
                      fabric_axis=(1, 0, 0), seed=7)
    return sp.build(3850.0 / 2e6 / 6.0)


def standard_cfg():
    cfg = Config()
    cfg.probe.f0 = 2.0e6
    cfg.acq.fs = 1.0 / DT
    cfg.acq.record_end_s = 67.5e-6
    cfg.acq.tilts_deg = (0.0,)
    cfg.err.enabled = False
    cfg.err.rim_reflection = -0.776
    return cfg


def standard_fabrics(base_axes):
    """The four fabric sets, exactly as used for the full-wave reference."""
    rng = np.random.default_rng(11)
    out = []
    for frac in FRACS:
        ax = base_axes.copy()
        if frac > 0:
            kk = int(frac * len(ax))
            idx = rng.choice(len(ax), kk, replace=False)
            v = rng.normal(size=(kk, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            ax[idx] = v
        out.append(ax)
    return out


def level_db(env, metric="max", window=None):
    """dB re E1 in the coda window. metric='max' is the historical protocol;
    'rms' self-averages ~24 resolution cells, cutting the realisation noise
    of the reference by ~5x, so it is the better discriminator between
    models once reference traces exist to score against. window is a
    (start_s, end_s) tuple; default is the historical 22-42 us."""
    lo, hi = (LO, HI) if window is None else (int(window[0] / DT),
                                             int(window[1] / DT))
    w = env[lo:hi]
    v = w.max() if metric == "max" else np.sqrt((w ** 2).mean())
    return 20.0 * np.log10(v / E1 + 1e-30)


def fw_reference_levels(metric="max", window=None):
    """FW levels from the saved reference traces (sim/ref); falls back to
    the recorded max-metric constants if the traces are not on disk."""
    import os
    ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")
    w0, w1 = (22e-6, 42e-6) if window is None else window
    out = []
    for tag in ("00", "25", "50", "100"):
        fp = os.path.join(ref, f"fw_fabric{tag}.npz")
        if not os.path.exists(fp):
            # no silent fallback: the FW constants are contaminated-era
            # numbers 13 dB off the decontaminated fabric00 truth
            raise FileNotFoundError(
                f"{fp} missing: run fw_reference.py first (refusing to "
                "fall back to the contaminated-era FW constants)")
        d = np.load(fp)
        tr, dt = d["trace"].ravel(), float(d["dt"])
        e = np.abs(hilbert(tr))
        t1 = 2 * 0.100 / 3850.0
        k0 = int(t1 / dt)
        e1 = e[max(k0 - int(2e-6 / dt), 0):k0 + int(2e-6 / dt)].max()
        lo, hi = int(w0 / dt), min(int(w1 / dt), len(e))
        w = e[lo:hi]
        v = w.max() if metric == "max" else np.sqrt((w ** 2).mean())
        out.append(20 * np.log10(v / e1 + 1e-30))
    return np.array(out)


def score(model_fn, build, cfg, fabrics, name="", metric="max", fw=None,
          window=None):
    """Run one model over the four fabrics; return dict of scores."""
    fw = fw_reference_levels(metric, window) if fw is None else fw
    levels, secs = [], 0.0
    for ax in fabrics:
        bb = dict(build); bb["axes"] = ax
        t0 = time.time()
        _, env = model_fn(bb, cfg)
        secs += time.time() - t0
        env = np.asarray(env, float)
        if np.any(env < 0):                    # waveform, not envelope
            env = np.abs(hilbert(env))
        levels.append(level_db(env, metric, window))
    levels = np.array(levels)
    dm, dfw = levels - levels[0], fw - fw[0]
    return dict(name=name, levels=levels, level_off=levels[0] - fw[0],
                sens_rms=float(np.sqrt(((dm - dfw)[1:] ** 2).mean())),
                s_per_trace=secs / len(fabrics), fw=fw)


def report(rows, fw=None):
    fw = fw_reference_levels() if fw is None else fw
    print(f"\n{'model':<26}{'lvl@0%':>8}{'offset':>8}{'sensRMS':>9}"
          f"{'d25':>7}{'d50':>7}{'d100':>7}{'s/tr':>7}")
    dfw = fw - fw[0]
    print(f"{'FULL WAVE (reference)':<26}{fw[0]:>7.1f}d{0.0:>+7.1f}d"
          f"{'-':>9}{dfw[1]:>+6.1f}d{dfw[2]:>+6.1f}d{dfw[3]:>+6.1f}d"
          f"{'275':>7}")
    for r in rows:
        dm = r["levels"] - r["levels"][0]
        print(f"{r['name']:<26}{r['levels'][0]:>7.1f}d{r['level_off']:>+7.1f}d"
              f"{r['sens_rms']:>8.1f}d{dm[1]:>+6.1f}d{dm[2]:>+6.1f}d"
              f"{dm[3]:>+6.1f}d{r['s_per_trace']:>7.2f}")


def main():
    build = standard_build()
    cfg = standard_cfg()
    fabrics = standard_fabrics(build["axes"])
    print(f"specimen {build['labels'].shape}, {len(build['axes'])} grains; "
          f"window {LO*DT*1e6:.0f}-{HI*DT*1e6:.0f} us, E1 {E1}")

    rows = []
    # harness validation: rungs 3 and 7 must reproduce their recorded scores
    rows.append(score(born.boundary_scatter, build, cfg, fabrics,
                      "rung3 incoherent"))
    print(f"   {rows[-1]['name']}: lvl {rows[-1]['levels'][0]:.1f} "
          f"(recorded -35.7)", flush=True)
    rows.append(score(born.kirchhoff_fresnel, build, cfg, fabrics,
                      "rung7 fresnel-cube"))
    print(f"   {rows[-1]['name']}: lvl {rows[-1]['levels'][0]:.1f} "
          f"(recorded -60.7)", flush=True)

    # rung 8: wavefront-aligned patches
    for mult in (0.5, 1.0, 2.0):
        rows.append(score(
            lambda b, c, m=mult: born.kirchhoff_shell(b, c, fres_mult=m),
            build, cfg, fabrics, f"rung8 shell x{mult}"))
        print(f"   {rows[-1]['name']}: lvl {rows[-1]['levels'][0]:.1f}",
              flush=True)
    # rung 8 partial coherence: medium-derived coherent/incoherent blend
    for mult in (0.5, 1.0, 2.0):
        rows.append(score(
            lambda b, c, m=mult: born.kirchhoff_shell(b, c, fres_mult=m,
                                                      partial=True),
            build, cfg, fabrics, f"rung8 partial x{mult}"))
        print(f"   {rows[-1]['name']}: lvl {rows[-1]['levels'][0]:.1f}",
              flush=True)
    # rung 9: add the P->S->P mode-conversion channel
    for mult in (1.0, 2.0):
        rows.append(score(
            lambda b, c, m=mult: born.kirchhoff_shell(b, c, fres_mult=m,
                                                      partial=True,
                                                      shear=True),
            build, cfg, fabrics, f"rung9 +shear x{mult}"))
        print(f"   {rows[-1]['name']}: lvl {rows[-1]['levels'][0]:.1f}",
              flush=True)
    # rung 10: plate-guided (top/bottom face image) paths
    rows.append(score(
        lambda b, c: born.kirchhoff_shell(b, c, fres_mult=2.0, partial=True,
                                          plate=True),
        build, cfg, fabrics, "rung10 +plate"))
    print(f"   {rows[-1]['name']}: lvl {rows[-1]['levels'][0]:.1f}", flush=True)
    rows.append(score(
        lambda b, c: born.kirchhoff_shell(b, c, fres_mult=2.0, partial=True,
                                          shear=True, plate=True),
        build, cfg, fabrics, "rung10 +shear+plate"))
    print(f"   {rows[-1]['name']}: lvl {rows[-1]['levels'][0]:.1f}", flush=True)
    report(rows)

    # RMS-metric rescoring against the saved reference traces, if present.
    # The max metric carries several dB of realisation noise in the
    # reference itself; RMS averages ~24 resolution cells and is the metric
    # that can actually rank models this close together.
    try:
        fw_rms = fw_reference_levels("rms")
    except FileNotFoundError:
        print("\n(no sim/ref traces yet: run fw_reference.py for RMS scoring)")
        return
    print("\n=== RMS metric (fw reference from saved traces) ===")
    fw_max = fw_reference_levels("max")
    print(f"fw max-metric from saved traces: "
          f"{np.array2string(fw_max, precision=1)} "
          f"(recorded {np.array2string(FW, precision=1)})")
    rms_rows = []
    for name, fn in (
            ("rung3 incoherent", born.boundary_scatter),
            ("rung8 partial x2.0",
             lambda b, c: born.kirchhoff_shell(b, c, fres_mult=2.0,
                                               partial=True)),
            ("rung9 +shear x2.0",
             lambda b, c: born.kirchhoff_shell(b, c, fres_mult=2.0,
                                               partial=True, shear=True)),
            ("rung10 +shear+plate",
             lambda b, c: born.kirchhoff_shell(b, c, fres_mult=2.0,
                                               partial=True, shear=True,
                                               plate=True))):
        rms_rows.append(score(fn, build, cfg, fabrics, name,
                              metric="rms", fw=fw_rms))
        print(f"   {name}: lvl {rms_rows[-1]['levels'][0]:.1f}", flush=True)
    report(rms_rows, fw=fw_rms)


if __name__ == "__main__":
    main()
