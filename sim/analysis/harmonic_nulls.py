"""Correct significance test for an azimuthal k=2 line.

The permutation null is invalid: shuffling azimuths destroys the 5-7 deg
glint correlation length, so it produces a WHITE null whose k=2 line is
tiny, and every red-noise process looks significant.  Replacements:

  (1) LOCAL-BACKGROUND line test.  Estimate the smooth (red) azimuthal
      power spectrum from the harmonics AROUND k=2 (excluding the
      fabric's own k=2 and k=4), then P2/bg ~ Exp(1) under the null.
  (2) SPECTRUM-PRESERVING PHASE-RANDOMISED surrogates: keep every |Y_k|
      except k=2, which is redrawn from the background level, randomise
      all phases, remeasure A2.  Gives the null distribution directly.
  (3) The isotropic_seed41_ppw6_calibration isotropic sweep as a physical control.
  (4) PHASE reproducibility across independent specimens - a red-noise
      process has uniformly random k=2 phase, so agreement between
      independent sweeps is the strongest evidence available."""
import numpy as np

RNG = np.random.default_rng(7)


def corr_length(az, y):
    """Azimuthal autocorrelation half-width (deg) of the residual after
    removing k<=4 - i.e. the glint correlation length."""
    o = np.argsort(az)
    az, y = np.asarray(az, float)[o], np.asarray(y, float)[o]
    th = np.radians(az)
    X = np.column_stack([np.ones_like(th)]
                        + [f(k * th) for k in (1, 2, 3, 4)
                           for f in (np.cos, np.sin)])
    r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r -= r.mean()
    n = len(r)
    ac = np.correlate(r, np.r_[r, r], mode="valid")[:n // 2]
    ac /= ac[0]
    step = float(np.median(np.diff(az)))
    i = int(np.argmax(ac < 0.5)) if np.any(ac < 0.5) else 0
    return i * step, ac[:12]


def line_test(az, y, k0=2, kfit=range(1, 21), exclude=(2, 4),
              n_surr=4000, verbose=False):
    """Return dict with A_k0, background-equivalent amplitude, ratio,
    analytic p (Exp(1)) and the surrogate p."""
    o = np.argsort(az)
    y = np.asarray(y, float)[o]
    n = len(y)
    Y = np.fft.rfft(y - y.mean())
    P = np.abs(Y) ** 2
    amp = 2 * np.abs(Y) / n
    ks = [k for k in kfit if k not in exclude and k < len(P)]
    # smooth red background: linear fit of log P vs log k
    c = np.polyfit(np.log(ks), np.log(P[ks]), 1)
    bg = float(np.exp(np.polyval(c, np.log(k0))))
    ratio = float(P[k0] / bg)
    p_an = float(np.exp(-ratio))
    # explicit spectrum-preserving surrogates
    mags = np.abs(Y).copy()
    hits = 0
    a_null = np.empty(n_surr)
    for i in range(n_surr):
        m = mags.copy()
        # k0 line redrawn from the background (Rayleigh at level bg)
        m[k0] = np.sqrt(bg * RNG.exponential())
        ph = RNG.uniform(0, 2 * np.pi, len(m))
        ph[0] = 0.0
        Ys = m * np.exp(1j * ph)
        ys = np.fft.irfft(Ys, n)
        A = 2 * np.abs(np.fft.rfft(ys - ys.mean())[k0]) / n
        a_null[i] = A
        hits += A >= amp[k0]
    return dict(A=float(amp[k0]),
                A_bg=float(2 * np.sqrt(bg) / n),
                ratio=ratio, p_analytic=p_an,
                p_surr=(hits + 1) / (n_surr + 1),
                null95=float(np.percentile(a_null, 95)),
                phase=float(np.degrees(np.angle(Y[k0])) / k0 % (360.0 / k0)))


def fmt(d):
    return (f"A2 {d['A']:5.3f} dB @ {d['phase']:6.1f} | red-noise null95 "
            f"{d['null95']:5.3f} | P2/bg {d['ratio']:6.2f} | "
            f"p_surr {d['p_surr']:.4f}")
