"""TEST 3, part 2: speckle STATISTICS at 2 vs 5 MHz.

The facet model says the specular lobe narrows as lam/D_g, so at 5 MHz
fewer facets contribute per resolution cell and the amplitude statistics
must get HEAVIER-TAILED (super-Rayleigh).  This is a shape prediction,
independent of level, and therefore immune to the attenuation confound.

Two measurements:

 (a) per-azimuth coda RMS across azimuth (what varies as the specimen is
     rotated).  n = 40 at each frequency, azimuthally autocorrelated, so
     the effective sample size is quoted.

 (b) the envelope SAMPLES inside the gate, pooled over azimuths after
     dividing out the across-azimuth mean depth profile, and decimated to
     one sample per envelope correlation time.  For fully developed
     speckle the amplitude is Rayleigh and the intensity moment ratio
     E[I^2]/E[I]^2 = 2 exactly.  Super-Rayleigh (>2) means few effective
     scatterers; K-distribution shape alpha from 2(1+1/alpha).
"""
import numpy as np
from scipy import stats
from scipy.signal import hilbert, sosfiltfilt

import t3_common as T

AZ5 = T.az_list("prod5mhz")
GATE = T.CODA_W


def gate_env(name, azs):
    c = T.cfg_of(name)
    f0 = c["f0_mhz"] * 1e6
    E, dts = [], []
    for a in azs:
        tr, dt = T.load_trace(name, a)
        fs = 1 / dt
        sos, _, _ = T.sos_of(fs, f0)
        e = np.abs(hilbert(sosfiltfilt(sos, tr)))
        E.append(e[int(GATE[0] * fs):int(GATE[1] * fs)])
        dts.append(dt)
    n = min(len(x) for x in E)
    return np.array([x[:n] for x in E]), dts[0], f0


def acl(y, step_deg):
    """1/e autocorrelation length of a zero-mean series, in degrees."""
    y = y - y.mean()
    c = np.correlate(y, y, "full")[len(y) - 1:]
    c /= c[0]
    k = np.argmax(c < np.exp(-1.0))
    if k == 0:
        return np.nan
    return (k - 1 + (c[k - 1] - np.exp(-1)) / (c[k - 1] - c[k])) * step_deg


def moments(x, label):
    x = np.asarray(x, float)
    I = x ** 2
    r = (I ** 2).mean() / I.mean() ** 2
    alpha = 2.0 / (r - 2.0) if r > 2 else np.inf
    return dict(label=label, n=len(x), mean=x.mean(), cv=x.std() / x.mean(),
                skew=stats.skew(x), kurt=stats.kurtosis(x), Iratio=r,
                alpha=alpha)


print("=== (a) per-azimuth coda amplitude, across azimuth ===")
rows = []
for nm, tag in (("fittest", "2 MHz"), ("prod5mhz", "5 MHz")):
    m = T.measure(nm, azs=AZ5)
    lin = m["coda_lin"]
    L = acl(m["coda_db"], 4.0)
    d = moments(lin, tag)
    d["acl"] = L
    d["sd_db"] = m["coda_db"].std(ddof=1)
    rows.append(d)
    print(f"  {tag}: n={d['n']}  sd {d['sd_db']:.2f} dB  CV {d['cv']:.3f}  "
          f"skew {d['skew']:+.2f}  exc-kurt {d['kurt']:+.2f}  "
          f"E[I2]/E[I]2 {d['Iratio']:.3f}  alpha {d['alpha']:.2f}")
    print(f"        azimuthal 1/e correlation length {L:.1f} deg "
          f"-> ~{176/max(L,1e-9):.0f} independent azimuths in the 176 deg "
          f"covered")
print("  Rayleigh reference: CV 0.523, skew +0.63, exc-kurt +0.25, "
      "E[I2]/E[I]2 = 2.000")

print("\n=== (b) envelope samples inside the gate (the sharp test) ===")
store = {}
for nm, tag in (("fittest", "2 MHz"), ("prod5mhz", "5 MHz")):
    E, dt, f0 = gate_env(nm, AZ5)
    prof = np.sqrt((E ** 2).mean(0))          # across-azimuth depth profile
    Z = E / prof                              # speckle, trend removed
    # envelope correlation time, measured
    y = Z - Z.mean()
    ac = np.array([np.mean(y[:, :Z.shape[1] - k] * y[:, k:])
                   for k in range(60)])
    ac /= ac[0]
    kk = np.argmax(ac < np.exp(-1.0))
    tau = kk * dt
    dec = max(int(round(tau / dt)), 1)
    s = Z[:, ::dec].ravel()
    d = moments(s, tag)
    d["tau_us"] = tau * 1e6
    d["dec"] = dec
    d["ncell"] = Z.shape[1] / dec
    store[tag] = s
    print(f"  {tag}: gate holds {Z.shape[1]} samples, envelope 1/e "
          f"correlation {tau*1e6:.3f} us -> {d['ncell']:.1f} independent "
          f"range cells/azimuth")
    print(f"        pooled n={d['n']}  CV {d['cv']:.3f}  skew {d['skew']:+.2f}"
          f"  exc-kurt {d['kurt']:+.2f}  E[I2]/E[I]2 {d['Iratio']:.3f}"
          f"  K-alpha {d['alpha']:.2f}")
    rows.append(d)

# block bootstrap over azimuths for the intensity ratio
print("\n  block bootstrap (resample whole azimuths, 4000 draws):")
rng = np.random.default_rng(7)
bs = {}
for nm, tag in (("fittest", "2 MHz"), ("prod5mhz", "5 MHz")):
    E, dt, f0 = gate_env(nm, AZ5)
    prof = np.sqrt((E ** 2).mean(0))
    Z = E / prof
    y = Z - Z.mean()
    ac = np.array([np.mean(y[:, :Z.shape[1] - k] * y[:, k:])
                   for k in range(60)])
    ac /= ac[0]
    dec = max(int(np.argmax(ac < np.exp(-1.0))), 1)
    Zd = Z[:, ::dec]
    out = []
    for _ in range(4000):
        i = rng.integers(0, len(AZ5), len(AZ5))
        s = Zd[i].ravel()
        I = s ** 2
        out.append((I ** 2).mean() / I.mean() ** 2)
    bs[tag] = np.array(out)
    lo, hi = np.percentile(out, [2.5, 97.5])
    print(f"    {tag}: E[I2]/E[I]2 = {np.mean(out):.3f} "
          f"95% CI [{lo:.3f}, {hi:.3f}]")
diff = bs["5 MHz"] - bs["2 MHz"]
print(f"    difference (5 - 2 MHz) = {diff.mean():+.3f} "
      f"95% CI [{np.percentile(diff,2.5):+.3f}, "
      f"{np.percentile(diff,97.5):+.3f}]  "
      f"P(diff>0) = {np.mean(diff>0):.4f}")
print("    MODEL REQUIRES: difference > 0 (heavier tail at 5 MHz)")

# KS against Rayleigh, on the decimated pooled samples
print("\n  KS test against Rayleigh (scale fitted); samples within an "
      "azimuth are\n  still weakly dependent so the p-value is optimistic:")
for tag, s in store.items():
    sc = np.sqrt((s ** 2).mean() / 2)
    ks = stats.kstest(s, "rayleigh", args=(0, sc))
    print(f"    {tag}: D = {ks.statistic:.4f}, p = {ks.pvalue:.3g}, "
          f"n = {len(s)}")

np.savez("t3_speckle.npz", **{k: v for k, v in store.items()})
