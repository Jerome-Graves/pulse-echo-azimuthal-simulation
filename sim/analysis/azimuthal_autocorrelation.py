"""Strand 3, step 2: azimuthal autocorrelation + harmonic spectrum of the
coda level, for the ISOTROPIC control (all of whose azimuthal variation is
speckle by construction) and for the anisotropic sweeps for comparison.
"""
import numpy as np

SCR = r"C:\Users\Jerome\AppData\Local\Temp\claude\C--Users-Jerome\72aa31c0-c0c1-48de-881e-9470fe03e8ba\scratchpad"
Z = np.load(SCR + r"\coda_levels.npz")


def circ_acf(x):
    """Unbiased-ish circular autocorrelation, normalised to rho(0)=1."""
    y = x - x.mean()
    n = len(y)
    F = np.fft.rfft(y, n=2 * n)          # linear (non-wrapped) autocorr
    ac = np.fft.irfft(F * np.conj(F), n=2 * n)[:n]
    ac /= (n - np.arange(n))             # unbiased normalisation
    return ac / ac[0]


def harmonics(x, kmax=40):
    """A_k = amplitude of the best-fit A cos(k th - ph) on a uniform grid."""
    n = len(x)
    F = np.fft.rfft(x - x.mean()) / n
    A = 2.0 * np.abs(F)
    return A[:kmax + 1]


def report(name, step_deg):
    lv = Z[name + "_lv"]
    rot = Z[name + "_rot"]
    n = len(lv)
    sd = lv.std(ddof=1)
    print(f"\n=== {name}  n={n}  step={step_deg} deg  sd={sd:.2f} dB ===")

    ac = circ_acf(lv)
    lags = np.arange(len(ac)) * step_deg
    # 1/e length
    below = np.where(ac < np.exp(-1.0))[0]
    l_e = lags[below[0]] if len(below) else np.nan
    # first zero crossing
    zc = np.where(ac < 0)[0]
    l_0 = lags[zc[0]] if len(zc) else np.nan
    # integral correlation length (sum to first zero crossing, Bartlett)
    m = zc[0] if len(zc) else min(len(ac), 30)
    tau_int = (1.0 + 2.0 * ac[1:m].sum()) * step_deg      # degrees
    neff = 360.0 / max(tau_int, step_deg)
    print(f"  acf(lag) : " + "  ".join(
        f"{lags[i]:.0f}d={ac[i]:+.2f}" for i in range(0, min(9, len(ac)))))
    print(f"  1/e correlation length      : {l_e:.1f} deg")
    print(f"  first zero crossing         : {l_0:.1f} deg")
    print(f"  integral corr length tau    : {tau_int:.2f} deg")
    print(f"  N_eff in 360 deg (=360/tau) : {neff:.1f}")

    A = harmonics(lv, kmax=min(40, len(lv) // 2))
    print(f"  A_k (dB amplitude) k=1..12 : " +
          " ".join(f"{A[k]:.2f}" for k in range(1, min(13, len(A)))))
    kk = np.arange(1, len(A))
    print(f"  rms A_k over k=1..{kk[-1]}      : {np.sqrt((A[1:]**2).mean()):.3f} dB")
    print(f"  A_2={A[2]:.3f}  A_4={A[4]:.3f}  "
          f"(rms modulation of that harmonic alone = A/sqrt2 = "
          f"{A[2]/np.sqrt(2):.3f}, {A[4]/np.sqrt(2):.3f} dB)")
    return lv, A, tau_int


for nm, st in (("iso_gcal", 1.0), ("rigid_seed11", 1.0), ("oos_seed23", 1.0),
               ("girdle_perp", 6.0), ("girdle_par", 6.0)):
    report(nm, st)

# ---- spectral shape of the isotropic control: white or coloured? ----
print("\n\n=== isotropic control: is the speckle white in azimuth? ===")
lv = Z["iso_gcal_lv"]
A = harmonics(lv, kmax=179)
P = A[1:] ** 2
k = np.arange(1, len(A))
for lo, hi in ((1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 179)):
    sel = (k >= lo) & (k <= hi)
    print(f"  k {lo:3d}-{hi:3d}: mean A_k = {np.sqrt(P[sel].mean()):.3f} dB   "
          f"power fraction = {P[sel].sum()/P.sum()*100:5.1f} %")
print(f"  total variance from harmonics = {0.5*P.sum():.2f} dB^2, "
      f"direct var = {lv.var():.2f} dB^2")

# ---- 180 degree symmetry check (same chord, opposite end) ----
print("\n=== 180 deg reciprocity check (same diameter, other end) ===")
for nm in ("iso_gcal", "rigid_seed11", "oos_seed23"):
    x = Z[nm + "_lv"]
    a, b = x[:180], x[180:]
    r = np.corrcoef(a, b)[0, 1]
    print(f"  {nm:<14} corr(theta, theta+180) = {r:+.3f}  "
          f"rms diff = {np.std(a-b):.2f} dB")
