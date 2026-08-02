"""dB reconciliation. The blocking item.

Three headline figures do not add up. Holding the physical coda fixed
across resolutions, the measured levels imply the numerical floor moves
about 22 dB between ppw 6 and 8, i.e. power as h^18, which no mechanism
produces.

Hypothesis to test: the figures were measured with DIFFERENT
normalisations, and coda/E1 double-counts. The backwall E1 is itself
attenuated by scattering, part of which is numerical, so refining the
grid raises E1 AND lowers the coda; their ratio then moves roughly twice
as far as either. Normalising by the SOURCE amplitude instead should
remove that.

Everything below is measured at MATCHED azimuths on the FIXED code path,
with one metric at a time, so nothing is compared across conventions.
"""
import os

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

SWD = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
C, F0, DIA = 3850.0, 2.0e6, 0.100
GATE, BAND = (24e-6, 36e-6), (0.8e6, 3.0e6)

# fixed-code-path sweeps of the SAME specimen (seed 11, girdle k=-8)
REAL = {6: "girdle_perp", 8: "girdle_perp_ppw8", 10: "lic_girdle_s11_ppw10"}
# Zero-contrast controls. All three are seed 11 on the same code path as
# the specimen sweeps, so the floor is measured on the geometry it is
# being compared against. zerocontrast_ppw6 is deliberately NOT used: it
# predates the production programme and holds only 22 of 30 azimuths.
FLOOR = {6: "zc_s11_ppw6", 8: "zerocontrast_ppw8", 10: "zc_s11_ppw10"}


def load(name):
    d = os.path.join(SWD, name)
    if not os.path.isdir(d):
        return None
    out = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz")):
            continue
        with np.load(os.path.join(d, f)) as z:
            tr = np.asarray(z["trace"], float).ravel()
            dt = float(z["dt"])
        fs = 1.0 / dt
        k0, w = int(2 * DIA / C * fs), int(2e-6 * fs)
        if k0 + w >= len(tr):
            continue
        er = np.abs(hilbert(tr))
        sos = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)],
                     btype="band", output="sos")
        ef = np.abs(hilbert(sosfiltfilt(sos, tr)))
        a, b = int(GATE[0] * fs), int(GATE[1] * fs)
        out[int(f[2:5])] = dict(
            bang=er.max(),
            e1=er[max(k0 - w, 0):k0 + w].max(),
            coda_raw=np.sqrt((er[a:b] ** 2).mean()),
            coda_bl=np.sqrt((ef[a:b] ** 2).mean()))
    return out


def db(x):
    return 20 * np.log10(x)


R = {p: load(n) for p, n in REAL.items()}
F = {p: load(n) for p, n in FLOOR.items()}
have = [p for p in (6, 8, 10) if R.get(p)]
common = set.intersection(*[set(R[p]) for p in have])
common = sorted(common)
print("fixed-code-path sweeps of seed 11 girdle, matched azimuths")
print("resolutions present: %s ; %d common azimuths\n"
      % (have, len(common)))

print("%-4s %10s %10s %10s %10s %10s"
      % ("ppw", "coda/E1", "coda/src", "E1/src", "coda_bl/E1", "n"))
lev = {}
for p in have:
    v = R[p]
    c = np.array([v[a]["coda_raw"] for a in common])
    cb = np.array([v[a]["coda_bl"] for a in common])
    e1 = np.array([v[a]["e1"] for a in common])
    bg = np.array([v[a]["bang"] for a in common])
    lev[p] = dict(coda_e1=db(c / e1).mean(), coda_src=db(c / bg).mean(),
                  e1_src=db(e1 / bg).mean(), codabl_e1=db(cb / e1).mean(),
                  coda_src_lin=(c / bg).mean())
    print("%-4d %9.2fd %9.2fd %9.2fd %9.2fd %10d"
          % (p, lev[p]["coda_e1"], lev[p]["coda_src"], lev[p]["e1_src"],
             lev[p]["codabl_e1"], len(common)))

print("\nSTEP-BY-STEP CHANGES (this is where the double count shows)")
print("%-10s %12s %12s %12s" % ("step", "coda/E1", "coda/src", "E1/src"))
for i in range(len(have) - 1):
    p, q = have[i], have[i + 1]
    print("%-10s %+11.2fd %+11.2fd %+11.2fd"
          % ("ppw %d->%d" % (p, q),
             lev[q]["coda_e1"] - lev[p]["coda_e1"],
             lev[q]["coda_src"] - lev[p]["coda_src"],
             lev[q]["e1_src"] - lev[p]["e1_src"]))

print("\nFLOOR MEASUREMENTS (zero contrast, same metric)")
for p in (6, 8, 10):
    if not F.get(p):
        print("  ppw %-2d  not yet run" % p)
        continue
    az = sorted(set(F[p]) & set(R[p]))
    if not az:
        print("  ppw %-2d  no matched azimuths (%d floor az available)"
              % (p, len(F[p])))
        continue
    fc = np.array([F[p][a]["coda_raw"] for a in az])
    fb = np.array([F[p][a]["bang"] for a in az])
    rc = np.array([R[p][a]["coda_raw"] for a in az])
    rb = np.array([R[p][a]["bang"] for a in az])
    f_src, r_src = db(fc / fb).mean(), db(rc / rb).mean()
    frac = 10 ** ((f_src - r_src) / 10)
    print("  ppw %-2d  real %7.2f dB   floor %7.2f dB   "
          "numerical %5.1f%% of power   (n=%d, source-normalised)"
          % (p, r_src, f_src, 100 * frac, len(az)))
