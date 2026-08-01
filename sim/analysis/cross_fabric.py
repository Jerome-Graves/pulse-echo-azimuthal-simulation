"""ADVERSARIAL TEST 1: does the measured coda level track sigma_d^2 ACROSS
fabrics, at FIXED solver settings?  Theory says eta ~ sigma_d^2, so in dB the
level must move 1:1 with 10log10(sigma_d^2).  A non-grain floor would be flat.

Uses only existing traces + CPU specimen builds.  GPU labeller disabled.
"""
import os, sys, json, glob
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
import specimen as SP
SP.DiskSpecimen._label_grid_gpu = staticmethod(lambda *a, **k: None)  # no GPU
import forward as F
from specimen import DiskSpecimen

OUT = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
C = 3850.0
T_BW = 2 * 0.100 / C
HBUILD = 1.0e-3   # sigma_d^2 shown insensitive to h


def build(conc, axis, seed):
    sp = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                      size_cv=0.35, concentration=conc, spatial_corr=0.0,
                      fabric_axis=axis, seed=seed)
    b = sp.build(HBUILD)
    lab, axes = b["labels"], np.asarray(b["axes"], float)
    ok = lab >= 0
    vol = np.bincount(lab[ok].ravel(), minlength=len(axes)).astype(float)
    keep = vol > 0
    return axes[keep], vol[keep] / vol[keep].sum()


def sig2(axes, vol, rot_deg):
    a = np.radians(rot_deg)
    n = np.array([np.cos(a), np.sin(a), 0.0])
    psi = np.arccos(np.clip(np.abs(axes @ n), 0.0, 1.0))
    v = np.interp(psi, F._PSI, F._VQP)
    vb = float(vol @ v)
    return float(vol @ (v - vb) ** 2) / vb ** 2      # sigma_d^2 (fractional)


def meas(name):
    d = os.path.join(OUT, name)
    az, M = [], []
    for f in sorted(glob.glob(os.path.join(d, "az*.npz"))):
        z = np.load(f); tr = np.asarray(z["trace"], float).ravel(); dt = float(z["dt"]); z.close()
        fs = 1 / dt
        e = np.abs(hilbert(tr))
        k0 = int(T_BW * fs); w = int(2e-6 * fs)
        lo = max(k0 - w, 0)
        E1 = e[lo:k0 + w].max()
        c = np.sqrt((e[int(24e-6 * fs):int(36e-6 * fs)] ** 2).mean())
        az.append(int(os.path.basename(f)[2:5])); M.append((c / E1) ** 2)
    return np.array(az), np.array(M)


CASES = [
    ("girdle_perp",  -8.0,  (1.0, 0.0, 0.0),   11, "single-raster ppw6"),
    ("girdle_par",   -8.0,  (0.0, 0.0, 1.0),   11, "single-raster ppw6"),
    ("rigid_seed11",  3.93, (0.866, 0.5, 0.0), 11, "LEGACY ppw6"),
    ("iso_gcal",      0.001,(1.0, 0.0, 0.0),   41, "LEGACY ppw6"),
    ("oos_seed23",    3.93, (-0.342, 0.94, 0.0), 23, "LEGACY ppw6"),
    ("fittest",       3.93, (1.0, 0.0, 0.0),    7, "LEGACY ppw6"),
    ("kappa8_seed17", 8.0,  (0.5, 0.866, 0.0), 17, "LEGACY ppw6"),
    ("gcheck_ppw8",   3.93, (0.866, 0.5, 0.0), 11, "LEGACY ppw8"),
    ("girdle_perp_ppw8", -8.0, (1.0, 0.0, 0.0), 11, "single-raster ppw8"),
]

print(f"{'sweep':<20}{'group':<22}{'n':>4}{'<sig_d^2>':>11}{'dB(sig2)':>10}"
      f"{'dB(<P>)':>9}")
rows = []
for nm, conc, ax, sd, grp in CASES:
    axes, vol = build(conc, ax, sd)
    az, P = meas(nm)
    s2 = np.array([sig2(axes, vol, a) for a in az])
    rows.append(dict(name=nm, grp=grp, az=az, P=P, s2=s2))
    print(f"{nm:<20}{grp:<22}{len(az):>4}{s2.mean():>11.4e}"
          f"{10*np.log10(s2.mean()):>10.2f}{10*np.log10(P.mean()):>9.2f}")

np.save(os.path.join(os.path.dirname(__file__), "ref2_rows.npy"), rows,
        allow_pickle=True)

print("\n--- within a fixed solver group, does level track sigma_d^2 1:1? ---")
for grp in ["single-raster ppw6", "LEGACY ppw6"]:
    sub = [r for r in rows if r["grp"] == grp]
    if len(sub) < 2:
        continue
    x = np.array([10 * np.log10(r["s2"].mean()) for r in sub])
    y = np.array([10 * np.log10(r["P"].mean()) for r in sub])
    print(f"\n{grp}: n_specimen={len(sub)}")
    for r, xi, yi in zip(sub, x, y):
        print(f"   {r['name']:<18} 10log10(sig2)={xi:8.2f}   10log10(<P>)={yi:8.2f}")
    if len(sub) > 2:
        sl, ic = np.polyfit(x, y, 1)
        print(f"   slope = {sl:+.3f}  (theory demands +1.000);  "
              f"corr = {np.corrcoef(x, y)[0,1]:+.3f}")
    else:
        print(f"   delta_sig2 = {x[1]-x[0]:+.2f} dB   delta_level = {y[1]-y[0]:+.2f} dB"
              f"   -> ratio {(y[1]-y[0])/(x[1]-x[0]):+.2f} (theory 1.00)")
