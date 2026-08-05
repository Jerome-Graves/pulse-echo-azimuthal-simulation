"""Step 1: reproduce the SOURCE-referenced measured levels and characterise
the recorded excitation pulse (the 'bang'), which is the amplitude reference.

CPU / disk only.  No CUDA, no FDTD.
"""
import os
import numpy as np
from scipy.signal import hilbert

SWD = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "out", "sweeps")))
C, F0, DIA = 3850.0, 2.0e6, 0.100
GATE = (24e-6, 36e-6)

REAL = {6: "girdle_seed11_ppw6_axis_perp", 8: "girdle_seed11_ppw8_dev", 10: "girdle_seed11_ppw10_licensing"}
ZC = {6: "girdle_seed11_ppw6_uniform_axis", 8: "girdle_seed11_ppw8_uniform_axis"}


def load_dir(name):
    d = os.path.join(SWD, name)
    out = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith("az") and f.endswith(".npz") and "tmp" not in f):
            continue
        with np.load(os.path.join(d, f)) as z:
            out[int(f[2:5])] = (np.asarray(z["trace"], float).ravel(),
                                float(z["dt"]))
    return out


def metrics(tr, dt):
    fs = 1.0 / dt
    e = np.abs(hilbert(tr))
    k0, w = int(2 * DIA / C * fs), int(2e-6 * fs)
    a, b = int(GATE[0] * fs), int(GATE[1] * fs)
    kb = int(np.argmax(e))
    return dict(bang=e.max(), t_bang=kb * dt,
                e1=e[max(k0 - w, 0):k0 + w].max(),
                t_e1=(max(k0 - w, 0) + int(np.argmax(e[max(k0-w, 0):k0+w])))*dt,
                coda=np.sqrt((e[a:b] ** 2).mean()), env=e, dt=dt)


def db(x):
    return 20 * np.log10(x)


print("=== A. source-referenced levels, seed 11 girdle k=-8, matched az ===")
R = {p: load_dir(n) for p, n in REAL.items()}
common = sorted(set.intersection(*[set(v) for v in R.values()]))
print(f"{len(common)} matched azimuths: {common[:4]} ... step "
      f"{common[1]-common[0]} deg")
M = {p: {a: metrics(*R[p][a]) for a in common} for p in R}
print(f"{'ppw':>4}{'coda/src dB':>13}{'E1/src dB':>12}{'coda/E1 dB':>12}"
      f"{'dt ns':>8}{'nt':>8}")
for p in (6, 8, 10):
    c = np.array([M[p][a]["coda"] for a in common])
    e1 = np.array([M[p][a]["e1"] for a in common])
    bg = np.array([M[p][a]["bang"] for a in common])
    tr, dt = R[p][common[0]]
    print(f"{p:>4}{db(c/bg).mean():>13.2f}{db(e1/bg).mean():>12.2f}"
          f"{db(c/e1).mean():>12.2f}{dt*1e9:>8.2f}{len(tr):>8d}")

print("\n=== B. zero-contrast control (no scattering): E1/src ===")
for p, n in ZC.items():
    Z = load_dir(n)
    az = sorted(set(Z) & set(common))
    if not az:
        az = sorted(Z)[:30]
    e1 = np.array([metrics(*Z[a])["e1"] for a in az])
    bg = np.array([metrics(*Z[a])["bang"] for a in az])
    cd = np.array([metrics(*Z[a])["coda"] for a in az])
    print(f"  ppw {p}: E1/src {db(e1/bg).mean():7.2f} dB  "
          f"coda/src {db(cd/bg).mean():7.2f} dB  (n={len(az)})")

print("\n=== C. shape of the recorded excitation (ppw10, az000) ===")
tr, dt = R[10][common[0]]
m = metrics(tr, dt)
fs = 1.0 / dt
e = m["env"]
kb = int(np.argmax(e))
print(f"  bang peak at t = {m['t_bang']*1e6:.3f} us "
      f"(source t0 = 1.2/f0 = {1.2/F0*1e6:.3f} us), E1 at "
      f"{m['t_e1']*1e6:.3f} us (2D/c = {2*DIA/C*1e6:.3f} us)")
# effective envelope duration of the bang and of E1
for tag, k, half in (("bang", kb, int(3e-6 * fs)),
                     ("E1", int(m["t_e1"] * fs), int(3e-6 * fs))):
    seg = e[max(k - half, 0):k + half]
    T = np.trapezoid(seg ** 2, dx=dt) / seg.max() ** 2
    print(f"  {tag:5s}: T_env = int env^2 dt / max^2 = {T*1e6:.4f} us "
          f"(= {T*F0:.4f} / f0)")

# theoretical Ricker
t = np.arange(-2000, 2000) * dt
a = (np.pi * F0 * t) ** 2
w = (1 - 2 * a) * np.exp(-a)
ew = np.abs(hilbert(w))
print(f"  ideal 2 MHz Ricker: peak env {ew.max():.4f} (peak w = 1), "
      f"T_env = {np.trapezoid(ew**2, dx=dt)/ew.max()**2*1e6:.4f} us "
      f"({np.trapezoid(ew**2, dx=dt)/ew.max()**2*F0:.4f}/f0)")

# spectrum of the bang vs a Ricker
seg = tr[:kb + int(3e-6 * fs)]
S = np.abs(np.fft.rfft(seg, 4096 * 8))
fr = np.fft.rfftfreq(4096 * 8, dt)
pk = fr[np.argmax(S)]
half = S.max() / 2
band = fr[S >= half]
print(f"  bang spectrum: peak {pk/1e6:.3f} MHz, -6 dB band "
      f"{band[0]/1e6:.2f}-{band[-1]/1e6:.2f} MHz "
      f"({100*(band[-1]-band[0])/pk:.0f} % fractional)")
Sr = np.abs(np.fft.rfft(w, 4096 * 8))
fr2 = np.fft.rfftfreq(4096 * 8, dt)
b2 = fr2[Sr >= Sr.max() / 2]
print(f"  Ricker      : peak {fr2[np.argmax(Sr)]/1e6:.3f} MHz, -6 dB band "
      f"{b2[0]/1e6:.2f}-{b2[-1]/1e6:.2f} MHz")

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "po_src_meas.npz"),
         common=np.array(common),
         coda10=np.array([M[10][a]["coda"] for a in common]),
         e110=np.array([M[10][a]["e1"] for a in common]),
         bang10=np.array([M[10][a]["bang"] for a in common]))
