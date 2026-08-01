"""ADVERSARIAL TEST 3 -- THE MISSING CONTROL, recovered from data already on disk.

In these sweeps ONLY the grain labels rotate.  rotation_test.rotated_grid builds
the disc mask on the axis-aligned solver grid (ins = X^2+Y^2 <= Rd^2), so the
staircased cylinder rim, the transducer, the sponge and the grid are IDENTICAL
at every azimuth.  Therefore:

   trace(t, az) = F(t)                 [lab-frame-fixed: rim/edge/source/sponge/
                                        staircase of the outer boundary]
                + g(t, az)             [grain scattering, decorrelates with az]

The coherent average over azimuths converges to F(t); the incoherent residual is
g.  This gives the grain-free floor WITHOUT running a new simulation -- exactly
the control the strand says does not exist.

Guard against a false positive: if F were really zero, the coherent mean of n
random-phase traces still has power P_inc/n.  So compare
   P_coh = |mean_az trace|^2   against   P_inc/n.
Excess over P_inc/n is a genuine lab-fixed component.
Also split the azimuths into two halves and cross-correlate the two independent
coherent means: a real F reproduces, sampling noise does not.
"""
import os, glob
import numpy as np
from scipy.signal import hilbert

OUT = r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\out\sweeps"
C = 3850.0
T_BW = 2 * 0.100 / C
GATE = (24e-6, 36e-6)


def load(name, nmax=400):
    d = os.path.join(OUT, name)
    trs, dts, azs = [], None, []
    for f in sorted(glob.glob(os.path.join(d, "az*.npz")))[:nmax]:
        z = np.load(f); tr = np.asarray(z["trace"], float).ravel(); dt = float(z["dt"]); z.close()
        trs.append(tr); dts = dt; azs.append(int(os.path.basename(f)[2:5]))
    n = min(len(t) for t in trs)
    return np.array([t[:n] for t in trs]), dts, np.array(azs)


def norm_by_backwall(T, dt):
    """scale each trace by its own backwall peak envelope, and align in time."""
    fs = 1 / dt
    k0 = int(T_BW * fs); w = int(2e-6 * fs)
    out, shifts = [], []
    for tr in T:
        e = np.abs(hilbert(tr))
        lo = max(k0 - w, 0)
        kp = lo + int(np.argmax(e[lo:k0 + w]))
        out.append(tr / e[kp]); shifts.append(kp)
    shifts = np.array(shifts)
    ref = int(round(shifts.mean()))
    al = []
    for tr, s in zip(out, shifts):
        al.append(np.roll(tr, ref - s))
    return np.array(out), np.array(al), shifts, ref


def gate_power(x, dt, gate=GATE):
    fs = 1 / dt
    seg = x[..., int(gate[0] * fs):int(gate[1] * fs)]
    return (np.abs(hilbert(seg, axis=-1)) ** 2).mean(axis=-1)


print("P_inc = mean single-azimuth coda power (dB re backwall peak)")
print("P_coh = coda power of the COHERENT AVERAGE over azimuths")
print("P_inc/n = what P_coh would be from pure sampling noise if there were "
      "NO lab-fixed component")
print("floor_frac = fraction of single-azimuth coda power that is lab-fixed\n")
hdr = (f"{'sweep':<20}{'n':>4}{'P_inc':>8}{'P_coh':>8}{'noise':>8}"
       f"{'floor':>8}{'frac%':>7}{'r_half':>8}{'align':>7}")
print(hdr)

for nm in ["iso_gcal", "rigid_seed11", "oos_seed23", "fittest",
           "girdle_perp", "girdle_par", "girdle_perp_ppw8", "gcheck_ppw8"]:
    T, dt, az = load(nm)
    for tag, X in (("raw", None), ("aligned", None)):
        pass
    Tn, Ta, sh, ref = norm_by_backwall(T, dt)
    for tag, X in (("raw", Tn), ("algn", Ta)):
        n = len(X)
        P_inc = gate_power(X, dt).mean()
        coh = X.mean(axis=0)
        P_coh = gate_power(coh, dt)
        noise = P_inc / n
        floor = max(P_coh - noise, 1e-30)
        # split-half reproducibility of the coherent mean inside the gate
        i = np.arange(n)
        A = X[i % 2 == 0].mean(axis=0); B = X[i % 2 == 1].mean(axis=0)
        fs = 1 / dt
        sl = slice(int(GATE[0] * fs), int(GATE[1] * fs))
        r = np.corrcoef(A[sl], B[sl])[0, 1]
        print(f"{nm:<20}{n:>4}{10*np.log10(P_inc):>8.1f}"
              f"{10*np.log10(P_coh):>8.1f}{10*np.log10(noise):>8.1f}"
              f"{10*np.log10(floor):>8.1f}{100*floor/P_inc:>7.1f}"
              f"{r:>8.3f}{tag:>7}")
