"""Can the grid mode be PREDICTED from the scheme, not just observed?

For a staggered-grid first derivative with coefficients c_n (n = 1..N) the
Fourier symbol is

    D(k) = (2/h) * sum_n c_n * sin((n - 1/2) * k h)

whose exact counterpart is k.  So

    phase velocity ratio  v_p/v = D(k)/k  = (2/(kh)) sum c_n sin((n-1/2)kh)
    group velocity ratio  v_g/v = dD/dk   = 2 sum c_n (n-1/2) cos((n-1/2)kh)

The failure that matters for backscatter is NOT phase error.  It is the
wavenumber where the GROUP velocity collapses: energy there stops
propagating, accumulates, and radiates as a spurious mode.  If that
wavenumber matches the kh where the measured coda energy piles up, the
artefact is predicted rather than merely reported - which is what turns
this from an erratum into a mechanism.

Note kh = 2*pi/ppw, so a source at ppw 6 sits at kh = 1.047, and its
SECOND harmonic content sits at kh = 2.09, deep into the bad region.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
import fdtd                                            # noqa: E402


def sym(c, kh):
    n = np.arange(1, len(c) + 1) - 0.5
    return 2.0 * (c[None, :] * np.sin(np.outer(kh, n))).sum(1)


def vg(c, kh):
    n = np.arange(1, len(c) + 1) - 0.5
    return 2.0 * (c[None, :] * n * np.cos(np.outer(kh, n))).sum(1)


kh = np.linspace(1e-4, np.pi, 20000)
print("kh = 2*pi/ppw, so ppw 4/5/6/8/10 -> kh "
      + "/".join(f"{2*np.pi/p:.2f}" for p in (4, 5, 6, 8, 10)))
print()
hdr = (f"{'scheme':<26}{'vg=0 at kh':>12}{'= ppw':>8}"
       f"{'vp err @ppw6':>14}{'vp err @ppw8':>14}{'vg @ 2x f0(ppw6)':>18}")
print(hdr)
print("-" * len(hdr))
rows = []
for order, khm, ms, tag in ((4, 1.6, False, "order 4 legacy"),
                            (6, 1.6, False, "order 6 legacy"),
                            (8, 1.6, False, "order 8 legacy (kh1.6)"),
                            (8, 2.0, True, "order 8 multistart (kh2.0)"),
                            (4, 2.0, True, "order 4 multistart"),
                            (6, 2.0, True, "order 6 multistart")):
    try:
        c = fdtd.optimised_coeffs(order, kh_max=khm, multistart=ms)
    except TypeError:
        c = fdtd.optimised_coeffs(order)
    g = vg(c, kh)
    s = np.where(np.diff(np.sign(g)))[0]
    kz = kh[s[0]] if len(s) else np.nan
    vp = sym(c, kh) / kh
    e6 = np.interp(2 * np.pi / 6, kh, vp) - 1
    e8 = np.interp(2 * np.pi / 8, kh, vp) - 1
    g2 = np.interp(2 * 2 * np.pi / 6, kh, vg(c, kh))
    rows.append((tag, kz, c))
    print(f"{tag:<26}{kz:>12.3f}{2*np.pi/kz if kz==kz else np.nan:>8.2f}"
          f"{100*e6:>13.2f}%{100*e8:>13.2f}%{g2:>18.3f}")

print("\nMEASURED: coda energy piles up near kh ~ 2.55 at ppw 5-6")
print("          (spectral peak near 2x the source frequency)")
print("\nEnergy at N times the source frequency sits at N * kh_source:")
for ppw in (5, 6, 8, 10):
    k0 = 2 * np.pi / ppw
    print(f"  ppw {ppw:>2}: source kh {k0:.3f}   2x -> {2*k0:.3f}   "
          f"3x -> {3*k0:.3f}"
          + ("   <- 2x is past vg=0" if 2 * k0 > rows[2][1] else ""))

print("\nGroup velocity ratio at the second harmonic, per resolution "
      "(order 8 legacy):")
c = rows[2][2]
for ppw in (4, 5, 6, 8, 10):
    k2 = 2 * (2 * np.pi / ppw)
    v = float(vg(c, np.array([min(k2, np.pi)]))[0])
    print(f"  ppw {ppw:>2}: kh(2f0) = {k2:5.3f}  vg/v = {v:+.3f}"
          + ("   STALLED / BACKWARD" if v < 0.2 else ""))
