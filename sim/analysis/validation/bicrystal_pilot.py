"""BICRYSTAL PILOT: how much spurious coda does a STAIRCASED elastic
interface generate, and does sub-cell material representation remove it?

Geometry chosen so the metric needs no reference solution: a single
planar ice/ice interface TILTED 30 deg to the grid, interrogated
monostatically. A correctly represented tilted plane steers its
specular reflection AWAY from the source, so almost everything
received back is spurious - staircase diffraction. Lower = better.

Variants at the same grid spacing:
  A staircased      - cell takes whichever crystal owns its centre
                      (what the production solver does today)
  B volume-averaged - boundary cells carry f*C_A + (1-f)*C_B with f the
                      exact volume fraction (4^3 supersampled). This is
                      arithmetic/Voigt blending: the simplest sub-cell
                      representation, and the one Schoenberg-Muir would
                      refine for anisotropic-anisotropic contacts.
  C smoothed        - B, then a 3-tap [1 2 1] low-pass along the plane
                      normal: a crude band-limited step.
Plus A at half the grid spacing, as the refinement comparison.
"""
import sys
import numpy as np
from scipy.signal import hilbert

SP = (r"C:\Users\Jerome\AppData\Local\Temp\claude\C--Users-Jerome"
      r"\72aa31c0-c0c1-48de-881e-9470fe03e8ba\scratchpad")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\pulse-echo-cof-sim\sim")
sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
from ringfwi import anisotropy as an              # noqa: E402
import fdtd                                       # noqa: E402

C_REF, F0, ORDER, SPONGE = 3850.0, 2.0e6, 8, 12
BOX = 0.040                      # 40 mm cube of pure ice
TILT = 30.0                      # interface tilt to the grid, degrees
X_IF = 0.020                     # interface passes through x = 20 mm
SRC_X = 0.004                    # source plane, 4 mm in
AXIS_A = np.array([1.0, 0.0, 0.0])
AXIS_B = np.array([np.cos(np.radians(55.0)), np.sin(np.radians(55.0)), 0.0])


def build(ppw, mode):
    h = C_REF / F0 / ppw
    n = int(round(BOX / h))
    u = (np.arange(n) + 0.5) * h
    X, Y, Z = np.meshgrid(u, u, u, indexing="ij")
    nrm = np.array([np.cos(np.radians(TILT)), np.sin(np.radians(TILT)), 0.0])
    d = (X - X_IF) * nrm[0] + (Y - BOX / 2) * nrm[1]
    if mode == "uniform":
        f = np.ones_like(d, dtype=np.float32)   # all crystal A: control
    elif mode == "stair":
        f = (d <= 0).astype(np.float32)
    else:                                   # exact volume fraction
        f = np.zeros_like(d, dtype=np.float32)
        s = (np.arange(4) + 0.5) / 4.0 - 0.5
        for dx in s:
            for dy in s:
                dd = d + dx * h * nrm[0] + dy * h * nrm[1]
                f += (dd <= 0)
        f /= 16.0
        if mode == "smooth":
            g = np.zeros_like(f)
            for k, wgt in ((-1, 0.25), (0, 0.5), (1, 0.25)):
                g += wgt * np.roll(f, k, axis=0)
            f = g
    lab = np.zeros((n, n, n), np.int32)
    CA, rho = an.polycrystal_stiffness_3d(lab, AXIS_A[None, :],
                                          c_couplant=1500.0,
                                          rho_couplant=300.0)
    CB, _ = an.polycrystal_stiffness_3d(lab, AXIS_B[None, :],
                                        c_couplant=1500.0,
                                        rho_couplant=300.0)
    # C is a dict of 21 Voigt components, each (n,n,n): blend per key.
    # Arithmetic (Voigt) blending of the stiffness is the simplest
    # sub-cell rule; density is uniform ice so it needs no blending.
    C = {k: (f * CA[k] + (1.0 - f) * CB[k]).astype(np.float32)
         for k in CA}
    return C, np.asarray(rho, np.float32), h, n


def run(ppw, mode):
    C, rho, h, n = build(ppw, mode)
    co = fdtd.optimised_coeffs(ORDER)
    dt = fdtd.safe_dt(C, rho, h, co, safety=0.5)
    nt = int(16e-6 / dt)
    wav = fdtd.ricker(F0, dt, nt)
    c = n // 2
    ix = int(round(SRC_X / h))
    er = max(int(3.0e-3 / h), 1)
    pts = [(c + dz, c + dy, ix) for dy in range(-er, er + 1)
           for dz in range(-er, er + 1) if dy * dy + dz * dz <= er * er]
    w = 1.0 / len(pts)
    tr = np.asarray(fdtd.forward_fused(
        C, rho, h, dt, nt, [(p, w) for p in pts], wav,
        [(pts, np.full(len(pts), w))], order=ORDER, coeffs=co,
        sponge_width=SPONGE), float).ravel()
    print(f"  {mode:9s} ppw {ppw:4.1f}: grid {n}^3, dt {dt*1e9:.2f} ns, "
          f"{nt} steps", flush=True)
    return tr, dt


def metric(tr, dt):
    """Spurious backscatter: RMS envelope in a window that starts after
    the source ringdown and ends before the box edges return."""
    fs = 1.0 / dt
    e = np.abs(hilbert(tr))
    src = e[:int(3e-6 * fs)].max()
    w = e[int(6e-6 * fs):int(13e-6 * fs)]
    seg = tr[int(6e-6 * fs):int(13e-6 * fs)]
    F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    fr = np.fft.rfftfreq(len(seg), dt)
    return (20 * np.log10(np.sqrt((w ** 2).mean()) / src),
            fr[np.argmax(F)] / 1e6, F[fr > 3e6].sum() / F.sum() * 100)


print(f"bicrystal: ice/ice interface tilted {TILT} deg, monostatic, "
      f"orientations 0 and 55 deg")
print("metric is the DIFFERENCE against a uniform-medium run, which\n"
      "cancels the source-rasterisation and sponge artefacts that\n"
      "otherwise dominate this geometry (first attempt measured those\n"
      "and returned identical numbers for every variant).\n")
out = {}
uni6, dt6 = run(6.0, "uniform")
for mode in ("stair", "vfrac", "smooth"):
    tr, dt = run(6.0, mode)
    out[mode] = metric(tr - uni6, dt)
uni12, dt12 = run(12.0, "uniform")
tr, dt = run(12.0, "stair")
out["stair@12"] = metric(tr - uni12, dt)

print("\n%-12s %-22s %-13s %s"
      % ("variant", "spurious re source (dB)", "peak (MHz)", "%>3MHz"))
for k in ("stair", "vfrac", "smooth", "stair@12"):
    v = out[k]
    print("%-12s %10.1f            %8.2f      %6.1f" % (k, v[0], v[1], v[2]))
b = out["stair"][0]
print("\nrelative to staircased at ppw 6:")
for k in ("vfrac", "smooth", "stair@12"):
    print("  %-10s %+6.1f dB" % (k, out[k][0] - b))
np.savez(SP + r"\bicrystal.npz", **{k: np.array(v) for k, v in out.items()})
