"""THE gold-standard cross-check: staircased FDTD vs grain-conforming FE.

Same 12-grain Voronoi box (same seeds, same per-grain stiffness tables),
run through both solvers. The A-B trick isolates grain scattering with
each solver's source function common-moded away: run each solver with the
REAL random c-axes (A) and with UNIFORM axes (B); the difference trace
contains only what the grain boundaries scattered. Each solver's A-B coda
is normalised by its OWN direct-arrival amplitude, so monopole-vs-force
source differences cancel too.

Verdict criterion: FE and FDTD (A-B) coda RMS within ~2-3 dB in the
matched window -> the staircase representation of grain boundaries is
validated at the ensemble-observable level by a method with NO staircase.
1 MHz centre frequency: FE mesh lambda/15 (0.5% dispersion), FDTD grid
lambda/12 (its cleanest regime).
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
from scipy.signal import hilbert                # noqa: E402

import fdtd                                     # noqa: E402
import fe_mesh                                  # noqa: E402
import fe_solver                                # noqa: E402

L = 0.030                                       # 30 mm box
F0 = 1.0e6
NG = 12
SRC = np.array([0.006, L / 2, L / 2])
REC = np.array([0.015, L / 2, L / 2])


def seeds():
    # rng 1, not 7: seed-set 7 contains a cell with nearly-coplanar faces
    # that defeats surface meshing (documented failure); the cross-check is
    # realisation-agnostic as long as BOTH solvers get the same specimen.
    rng = np.random.default_rng(1)
    return rng.uniform(0.12 * L, 0.88 * L, (NG, 3))


def axes_sets():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(NG, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.tile(a[0][None, :], (NG, 1))
    return a, b


def run_fe(h_mesh):
    import os
    sd = seeds()
    if os.path.exists("fe_cc_mesh.npz"):
        d = np.load("fe_cc_mesh.npz")
        nodes, tets, grain = d["nodes"], d["tets"], d["grain"]
        print(f"reusing mesh: {len(tets)} tets", flush=True)
    else:
        nodes, tets, grain = fe_mesh.mesh_polycrystal(
            sd, ((0, L), (0, L), (0, L)), h_mesh, "fe_cc_mesh.npz")
    aA, aB = axes_sets()
    dt = fe_solver.stable_dt(nodes, tets, 4046.0)
    nt = int(9e-6 / dt)
    wav = fdtd.ricker(F0, dt, nt)

    def nearest(x):
        return int(np.argmin(np.sum((nodes - x) ** 2, axis=1)))
    src, rec = nearest(SRC), nearest(REC)
    out = {}
    for tag, ax in (("A", aA), ("B", aB)):
        t0 = time.time()
        out[tag] = fe_solver.forward_fe(nodes, tets, grain, ax, dt, nt,
                                        src, wav, [rec])[:, 0]
        print(f"FE {tag}: {time.time()-t0:.0f}s "
              f"({len(tets)} tets, {nt} steps)", flush=True)
    return out["A"], out["B"], dt


def run_fdtd():
    sd = seeds()
    h = 4046.0 / F0 / 12.0
    n = int(round(L / h))
    m = 13
    N = n + 2 * m
    ax_g = (np.arange(n) + 0.5) * h
    Z, Y, X = np.meshgrid(ax_g, ax_g, ax_g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    d2 = ((P[:, None, :] - sd[None, :, :]) ** 2).sum(-1)
    core = d2.argmin(1).astype(np.int16).reshape(n, n, n)
    lab = np.full((N, N, N), 0, np.int16)
    lab[m:m+n, m:m+n, m:m+n] = core          # box fully filled with grains
    aA, aB = axes_sets()
    co = fdtd.optimised_coeffs(8)
    mats = (lab + 1).astype(np.uint8)
    Ct, rho_t = fdtd.material_tables(aA)
    dt = fdtd.safe_dt_labels(mats, Ct, rho_t, h, co, safety=0.5)
    nt = int(9e-6 / dt)
    wav = fdtd.ricker(F0, dt, nt)

    def cell(x):
        return tuple(int(m + q / h) for q in (x[2], x[1], x[0]))  # (z,y,x)
    src, rec = cell(SRC), cell(REC)
    out = {}
    for tag, ax in (("A", aA), ("B", aB)):
        t0 = time.time()
        tr = np.asarray(fdtd.forward_fused_labels(
            lab, ax, h, dt, nt, [(src, 1.0)], wav, [([rec], [1.0])],
            order=8, coeffs=co, sponge_width=10), float).ravel()
        out[tag] = tr
        print(f"FDTD {tag}: {time.time()-t0:.0f}s "
              f"(grid {lab.shape}, {nt} steps)", flush=True)
    return out["A"], out["B"], dt


def analyse(a, b, dt, name):
    fs = 1.0 / dt
    t0_w = 1.2 / F0                              # ricker centre delay
    Ld = np.linalg.norm(REC - SRC)
    t_dir = Ld / 4046.0 + t0_w
    e_dir = np.abs(hilbert(b))
    k = int(t_dir * fs)
    direct = e_dir[max(k - int(1e-6 * fs), 0):k + int(1e-6 * fs)].max()
    diff = a - b
    # coda window: after the direct pulse, before first edge returns
    w0 = int((t_dir + 1.5e-6) * fs)
    w1 = int(min((2 * 0.015 / 3800.0 + t_dir + 2.5e-6), 8.8e-6) * fs)
    cw = np.abs(hilbert(diff))[w0:w1]
    lvl = 20 * np.log10(np.sqrt((cw ** 2).mean()) / direct + 1e-30)
    print(f"{name}: direct {direct:.3e}, A-B coda RMS {lvl:.1f} dB "
          f"re direct  (window {w0/fs*1e6:.1f}-{w1/fs*1e6:.1f} us)")
    return lvl


def main():
    h_mesh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.27e-3
    fa, fb, dtf = run_fdtd()
    lf = analyse(fa, fb, dtf, "FDTD (staircase)")
    ea, eb, dte = run_fe(h_mesh)
    le = analyse(ea, eb, dte, "FE (conforming)")
    print(f"\nCROSS-CHECK: FDTD {lf:.1f} vs FE {le:.1f} dB -> "
          f"difference {lf-le:+.1f} dB "
          f"({'PASS' if abs(lf-le) < 3.0 else 'INVESTIGATE'} at 3 dB)")


if __name__ == "__main__":
    main()
