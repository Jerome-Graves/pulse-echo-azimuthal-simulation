import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
"""Arbiter shot round 4: src/rec moved 6 mm deeper (same 9 mm path).

Round 3 (source-matched) delivered the attenuation cross-validation
(direct A/B: FDTD -1.37 dB vs FE -1.34 dB - 0.03 dB agreement!) and
localised the remaining coda-window contamination to mirror paths off
the x=0 boundary 6 mm behind the source: shell-entrance reflection AND
the A-side pad/specimen interface echo both arrive at 6.39 us (onset
~5.5), matching the observed ramp. With SRC x=12, REC x=21 every x=0
mirror arrives after 9.3 us - outside the 4.9-6.2 us window entirely,
at zero remesh cost. Any residual ~5.9 us feature is then numerical
S leakage of the discrete monopole, cleanly separated.

FDTD runs the same shifted geometry (its own margin/sponge as always).
"""
import time

import numpy as np

import fdtd                                     # noqa: E402
import fe_solver_p2plus as pp                   # noqa: E402
from fe_crosscheck import L, F0, seeds, axes_sets  # noqa: E402
from fe_p2plus_ab2 import get_mesh, SHELL_FRAC, SHELL_RATE  # noqa: E402

SRC = np.array([0.012, L / 2, L / 2])
REC = np.array([0.021, L / 2, L / 2])
W = (4.9e-6, 6.2e-6)
T = 7.0e-6


def analyse(a, b, dt, name):
    from scipy.signal import hilbert
    fs = 1.0 / dt
    t_dir = np.linalg.norm(REC - SRC) / 4046.0 + 1.2 / F0
    e_dir = np.abs(hilbert(b))
    k = int(t_dir * fs)
    direct = e_dir[max(k - int(1e-6 * fs), 0):k + int(1e-6 * fs)].max()
    cw = np.abs(hilbert(a - b))[int(W[0] * fs):int(W[1] * fs)]
    lvl = 20 * np.log10(np.sqrt((cw ** 2).mean()) / direct + 1e-30)
    print(f"{name}: direct {direct:.3e}, A-B coda RMS {lvl:.1f} dB "
          f"re direct (window {W[0]*1e6:.1f}-{W[1]*1e6:.1f} us)",
          flush=True)
    return lvl


def run_fdtd():
    sd = seeds()
    h = 4046.0 / F0 / 12.0
    n = int(round(L / h))
    m = 13
    ax_g = (np.arange(n) + 0.5) * h
    Z, Y, X = np.meshgrid(ax_g, ax_g, ax_g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    d2 = ((P[:, None, :] - sd[None, :, :]) ** 2).sum(-1)
    core = d2.argmin(1).astype(np.int16).reshape(n, n, n)
    N = n + 2 * m
    lab = np.full((N, N, N), 0, np.int16)
    lab[m:m+n, m:m+n, m:m+n] = core
    aA, aB = axes_sets()
    co = fdtd.optimised_coeffs(8)
    mats = (lab + 1).astype(np.uint8)
    Ct, rho_t = fdtd.material_tables(aA)
    dt = fdtd.safe_dt_labels(mats, Ct, rho_t, h, co, safety=0.5)
    nt = int(9e-6 / dt)
    wav = fdtd.ricker(F0, dt, nt)

    def cell(x):
        return tuple(int(m + q / h) for q in (x[2], x[1], x[0]))
    src, rec = cell(SRC), cell(REC)
    out = {}
    for tag, ax in (("A", aA), ("B", aB)):
        t0 = time.time()
        tr = np.asarray(fdtd.forward_fused_labels(
            lab, ax, h, dt, nt, [(src, 1.0)], wav, [([rec], [1.0])],
            order=8, coeffs=co, sponge_width=10), float).ravel()
        out[tag] = tr
        print(f"FDTD {tag}: {time.time()-t0:.0f}s", flush=True)
    return out["A"], out["B"], dt


def run_p2plus():
    nodes10, tets10, grain = get_mesh()
    nodes15, tets15 = pp.build_nodes(nodes10, tets10)
    aA, aB = axes_sets()
    out = {}
    for tag, ax in (("A", aA), ("B", aB)):
        ax13 = np.vstack([ax, ax[:1]])
        vol, D = pp.precompute(nodes15, tets15, grain, ax13)
        md = pp.lumped_mass(nodes15, tets15, vol)
        dt = pp.stable_dt(tets15, D, md)
        nt = int(T / dt)
        wav = fdtd.ricker(F0, dt, nt)
        t0 = time.time()
        out[tag] = (pp.forward(nodes15, tets15, grain, ax13, dt, nt, wav,
                               SRC, [REC], shell_frac=SHELL_FRAC,
                               shell_rate=SHELL_RATE, shell_pow=4,
                               src_type="monopole", rec_type="div",
                               D=D, vol=vol)[:, 0], dt)
        print(f"P2+ {tag}: {time.time()-t0:.0f}s ({nt} steps, "
              f"dt {dt*1e9:.2f} ns)", flush=True)
    fs = 5.0e8
    tg = np.arange(0.0, T, 1.0 / fs)
    za = np.interp(tg, np.arange(len(out["A"][0])) * out["A"][1],
                   out["A"][0])
    zb = np.interp(tg, np.arange(len(out["B"][0])) * out["B"][1],
                   out["B"][0])
    return za, zb, 1.0 / fs


def main():
    fa, fb, dtf = run_fdtd()
    lf = analyse(fa, fb, dtf, "FDTD (deep src/rec)")
    ea, eb, dte = run_p2plus()
    np.savez("fe_p2plus_ab4_traces.npz", fa=fa, fb=fb, dtf=dtf,
             ea=ea, eb=eb, dte=dte)
    le = analyse(ea, eb, dte, "P2+ (deep src/rec)")
    print(f"\nCROSS-CHECK 4: FDTD {lf:.1f} vs P2+ {le:.1f} dB -> "
          f"difference {lf-le:+.1f} dB "
          f"({'PASS' if abs(lf-le) < 3.0 else 'INVESTIGATE'} at 3 dB)",
          flush=True)


if __name__ == "__main__":
    main()
