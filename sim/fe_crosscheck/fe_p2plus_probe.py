import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
"""ML2n15 validation ladder, run in one logged pass:
  1. element identities: cond(Vandermonde), partition of unity, delta
     property, weight closure (must all be exact/machine).
  2. f64 patch test on the straightened gmsh TET10 mesh + built face/
     interior nodes: linear field -> interior forces vanish.
  3. structured-bar two-receiver dispersion vs exact analytic qP:
     paper predicts e = 1.89*nE^-4 = 0.046% at lambda/8 (HRZ-TET10
     measured -3.11% on the same bar, P1 -2.48%). Also at dt/2 to
     confirm dt-independence. Shell OFF (rate=0).
"""
import time

import numpy as np

import fdtd                                     # noqa: E402
import fe_solver_p2plus as pp                   # noqa: E402
from fe_p2_probe import kuhn_bar, xcorr_dt      # noqa: E402

RHO = 917.0


def element_checks():
    print(f"cond(Vandermonde) = {pp._CONDV:.2f}", flush=True)
    rng = np.random.default_rng(1)
    pts = rng.dirichlet(np.ones(4), 200)
    S = np.array([pp.shape_at(L) for L in pts])
    print(f"partition of unity: max|sum N - 1| = "
          f"{np.abs(S.sum(1) - 1).max():.2e}", flush=True)
    Dm = np.array([pp.shape_at(L) for L in pp._BARY])
    print(f"delta property: max|N(nodes) - I| = "
          f"{np.abs(Dm - np.eye(15)).max():.2e}", flush=True)
    print(f"weight closure: sum(W15) - 1 = {pp.W15.sum() - 1.0:+.2e}",
          flush=True)


def patch_test():
    import os
    import fe_solver_p2 as p2
    if not os.path.exists("fe_p2_test.npz"):
        print("no fe_p2_test.npz - skipping patch", flush=True)
        return
    d = np.load("fe_p2_test.npz")
    nodes10, tets10 = p2.straighten(d["nodes"], d["tets"]), d["tets"]
    nodes15, tets15 = pp.build_nodes(nodes10, tets10)
    axes = np.array([[1.0, 0.0, 0.0]])
    grain = np.zeros(len(tets15), np.int32)
    vol, D = pp.precompute(nodes15, tets15, grain, axes)
    # f64 D for the patch (precompute casts to f32; redo the contraction)
    p = nodes15[tets15[:, :4]]
    a = p[:, 1] - p[:, 0]; b = p[:, 2] - p[:, 0]; c = p[:, 3] - p[:, 0]
    v64 = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    Jin = np.linalg.inv(np.stack([a, b, c], axis=2))
    Ct, _ = fdtd.material_tables(axes)
    C6 = Ct.reshape(-1, 6, 6)[grain + 1]
    C4 = pp._voigt_to_tensor(C6)
    D64 = np.einsum("m,maj,mijkl,mbl->mabki", v64, Jin, C4, Jin,
                    optimize=True).reshape(-1, 27, 3)
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3)) * 1e-6
    u = nodes15 @ A.T
    f = pp._apply_K_np(tets15, D64, u)
    lo, hi = nodes15.min(0), nodes15.max(0)
    interior = ((nodes15 > lo + 1e-9) & (nodes15 < hi - 1e-9)).all(axis=1)
    eps = np.array([A[0, 0], A[1, 1], A[2, 2],
                    A[1, 2] + A[2, 1], A[0, 2] + A[2, 0], A[0, 1] + A[1, 0]])
    sig = C6[0] @ eps
    scale = np.abs(sig).max() * (v64.mean() ** (2 / 3))
    res = np.abs(f[interior]).max() / scale
    print(f"PATCH ({len(tets15)} tets, {len(nodes15)} nodes): interior "
          f"residual {res:.2e} (re traction*area) -> "
          f"{'PASS' if res < 1e-8 else 'FAIL'}", flush=True)


def speed_test():
    h = 0.5e-3
    nodes4, tets4, nodes10, tets10 = kuhn_bar(80, 24, 24, h)
    nodes15, tets15 = pp.build_nodes(nodes10, tets10)
    print(f"bar: {len(tets15)} tets, {len(nodes15)} nodes (ML2n15)",
          flush=True)
    axes = np.array([[1.0, 0.0, 0.0]])
    grain = np.zeros(len(tets15), np.int32)
    vol, D = pp.precompute(nodes15, tets15, grain, axes)
    md = pp.lumped_mass(nodes15, tets15, vol)
    print(f"mass closure: {md.sum():.6e} vs "
          f"{RHO*vol.sum():.6e}", flush=True)
    Ct, _ = fdtd.material_tables(axes)
    v_exp = np.sqrt(Ct.reshape(-1, 6, 6)[1][0, 0] / RHO)
    print(f"analytic qP along c: {v_exp:.1f} m/s", flush=True)

    t0 = time.time()
    dt0 = pp.stable_dt(tets15, D, md)
    print(f"power-iteration dt = {dt0*1e9:.2f} ns "
          f"({time.time()-t0:.0f}s)", flush=True)

    F0 = 1.0e6
    src = np.array([8e-3, 6e-3, 6e-3])
    r1 = np.array([16e-3, 6e-3, 6e-3])
    r2 = np.array([28e-3, 6e-3, 6e-3])
    t0w = 1.2 / F0
    t1e = np.linalg.norm(r1 - src) / v_exp + t0w
    t2e = np.linalg.norm(r2 - src) / v_exp + t0w
    gap = np.linalg.norm(r2 - r1)
    for fac in (1, 2):
        dt = dt0 / fac
        nt = int(8.0e-6 / dt)
        wav = fdtd.ricker(F0, dt, nt)
        t0 = time.time()
        rec = pp.forward(nodes15, tets15, grain, axes, dt, nt, wav,
                         src, [r1, r2], shell_rate=0.0, D=D, vol=vol)
        print(f"run dt/{fac}: {time.time()-t0:.0f}s ({nt} steps)",
              flush=True)
        if not np.isfinite(rec).all() or np.abs(rec).max() > 1e6:
            print(f"dt/{fac}: UNSTABLE (max {np.abs(rec).max():.1e})",
                  flush=True)
            continue
        lag = xcorr_dt(rec[:, 0], rec[:, 1], dt, t1e, t2e)
        v = gap / lag
        print(f"ML2n15 dt/{fac}: lag {lag*1e6:.4f} us -> v = {v:.1f} m/s "
              f"({(v/v_exp-1)*100:+.3f}% vs analytic; paper predicts "
              f"~0.05% at lambda/8; HRZ-TET10 was -3.11%)", flush=True)


def main():
    element_checks()
    patch_test()
    speed_test()


if __name__ == "__main__":
    main()
