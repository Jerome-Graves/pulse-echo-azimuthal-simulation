"""Decisive probes for the TET10 solver's 19%-slow, dt-dependent speed.

Three probes, in order of cheapness, each isolating one layer:
  1. PATCH (f64, post-straighten): linear field u = A.x on the gmsh test
     mesh -> Gauss-point strains must equal sym(A) exactly and interior
     nodal forces must vanish. Tests B matrices + assembly, no dynamics.
  2. MASS: md.sum() vs rho*V_total (should be exact by construction of
     the normalised HRZ weights - printed to close the question).
  3. SPEED, structured mesh, TWO RECEIVERS: a hand-built Kuhn-triangulated
     bar with EXACT edge midpoints (no gmsh), c-axis along x so the
     analytic qP speed is sqrt(C33/rho) with zero ambiguity. Speed is
     measured receiver-to-receiver by windowed cross-correlation, which
     common-modes away the source onset, the ricker delay convention and
     any threshold choice. shell_damp=0 removes the absorber confound.
     The SAME corner mesh runs through the P1 solver as a control, and
     the P2 run repeats at dt/2 to re-test the dt-dependence with an
     honest measurement.
"""
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys
import time

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))

import fdtd                                      # noqa: E402
import fe_solver_p1 as fe_solver                                 # noqa: E402
import fe_solver_p2 as p2                        # noqa: E402

RHO = 917.0
EDGES = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]   # gmsh tet10


# ---------------------------------------------------------------- probe 1
def precompute_f64(nodes, tets10):
    """f64 twin of p2.precompute_p2 so the patch residual is not hidden
    under float32 noise."""
    p = nodes[tets10[:, :4]]
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    c = p[:, 3] - p[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    J = np.stack([a, b, c], axis=2)
    Jin = np.linalg.inv(J)
    m = len(tets10)
    dLdx = np.zeros((m, 4, 3))
    dLdx[:, 1:, :] = Jin
    dLdx[:, 0, :] = -Jin.sum(axis=1)
    Bg = np.zeros((m, 4, 6, 30))
    for gp in range(4):
        g_b = p2._shape_grads_bary(p2._GPTS[gp])
        gN = np.einsum("na,mak->mnk", g_b, dLdx)
        for k in range(10):
            gx, gy, gz = gN[:, k, 0], gN[:, k, 1], gN[:, k, 2]
            Bg[:, gp, 0, 3*k+0] = gx
            Bg[:, gp, 1, 3*k+1] = gy
            Bg[:, gp, 2, 3*k+2] = gz
            Bg[:, gp, 3, 3*k+1] = gz; Bg[:, gp, 3, 3*k+2] = gy
            Bg[:, gp, 4, 3*k+0] = gz; Bg[:, gp, 4, 3*k+2] = gx
            Bg[:, gp, 5, 3*k+0] = gy; Bg[:, gp, 5, 3*k+1] = gx
    return vol, Bg


def patch_test(nodes, tets10, C6):
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3)) * 1e-6
    eps = np.array([A[0, 0], A[1, 1], A[2, 2],
                    A[1, 2] + A[2, 1], A[0, 2] + A[2, 0], A[0, 1] + A[1, 0]])
    vol, Bg = precompute_f64(nodes, tets10)
    u = nodes @ A.T                                        # (n, 3)
    ue = u[tets10].reshape(-1, 30)
    eg = np.einsum("mgik,mk->mgi", Bg, ue)                 # (m, 4, 6)
    strain_err = np.abs(eg - eps).max() / np.abs(eps).max()

    sig = C6 @ eps                                         # uniform material
    n = len(nodes)
    f = np.zeros((n, 3))
    gw = vol[:, None] * 0.25
    for gp in range(4):
        fe = -np.einsum("mjk,j->mk", Bg[:, gp], sig) * gw
        np.add.at(f, tets10.ravel(),
                  fe.reshape(-1, 10, 3).reshape(-1, 3))
    lo, hi = nodes.min(0), nodes.max(0)
    margin = 1e-9
    interior = ((nodes > lo + margin) & (nodes < hi - margin)).all(axis=1)
    scale = np.abs(sig).max() * (vol.mean() ** (2 / 3))    # ~traction*area
    res = np.abs(f[interior]).max() / scale
    print(f"PATCH: gp-strain rel err {strain_err:.2e}, "
          f"interior force residual {res:.2e} (re traction*area) "
          f"-> {'PASS' if strain_err < 1e-9 and res < 1e-8 else 'FAIL'}",
          flush=True)
    return strain_err, res


# ---------------------------------------------------------------- probe 3
def kuhn_bar(nx, ny, nz, h):
    """Structured Kuhn (6-tet) bar mesh; returns P1 (nodes4, tets4) and
    TET10 (nodes10, tets10) with EXACT midpoints, gmsh edge order."""
    gx = np.arange(nx + 1) * h
    gy = np.arange(ny + 1) * h
    gz = np.arange(nz + 1) * h
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    nodes4 = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    import itertools
    tets = []
    steps = np.eye(3, dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c0 = np.array([i, j, k])
                for perm in itertools.permutations(range(3)):
                    v = [c0]
                    for p_ in perm:
                        v.append(v[-1] + steps[p_])
                    tets.append([nid(*q) for q in v])
    tets4 = np.asarray(tets, np.int64)

    mid = {}
    nextid = len(nodes4)
    tets10 = np.zeros((len(tets4), 10), np.int64)
    tets10[:, :4] = tets4
    midpts = []
    for e, (a, b) in enumerate(EDGES):
        for t in range(len(tets4)):
            key = (min(tets4[t, a], tets4[t, b]),
                   max(tets4[t, a], tets4[t, b]))
            if key not in mid:
                mid[key] = nextid
                midpts.append(0.5 * (nodes4[key[0]] + nodes4[key[1]]))
                nextid += 1
            tets10[t, 4 + e] = mid[key]
    nodes10 = np.vstack([nodes4, np.asarray(midpts)])
    return nodes4, tets4, nodes10, tets10


def mass_check(nodes10, tets10):
    p = nodes10[tets10[:, :4]]
    a = p[:, 1] - p[:, 0]; b = p[:, 2] - p[:, 0]; c = p[:, 3] - p[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    ref = p2._ref_mass_diag()
    w10 = ref / ref.sum()
    md = np.zeros(len(nodes10))
    np.add.at(md, tets10.ravel(),
              (RHO * vol[:, None] * w10[None, :]).ravel())
    print(f"MASS: sum(md)={md.sum():.6e} vs rho*V={RHO*vol.sum():.6e} "
          f"(rel {(md.sum()/(RHO*vol.sum())-1):+.2e}); "
          f"HRZ weights corner {w10[:4].mean():.4f} edge {w10[4:].mean():.4f}",
          flush=True)


def xcorr_dt(tr1, tr2, dt, t1_exp, t2_exp, half=1.2e-6):
    """Sub-sample lag between windowed arrivals via parabolic peak."""
    fs = 1.0 / dt
    def win(tr, tc):
        a = max(int((tc - half) * fs), 0)
        b = min(int((tc + half) * fs), len(tr))
        w = np.zeros_like(tr)
        w[a:b] = tr[a:b] * np.hanning(b - a)
        return w
    w1, w2 = win(tr1, t1_exp), win(tr2, t2_exp)
    c = np.correlate(w2, w1, "full")
    k = int(np.argmax(c))
    if 0 < k < len(c) - 1:                       # parabolic interpolation
        d = 0.5 * (c[k - 1] - c[k + 1]) / (c[k - 1] - 2 * c[k] + c[k + 1])
    else:
        d = 0.0
    return (k + d - (len(tr1) - 1)) * dt


def speed_test():
    h = 0.5e-3
    nx, ny, nz = 80, 24, 24                       # 40 x 12 x 12 mm bar
    nodes4, tets4, nodes10, tets10 = kuhn_bar(nx, ny, nz, h)
    print(f"structured bar: {len(tets4)} tets, "
          f"{len(nodes10)} P2 nodes", flush=True)
    mass_check(nodes10, tets10)

    axes = np.array([[1.0, 0.0, 0.0]])            # c-axis along x
    grain = np.zeros(len(tets4), np.int32)
    Ct, _ = fdtd.material_tables(axes)
    C6 = Ct.reshape(-1, 6, 6)[1].astype(float)
    v_exp = np.sqrt(C6[0, 0] / RHO)               # qP along c, exact
    print(f"analytic qP along c-axis: {v_exp:.1f} m/s", flush=True)

    F0 = 1.0e6
    src = np.array([8e-3, 6e-3, 6e-3])
    r1 = np.array([16e-3, 6e-3, 6e-3])
    r2 = np.array([28e-3, 6e-3, 6e-3])
    t0w = 1.2 / F0
    t1e = np.linalg.norm(r1 - src) / v_exp + t0w
    t2e = np.linalg.norm(r2 - src) / v_exp + t0w
    T = 8.0e-6
    gap = np.linalg.norm(r2 - r1)

    def measure(rec, dt, tag):
        if not np.isfinite(rec).all() or np.abs(rec).max() > 1e6:
            print(f"{tag}: UNSTABLE (max {np.abs(rec).max():.1e})",
                  flush=True)
            return None
        lag = xcorr_dt(rec[:, 0], rec[:, 1], dt, t1e, t2e)
        v = gap / lag
        print(f"{tag}: lag {lag*1e6:.4f} us -> v = {v:.1f} m/s "
              f"({(v/v_exp-1)*100:+.2f}% vs analytic)", flush=True)
        return v

    # P1 control on the identical corner mesh
    dt1 = fe_solver.stable_dt(nodes4, tets4, v_exp)
    nt1 = int(T / dt1)
    wav1 = fdtd.ricker(F0, dt1, nt1)
    t0 = time.time()
    rec = fe_solver.forward_fe(nodes4, tets4, grain, axes, dt1, nt1,
                               0, wav1, None, shell_damp=0.0,
                               src_xyz=src, rec_xyz=[r1, r2])
    print(f"P1 run: {time.time()-t0:.0f}s ({nt1} steps)", flush=True)
    measure(rec, dt1, "P1  (h=0.5mm, lam/8 cells)")

    # P2: HRZ lumped vs k Jacobi consistent-mass iterations, + dt/2 check
    dt2 = p2.stable_dt_p2(nodes10, tets10, v_exp)
    for fac, k in ((1, 0), (1, 1), (1, 2), (1, 3), (2, 3)):
        dt = dt2 / fac
        nt = int(T / dt)
        wav = fdtd.ricker(F0, dt, nt)
        t0 = time.time()
        rec = p2.forward_fe_p2(nodes10, tets10, grain, axes, dt, nt, wav,
                               src, [r1, r2], shell_damp=0.0, mass_iter=k)
        tag = f"P2 k={k} dt/{fac}"
        print(f"{tag} run: {time.time()-t0:.0f}s ({nt} steps)", flush=True)
        measure(rec, dt, tag)


def main():
    import os
    if os.path.exists("fe_p2_test.npz"):
        d = np.load("fe_p2_test.npz")
        nodes, tets10 = d["nodes"], d["tets"]
        nodes = p2.straighten(nodes, tets10)
        axes = np.array([[1.0, 0.0, 0.0]])
        Ct, _ = fdtd.material_tables(axes)
        C6 = Ct.reshape(-1, 6, 6)[1].astype(float)
        print(f"gmsh mesh: {len(tets10)} tets (straightened)", flush=True)
        patch_test(nodes, tets10, C6)
    else:
        print("no fe_p2_test.npz - skipping gmsh patch test", flush=True)

    # mass closure (exact by construction; printed to close the question)
    speed_test()


if __name__ == "__main__":
    main()
