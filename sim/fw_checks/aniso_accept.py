import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
"""Conclusive anisotropy test: separate GRID anisotropy from MATERIAL anisotropy.

The previous attempt left a 59 ns residual against a 61 ns group-vs-phase
signal, so it could not say which velocity the solver actually follows. The
worst point was 45 degrees, which is exactly the grid diagonal, and that is
the tell: a cubic FDTD grid has its OWN directional error. Numerical
dispersion along a face normal differs from dispersion along a body diagonal
even in a perfectly isotropic medium, so part of the "anisotropy" being
measured was the mesh, not the ice.

Method:
  1. run an ISOTROPIC medium through the identical geometry. Every arrival
     should be identical; whatever angular variation appears is pure grid
     artefact.
  2. run anisotropic ice through the same geometry.
  3. subtract. What survives is material anisotropy alone, and it can be
     scored against group and phase velocity without the mesh confounding it.

Delays are measured by CROSS-CORRELATION against the 0-degree trace rather
than by envelope peak. The constant pick bias (132 ns previously, the Ricker
envelope peak lagging the onset) cancels exactly in a relative measurement,
and correlation resolves well below one sample.
"""
import numpy as np


def iso_stiffness(vp, vs, rho, shape):
    """Isotropic stiffness dict matching the solver's Voigt convention."""
    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2 * mu
    C = {f"C{i}{j}": np.zeros(shape) for i in range(1, 7) for j in range(i, 7)}
    for k in ("C11", "C22", "C33"):
        C[k] = np.full(shape, lam + 2 * mu)
    for k in ("C12", "C13", "C23"):
        C[k] = np.full(shape, lam)
    for k in ("C44", "C55", "C66"):
        C[k] = np.full(shape, mu)
    return C


def xcorr_delay(a, b, dt, max_lag=None):
    """Delay of `a` relative to `b`, sub-sample, by parabolic peak fit.

    Both traces are mean-removed and energy-normalised first so a difference
    in amplitude cannot bias the lag.
    """
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    a = a / (np.linalg.norm(a) + 1e-30)
    b = b / (np.linalg.norm(b) + 1e-30)
    c = np.correlate(a, b, mode="full")
    lags = np.arange(-len(b) + 1, len(a))
    if max_lag is not None:
        m = np.abs(lags) <= max_lag
        c, lags = c[m], lags[m]
    k = int(np.argmax(c))
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        d = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / d if abs(d) > 1e-30 else 0.0
    else:
        frac = 0.0
    return (lags[k] + frac) * dt


def run(angles=(0, 20, 30, 45, 60, 75, 90), n=200, r_cells=80, h=4.81e-4,
        f0=2.0e6, order=4, safety=0.5, sponge=16, verbose=True):
    import os
    import sys
    sys.path.insert(0, os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "vendor")))
    from ringfwi import anisotropy as an
    import fdtd
    import aniso_check as AC

    nz = ny = nx = n
    src = (n // 2, n // 2, n // 2)
    lab = np.zeros((nz, ny, nx), np.int32)

    recs, offs = [], []
    for th in angles:
        a = np.radians(th)
        dz = int(round(r_cells * np.cos(a)))
        dx = int(round(r_cells * np.sin(a)))
        recs.append((src[0] + dz, src[1], src[2] + dx))
        offs.append((dz, dx))

    Ca, rho = an.polycrystal_stiffness_3d(lab, np.array([[0.0, 0.0, 1.0]]))
    # isotropic control at the basal-plane speed, where ice group == phase
    Ci = iso_stiffness(3897.5, 1950.0, 917.0, lab.shape)

    coeffs = fdtd.optimised_coeffs(order)
    dt = fdtd.safe_dt(Ca, rho, h, coeffs, safety=safety)
    nt = int(2.6 * r_cells * h / 3700.0 / dt)
    wav = fdtd.ricker(f0, dt, nt)
    rg = [([q], np.array([1.0])) for q in recs]

    out = {}
    for tag, CC in (("iso", Ci), ("aniso", Ca)):
        if verbose:
            print(f"  running {tag} ... ", end="", flush=True)
        r = fdtd.forward_fused(CC, rho, h, dt, nt, [(src, 1.0)], wav, rg,
                               order=order, coeffs=coeffs, sponge_width=sponge)
        out[tag] = np.asarray(r, float)
        if verbose:
            print("done", flush=True)

    Ct = AC.ice_voigt((0, 0, 1))
    rows = []
    v_iso = 3897.5
    win = int(1.8e-6 / dt)                 # window half-width around the P arrival
    for i, th in enumerate(angles):
        dz, dx = offs[i]
        R = np.hypot(dz, dx) * h
        # Compare the SAME receiver in the two media. Grid direction, path
        # length, sponge proximity and radiation pattern are then identical,
        # so they cancel exactly and only the material anisotropy survives.
        # (Referencing to the 0-degree trace instead left 627 ns of apparent
        # "grid anisotropy" where geometry predicts about 70, because the
        # receivers differ in every one of those respects.)
        k0 = int((R / v_iso) / dt)
        lo, hi = max(k0 - win, 0), min(k0 + win, out["iso"].shape[0])
        d_mat = xcorr_delay(out["aniso"][lo:hi, i], out["iso"][lo:hi, i], dt,
                            max_lag=int(1.2e-6 / dt))

        rv = np.array([dx * h, 0.0, dz * h])          # (x, y, z), c-axis = z
        tg, _ = AC.expected_arrival(Ct, 917.0, rv, 0)
        a = np.radians(th)
        vp = AC.phase_velocities(Ct, 917.0,
                                 np.array([np.sin(a), 0.0, np.cos(a)]))[0][0]
        tp = R / vp
        rows.append(dict(angle=th, R=R, d_mat=d_mat, t_group=tg, t_phase=tp,
                         pred_group=tg - R / v_iso,
                         pred_phase=tp - R / v_iso))

    return dict(rows=rows, dt=dt, nt=nt, n=n, r_cells=r_cells, h=h)


def report(res):
    rows = res["rows"]
    print()
    print(f" grid {res['n']}^3, dt={res['dt']*1e9:.2f} ns, {res['nt']} steps, "
          f"radius {res['r_cells']*res['h']*1e3:.1f} mm")
    print(" aniso(theta) vs iso(theta) at the SAME receiver")
    print()
    print(f"{'ang':>5} {'measured':>11} {'pred GROUP':>11} {'pred PHASE':>11} "
          f"{'err GRP':>9} {'err PHS':>9}")
    for r in rows:
        print(f"{r['angle']:>5} {r['d_mat']*1e9:>+10.1f}n "
              f"{r['pred_group']*1e9:>+10.1f}n {r['pred_phase']*1e9:>+10.1f}n "
              f"{(r['d_mat']-r['pred_group'])*1e9:>+8.1f}n "
              f"{(r['d_mat']-r['pred_phase'])*1e9:>+8.1f}n")
    m = np.array([r["d_mat"] for r in rows])
    g = np.array([r["pred_group"] for r in rows])
    p = np.array([r["pred_phase"] for r in rows])
    rg, rp = np.std(m - g) * 1e9, np.std(m - p) * 1e9
    sep = np.std(g - p) * 1e9
    sig = (m.max() - m.min()) * 1e9
    print()
    print(f" signal being measured : {sig:6.1f} ns")
    print(f" RMS residual vs GROUP : {rg:6.1f} ns")
    print(f" RMS residual vs PHASE : {rp:6.1f} ns")
    print(f" group-phase separation: {sep:6.1f} ns")
    print()
    if sep > 2 * min(rg, rp):
        w = "GROUP" if rg < rp else "PHASE"
        print(f" CONCLUSIVE: the solver follows {w} velocity.")
    else:
        print(f" Anisotropy reproduced to {rg:.0f} ns on a {sig:.0f} ns signal,")
        print(f" but group and phase differ by only {sep:.0f} ns here, so this")
        print(" test cannot separate them. Longer propagation would.")
    return dict(rms_group=rg, rms_phase=rp, sep=sep, signal=sig)
