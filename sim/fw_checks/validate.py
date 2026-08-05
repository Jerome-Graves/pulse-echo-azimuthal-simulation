"""Validate the ray forward model against a full-wave 3D elastic solve.

Everything in this project rests on `forward.py`, which is a ray model. Its
geometric predictions are safe regardless (the mirror degeneracy is algebra),
but its AMPLITUDE predictions are not, and the headline conclusion, that the
8-bit digitiser is the binding limit, depends entirely on how big a grain
boundary echo is relative to the rim echo.

The weakest assumption is the reflection coefficient. `ray_echoes` uses a
scalar fluid-like impedance step, R = (Z2-Z1)/(Z2+Z1), scaled by an obliquity
factor. The true problem at an interface between two misoriented anisotropic
crystals requires the full boundary conditions and produces P->P, P->SV and
P->SH. If conversion carries away significant energy, or if the anisotropic
P->P coefficient departs from the impedance form, the ray model has the wrong
coda amplitude and the ADC conclusion moves.

`two_grain_reflection` measures the true P->P coefficient directly by
differencing two full-wave runs, one with the interface and one homogeneous,
which cancels source spreading and the domain-edge arrivals. No absorbing
boundaries are available in the solver, so the record is windowed before the
far-face echo returns.
"""
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys
import time

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor"))))
from ringfwi import anisotropy as an, elastic3d as e3   # noqa: E402

import forward as F                                     # noqa: E402

RHO = an.ICE_RHO


def _axis(colat_deg, az_deg=0.0):
    t, a = np.radians(colat_deg), np.radians(az_deg)
    return np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])


def two_grain_reflection(colat_b_list=(0, 20, 40, 60, 90), colat_a=90.0,
                         f0=2.0e6, ppw=8.0, n_trans=28, x_iface_frac=0.42,
                         device="auto", verbose=True):
    """Full-wave P->P reflection coefficient at a single grain boundary.

    Propagation is along +x. Grain A occupies the first part of the domain,
    grain B the rest, and the interface normal is exactly along the ray, so
    the ray model's obliquity factor is 1 and only the impedance form is
    under test.
    """
    c_ref = 3850.0
    lam = c_ref / f0
    h = lam / ppw
    nx = int(np.ceil(6.0 * lam / h))
    nx = max(nx, 90)
    ny = nz = n_trans
    ix_if = int(nx * x_iface_frac)

    dt = 0.35 * h / 4100.0
    # far-face echo returns at 2*(nx-ix_src)*h/c; stop before it arrives
    ix_src = 3
    t_iface = 2.0 * (ix_if - ix_src) * h / c_ref
    t_far = 2.0 * (nx - 1 - ix_src) * h / c_ref
    nt = int(min(t_far * 0.92, t_iface * 2.4) / dt)

    _, p = F.gaussian_pulse(f0, 0.6, 1.0 / dt)
    wav = np.zeros(nt)                       # solver reads wavelet[n] for all n
    wav[:min(len(p), nt)] = p[:min(len(p), nt)]
    src = (nz // 2, ny // 2, ix_src)
    rec = [src]

    rows = []
    for cb in colat_b_list:
        na, nb = _axis(colat_a), _axis(cb)
        lab2 = np.zeros((nz, ny, nx), np.int32)
        lab2[:, :, ix_if:] = 1
        C2, rho2 = an.polycrystal_stiffness_3d(lab2, np.stack([na, nb]))
        C1, rho1 = an.polycrystal_stiffness_3d(np.zeros_like(lab2),
                                               np.stack([na, na]))
        kw = dict(source="explosive", record="pressure", device=device)
        d2, _ = e3.forward(C2, rho2, h, dt, nt, src, wav, rec, **kw)
        d1, _ = e3.forward(C1, rho1, h, dt, nt, src, wav, rec, **kw)
        d2, d1 = np.asarray(d2, float), np.asarray(d1, float)
        diff = (d2 - d1).ravel()          # isolates the interface reflection

        # reference: the same pulse reflected off a perfect mirror would be
        # the incident amplitude, so normalise by a total-reflection run
        w0 = max(int(t_iface / dt * 0.75), 1)
        w1 = min(int(t_iface / dt * 1.35), nt - 1)
        amp = np.abs(_env(diff[w0:w1])).max() if w1 > w0 + 4 else np.nan

        # ray-model prediction for the same pair
        d_hat = np.array([1.0, 0.0, 0.0])
        va = float(F.grain_speeds(na[None, :], d_hat)[0])
        vb = float(F.grain_speeds(nb[None, :], d_hat)[0])
        r_ray = (RHO * vb - RHO * va) / (RHO * vb + RHO * va)
        rows.append(dict(colat_b=cb, v_a=va, v_b=vb, r_ray=r_ray,
                         amp_fullwave=amp, t_iface_us=t_iface * 1e6))
        if verbose:
            print(f"   colat_B={cb:>3}  v {va:6.1f} -> {vb:6.1f} m/s   "
                  f"R_ray={r_ray:+.5f}   fullwave amp={amp:.4e}")

    # calibrate the arbitrary full-wave amplitude scale on the strongest pair,
    # then test whether the SHAPE of R versus misorientation matches
    rr = np.array([r["r_ray"] for r in rows])
    aa = np.array([r["amp_fullwave"] for r in rows])
    ok = np.isfinite(aa) & (np.abs(rr) > 1e-6)
    scale = float(np.sum(np.abs(rr[ok]) * aa[ok]) / np.sum(aa[ok] ** 2)) if ok.any() else np.nan
    for r, a in zip(rows, aa):
        r["r_fullwave"] = a * scale
        r["ratio"] = (a * scale) / r["r_ray"] if abs(r["r_ray"]) > 1e-6 else np.nan
    return dict(rows=rows, h=h, nx=nx, nt=nt, dt=dt, f0=f0,
                grid=(nz, ny, nx), scale=scale)


def _env(x):
    from scipy.signal import hilbert
    return np.abs(hilbert(np.asarray(x, float)))


def report(res):
    print(f"\n grid {res['grid']}, h={res['h']*1e3:.3f} mm, "
          f"{res['nt']} steps of {res['dt']*1e9:.1f} ns, f0={res['f0']/1e6:.1f} MHz")
    print(f"{'colat B':>8} {'R_ray':>10} {'R_fullwave':>12} {'ratio':>8}")
    for r in res["rows"]:
        print(f"{r['colat_b']:>8} {r['r_ray']:>+10.5f} {r['r_fullwave']:>+12.5f} "
              f"{r['ratio']:>8.2f}" if np.isfinite(r["ratio"])
              else f"{r['colat_b']:>8} {r['r_ray']:>+10.5f} {r['r_fullwave']:>+12.5f}"
                   f" {'-':>8}")
    rt = np.array([r["ratio"] for r in res["rows"] if np.isfinite(r["ratio"])])
    if len(rt):
        print(f"\n ratio spread {rt.min():.2f} to {rt.max():.2f} "
              f"(1.00 everywhere = the impedance form is exact)")


if __name__ == "__main__":
    t0 = time.time()
    res = two_grain_reflection()
    report(res)
    print(f"\n total {time.time()-t0:.0f}s")


def _iso_C(vp, vs, rho):
    """Isotropic stiffness dict. Here R = (Z2-Z1)/(Z2+Z1) is EXACT by classical
    theory, which makes this a non-circular calibration of the full-wave
    amplitude scale."""
    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2 * mu
    C = {f"C{i}{j}": 0.0 for i in range(1, 7) for j in range(i, 7)}
    for k in ("C11", "C22", "C33"):
        C[k] = lam + 2 * mu
    for k in ("C12", "C13", "C23"):
        C[k] = lam
    for k in ("C44", "C55", "C66"):
        C[k] = mu
    return C


def absolute_scale(f0=2.0e6, ppw=8.0, n_trans=28, x_iface_frac=0.42,
                   dv=(3850.0, 4300.0), device="auto"):
    """Calibrate the full-wave amplitude scale on a KNOWN isotropic step."""
    c_ref = 3850.0
    h = c_ref / f0 / ppw
    nx = max(int(np.ceil(6.0 * c_ref / f0 / h)), 90)
    ny = nz = n_trans
    ix_if, ix_src = int(nx * x_iface_frac), 3
    dt = 0.35 * h / 4600.0
    t_if = 2.0 * (ix_if - ix_src) * h / c_ref
    t_far = 2.0 * (nx - 1 - ix_src) * h / c_ref
    nt = int(min(t_far * 0.92, t_if * 2.4) / dt)
    _, p = F.gaussian_pulse(f0, 0.6, 1.0 / dt)
    wav = np.zeros(nt); wav[:min(len(p), nt)] = p[:min(len(p), nt)]
    src = (nz // 2, ny // 2, ix_src); rec = [src]

    v1, v2 = dv
    rho1 = rho2 = RHO
    r_exact = (rho2 * v2 - rho1 * v1) / (rho2 * v2 + rho1 * v1)

    Ca, Cb = _iso_C(v1, v1 / 2, RHO), _iso_C(v2, v2 / 2, RHO)
    shape = (nz, ny, nx)
    Cmix = {}
    for k in Ca:
        a = np.full(shape, float(Ca[k])); a[:, :, ix_if:] = float(Cb[k])
        Cmix[k] = a
    Chom = {k: np.full(shape, float(Ca[k])) for k in Ca}
    rho_a = np.full(shape, RHO)
    kw = dict(source="explosive", record="pressure", device=device)
    d2, _ = e3.forward(Cmix, rho_a, h, dt, nt, src, wav, rec, **kw)
    d1, _ = e3.forward(Chom, rho_a, h, dt, nt, src, wav, rec, **kw)
    diff = (np.asarray(d2, float) - np.asarray(d1, float)).ravel()
    w0, w1 = max(int(t_if / dt * 0.75), 1), min(int(t_if / dt * 1.35), nt - 1)
    amp = float(np.abs(_env(diff[w0:w1])).max())
    return dict(r_exact=r_exact, amp=amp, scale=abs(r_exact) / amp,
                v1=v1, v2=v2, h=h, nt=nt)


def full_wave_trace(build, f0=2.0e6, ppw=6.0, record_to=None, elem_d=6.35e-3,
                    c_out=1500.0, rho_out=300.0, margin_cells=8,
                    device="auto", verbose=True):
    """One complete A-scan from the full-wave solver, same specimen as the ray model.

    Uses the REAL disk geometry (100 x 35 mm, real grains) but at a lower
    centre frequency, because cost scales as (D/lambda)^4 once the time step
    is included. At 2 MHz the grains are still ~9 wavelengths across, so the
    specimen is in the same geometric scattering regime as the 5 MHz
    experiment; only the absolute resolution drops.

    SURROUND. The real specimen is ice, then a ~25 um oil film, then air. At
    2 MHz the wavelength is 1.93 mm, so the film is 1/77 of a wavelength and
    acoustically invisible: the rim behaves as ice against air, R ~ -1. The
    solver has no vacuum, and real air (c = 343 m/s) has a wavelength far
    below one cell so it cannot be resolved. Instead the surround is a
    fictitious LIGHT fluid: normal sound speed, density 10 kg/m^3. Impedance
    is rho*c, so dropping the density alone gives the ice/air reflection
    coefficient while keeping the wavelength resolvable.

    MARGIN. The disk must not touch the grid boundary. Inscribing it exactly
    puts the curved ice surface into the box edge as a sharp wedge and the
    run goes unstable: an earlier attempt with no margin decayed correctly to
    55 us then grew by a factor of 1400 by 110 us.
    """
    labels = build["labels"]
    axes = build["axes"]
    h_spec = build["h"]
    nx_s, ny_s, nz_s = labels.shape
    D = nx_s * h_spec
    c_ref = 3850.0
    lam = c_ref / f0
    h = lam / ppw

    # resample the label volume onto the solver grid, then reorder to (z,y,x)
    nd = int(round(D / h)); ndz = int(round(nz_s * h_spec / h))
    gx = np.clip((np.arange(nd) + 0.5) * h / h_spec, 0, nx_s - 1).astype(int)
    gz = np.clip((np.arange(ndz) + 0.5) * h / h_spec, 0, nz_s - 1).astype(int)
    core = labels[np.ix_(gx, gx, gz)].transpose(2, 1, 0).astype(np.int32)
    m = int(margin_cells)
    nz, nx = ndz + 2 * m, nd + 2 * m
    lab = np.full((nz, nx, nx), -1, np.int32)      # -1 = surround fluid
    lab[m:m + ndz, m:m + nd, m:m + nd] = core

    if verbose:
        print(f"  grid (z,y,x) = {lab.shape} = {lab.size/1e6:.2f}M cells, "
              f"h = {h*1e3:.3f} mm, lambda = {lam*1e3:.2f} mm")

    C, rho = an.polycrystal_stiffness_3d(lab, axes, c_couplant=c_out,
                                         rho_couplant=rho_out)
    dt = 0.35 * h / 4100.0
    record_to = record_to or (4.4 * D / c_ref)
    nt = int(record_to / dt)
    _, p = F.gaussian_pulse(f0, 0.6, 1.0 / dt)
    wav = np.zeros(nt); wav[:min(len(p), nt)] = p[:min(len(p), nt)]

    # probe on the rim at +x, mid height, spanning the element footprint
    cz, cy, cx = nz // 2, nx // 2, nx // 2
    Rc = nd // 2
    er = max(int(elem_d / 2 / h), 1)
    ix_probe = None
    for k in range(Rc - 1, 0, -1):                 # first ice cell inside the rim
        if lab[cz, cy, cx + k] >= 0:
            ix_probe = cx + k - 1
            break
    pts = []
    for dy in range(-er, er + 1):
        for dz in range(-er, er + 1):
            if dy * dy + dz * dz > er * er:
                continue
            iz, iy = cz + dz, cy + dy
            if 0 <= iz < nz and 0 <= iy < nx and lab[iz, iy, ix_probe] >= 0:
                pts.append((iz, iy, ix_probe))
    w = 1.0 / max(len(pts), 1)
    src_pts = [(p_, w) for p_ in pts]
    rec_groups = [(pts, np.full(len(pts), w))]

    if verbose:
        print(f"  {nt} steps of {dt*1e9:.1f} ns -> {record_to*1e6:.0f} us, "
              f"element = {len(pts)} cells")
    t0 = time.time()
    rec, _ = e3.forward(C, rho, h, dt, nt, pts[0], wav, [pts[0]],
                        source="explosive", record="pressure",
                        src_pts=src_pts, rec_groups=rec_groups, device=device)
    wall = time.time() - t0
    trace = np.asarray(rec, float).ravel()
    if verbose:
        print(f"  solved in {wall:.0f}s")
    return dict(t=np.arange(nt) * dt, trace=trace, h=h, dt=dt, nt=nt,
                f0=f0, grid=lab.shape, wall_s=wall, margin=m,
                r_rim=(rho_out * c_out - RHO * c_ref) / (rho_out * c_out + RHO * c_ref))


def full_wave_trace_fast(build, f0=2.0e6, ppw=4.0, record_to=None,
                         elem_d=6.35e-3, c_out=1500.0, rho_out=300.0,
                         sponge=10, order=4, fluid_damp=0.005, verbose=True):
    """Full-wave A-scan using the optimised solver in `fdtd`.

    Differences from `full_wave_trace`, all of which came out of debugging it:
      * fused CUDA kernels instead of cupy array ops (3.8-9x)
      * 4th-order optimised stencils, so ppw 4 replaces ppw 6 (~3x fewer cells)
      * a sponge boundary instead of a wide quiet margin, which both fixes the
        late-time blow-ups and shrinks the domain
      * a RICKER source. The old Gaussian-modulated cosine summed to +0.373,
        and since an explosive source accumulates (s += wavelet每 step) that
        left a permanent static stress on the trace. Envelope picking then
        tracked the pedestal rather than the echoes, which is why E1 was
        invisible in the first attempts.
    """
    import time
    import fdtd

    labels, axes, h_spec = build["labels"], build["axes"], build["h"]
    nx_s, ny_s, nz_s = labels.shape
    D = nx_s * h_spec
    c_ref = 3850.0
    lam = c_ref / f0
    h = lam / ppw

    nd = int(round(D / h)); ndz = int(round(nz_s * h_spec / h))
    gx = np.clip((np.arange(nd) + 0.5) * h / h_spec, 0, nx_s - 1).astype(int)
    gz = np.clip((np.arange(ndz) + 0.5) * h / h_spec, 0, nz_s - 1).astype(int)
    core = labels[np.ix_(gx, gx, gz)].transpose(2, 1, 0).astype(np.int32)
    m = int(sponge) + 2
    nz, nx = ndz + 2 * m, nd + 2 * m
    lab = np.full((nz, nx, nx), -1, np.int32)
    lab[m:m + ndz, m:m + nd, m:m + nd] = core

    C, rho = an.polycrystal_stiffness_3d(lab, axes, c_couplant=c_out,
                                         rho_couplant=rho_out)
    coeffs = fdtd.optimised_coeffs(order)
    # dt from the speed the DISCRETE scheme supports, not the fastest physical
    # wave: the stress update uses cell-centre stiffness while the velocity
    # update divides by a face-averaged density, so at the rim ice stiffness
    # pairs with a lighter density and the effective speed exceeds 4046 m/s.
    dt = fdtd.safe_dt(C, rho, h, coeffs, safety=0.5)

    # Damp the surround. The fluid exists only to give the rim its reflection
    # coefficient; nothing there is measured. Without this the solid/fluid
    # interface supports a slowly growing mode (a fraction of a percent per
    # step) that is invisible over hundreds of steps and destroys the run over
    # tens of thousands. Measured over 20000 steps: growth 1.6e4 undamped,
    # 1.00 at 0.01/cell, while changing the ice trace by ~1%.
    from scipy import ndimage
    dmask = None
    if fluid_damp > 0:
        dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
        dmask = np.exp(-fluid_damp * dist).astype(np.float32)
    record_to = record_to or (4.6 * D / c_ref)
    nt = int(record_to / dt)
    wav = fdtd.ricker(f0, dt, nt)

    cz, cy, cx = nz // 2, nx // 2, nx // 2
    Rc = nd // 2
    ix_probe = None
    for k in range(Rc - 1, 0, -1):
        if lab[cz, cy, cx + k] >= 0:
            ix_probe = cx + k - 1
            break
    er = max(int(elem_d / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ix_probe)
           for dy in range(-er, er + 1) for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er
           and 0 <= cz + dz < nz and 0 <= cy + dy < nx
           and lab[cz + dz, cy + dy, ix_probe] >= 0]
    w = 1.0 / max(len(pts), 1)

    if verbose:
        print(f"  grid {lab.shape} = {lab.size/1e6:.2f}M cells, h={h*1e3:.3f} mm, "
              f"ppw={ppw}, order={order}")
        print(f"  {nt} steps of {dt*1e9:.1f} ns -> {record_to*1e6:.0f} us, "
              f"element {len(pts)} cells, sponge {sponge}")
    t0 = time.time()
    rec = fdtd.forward_fused(C, rho, h, dt, nt, [(p, w) for p in pts], wav,
                             [(pts, np.full(len(pts), w))], order=order,
                             coeffs=coeffs, sponge_width=sponge,
                             damp_mask=dmask)
    wall = time.time() - t0
    if verbose:
        print(f"  solved in {wall:.0f}s "
              f"({lab.size*nt/wall/1e9:.2f} G cell-updates/s)")
    return dict(t=np.arange(nt) * dt, trace=np.asarray(rec, float).ravel(),
                h=h, dt=dt, nt=nt, f0=f0, grid=lab.shape, wall_s=wall,
                cells=lab.size,
                r_rim=(rho_out * c_out - RHO * c_ref) / (rho_out * c_out + RHO * c_ref))
