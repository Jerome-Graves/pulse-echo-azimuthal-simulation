"""Is the coda physical scattering, or the staircased mesh?

Polarity inversion cannot answer this: it returned EXACTLY zero residual, so
the scheme is perfectly linear, and physical scattering, numerical dispersion
and staircase scattering are all linear operations that flip sign together.
It measures the roundoff floor (nil here) and nothing else.

Resolution convergence cannot answer it cleanly either, because refining the
grid changes the artefact AND re-samples the geometry at the same time. A
monotonic trend over three points is suggestive, not proof.

This test separates them. Take the IDENTICAL microstructure and re-sample it
rotated by some angle about the disc axis, then put the probe at the matching
rotated azimuth so the physical experiment is unchanged:

  * PHYSICAL scattering is invariant. Same grains, same c-axes, same relative
    geometry, described in rotated coordinates.
  * STAIRCASE artefact is not. Every boundary is stepped differently by the
    mesh, so any mesh-induced scattering changes character.

Comparing coda level and waveform across rotations therefore measures the
artefact directly instead of inferring it.

⚠ SUPERSEDED AS A STAIRCASE VALIDATION (2026-07-29). The code never
implemented the premise above: the probe stays on the +x rim while the
specimen rotates, so each rotation interrogates a DIFFERENT diameter
(different grain chain, different beam-to-fabric angle) - an azimuth
sweep, not a mesh-invariance test. Measured at production settings
(order 8, ppw 6, rigid rotated_grid): peak spread 13.7 dB across
rotations {0,17,31,47} = the physical azimuthal fabric pattern + grain
glint, exactly as an azimuth sweep should show. The historical "~2 dB
invariance" at order 4 / ppw 4 was the coarse grid's staircase noise
floor flattening the physics - it validated nothing. Staircase
validity rests on the FE arbiter instead (fe_p2plus_ab5: conforming
mesh agrees to 2.9 dB, attenuation to 0.03 dB), which is stronger
evidence than this test could ever give. `rotated_grid` itself remains
production machinery for sweeps. Same-day ppw record (order 8, rot 0,
60 mm specimen): ppw6 -> ppw8 coda deltas peak -1.30 / rms -1.97 dB,
so production coda LEVELS carry ~2 dB discretisation allowance (fit
impact second-order: absolute level is a free-gain nuisance).
"""
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor"))))
from ringfwi import anisotropy as an          # noqa: E402
from scipy import ndimage                     # noqa: E402
from scipy.signal import hilbert              # noqa: E402

import fdtd                                    # noqa: E402
from specimen import DiskSpecimen              # noqa: E402


def rotated_grid(build, h, sponge, rot_deg, single_raster=False):
    """Resample the specimen onto the solver grid, rotated about the disc axis.

    The c-axes are rotated by the same angle, so the physical microstructure
    is untouched: only its description relative to the mesh changes.

    FRAME (fixed 2026-07-29): the resample reads specimen point R(+rot)*x
    at solver point x, i.e. the GEOMETRY appears rotated by -rot in the
    solver frame, so the axes must also be rotated by -rot (axes @ R).
    Pre-fix code rotated the axes by +rot (axes @ R.T): each azimuth was
    then a genuinely different specimen (axes spun +2*rot relative to
    their own grains), not a rigid rotation. Sweeps generated before the
    fix are tagged legacy via the absence of axes_convention='rigid2' in
    their config.json and keep the old reporting convention in fit_sweep.
    """
    labels, axes, h_spec = build["labels"], build["axes"], build["h"]
    nx_s, _, nz_s = labels.shape
    D = nx_s * h_spec
    nd = int(round(D / h))
    ndz = int(round(nz_s * h_spec / h))

    a = np.radians(rot_deg)
    ca, sa = np.cos(a), np.sin(a)
    u = (np.arange(nd) + 0.5) * h - D / 2.0
    X, Y = np.meshgrid(u, u, indexing="ij")
    Xr = ca * X - sa * Y
    Yr = sa * X + ca * Y
    gi = np.rint(Xr / h_spec + (nx_s - 1) / 2.0).astype(int)
    gj = np.rint(Yr / h_spec + (nx_s - 1) / 2.0).astype(int)
    ok = (gi >= 0) & (gi < nx_s) & (gj >= 0) & (gj < nx_s)
    gi, gj = np.clip(gi, 0, nx_s - 1), np.clip(gj, 0, nx_s - 1)
    gz = np.clip((np.arange(ndz) + 0.5) * h / h_spec, 0, nz_s - 1).astype(int)

    if single_raster:
        # SINGLE rasterisation: evaluate the Laguerre tessellation
        # directly at the rotated solver points instead of nearest-
        # neighbour resampling an ALREADY rasterised label volume.
        # The default path rasterises twice (once in specimen.build at
        # spacing h_spec, again by the np.rint above), which aliases
        # every grain boundary a second time and is a prime suspect for
        # the beam-referenced azimuthal artefact. Needs build['weights']
        # (added 2026-07-31); falls back if an older build dict is
        # passed in.
        # Implemented by rotating the SEEDS rather than the query
        # points, which is identical (the power metric is rigid:
        # argmin_s |R(a)x - s|^2 - w_s == argmin_s |x - R(-a)s|^2 - w_s)
        # and lets the GPU labeller do the work on an axis-aligned
        # grid. Seeds rotate by `@ R`, exactly like the c-axes.
        from specimen import DiskSpecimen as _DS
        wts = build.get("weights")
        if wts is not None:
            Rd = 0.5 * D
            zc = (np.arange(ndz) - (ndz - 1) / 2) * h
            Rm = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
            sd = np.asarray(build["seeds"], float) @ Rm
            raw = _DS._label_grid_gpu(u, zc, Rd, sd, wts)
            if raw is None:
                X3, Y3, Z3 = np.meshgrid(u, u, zc, indexing="ij")
                ins = (X3 ** 2 + Y3 ** 2) <= Rd ** 2
                raw = np.full(X3.shape, -1, np.int32)
                P = np.stack([X3[ins], Y3[ins], Z3[ins]], axis=1)
                raw[ins] = _DS._assign(P, sd, wts).astype(np.int32)
            core = np.asarray(raw, np.int32).transpose(2, 1, 0)
        else:
            single_raster = False
    if not single_raster:
        core = np.where(ok[:, :, None],
                        labels[gi, gj][:, :, gz], -1).astype(np.int32)
        core = core.transpose(2, 1, 0)
    m = sponge + 3
    nz, nx = ndz + 2 * m, nd + 2 * m
    lab = np.full((nz, nx, nx), -1, np.int32)
    lab[m:m + ndz, m:m + nd, m:m + nd] = core

    R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    # row vectors: axes @ R = R.T @ v = rotation by -rot, matching the
    # geometry resample above (rigid rotation of the whole specimen).
    return lab, nd, m, axes @ R


def one_rotation(build, rot_deg, order=4, ppw=4.0, f0=2.0e6, sponge=10,
                 fluid_damp=0.02, elem_d=6.35e-3, verbose=True):
    c_ref = 3850.0
    h = c_ref / f0 / ppw
    lab, nd, m, axes_rot = rotated_grid(build, h, sponge, rot_deg)
    nz, nx = lab.shape[0], lab.shape[1]
    axes_uniform = np.tile(axes_rot[0][None, :], (len(axes_rot), 1))

    coeffs = fdtd.optimised_coeffs(order)
    Cr, rho = an.polycrystal_stiffness_3d(lab, axes_rot, c_couplant=1500.0,
                                          rho_couplant=300.0)
    dt = fdtd.safe_dt(Cr, rho, h, coeffs, safety=0.5)
    D = nd * h
    nt = int(2.6 * D / c_ref / dt)
    wav = fdtd.ricker(f0, dt, nt)
    dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    dmask = np.exp(-fluid_damp * dist).astype(np.float32)

    cz, cy, cx = nz // 2, nx // 2, nx // 2
    Rc = nd // 2
    ixp = next(cx + k - 1 for k in range(Rc - 1, 0, -1)
               if lab[cz, cy, cx + k] >= 0)
    er = max(int(elem_d / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
           for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er and lab[cz + dz, cy + dy, ixp] >= 0]
    w = 1.0 / len(pts)
    if verbose:
        print(f"   rot {rot_deg:>2} deg: grid {lab.shape}, {nt} steps", flush=True)

    tr = {}
    for tag, ax in (("A", axes_rot), ("B", axes_uniform)):
        C, rho = an.polycrystal_stiffness_3d(lab, ax, c_couplant=1500.0,
                                             rho_couplant=300.0)
        r = fdtd.forward_fused(C, rho, h, dt, nt, [(p, w) for p in pts], wav,
                               [(pts, np.full(len(pts), w))], order=order,
                               coeffs=coeffs, sponge_width=sponge,
                               damp_mask=dmask)
        tr[tag] = np.asarray(r, float).ravel()

    a, b = tr["A"], tr["B"]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(rot=rot_deg, ok=False)
    diff = a - b
    fs = 1.0 / dt
    t1 = 2 * D / c_ref
    r0, r1 = int(t1 * 0.86 * fs), min(int(t1 * 1.14 * fs), nt - 1)
    w0, w1 = int(0.42 * t1 * fs), int(t1 * 0.86 * fs)   # clear of the main bang
    E1 = np.abs(hilbert(a))[r0:r1].max()
    env = np.abs(hilbert(diff))[w0:w1]
    return dict(rot=rot_deg, ok=True, E1=E1,
                peak_db=20 * np.log10(env.max() / E1 + 1e-30),
                rms_db=20 * np.log10(np.sqrt((diff[w0:w1] ** 2).mean()) / E1 + 1e-30),
                energy_db=10 * np.log10((diff[w0:w1] ** 2).sum() / E1 ** 2 + 1e-30))


def main():
    print("SUPERSEDED as a staircase validation (2026-07-29): the probe "
          "does not follow the rotation, so this run is an azimuth sweep "
          "in disguise - see the module docstring. The verdict text below "
          "is kept for the historical record only.\n")
    sp = DiskSpecimen(diameter_m=0.060, thickness_m=0.025, n_grains=40,
                      size_cv=0.35, concentration=3.93, spatial_corr=0.0,
                      fabric_axis=(1, 0, 0), seed=7)
    build = sp.build(3850.0 / 2e6 / 6.0)
    print(f"specimen {build['labels'].shape}, {len(build['axes'])} grains")
    print("same physical specimen, described at different angles to the mesh\n")
    rows = [one_rotation(build, r) for r in (0, 17, 31, 47)]
    print()
    print(f"{'rotation':>9} {'coda peak':>11} {'coda rms':>10} {'coda energy':>12}")
    for r in rows:
        if not r["ok"]:
            print(f"{r['rot']:>8}d {'UNSTABLE':>11}")
            continue
        print(f"{r['rot']:>8}d {r['peak_db']:>10.1f}d {r['rms_db']:>9.1f}d "
              f"{r['energy_db']:>11.1f}d")
    g = [r for r in rows if r["ok"]]
    if len(g) > 1:
        pk = np.array([r["peak_db"] for r in g])
        rm = np.array([r["rms_db"] for r in g])
        print(f"\n peak   spread across rotations: {pk.max()-pk.min():.1f} dB")
        print(f" rms    spread across rotations: {rm.max()-rm.min():.1f} dB")
        print()
        if pk.max() - pk.min() < 2.0:
            print(" Coda level is INVARIANT to how the mesh samples the grains.")
            print(" That is the signature of PHYSICAL scattering: a staircase")
            print(" artefact would change when every boundary is re-stepped.")
        else:
            print(" Coda level DEPENDS on the mesh sampling, so a significant")
            print(f" part of it ({pk.max()-pk.min():.1f} dB) is staircase artefact.")


if __name__ == "__main__":
    main()
