"""Is the measured CODA converged, or is it numerics?

The -41.7 dB grain-coda level is the number that justifies changing the
instrument, so it has to be shown to be physics rather than an artefact of the
discretisation. Grid anisotropy is 0.98% at order 4 / ppw 4 against 6.9% for
ice itself, which is large enough to worry about: a mesh that scatters can
manufacture a coda, and a mesh that disperses can distort a real one.

The test is convergence. Run the same A-B experiment (real fabric minus
uniform fabric, which cancels everything common) at several discretisations:

    order 4, ppw 4   grid anisotropy 0.98%
    order 8, ppw 4   grid anisotropy 0.053%   (19x less, same cell count)
    order 4, ppw 6   grid anisotropy 0.22%    (4.4x less, 3.4x more cells)

If the coda level is the same at all three, it is converged and physical: an
artefact would change when the artefact's size changes by 19x. If it moves,
the number is not yet trustworthy and the resolution has to go up until it
stops moving.
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
from ringfwi import anisotropy as an          # noqa: E402
from scipy import ndimage                     # noqa: E402
from scipy.signal import hilbert              # noqa: E402

import fdtd                                    # noqa: E402
from specimen import DiskSpecimen              # noqa: E402


def build_grid(build, h, sponge, c_out=1500.0, rho_out=300.0):
    labels, h_spec = build["labels"], build["h"]
    nx_s, _, nz_s = labels.shape
    D = nx_s * h_spec
    nd = int(round(D / h))
    ndz = int(round(nz_s * h_spec / h))
    gx = np.clip((np.arange(nd) + 0.5) * h / h_spec, 0, nx_s - 1).astype(int)
    gz = np.clip((np.arange(ndz) + 0.5) * h / h_spec, 0, nz_s - 1).astype(int)
    core = labels[np.ix_(gx, gx, gz)].transpose(2, 1, 0).astype(np.int32)
    m = sponge + 3
    nz, nx = ndz + 2 * m, nd + 2 * m
    lab = np.full((nz, nx, nx), -1, np.int32)
    lab[m:m + ndz, m:m + nd, m:m + nd] = core
    return lab, nd, m


def one_case(build, order, ppw, f0=2.0e6, sponge=10, fluid_damp=0.02,
             elem_d=6.35e-3, verbose=True):
    """Coda-to-E1 ratio for one discretisation, via A-B differencing."""
    c_ref = 3850.0
    h = c_ref / f0 / ppw
    lab, nd, m = build_grid(build, h, sponge)
    nz, nx = lab.shape[0], lab.shape[1]
    axes = build["axes"]
    axes_uniform = np.tile(axes[0][None, :], (len(axes), 1))

    coeffs = fdtd.optimised_coeffs(order)
    Cref, rho = an.polycrystal_stiffness_3d(lab, axes, c_couplant=1500.0,
                                            rho_couplant=300.0)
    dt = fdtd.safe_dt(Cref, rho, h, coeffs, safety=0.5)
    D = nd * h
    record_to = 2.6 * D / c_ref
    nt = int(record_to / dt)
    wav = fdtd.ricker(f0, dt, nt)

    dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    dmask = np.exp(-fluid_damp * dist).astype(np.float32)

    cz, cy, cx = nz // 2, nx // 2, nx // 2
    Rc = nd // 2
    ix_probe = next(cx + k - 1 for k in range(Rc - 1, 0, -1)
                    if lab[cz, cy, cx + k] >= 0)
    er = max(int(elem_d / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ix_probe)
           for dy in range(-er, er + 1) for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er
           and lab[cz + dz, cy + dy, ix_probe] >= 0]
    w = 1.0 / len(pts)

    if verbose:
        print(f"   order {order}, ppw {ppw}: grid {lab.shape} "
              f"= {lab.size/1e6:.1f}M, {nt} steps, dt {dt*1e9:.1f} ns",
              flush=True)

    traces = {}
    for tag, ax in (("A", axes), ("B", axes_uniform)):
        C, rho = an.polycrystal_stiffness_3d(lab, ax, c_couplant=1500.0,
                                             rho_couplant=300.0)
        t0 = time.time()
        r = fdtd.forward_fused(C, rho, h, dt, nt, [(p, w) for p in pts], wav,
                               [(pts, np.full(len(pts), w))], order=order,
                               coeffs=coeffs, sponge_width=sponge,
                               damp_mask=dmask)
        traces[tag] = np.asarray(r, float).ravel()
        if verbose:
            print(f"      run {tag}: {time.time()-t0:.0f}s", flush=True)

    a, b = traces["A"], traces["B"]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(order=order, ppw=ppw, ok=False)

    diff = a - b                       # grain-orientation signal alone
    fs = 1.0 / dt
    t_e1 = 2 * D / c_ref
    r0, r1 = int(t_e1 * 0.86 * fs), min(int(t_e1 * 1.14 * fs), nt - 1)
    # Coda window set FROM THE DATA, not assumed. Source and receiver are
    # co-located on the rim, so the direct near-field dominates the early
    # record: measured at 1.000 in 0-5 us against 0.018 for E1, i.e. 55x
    # LARGER than the echo. An earlier window starting at 4 us therefore
    # measured the near-field and reported a "coda" bigger than E1, which is
    # impossible for boundary reflections at R ~ 0.01-0.035.
    # Start after the near-field has decayed (0.3 * t_E1) and stop before E1.
    w0, w1 = int(0.30 * t_e1 * fs), int(t_e1 * 0.86 * fs)
    E1 = np.abs(hilbert(a))[r0:r1].max()
    cd = np.abs(hilbert(diff))[w0:w1]
    return dict(order=order, ppw=ppw, ok=True, E1=E1,
                peak_db=20 * np.log10(cd.max() / E1 + 1e-30),
                rms_db=20 * np.log10(np.sqrt((diff[w0:w1] ** 2).mean()) / E1 + 1e-30),
                cells=lab.size, nt=nt)


def main():
    sp = DiskSpecimen(diameter_m=0.060, thickness_m=0.025, n_grains=40,
                      size_cv=0.35, concentration=3.93, spatial_corr=0.0,
                      fabric_axis=(1, 0, 0), seed=7)
    build = sp.build(3850.0 / 2e6 / 6.0)
    print(f"specimen {build['labels'].shape}, {len(build['axes'])} grains\n")
    rows = []
    for order, ppw in ((4, 4.0), (8, 4.0), (4, 6.0)):
        rows.append(one_case(build, order, ppw))
    print()
    print(f"{'order':>6} {'ppw':>5} {'grid aniso':>11} {'coda peak':>11} {'coda rms':>10}")
    ga = {(4, 4.0): "0.98%", (8, 4.0): "0.053%", (4, 6.0): "0.22%"}
    for r in rows:
        if not r["ok"]:
            print(f"{r['order']:>6} {r['ppw']:>5} {'-':>11} {'UNSTABLE':>11}")
            continue
        print(f"{r['order']:>6} {r['ppw']:>5} {ga[(r['order'],r['ppw'])]:>11} "
              f"{r['peak_db']:>10.1f}d {r['rms_db']:>9.1f}d")
    good = [r for r in rows if r["ok"]]
    if len(good) > 1:
        pk = np.array([r["peak_db"] for r in good])
        print(f"\n spread across discretisations: {pk.max()-pk.min():.1f} dB")
        if pk.max() - pk.min() < 2.0:
            print(" CONVERGED: the coda is physical, not a grid artefact.")
        else:
            print(" NOT CONVERGED: the coda level still depends on resolution.")


if __name__ == "__main__":
    main()
