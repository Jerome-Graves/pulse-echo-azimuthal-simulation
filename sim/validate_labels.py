"""Validate the label-indexed solver against the dense fused solver.

Three gates, in order of cost:
  1. --quick : small grid, few hundred steps, dense vs labels trace compare.
     Must agree to float roundoff (they are the same arithmetic).
  2. --ref   : full 2 MHz fabric-0 trace via labels vs ref/fw_fabric00.npz.
     Pass = correlation > 0.999 and coda level within 0.2 dB.
  3. --probe5: build the 5 MHz full-disk grid, report memory, time 30 steps,
     project the per-trace wall time. No trace saved.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
from ringfwi import anisotropy as an          # noqa: E402
from scipy import ndimage                     # noqa: E402
from scipy.signal import hilbert              # noqa: E402

import coda_convergence as CC                  # noqa: E402
import fdtd                                    # noqa: E402
import ladder                                  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")
C_REF = 3850.0


def setup(f0, build, ppw=6.0, sponge=10, fluid_damp=0.02):
    # fluid_damp 0.02, NOT 0.01: at 0.01 the rim instability survives
    # (~0.2-0.5%/step, argmax at end of record) and inflated the coda by
    # ~15 dB in every pre-2026-07-25 reference. 0.03 kills it (tail
    # 47x E1 -> 0.02x). Damping must start AT the first fluid cell:
    # offsetting the ramp off the rim revives the instability (tail 524x).
    # Cost: E1 drops ~18% (rim reflection absorbs); coda/E1 ~rate-stable.
    h = C_REF / f0 / ppw
    lab, nd, m = CC.build_grid(build, h, sponge)
    nz, nx = lab.shape[0], lab.shape[1]
    co = fdtd.optimised_coeffs(8)
    mats = (lab + 1).astype(np.uint8)
    Ct, rho_t = fdtd.material_tables(build["axes"])
    dt = fdtd.safe_dt_labels(mats, Ct, rho_t, h, co, safety=0.5)
    D = nd * h
    nt = int(2.6 * D / C_REF / dt)
    wav = fdtd.ricker(f0, dt, nt)
    dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    dm = np.exp(-fluid_damp * dist).astype(np.float32)
    cz, cy, cx = nz // 2, nx // 2, nx // 2
    Rc = nd // 2
    ixp = next(cx + k - 1 for k in range(Rc - 1, 0, -1)
               if lab[cz, cy, cx + k] >= 0)
    er = max(int(6.35e-3 / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
           for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er and lab[cz + dz, cy + dy, ixp] >= 0]
    w = 1.0 / len(pts)
    return dict(lab=lab, h=h, dt=dt, nt=nt, wav=wav, dm=dm, co=co,
                pts=pts, w=w, build=build, D=D)


def quick():
    """Small specimen, truncated run, dense vs labels."""
    from specimen import DiskSpecimen
    sp = DiskSpecimen(diameter_m=0.060, thickness_m=0.025, n_grains=40,
                      size_cv=0.35, concentration=3.93, spatial_corr=0.0,
                      fabric_axis=(1, 0, 0), seed=7)
    build = sp.build(C_REF / 2e6 / 6.0)
    st = setup(2e6, build)
    nt = st["nt"]          # FULL length: slow-growth regressions are
                           # invisible at 400 steps (review finding)
    lab, co, dt, h = st["lab"], st["co"], st["dt"], st["h"]
    srcs = [(p, st["w"]) for p in st["pts"]]
    recs = [(st["pts"], np.full(len(st["pts"]), st["w"]))]

    C, rho = an.polycrystal_stiffness_3d(lab, st["build"]["axes"],
                                         c_couplant=1500.0, rho_couplant=300.0)
    t0 = time.time()
    a = np.asarray(fdtd.forward_fused(C, rho, h, dt, nt, srcs, st["wav"],
                                      recs, order=8, coeffs=co,
                                      sponge_width=10,
                                      damp_mask=st["dm"]), float).ravel()
    ta = time.time() - t0
    t0 = time.time()
    b = np.asarray(fdtd.forward_fused_labels(lab, st["build"]["axes"], h, dt,
                                             nt, srcs, st["wav"], recs,
                                             order=8, coeffs=co,
                                             sponge_width=10,
                                             damp_mask=st["dm"]),
                   float).ravel()
    tb = time.time() - t0
    scale = np.abs(a).max()
    err = np.abs(a - b).max() / scale
    print(f"quick: {nt} steps, dense {ta:.1f}s vs labels {tb:.1f}s")
    print(f"       max |diff| / max |trace| = {err:.2e}  "
          f"({'PASS' if err == 0.0 else 'FAIL'} at BIT-IDENTICAL)")


def ref():
    st = setup(2e6, ladder.standard_build())
    srcs = [(p, st["w"]) for p in st["pts"]]
    recs = [(st["pts"], np.full(len(st["pts"]), st["w"]))]
    print(f"grid {st['lab'].shape}, {st['nt']} steps, "
          f"dt {st['dt']*1e9:.2f} ns", flush=True)
    t0 = time.time()
    tr = np.asarray(fdtd.forward_fused_labels(
        st["lab"], st["build"]["axes"], st["h"], st["dt"], st["nt"], srcs,
        st["wav"], recs, order=8, coeffs=st["co"], sponge_width=10,
        damp_mask=st["dm"]), float).ravel()
    el = time.time() - t0
    d = np.load(os.path.join(REF, "fw_fabric00.npz"))
    refr, dtr = d["trace"].ravel(), float(d["dt"])
    n = min(len(tr), len(refr))
    corr = float(np.corrcoef(tr[:n], refr[:n])[0, 1])

    def coda_level(x, dt):
        t1 = 2 * 0.100 / C_REF
        e = np.abs(hilbert(x))
        k0 = int(t1 / dt)
        E1 = e[max(k0 - int(2e-6 / dt), 0):k0 + int(2e-6 / dt)].max()
        lo, hi = int(22e-6 / dt), min(int(42e-6 / dt), len(e))
        return 20 * np.log10(e[lo:hi].max() / E1 + 1e-30)

    la, lb = coda_level(tr, st["dt"]), coda_level(refr, dtr)
    print(f"labels run: {el:.0f}s; correlation vs reference {corr:.6f}")
    print(f"coda level: labels {la:.2f} dB vs reference {lb:.2f} dB "
          f"(diff {abs(la-lb):.2f})")
    print("PASS" if corr > 0.999 and abs(la - lb) < 0.2 else "FAIL")


def probe5():
    import cupy as cp
    from specimen import DiskSpecimen
    f0 = 5.0e6
    h = C_REF / f0 / 6.0
    sp = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                      size_cv=0.35, concentration=3.93, spatial_corr=0.0,
                      fabric_axis=(1, 0, 0), seed=7)
    print("building 5 MHz specimen + grid (CPU)...", flush=True)
    build = sp.build(h)
    st = setup(f0, build)
    lab = st["lab"]
    print(f"grid {lab.shape} = {lab.size/1e6:.0f}M cells, {st['nt']} steps, "
          f"dt {st['dt']*1e9:.2f} ns", flush=True)
    est = lab.size * (9 * 4 + 1 + 4) / 1e9
    print(f"estimated device memory ~ {est:.1f} GB "
          f"(+ slab scratch + overhead)", flush=True)
    srcs = [(p, st["w"]) for p in st["pts"]]
    recs = [(st["pts"], np.full(len(st["pts"]), st["w"]))]
    nt_probe = 30
    t0 = time.time()
    fdtd.forward_fused_labels(lab, build["axes"], st["h"], st["dt"], nt_probe,
                              srcs, st["wav"][:nt_probe], recs, order=8,
                              coeffs=st["co"], sponge_width=10,
                              damp_mask=st["dm"])
    el = time.time() - t0
    per = el / nt_probe
    used = cp.get_default_memory_pool().used_bytes() / 1e9
    print(f"30 steps in {el:.1f}s -> {per*1e3:.0f} ms/step, "
          f"device pool {used:.1f} GB")
    print(f"projected full trace: {per*st['nt']/60:.0f} min "
          f"({st['nt']} steps)")


if __name__ == "__main__":
    if "--ref" in sys.argv:
        ref()
    elif "--probe5" in sys.argv:
        probe5()
    else:
        quick()
