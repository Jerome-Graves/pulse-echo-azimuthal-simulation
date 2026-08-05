"""Full simulation programme for the Ultrasonics paper.

Ordered by value per GPU hour, and RESUMABLE: any azimuth whose .npz
already exists is skipped, so the script can be stopped and restarted
without losing work.

  1  ppw 10 licensing run   seed 11 girdle, 30 az. Decides whether the
                            matrix can run at ppw 8. 2.2 h
  2  tessellation matrix    5 new seeds x 2 fabrics, ppw 8, 30 az. The
                            headline result and the ensemble at once. 8.9 h
  3  contrast ladder        seed 11 girdle, stiffness interpolated a
                            fraction f from the orientation-mean tensor
                            toward each grain's own, f = 0 .. 0.75. This
                            REPLACES the uniform-orientation control,
                            which was blind to the contrast-carrying
                            staircase. 3.6 h
  4  numerical floor        zero contrast at ppw 6 and ppw 10. 2.5 h

Total about 17 GPU hours. Progress is written to master_status.txt.
"""
import os as _os
import sys as _sys
# helper modules shared with sibling directories
for _d in ('../validation',):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import json
import os
import sys
import time
import traceback

import numpy as np
from scipy import ndimage

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
import fdtd                                            # noqa: E402
from rotation_test import rotated_grid                 # noqa: E402
from specimen import DiskSpecimen                      # noqa: E402

SWD = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "sweeps")))
STATUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "master_status.txt")
C_REF, F0, ORDER, KHM = 3850.0, 2.0e6, 8, 2.0
SPONGE, DAMP, ELEM, RECF = 10, 0.02, 6.35e-3, 2.7
DIA, THK, NG, CV = 0.100, 0.035, 100, 0.35

GIRDLE = dict(kappa=-8.0, axis=(1.0, 0.0, 0.0), tag="girdle")
SINGLE = dict(kappa=3.93, axis=(0.866, 0.5, 0.0), tag="single")

JOBS = []
# --- 1. licensing run: does the facet correlation survive ppw 6->10? ---
JOBS.append(dict(name="girdle_seed11_ppw10_licensing", ppw=10.0, step=12,
                 seed=11, mode="normal", **GIRDLE))
# --- 2. tessellation x fabric matrix at ppw 8 --------------------------
# Weighted toward TESSELLATION count rather than fabric count. Independent
# microstructures serve the headline claim (the coda is predicted by grain
# geometry), which currently rests on one; the fabric contrast serves the
# blindness result, which already has the rank test and the 22-observable
# panel behind it. Seven new girdle seeds give eight tessellations with
# seed 11; three of them repeated in single maximum give four
# same-tessellation fabric pairs.
for sd in (23, 41, 17, 7, 53, 71, 89):
    JOBS.append(dict(name="girdle_seed%d_ppw8_ensemble" % sd, ppw=8.0, step=12,
                     seed=sd, mode="normal", **GIRDLE))
for sd in (23, 41, 17):
    JOBS.append(dict(name="singlemax_seed%d_ppw8_ensemble" % sd, ppw=8.0, step=12,
                     seed=sd, mode="normal", **SINGLE))
# --- 3. contrast-scaling ladder (f = 1 is the existing 60-az sweep) ----
for f in (0.0, 0.25, 0.5, 0.75):
    JOBS.append(dict(name="girdle_seed11_ppw8_contrast_f%03d" % int(f * 100), ppw=8.0,
                     step=12, seed=11, mode="contrast", frac=f, **GIRDLE))
# --- 4. numerical floor at the other two resolutions -------------------
JOBS.append(dict(name="girdle_seed11_ppw6_zerocontrast", ppw=6.0, step=12, seed=11,
                 mode="zero", **GIRDLE))
JOBS.append(dict(name="girdle_seed11_ppw10_zerocontrast", ppw=10.0, step=12, seed=11,
                 mode="zero", **GIRDLE))


def note(msg):
    line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(STATUS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def tables_for(job, axes):
    """(Ct, rho_t) with the job's contrast treatment applied."""
    Ct, rho_t = fdtd.material_tables(axes)
    if job["mode"] == "zero":
        # every grain identical: no acoustic contrast anywhere
        Ct = Ct.copy()
        Ct[1:] = Ct[1]
    elif job["mode"] == "contrast":
        # interpolate each grain a fraction f from the orientation mean
        # toward its own tensor. Physical scattering scales as f^2; any
        # component that does not is numerical.
        Ct = Ct.copy()
        mean = Ct[1:].mean(axis=0, keepdims=True)
        Ct[1:] = mean + job["frac"] * (Ct[1:] - mean)
    return Ct, rho_t


def run_job(job):
    out = os.path.join(SWD, job["name"])
    os.makedirs(out, exist_ok=True)
    cfg = {"f0_mhz": 2.0, "ppw": job["ppw"], "order": ORDER,
           "az_start": 0, "az_stop": 360, "az_step": job["step"],
           "record_factor": RECF, "fluid_damp": DAMP, "sponge": SPONGE,
           "element_d_mm": 6.35, "diameter_mm": 100.0, "thickness_mm": 35.0,
           "n_grains": NG, "size_cv": CV, "concentration": job["kappa"],
           "spatial_corr": job.get("spatial_corr", 0.0),
           "fabric_axis": list(job["axis"]),
           "seed": job["seed"], "axes_convention": "rigid2",
           "fd_kh_max": KHM, "rasterise": "single", "name": job["name"],
           "treatment": job["mode"]}
    if job["mode"] == "contrast":
        cfg["contrast_fraction"] = job["frac"]
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)

    az = list(range(0, 360, job["step"]))
    todo = [a for a in az
            if not os.path.exists(os.path.join(out, "az%03d.npz" % a))]
    if not todo:
        note("  %s already complete" % job["name"])
        return
    h = C_REF / F0 / job["ppw"]
    co = fdtd.optimised_coeffs(ORDER, kh_max=KHM, multistart=True)
    build = DiskSpecimen(diameter_m=DIA, thickness_m=THK, n_grains=NG,
                         size_cv=CV, concentration=job["kappa"],
                         spatial_corr=job.get("spatial_corr", 0.0),
                         fabric_axis=job["axis"],
                         seed=job["seed"]).build(h)
    note("  %s: %d/%d azimuths to run, h=%.3f mm"
         % (job["name"], len(todo), len(az), h * 1e3))
    times = []
    for i, a in enumerate(todo):
        t0 = time.time()
        lab, nd, m, axr = rotated_grid(build, h, SPONGE, a,
                                       single_raster=True)
        Ct, rho_t = tables_for(job, axr)
        dt = fdtd.safe_dt_labels((lab + 1).astype(np.uint8), Ct, rho_t, h,
                                 co, safety=0.5)
        nt = int(RECF * nd * h / C_REF / dt)
        wav = fdtd.ricker(F0, dt, nt)
        dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
        dm = np.exp(-DAMP * dist).astype(np.float32)
        nz, nyy = lab.shape[0], lab.shape[1]
        cz, cy, cx = nz // 2, nyy // 2, nyy // 2
        ixp = next(cx + k - 1 for k in range(nd // 2 - 1, 0, -1)
                   if lab[cz, cy, cx + k] >= 0)
        er = max(int(ELEM / 2 / h), 1)
        pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
               for dz in range(-er, er + 1)
               if dy * dy + dz * dz <= er * er
               and lab[cz + dz, cy + dy, ixp] >= 0]
        w = 1.0 / len(pts)
        tr = np.asarray(fdtd.forward_fused_labels(
            lab, axr, h, dt, nt, [(p, w) for p in pts], wav,
            [(pts, np.full(len(pts), w))], order=ORDER, coeffs=co,
            sponge_width=SPONGE, damp_mask=dm,
            mat_tables=(Ct, rho_t)), float).ravel()
        np.savez_compressed(os.path.join(out, "az%03d.npz" % a),
                            trace=tr.astype(np.float32), dt=dt, az=a)
        times.append(time.time() - t0)
        if (i + 1) % 5 == 0 or i == len(todo) - 1:
            left = (len(todo) - i - 1) * float(np.mean(times)) / 60
            note("    %s az %3d (%d/%d) %.0f s  ~%.0f min left"
                 % (job["name"], a, i + 1, len(todo), times[-1], left))


def bicrystal_ladder():
    """Stage 0: extend the tilted-interface test to a 7-point resolution
    ladder so the convergence ORDER is fitted rather than estimated from
    three points. Contains no grains, so this result is independent of
    grain size, which answers the objection that a criterion derived at
    D/lambda = 9 may not transfer. About 20 minutes."""
    import tilt_testbed as TB
    res = {}
    for ppw in (4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0):
        errs = []
        base = None
        for t in (0.0, 15.0, 22.5, 30.0, 45.0):
            bi, dt, d = TB.run(t, ppw, "naive", 0.0, 51.0)
            ho, _, _ = TB.run(t, ppw, "naive", 0.0, 51.0, homog=True,
                              dt_in=dt)
            amp = TB.echo_amp(bi, ho, dt, d, TB.vA(0.0))
            if base is None:
                base = amp
            else:
                errs.append(abs(20 * np.log10(amp / base)))
        res[ppw] = float(np.mean(errs))
        note("    bicrystal ppw %4.1f : mean |error| %5.2f dB"
             % (ppw, res[ppw]))
    x = np.log(1.0 / np.array(sorted(res)))
    y = np.log([1 - 10 ** (-res[p] / 20) for p in sorted(res)])
    p = np.polyfit(x, y, 1)[0]
    note("    fitted convergence order: h^%.2f  (first order = 1.00)" % p)
    np.savez(os.path.join(os.path.dirname(STATUS), "bicrystal_ladder.npz"),
             ppw=np.array(sorted(res)),
             err=np.array([res[k] for k in sorted(res)]), order=p)


if __name__ == "__main__":
    note("=" * 60)
    note("MASTER RUN: bicrystal ladder + %d sweeps" % len(JOBS))
    t_start = time.time()
    note("[0] bicrystal resolution ladder")
    try:
        bicrystal_ladder()
    except Exception:
        note("  bicrystal ladder FAILED, continuing:\n"
             + traceback.format_exc())
    for j, job in enumerate(JOBS, 1):
        note("[%d/%d] %s  (ppw %.0f, seed %d, %s)"
             % (j, len(JOBS), job["name"], job["ppw"], job["seed"],
                job["mode"]))
        try:
            run_job(job)
        except Exception:
            note("  FAILED, continuing:\n" + traceback.format_exc())
    note("ALL DONE in %.1f h" % ((time.time() - t_start) / 3600))
