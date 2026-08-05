"""Does SINGLE rasterisation reduce the numerical artefact?

Same specimen (seed 11) and the SAME legacy FD coefficients as
out/sweeps/rigid_seed11, so the only difference is how the rotated
label field is produced:
  double (rigid_seed11, already on disk) - nearest-neighbour resample
      of an already-rasterised label volume
  single (run here)                      - tessellation evaluated on
      the solver grid directly, by rotating the SEEDS

10 antipodal pairs, so the headline metric is the truth-free
reciprocity error: azimuth a and a+180 traverse the identical chord,
so any arrival-time difference is pure numerical error.
"""
import os
import sys
import numpy as np
from scipy import ndimage
from scipy.signal import hilbert

SP = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim", "results")))
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vendor"))))
import fdtd                                       # noqa: E402
from rotation_test import rotated_grid            # noqa: E402
from specimen import DiskSpecimen                 # noqa: E402

C_REF, F0, PPW, ORDER, SPONGE, DAMP = 3850.0, 2.0e6, 6.0, 8, 10, 0.02
ELEM, RECF = 6.35e-3, 2.7
PAIRS = [(a, a + 180) for a in (0, 18, 36, 54, 72, 90, 108, 126, 144, 162)]
REPO = ((os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")) +
        r"\out\sweeps\rigid_seed11"))

h = C_REF / F0 / PPW
co = fdtd.optimised_coeffs(ORDER, kh_max=1.6, multistart=False)  # legacy
sp = DiskSpecimen(diameter_m=0.100, thickness_m=0.035, n_grains=100,
                  size_cv=0.35, concentration=3.93, spatial_corr=0.0,
                  fabric_axis=(0.866, 0.5, 0.0), seed=11)
build = sp.build(h)
print(f"specimen {build['labels'].shape}; legacy coeffs "
      f"{np.array2string(co, precision=6)}", flush=True)


def run(rot):
    lab, nd, m, axes = rotated_grid(build, h, SPONGE, rot,
                                    single_raster=True)
    nz, nyy = lab.shape[0], lab.shape[1]
    Ct, rho_t = fdtd.material_tables(axes)
    dt = fdtd.safe_dt_labels((lab + 1).astype(np.uint8), Ct, rho_t, h,
                             co, safety=0.5)
    nt = int(RECF * nd * h / C_REF / dt)
    wav = fdtd.ricker(F0, dt, nt)
    dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    dm = np.exp(-DAMP * dist).astype(np.float32)
    cz, cy, cx = nz // 2, nyy // 2, nyy // 2
    ixp = next(cx + k - 1 for k in range(nd // 2 - 1, 0, -1)
               if lab[cz, cy, cx + k] >= 0)
    er = max(int(ELEM / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
           for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er and lab[cz + dz, cy + dy, ixp] >= 0]
    w = 1.0 / len(pts)
    tr = np.asarray(fdtd.forward_fused_labels(
        lab, axes, h, dt, nt, [(p, w) for p in pts], wav,
        [(pts, np.full(len(pts), w))], order=ORDER, coeffs=co,
        sponge_width=SPONGE, damp_mask=dm), float).ravel()
    return tr, dt


def diag(tr, dt):
    fs = 1.0 / dt
    e = np.abs(hilbert(tr))
    k0 = int(2 * 0.100 / C_REF * fs)
    a = max(k0 - int(2e-6 * fs), 0)
    s = e[a:k0 + int(2e-6 * fs)]
    E1 = s.max()
    ip = int(np.argmax(s))
    j = np.where(s[:ip + 1] >= 0.25 * E1)[0]
    tof = (a + (int(j[0]) if len(j) else ip)) / fs
    seg = tr[int(24e-6 * fs):int(36e-6 * fs)]
    F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    fr = np.fft.rfftfreq(len(seg), dt)
    coda = np.sqrt((e[int(24e-6 * fs):int(36e-6 * fs)] ** 2).mean())
    return (tof, 20 * np.log10(coda / E1), fr[np.argmax(F)] / 1e6,
            F[fr > 3e6].sum() / F.sum() * 100)


def repo(rot):
    with np.load(os.path.join(REPO, f"az{rot:03d}.npz")) as z:
        return diag(np.asarray(z["trace"], float).ravel(), float(z["dt"]))


rows = {"double": [], "single": []}
for i, (a, b) in enumerate(PAIRS):
    ra, rb = repo(a), repo(b)
    rows["double"].append((abs(ra[0] - rb[0]) * 1e6, ra[1], rb[1],
                           ra[2], rb[2], ra[3], rb[3]))
    ta, dta = run(a)
    tb, dtb = run(b)
    da, db = diag(ta, dta), diag(tb, dtb)
    rows["single"].append((abs(da[0] - db[0]) * 1e6, da[1], db[1],
                           da[2], db[2], da[3], db[3]))
    print(f"  pair {i+1}/10 ({a:3d},{b:3d}): double dToF "
          f"{rows['double'][-1][0]:.3f} | single {rows['single'][-1][0]:.3f} us",
          flush=True)

print("\n%-8s %-14s %-14s %-13s %s"
      % ("raster", "recip rms (us)", "coda/E1 (dB)", "peak (MHz)", "%>3MHz"))
for k in ("double", "single"):
    r = np.array(rows[k])
    print("%-8s %8.3f       %8.2f       %7.2f      %6.1f"
          % (k, np.sqrt((r[:, 0] ** 2).mean()),
             np.r_[r[:, 1], r[:, 2]].mean(), np.r_[r[:, 3], r[:, 4]].mean(),
             np.r_[r[:, 5], r[:, 6]].mean()))
D, S = np.array(rows["double"]), np.array(rows["single"])
for lab, ia, ib in (("reciprocity", 0, None), ("coda/E1", 1, 2),
                    ("peak MHz", 3, 4), ("%>3MHz", 5, 6)):
    d = (D[:, ia] - S[:, ia]) if ib is None else \
        (np.r_[D[:, ia], D[:, ib]] - np.r_[S[:, ia], S[:, ib]])
    se = d.std(ddof=1) / np.sqrt(len(d))
    print("paired (double - single) %-12s %+7.3f +- %.3f  (%.1f sigma)"
          % (lab, d.mean(), se, abs(d.mean()) / se if se else 0))
np.savez(os.path.join(SP, "raster_test.npz"), double=D, single=S)
