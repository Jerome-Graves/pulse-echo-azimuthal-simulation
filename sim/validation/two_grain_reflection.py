"""Two validations, run back to back on the GPU.

(A) ZERO-CONTRAST CONTROL.  The identical specimen, identical grid,
    identical everything, except every grain is given the SAME c-axis.
    The tessellation is still there but it is acoustically invisible, so
    a correct solver must return no coda.  Whatever coda IS returned is
    the numerical floor, measured rather than inferred.  There has never
    been such a measurement in this project: isotropic_seed41_ppw6_calibration is NOT one, because
    random orientations still give full grain-to-grain contrast.

(B) BICRYSTAL VALIDATION.  Two grains, one flat boundary normal to the
    beam at 40 mm depth, where the reflection coefficient is known in
    closed form: at normal incidence between two media of equal density,
    R = (v2 - v1) / (v2 + v1).  Sweep the c-axis pair to vary R over the
    whole range ice can produce, and check the measured echo is LINEAR in
    R with the right slope.  This validates the forward model against
    exact theory rather than against a correlation, so it does not depend
    on any null, p-value or analysis choice.

Both use the exact production recipe from sweep_runner: ppw 8, order 8,
multistart coefficients at kh_max 2.0, sponge 10, fluid damp 0.02,
record_factor 2.7, 6.35 mm element.
"""
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.signal import hilbert

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
import fdtd                                            # noqa: E402
import forward as F                                    # noqa: E402
from rotation_test import rotated_grid                 # noqa: E402
from specimen import DiskSpecimen                      # noqa: E402

OUT = (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "sweeps")))
SP = os.path.dirname(os.path.abspath(__file__))
C_REF, F0, PPW, ORDER = 3850.0, 2.0e6, 8.0, 8
SPONGE, DAMP, ELEM, RECF, KHM = 10, 0.02, 6.35e-3, 2.7, 2.0
DIA, THK = 0.100, 0.035
GATE = (24e-6, 36e-6)
h = C_REF / F0 / PPW
CO = fdtd.optimised_coeffs(ORDER, kh_max=KHM, multistart=True)


def solve(lab, axes, nd, tag):
    nz, nyy = lab.shape[0], lab.shape[1]
    Ct, rho_t = fdtd.material_tables(axes)
    dt = fdtd.safe_dt_labels((lab + 1).astype(np.uint8), Ct, rho_t, h, CO,
                             safety=0.5)
    D = nd * h
    nt = int(RECF * D / C_REF / dt)
    wav = fdtd.ricker(F0, dt, nt)
    dist = ndimage.distance_transform_edt((lab < 0).astype(np.float32))
    dm = np.exp(-DAMP * dist).astype(np.float32)
    cz, cy, cx = nz // 2, nyy // 2, nyy // 2
    Rc = nd // 2
    ixp = next(cx + k - 1 for k in range(Rc - 1, 0, -1)
               if lab[cz, cy, cx + k] >= 0)
    er = max(int(ELEM / 2 / h), 1)
    pts = [(cz + dz, cy + dy, ixp) for dy in range(-er, er + 1)
           for dz in range(-er, er + 1)
           if dy * dy + dz * dz <= er * er and lab[cz + dz, cy + dy, ixp] >= 0]
    w = 1.0 / len(pts)
    t0 = time.time()
    tr = np.asarray(fdtd.forward_fused_labels(
        lab, axes, h, dt, nt, [(p, w) for p in pts], wav,
        [(pts, np.full(len(pts), w))], order=ORDER, coeffs=CO,
        sponge_width=SPONGE, damp_mask=dm), float).ravel()
    print(f"    {tag}: {nt} steps, {time.time()-t0:.0f} s", flush=True)
    return tr, dt


def coda_db(tr, dt):
    fs = 1.0 / dt
    e = np.abs(hilbert(tr))
    k0, w = int(2 * DIA / C_REF * fs), int(2e-6 * fs)
    E1 = e[max(k0 - w, 0):k0 + w].max()
    cd = np.sqrt((e[int(GATE[0] * fs):int(GATE[1] * fs)] ** 2).mean())
    return 20 * np.log10(cd / E1), E1, e.max()


# ===================== (A) ZERO-CONTRAST CONTROL =====================
print("=" * 70)
print("(A) ZERO-CONTRAST CONTROL  -  measuring the numerical coda floor")
print("=" * 70, flush=True)
sp = DiskSpecimen(diameter_m=DIA, thickness_m=THK, n_grains=100,
                  size_cv=0.35, concentration=-8.0, spatial_corr=0.0,
                  fabric_axis=(1.0, 0.0, 0.0), seed=11)
build = sp.build(h)
AZ = [0, 60, 120, 180, 240, 300]
rows = []
for az in AZ:
    lab, nd, m, axr = rotated_grid(build, h, SPONGE, az, single_raster=True)
    uni = np.tile(axr[0][None, :], (len(axr), 1))       # every grain the same
    tr, dt = solve(lab, uni, nd, f"az {az} uniform-axes")
    cz, e1, bang = coda_db(tr, dt)
    with np.load(os.path.join(OUT, "girdle_seed11_ppw8_dev",
                              f"az{az:03d}.npz")) as z:
        rt, rdt = np.asarray(z["trace"], float).ravel(), float(z["dt"])
    cr, e1r, bangr = coda_db(rt, rdt)
    rows.append((az, cr, cz))
    print(f"  az {az:3d}:  real fabric {cr:7.2f} dB   "
          f"zero-contrast {cz:7.2f} dB   difference {cr-cz:+6.2f} dB",
          flush=True)
R = np.array(rows)
print(f"\n  MEAN real {R[:,1].mean():.2f} dB   "
      f"MEAN zero-contrast {R[:,2].mean():.2f} dB")
print(f"  numerical floor sits {R[:,1].mean()-R[:,2].mean():.2f} dB below "
      f"the real coda")
frac = 10 ** ((R[:, 2].mean() - R[:, 1].mean()) / 10)
print(f"  => {100*frac:.1f}% of the ppw8 coda POWER is numerical, "
      f"{100*(1-frac):.1f}% is physical")
np.savez(os.path.join(SP, "zerocontrast.npz"), rows=R)

# ===================== (B) BICRYSTAL VALIDATION =====================
print("\n" + "=" * 70)
print("(B) BICRYSTAL  -  measured echo vs the closed-form reflection "
      "coefficient")
print("=" * 70, flush=True)
DB = 0.040                                  # boundary depth from the rim
nd = int(round(DIA / h))
ndz = int(round(THK / h))
mm = SPONGE + 3
u = (np.arange(nd) + 0.5) * h - DIA / 2
X, Y = np.meshgrid(u, u, indexing="ij")
ins = (X ** 2 + Y ** 2) <= (DIA / 2) ** 2
core2 = np.where(ins, np.where(X > (DIA / 2 - DB), 0, 1), -1).astype(np.int32)
lab2 = np.full((ndz + 2 * mm, nd + 2 * mm, nd + 2 * mm), -1, np.int32)
lab2[mm:mm + ndz, mm:mm + nd, mm:mm + nd] = \
    np.repeat(core2[None, :, :], ndz, axis=0).transpose(0, 2, 1)
print(f"  grid {lab2.shape}, boundary at {DB*1e3:.0f} mm depth, "
      f"echo expected at {2*DB/C_REF*1e6 + 1.2/F0*1e6:.1f} us", flush=True)


def vof(psi):
    return float(np.interp(psi, F._PSI, F._VQP))


PAIRS = [(0.0, 0.0), (0.0, 20.0), (0.0, 35.0), (0.0, 51.0),
         (0.0, 90.0), (90.0, 51.0)]
res = []
for pa, pb in PAIRS:
    axes = np.array([[np.cos(np.radians(p)), 0.0, np.sin(np.radians(p))]
                     for p in (pa, pb)])
    va, vb = vof(np.radians(pa)), vof(np.radians(pb))
    Rth = (vb - va) / (vb + va)
    tr, dt = solve(lab2, axes, nd, f"psi {pa:.0f}/{pb:.0f}")
    fs = 1.0 / dt
    e = np.abs(hilbert(tr))
    tb = 2 * DB / va + 1.2 / F0
    wb = int(1.5e-6 * fs)
    ib = int(tb * fs)
    Eb = e[max(ib - wb, 0):ib + wb].max()
    k0, w = int(2 * DIA / C_REF * fs), int(2e-6 * fs)
    E1 = e[max(k0 - w, 0):k0 + w].max()
    res.append((pa, pb, Rth, Eb / E1, va, vb))
    print(f"  psi {pa:5.1f}/{pb:5.1f}  v {va:6.1f}/{vb:6.1f}  "
          f"R_theory {Rth:+.5f}   Eb/E1 {Eb/E1:.5f}", flush=True)

A = np.array([(r[2], r[3]) for r in res])
nz_ = np.abs(A[:, 0]) > 1e-9
sl, ic = np.polyfit(np.abs(A[nz_, 0]), A[nz_, 1], 1)
rr = np.corrcoef(np.abs(A[nz_, 0]), A[nz_, 1])[0, 1]
print(f"\n  |R_theory| vs measured Eb/E1 :  slope {sl:.3f}, "
      f"intercept {ic:+.5f}, r = {rr:+.4f}  (n={nz_.sum()})")
print(f"  zero-contrast pair (0/0) gave Eb/E1 = {A[~nz_, 1][0]:.5f} "
      f"<- must be ~0")
Rbw = sl * 1.0
print(f"  implied backwall reference: slope x (d_b/d_1) = "
      f"{sl*DB/(DIA):.4f}")
print("  a slope that is CONSTANT across R (r near 1) validates the model's "
      "form;\n  the zero pair validates that nothing is manufactured.")
np.savez(os.path.join(SP, "bicrystal_val.npz"), res=np.array(res))
print("\nDONE", flush=True)
