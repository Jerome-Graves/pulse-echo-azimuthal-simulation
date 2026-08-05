"""How many grains does the beam actually interrogate, and what speckle
floor does that imply?

The coda window (24-36 us) samples backscatter from 46-69 mm along the
100 mm chord.  Count the grain-boundary FACETS inside the 6.35 mm beam
column over that depth range, for the real seed-11 girdle specimen.

For an incoherent sum of N equal scatterers the envelope is Rayleigh and
the level has a standard deviation of 5.57 dB per independent look; with
N discrete facets the per-azimuth level fluctuates by ~5.57/sqrt(N_eff).
"""
import os
import sys

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
from specimen import DiskSpecimen              # noqa: E402

C_REF, F0, PPW = 3850.0, 2.0e6, 8.0
ELEM = 6.35e-3
D = 0.100
CODA_T = (24e-6, 36e-6)

h = C_REF / F0 / PPW
sp = DiskSpecimen(diameter_m=D, thickness_m=0.035, n_grains=100,
                  size_cv=0.35, concentration=-8.0, spatial_corr=0.0,
                  fabric_axis=(1.0, 0.0, 0.0), seed=11)
b = sp.build(h)
lab = b["labels"]                       # (nx, ny, nz), -1 outside
nx, ny, nz = lab.shape
print(f"grid {lab.shape}  h={h*1e3:.3f} mm")
ok = lab >= 0
vol = np.bincount(lab[ok].ravel(), minlength=lab.max() + 1) * h ** 3
vol = vol[vol > 0]
deq = (6 * vol / np.pi) ** (1 / 3) * 1e3
print(f"{len(vol)} grains, equivalent diameter {deq.mean():.1f} +- "
      f"{deq.std():.1f} mm  (range {deq.min():.1f}-{deq.max():.1f})")

# beam: enters at +x rim, travels in -x.  depth d along the chord.
d0, d1 = CODA_T[0] * C_REF / 2, CODA_T[1] * C_REF / 2
print(f"\ncoda window {CODA_T[0]*1e6:.0f}-{CODA_T[1]*1e6:.0f} us "
      f"-> depth {d0*1e3:.0f}-{d1*1e3:.0f} mm along the {D*1e3:.0f} mm chord")

xc = (np.arange(nx) + 0.5) * h - nx * h / 2
yc = (np.arange(ny) + 0.5) * h - ny * h / 2
zc = (np.arange(nz) + 0.5) * h - nz * h / 2
er = ELEM / 2

nax = 60
ngr, nfac, nline = [], [], []
for k in range(nax):
    a = np.radians(k * 360.0 / nax)
    ca, sa = np.cos(a), np.sin(a)
    # beam axis in specimen frame: from rim at +n inward along -n
    n = np.array([ca, sa, 0.0])
    t = np.array([-sa, ca, 0.0])
    # sample the beam column
    s = np.arange(d0, d1, h)                       # along-beam distance
    off = np.arange(-er, er + h, h)
    P = (D / 2 * n[None, None, None, :]
         - s[:, None, None, None] * n[None, None, None, :]
         + off[None, :, None, None] * t[None, None, None, :]
         + zc[None, None, :, None] * np.array([0, 0, 1.0]))
    P = P.reshape(-1, 3)
    r2 = P[:, 0] ** 2 + P[:, 1] ** 2
    P = P[(r2 <= (D / 2) ** 2) & (np.abs(P[:, 2]) <= zc.max())]
    gi = np.clip(np.rint((P[:, 0] + nx * h / 2) / h - 0.5), 0, nx - 1).astype(int)
    gj = np.clip(np.rint((P[:, 1] + ny * h / 2) / h - 0.5), 0, ny - 1).astype(int)
    gk = np.clip(np.rint((P[:, 2] + nz * h / 2) / h - 0.5), 0, nz - 1).astype(int)
    L = lab[gi, gj, gk]
    L = L[L >= 0]
    ngr.append(len(np.unique(L)))
    # on-axis line: count label changes = boundary crossings
    sl = np.arange(d0, d1, h / 4)
    Q = D / 2 * n[None, :] - sl[:, None] * n[None, :]
    qi = np.clip(np.rint((Q[:, 0] + nx * h / 2) / h - 0.5), 0, nx - 1).astype(int)
    qj = np.clip(np.rint((Q[:, 1] + ny * h / 2) / h - 0.5), 0, ny - 1).astype(int)
    qk = nz // 2
    L2 = lab[qi, qj, qk]
    nline.append(int((np.diff(L2) != 0).sum()))

ngr, nline = np.array(ngr), np.array(nline)
print(f"\ngrains touched by the beam COLUMN in the coda window: "
      f"{ngr.mean():.1f} +- {ngr.std():.1f}  (range {ngr.min()}-{ngr.max()})")
print(f"boundary crossings on the beam AXIS in the coda window: "
      f"{nline.mean():.1f} +- {nline.std():.1f}  (range {nline.min()}-{nline.max()})")

print("\nspeckle floor implied (Rayleigh, 5.57 dB per independent look):")
pulse = C_REF / F0 * 2 / 2                      # 2-cycle gate, one-way depth
ncell = (d1 - d0) / pulse
print(f"  range gates in the window: {ncell:.0f}")
for tag, N in (("axis facets only", nline.mean()),
               ("facets x range-gate diversity", nline.mean()),
               ("+ 3 frequency sub-bands", nline.mean() * 3)):
    print(f"  {tag:<32} N={N:5.1f}  sigma = {5.57/np.sqrt(max(N,1)):5.2f} dB")

# azimuthal decorrelation: how far must the disk turn for a fresh chain?
print("\nazimuthal independence:")
arc = 2 * np.arcsin(ELEM / D) * 180 / np.pi
print(f"  beam width {ELEM*1e3:.2f} mm on a {D*1e3:.0f} mm disc -> "
      f"{arc:.1f} deg per beam width -> {360/arc:.0f} independent "
      f"azimuths max")
print(f"  the 60-azimuth sweep steps 6 deg, so adjacent azimuths overlap "
      f"by {100*(1-6/arc):.0f}%" if arc > 6 else "  6 deg steps are independent")
