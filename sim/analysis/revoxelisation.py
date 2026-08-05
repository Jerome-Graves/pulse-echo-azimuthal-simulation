"""Does refining the grid change the microstructure it represents?

Supports the convergence argument of Section 4. That argument compares
the scattered field at three grid spacings and attributes the change to
discretisation error. The comparison is only valid if the three grids
represent the SAME microstructure. They do not represent it identically:
re-voxelising a tessellation at a finer spacing changes which cells fall
either side of each boundary, so the discrete boundary-face count and the
distribution of facet normals both move between rungs of the ladder. If
that movement is large it is a confound, because the specimen would then
differ between resolutions and not merely its discretisation.

This quantifies it directly from the label arrays, with no simulation.
Two quantities are measured at each resolution:

  discrete boundary area   the number of faces between voxels carrying
                           different grain labels, times h squared. The
                           continuum limit is the true interfacial area,
                           so the trend with h shows how far from that
                           limit each rung sits.
  normal-direction mix     the fraction of those faces whose normal lies
                           along x, y and z. A staircased plane trades
                           area between the three axes as it is re-sampled,
                           so a drift here means the facets present a
                           different aspect to the beam.

Reads:  nothing. The specimen is rebuilt on the CPU from its seed, which
        is cheap at these grid spacings and needs no GPU.
Writes: ../results/revoxelisation.log and .npz
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('..',):
    _p = os.path.normpath(os.path.join(_HERE, _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)
        sys.path[:0] = [os.path.join(sys.path[0], _d)
                        for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))

from specimen import DiskSpecimen                          # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))

# Specimen of Section 3, and the three rungs of the convergence ladder.
DIA, THK, N_GRAINS, SIZE_CV = 0.100, 0.035, 100, 0.35
KAPPA, AXIS, SEED = -8.0, (1.0, 0.0, 0.0), 11
C_REF, F0 = 3850.0, 2.0e6
PPWS = (6.0, 8.0, 10.0)


def label_volume(ppw):
    """Grain-label array at the given points per wavelength.

    Built from the seed rather than loaded, so this is reproducible from
    the repository alone and does not depend on any saved sweep.
    """
    h = C_REF / F0 / ppw
    build = DiskSpecimen(diameter_m=DIA, thickness_m=THK,
                         n_grains=N_GRAINS, size_cv=SIZE_CV,
                         concentration=KAPPA, spatial_corr=0.0,
                         fabric_axis=AXIS, seed=SEED).build(h)
    # build() returns a dict; 'labels' is the grain-label array and
    # cells outside the disc carry a negative label.
    return np.asarray(build['labels']), h


def boundary_faces(lab):
    """Faces between differing labels, counted per axis.

    Only interior faces are counted: a face against the surrounding
    couplant, which carries label < 0, is a specimen boundary and not a
    grain boundary, and including it would swamp the quantity of interest.
    """
    counts = []
    for axis in range(3):
        a = np.take(lab, np.arange(lab.shape[axis] - 1), axis=axis)
        b = np.take(lab, np.arange(1, lab.shape[axis]), axis=axis)
        interior = (a >= 0) & (b >= 0)
        counts.append(int(np.count_nonzero((a != b) & interior)))
    return counts


def main():
    os.makedirs(RESULTS, exist_ok=True)
    log = os.path.join(RESULTS, 'revoxelisation.log')
    rows = []

    def note(msg):
        print(msg, flush=True)
        with open(log, 'a', encoding='utf-8') as fh:
            fh.write(msg + '\n')

    note('=' * 68)
    note('Re-voxelisation of the seed %d girdle specimen' % SEED)
    note('%-6s %8s %12s %14s %22s'
         % ('ppw', 'h (mm)', 'faces', 'area (cm^2)', 'normal mix x:y:z'))

    for ppw in PPWS:
        t0 = time.time()
        lab, h = label_volume(ppw)
        nx, ny, nz = boundary_faces(lab)
        total = nx + ny + nz
        area_cm2 = total * (h * 1e2) ** 2
        mix = np.array([nx, ny, nz], float) / total
        rows.append((ppw, h, total, area_cm2, mix))
        note('%-6.0f %8.3f %12d %14.1f   %5.3f %5.3f %5.3f   (%.0f s)'
             % (ppw, h * 1e3, total, area_cm2, mix[0], mix[1], mix[2],
                time.time() - t0))

    note('')
    base = rows[-1]
    note('Referred to the finest grid, %s ppw:' % int(base[0]))
    for ppw, _h, _n, area, mix in rows:
        d_area = 20.0 * np.log10(area / base[3])
        d_mix = float(np.max(np.abs(mix - base[4])))
        note('  ppw %-3.0f  area %+6.2f dB   largest normal-mix shift %+.4f'
             % (ppw, d_area, d_mix))

    note('')
    note('The scattered field changes by 7.98 dB between ppw 6 and 8 and')
    note('2.21 dB between 8 and 10 (Table 4). Compare those with the area')
    note('column above: if the represented boundary area moves by a small')
    note('fraction of a decibel the ladder is comparing one specimen at')
    note('three resolutions, which is what the convergence claim assumes.')

    np.savez(os.path.join(RESULTS, 'revoxelisation.npz'),
             ppw=np.array([r[0] for r in rows]),
             faces=np.array([r[2] for r in rows]),
             area_cm2=np.array([r[3] for r in rows]),
             normal_mix=np.array([r[4] for r in rows]))


if __name__ == '__main__':
    main()
