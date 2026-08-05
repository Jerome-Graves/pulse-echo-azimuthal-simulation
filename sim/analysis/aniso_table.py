"""Single-crystal elastic anisotropy of ice against common NDE materials.

The physically correct comparator for Born-approximation scattering power
is the normalised standard deviation of the longitudinal modulus
C1111 = c_ijkl n_i n_j n_k n_l over uniformly random crystal
orientations, because scattering power goes as its square.

Ice constants are taken from the project's own material module so that
they are consistent with the simulations. The metal constants are
standard literature values and are flagged for verification.
"""
import os
import sys

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
from ringfwi import anisotropy as an   # noqa: E402


def voigt_to_c(M):
    """6x6 Voigt -> 3x3x3x3."""
    idx = {(0, 0): 0, (1, 1): 1, (2, 2): 2,
           (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4,
           (0, 1): 5, (1, 0): 5}
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = M[idx[(i, j)], idx[(k, l)]]
    return C


def cubic(c11, c12, c44):
    M = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            M[i, j] = c12
        M[i, i] = c11
        M[i + 3, i + 3] = c44
    return M


def hexagonal(c11, c33, c44, c12, c13):
    M = np.zeros((6, 6))
    M[0, 0] = M[1, 1] = c11
    M[2, 2] = c33
    M[0, 1] = M[1, 0] = c12
    M[0, 2] = M[2, 0] = M[1, 2] = M[2, 1] = c13
    M[3, 3] = M[4, 4] = c44
    M[5, 5] = 0.5 * (c11 - c12)
    return M


def stats(M, n=200000, seed=0):
    """mean and sd of C1111 over uniformly random directions."""
    C = voigt_to_c(M)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    L = np.einsum('ijkl,ai,aj,ak,al->a', C, v, v, v, v)
    return L.mean(), L.std()


# ice, from the project's own module, so it matches the simulations
ICE = an.ti_stiffness_6(**an.ICE_MATERIAL) / 1e9
MATS = [
    ('Ice Ih (hexagonal)', ICE, 'project module, after Gammon et al.'),
    ('alpha-Ti (hexagonal)', hexagonal(162.4, 180.7, 46.7, 92.0, 69.0), 'VERIFY'),
    ('Al (cubic)', cubic(108.2, 61.3, 28.5), 'VERIFY'),
    ('Cu (cubic)', cubic(168.4, 121.4, 75.4), 'VERIFY'),
    ('Ni (cubic)', cubic(246.5, 147.3, 124.7), 'VERIFY'),
    ('alpha-Fe (cubic)', cubic(231.4, 134.7, 116.4), 'VERIFY'),
    ('Austenitic steel 316', cubic(206.0, 133.0, 119.0), 'VERIFY'),
]

print('%-22s %10s %10s %10s   %s'
      % ('material', 'mean(GPa)', 'sd/mean %', 'vs ice dB', 'source'))
ref = None
rows = []
for name, M, src in MATS:
    m, s = stats(M)
    frac = 100.0 * s / m
    if ref is None:
        ref = frac
    db = 20.0 * np.log10(frac / ref)
    rows.append((name, m, frac, db, src))
    print('%-22s %10.1f %10.2f %+10.1f   %s' % (name, m, frac, db, src))

print()
print('LaTeX rows:')
for name, m, frac, db, src in rows:
    tag = '' if src == 'VERIFY' else ''
    print('%s & %.2f & %+.1f \\\\' % (name.replace('alpha-', r'$\alpha$-'),
                                      frac, db))
