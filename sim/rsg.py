"""Minimal 3D rotated-staggered-grid (RSG) elastic solver, CuPy.

WHY. The production solver is a STANDARD staggered grid (SSG), where the
six stress components live at four different half-cell offsets. At a
material interface that is not grid-aligned, each component therefore
samples a different mixture of the two media, and the material parameters
must be interpolated. Bohlen & Saenger (2006, Geophysics 71:T109) show
this is the dominant error at dipping interfaces.

The rotated staggered grid (Saenger, Gold & Shapiro 2000) puts ALL stress
components at one point and ALL velocity components at another, and takes
derivatives along the four body diagonals of the cell. No material
interpolation is required anywhere, so an arbitrary anisotropic stiffness
can be applied at a single point. That is exactly the property the
staircase problem needs.

OPERATOR. With the four body diagonals a_m in {(1,1,1),(1,1,-1),(1,-1,1),
(1,-1,-1)} and the SAME staggered coefficients c_n the SSG uses,

    D_m f = (1/h) sum_n c_n [ f(x + (n-1/2) h a_m) - f(x - (n-1/2) h a_m) ]
          ~ a_m . grad f

and because sum_m (a_m)_i (a_m)_j = 4 delta_ij,

    df/dx_i = (1/4) sum_m (a_m)_i D_m f .

Every tap lands exactly on the other sub-grid, so no interpolation.

This is deliberately a plain-array implementation, not a fused kernel:
the testbed domain is small and the point is to measure accuracy, not
speed. If RSG wins here, THEN it is worth rewriting the production kernel.
"""
import numpy as np

DIAG = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]], float)
# Voigt pairs for building the symmetric gradient / divergence
_VP = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def _sh(xp, f, off):
    """f shifted so that out[i] = f[i + off] (zero-padded)."""
    o = np.asarray(off, int)
    out = xp.zeros_like(f)
    src = [slice(max(o[d], 0), f.shape[d] + min(o[d], 0)) for d in range(3)]
    dst = [slice(max(-o[d], 0), f.shape[d] - max(o[d], 0)) for d in range(3)]
    out[tuple(dst)] = f[tuple(src)]
    return out


def _grad(xp, f, c, h, forward):
    """Cartesian gradient of `f` evaluated on the OTHER sub-grid.

    forward=True  : f lives on the coarse grid (index p at p*h), result on
                    the half-offset grid; tap index along an axis with
                    a=+1 is i+n, with a=-1 it is i+1-n.
    forward=False : the reverse map (result on the coarse grid).
    """
    g = [xp.zeros_like(f) for _ in range(3)]
    M = len(c)
    for a in DIAG:
        D = xp.zeros_like(f)
        for n in range(1, M + 1):
            if forward:
                pi = [n if a[d] > 0 else 1 - n for d in range(3)]
                mi = [1 - n if a[d] > 0 else n for d in range(3)]
            else:
                pi = [n - 1 if a[d] > 0 else -n for d in range(3)]
                mi = [-n if a[d] > 0 else n - 1 for d in range(3)]
            D += c[n - 1] * (_sh(xp, f, pi) - _sh(xp, f, mi))
        D /= h
        for d in range(3):
            if a[d] > 0:
                g[d] += 0.25 * D
            else:
                g[d] -= 0.25 * D
    # arrays are indexed (z, y, x), so g[0] is d/dz. Return in (x, y, z)
    # order to match the Voigt convention used by the caller. (The diagonal
    # set is invariant under axis permutation up to a sign that cancels in
    # (a_m)_i D_m, so reversing is legitimate.)
    return [g[2], g[1], g[0]]


def forward_rsg(Cfield, rho, h, dt, nt, src_idx, wavelet, rec_idx,
                coeffs, sponge_width=12, sponge_strength=0.16,
                progress=None):
    """Cfield: (6,6,nz,ny,nx) stiffness AT THE STRESS POINTS. rho: scalar
    or (nz,ny,nx) at the velocity points. Monopole source, pressure rec."""
    import cupy as cp
    xp = cp
    c = np.asarray(coeffs, float)
    C = xp.asarray(np.asarray(Cfield, np.float32))
    shp = C.shape[2:]
    rho_a = xp.asarray(np.broadcast_to(np.asarray(rho, np.float32),
                                       shp).copy())
    v = [xp.zeros(shp, cp.float32) for _ in range(3)]
    s = [xp.zeros(shp, cp.float32) for _ in range(6)]

    # cosine-taper sponge on all six faces
    w = xp.ones(shp, cp.float32)
    if sponge_width > 0:
        ramp = xp.asarray(
            (1.0 - sponge_strength
             * (0.5 * (1 + np.cos(np.pi * np.arange(sponge_width)
                                  / sponge_width)))[::-1]).astype(np.float32))
        for d in range(3):
            sl = [slice(None)] * 3
            sh_ = [1, 1, 1]
            sh_[d] = sponge_width
            r = ramp.reshape(sh_)
            sl[d] = slice(0, sponge_width)
            w[tuple(sl)] *= r
            sl[d] = slice(shp[d] - sponge_width, shp[d])
            w[tuple(sl)] *= r[::-1] if d == 0 else xp.flip(r, axis=d)

    out = np.zeros(nt, np.float64)
    for it in range(nt):
        # --- stress update: sdot = C : sym(grad v),  v -> stress points
        gx = _grad(xp, v[0], c, h, True)
        gy = _grad(xp, v[1], c, h, True)
        gz = _grad(xp, v[2], c, h, True)
        e = [gx[0], gy[1], gz[2],
             gy[2] + gz[1], gx[2] + gz[0], gx[1] + gy[0]]
        for I in range(6):
            acc = C[I, 0] * e[0]
            for J in range(1, 6):
                acc = acc + C[I, J] * e[J]
            s[I] += dt * acc
        s[0] += dt * wavelet[it] * xp.asarray(src_idx)
        s[1] += dt * wavelet[it] * xp.asarray(src_idx)
        s[2] += dt * wavelet[it] * xp.asarray(src_idx)
        for I in range(6):
            s[I] *= w
        # --- velocity update: rho vdot = div sigma, stress -> vel points
        d0 = _grad(xp, s[0], c, h, False)
        d1 = _grad(xp, s[1], c, h, False)
        d2 = _grad(xp, s[2], c, h, False)
        d3 = _grad(xp, s[3], c, h, False)
        d4 = _grad(xp, s[4], c, h, False)
        d5 = _grad(xp, s[5], c, h, False)
        v[0] += dt / rho_a * (d0[0] + d5[1] + d4[2])
        v[1] += dt / rho_a * (d5[0] + d1[1] + d3[2])
        v[2] += dt / rho_a * (d4[0] + d3[1] + d2[2])
        for d in range(3):
            v[d] *= w
        out[it] = float(-(s[0] + s[1] + s[2])[rec_idx].sum() / 3.0)
        if progress is not None and it % 200 == 0:
            progress(it / nt)
    return out


def stable_dt(Cmax, rho_min, h, coeffs, safety=0.5):
    """RSG CFL: the diagonal operator gives sqrt(3) larger effective k."""
    ssum = float(np.abs(np.asarray(coeffs)).sum())
    vmax = float(np.sqrt(Cmax / rho_min))
    return safety * h / (vmax * 2.0 * ssum * np.sqrt(3.0))
