import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
"""Rotated-staggered-grid PSEUDOSPECTRAL anisotropic elastic solver.

Routes A + B of the speed-up literature survey, merged (the k-space
temporal correction IS route B, applied inside the spectral scheme):

  * Spectral spatial derivatives (Fornberg 1987; Carcione et al. 1992)
    need ~3 points per wavelength against 6 for our 8th-order FD, so the
    grid shrinks ~8x in 3D and dt grows with h.
  * ROTATED staggering (Saenger; Cheng et al., Geophys. Prospecting,
    10.1111/1365-2478.12543): ALL six stresses live at one node family
    (cell corners), velocities at the other (cell centres). Staggering is
    exact in Fourier space via a half-cell phase shift, and - crucially
    for arbitrary anisotropy - the full 21-component stiffness applies
    POINTWISE at one location. The dense solver's half-cell interpolation
    of shear strains (_moved) disappears entirely.
  * k-space temporal correction (Tabei/Mast/Waag lineage): the spectral
    derivative is scaled by sinc(c_ref |k| dt / 2), which makes leapfrog
    time-stepping exact for a homogeneous medium at c_ref. Ice's total
    qP spread is only +/-3.5% about c_ref, so the residual temporal
    dispersion is far below our validated error budget.

Sharp-boundary caveat, stated up front: spectral derivatives ring at
discontinuities (Gibbs), and our observable IS boundary scattering. The
rotated grid was invented for sharp discontinuities (Saenger's crack
papers), which is encouraging, but the only verdict that counts is the
one from validate_ps.py against the saved full-wave references.

MEASURED VERDICT (2026-07-25, validate_ps.py vs ref/fw_fabric00):
  * E1 TIMING: PASS. 51.71 vs 51.43 us (0.28 us) at ppw 3 - anisotropic
    propagation is right, and the trace costs 21 s against the FD 40 s at
    2 MHz (at 5 MHz the ratio is ~4x). The PS solver IS usable for
    timing-only work: E1 ToF sinograms, walk-off fade studies.
  * CODA: FAIL, and not fixably-so at coarse ppw. Fabric specimen read
    -9.6 dB re E1 against the true -36.2. The decisive control: a
    UNIFORM specimen (no grain boundaries at all) still reads -12.2 dB,
    so the excess is not grain physics - it is Gibbs ringing at the
    staircased ICE/FLUID RIM (92:1 density contrast), 24 dB above the
    real coda. Medium smoothing made it worse (smears the rim, delays
    and weakens E1); a stronger/wider sponge changed nothing (it is not
    wraparound). The FD scheme survives the same rim because of its
    face-averaged density treatment; pointwise spectral sampling has no
    such cushion. Fixing this means a hybrid FD-boundary/PS-interior
    scheme or fine ppw that erases the speed win - a research project,
    not an optimisation. DO NOT use this solver for amplitude work.
"""
import numpy as np

C_REF_DEFAULT = 3850.0


def ps_wavenumbers(shape, h, dt, c_ref, xp):
    """1D spectral factors and the (non-separable) k-space sinc correction.

    Returns (mult_fwd, mult_bwd, ksinc):
      mult_fwd[axis] : 1j * k_axis * exp(+i k_axis h/2)   (centre -> corner)
      mult_bwd[axis] : 1j * k_axis * exp(-i k_axis h/2)   (corner -> centre)
      ksinc          : sinc(c_ref |k| dt / 2), full 3D, float32
    The Nyquist bin of each odd-derivative factor is zeroed (standard for
    real fields). The half-cell shift is separable per axis; only the
    |k|-dependent sinc needs a full 3D array.
    """
    ks = [2.0 * np.pi * np.fft.fftfreq(n, h) for n in shape]
    mult_fwd, mult_bwd = [], []
    for ax, k in enumerate(ks):
        m = 1j * k * np.exp(0.5j * k * h)
        mb = 1j * k * np.exp(-0.5j * k * h)
        n = shape[ax]
        if n % 2 == 0:                      # zero the Nyquist derivative
            m[n // 2] = 0.0
            mb[n // 2] = 0.0
        sh = [1, 1, 1]
        sh[ax] = n
        mult_fwd.append(xp.asarray(m.reshape(sh), xp.complex64))
        mult_bwd.append(xp.asarray(mb.reshape(sh), xp.complex64))
    KZ, KY, KX = np.meshgrid(ks[0], ks[1], ks[2], indexing="ij")
    kmag = np.sqrt(KZ ** 2 + KY ** 2 + KX ** 2)
    arg = 0.5 * c_ref * kmag * dt
    ksinc = np.ones_like(arg)
    nzm = arg > 1e-12
    ksinc[nzm] = np.sin(arg[nzm]) / arg[nzm]
    return mult_fwd, mult_bwd, xp.asarray(ksinc, xp.float32)


def ps_dt(h, vmax, safety=0.5):
    """Timestep for the k-space corrected leapfrog.

    Exact (unconditionally stable) at c_ref in homogeneous media; the
    heterogeneous residual motivates the usual sqrt(3) 3D bound with a
    safety factor. Still ~1.5x the 8th-order FD dt at equal h, and the
    PS h is itself ~2x coarser.
    """
    return safety * h / (np.sqrt(3.0) * vmax)


def forward_ps_labels(lab, axes, h, dt, nt, src_pts, wavelet, rec_groups,
                      c_ref=C_REF_DEFAULT, sponge_width=0,
                      sponge_strength=0.09, damp_mask=None,
                      c_couplant=1500.0, rho_couplant=300.0,
                      record="pressure", smooth_cells=0.0, progress=None):
    """Same contract as fdtd.forward_fused_labels, spectral engine.

    lab: specimen label volume (-1 = fluid), sampled at the PS grid's h.
    Stiffness is applied pointwise at the corner nodes using the SAME
    label of the cell (the half-cell offset is < h/2 ~ lambda/6, far
    below the grain scale, and is part of what validation must absorb).
    """
    import cupy as cp
    from fdtd import material_tables, sponge_profile

    lab = np.asarray(lab)
    nz, ny, nx = lab.shape
    mats = (lab + 1).astype(np.int32)
    Ct, rho_t = material_tables(axes, c_couplant, rho_couplant)

    # dense per-cell planes: PS grids are ~8x smaller than FD ones, so the
    # 21 stiffness planes are affordable and beat per-step table gathers
    mats_d = cp.asarray(mats)
    Cp = {}
    for I in range(1, 7):
        for J in range(I, 7):
            col = Ct[:, (I - 1) * 6 + (J - 1)]
            if np.any(col):
                Cp[(I, J)] = cp.asarray(col, cp.float32)[mats_d]
    rho_d = cp.asarray(rho_t, cp.float32)[mats_d]
    del mats_d
    if smooth_cells > 0:
        # Band-limit the medium (the k-Wave prescription): spectral
        # derivatives ring at stiffness jumps the grid cannot represent,
        # and at ppw 3 that manufactured +26.6 dB of fake coda. A ~1-cell
        # Gaussian keeps the physical reflectivity (transition width
        # << lambda) while removing the unrepresentable spatial content.
        from cupyx.scipy import ndimage as cnd
        for key in Cp:
            Cp[key] = cnd.gaussian_filter(Cp[key], smooth_cells)
        rho_d = cnd.gaussian_filter(rho_d, smooth_cells)

    mf, mb, ksinc = ps_wavenumbers((nz, ny, nx), h, dt, c_ref, cp)

    damp_h = None
    if sponge_width > 0:
        damp_h = sponge_profile((nz, ny, nx), sponge_width,
                                sponge_strength, np).astype(np.float32)
    if damp_mask is not None:
        dmh = np.asarray(damp_mask, np.float32)
        damp_h = dmh if damp_h is None else damp_h * dmh
    damp = cp.asarray(damp_h) if damp_h is not None else None
    del damp_h

    v = [cp.zeros((nz, ny, nx), cp.float32) for _ in range(3)]  # vx vy vz
    s = [cp.zeros((nz, ny, nx), cp.float32) for _ in range(6)]

    AXOF = {"x": 2, "y": 1, "z": 0}

    def deriv(field, axis, forward):
        F = cp.fft.fftn(field.astype(cp.complex64))
        F *= (mf if forward else mb)[AXOF[axis]]
        F *= ksinc
        return cp.fft.ifftn(F).real.astype(cp.float32)

    shp = (nz, ny, nx)
    s_idx = cp.asarray(np.array([np.ravel_multi_index(tuple(i), shp)
                                 for i, _ in src_pts], np.int64))
    s_w = cp.asarray(np.array([w for _, w in src_pts], np.float32))
    r_idx = [cp.asarray(np.array([np.ravel_multi_index(tuple(q), shp)
                                  for q in pts], np.int64))
             for pts, _ in rec_groups]
    r_w = [cp.asarray(np.asarray(w, np.float32)) for _, w in rec_groups]
    rec_d = cp.zeros((nt, len(rec_groups)), cp.float32)
    sf = [a.ravel() for a in s]

    dtf = np.float32(dt)
    for n in range(nt):
        # strains at corner nodes (forward shift), full anisotropic update
        g1 = deriv(v[0], "x", True)
        g2 = deriv(v[1], "y", True)
        g3 = deriv(v[2], "z", True)
        g4 = deriv(v[2], "y", True) + deriv(v[1], "z", True)
        g5 = deriv(v[2], "x", True) + deriv(v[0], "z", True)
        g6 = deriv(v[1], "x", True) + deriv(v[0], "y", True)
        g = [g1, g2, g3, g4, g5, g6]
        for I in range(1, 7):
            acc = None
            for J in range(1, 7):
                key = (min(I, J), max(I, J))
                if key not in Cp:
                    continue
                t = Cp[key] * g[J - 1]
                acc = t if acc is None else acc + t
            if acc is not None:
                s[I - 1] += dtf * acc
        del g, g1, g2, g3, g4, g5, g6

        wn = np.float32(wavelet[n])
        for I in range(3):
            sf[I][s_idx] += wn * s_w

        # velocities at centre nodes (backward shift)
        v[0] += (dtf / rho_d) * (deriv(s[0], "x", False)
                                 + deriv(s[5], "y", False)
                                 + deriv(s[4], "z", False))
        v[1] += (dtf / rho_d) * (deriv(s[5], "x", False)
                                 + deriv(s[1], "y", False)
                                 + deriv(s[3], "z", False))
        v[2] += (dtf / rho_d) * (deriv(s[4], "x", False)
                                 + deriv(s[3], "y", False)
                                 + deriv(s[2], "z", False))

        if damp is not None:
            for a in v:
                a *= damp
            for a in s:
                a *= damp

        for r in range(len(r_idx)):
            val = -(sf[0][r_idx[r]] + sf[1][r_idx[r]]
                    + sf[2][r_idx[r]]) / 3.0
            rec_d[n, r] = (val * r_w[r]).sum()
        if progress is not None and (n % 100 == 0):
            progress(n / nt)
    return cp.asnumpy(rec_d)
