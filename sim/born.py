"""Born single-scatter forward model: the next rung above the ray model.

WHY THE RAY MODEL FAILED (measured, not assumed):
  * coda 14 dB too low, 6 dB scatter
  * sensitivity WRONG SIGN at 25% fabric randomisation (full-wave -4.8 dB,
    ray +5.2 dB)
It counts only SPECULAR returns. A tilted grain boundary reflects away from
the transducer, so the ray model records almost nothing from it, yet 92% of
boundaries are struck obliquely. The missing energy is non-specular
backscatter, and that is exactly the channel that responds to misorientation,
which is why calibration cannot rescue it.

THE FIX IN PRINCIPLE:
Stop treating grains as mirrors. In Born theory a polycrystal backscatters
because the stiffness field has spatial fluctuations; a volume element with
impedance deviation dZ re-radiates in ALL directions, and the transducer sees
the component at 2k (backscatter). No specular condition appears anywhere, so
obliquely-oriented boundaries contribute naturally.

This is the standard grain-noise model in ultrasonic NDE. The backscattered
signal at two-way time t is the coherent sum over the shell at depth ct/2:

    A(t) = SUM_cells  (dZ/Z) * B(r)^2 / r^2 * exp(2 i k r)

with B the beam amplitude and 1/r^2 two-way spreading. Because the phases are
essentially random the envelope follows the RMS, which is what the coda
measurement compares against.
"""
import numpy as np

RHO = 917.0


def sh_speeds(axes, direction, C44=3.014e9, C66=(13.929e9 - 7.082e9) / 2.0):
    """SH shear speed of every grain for one propagation direction.

    Transversely isotropic ice: v_SH^2 = (C66 sin^2 psi + C44 cos^2 psi)/rho
    with psi the angle between propagation and the c-axis. Used to scale the
    P->S conversion channel by the SHEAR contrast across a face, which is
    the honest driver of conversion (the toy version used the P contrast).
    """
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    cos2 = np.clip(np.abs(np.asarray(axes, float) @ d), 0.0, 1.0) ** 2
    return np.sqrt((C66 * (1.0 - cos2) + C44 * cos2) / RHO)


def impedance_field(labels, axes, direction, c_ref=3850.0):
    """Per-cell acoustic impedance for propagation along `direction`.

    Uses the same qP(psi) table as the ray model, so any difference between
    the two models is the SCATTERING PHYSICS, not the velocity model.
    """
    import forward as F
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    v_grain = F.grain_speeds(np.asarray(axes, float), d)      # (n_grains,)
    Z = RHO * v_grain
    out = np.zeros(labels.shape, np.float32)
    inside = labels >= 0
    out[inside] = Z[labels[inside]]
    return out, inside


def simulate(build, cfg, azimuth_rad=0.0, f0=None, fs=None, n_t=None,
             beam_half_angle=None, c_ref=3850.0, coherent=True):
    """Born backscatter A-scan. Returns (t, trace).

    Geometry matches the ray model exactly: probe on the rim at `azimuth_rad`
    looking along the diameter, same element size, same pulse.
    """
    import forward as F
    labels, axes, h = build["labels"], build["axes"], build["h"]
    p = cfg.probe
    f0 = f0 or p.f0
    fs = fs or cfg.acq.fs
    n_t = n_t or int(cfg.acq.record_end_s * fs)
    R = cfg.specimen.diameter_m / 2.0
    lam = c_ref / f0
    k = 2 * np.pi / lam

    nx, ny, nz = labels.shape
    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    origin = np.array([R * ca, R * sa, 0.0])
    dhat = np.array([-ca, -sa, 0.0])

    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij")
    X = (ix - (nx - 1) / 2.0) * h
    Y = (iy - (ny - 1) / 2.0) * h
    Z3 = (iz - (nz - 1) / 2.0) * h
    rx, ry, rz = X - origin[0], Y - origin[1], Z3 - origin[2]
    r = np.sqrt(rx * rx + ry * ry + rz * rz)
    r = np.maximum(r, h)

    # angle off the beam axis, and the piston directivity there
    cosang = (rx * dhat[0] + ry * dhat[1] + rz * dhat[2]) / r
    Zf, inside = impedance_field(labels, axes, dhat, c_ref)

    ang = np.arccos(np.clip(cosang, -1.0, 1.0))
    a_el = p.element_d_m / 2.0
    xarg = k * a_el * np.sin(ang)
    from scipy.special import j1
    B = np.ones_like(xarg)
    nzm = np.abs(xarg) > 1e-9
    B[nzm] = 2.0 * j1(xarg[nzm]) / xarg[nzm]
    B = np.where(cosang > 0, B, 0.0)          # nothing behind the probe

    # IMPEDANCE FLUCTUATION is the scattering source. Deviation from the local
    # mean, not from a global constant: a uniformly fast specimen does not
    # backscatter, only variation does. This is what makes run B (all c-axes
    # identical) produce no coda, exactly as in the full-wave A-B experiment.
    Zm = Zf[inside].mean() if inside.any() else 1.0
    dZ = np.where(inside, (Zf - Zm) / Zm, 0.0).astype(np.float64)

    # two-way amplitude weight for each cell
    wgt = dZ * (B ** 2) / (r ** 2)
    wgt = np.where(inside, wgt, 0.0)

    t_cell = 2.0 * r / c_ref
    bins = np.rint(t_cell * fs).astype(np.int64)
    ok = (bins >= 0) & (bins < n_t) & inside & (np.abs(wgt) > 0)

    series = np.zeros(n_t)
    if coherent:
        # random-but-deterministic phase from the exact path length; this is
        # what makes the sum incoherent in practice and gives a speckle-like
        # envelope rather than a single spike
        ph = np.cos(2.0 * k * r)
        np.add.at(series, bins[ok], (wgt * ph)[ok])
    else:
        np.add.at(series, bins[ok], np.abs(wgt[ok]))

    _, pulse = F.gaussian_pulse(f0, p.frac_bw, fs)
    trace = np.convolve(series, pulse, mode="same")

    # far-rim echo, same treatment as the ray model so E1 is comparable
    t_e1 = 2.0 * (2 * R) / c_ref
    kk = int(round(t_e1 * fs))
    if 0 <= kk < n_t:
        imp = np.zeros(n_t)
        imp[kk] = cfg.err.rim_reflection
        trace = trace + np.convolve(imp, pulse, mode="same")
    return np.arange(n_t) / fs, trace


# ══════════════════ rung 3: incoherent BOUNDARY scattering ═══════════════
"""Why the volume-Born attempt above failed, and what replaces it.

MEASURED: coda +6 to +24 dB, i.e. LOUDER than E1. Impossible.

DIAGNOSIS: `simulate` sums every CELL as an independent scatterer, so the
total grows as sqrt(number of cells) and the answer depends on the grid
spacing. That is unphysical. Cells inside a single grain share an impedance
and do not scatter at all; only variation on the scale of half a wavelength
does. A uniform region with dZ != 0 changes phase velocity, it does not
backscatter.

There is also a regime error. Volume-Born (Rayleigh) applies when grains are
SMALL against the wavelength. Here grains are 8 to 28 wavelengths, which is
the GEOMETRIC regime, where scattering happens at grain BOUNDARIES.

RUNG 3 therefore sums boundary AREA, not volume, and adds POWER incoherently:

    env(t)^2 = SUM_boundary_faces  R^2 * B(r)^4 / r^4 * dA

R is the true impedance-contrast coefficient for the two grains meeting at
that face. Crucially there is NO specular condition: every boundary face
contributes according to its area, whatever its tilt. That is precisely the
non-specular channel the ray model lacks, and it is grid-independent because
boundary area converges as the mesh refines.
"""


def boundary_scatter(build, cfg, azimuth_rad=0.0, f0=None, fs=None, n_t=None,
                     c_ref=3850.0, gain=1.0, unit_contrast=False,
                     face_weight="l1"):
    """Incoherent boundary-scattering envelope. Returns (t, envelope).

    unit_contrast=True replaces every face's R with 1, giving the purely
    GEOMETRIC envelope (boundary area x beam x spreading). Because the grain
    axes are iid, the expected physical envelope factorises as
    env = geometric_env * sqrt(E[R^2]), with E[R^2] an ODF expectation
    computable by quadrature - this is what makes the iterative fabric fit
    (fit_fabric.py) smooth and realisation-noise-free.

    face_weight: 'l1' (default) counts every label-change face as area h^2,
    which for a wall of unit normal n converges to the L1-projected area
    (|n1|+|n2|+|n3|) * A_true, inflated by 1..sqrt(3) depending on wall
    orientation. 'normal' weights each axis-ax face by |n . e_ax| (n from the
    Laguerre seed pair in build['seeds'], the exact wall normal), so the
    summed weights converge to sum n_i^2 = 1 times A_true: exact area,
    mesh-orientation independent. MEASURED on the 2 MHz standard specimen
    (mesh rotated 45 deg vs not, unit_contrast RMS in 24-36 us): the L1
    inflation is azimuth-flat to 0.014 dB (4-fold, i.e. absorbed by the
    fit's free gain), so 'l1' is not injecting a mesh-locked azimuthal
    signature; 'normal' exists to make the absolute level mesh-independent
    (the 45-deg mesh reads +0.40 dB under 'l1').
    """
    from scipy.special import j1
    labels, axes, h = build["labels"], build["axes"], build["h"]
    p = cfg.probe
    f0 = f0 or p.f0
    fs = fs or cfg.acq.fs
    n_t = n_t or int(cfg.acq.record_end_s * fs)
    R = cfg.specimen.diameter_m / 2.0
    k = 2 * np.pi * f0 / c_ref

    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    origin = np.array([R * ca, R * sa, 0.0])
    dhat = np.array([-ca, -sa, 0.0])
    Zf, inside = impedance_field(labels, axes, dhat, c_ref)

    nx, ny, nz = labels.shape
    acc = np.zeros(n_t)
    a_el = p.element_d_m / 2.0
    # walk the three face directions; each interior face where the label
    # changes is a boundary element of area h^2
    for ax in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1); s2[ax] = slice(1, None)
        s1, s2 = tuple(s1), tuple(s2)
        la, lb = labels[s1], labels[s2]
        face = (la != lb) & (la >= 0) & (lb >= 0)
        if not face.any():
            continue
        Za, Zb = Zf[s1][face], Zf[s2][face]
        Rc = (Zb - Za) / (Zb + Za)
        if unit_contrast:
            Rc = np.ones_like(Rc)
        idx = np.argwhere(face)
        # face centre in physical coordinates
        cc = idx.astype(float)
        cc[:, ax] += 0.5
        Xc = (cc[:, 0] - (nx - 1) / 2.0) * h
        Yc = (cc[:, 1] - (ny - 1) / 2.0) * h
        Zc = (cc[:, 2] - (nz - 1) / 2.0) * h
        rx, ry, rz = Xc - origin[0], Yc - origin[1], Zc - origin[2]
        r = np.sqrt(rx * rx + ry * ry + rz * rz)
        r = np.maximum(r, h)
        cosang = (rx * dhat[0] + ry * dhat[1] + rz * dhat[2]) / r
        ang = np.arccos(np.clip(cosang, -1.0, 1.0))
        xarg = k * a_el * np.sin(ang)
        B = np.ones_like(xarg)
        m = np.abs(xarg) > 1e-9
        B[m] = 2.0 * j1(xarg[m]) / xarg[m]
        B = np.where(cosang > 0, B, 0.0)
        # POWER contribution: R^2 x area x two-way beam^4 / r^4
        pw = (Rc ** 2) * (h * h) * (B ** 4) / (r ** 4)
        if face_weight == "normal":
            # exact-area correction: a wall of unit normal n crosses this
            # axis with face density |n.e_ax|, so weighting each face by
            # |n.e_ax| makes the three axis passes sum to n1^2+n2^2+n3^2 = 1
            # times the true area instead of |n1|+|n2|+|n3| times it.
            seeds = build["seeds"]
            ga = la[face].astype(np.int64)
            gb = lb[face].astype(np.int64)
            nrm = seeds[gb] - seeds[ga]
            nn = np.linalg.norm(nrm, axis=1)
            good = nn > 1e-12
            wA = np.ones(len(nn))
            wA[good] = np.abs(nrm[good, ax]) / nn[good]
            pw = pw * wA
        tb = np.rint((2.0 * r / c_ref) * fs).astype(np.int64)
        ok = (tb >= 0) & (tb < n_t) & (pw > 0)
        np.add.at(acc, tb[ok], pw[ok])

    env = gain * np.sqrt(acc)
    # smooth over the pulse duration, since power adds within a resolution cell
    npul = max(int(fs / (p.frac_bw * f0)), 1)
    if npul > 1:
        env = np.convolve(env, np.ones(npul) / npul, mode="same")
    return np.arange(n_t) / fs, env


def boundary_coherent(build, cfg, azimuth_rad=0.0, f0=None, fs=None, n_t=None,
                      c_ref=3850.0, gain=1.0, obliquity=0.0):
    """Rung 4: boundary faces summed WITH phase, so they interfere.

    Rung 3 summed power incoherently and got the sign right everywhere but
    under-responded (full-wave -15.3 dB at full randomisation, rung 3 -8.9).
    An incoherent sum cannot reproduce a coda that FALLS as the fabric
    randomises, because that decrease is destructive interference between
    faces inside one resolution cell. Restoring exp(2ikr) lets that happen.

    `obliquity` blends in a tilt weight: 0 = pure area (rung 3 behaviour,
    every face equal), 1 = full |n.d| specular weighting (ray-model
    behaviour). The truth is in between, and sweeping it locates where.
    """
    from scipy.special import j1
    labels, axes, seeds, h = (build["labels"], build["axes"], build["seeds"],
                              build["h"])
    p = cfg.probe
    f0 = f0 or p.f0
    fs = fs or cfg.acq.fs
    n_t = n_t or int(cfg.acq.record_end_s * fs)
    R = cfg.specimen.diameter_m / 2.0
    k = 2 * np.pi * f0 / c_ref

    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    origin = np.array([R * ca, R * sa, 0.0])
    dhat = np.array([-ca, -sa, 0.0])
    Zf, inside = impedance_field(labels, axes, dhat, c_ref)
    nx, ny, nz = labels.shape
    a_el = p.element_d_m / 2.0
    re = np.zeros(n_t)
    im = np.zeros(n_t)

    for ax in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1); s2[ax] = slice(1, None)
        s1, s2 = tuple(s1), tuple(s2)
        la, lb = labels[s1], labels[s2]
        face = (la != lb) & (la >= 0) & (lb >= 0)
        if not face.any():
            continue
        Za, Zb = Zf[s1][face], Zf[s2][face]
        Rc = (Zb - Za) / (Zb + Za)
        idx = np.argwhere(face)
        cc = idx.astype(float); cc[:, ax] += 0.5
        Xc = (cc[:, 0] - (nx - 1) / 2.0) * h
        Yc = (cc[:, 1] - (ny - 1) / 2.0) * h
        Zc = (cc[:, 2] - (nz - 1) / 2.0) * h
        rx, ry, rz = Xc - origin[0], Yc - origin[1], Zc - origin[2]
        r = np.maximum(np.sqrt(rx * rx + ry * ry + rz * rz), h)
        cosang = (rx * dhat[0] + ry * dhat[1] + rz * dhat[2]) / r
        ang = np.arccos(np.clip(cosang, -1.0, 1.0))
        xarg = k * a_el * np.sin(ang)
        B = np.ones_like(xarg); m = np.abs(xarg) > 1e-9
        B[m] = 2.0 * j1(xarg[m]) / xarg[m]
        B = np.where(cosang > 0, B, 0.0)

        wt = np.ones_like(Rc)
        if obliquity > 0:
            ga = labels[s1][face]; gb = labels[s2][face]
            nrm = seeds[gb] - seeds[ga]
            nn = np.linalg.norm(nrm, axis=1)
            good = nn > 1e-12
            ob = np.ones_like(Rc)
            ob[good] = np.abs(nrm[good] @ dhat) / nn[good]
            wt = (1.0 - obliquity) + obliquity * ob

        amp = Rc * wt * (h * h) * (B ** 2) / (r ** 2)
        ph = 2.0 * k * r
        tb = np.rint((2.0 * r / c_ref) * fs).astype(np.int64)
        ok = (tb >= 0) & (tb < n_t)
        np.add.at(re, tb[ok], (amp * np.cos(ph))[ok])
        np.add.at(im, tb[ok], (amp * np.sin(ph))[ok])

    env = gain * np.sqrt(re ** 2 + im ** 2)
    npul = max(int(fs / (p.frac_bw * f0)), 1)
    if npul > 1:
        env = np.convolve(env, np.ones(npul) / npul, mode="same")
    return np.arange(n_t) / fs, env


def boundary_signed(build, cfg, azimuth_rad=0.0, f0=None, fs=None, n_t=None,
                    c_ref=3850.0, gain=1.0, obliquity=0.0):
    """Rung 5: signed boundary sum, phase applied ONCE via the pulse.

    Rung 4 gave -98 dB, seventy decibels low, because it applied the phase
    twice: the time-bin position already carries the propagation phase once
    the series is convolved with an oscillating pulse, and multiplying by
    cos(2kr) on top made faces with opposite-signed R cancel almost
    completely.

    Here the SIGNED reflection coefficients are binned by two-way time and the
    pulse supplies the phase, exactly as in the ray model. The difference from
    the ray model is that every face contributes according to its AREA with no
    specular condition, which is the non-specular channel that matters.
    """
    import forward as F
    from scipy.special import j1
    labels, axes, seeds, h = (build["labels"], build["axes"], build["seeds"],
                              build["h"])
    p = cfg.probe
    f0 = f0 or p.f0
    fs = fs or cfg.acq.fs
    n_t = n_t or int(cfg.acq.record_end_s * fs)
    R = cfg.specimen.diameter_m / 2.0
    k = 2 * np.pi * f0 / c_ref
    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    origin = np.array([R * ca, R * sa, 0.0])
    dhat = np.array([-ca, -sa, 0.0])
    Zf, inside = impedance_field(labels, axes, dhat, c_ref)
    nx, ny, nz = labels.shape
    a_el = p.element_d_m / 2.0
    series = np.zeros(n_t)

    for ax in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1); s2[ax] = slice(1, None)
        s1, s2 = tuple(s1), tuple(s2)
        la, lb = labels[s1], labels[s2]
        face = (la != lb) & (la >= 0) & (lb >= 0)
        if not face.any():
            continue
        Za, Zb = Zf[s1][face], Zf[s2][face]
        Rc = (Zb - Za) / (Zb + Za)
        idx = np.argwhere(face)
        cc = idx.astype(float); cc[:, ax] += 0.5
        Xc = (cc[:, 0] - (nx - 1) / 2.0) * h
        Yc = (cc[:, 1] - (ny - 1) / 2.0) * h
        Zc = (cc[:, 2] - (nz - 1) / 2.0) * h
        rx, ry, rz = Xc - origin[0], Yc - origin[1], Zc - origin[2]
        r = np.maximum(np.sqrt(rx * rx + ry * ry + rz * rz), h)
        cosang = (rx * dhat[0] + ry * dhat[1] + rz * dhat[2]) / r
        ang = np.arccos(np.clip(cosang, -1.0, 1.0))
        xarg = k * a_el * np.sin(ang)
        B = np.ones_like(xarg); m = np.abs(xarg) > 1e-9
        B[m] = 2.0 * j1(xarg[m]) / xarg[m]
        B = np.where(cosang > 0, B, 0.0)
        wt = np.ones_like(Rc)
        if obliquity > 0:
            ga = labels[s1][face]; gb = labels[s2][face]
            nrm = seeds[gb] - seeds[ga]
            nn = np.linalg.norm(nrm, axis=1)
            good = nn > 1e-12
            ob = np.ones_like(Rc)
            ob[good] = np.abs(nrm[good] @ dhat) / nn[good]
            wt = (1.0 - obliquity) + obliquity * ob
        amp = Rc * wt * (h * h) * (B ** 2) / (r ** 2)
        tb = np.rint((2.0 * r / c_ref) * fs).astype(np.int64)
        ok = (tb >= 0) & (tb < n_t)
        np.add.at(series, tb[ok], amp[ok])

    _, pulse = F.gaussian_pulse(f0, p.frac_bw, fs)
    from scipy.signal import hilbert
    return np.arange(n_t) / fs, gain * np.abs(hilbert(np.convolve(series, pulse, mode="same")))


def kirchhoff_facets(build, cfg, azimuth_rad=0.0, f0=None, fs=None, n_t=None,
                     c_ref=3850.0, gain=1.0):
    """Rung 6: Kirchhoff facets - coherent WITHIN a facet, incoherent ACROSS.

    WHAT THE EARLIER RUNGS TAUGHT:
      * rung 3 (all faces incoherent)  level -35.7, sens err 4.7 dB. Best so
        far, but its amplitude law R*A/r^2 is a POINT-scatterer law, while
        these grains are 5 wavelengths across.
      * rungs 4 and 5 (all faces coherent) collapsed to -91 and -98 dB.
        Not a bug: faces at equal range carry random-sign R, so a global
        coherent sum cancels as 1/sqrt(N). Checked numerically,
        sqrt(1e4)*4e-7 = -86 dB, which is what came out. The coda really is
        incoherent speckle.

    So NEITHER extreme is right. Cells on the SAME grain boundary belong to
    one physical facet and must add coherently, because that is what produces
    a specular lobe and its diffraction skirts. DIFFERENT facets are
    independent and add in power. Grouping by grain pair does exactly that.

    The Kirchhoff amplitude for one facet, two-way, is

        A = (k / 2 pi) * R * SUM_faces (B^2 / r^2) exp(2 i k r) dA

    The k/2pi = 1/lambda prefactor is what the earlier rungs were missing; it
    is the difference between point-scatterer and facet scaling, worth
    10 log10(r/lambda) ~ 14 dB at mid-record here.
    """
    from scipy.special import j1
    labels, axes, h = build["labels"], build["axes"], build["h"]
    p = cfg.probe
    f0 = f0 or p.f0
    fs = fs or cfg.acq.fs
    n_t = n_t or int(cfg.acq.record_end_s * fs)
    R = cfg.specimen.diameter_m / 2.0
    lam = c_ref / f0
    k = 2 * np.pi / lam
    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    origin = np.array([R * ca, R * sa, 0.0])
    dhat = np.array([-ca, -sa, 0.0])
    Zf, _ = impedance_field(labels, axes, dhat, c_ref)
    nx, ny, nz = labels.shape
    a_el = p.element_d_m / 2.0
    ng = len(axes)

    pid, rr, amp_re, amp_im = [], [], [], []
    for ax in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1); s2[ax] = slice(1, None)
        s1, s2 = tuple(s1), tuple(s2)
        la, lb = labels[s1], labels[s2]
        face = (la != lb) & (la >= 0) & (lb >= 0)
        if not face.any():
            continue
        ga, gb = la[face].astype(np.int64), lb[face].astype(np.int64)
        Za, Zb = Zf[s1][face], Zf[s2][face]
        Rc = (Zb - Za) / (Zb + Za)
        idx = np.argwhere(face).astype(float); idx[:, ax] += 0.5
        Xc = (idx[:, 0] - (nx - 1) / 2.0) * h
        Yc = (idx[:, 1] - (ny - 1) / 2.0) * h
        Zc = (idx[:, 2] - (nz - 1) / 2.0) * h
        rx, ry, rz = Xc - origin[0], Yc - origin[1], Zc - origin[2]
        r = np.maximum(np.sqrt(rx * rx + ry * ry + rz * rz), h)
        cosang = (rx * dhat[0] + ry * dhat[1] + rz * dhat[2]) / r
        ang = np.arccos(np.clip(cosang, -1.0, 1.0))
        xa = k * a_el * np.sin(ang)
        B = np.ones_like(xa); m = np.abs(xa) > 1e-9
        B[m] = 2.0 * j1(xa[m]) / xa[m]
        B = np.where(cosang > 0, B, 0.0)
        a = (k / (2 * np.pi)) * Rc * (h * h) * (B ** 2) / (r ** 2)
        ph = 2.0 * k * r
        lo_, hi_ = np.minimum(ga, gb), np.maximum(ga, gb)
        pid.append(lo_ * ng + hi_)
        rr.append(r); amp_re.append(a * np.cos(ph)); amp_im.append(a * np.sin(ph))
    if not pid:
        return np.arange(n_t) / fs, np.zeros(n_t)
    pid = np.concatenate(pid); rr = np.concatenate(rr)
    ar = np.concatenate(amp_re); ai = np.concatenate(amp_im)

    # group by grain pair = one physical facet
    order = np.argsort(pid, kind="stable")
    pid, rr, ar, ai = pid[order], rr[order], ar[order], ai[order]
    cuts = np.flatnonzero(np.diff(pid)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(pid)]))

    acc = np.zeros(n_t)
    for s, e in zip(starts, ends):
        Fre, Fim = ar[s:e].sum(), ai[s:e].sum()      # coherent within facet
        pw = Fre * Fre + Fim * Fim
        if pw <= 0:
            continue
        rm = rr[s:e].mean()
        tb = int(round((2.0 * rm / c_ref) * fs))
        if 0 <= tb < n_t:
            acc[tb] += pw                            # incoherent across facets
    env = gain * np.sqrt(acc)
    npul = max(int(fs / (p.frac_bw * f0)), 1)
    if npul > 1:
        env = np.convolve(env, np.ones(npul) / npul, mode="same")
    return np.arange(n_t) / fs, env


def kirchhoff_fresnel(build, cfg, azimuth_rad=0.0, f0=None, fs=None, n_t=None,
                      c_ref=3850.0, gain=1.0, fres_mult=1.0):
    """Rung 7: coherence limited to ONE FRESNEL ZONE.

    THE NUMBER THAT SHOULD HAVE COME FIRST: the Fresnel radius here is
    sqrt(lambda r) = sqrt(1.93 mm x 50 mm) = 9.7 mm, and the grains are about
    10 mm. Grain boundaries are therefore almost exactly one Fresnel zone.

    That explains every previous rung:
      * rung 3 summed everything incoherently, i.e. assumed coherence length
        ZERO (point scatterers). Level -35.7, close by luck.
      * rung 6 summed each whole facet coherently, i.e. assumed coherence
        length INFINITE (large specular mirror). A tilted mirror sends its
        specular lobe away and the coherent sum cancels, giving -65.8.
      * rungs 4 and 5 summed globally coherently and cancelled to -91.

    Neither limit applies. Waves stay coherent across roughly one Fresnel
    zone and decorrelate beyond it, so the sum must be coherent within a
    patch of radius sqrt(lambda r) and incoherent across patches. That is the
    physically correct intermediate, and it is the only free parameter here.
    """
    from scipy.special import j1
    labels, axes, h = build["labels"], build["axes"], build["h"]
    p = cfg.probe
    f0 = f0 or p.f0
    fs = fs or cfg.acq.fs
    n_t = n_t or int(cfg.acq.record_end_s * fs)
    R = cfg.specimen.diameter_m / 2.0
    lam = c_ref / f0
    k = 2 * np.pi / lam
    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    origin = np.array([R * ca, R * sa, 0.0])
    dhat = np.array([-ca, -sa, 0.0])
    Zf, _ = impedance_field(labels, axes, dhat, c_ref)
    nx, ny, nz = labels.shape
    a_el = p.element_d_m / 2.0

    P, RR, AR, AI = [], [], [], []
    for ax in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1); s2[ax] = slice(1, None)
        s1, s2 = tuple(s1), tuple(s2)
        la, lb = labels[s1], labels[s2]
        face = (la != lb) & (la >= 0) & (lb >= 0)
        if not face.any():
            continue
        Za, Zb = Zf[s1][face], Zf[s2][face]
        Rc = (Zb - Za) / (Zb + Za)
        idx = np.argwhere(face).astype(float); idx[:, ax] += 0.5
        X = (idx[:, 0] - (nx - 1) / 2.0) * h
        Y = (idx[:, 1] - (ny - 1) / 2.0) * h
        Z = (idx[:, 2] - (nz - 1) / 2.0) * h
        rx, ry, rz = X - origin[0], Y - origin[1], Z - origin[2]
        r = np.maximum(np.sqrt(rx * rx + ry * ry + rz * rz), h)
        cosang = (rx * dhat[0] + ry * dhat[1] + rz * dhat[2]) / r
        angv = np.arccos(np.clip(cosang, -1.0, 1.0))
        xa = k * a_el * np.sin(angv)
        B = np.ones_like(xa); m = np.abs(xa) > 1e-9
        B[m] = 2.0 * j1(xa[m]) / xa[m]
        B = np.where(cosang > 0, B, 0.0)
        a = (k / (2 * np.pi)) * Rc * (h * h) * (B ** 2) / (r ** 2)
        ph = 2.0 * k * r
        # patch id: Fresnel-sized cells in space, so coherence stops there
        fr = fres_mult * np.sqrt(lam * r)
        px = np.floor(X / fr).astype(np.int64)
        py = np.floor(Y / fr).astype(np.int64)
        pz = np.floor(Z / fr).astype(np.int64)
        P.append(((px + 512) * 1024 + (py + 512)) * 1024 + (pz + 512))
        RR.append(r); AR.append(a * np.cos(ph)); AI.append(a * np.sin(ph))
    if not P:
        return np.arange(n_t) / fs, np.zeros(n_t)
    P = np.concatenate(P); RR = np.concatenate(RR)
    AR = np.concatenate(AR); AI = np.concatenate(AI)
    o = np.argsort(P, kind="stable")
    P, RR, AR, AI = P[o], RR[o], AR[o], AI[o]
    cuts = np.flatnonzero(np.diff(P)) + 1
    st = np.concatenate(([0], cuts)); en = np.concatenate((cuts, [len(P)]))
    acc = np.zeros(n_t)
    for s, e in zip(st, en):
        pw = AR[s:e].sum() ** 2 + AI[s:e].sum() ** 2
        if pw <= 0:
            continue
        tb = int(round((2.0 * RR[s:e].mean() / c_ref) * fs))
        if 0 <= tb < n_t:
            acc[tb] += pw
    env = gain * np.sqrt(acc)
    npul = max(int(fs / (p.frac_bw * f0)), 1)
    if npul > 1:
        env = np.convolve(env, np.ones(npul) / npul, mode="same")
    return np.arange(n_t) / fs, env


def kirchhoff_shell(build, cfg, azimuth_rad=0.0, f0=None, fs=None, n_t=None,
                    c_ref=3850.0, gain=1.0, fres_mult=1.0, partial=False,
                    shear=False, c_shear=1813.0, plate=False):
    """Rung 8: Fresnel patches aligned to the WAVEFRONT, not the mesh axes.

    WHY RUNG 7 UNDER-READ BY 25 dB: it tiled space with CARTESIAN CUBES of
    Fresnel size. Monostatic iso-phase surfaces are SPHERES around the probe,
    so the strong contributors are boundary facets tangent to those spheres.
    A cube face-on to the beam axis is tangent-aligned only near one azimuth;
    everywhere else the cube cuts obliquely through the wavefront, splitting
    every tangent facet across several cubes. Split halves are power-added,
    so an N-way split costs 10*log10(N) dB, and the radial cube extent mixes
    ~10 phase cycles into each patch on top. The geometry of the patches, not
    the physics in them, set the level.

    THE FIX: build the patches ON the wavefront. Spherical coordinates about
    the probe: range shells whose width is the Fresnel length s = m*sqrt(lam r)
    (shell index n = 2*sqrt(r/lam)/m, so the width comes out right
    automatically), polar bands of arc s, azimuthal sectors of arc s. Every
    patch is a Fresnel-sized tile of the wavefront, which is exactly the
    region a real wave keeps in phase. Band/sector counts are derived from
    the SHELL-CENTRE radius so the tiling is deterministic; per-face sizing
    (as in rung 7) wobbles at bin edges and splits patches by accident.

    PARTIAL COHERENCE (partial=True): the full wave propagates through grains
    with +/- a few percent velocity, so by depth r the phase has random-walked
    by sigma_phi^2 = k^2 <eps^2> ell * 2r  (eps = fractional velocity
    fluctuation along the beam, ell = grain size; standard travel-time
    variance). Where sigma_phi is large the deterministic Kirchhoff
    cancellation cannot survive, and each face scatters independently. Blend
    the two regimes with w = exp(-sigma_phi^2):

        P_patch = w * |sum a_kirchhoff|^2  +  (1 - w) * sum a_rung3^2

    The coherent term uses Kirchhoff facet units (valid once a patch
    assembles faces into a >= lambda mirror); the incoherent term uses rung
    3's per-face law, whose level matched the full wave. NOTHING IS FITTED:
    eps and ell come from the specimen, so w also responds to the fabric,
    which is a second sensitivity channel the other rungs lack.

    MEASURED OUTCOMES (ladder.py, dB level at fabric 0 / sensitivity RMS):
      * pure wavefront patches: -54.3 to -63.1 over fres_mult 0.5-2.0. The
        realignment recovered at most 6 of rung 7's 25 dB, so the Cartesian
        cubes were NOT the main cause: deterministic Kirchhoff cancellation
        is intrinsically too strong however the patches are shaped.
      * partial=True: -38.2/-40.2/-41.3 with sens 4.0/3.4/3.0 dB at
        fres_mult 0.5/1.0/2.0. Best sensitivity on the ladder; the
        phase-jitter blend is the physics that matters.
      * a max-statistics explanation for the residual was tested and mostly
        DIED: on a real full-wave coda, max(speckle env)/max(smoothed env)
        is only +1.7 dB (the naive Rayleigh estimate of +5.6 fails because
        the window max sits at the decaying window's start). So roughly
        6 dB of full-wave excess over the incoherent sum is real physics:
        candidate channels are rim-cavity reverberation, multiple
        scattering, and scattering-broadened beams.

    MODE CONVERSION (shear=True): the full wave sits ABOVE even the fully
    incoherent PP boundary sum, so a channel is missing, and the obvious one
    is P->S->P. A face at depth r returns converted energy at
    t = r/c_p + r/c_s, which for c_s = sqrt(C44/rho) = 1813 m/s puts
    conversions from ~40 mm depth squarely in the 22-42 us coda window.
    Per face the converted amplitude is taken as R_pp * sin(2*theta_inc)
    with theta_inc from the (staircase) face normal - the grid-axis normal
    distribution over a curved boundary approximates the true normal mix.
    Conversions ride different path physics face-to-face, so they are added
    incoherently. The receive-side S directivity is approximated by the same
    piston B; that overstates the S beam width, so treat this channel as an
    order-of-magnitude test, not calibrated physics.
    MEASURED: +2 dB of level on top of partial (-39.2 at fres_mult 2.0),
    sensitivity held at 3.1 dB. Helpful, not decisive.

    PLATE GUIDING (plate=True): the disk is a 35 mm plate whose top and
    bottom faces reflect at rim_reflection (~-0.78 against the fluid
    surround, ~-1 against air in the real rig). The piston beam is wider
    than the plate beyond ~30 mm depth, so power the free-space B^2/r^2 law
    writes off as gone is partly returned by plate-face bounces - a channel
    that GROWS with depth, matching where the models under-read. Method of
    images: probe images at z = +/- t_plate see each face through one
    bounce, weighted |R_plate| per bounce, and each (out, back) leg pair is
    an independent incoherent path binned at (r_out + r_back)/c. The
    (direct, direct) pair is the ordinary term and is excluded here.
    Single-bounce images only; the |R|^2 and B-lobe cost of double bounces
    makes them a small correction on a correction.
    """
    import forward as F
    from scipy.special import j1
    labels, axes, h = build["labels"], build["axes"], build["h"]
    p = cfg.probe
    f0 = f0 or p.f0
    fs = fs or cfg.acq.fs
    n_t = n_t or int(cfg.acq.record_end_s * fs)
    R = cfg.specimen.diameter_m / 2.0
    lam = c_ref / f0
    k = 2 * np.pi / lam
    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    origin = np.array([R * ca, R * sa, 0.0])
    dhat = np.array([-ca, -sa, 0.0])
    e2 = np.array([-sa, ca, 0.0])            # dhat x z: azimuth basis
    Zf, _ = impedance_field(labels, axes, dhat, c_ref)
    nx, ny, nz = labels.shape
    a_el = p.element_d_m / 2.0

    # medium-derived phase-decorrelation rate for the partial blend
    vgr = F.grain_speeds(np.asarray(axes, float), dhat)
    eps2 = (vgr.std() / vgr.mean()) ** 2
    ell = cfg.specimen.mean_grain_size()

    acc_ps = np.zeros(n_t)
    acc_pl = np.zeros(n_t)
    P, RR, AR, AI, A2K, A23 = [], [], [], [], [], []
    for ax in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1); s2[ax] = slice(1, None)
        s1, s2 = tuple(s1), tuple(s2)
        la, lb = labels[s1], labels[s2]
        face = (la != lb) & (la >= 0) & (lb >= 0)
        if not face.any():
            continue
        Za, Zb = Zf[s1][face], Zf[s2][face]
        Rc = (Zb - Za) / (Zb + Za)
        idx = np.argwhere(face).astype(float); idx[:, ax] += 0.5
        X = (idx[:, 0] - (nx - 1) / 2.0) * h
        Y = (idx[:, 1] - (ny - 1) / 2.0) * h
        Z = (idx[:, 2] - (nz - 1) / 2.0) * h
        rx, ry, rz = X - origin[0], Y - origin[1], Z - origin[2]
        r = np.maximum(np.sqrt(rx * rx + ry * ry + rz * rz), h)
        cosang = (rx * dhat[0] + ry * dhat[1] + rz * dhat[2]) / r
        angv = np.arccos(np.clip(cosang, -1.0, 1.0))
        xa = k * a_el * np.sin(angv)
        B = np.ones_like(xa); m = np.abs(xa) > 1e-9
        B[m] = 2.0 * j1(xa[m]) / xa[m]
        B = np.where(cosang > 0, B, 0.0)
        ak = (k / (2 * np.pi)) * Rc * (h * h) * (B ** 2) / (r ** 2)
        a3 = Rc * h * (B ** 2) / (r ** 2)
        ph = 2.0 * k * r

        if shear:
            # face normal is the grid axis; incidence angle from it
            cosi = np.abs((rx, ry, rz)[ax]) / r
            sin2i = 2.0 * cosi * np.sqrt(np.maximum(1.0 - cosi * cosi, 0.0))
            # NEGATIVE RESULT, kept for the record: scaling conversions with
            # the SH contrast across the face (sh_speeds) is the "more
            # honest" per-face physics but scored WORSE (sens 4.3 vs 3.1):
            # SH contrast persists as the fabric randomises, so the shear
            # coda refuses to fall while the full wave falls 15 dB. The
            # receive-side S->signal conversion loss (not modelled) evidently
            # suppresses this channel in reality; the P-contrast toy keeps
            # the channel small in the right way.
            aps = Rc * sin2i * h * (B ** 2) / (r ** 2)
            tps = np.rint(r * (1.0 / c_ref + 1.0 / c_shear) * fs).astype(np.int64)
            okp = (tps >= 0) & (tps < n_t)
            np.add.at(acc_ps, tps[okp], (aps * aps)[okp])

        if plate:
            tpl = cfg.specimen.thickness_m
            Rpl = abs(cfg.err.rim_reflection)
            legs = []
            for zoff, bounce in ((0.0, 0), (tpl, 1), (-tpl, 1)):
                rzm = rz - zoff
                rm2 = np.maximum(np.sqrt(rx * rx + ry * ry + rzm * rzm), h)
                cm = (rx * dhat[0] + ry * dhat[1] + rzm * dhat[2]) / rm2
                xm = k * a_el * np.sqrt(np.maximum(1.0 - cm * cm, 0.0))
                Bm = np.ones_like(xm); mm = np.abs(xm) > 1e-9
                Bm[mm] = 2.0 * j1(xm[mm]) / xm[mm]
                Bm = np.where(cm > 0, Bm, 0.0)
                legs.append(((Rpl ** bounce) * Bm / rm2, rm2))
            for mi in range(3):
                for ni in range(3):
                    if mi == 0 and ni == 0:
                        continue          # the direct-direct path is the main term
                    Lm, rmo = legs[mi]; Ln, rno = legs[ni]
                    a_img = Rc * h * Lm * Ln
                    tbp = np.rint(((rmo + rno) / c_ref) * fs).astype(np.int64)
                    okq = (tbp >= 0) & (tbp < n_t)
                    np.add.at(acc_pl, tbp[okq], (a_img * a_img)[okq])

        # wavefront-aligned patch id: (range shell, polar band, azimuth sector)
        ir = np.floor(2.0 * np.sqrt(r / lam) / fres_mult).astype(np.int64)
        r_c = lam * ((ir + 0.5) * fres_mult / 2.0) ** 2
        s_c = fres_mult * np.sqrt(lam * r_c)
        dth = s_c / r_c
        ith = np.floor(angv / dth).astype(np.int64)
        th_c = (ith + 0.5) * dth
        nphi = np.maximum(1, np.rint(2 * np.pi * r_c * np.sin(th_c)
                                     / s_c)).astype(np.int64)
        phi = np.arctan2(rx * e2[0] + ry * e2[1], rz)
        iph = np.minimum((np.floor((phi + np.pi) / (2 * np.pi) * nphi))
                         .astype(np.int64), nphi - 1)
        P.append((ir * 4096 + ith) * 8192 + iph)
        RR.append(r)
        AR.append(ak * np.cos(ph)); AI.append(ak * np.sin(ph))
        A2K.append(ak * ak); A23.append(a3 * a3)
    if not P:
        return np.arange(n_t) / fs, np.zeros(n_t)
    P = np.concatenate(P); RR = np.concatenate(RR)
    AR = np.concatenate(AR); AI = np.concatenate(AI)
    A2K = np.concatenate(A2K); A23 = np.concatenate(A23)
    o = np.argsort(P, kind="stable")
    P, RR, AR, AI = P[o], RR[o], AR[o], AI[o]
    A2K, A23 = A2K[o], A23[o]
    cuts = np.flatnonzero(np.diff(P)) + 1
    st = np.concatenate(([0], cuts)); en = np.concatenate((cuts, [len(P)]))
    acc = np.zeros(n_t)
    for s, e in zip(st, en):
        coh = AR[s:e].sum() ** 2 + AI[s:e].sum() ** 2
        rm = RR[s:e].mean()
        if partial:
            w = np.exp(-2.0 * k * k * eps2 * ell * rm)
            pw = w * coh + (1.0 - w) * A23[s:e].sum()
        else:
            pw = coh
        if pw <= 0:
            continue
        tb = int(round((2.0 * rm / c_ref) * fs))
        if 0 <= tb < n_t:
            acc[tb] += pw
    env = gain * np.sqrt(acc + acc_ps + acc_pl)
    npul = max(int(fs / (p.frac_bw * f0)), 1)
    if npul > 1:
        env = np.convolve(env, np.ones(npul) / npul, mode="same")
    return np.arange(n_t) / fs, env
