"""Forward model: the A-scan the transducer actually records at one azimuth.

This is a RAY model, not full-wave, and that is a deliberate choice rather
than a shortcut. The grains are 8 to 28 wavelengths across (see config
summary), so the specimen is in the geometric scattering regime: each grain
is individually resolvable and the echo train is a sequence of discrete
boundary reflections, not a diffuse speckle field. The full 3D elastic solve
is the validation reference, not the workhorse. At ppw 6 a 180-angle scan is
about 5 h on the RTX 4070, which is useless for iterating on a reconstruction.

What is modelled:
  * qP velocity per grain from the Christoffel solution (6.95% anisotropy,
    4045.8 m/s along the c-axis down to 3776 m/s at 50.4 deg)
  * reflection at every grain boundary from the impedance contrast
  * EXACT boundary normals. Laguerre (power diagram) cell walls are planes
    whose normal is the difference of the two seed points, so obliquity is
    computed, not fudged
  * two-way transmission loss through every interface, so scattering
    attenuation emerges from the microstructure instead of being a fitted
    constant
  * the retroreflection constraint: a ray launched off the diameter returns
    to the rim displaced from where it started, and is only detected if it
    lands back on the element. This is what makes only the diameter
    retroreflect on a circular specimen.
  * E1, E2, E3 from repeated internal round trips, plus first-order
    rim-to-boundary multiples which populate the coda between E1 and E2
"""
import os
import sys

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))
from ringfwi import anisotropy as an   # noqa: E402

RHO_ICE = an.ICE_RHO

# qP speed vs angle from the c-axis, tabulated once (the Christoffel solve is
# far too slow to call per grain per ray per angle)
_PSI = np.linspace(0.0, np.pi / 2.0, 901)
_VQP = np.array([an.ice_qp_vs_caxis(p) for p in _PSI])


def qp_speed_at(psi):
    """qP speed at angle psi (rad) between propagation and the c-axis."""
    return np.interp(np.abs(psi), _PSI, _VQP)


def grain_speeds(axes, direction):
    """qP speed of every grain for one propagation direction."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    cospsi = np.clip(np.abs(np.asarray(axes, float) @ d), 0.0, 1.0)
    return qp_speed_at(np.arccos(cospsi))


# ─────────────────────────────── the pulse ───────────────────────────────
def gaussian_pulse(f0, frac_bw, fs):
    """Gaussian-modulated cosine with the requested -6 dB fractional bandwidth."""
    B = frac_bw * f0
    sigma_t = np.sqrt(2.0 * np.log(2.0)) / (np.pi * B)
    n = int(np.ceil(3.0 * sigma_t * fs))
    t = np.arange(-n, n + 1) / fs
    return t, np.exp(-0.5 * (t / sigma_t) ** 2) * np.cos(2 * np.pi * f0 * t)


# ────────────────────────────── ray marching ─────────────────────────────
def march(labels, h, p0, d, s_max, ds):
    """Sample grain labels along a ray. Returns (arc length, label)."""
    n = max(int(s_max / ds), 2)
    s = (np.arange(n) + 0.5) * ds
    pts = p0[None, :] + s[:, None] * d[None, :]
    shape = np.array(labels.shape)
    idx = np.rint(pts / h + (shape - 1) / 2.0).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < shape), axis=1)
    lab = np.full(n, -1, np.int32)
    if ok.any():
        j = idx[ok]
        lab[ok] = labels[j[:, 0], j[:, 1], j[:, 2]]
    return s, lab


def segments(s, lab):
    """Compress a marched ray into runs of constant grain.

    Returns (grain ids, entry arc length, exit arc length) for cells inside
    the specimen only.
    """
    inside = lab >= 0
    if not inside.any():
        return np.empty(0, int), np.empty(0), np.empty(0)
    change = np.flatnonzero(np.diff(lab) != 0)
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(lab)]))
    keep = lab[starts] >= 0
    starts, ends = starts[keep], ends[keep]
    return lab[starts], s[starts], s[np.minimum(ends, len(s) - 1)]


# ───────────────────────── one ray, one azimuth ──────────────────────────
def ray_echoes(labels, seeds, axes, h, R, p0, d, ds, cfg,
               weight=1.0, detect=True):
    """Arrival times and signed amplitudes for a single ray.

    `detect` is False when the geometry says the returning ray misses the
    element; boundary echoes are still generated (they scatter in all
    directions) but the specular rim echo is not.
    """
    chord = 2.0 * np.sqrt(max(R ** 2 - _perp_dist2(p0, d, R), 0.0))
    s, lab = march(labels, h, p0, d, chord * 1.02, ds)
    gid, s0, s1 = segments(s, lab)
    if len(gid) == 0:
        return np.empty(0), np.empty(0)

    v = grain_speeds(axes[gid], d)                 # speed in each segment
    L = np.maximum(s1 - s0, 0.0)
    dt = L / v
    t_cum = np.concatenate(([0.0], np.cumsum(dt)))  # one-way time to each face
    Z = RHO_ICE * v

    # reflection coefficient at each internal boundary, with the exact
    # Laguerre plane normal (power-diagram walls are planes)
    nb = len(gid) - 1
    times, amps = [], []
    trans = 1.0                                     # running two-way transmission
    for i in range(nb):
        z1, z2 = Z[i], Z[i + 1]
        r = (z2 - z1) / (z2 + z1)
        nrm = seeds[gid[i + 1]] - seeds[gid[i]]
        nn = np.linalg.norm(nrm)
        obl = abs(float(nrm @ d) / nn) if nn > 1e-12 else 1.0
        if detect:
            times.append(2.0 * t_cum[i + 1])
            amps.append(weight * trans * r * obl)
        trans *= (1.0 - r * r)                      # energy carried onward
    t_one = t_cum[-1]

    # far-rim echo: ice to air (the oil film is thin against 0.77 mm), and
    # the near rim reflects internally to make E2, E3, ...
    if detect:
        r_far = cfg.err.rim_reflection
        r_near = cfg.err.near_rim_internal
        for k in range(1, cfg.acq.n_reverb + 1):
            amp = weight * (trans ** k) * (r_far ** k) * (r_near ** (k - 1))
            times.append(2.0 * k * t_one)
            amps.append(amp)
            # first-order multiples: rim echo re-reflected off each boundary
            if k == 1 and cfg.acq.include_multiples:
                for i in range(nb):
                    z1, z2 = Z[i], Z[i + 1]
                    r = (z2 - z1) / (z2 + z1)
                    times.append(2.0 * t_one + 2.0 * (t_one - t_cum[i + 1]))
                    amps.append(weight * trans * r_far * r * 0.5)
    return np.asarray(times), np.asarray(amps)


def _perp_dist2(p0, d, R):
    """Squared perpendicular distance from the disk axis to the ray line."""
    p = np.asarray(p0, float)[:2]
    dd = np.asarray(d, float)[:2]
    n2 = dd @ dd
    if n2 < 1e-18:
        return 0.0
    t = -(p @ dd) / n2
    q = p + t * dd
    return float(q @ q)


def _returns_to_element(alpha, beta, R, elem_d):
    """Does a ray launched off-axis come back onto the element?

    In-plane: a chord makes equal angles with the radii at both ends, so a ray
    at alpha reflects to a rim point displaced by an arc of 4*alpha*R.
    Out-of-plane: the cylindrical rim preserves the z component, so the ray
    returns at z = 2*L*tan(beta).
    """
    lateral = abs(4.0 * alpha * R)
    axial = abs(2.0 * 2.0 * R * np.tan(beta))
    return (lateral <= elem_d / 2.0) and (axial <= elem_d / 2.0)


def directivity(alpha, k, a):
    """Circular piston directivity, 2*J1(x)/x."""
    from scipy.special import j1
    x = k * a * np.sin(alpha)
    out = np.ones_like(np.atleast_1d(x), dtype=float)
    nz = np.abs(x) > 1e-9
    out[nz] = 2.0 * j1(x[nz]) / x[nz]
    return out if np.ndim(alpha) else float(out[0])


# ─────────────────────── full A-scan at one azimuth ──────────────────────
def simulate_angle(build, cfg, azimuth_rad, tilt_rad=0.0, ds=None,
                   n_alpha=None, n_beta=None):
    """Synthesise the A-scan recorded at one probe azimuth.

    Returns (t, trace) with t in seconds from the transmit instant.
    """
    labels, axes, seeds, h = (build["labels"], build["axes"],
                              build["seeds"], build["h"])
    R = cfg.specimen.diameter_m / 2.0
    p = cfg.probe
    ds = ds or h * 0.5
    n_alpha = n_alpha or cfg.acq.n_rays_inplane
    n_beta = n_beta or cfg.acq.n_rays_outplane

    # probe on the rim, looking in along the diameter, optionally tilted out
    # of plane. When tilted it starts low so the ray has the full thickness
    # to climb through before it reaches the far rim.
    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    cb, sb = np.cos(tilt_rad), np.sin(tilt_rad)
    z0 = -np.sign(tilt_rad) * cfg.specimen.thickness_m / 2.0 * 0.9
    origin = np.array([R * ca, R * sa, z0])
    inward = np.array([-ca * cb, -sa * cb, sb])
    tang = np.array([-sa, ca, 0.0])
    zhat = np.cross(inward, tang)
    zhat /= np.linalg.norm(zhat)

    th = p.beam_half_angle(cfg.solver.c_ref)
    k = 2 * np.pi * p.f0 / cfg.solver.c_ref
    a = p.element_d_m / 2.0
    alphas = np.linspace(-th, th, n_alpha) if n_alpha > 1 else np.array([0.0])
    betas = np.linspace(-th, th, n_beta) if n_beta > 1 else np.array([0.0])

    all_t, all_a = [], []
    for al in alphas:
        for be in betas:
            d = (inward * (np.cos(al) * np.cos(be))
                 + tang * np.sin(al) + zhat * np.sin(be))
            d /= np.linalg.norm(d)
            # transmit x receive directivity (reciprocity)
            w = float(directivity(np.array([al]), k, a)[0]
                      * directivity(np.array([be]), k, a)[0]) ** 2
            if abs(w) < 1e-4:
                continue
            det = _returns_to_element(al, be, R, p.element_d_m)
            t_i, a_i = ray_echoes(labels, seeds, axes, h, R, origin, d, ds,
                                  cfg, weight=w, detect=True)
            if not det:                       # rim specular lost, coda remains
                a_i = a_i * cfg.acq.offaxis_rim_suppression
            all_t.append(t_i)
            all_a.append(a_i)

    if not all_t:
        n = int(cfg.acq.record_end_s * cfg.acq.fs)
        return np.arange(n) / cfg.acq.fs, np.zeros(n)

    times = np.concatenate(all_t)
    amps = np.concatenate(all_a)

    # bin the reflectivity series onto the sample grid, then convolve
    fs = cfg.acq.fs
    n = int(cfg.acq.record_end_s * fs)
    t = np.arange(n) / fs
    series = np.zeros(n)
    bins = np.rint(times * fs).astype(np.int64)
    ok = (bins >= 0) & (bins < n)
    np.add.at(series, bins[ok], amps[ok])

    _, pulse = gaussian_pulse(p.f0, p.frac_bw, fs)
    trace = np.convolve(series, pulse, mode="same")
    return t, trace
