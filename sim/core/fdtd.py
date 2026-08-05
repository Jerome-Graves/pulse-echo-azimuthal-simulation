"""Optimised forward-only 3D anisotropic elastic FDTD.

Deliberately a SEPARATE solver from ringfwi.elastic3d rather than a patch to
it: that one carries the adjoint and gradient machinery openUSCT's FWI
depends on, and changing its numerics underneath would be reckless. This one
is forward-only, so it can take liberties that a gradient solver cannot.

Four changes over the reference solver, in the order they matter:

1. SPONGE ABSORBING LAYER.
   The reference has no absorbing boundary at all. Every full-disk run we
   attempted went unstable in late time (one decayed correctly to 55 us then
   grew 1400x by 110 us; a lighter surround produced outright NaN). Waves
   reaching the grid edge have nowhere to go. A tapered damping region
   removes them, which fixes the blow-ups AND lets the domain shrink, since
   the specimen no longer needs a wide margin of quiet fluid around it.

2. HIGHER-ORDER SPATIAL STENCILS.
   The reference is 2nd order and needs about 10 points per wavelength. An
   8th-order staggered stencil needs about 4. In 3D that is (10/4)^3 ~ 15x
   fewer cells, and because dt scales with h the step count drops too, offset
   by the ~1.29x stricter CFL of the wider stencil.

3. DISPERSION-OPTIMISED COEFFICIENTS.
   Taylor coefficients maximise formal order at k->0, which is the wrong
   target: what matters is phase error across the band actually being
   propagated. The optimised tables here are solved numerically by minimising
   peak phase error over a specified band, not copied from a table.

4. FEWER TEMPORARIES.
   FDTD is memory-bandwidth bound, not compute bound. The reference builds a
   fresh array for every derivative and every stiffness product; the stress
   update alone can allocate dozens of full-grid temporaries per step. Here
   the derivative buffers are pre-allocated once and reused, and the stress
   accumulation is done in place.
"""
import numpy as np

# Half-cell offsets (z, y, x) of each staggered position, matching elastic3d.
_POS = {"c": (0, 0, 0), "yz": (1, 1, 0), "xz": (1, 0, 1), "xy": (0, 1, 1)}
_VPOS = {1: "c", 2: "c", 3: "c", 4: "yz", 5: "xz", 6: "xy"}


# ───────────────────────── stencil coefficients ──────────────────────────
def taylor_coeffs(order):
    """Standard staggered-grid coefficients, exact to `order` in h.

    Solves the linear system that makes sum_k c_k * sin((k-1/2) t) reproduce
    t/2 to the requested order, which is the staggered analogue of the usual
    Taylor construction.
    """
    n = order // 2
    k = np.arange(1, n + 1)
    A = np.zeros((n, n))
    for r in range(n):
        A[r] = (2 * k - 1) ** (2 * r + 1)
    b = np.zeros(n)
    b[0] = 1.0
    c = np.linalg.solve(A, b)
    return c


def optimised_coeffs(order, kh_max=1.6, n_grid=400, multistart=True):
    """Coefficients minimising PEAK phase error over kh in (0, kh_max].

    The effective wavenumber of a staggered stencil is
        k_eff h = 2 * sum_k c_k sin((k - 1/2) kh)
    and we want k_eff = k across the band. Taylor coefficients nail kh -> 0
    and drift badly near the Nyquist end; minimising the worst case instead
    buys roughly one extra point per wavelength of usable band.

    CONSISTENCY IS ENFORCED, and it matters. Left unconstrained the optimiser
    happily trades away accuracy at kh -> 0 to flatten the error near Nyquist,
    which looks like a huge win on a peak-error metric and is useless in
    practice: a broadband pulse carries real energy at low wavenumber and that
    content would propagate at the wrong speed. Requiring
    sum_j 2 c_j (j + 1/2) = 1 keeps the stencil exact in the long-wave limit
    and optimises only the remaining freedom. For order 2 there is no freedom
    left, so it correctly returns the Taylor coefficient.
    """
    from scipy.optimize import minimize
    n = order // 2
    kh = np.linspace(1e-3, kh_max, n_grid)
    K = np.stack([2.0 * np.sin((j + 0.5) * kh) for j in range(n)], axis=1)
    w = 2.0 * (np.arange(n) + 0.5)          # consistency weights

    c0 = taylor_coeffs(order)
    if n == 1:
        return c0

    def expand(free):
        """Recover c_0 from the consistency constraint."""
        c = np.empty(n)
        c[1:] = free
        c[0] = (1.0 - w[1:] @ free) / w[0]
        return c

    def worst(free):
        return np.max(np.abs(K @ expand(free) - kh) / kh)

    # MULTI-START (2026-07-31). A single Nelder-Mead from the Taylor
    # start does NOT converge: on order 8 / kh_max 1.6 it stalls at
    # peak error 4.36e-4, while restarting from perturbed simplices
    # reaches 5.99e-5 on the IDENTICAL objective - 7.3x better for a
    # few milliseconds of extra CPU. Found by an adversarial review of
    # the grid-noise pathology; the stalled solve was leaving free
    # accuracy on the table in exactly the 3-5 MHz band where the
    # staircase noise lives.
    if not multistart:
        # EXACT pre-2026-07-31 behaviour, kept so sweeps generated
        # before the fix stay reproducible (sweep_runner selects this
        # for any sweep whose config predates the fd_kh_max key).
        res = minimize(worst, c0[1:], method="Nelder-Mead",
                       options=dict(maxiter=40000, maxfev=40000,
                                    xatol=1e-13, fatol=1e-15))
        return expand(res.x) if worst(res.x) < worst(c0[1:]) else c0

    best, bx = worst(c0[1:]), c0[1:]
    rng = np.random.default_rng(0)          # deterministic
    starts = [c0[1:]] + [c0[1:] * (1.0 + 0.15 * rng.standard_normal(n - 1))
                         for _ in range(24)]
    for s in starts:
        r = minimize(worst, s, method="Nelder-Mead",
                     options=dict(maxiter=40000, maxfev=40000,
                                  xatol=1e-14, fatol=1e-16))
        # polish: a second pass from the winner escapes simplex stalls
        r = minimize(worst, r.x, method="Nelder-Mead",
                     options=dict(maxiter=40000, maxfev=40000,
                                  xatol=1e-14, fatol=1e-16))
        if worst(r.x) < best:
            best, bx = worst(r.x), r.x
    return expand(bx)


def ppw_for_error(c, tol=1e-3, kh_hi=3.0):
    """Points per wavelength at which this stencil's phase error stays under `tol`."""
    kh = np.linspace(1e-3, kh_hi, 3000)
    n = len(c)
    keff = sum(2.0 * c[j] * np.sin((j + 0.5) * kh) for j in range(n))
    err = np.abs(keff - kh) / kh
    ok = kh[err < tol]
    return float(2 * np.pi / ok.max()) if len(ok) else np.inf


def cfl_limit(c, dim=3):
    """Stable dt = cfl_limit * h / v_max, from the standard staggered condition.

        dt <= h / (v * sqrt(D) * sum_j |c_j|)

    An earlier version carried a spurious extra factor of 2, making every run
    twice as slow as it needed to be. Use `safe_dt` rather than this directly:
    the speed that belongs in the denominator is `max_grid_speed`, not the
    fastest physical wave, and getting that wrong is what destabilised every
    full-disc run in this project.
    """
    return 1.0 / (np.sqrt(dim) * np.sum(np.abs(c)))


# ───────────────────────── staggered derivatives ─────────────────────────
def _deriv(xp, f, ax, coeffs, inv_h, out, forward):
    """Staggered derivative along `ax` into pre-allocated `out`.

    forward=True : integer -> half position (elastic3d's _Df)
    forward=False: half -> integer position (elastic3d's _Db)
    """
    out.fill(0)
    n = f.shape[ax]
    for j, c in enumerate(coeffs):
        if forward:
            hi_lo, lo_lo = j + 1, j            # f[i+j+1] - f[i-j]
        else:
            hi_lo, lo_lo = j, j + 1            # f[i+j]   - f[i-j-1]
        a0 = hi_lo
        b0 = lo_lo
        lo_len = n - a0 - b0
        if lo_len <= 0:
            break
        sl_out = [slice(None)] * f.ndim
        sl_hi = [slice(None)] * f.ndim
        sl_lo = [slice(None)] * f.ndim
        sl_out[ax] = slice(b0, b0 + lo_len)
        sl_hi[ax] = slice(a0 + b0, a0 + b0 + lo_len)
        sl_lo[ax] = slice(0, lo_len)
        out[tuple(sl_out)] += (c * inv_h) * (f[tuple(sl_hi)] - f[tuple(sl_lo)])
    return out


def _avg_axis(xp, f, ax, d, out):
    out.fill(0)
    hi = [slice(None)] * f.ndim
    lo = [slice(None)] * f.ndim
    hi[ax] = slice(1, None)
    lo[ax] = slice(0, -1)
    hi, lo = tuple(hi), tuple(lo)
    if d > 0:
        out[lo] = 0.5 * (f[lo] + f[hi])
    else:
        out[hi] = 0.5 * (f[lo] + f[hi])
    return out


def _move(xp, f, src, dst, buf):
    for ax in range(3):
        d = _POS[dst][ax] - _POS[src][ax]
        if d:
            f = _avg_axis(xp, f, ax - 3, d, buf[ax]).copy()
    return f


# ─────────────────────────── sponge boundary ─────────────────────────────
def sponge_profile(shape, width, strength=0.09, xp=np):
    """Multiplicative per-step damping, 1 in the interior, tapering at the edges.

    A quadratic taper rather than a step: an abrupt damping front reflects
    almost as badly as the hard boundary it replaces.
    """
    nz, ny, nx = shape
    d = xp.ones(shape, dtype=xp.float32)
    if width <= 0:
        return d
    for ax, n in enumerate((nz, ny, nx)):
        w = min(width, n // 2)
        if w <= 0:
            continue
        prof = np.ones(n)
        t = (np.arange(w) / w)
        taper = np.exp(-strength * (1.0 - t) ** 2 * w)
        prof[:w] = taper
        prof[n - w:] = taper[::-1]
        sh = [1, 1, 1]
        sh[ax] = n
        d *= xp.asarray(prof.reshape(sh), dtype=xp.float32)
    return d


def forward(C, rho, h, dt, nt, src_pts, wavelet, rec_groups, order=8,
            coeffs=None, sponge_width=0, sponge_strength=0.09,
            device="auto", record="pressure", progress=None):
    """Forward-only 3D anisotropic elastic FDTD with sponge boundaries.

    Signature deliberately mirrors ringfwi.elastic3d.forward for the pieces
    that matter, so the two can be swapped in a validation harness.

    C          : dict of the 21 Voigt stiffnesses, each (nz, ny, nx) or scalar
    rho        : (nz, ny, nx)
    src_pts    : list of ((iz, iy, ix), weight)
    rec_groups : list over receivers of (list of (iz, iy, ix), weights)
    """
    xp = np
    use_gpu = False
    if device in ("auto", "gpu"):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                xp, use_gpu = cp, True
        except Exception:
            if device == "gpu":
                raise
    dtype = xp.float32 if use_gpu else np.float64

    if coeffs is None:
        coeffs = optimised_coeffs(order)
    coeffs = np.asarray(coeffs, float)

    rho_h = np.asarray(rho, float)
    nz, ny, nx = rho_h.shape
    inv_h = 1.0 / h
    # Effective density AT THE VELOCITY NODES, not at cell centres.
    # vx/vy/vz each sit half a cell off along their own axis, so the density
    # they feel is the average of the two cells they straddle. Using the
    # cell-centred value instead is wrong wherever density jumps, and it is
    # what made a light surround explode: at a 92:1 ice/fluid contrast the
    # velocity update divides by a density that belongs to neither side.
    # Harmonic averaging is the standard heterogeneous-media treatment and is
    # what keeps a sharp contrast stable.
    rho_vx = xp.asarray(face_density(rho_h, 2), dtype)
    rho_vy = xp.asarray(face_density(rho_h, 1), dtype)
    rho_vz = xp.asarray(face_density(rho_h, 0), dtype)

    def as_dev(v):
        a = np.asarray(v, float)
        return xp.asarray(a if a.ndim == 3 else np.full((nz, ny, nx), float(a)),
                          dtype)

    Cd = {k: as_dev(v) for k, v in C.items()}
    active = {(I, J): bool(np.any(np.asarray(C[f"C{min(I,J)}{max(I,J)}"])))
              for I in range(1, 7) for J in range(1, 7)}

    vx = xp.zeros((nz, ny, nx), dtype)
    vy = xp.zeros((nz, ny, nx), dtype)
    vz = xp.zeros((nz, ny, nx), dtype)
    s = {I: xp.zeros((nz, ny, nx), dtype) for I in range(1, 7)}

    # pre-allocated scratch: the whole point of item 4
    buf = {k: xp.zeros((nz, ny, nx), dtype) for k in
           ("d0", "d1", "d2", "acc", "m0", "m1", "m2")}
    mbuf = [buf["m0"], buf["m1"], buf["m2"]]

    damp = (sponge_profile((nz, ny, nx), sponge_width, sponge_strength, xp)
            if sponge_width > 0 else None)
    if damp is not None:
        damp = damp.astype(dtype)

    Z, Y, X = -3, -2, -1
    n_rx = len(rec_groups)
    rec = np.zeros((nt, n_rx))
    grp = [([tuple(p) for p in pts], np.asarray(w, float))
           for pts, w in rec_groups]

    for n in range(nt):
        gam = {}
        gam[1] = _deriv(xp, vx, X, coeffs, inv_h, buf["d0"], False).copy()
        gam[2] = _deriv(xp, vy, Y, coeffs, inv_h, buf["d0"], False).copy()
        gam[3] = _deriv(xp, vz, Z, coeffs, inv_h, buf["d0"], False).copy()
        gam[4] = (_deriv(xp, vz, Y, coeffs, inv_h, buf["d0"], True).copy()
                  + _deriv(xp, vy, Z, coeffs, inv_h, buf["d1"], True))
        gam[5] = (_deriv(xp, vz, X, coeffs, inv_h, buf["d0"], True).copy()
                  + _deriv(xp, vx, Z, coeffs, inv_h, buf["d1"], True))
        gam[6] = (_deriv(xp, vy, X, coeffs, inv_h, buf["d0"], True).copy()
                  + _deriv(xp, vx, Y, coeffs, inv_h, buf["d1"], True))

        moved = {}

        def strain_at(J, pos):
            key = (J, pos)
            if key not in moved:
                moved[key] = (gam[J] if _VPOS[J] == pos
                              else _move(xp, gam[J], _VPOS[J], pos, mbuf))
            return moved[key]

        for I in range(1, 7):
            pos = _VPOS[I]
            acc = buf["acc"]
            acc.fill(0)
            any_term = False
            for J in range(1, 7):
                if not active[(I, J)]:
                    continue
                acc += Cd[f"C{min(I,J)}{max(I,J)}"] * strain_at(J, pos)
                any_term = True
            if any_term:
                s[I] += dt * acc

        for idx, w in src_pts:
            for I in (1, 2, 3):
                s[I][idx] += wavelet[n] * w

        vx += (dt / rho_vx) * (
            _deriv(xp, s[1], X, coeffs, inv_h, buf["d0"], True)
            + _deriv(xp, s[6], Y, coeffs, inv_h, buf["d1"], False)
            + _deriv(xp, s[5], Z, coeffs, inv_h, buf["d2"], False))
        vy += (dt / rho_vy) * (
            _deriv(xp, s[6], X, coeffs, inv_h, buf["d0"], False)
            + _deriv(xp, s[2], Y, coeffs, inv_h, buf["d1"], True)
            + _deriv(xp, s[4], Z, coeffs, inv_h, buf["d2"], False))
        vz += (dt / rho_vz) * (
            _deriv(xp, s[5], X, coeffs, inv_h, buf["d0"], False)
            + _deriv(xp, s[4], Y, coeffs, inv_h, buf["d1"], False)
            + _deriv(xp, s[3], Z, coeffs, inv_h, buf["d2"], True))

        if damp is not None:
            vx *= damp; vy *= damp; vz *= damp
            for I in range(1, 7):
                s[I] *= damp

        if record == "pressure":
            p = -(s[1] + s[2] + s[3]) / 3.0
        else:
            p = {"vx": vx, "vy": vy, "vz": vz}[record]
        for r, (pts, wts) in enumerate(grp):
            vals = xp.stack([p[q] for q in pts])
            rec[n, r] = float(xp.asnumpy(vals @ xp.asarray(wts, dtype))
                              if use_gpu else vals @ wts)
        if progress is not None and (n % 200 == 0):
            progress(n / nt)
    return rec


# ═══════════════════ fused CUDA kernels (item 3) ═══════════════════════
# The array-op implementation above is memory-bandwidth bound: every
# derivative is a separate full-grid read-modify-write, and the stress update
# walks up to 36 of them per step. Measured against the reference solver it
# was marginally SLOWER (5.5s vs 5.2s), so buffer pre-allocation alone bought
# nothing. These kernels read each field once into registers, do all the
# stencil taps there, and write once.
#
# Tap indexing reproduces _deriv exactly, so results are comparable term by
# term (float addition order aside):
#   forward : out[i] += c_j * (f[i+j+1] - f[i-j]),   valid j <= i < n-j-1
#   backward: out[i] += c_j * (f[i+j]   - f[i-j-1]), valid j+1 <= i < n-j

_VEL_SRC = r'''
extern "C" __global__
void vel_update(const float* __restrict__ s1, const float* __restrict__ s2,
                const float* __restrict__ s3, const float* __restrict__ s4,
                const float* __restrict__ s5, const float* __restrict__ s6,
                float* __restrict__ vx, float* __restrict__ vy,
                float* __restrict__ vz,
                const float* __restrict__ rvx, const float* __restrict__ rvy,
                const float* __restrict__ rvz,
                const float* __restrict__ c, const int nc,
                const float dt_h, const int nz, const int ny, const int nx)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long ncell = (long)nz * ny * nx;
    if (idx >= ncell) return;
    int ix = (int)(idx % nx);
    int iy = (int)((idx / nx) % ny);
    int iz = (int)(idx / ((long)nx * ny));

    const long sx = 1, sy = nx, sz = (long)nx * ny;

    float ax = 0.f, ay = 0.f, az = 0.f;
    for (int j = 0; j < nc; ++j) {
        float cj = c[j];
        // ---- x direction
        if (ix >= j && ix < nx - j - 1)            // forward in x
            ax += cj * (s1[idx + (long)(j+1)*sx] - s1[idx - (long)j*sx]);
        if (ix >= j + 1 && ix < nx - j) {          // backward in x
            ay += cj * (s6[idx + (long)j*sx] - s6[idx - (long)(j+1)*sx]);
            az += cj * (s5[idx + (long)j*sx] - s5[idx - (long)(j+1)*sx]);
        }
        // ---- y direction
        if (iy >= j && iy < ny - j - 1)            // forward in y
            ay += cj * (s2[idx + (long)(j+1)*sy] - s2[idx - (long)j*sy]);
        if (iy >= j + 1 && iy < ny - j) {          // backward in y
            ax += cj * (s6[idx + (long)j*sy] - s6[idx - (long)(j+1)*sy]);
            az += cj * (s4[idx + (long)j*sy] - s4[idx - (long)(j+1)*sy]);
        }
        // ---- z direction
        if (iz >= j && iz < nz - j - 1)            // forward in z
            az += cj * (s3[idx + (long)(j+1)*sz] - s3[idx - (long)j*sz]);
        if (iz >= j + 1 && iz < nz - j) {          // backward in z
            ax += cj * (s5[idx + (long)j*sz] - s5[idx - (long)(j+1)*sz]);
            ay += cj * (s4[idx + (long)j*sz] - s4[idx - (long)(j+1)*sz]);
        }
    }
    vx[idx] += dt_h * ax / rvx[idx];
    vy[idx] += dt_h * ay / rvy[idx];
    vz[idx] += dt_h * az / rvz[idx];
}
'''

_KERNEL_CACHE = {}


def _get_vel_kernel():
    if "vel" not in _KERNEL_CACHE:
        import cupy as cp
        _KERNEL_CACHE["vel"] = cp.RawKernel(_VEL_SRC, "vel_update")
    return _KERNEL_CACHE["vel"]


# The stress update is the expensive half: 9 derivatives for the strains, then
# up to 36 stiffness products, each of which the array path runs as its own
# full-grid pass. Split into two kernels rather than one, because the
# staggered-position moves need neighbour access on the strains, so they have
# to exist before they can be moved.
#
# Position codes (z, y, x half-cell offsets), matching _POS:
#   0 = c (0,0,0)   1 = yz (1,1,0)   2 = xz (1,0,1)   3 = xy (0,1,1)

_STRAIN_SRC = r'''
extern "C" __global__
void strains(const float* __restrict__ vx, const float* __restrict__ vy,
             const float* __restrict__ vz,
             float* __restrict__ g1, float* __restrict__ g2,
             float* __restrict__ g3, float* __restrict__ g4,
             float* __restrict__ g5, float* __restrict__ g6,
             const float* __restrict__ c, const int nc, const float inv_h,
             const int nz, const int ny, const int nx)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)nz * ny * nx) return;
    int ix = (int)(idx % nx);
    int iy = (int)((idx / nx) % ny);
    int iz = (int)(idx / ((long)nx * ny));
    const long sx = 1, sy = nx, sz = (long)nx * ny;

    float a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f;
    for (int j = 0; j < nc; ++j) {
        float cj = c[j];
        if (ix >= j+1 && ix < nx-j)                        // Db_x(vx)
            a1 += cj * (vx[idx + (long)j*sx] - vx[idx - (long)(j+1)*sx]);
        if (iy >= j+1 && iy < ny-j)                        // Db_y(vy)
            a2 += cj * (vy[idx + (long)j*sy] - vy[idx - (long)(j+1)*sy]);
        if (iz >= j+1 && iz < nz-j)                        // Db_z(vz)
            a3 += cj * (vz[idx + (long)j*sz] - vz[idx - (long)(j+1)*sz]);
        if (iy >= j && iy < ny-j-1)                        // Df_y(vz)
            a4 += cj * (vz[idx + (long)(j+1)*sy] - vz[idx - (long)j*sy]);
        if (iz >= j && iz < nz-j-1)                        // Df_z(vy)
            a4 += cj * (vy[idx + (long)(j+1)*sz] - vy[idx - (long)j*sz]);
        if (ix >= j && ix < nx-j-1)                        // Df_x(vz)
            a5 += cj * (vz[idx + (long)(j+1)*sx] - vz[idx - (long)j*sx]);
        if (iz >= j && iz < nz-j-1)                        // Df_z(vx)
            a5 += cj * (vx[idx + (long)(j+1)*sz] - vx[idx - (long)j*sz]);
        if (ix >= j && ix < nx-j-1)                        // Df_x(vy)
            a6 += cj * (vy[idx + (long)(j+1)*sx] - vy[idx - (long)j*sx]);
        if (iy >= j && iy < ny-j-1)                        // Df_y(vx)
            a6 += cj * (vx[idx + (long)(j+1)*sy] - vx[idx - (long)j*sy]);
    }
    g1[idx]=a1*inv_h; g2[idx]=a2*inv_h; g3[idx]=a3*inv_h;
    g4[idx]=a4*inv_h; g5[idx]=a5*inv_h; g6[idx]=a6*inv_h;
}
'''

_STRESS_SRC = r'''
__device__ __forceinline__ int _off(int code, int ax)
{   // half-cell offset of position `code` along axis ax (0=z,1=y,2=x)
    const int P[4][3] = {{0,0,0},{1,1,0},{1,0,1},{0,1,1}};
    return P[code][ax];
}

// value of strain `g` moved from position `spos` to `dpos` at this cell.
// Separable half-cell averaging, reproduced as a product-set average so it
// matches the sequential _avg_axis calls in the array path exactly.
__device__ float _moved(const float* __restrict__ g, int spos, int dpos,
                        long idx, int iz, int iy, int ix,
                        int nz, int ny, int nx, long sy, long sz)
{
    int d[3]; long strd[3]; int pos[3], n[3];
    d[0]=_off(dpos,0)-_off(spos,0); strd[0]=sz; pos[0]=iz; n[0]=nz;
    d[1]=_off(dpos,1)-_off(spos,1); strd[1]=sy; pos[1]=iy; n[1]=ny;
    d[2]=_off(dpos,2)-_off(spos,2); strd[2]=1;  pos[2]=ix; n[2]=nx;

    int lo[3], hi[3];
    float w = 1.0f;
    for (int a = 0; a < 3; ++a) {
        if (d[a] > 0) {                      // integer -> half: {0, +1}
            if (pos[a] > n[a]-2) return 0.0f;
            lo[a]=0; hi[a]=1; w *= 0.5f;
        } else if (d[a] < 0) {               // half -> integer: {-1, 0}
            if (pos[a] < 1) return 0.0f;
            lo[a]=-1; hi[a]=0; w *= 0.5f;
        } else { lo[a]=0; hi[a]=0; }
    }
    float acc = 0.0f;
    for (int a0 = lo[0]; a0 <= hi[0]; ++a0)
      for (int a1 = lo[1]; a1 <= hi[1]; ++a1)
        for (int a2 = lo[2]; a2 <= hi[2]; ++a2)
            acc += g[idx + a0*strd[0] + a1*strd[1] + a2*strd[2]];
    return acc * w;
}

extern "C" __global__
void stress_acc(const float* __restrict__ g1, const float* __restrict__ g2,
                const float* __restrict__ g3, const float* __restrict__ g4,
                const float* __restrict__ g5, const float* __restrict__ g6,
                float* __restrict__ s1, float* __restrict__ s2,
                float* __restrict__ s3, float* __restrict__ s4,
                float* __restrict__ s5, float* __restrict__ s6,
                const float* __restrict__ Cflat,   // 36 planes, row-major I,J
                const int* __restrict__ act,       // 36 flags
                const float dt, const int nz, const int ny, const int nx)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long ncell = (long)nz * ny * nx;
    if (idx >= ncell) return;
    int ix = (int)(idx % nx);
    int iy = (int)((idx / nx) % ny);
    int iz = (int)(idx / ((long)nx * ny));
    const long sy = nx, sz = (long)nx * ny;

    const float* G[6] = {g1,g2,g3,g4,g5,g6};
    float*       S[6] = {s1,s2,s3,s4,s5,s6};
    const int VPOS[6] = {0,0,0,1,2,3};

    for (int I = 0; I < 6; ++I) {
        float acc = 0.0f;
        int any = 0;
        for (int J = 0; J < 6; ++J) {
            if (!act[I*6+J]) continue;
            float gv = (VPOS[J]==VPOS[I]) ? G[J][idx]
                     : _moved(G[J], VPOS[J], VPOS[I], idx, iz,iy,ix, nz,ny,nx, sy,sz);
            // long long: under Windows LLP64 `long` is 32-bit and the
            // 36-plane offset overflows above ~59.7M cells (review finding)
            acc += Cflat[(long long)(I*6+J)*(long long)ncell + idx] * gv;
            any = 1;
        }
        if (any) S[I][idx] += dt * acc;
    }
}
'''


def _get_kernel(name, src, fn, options=()):
    key = (name, tuple(options))
    if key not in _KERNEL_CACHE:
        import cupy as cp
        _KERNEL_CACHE[key] = cp.RawKernel(src, fn, options=tuple(options))
    return _KERNEL_CACHE[key]


def _stress_unrolled_src():
    """Generate a fully unrolled stress kernel.

    Profiling showed the generic stress kernel is 89% of the step time at
    0.31 G cells/s (the other kernels run at 7-8): runtime-indexed pointer
    arrays and _movedL's runtime loops defeat the compiler. Every (I, J)
    term's position move is a compile-time constant, so this generator
    emits all 36 terms with hardcoded offsets and guards. The tap ORDER
    inside each moved average and the J-ascending accumulation replicate
    the generic kernel exactly, so the result is bit-identical - enforced
    by validate_labels.py --quick, which is the gate for ANY edit here.
    """
    POS = {0: (0, 0, 0), 1: (1, 1, 0), 2: (1, 0, 1), 3: (0, 1, 1)}
    VP = [0, 0, 0, 1, 2, 3]
    GN = ["g1", "g2", "g3", "g4", "g5", "g6"]
    SN = ["s1", "s2", "s3", "s4", "s5", "s6"]
    AXV = [("iz", "nz", "sz"), ("iy", "ny", "sy"), ("ix", "nx", "(long)1")]

    def moved_expr(gname, spos, dpos):
        d = [POS[dpos][a] - POS[spos][a] for a in range(3)]
        conds, taps = [], [[]]
        for a in range(3):
            pos, n, stride = AXV[a]
            if d[a] > 0:
                conds.append(f"{pos} <= {n}-2")
                offs = [0, 1]
            elif d[a] < 0:
                conds.append(f"{pos} >= 1")
                offs = [-1, 0]
            else:
                offs = [0]
            taps = [t + [(stride, o)] for t in taps for o in offs]
        terms = []
        for t in taps:
            off = " + ".join(f"({s})*({o})" for s, o in t if o != 0)
            terms.append(f"{gname}[gidx{(' + ' + off) if off else ''}]")
        expr = " + ".join(terms)
        w = 0.25 if len(taps) == 4 else (0.5 if len(taps) == 2 else 1.0)
        cond = " && ".join(conds)
        return f"(({cond}) ? ({expr}) * {w}f : 0.0f)"

    lines = []
    for p in range(4):
        for J in range(6):
            if VP[J] == p:
                lines.append(f"    float q{p}_{J} = {GN[J]}[gidx];")
            else:
                lines.append(f"    float q{p}_{J} = "
                             f"{moved_expr(GN[J], VP[J], p)};")
    for I in range(6):
        p = VP[I]
        lines.append(f"    {{ float acc = 0.0f; int any = 0;")
        for J in range(6):
            lines.append(f"      if (act[{I*6+J}]) "
                         f"{{ acc += Cm[{I*6+J}] * q{p}_{J}; any = 1; }}")
        # DEFERRED damping: the dense path damps s at the END of step n and
        # this kernel adds dt*acc at step n+1; multiplying by damp on the
        # way in commutes those two writes exactly (kV always reads s
        # BETWEEN the add and the damp in both schemes, and receiver /
        # source cells sit at damp = 1). Saves the whole damp_s pass:
        # ~50 B/cell-step of traffic.
        lines.append(f"      float sv = {SN[I]}[idx] * dmp;")
        lines.append(f"      if (any) sv += dt * acc;")
        lines.append(f"      {SN[I]}[idx] = sv; }}")
    body = "\n".join(lines)

    return r'''
extern "C" __global__
void stress_lab_u(const float* __restrict__ g1, const float* __restrict__ g2,
                  const float* __restrict__ g3, const float* __restrict__ g4,
                  const float* __restrict__ g5, const float* __restrict__ g6,
                  float* __restrict__ s1, float* __restrict__ s2,
                  float* __restrict__ s3, float* __restrict__ s4,
                  float* __restrict__ s5, float* __restrict__ s6,
                  const unsigned char* __restrict__ lab,
                  const float* __restrict__ Ct,
                  const int* __restrict__ act,
                  const float* __restrict__ damp, const int use_damp,
                  const float dt, const int nz, const int ny, const int nx,
                  const int z0, const int z1, const int zg0)
{
    long lidx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long ncore = (long)(z1 - z0) * ny * nx;
    if (lidx >= ncore) return;
    int ix = (int)(lidx % nx);
    int iy = (int)((lidx / nx) % ny);
    int iz = z0 + (int)(lidx / ((long)nx * ny));
    long idx  = ((long)iz * ny + iy) * nx + ix;
    long gidx = ((long)(iz - zg0) * ny + iy) * nx + ix;
    const long sy = nx, sz = (long)nx * ny;
    const float* Cm = Ct + (long)lab[idx] * 36;
    const float dmp = use_damp ? damp[idx] : 1.0f;

''' + body + r'''
}
'''


def _stress_smem_src(TX=32, TY=8):
    """Shared-memory-tiled twin of stress_lab_u - bit-identical maths.

    NEGATIVE RESULT (2026-07-29), kept for the record like the fused
    kernel before it: gate PASSED at 0.00e+00 (full 2822 steps) but at
    194M cells it measured 129 ms/step vs 112 for the unrolled kernel -
    the halo loads (1.33x overfetch) plus two __syncthreads per z-plane
    plus the occupancy cost of 24.5 KB smem/block exceed the L2 saving.
    Together with the fused-step result this brackets the design space:
    on this GPU the L2 handles the g windows better than explicit
    staging, and stress_lab_u (default) is within ~2x of the summed
    per-kernel DRAM floors. Do not re-attempt smem staging here without
    a DIFFERENT mechanism (e.g. persistent-blocks z-marching that also
    caches s in registers).

    Original motivation, for context: the six g slabs' rolling 3-plane
    z-windows (23 MB) contend for the L2 and the unrolled kernel
    re-reads each g value up to ~9x from L2 (measured 43.7 ms vs a
    ~22 ms DRAM floor). Staging 3 rolling z-planes of all six g arrays
    per 32x8 tile makes each g value one DRAM read. Same expression
    tree, tap order and weight literals as stress_lab_u -> bit-identical
    (validate_labels.py --quick is the gate for ANY edit here).
    """
    POS = {0: (0, 0, 0), 1: (1, 1, 0), 2: (1, 0, 1), 3: (0, 1, 1)}
    VP = [0, 0, 0, 1, 2, 3]
    SN = ["s1", "s2", "s3", "s4", "s5", "s6"]
    NAMES = ["iz", "iy", "ix"]
    DIMS = ["nz", "ny", "nx"]
    SLOT = {-1: "sm1", 0: "s0c", 1: "sp1"}

    def moved_expr(J, spos, dpos):
        d = [POS[dpos][a] - POS[spos][a] for a in range(3)]
        conds, taps = [], [[]]
        for a in range(3):
            if d[a] > 0:
                conds.append(f"{NAMES[a]} <= {DIMS[a]}-2")
                offs = [0, 1]
            elif d[a] < 0:
                conds.append(f"{NAMES[a]} >= 1")
                offs = [-1, 0]
            else:
                offs = [0]
            taps = [t + [o] for t in taps for o in offs]
        terms = [f"G{J+1}[{SLOT[t[0]]}][ty+1+({t[1]})][tx+1+({t[2]})]"
                 for t in taps]
        expr = " + ".join(terms)
        w = 0.25 if len(taps) == 4 else (0.5 if len(taps) == 2 else 1.0)
        return f"(({' && '.join(conds)}) ? ({expr}) * {w}f : 0.0f)"

    lines = []
    for p in range(4):
        for J in range(6):
            if VP[J] == p:
                lines.append(f"            float q{p}_{J} = "
                             f"G{J+1}[s0c][ty+1][tx+1];")
            else:
                lines.append(f"            float q{p}_{J} = "
                             f"{moved_expr(J, VP[J], p)};")
    for I in range(6):
        p = VP[I]
        lines.append("            { float acc = 0.0f; int any = 0;")
        for J in range(6):
            lines.append(f"              if (act[{I*6+J}]) "
                         f"{{ acc += Cm[{I*6+J}] * q{p}_{J}; any = 1; }}")
        lines.append(f"              float sv = {SN[I]}[idx] * dmp;")
        lines.append("              if (any) sv += dt * acc;")
        lines.append(f"              {SN[I]}[idx] = sv; }}")
    body = "\n".join(lines)

    tmpl = r'''
extern "C" __global__
void stress_lab_smem(const float* __restrict__ g1,
                     const float* __restrict__ g2,
                     const float* __restrict__ g3,
                     const float* __restrict__ g4,
                     const float* __restrict__ g5,
                     const float* __restrict__ g6,
                     float* __restrict__ s1, float* __restrict__ s2,
                     float* __restrict__ s3, float* __restrict__ s4,
                     float* __restrict__ s5, float* __restrict__ s6,
                     const unsigned char* __restrict__ lab,
                     const float* __restrict__ Ct,
                     const int* __restrict__ act,
                     const float* __restrict__ damp, const int use_damp,
                     const float dt, const int nz, const int ny,
                     const int nx, const int z0, const int z1,
                     const int zg0)
{
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int ix = blockIdx.x * TXQ + tx;
    const int iy = blockIdx.y * TYQ + ty;
    const int tid = ty * TXQ + tx;
    const int zg1 = (z1 + 1 < nz) ? (z1 + 1) : nz;
    __shared__ float G1[3][TYQ+2][TXQ+2];
    __shared__ float G2[3][TYQ+2][TXQ+2];
    __shared__ float G3[3][TYQ+2][TXQ+2];
    __shared__ float G4[3][TYQ+2][TXQ+2];
    __shared__ float G5[3][TYQ+2][TXQ+2];
    __shared__ float G6[3][TYQ+2][TXQ+2];
    int sm1 = 0, s0c = 1, sp1 = 2;

#define LOAD_PLANE(PZ, SL) do { \
    int pz = (PZ); \
    for (int q = tid; q < (TYQ+2)*(TXQ+2); q += TXQ*TYQ) { \
        int ly = q / (TXQ+2), lx = q % (TXQ+2); \
        int gy = blockIdx.y * TYQ + ly - 1; \
        int gx = blockIdx.x * TXQ + lx - 1; \
        float v1=0.f, v2=0.f, v3=0.f, v4=0.f, v5=0.f, v6=0.f; \
        if (pz >= zg0 && pz < zg1 && gy >= 0 && gy < ny \
            && gx >= 0 && gx < nx) { \
            long long gp = ((long long)(pz - zg0) * ny + gy) * nx + gx; \
            v1 = g1[gp]; v2 = g2[gp]; v3 = g3[gp]; \
            v4 = g4[gp]; v5 = g5[gp]; v6 = g6[gp]; \
        } \
        G1[SL][ly][lx] = v1; G2[SL][ly][lx] = v2; G3[SL][ly][lx] = v3; \
        G4[SL][ly][lx] = v4; G5[SL][ly][lx] = v5; G6[SL][ly][lx] = v6; \
    } } while (0)

    LOAD_PLANE(z0 - 1, sm1);
    LOAD_PLANE(z0,     s0c);
    for (int iz = z0; iz < z1; ++iz) {
        LOAD_PLANE(iz + 1, sp1);
        __syncthreads();
        if (iy < ny && ix < nx) {
            long long idx = ((long long)iz * ny + iy) * nx + ix;
            const float* Cm = Ct + (long long)lab[idx] * 36;
            const float dmp = use_damp ? damp[idx] : 1.0f;
BODYQ
        }
        __syncthreads();
        int tsw = sm1; sm1 = s0c; s0c = sp1; sp1 = tsw;
    }
#undef LOAD_PLANE
}
'''
    return (tmpl.replace("TXQ", str(TX)).replace("TYQ", str(TY))
                .replace("BODYQ", body))


def forward_fused(C, rho, h, dt, nt, src_pts, wavelet, rec_groups, order=8,
                  coeffs=None, sponge_width=0, sponge_strength=0.09,
                  damp_mask=None, record="pressure", progress=None):
    """Same physics as `forward`, driven by the fused CUDA kernels.

    Requires CUDA. Stiffness is uploaded once as a flat (36, nz, ny, nx) block
    so the stress kernel indexes it without a Python-level dict lookup per
    term, and inactive (I, J) pairs are skipped via a flag array rather than
    by branching in Python.
    """
    import cupy as cp
    if coeffs is None:
        coeffs = optimised_coeffs(order)
    coeffs = np.asarray(coeffs, float)
    cd = cp.asarray(coeffs, cp.float32)

    rho_h = np.asarray(rho, float)
    nz, ny, nx = rho_h.shape

    rvx = cp.asarray(face_density(rho_h, 2), cp.float32)
    rvy = cp.asarray(face_density(rho_h, 1), cp.float32)
    rvz = cp.asarray(face_density(rho_h, 0), cp.float32)

    Cflat = np.zeros((36, nz, ny, nx), np.float32)
    act = np.zeros(36, np.int32)
    for I in range(1, 7):
        for J in range(1, 7):
            a = np.asarray(C[f"C{min(I,J)}{max(I,J)}"], float)
            if not np.any(a):
                continue
            Cflat[(I - 1) * 6 + (J - 1)] = a if a.ndim == 3 else float(a)
            act[(I - 1) * 6 + (J - 1)] = 1
    Cflat = cp.asarray(Cflat).reshape(-1)
    actd = cp.asarray(act)

    v = [cp.zeros((nz, ny, nx), cp.float32) for _ in range(3)]
    s = [cp.zeros((nz, ny, nx), cp.float32) for _ in range(6)]
    g = [cp.zeros((nz, ny, nx), cp.float32) for _ in range(6)]
    damp = (sponge_profile((nz, ny, nx), sponge_width, sponge_strength, cp)
            if sponge_width > 0 else None)
    if damp_mask is not None:
        dm = cp.asarray(np.asarray(damp_mask, np.float32))
        damp = dm if damp is None else damp * dm

    kS = _get_kernel("strain", _STRAIN_SRC, "strains")
    kT = _get_kernel("stress", _STRESS_SRC, "stress_acc")
    kV = _get_vel_kernel()
    tpb = 256
    ncell = nz * ny * nx
    blocks = int((ncell + tpb - 1) // tpb)
    N = (np.int32(nz), np.int32(ny), np.int32(nx))
    inv_h = np.float32(1.0 / h)
    dtf = np.float32(dt)
    dt_h = np.float32(dt / h)

    # Flatten source and receiver footprints ONCE. Looping over element cells
    # in Python cost more than the physics: a 113-cell element meant 339 GPU
    # scalar writes plus 113 scalar reads AND a device sync every single step,
    # which dropped throughput from 1.15 to 0.23 G cell-updates/s.
    shp = (nz, ny, nx)
    s_idx = cp.asarray(np.array([np.ravel_multi_index(tuple(idx), shp)
                                 for idx, _ in src_pts], np.int64))
    s_w = cp.asarray(np.array([w for _, w in src_pts], np.float32))
    r_idx = [cp.asarray(np.array([np.ravel_multi_index(tuple(q), shp)
                                  for q in pts], np.int64))
             for pts, _ in rec_groups]
    r_w = [cp.asarray(np.asarray(w, np.float32)) for _, w in rec_groups]
    rec_d = cp.zeros((nt, len(rec_groups)), cp.float32)   # stays on device

    for n in range(nt):
        kS((blocks,), (tpb,), (v[0], v[1], v[2], *g, cd,
                               np.int32(len(coeffs)), inv_h, *N))
        kT((blocks,), (tpb,), (*g, *s, Cflat, actd, dtf, *N))
        wn = np.float32(wavelet[n])
        for I in range(3):
            s[I].ravel()[s_idx] += wn * s_w
        kV((blocks,), (tpb,), (*s, v[0], v[1], v[2], rvx, rvy, rvz, cd,
                               np.int32(len(coeffs)), dt_h, *N))
        if damp is not None:
            for a in v:
                a *= damp
            for a in s:
                a *= damp
        p = -(s[0] + s[1] + s[2]) / 3.0 if record == "pressure" else \
            {"vx": v[0], "vy": v[1], "vz": v[2]}[record]
        pf = p.ravel()
        for r in range(len(r_idx)):
            rec_d[n, r] = (pf[r_idx[r]] * r_w[r]).sum()
        if progress is not None and (n % 200 == 0):
            progress(n / nt)
    return cp.asnumpy(rec_d)          # ONE transfer, not nt of them


# ─────────────────────── source wavelets and picking ─────────────────────
def ricker(f0, dt, nt, t0=None):
    """Ricker wavelet: the second derivative of a Gaussian, so its integral is
    ZERO by construction.

    This matters because an explosive source accumulates: the update is
    s += wavelet[n] every step, so any DC content in the wavelet integrates
    into a permanent static stress. A Gaussian-modulated cosine looks
    band-limited but still summed to +0.373 in testing, leaving a -0.125
    pedestal on the trace that never decayed. Envelope picking on that is
    meaningless, because the Hilbert envelope of (large DC + small echo) just
    tracks the DC. It is what hid E1 in the earlier full-wave runs.
    """
    t0 = t0 if t0 is not None else 1.2 / f0
    t = np.arange(nt) * dt - t0
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def detrend_hp(x, dt, f_lo, order=2):
    """Remove DC and any slow drift before envelope work."""
    from scipy.signal import butter, filtfilt
    fs = 1.0 / dt
    b, a = butter(order, f_lo / (fs / 2.0), "high")
    return filtfilt(b, a, np.asarray(x, float))


def pick_peak(t, x, t_lo=None, t_hi=None, dt=None, f_lo=None,
              edge_frac=0.04):
    """Envelope peak in a window, with the traps that bit us ruled out.

    * high-passes first, so a DC pedestal cannot dominate the envelope
    * excludes `edge_frac` of the record at BOTH ends, because filtfilt leaves
      transients there that are easily larger than a real echo (one produced a
      spike 12x the true arrival and argmax took it every time)
    * refuses to return a pick that lands on a window edge, which is the
      signature of "no peak found" rather than a detection
    """
    from scipy.signal import hilbert
    x = np.asarray(x, float)
    n = len(x)
    if dt is None:
        dt = float(t[1] - t[0])
    if f_lo:
        x = detrend_hp(x, dt, f_lo)
    else:
        x = x - x.mean()
    e = np.abs(hilbert(x))
    g = max(int(edge_frac * n), 4)
    lo = max(int((t_lo or 0.0) / dt), g)
    hi = min(int((t_hi or t[-1]) / dt), n - g)
    if hi <= lo + 2:
        return np.nan, 0.0, False
    k = int(np.argmax(e[lo:hi]))
    ok = 0 < k < (hi - lo - 1)                  # not sitting on a window edge
    kk = lo + k
    if 0 < kk < n - 1:                          # parabolic refinement
        y0, y1, y2 = e[kk - 1], e[kk], e[kk + 1]
        d = y0 - 2 * y1 + y2
        kk = kk + (0.5 * (y0 - y2) / d if abs(d) > 1e-30 else 0.0)
    return kk * dt, float(e[lo:hi].max()), ok


def face_density(rho, ax):
    """Effective density at velocity nodes offset half a cell along `ax`.

    ARITHMETIC, not harmonic. The velocity update is v_t = (1/rho) div(sigma);
    integrating over the control volume centred on the velocity node gives the
    VOLUME average of the densities it straddles. Harmonic averaging is the
    right treatment for the shear modulus at shear-stress nodes, where
    compliances add in series, and applying it to density instead understates
    it badly: across the ice/fluid rim it gives 452 kg/m3 where the correct
    value is 608, inflating the effective wave speed from 4967 to 5762 m/s and
    pushing the scheme toward instability rather than away from it.
    """
    a = np.asarray(rho, float)
    m = np.empty_like(a)
    hi = [slice(None)] * 3; lo = [slice(None)] * 3
    hi[ax] = slice(1, None); lo[ax] = slice(0, -1)
    m[tuple(lo)] = 0.5 * (a[tuple(lo)] + a[tuple(hi)])
    last = [slice(None)] * 3; last[ax] = slice(-1, None)
    m[tuple(last)] = a[tuple(last)]
    return m


def max_grid_speed(C, rho):
    """Worst-case wave speed the DISCRETE scheme can support on this grid.

    Not the same as the fastest physical wave. The staggered update pairs the
    stiffness of a cell with a FACE-AVERAGED density, so at a sharp contrast a
    stiff cell can end up dividing by a density belonging partly to its light
    neighbour. On a 100 mm ice disc in a light surround that pairs ice C33
    (15.01 GPa) with a face density of 452 kg/m3 instead of 917, giving an
    effective 5763 m/s against the 4046 m/s of ice itself: 42% over budget.

    Setting dt from the physical speed therefore violates CFL in a thin shell
    at the rim. It does not blow up immediately, it grows about 0.2% per step,
    which is invisible over a few hundred steps and fatal over several
    thousand. That is the bug that wrecked every full-disc run in this project,
    including the original elastic3d ones.
    """
    rho_h = np.asarray(rho, float)
    diag = [np.asarray(C[k], float) for k in ("C11", "C22", "C33")]
    Cmax = np.maximum.reduce([np.broadcast_to(a, rho_h.shape) for a in diag])

    rmin = rho_h.copy()
    for ax in range(3):
        face = face_density(rho_h, ax)
        np.minimum(rmin, face, out=rmin)
    return float(np.sqrt(Cmax / rmin).max())


def safe_dt(C, rho, h, coeffs, safety=0.85):
    """Largest stable timestep for this grid and stencil."""
    return safety * cfl_limit(np.asarray(coeffs, float)) * h / max_grid_speed(C, rho)


# ═══════════════ label-indexed solver (5 MHz memory fix) ═════════════════
# forward_fused stores stiffness as 36 FULL float32 volumes (symmetric pairs
# duplicated): 144 B/cell of a ~220 B/cell footprint, which caps the 4070 at
# 3 MHz (a 5 MHz full disk is 194M cells ~ 43 GB). But the medium has only
# ~100 distinct materials, so a 1-byte material label per cell plus a
# (n_mat, 36) lookup table carries identical information. Likewise the strain
# scratch (24 B/cell) is only ever consumed within +/-1 cell by the staggered
# moves, so strains are computed in z-slabs with a one-cell halo instead of
# full volumes. Net: ~41 B/cell -> a 5 MHz full disk fits in ~8.5 GB.
#
# The NUMERICS ARE UNCHANGED: same stencils, same per-cell stiffness values
# (looked up rather than stored), same arithmetic face-density rule, same
# damping. Validate against ref/fw_fabric00.npz before trusting any run.

_STRAIN_SLAB_SRC = r'''
extern "C" __global__
void strains_slab(const float* __restrict__ vx, const float* __restrict__ vy,
                  const float* __restrict__ vz,
                  float* __restrict__ g1, float* __restrict__ g2,
                  float* __restrict__ g3, float* __restrict__ g4,
                  float* __restrict__ g5, float* __restrict__ g6,
                  const float* __restrict__ c, const int nc,
                  const float inv_h, const int nz, const int ny,
                  const int nx, const int zg0, const int zg1)
{
    long lidx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long nslab = (long)(zg1 - zg0) * ny * nx;
    if (lidx >= nslab) return;
    int ix = (int)(lidx % nx);
    int iy = (int)((lidx / nx) % ny);
    int iz = zg0 + (int)(lidx / ((long)nx * ny));
    long idx = ((long)iz * ny + iy) * nx + ix;
    const long sx = 1, sy = nx, sz = (long)nx * ny;

    float a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f;
    for (int j = 0; j < nc; ++j) {
        float cj = c[j];
        if (ix >= j+1 && ix < nx-j)
            a1 += cj * (vx[idx + (long)j*sx] - vx[idx - (long)(j+1)*sx]);
        if (iy >= j+1 && iy < ny-j)
            a2 += cj * (vy[idx + (long)j*sy] - vy[idx - (long)(j+1)*sy]);
        if (iz >= j+1 && iz < nz-j)
            a3 += cj * (vz[idx + (long)j*sz] - vz[idx - (long)(j+1)*sz]);
        if (iy >= j && iy < ny-j-1)
            a4 += cj * (vz[idx + (long)(j+1)*sy] - vz[idx - (long)j*sy]);
        if (iz >= j && iz < nz-j-1)
            a4 += cj * (vy[idx + (long)(j+1)*sz] - vy[idx - (long)j*sz]);
        if (ix >= j && ix < nx-j-1)
            a5 += cj * (vz[idx + (long)(j+1)*sx] - vz[idx - (long)j*sx]);
        if (iz >= j && iz < nz-j-1)
            a5 += cj * (vx[idx + (long)(j+1)*sz] - vx[idx - (long)j*sz]);
        if (ix >= j && ix < nx-j-1)
            a6 += cj * (vy[idx + (long)(j+1)*sx] - vy[idx - (long)j*sx]);
        if (iy >= j && iy < ny-j-1)
            a6 += cj * (vx[idx + (long)(j+1)*sy] - vx[idx - (long)j*sy]);
    }
    g1[lidx]=a1*inv_h; g2[lidx]=a2*inv_h; g3[lidx]=a3*inv_h;
    g4[lidx]=a4*inv_h; g5[lidx]=a5*inv_h; g6[lidx]=a6*inv_h;
}
'''

_STRAIN_ZP_SRC = r'''
// z-column register pipeline (route C, practical form): at 194M cells the
// +/-4-plane z-taps (23 MB of window per field) fall out of the 36 MB L2
// and every tap is a DRAM miss - measured 3.5 G cells/s vs 7.4 on small
// grids. Each thread owns one (x, y) column and marches z, carrying the
// nine z-planes it taps in registers; x/y taps stay as global reads with
// same-plane cache locality. Term order inside the j-loop is IDENTICAL to
// strains_slab, so the result is bit-identical (gated).
extern "C" __global__
void strains_zp(const float* __restrict__ vx, const float* __restrict__ vy,
                const float* __restrict__ vz,
                float* __restrict__ g1, float* __restrict__ g2,
                float* __restrict__ g3, float* __restrict__ g4,
                float* __restrict__ g5, float* __restrict__ g6,
                const float* __restrict__ c, const int nc,
                const float inv_h, const int nz, const int ny,
                const int nx, const int zg0, const int zg1)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y;
    if (ix >= nx || iy >= ny) return;
    const long sx = 1, sy = nx, sz = (long)nx * ny;
    const long base = (long)iy * nx + ix;

    float rx[9], ry[9], rz9[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        int z = zg0 - 4 + k;
        bool ok = (z >= 0 && z < nz);
        long p = base + (long)z * sz;
        rx[k]  = ok ? vx[p] : 0.f;
        ry[k]  = ok ? vy[p] : 0.f;
        rz9[k] = ok ? vz[p] : 0.f;
    }
    for (int iz = zg0; iz < zg1; ++iz) {
        long idx  = base + (long)iz * sz;
        long lidx = base + (long)(iz - zg0) * sz;
        float a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j >= nc) break;
            float cj = c[j];
            if (ix >= j+1 && ix < nx-j)
                a1 += cj * (vx[idx + (long)j*sx] - vx[idx - (long)(j+1)*sx]);
            if (iy >= j+1 && iy < ny-j)
                a2 += cj * (vy[idx + (long)j*sy] - vy[idx - (long)(j+1)*sy]);
            if (iz >= j+1 && iz < nz-j)
                a3 += cj * (rz9[4+j] - rz9[3-j]);
            if (iy >= j && iy < ny-j-1)
                a4 += cj * (vz[idx + (long)(j+1)*sy] - vz[idx - (long)j*sy]);
            if (iz >= j && iz < nz-j-1)
                a4 += cj * (ry[5+j] - ry[4-j]);
            if (ix >= j && ix < nx-j-1)
                a5 += cj * (vz[idx + (long)(j+1)*sx] - vz[idx - (long)j*sx]);
            if (iz >= j && iz < nz-j-1)
                a5 += cj * (rx[5+j] - rx[4-j]);
            if (ix >= j && ix < nx-j-1)
                a6 += cj * (vy[idx + (long)(j+1)*sx] - vy[idx - (long)j*sx]);
            if (iy >= j && iy < ny-j-1)
                a6 += cj * (vx[idx + (long)(j+1)*sy] - vx[idx - (long)j*sy]);
        }
        g1[lidx]=a1*inv_h; g2[lidx]=a2*inv_h; g3[lidx]=a3*inv_h;
        g4[lidx]=a4*inv_h; g5[lidx]=a5*inv_h; g6[lidx]=a6*inv_h;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            rx[k]=rx[k+1]; ry[k]=ry[k+1]; rz9[k]=rz9[k+1];
        }
        int zn = iz + 5;
        bool ok = (zn >= 0 && zn < nz);
        long p = base + (long)zn * sz;
        rx[8]  = ok ? vx[p] : 0.f;
        ry[8]  = ok ? vy[p] : 0.f;
        rz9[8] = ok ? vz[p] : 0.f;
    }
}
'''

_VEL_ZP_SRC = r'''
// z-pipeline velocity update: only s3, s4, s5 carry z-taps, so three ring
// buffers suffice; s1/s2/s6 x/y taps stay as cache-local global reads.
// Same term order as vel_lab - bit-identical (gated).
extern "C" __global__
void vel_zp(const float* __restrict__ s1, const float* __restrict__ s2,
            const float* __restrict__ s3, const float* __restrict__ s4,
            const float* __restrict__ s5, const float* __restrict__ s6,
            float* __restrict__ vx, float* __restrict__ vy,
            float* __restrict__ vz,
            const unsigned char* __restrict__ lab,
            const float* __restrict__ rho_t,
            const float* __restrict__ damp, const int use_damp,
            const float* __restrict__ c, const int nc,
            const float dt_h, const int nz, const int ny, const int nx)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y;
    if (ix >= nx || iy >= ny) return;
    const long sx = 1, sy = nx, sz = (long)nx * ny;
    const long base = (long)iy * nx + ix;

    float r3[9], r4[9], r5[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        int z = k - 4;
        bool ok = (z >= 0 && z < nz);
        long p = base + (long)z * sz;
        r3[k] = ok ? s3[p] : 0.f;
        r4[k] = ok ? s4[p] : 0.f;
        r5[k] = ok ? s5[p] : 0.f;
    }
    for (int iz = 0; iz < nz; ++iz) {
        long idx = base + (long)iz * sz;
        float ax = 0.f, ay = 0.f, az = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j >= nc) break;
            float cj = c[j];
            if (ix >= j && ix < nx - j - 1)
                ax += cj * (s1[idx + (long)(j+1)*sx] - s1[idx - (long)j*sx]);
            if (ix >= j + 1 && ix < nx - j) {
                ay += cj * (s6[idx + (long)j*sx] - s6[idx - (long)(j+1)*sx]);
                az += cj * (s5[idx + (long)j*sx] - s5[idx - (long)(j+1)*sx]);
            }
            if (iy >= j && iy < ny - j - 1)
                ay += cj * (s2[idx + (long)(j+1)*sy] - s2[idx - (long)j*sy]);
            if (iy >= j + 1 && iy < ny - j) {
                ax += cj * (s6[idx + (long)j*sy] - s6[idx - (long)(j+1)*sy]);
                az += cj * (s4[idx + (long)j*sy] - s4[idx - (long)(j+1)*sy]);
            }
            if (iz >= j && iz < nz - j - 1)
                az += cj * (r3[5+j] - r3[4-j]);
            if (iz >= j + 1 && iz < nz - j) {
                ax += cj * (r5[4+j] - r5[3-j]);
                ay += cj * (r4[4+j] - r4[3-j]);
            }
        }
        float r0 = rho_t[lab[idx]];
        float rvx = (ix < nx-1) ? 0.5f*(r0 + rho_t[lab[idx+sx]]) : r0;
        float rvy = (iy < ny-1) ? 0.5f*(r0 + rho_t[lab[idx+sy]]) : r0;
        float rvz = (iz < nz-1) ? 0.5f*(r0 + rho_t[lab[idx+sz]]) : r0;
        float dmp = use_damp ? damp[idx] : 1.0f;
        vx[idx] = (vx[idx] + dt_h * ax / rvx) * dmp;
        vy[idx] = (vy[idx] + dt_h * ay / rvy) * dmp;
        vz[idx] = (vz[idx] + dt_h * az / rvz) * dmp;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            r3[k]=r3[k+1]; r4[k]=r4[k+1]; r5[k]=r5[k+1];
        }
        int zn = iz + 5;
        bool ok = (zn >= 0 && zn < nz);
        long p = base + (long)zn * sz;
        r3[8] = ok ? s3[p] : 0.f;
        r4[8] = ok ? s4[p] : 0.f;
        r5[8] = ok ? s5[p] : 0.f;
    }
}
'''

_STRESS_LAB_SRC = r'''
__device__ __forceinline__ int _offL(int code, int ax)
{
    const int P[4][3] = {{0,0,0},{1,1,0},{1,0,1},{0,1,1}};
    return P[code][ax];
}

// identical maths to _moved in the dense kernel, on the slab-local g array
__device__ float _movedL(const float* __restrict__ g, int spos, int dpos,
                         long gidx, int iz, int iy, int ix,
                         int nz, int ny, int nx, long sy, long sz)
{
    int d[3]; long strd[3]; int pos[3], n[3];
    d[0]=_offL(dpos,0)-_offL(spos,0); strd[0]=sz; pos[0]=iz; n[0]=nz;
    d[1]=_offL(dpos,1)-_offL(spos,1); strd[1]=sy; pos[1]=iy; n[1]=ny;
    d[2]=_offL(dpos,2)-_offL(spos,2); strd[2]=1;  pos[2]=ix; n[2]=nx;

    int lo[3], hi[3];
    float w = 1.0f;
    for (int a = 0; a < 3; ++a) {
        if (d[a] > 0) {
            if (pos[a] > n[a]-2) return 0.0f;
            lo[a]=0; hi[a]=1; w *= 0.5f;
        } else if (d[a] < 0) {
            if (pos[a] < 1) return 0.0f;
            lo[a]=-1; hi[a]=0; w *= 0.5f;
        } else { lo[a]=0; hi[a]=0; }
    }
    float acc = 0.0f;
    for (int a0 = lo[0]; a0 <= hi[0]; ++a0)
      for (int a1 = lo[1]; a1 <= hi[1]; ++a1)
        for (int a2 = lo[2]; a2 <= hi[2]; ++a2)
            acc += g[gidx + a0*strd[0] + a1*strd[1] + a2*strd[2]];
    return acc * w;
}

extern "C" __global__
void stress_lab(const float* __restrict__ g1, const float* __restrict__ g2,
                const float* __restrict__ g3, const float* __restrict__ g4,
                const float* __restrict__ g5, const float* __restrict__ g6,
                float* __restrict__ s1, float* __restrict__ s2,
                float* __restrict__ s3, float* __restrict__ s4,
                float* __restrict__ s5, float* __restrict__ s6,
                const unsigned char* __restrict__ lab,
                const float* __restrict__ Ct,      // (n_mat, 36)
                const int* __restrict__ act,       // 36 flags
                const float dt, const int nz, const int ny, const int nx,
                const int z0, const int z1, const int zg0)
{
    long lidx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long ncore = (long)(z1 - z0) * ny * nx;
    if (lidx >= ncore) return;
    int ix = (int)(lidx % nx);
    int iy = (int)((lidx / nx) % ny);
    int iz = z0 + (int)(lidx / ((long)nx * ny));
    long idx  = ((long)iz * ny + iy) * nx + ix;
    long gidx = ((long)(iz - zg0) * ny + iy) * nx + ix;
    const long sy = nx, sz = (long)nx * ny;

    const float* G[6] = {g1,g2,g3,g4,g5,g6};
    float*       S[6] = {s1,s2,s3,s4,s5,s6};
    const int VPOS[6] = {0,0,0,1,2,3};
    const float* Cm = Ct + (long)lab[idx] * 36;
    // NO damping here: the velocity update must read UNDAMPED stress (the
    // dense path damps s only after kV). Folding damp into this kernel
    // changed the absorber discretisation and broke bit-equivalence
    // (1e-3). SUPERSEDED: production uses stress_lab_u with DEFERRED
    // damping (commutes); damp_s no longer exists in the driver. Kept as
    // the readable reference implementation only.

    for (int I = 0; I < 6; ++I) {
        float acc = 0.0f;
        int any = 0;
        for (int J = 0; J < 6; ++J) {
            if (!act[I*6+J]) continue;
            float gv = (VPOS[J]==VPOS[I]) ? G[J][gidx]
                     : _movedL(G[J], VPOS[J], VPOS[I], gidx,
                               iz,iy,ix, nz,ny,nx, sy,sz);
            acc += Cm[I*6+J] * gv;
            any = 1;
        }
        if (any) S[I][idx] += dt * acc;
    }
}
'''

_DAMP_S_SRC = r'''
extern "C" __global__
void damp_s(float* __restrict__ s1, float* __restrict__ s2,
            float* __restrict__ s3, float* __restrict__ s4,
            float* __restrict__ s5, float* __restrict__ s6,
            const float* __restrict__ damp, const long ncell)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncell) return;
    float d = damp[idx];
    s1[idx] *= d; s2[idx] *= d; s3[idx] *= d;
    s4[idx] *= d; s5[idx] *= d; s6[idx] *= d;
}
'''

_VEL_LAB_SRC = r'''
extern "C" __global__
void vel_lab(const float* __restrict__ s1, const float* __restrict__ s2,
             const float* __restrict__ s3, const float* __restrict__ s4,
             const float* __restrict__ s5, const float* __restrict__ s6,
             float* __restrict__ vx, float* __restrict__ vy,
             float* __restrict__ vz,
             const unsigned char* __restrict__ lab,
             const float* __restrict__ rho_t,
             const float* __restrict__ damp, const int use_damp,
             const float* __restrict__ c, const int nc,
             const float dt_h, const int nz, const int ny, const int nx)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long ncell = (long)nz * ny * nx;
    if (idx >= ncell) return;
    int ix = (int)(idx % nx);
    int iy = (int)((idx / nx) % ny);
    int iz = (int)(idx / ((long)nx * ny));
    const long sx = 1, sy = nx, sz = (long)nx * ny;

    float ax = 0.f, ay = 0.f, az = 0.f;
    for (int j = 0; j < nc; ++j) {
        float cj = c[j];
        if (ix >= j && ix < nx - j - 1)
            ax += cj * (s1[idx + (long)(j+1)*sx] - s1[idx - (long)j*sx]);
        if (ix >= j + 1 && ix < nx - j) {
            ay += cj * (s6[idx + (long)j*sx] - s6[idx - (long)(j+1)*sx]);
            az += cj * (s5[idx + (long)j*sx] - s5[idx - (long)(j+1)*sx]);
        }
        if (iy >= j && iy < ny - j - 1)
            ay += cj * (s2[idx + (long)(j+1)*sy] - s2[idx - (long)j*sy]);
        if (iy >= j + 1 && iy < ny - j) {
            ax += cj * (s6[idx + (long)j*sy] - s6[idx - (long)(j+1)*sy]);
            az += cj * (s4[idx + (long)j*sy] - s4[idx - (long)(j+1)*sy]);
        }
        if (iz >= j && iz < nz - j - 1)
            az += cj * (s3[idx + (long)(j+1)*sz] - s3[idx - (long)j*sz]);
        if (iz >= j + 1 && iz < nz - j) {
            ax += cj * (s5[idx + (long)j*sz] - s5[idx - (long)(j+1)*sz]);
            ay += cj * (s4[idx + (long)j*sz] - s4[idx - (long)(j+1)*sz]);
        }
    }
    // arithmetic face density from the material table, matching face_density
    // exactly (including the last-slice copy at the +edge)
    float r0 = rho_t[lab[idx]];
    float rvx = (ix < nx-1) ? 0.5f*(r0 + rho_t[lab[idx+sx]]) : r0;
    float rvy = (iy < ny-1) ? 0.5f*(r0 + rho_t[lab[idx+sy]]) : r0;
    float rvz = (iz < nz-1) ? 0.5f*(r0 + rho_t[lab[idx+sz]]) : r0;
    float dmp = use_damp ? damp[idx] : 1.0f;
    vx[idx] = (vx[idx] + dt_h * ax / rvx) * dmp;
    vy[idx] = (vy[idx] + dt_h * ay / rvy) * dmp;
    vz[idx] = (vz[idx] + dt_h * az / rvz) * dmp;
}
'''


def _fused_step_src(TZ=4, TY=8, TX=16):
    """Generate the fused strain+stress kernel with shared-memory strains.

    At 194M cells the strain slab (125 MB) no longer fits the 36 MB L2, so
    the two-kernel path pays DRAM twice for every strain value (write in
    kS, read in kT) - measured 128 ms/step at 5 MHz vs the ~5 G cells/s
    the same code sustains when the slab is L2-resident. This kernel
    computes the strains for a (TZ+2, TY+2, TX+2) halo tile into shared
    memory and consumes them immediately, so strains NEVER touch DRAM, and
    the g arrays disappear from the memory budget along with 20 of the 22
    kernel launches per step.

    Numerics: the strain expressions, move tap order, row accumulation
    order and deferred damping replicate the two-kernel path statement for
    statement; validate_labels.py --quick (bit-equivalence vs the dense
    solver) is the gate for ANY edit here.
    """
    POS = {0: (0, 0, 0), 1: (1, 1, 0), 2: (1, 0, 1), 3: (0, 1, 1)}
    VP = [0, 0, 0, 1, 2, 3]
    SN = ["s1", "s2", "s3", "s4", "s5", "s6"]
    NSH = (TZ + 2) * (TY + 2) * (TX + 2)
    AX = [("iz", "nz", "ssz"), ("iy", "ny", "ssy"), ("ix", "nx", "(long)1")]

    def moved_expr(J, spos, dpos):
        d = [POS[dpos][a] - POS[spos][a] for a in range(3)]
        conds, taps = [], [[]]
        for a in range(3):
            pos, n, stride = AX[a]
            if d[a] > 0:
                conds.append(f"{pos} <= {n}-2")
                offs = [0, 1]
            elif d[a] < 0:
                conds.append(f"{pos} >= 1")
                offs = [-1, 0]
            else:
                offs = [0]
            taps = [t + [(stride, o)] for t in taps for o in offs]
        terms = []
        for t in taps:
            off = " + ".join(f"({s})*({o})" for s, o in t if o != 0)
            terms.append(f"SH{J}[shidx{(' + ' + off) if off else ''}]")
        expr = " + ".join(terms)
        w = 0.25 if len(taps) == 4 else (0.5 if len(taps) == 2 else 1.0)
        return f"(({' && '.join(conds)}) ? ({expr}) * {w}f : 0.0f)"

    q = []
    for p in range(4):
        for J in range(6):
            if VP[J] == p:
                q.append(f"    float q{p}_{J} = SH{J}[shidx];")
            else:
                q.append(f"    float q{p}_{J} = "
                         f"{moved_expr(J, VP[J], p)};")
    rows = []
    for I in range(6):
        p = VP[I]
        rows.append("    { float acc = 0.0f; int any = 0;")
        for J in range(6):
            rows.append(f"      if (act[{I*6+J}]) "
                        f"{{ acc += Cm[{I*6+J}] * q{p}_{J}; any = 1; }}")
        rows.append(f"      float sv = {SN[I]}[idx] * dmp;")
        rows.append("      if (any) sv += dt * acc;")
        rows.append(f"      {SN[I]}[idx] = sv; }}")

    return (r'''
extern "C" __global__
void fused_step(const float* __restrict__ vx, const float* __restrict__ vy,
                const float* __restrict__ vz,
                float* __restrict__ s1, float* __restrict__ s2,
                float* __restrict__ s3, float* __restrict__ s4,
                float* __restrict__ s5, float* __restrict__ s6,
                const unsigned char* __restrict__ lab,
                const float* __restrict__ Ct,
                const int* __restrict__ act,
                const float* __restrict__ damp, const int use_damp,
                const float* __restrict__ c, const int nc,
                const float inv_h, const float dt,
                const int nz, const int ny, const int nx)
{
''' + f'''
    __shared__ float SH0[{NSH}]; __shared__ float SH1[{NSH}];
    __shared__ float SH2[{NSH}]; __shared__ float SH3[{NSH}];
    __shared__ float SH4[{NSH}]; __shared__ float SH5[{NSH}];
    const int TX = {TX}, TY = {TY}, TZ = {TZ};
''' + r'''
    const int bx0 = blockIdx.x * TX, by0 = blockIdx.y * TY,
              bz0 = blockIdx.z * TZ;
    const int tid = (threadIdx.z * TY + threadIdx.y) * TX + threadIdx.x;
    const int NT = TX * TY * TZ;
    const int NSH = (TZ + 2) * (TY + 2) * (TX + 2);
    const long sx = 1, sy = nx, sz = (long)nx * ny;

    for (int t = tid; t < NSH; t += NT) {
        int hx = bx0 + (t % (TX + 2)) - 1;
        int hy = by0 + ((t / (TX + 2)) % (TY + 2)) - 1;
        int hz = bz0 + (t / ((TX + 2) * (TY + 2))) - 1;
        float a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f;
        if (hx >= 0 && hx < nx && hy >= 0 && hy < ny
            && hz >= 0 && hz < nz) {
            long idx = ((long)hz * ny + hy) * nx + hx;
            for (int j = 0; j < nc; ++j) {
                float cj = c[j];
                if (hx >= j+1 && hx < nx-j)
                    a1 += cj * (vx[idx + (long)j*sx] - vx[idx - (long)(j+1)*sx]);
                if (hy >= j+1 && hy < ny-j)
                    a2 += cj * (vy[idx + (long)j*sy] - vy[idx - (long)(j+1)*sy]);
                if (hz >= j+1 && hz < nz-j)
                    a3 += cj * (vz[idx + (long)j*sz] - vz[idx - (long)(j+1)*sz]);
                if (hy >= j && hy < ny-j-1)
                    a4 += cj * (vz[idx + (long)(j+1)*sy] - vz[idx - (long)j*sy]);
                if (hz >= j && hz < nz-j-1)
                    a4 += cj * (vy[idx + (long)(j+1)*sz] - vy[idx - (long)j*sz]);
                if (hx >= j && hx < nx-j-1)
                    a5 += cj * (vz[idx + (long)(j+1)*sx] - vz[idx - (long)j*sx]);
                if (hz >= j && hz < nz-j-1)
                    a5 += cj * (vx[idx + (long)(j+1)*sz] - vx[idx - (long)j*sz]);
                if (hx >= j && hx < nx-j-1)
                    a6 += cj * (vy[idx + (long)(j+1)*sx] - vy[idx - (long)j*sx]);
                if (hy >= j && hy < ny-j-1)
                    a6 += cj * (vx[idx + (long)(j+1)*sy] - vx[idx - (long)j*sy]);
            }
        }
        SH0[t]=a1*inv_h; SH1[t]=a2*inv_h; SH2[t]=a3*inv_h;
        SH3[t]=a4*inv_h; SH4[t]=a5*inv_h; SH5[t]=a6*inv_h;
    }
    __syncthreads();

    const int ix = bx0 + threadIdx.x;
    const int iy = by0 + threadIdx.y;
    const int iz = bz0 + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;
    const long idx = ((long)iz * ny + iy) * nx + ix;
    const long shidx = ((long)(threadIdx.z + 1) * (TY + 2)
                        + threadIdx.y + 1) * (TX + 2) + threadIdx.x + 1;
    const long ssy = TX + 2, ssz = (long)(TX + 2) * (TY + 2);
    const float* Cm = Ct + (long)lab[idx] * 36;
    const float dmp = use_damp ? damp[idx] : 1.0f;

''' + "\n".join(q) + "\n" + "\n".join(rows) + r'''
}
''')


def absorption_mask(lab, alpha_np_m, c_ref, dt):
    """Per-step amplitude factor implementing intrinsic absorption in ICE.

    For a narrowband pulse, constant-Q absorption is well approximated by a
    frequency-independent per-step loss exp(-alpha * c * dt): the wave
    travels c*dt per step, so the accumulated loss over a path L is
    exp(-alpha * L) exactly (validated: E1 over 200 mm matches analytic to
    <2%). alpha_np_m is at the CENTRE frequency; band-edge error is second
    order for our 60% bandwidth. Set alpha from literature/measured ice
    attenuation at the operating temperature (tunable, temperature matters).
    Multiply into the existing damp_mask; fluid cells are left at 1 here
    (the fluid has its own, much stronger, stability damping).
    """
    lab = np.asarray(lab)
    f = float(np.exp(-alpha_np_m * c_ref * dt))
    m = np.ones(lab.shape, np.float32)
    m[lab >= 0] = f
    return m


def material_tables(axes, c_couplant=1500.0, rho_couplant=300.0):
    """Per-material stiffness (n_mat, 36) and density (n_mat,) tables.

    Material 0 is the fluid surround (specimen label -1); material m is
    grain m-1. Values come from the SAME polycrystal_stiffness_3d call the
    dense path uses, evaluated on an (n_mat, 1, 1) label grid, so every
    cell's stiffness is bit-identical to what forward_fused would store.
    """
    from ringfwi import anisotropy as an
    n_g = len(axes)
    ids = np.arange(-1, n_g, dtype=np.int32).reshape(-1, 1, 1)
    C, rho = an.polycrystal_stiffness_3d(ids, np.asarray(axes, float),
                                         c_couplant=c_couplant,
                                         rho_couplant=rho_couplant)
    n_mat = n_g + 1
    Ct = np.zeros((n_mat, 36), np.float32)
    for I in range(1, 7):
        for J in range(1, 7):
            a = np.asarray(C[f"C{min(I,J)}{max(I,J)}"], float)
            a = np.broadcast_to(a, (n_mat, 1, 1)).reshape(-1)
            Ct[:, (I - 1) * 6 + (J - 1)] = a
    rho_t = np.broadcast_to(np.asarray(rho, float),
                            (n_mat, 1, 1)).reshape(-1).astype(np.float32)
    return Ct, rho_t


def safe_dt_labels(mats, Ct, rho_t, h, coeffs, safety=0.85):
    """safe_dt without materialising per-cell stiffness volumes."""
    diag = Ct[:, [0, 7, 14]].max(axis=1)          # C11, C22, C33 per material
    Cmax = diag[mats]
    rho = rho_t[mats].astype(float)
    rmin = rho.copy()
    for ax in range(3):
        np.minimum(rmin, face_density(rho, ax), out=rmin)
    vmax = float(np.sqrt(Cmax / rmin).max())
    return safety * cfl_limit(np.asarray(coeffs, float)) * h / vmax


def forward_fused_labels(lab, axes, h, dt, nt, src_pts, wavelet, rec_groups,
                         order=8, coeffs=None, sponge_width=0,
                         sponge_strength=0.09, damp_mask=None,
                         c_couplant=1500.0, rho_couplant=300.0,
                         record="pressure", slab=32, stress_smem=False,
                         progress=None, mat_tables=None):
    """forward_fused semantics at ~41 B/cell: label-indexed stiffness and
    z-slab strain scratch. `lab` is the specimen label volume (-1 = fluid).

    `mat_tables` optionally supplies (Ct, rho_t) directly, bypassing the
    per-grain c-axis lookup, so labels can denote EFFECTIVE media rather
    than grains. Default None keeps the original behaviour bit-for-bit.
    """
    import cupy as cp
    if coeffs is None:
        coeffs = optimised_coeffs(order)
    coeffs = np.asarray(coeffs, float)
    # review finding: the zp kernels carry 9-deep z-rings and hard-cap the
    # tap loop at 4 coefficients; longer stencils would be silently
    # truncated into WRONG physics, so refuse them loudly.
    if len(coeffs) > 4:
        raise ValueError(
            f"forward_fused_labels supports order <= 8 ({len(coeffs)} "
            f"coefficients given): the z-pipeline kernels hold only "
            f"+/-4 z-planes in registers. Use forward_fused for higher "
            f"orders.")
    cd = cp.asarray(coeffs, cp.float32)

    lab = np.asarray(lab)
    # review finding: the uint8 material index wraps silently at >= 255
    # grains (grain 255 becomes FLUID), and labels beyond the axes table
    # read out of bounds on the GPU. Refuse loudly.
    lmin, lmax = int(lab.min()), int(lab.max())
    if lmin < -1 or lmax > 254:
        raise ValueError(
            f"labels must lie in [-1, 254]; got [{lmin}, {lmax}]: the "
            f"uint8 material index would wrap silently (>=255 grains "
            f"needs a wider label type).")
    if mat_tables is None and lmax >= len(axes):
        raise ValueError(
            f"label {lmax} has no c-axis (len(axes) = {len(axes)}).")
    nz, ny, nx = lab.shape
    mats = (lab + 1).astype(np.uint8)             # 0 = fluid
    if mat_tables is None:
        Ct, rho_t = material_tables(axes, c_couplant, rho_couplant)
    else:
        # EQUIVALENT-MEDIUM PATH (2026-08-01): the caller supplies the
        # (n_mat, 36) stiffness and (n_mat,) density tables directly, so a
        # material need not be a whole grain with a single c-axis. This is
        # what lets boundary cells carry an effective anisotropic tensor
        # (anti-aliased or Schoenberg-Muir) instead of being snapped to
        # whichever grain owns the cell centre. Row 0 is still the label
        # -1 material, matching the `mats = lab + 1` convention above.
        Ct, rho_t = mat_tables
        Ct = np.ascontiguousarray(Ct, np.float32).reshape(len(rho_t), 36)
        rho_t = np.ascontiguousarray(rho_t, np.float32)
        if lmax + 1 >= len(rho_t):
            raise ValueError(
                f"label {lmax} has no row in mat_tables "
                f"(n_mat = {len(rho_t)}, need >= {lmax + 2}).")
    act = np.ascontiguousarray(
        (np.abs(Ct) > 0).any(axis=0).astype(np.int32))

    lab_d = cp.asarray(mats)
    Ct_d = cp.asarray(Ct.reshape(-1))
    rho_d = cp.asarray(rho_t)
    act_d = cp.asarray(act)

    v = [cp.zeros((nz, ny, nx), cp.float32) for _ in range(3)]
    s = [cp.zeros((nz, ny, nx), cp.float32) for _ in range(6)]

    # combined damping built ON THE CPU and uploaded once: the device path
    # held three full-volume transients (sponge, mask, product), and on a
    # card that also drives the display that transient peak tipped a 194M
    # grid into WDDM paging - the silent 25x slowdown, not an OOM error.
    damp_h = None
    if sponge_width > 0:
        damp_h = sponge_profile((nz, ny, nx), sponge_width,
                                sponge_strength, np).astype(np.float32)
    if damp_mask is not None:
        dm = np.asarray(damp_mask, np.float32)
        damp_h = dm if damp_h is None else damp_h * dm
    shp = (nz, ny, nx)
    s_idx_h = np.array([np.ravel_multi_index(tuple(i), shp)
                        for i, _ in src_pts], np.int64)
    r_idx_h = [np.array([np.ravel_multi_index(tuple(q), shp) for q in pts],
                        np.int64) for pts, _ in rec_groups]
    # review finding: deferred damping records PRE-damp stress, which is
    # only identical to the dense path where damp == 1 exactly. That held
    # by construction for every run so far; now it is enforced.
    if damp_h is not None:
        # tolerance 0.999: intrinsic-absorption masks legitimately put
        # damp = 1 - O(1e-5) in ice (recording bias = ONE step's absorption,
        # physically nil); sponge/fluid damping (the real trap) is far below.
        dflat = damp_h.ravel()
        bad = [int(q) for q in s_idx_h if dflat[q] < 0.999]
        for ri in r_idx_h:
            bad += [int(q) for q in ri if dflat[q] < 0.999]
        if bad:
            raise ValueError(
                f"{len(bad)} source/receiver cells sit where damp < 0.999 "
                f"(e.g. flat index {bad[0]}): the deferred-damping scheme "
                f"records pre-damp stress there and would silently diverge "
                f"from the dense path. Move the footprint out of the "
                f"sponge/fluid-damped region.")
    damp = cp.asarray(damp_h) if damp_h is not None else None
    del damp_h
    damp_arg = damp if damp is not None else cp.zeros(1, cp.float32)
    use_damp = np.int32(0 if damp is None else 1)

    # NEGATIVE RESULT, kept for the record: the fused shared-memory kernel
    # (_fused_step_src) measured 184 ms/step at 194M cells vs 128 for this
    # two-kernel slab path - its halo-recompute overhead exceeds the cache
    # saving at every size measured. Both are bit-equivalence-gated, so
    # this is purely a speed choice.
    # NEGATIVE RESULT #3 (2026-07-29), completing the kernel-route
    # bracket: double-buffered g + two streams overlapping slab k+1's
    # strains with slab k's stress measured 114 vs 112 ms/step at 194M
    # (gate 0.00e+00 - correct but useless). Both kernels are DRAM-
    # bandwidth-bound; concurrency just time-slices the same bus. With
    # the fused-step and smem-stress results this proves the solver is
    # AT the memory wall on this GPU - remaining speed lives in protocol
    # (causal coda/E1 split) and hardware, not kernels.
    g = [cp.zeros((slab + 2, ny, nx), cp.float32) for _ in range(6)]
    kS = _get_kernel("strains_zp", _STRAIN_ZP_SRC, "strains_zp")
    if stress_smem:
        # shared-memory-tiled stress: at 194M cells the 6 g z-windows
        # (23 MB) thrash the L2; staging 3 rolling planes per 32x8 tile
        # in smem cuts g traffic to one DRAM read per cell. Identical
        # expression tree and tap order -> bit-identical (gated).
        kT = _get_kernel("stress_lab_smem", _stress_smem_src(),
                         "stress_lab_smem")
    else:
        kT = _get_kernel("stress_lab_u", _stress_unrolled_src(),
                         "stress_lab_u")
    # review-measured: 72 regs caps vel_zp at 50% occupancy; capping at 56
    # (no spills) + 128-wide column blocks = 1.21x on this kernel, and the
    # register allocator does not change the FP graph (bit-identical, gated)
    kV = _get_kernel("vel_zp", _VEL_ZP_SRC, "vel_zp",
                     options=("--maxrregcount=56",))
    tpb = 256
    N = (np.int32(nz), np.int32(ny), np.int32(nx))
    inv_h = np.float32(1.0 / h)
    dtf = np.float32(dt)
    dt_h = np.float32(dt / h)
    z_ranges = []
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        zg0, zg1 = max(z0 - 1, 0), min(z1 + 1, nz)
        nsl = (zg1 - zg0) * ny * nx
        ncr = (z1 - z0) * ny * nx
        z_ranges.append((z0, z1, zg0, zg1,
                         int((nsl + tpb - 1) // tpb),
                         int((ncr + tpb - 1) // tpb)))
    if stress_smem:                      # 2D tile grid, marches z per slab
        bT_g = (int((nx + 31) // 32), int((ny + 7) // 8))
        bT_b = (32, 8)

    s_idx = cp.asarray(s_idx_h)
    s_w = cp.asarray(np.array([w for _, w in src_pts], np.float32))
    r_idx = [cp.asarray(ri) for ri in r_idx_h]
    r_w = [cp.asarray(np.asarray(w, np.float32)) for _, w in rec_groups]
    rec_d = cp.zeros((nt, len(rec_groups)), cp.float32)

    sf = [a.ravel() for a in s]
    vf = [a.ravel() for a in v]
    tpbc = 128                    # column kernels: 128 measured >= 256
    gcol = (int((nx + tpbc - 1) // tpbc), ny)   # one thread per (x, y) column
    for n in range(nt):
        for z0, z1, zg0, zg1, bS, bT in z_ranges:
            kS(gcol, (tpbc,), (v[0], v[1], v[2], *g, cd,
                              np.int32(len(coeffs)), inv_h, *N,
                              np.int32(zg0), np.int32(zg1)))
            args_T = (*g, *s, lab_d, Ct_d, act_d, damp_arg,
                      use_damp, dtf, *N,
                      np.int32(z0), np.int32(z1), np.int32(zg0))
            if stress_smem:
                kT(bT_g, bT_b, args_T)
            else:
                kT((bT,), (tpb,), args_T)
        wn = np.float32(wavelet[n])
        for I in range(3):
            sf[I][s_idx] += wn * s_w
        kV(gcol, (tpbc,), (*s, v[0], v[1], v[2], lab_d, rho_d, damp_arg,
                          use_damp, cd, np.int32(len(coeffs)), dt_h, *N))
        # gather ONLY the receiver cells: building a full-volume pressure
        # array per step was 776 MB of allocation and 2.3 GB of traffic per
        # step at 5 MHz, and the transient peak is what triggered paging
        for r in range(len(r_idx)):
            if record == "pressure":
                val = -(sf[0][r_idx[r]] + sf[1][r_idx[r]]
                        + sf[2][r_idx[r]]) / 3.0
            else:
                val = {"vx": vf[0], "vy": vf[1],
                       "vz": vf[2]}[record][r_idx[r]]
            rec_d[n, r] = (val * r_w[r]).sum()
        if progress is not None and (n % 200 == 0):
            progress(n / nt)
    return cp.asnumpy(rec_d)


def harmonic_shear(C, shape):
    """Harmonic-average the shear stiffnesses onto their staggered nodes.

    Shear-stress components live at cell CORNERS: s4 (yz) is offset half a
    cell in z and y, s5 (xz) in z and x, s6 (xy) in y and x. Sampling C44/C55/
    C66 from the cell centre therefore gives a node straddling the rim the
    FULL ice shear modulus even when half its support is inviscid fluid, so
    the grid can carry shear stress inside a fluid. That is unphysical and it
    drives a slow interface instability: growth of a fraction of a percent per
    step, invisible over hundreds of steps and fatal over tens of thousands.

    Harmonic averaging over the four surrounding cells is the standard
    Moczo-Kristek treatment, and it does the right thing automatically at a
    solid/fluid boundary: the harmonic mean of anything with a zero is zero,
    so every node touching fluid gets exactly zero shear and free slip is
    enforced without any explicit interface bookkeeping.
    """
    # (Voigt shear index) -> the two axes it is staggered along, as (z,y,x)=0,1,2
    PLANES = {"C44": (0, 1), "C55": (0, 2), "C66": (1, 2)}
    out = dict(C)
    for key, axes in PLANES.items():
        a = np.asarray(C[key], float)
        if a.ndim != 3:
            a = np.full(shape, float(a))
        inv = np.zeros_like(a)
        nz_mask = a > 0
        inv[nz_mask] = 1.0 / a[nz_mask]
        acc = np.zeros_like(a)
        zero_any = np.zeros(a.shape, bool)
        for d0 in (0, 1):
            for d1 in (0, 1):
                sh = [slice(None)] * 3
                sh[axes[0]] = slice(d0, a.shape[axes[0]] - 1 + d0)
                sh[axes[1]] = slice(d1, a.shape[axes[1]] - 1 + d1)
                tgt = [slice(None)] * 3
                tgt[axes[0]] = slice(0, a.shape[axes[0]] - 1)
                tgt[axes[1]] = slice(0, a.shape[axes[1]] - 1)
                acc[tuple(tgt)] += inv[tuple(sh)]
                zero_any[tuple(tgt)] |= ~nz_mask[tuple(sh)]
        eff = np.zeros_like(a)
        good = (~zero_any) & (acc > 0)
        eff[good] = 4.0 / acc[good]
        out[key] = eff
    return out
