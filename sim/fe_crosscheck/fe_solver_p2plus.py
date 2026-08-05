"""ML2n15: the Geevers-Mulder-van der Vegt degree-2 MASS-LUMPED tet.

Why this element (the whole P2 saga in three lines): plain TET10 + HRZ
lumping degrades to P1-grade dispersion (-3.1% at lambda/8, measured),
and the consistent-mass Jacobi fix diverges (TET10 M_c not diagonally
dominant - documented in fe_solver_p2.py). ML2n15 enriches P2 with the
4 face bubbles + interior bubble so that a POSITIVE nodal quadrature
exists: mass is truly diagonal AND dispersion is 4th order,
e_disp = 1.89 * nE^-4 (Geevers, Mulder & van der Vegt 2018, SIAM J.
Sci. Comp. 40(5), arXiv:1803.10065, Tables 1 & 4) -> 0.046% at
lambda/8 vs 3.1% for HRZ-TET10.

Element (their Table 1, reference tet (0,0,0),(1,0,0),(0,1,0),(0,0,1)):
  4 vertices        weight 17/5040
  6 edge midpoints  weight  2/315
  4 face centroids  weight  9/560
  1 centroid        weight 16/315     (sum = 1/6 = ref volume, exact)
  U = P2 (+) {Li Lj Lk} face bubbles (+) L1L2L3L4 interior bubble.

Implementation choices that matter:
  * straight (affine) tets -> stiffness is EXACT analytic: the reference
    tensor Ghat[a,alpha,b,beta] = (1/V)int dN_a/dxi_alpha dN_b/dxi_beta
    is computed symbolically once (barycentric factorial formula), and
    per element only Jin (3x3) + V + the contracted D = V*Jin*C*Jin
    (27x3) are stored. Step = one shared GEMM + one batched GEMM.
    No quadrature loop, no Bg storage (the TET10 solver's 20 GB trap).
  * dt from POWER ITERATION on M^-1 K per mesh (no formula guesswork;
    sliver elements set the bound, not the mean shape).
  * damping shell is RATE-scaled: damp = exp(-shell_rate*dt*q^2), so
    the absorbed fraction per unit TIME is dt-invariant (the per-step
    shell in fe_solver/fe_solver_p2 changes physics with dt - the
    adversarial fleet's confirmed bug, kept there for history).
"""
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys
from math import factorial

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor"))))

import fdtd                                     # noqa: E402
from fe_solver_p1 import locate                    # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]   # gmsh pair order
FACES = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]       # opposite vertex f
# normalised node weights (6*omega, so they sum to 1): mass_a = rho*V*W15[a]
W15 = np.array([17 / 840] * 4 + [4 / 105] * 6 + [27 / 280] * 4 + [32 / 105])


# ---------------------------------------------------- exact reference algebra
def _pmul(p, q):
    r = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            e = tuple(a + b for a, b in zip(e1, e2))
            r[e] = r.get(e, 0.0) + c1 * c2
    return r


def _pdL(p, i):
    r = {}
    for e, c in p.items():
        if e[i]:
            e2 = list(e); e2[i] -= 1
            r[tuple(e2)] = r.get(tuple(e2), 0.0) + c * e[i]
    return r


def _pint(p):
    """(1/V) * integral over the reference tet of a barycentric poly."""
    s = 0.0
    for e, c in p.items():
        num = 1
        for q in e:
            num *= factorial(q)
        s += c * 6.0 * num / factorial(sum(e) + 3)
    return s


def _peval(p, L):
    return sum(c * np.prod([L[i] ** e[i] for i in range(4)])
               for e, c in p.items())


def _reference():
    """Nodal basis (poly dicts), node barycentrics, and the exact
    stiffness tensor Ghat[a,alpha,b,beta] (15,3,15,3)."""
    # node barycentric coordinates, element-local order:
    # 4 corners, 6 edges (gmsh pairs), 4 faces (opposite-vertex), centre
    bary = []
    eye = np.eye(4)
    for a in range(4):
        bary.append(eye[a])
    for a, b in EDGES:
        bary.append(0.5 * (eye[a] + eye[b]))
    for f in FACES:
        bary.append(sum(eye[i] for i in f) / 3.0)
    bary.append(np.full(4, 0.25))
    bary = np.array(bary)

    # monomial basis of U = P2 (+) Bf (+) Be, as exponent dicts
    mono = []
    for i in range(4):
        e = [0] * 4; e[i] = 2
        mono.append({tuple(e): 1.0})
    for i in range(4):
        for j in range(i + 1, 4):
            e = [0] * 4; e[i] = 1; e[j] = 1
            mono.append({tuple(e): 1.0})
    for f in FACES:
        e = [0] * 4
        for i in f:
            e[i] = 1
        mono.append({tuple(e): 1.0})
    mono.append({(1, 1, 1, 1): 1.0})

    V = np.array([[_peval(m, b) for m in mono] for b in bary])
    Vi = np.linalg.inv(V)
    condV = np.linalg.cond(V)
    basis = []
    for a in range(15):
        p = {}
        for b in range(15):
            for e, c in mono[b].items():
                p[e] = p.get(e, 0.0) + Vi[b, a] * c
        basis.append({e: c for e, c in p.items() if abs(c) > 1e-13})

    # gradients wrt reference coords xi_alpha: d/dxi_a = d/dL_a - d/dL_0
    grads = []
    for p in basis:
        g0 = _pdL(p, 0)
        ga = []
        for al in (1, 2, 3):
            g = dict(_pdL(p, al))
            for e, c in g0.items():
                g[e] = g.get(e, 0.0) - c
            ga.append(g)
        grads.append(ga)

    G = np.zeros((15, 3, 15, 3))
    for a in range(15):
        for al in range(3):
            for b in range(a, 15):
                for be in range(3):
                    v = _pint(_pmul(grads[a][al], grads[b][be]))
                    G[a, al, b, be] = v
                    G[b, be, a, al] = v
    return bary, basis, G, condV


_BARY, _BASIS, _GHAT, _CONDV = _reference()
# W[a,alpha,beta,k] = sum_b Ghat[a,alpha,b,beta] u[b,k]  as one GEMM.
# _G135_64 stays exact for verification paths (f32 truncation of the
# reference integrals, amplified by sliver Jacobians, showed up as a
# fake 4e-5 patch-test failure); the GPU gets the f32 copy.
_G135_64 = np.ascontiguousarray(
    _GHAT.transpose(0, 1, 3, 2).reshape(135, 15))
_G135 = _G135_64.astype(np.float32)


def shape_at(lam):
    """The 15 nodal shape values at barycentric point lam (for src/rec)."""
    return np.array([_peval(p, lam) for p in _BASIS])


def shape_grads_at(lam, Jin_e):
    """Physical gradients dN_a/dx_i (15,3) at barycentric lam inside an
    element with inverse Jacobian Jin_e. Used for the monopole source
    (f_a = M grad N_a: centre of dilatation, radiates P ONLY - a point
    FORCE also radiates S, which the FDTD's stress-monopole source does
    not, and that S content was the -19 dB A-B 'floor') and for the
    dilatation receiver (div u = sum_a u_a . grad N_a)."""
    g = np.zeros((15, 4))
    for a, p in enumerate(_BASIS):
        for i in range(4):
            g[a, i] = _peval(_pdL(p, i), lam)
    gxi = g[:, 1:] - g[:, :1]                    # d/dxi_a = d/dL_a - d/dL_0
    return gxi @ Jin_e


def _voigt_to_tensor(C6):
    vi = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    C = np.zeros((len(C6), 3, 3, 3, 3), np.float64)
    for I, (i, j) in enumerate(vi):
        for J, (k, l) in enumerate(vi):
            v = C6[:, I, J]
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    C[:, a, b, c, d] = v
    return C


# ------------------------------------------------------------- mesh building
def build_nodes(nodes10, tets10):
    """Extend a (straightened) TET10 mesh with shared face-centroid nodes
    and per-element centroids -> nodes15, tets15 (m,15)."""
    n0 = len(nodes10)
    m = len(tets10)
    tets15 = np.zeros((m, 15), np.int64)
    tets15[:, :10] = tets10
    fid = {}
    fpos = []
    nid = n0
    for fi, f in enumerate(FACES):
        for t in range(m):
            key = tuple(sorted(tets10[t, list(f)]))
            if key not in fid:
                fid[key] = nid
                fpos.append(nodes10[list(key)].mean(axis=0))
                nid += 1
            tets15[t, 10 + fi] = fid[key]
    cpos = nodes10[tets10[:, :4]].mean(axis=1)
    tets15[:, 14] = nid + np.arange(m)
    nodes15 = np.vstack([nodes10, np.asarray(fpos), cpos])
    return nodes15, tets15


def precompute(nodes15, tets15, grain, axes, rho=917.0):
    """Per-element V, Jin and the contracted stiffness D (m,27,3):
    D[(alpha,beta,k), i] = V * Jin[alpha,j] C_ijkl Jin[beta,l]."""
    p = nodes15[tets15[:, :4]]
    a = p[:, 1] - p[:, 0]; b = p[:, 2] - p[:, 0]; c = p[:, 3] - p[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    J = np.stack([a, b, c], axis=2)
    Jin = np.linalg.inv(J)
    Ct, _ = fdtd.material_tables(axes)
    C4 = _voigt_to_tensor(Ct.reshape(-1, 6, 6)[grain + 1])
    D = np.einsum("m,maj,mijkl,mbl->mabki", vol, Jin, C4, Jin,
                  optimize=True)
    return (vol.astype(np.float32),
            np.ascontiguousarray(D.reshape(-1, 27, 3)).astype(np.float32))


def lumped_mass(nodes15, tets15, vol, rho=917.0):
    md = np.zeros(len(nodes15), np.float64)
    np.add.at(md, tets15.ravel(),
              (rho * vol[:, None] * W15[None, :]).ravel())
    return md.astype(np.float32)


# ---------------------------------------------------------------- the solver
def _apply_K_np(tets15, D, u):
    """f = -K u in float64 numpy (patch test / verification path)."""
    ue = u[tets15]                                        # (m,15,3)
    W = np.matmul(_G135_64, ue)                           # (m,135,3)
    W = W.reshape(-1, 15, 27)
    fe = -np.matmul(W, D.astype(np.float64))              # (m,15,3)
    f = np.zeros_like(u)
    np.add.at(f, tets15.ravel(), fe.reshape(-1, 3))
    return f


def stable_dt(tets15, D, md, iters=60, safety=0.85, seed=0):
    """dt = safety * 2/omega_max, omega_max^2 from power iteration on
    M^-1 K (exact per-mesh bound - slivers included, no formula)."""
    import cupy as cp
    n = len(md)
    d_t = cp.asarray(tets15)
    d_D = cp.asarray(D)
    d_G = cp.asarray(_G135)
    d_mi = cp.asarray(1.0 / md)[:, None]
    d_scat = (d_t.ravel()[:, None] * 3 + cp.arange(3)[None, :]).ravel()
    rng = np.random.default_rng(seed)
    x = cp.asarray(rng.standard_normal((n, 3)).astype(np.float32))
    lam = 0.0
    for _ in range(iters):
        ue = x[d_t]
        W = cp.matmul(d_G[None], ue).reshape(-1, 15, 27)
        fe = cp.matmul(W, d_D)
        y = cp.zeros((n, 3), cp.float32)
        cp.add.at(y.ravel(), d_scat, fe.reshape(-1))
        y *= d_mi
        lam = float(cp.linalg.norm(y.ravel()) / cp.linalg.norm(x.ravel()))
        x = y / cp.linalg.norm(y.ravel())
    wmax = np.sqrt(lam * 1.05)                  # 5% margin, iterate underest.
    return safety * 2.0 / wmax


def forward(nodes15, tets15, grain, axes, dt, nt, wavelet, src_xyz,
            rec_xyz, rho=917.0, shell_frac=0.18, shell_rate=1.4e8,
            shell_pow=2, src_type="force_x", rec_type="ux",
            src_radius=1.0e-3, D=None, vol=None, progress=None):
    """shell_rate is per SECOND (damp = exp(-rate*dt*q^pow)): absorption
    per unit time is dt-invariant. MEASURED (fe_arbiter_round1_baseline forensics):
    rate=1.4e8 with pow=2 is an overdamped BRICK WALL - it reflects
    ~-16 dB at the shell entrance and the box rings at -20 dB re direct
    for the whole record. An absorber must be weak and wide: for a
    ~2-wavelength shell use pow=4 with rate*w/(5*v) ~ 4-5 (one-way
    -40 dB, two-way dead, soft adiabatic entry)."""
    import cupy as cp
    if D is None:
        vol, D = precompute(nodes15, tets15, grain, axes, rho)
    md = lumped_mass(nodes15, tets15, vol, rho)
    n = len(nodes15)

    d_t = cp.asarray(tets15)
    d_D = cp.asarray(D)
    d_G = cp.asarray(_G135)
    d_mi = cp.asarray(1.0 / md)[:, None]
    d_scat = (cp.asarray(tets15).ravel()[:, None] * 3
              + cp.arange(3)[None, :]).ravel()

    lo, hi = nodes15.min(0), nodes15.max(0)
    t = np.minimum(nodes15 - lo, hi - nodes15).min(axis=1)
    w = (hi - lo).min() * shell_frac
    q = np.clip(1.0 - t / w, 0, 1)
    damp = np.exp(-shell_rate * dt * q ** shell_pow)
    d_damp = cp.asarray(damp.astype(np.float32))[:, None]

    def _jin_of(ti):
        pc = nodes15[tets15[ti, :4]]
        J = np.column_stack([pc[1] - pc[0], pc[2] - pc[0], pc[3] - pc[0]])
        return np.linalg.inv(J)

    def _locate(x):
        # locate() seeds from the 30 nearest NODES, but ~87% of the
        # 15-node mesh's nodes never appear in the corner connectivity
        # (edge/face/interior) - searching them finds no candidate tet.
        # Restrict to corner nodes; tet ordering is preserved.
        corners = np.unique(tets15[:, :4])
        inv = np.zeros(len(nodes15), np.int64)
        inv[corners] = np.arange(len(corners))
        return locate(nodes15[corners], inv[tets15[:, :4]], x)

    # source/receiver weights as (15,3) matrices so force_x/monopole and
    # ux/div share one code path (weights fixed, physics in the values)
    if src_type == "monopole_ball":
        # Gaussian cloud of point moments (one per element centre inside
        # 3 sigma). A SINGLE discrete monopole leaks numerical shear at
        # ~-40 dB (measured: the 5.9 us S-arrival hump in the round-4 B
        # trace), and in the A runs that shear scatters off grains into
        # P coda the FDTD's pure-P source cannot have. ~3000 elements
        # average the per-element gradient error down ~30 dB while the
        # far field stays a monopole (ball transit ~0.25 us << ricker).
        sig = src_radius
        cent = nodes15[tets15[:, :4]].mean(axis=1)
        r2 = ((cent - np.asarray(src_xyz)) ** 2).sum(axis=1)
        sel = np.where(r2 < (3.0 * sig) ** 2)[0]
        wgt = np.exp(-r2[sel] / (2.0 * sig ** 2))
        wgt /= wgt.sum()
        acc = {}
        lam_c = np.full(4, 0.25)
        for e, we in zip(sel, wgt):
            g = shape_grads_at(lam_c, _jin_of(int(e))) * we
            for a in range(15):
                nid = int(tets15[e, a])
                acc[nid] = acc.get(nid, 0.0) + g[a]
        ids = np.array(sorted(acc.keys()), np.int64)
        sw = np.stack([acc[i] for i in ids])
        print(f"monopole_ball: {len(sel)} elements, "
              f"{len(ids)} source nodes", flush=True)
        d_sids = cp.asarray(ids)
        d_sw = cp.asarray(sw.astype(np.float32))
    else:
        sti, slam = _locate(np.asarray(src_xyz))
        if src_type == "monopole":
            sw = shape_grads_at(slam, _jin_of(sti))
        else:                                    # x point force (radiates S!)
            sw = np.zeros((15, 3)); sw[:, 0] = shape_at(slam)
        d_sids = cp.asarray(tets15[sti])
        d_sw = cp.asarray(sw.astype(np.float32))
    rec_pack = []
    for x in rec_xyz:
        ti, lam = _locate(np.asarray(x))
        if rec_type == "div":
            rw = shape_grads_at(lam, _jin_of(ti))
        else:
            rw = np.zeros((15, 3)); rw[:, 0] = shape_at(lam)
        rec_pack.append((cp.asarray(tets15[ti]),
                         cp.asarray(rw.astype(np.float32))))

    u = cp.zeros((n, 3), cp.float32)
    v = cp.zeros((n, 3), cp.float32)
    m = len(tets15)
    d_wav = cp.asarray(np.asarray(wavelet, np.float32))
    # rec samples accumulate ON DEVICE; float() readback per step forces
    # a full sync/stall (the FDTD hit this same trap: 5x). Copy once.
    rec_dev = cp.zeros((nt, len(rec_pack)), cp.float32)
    for it in range(nt):
        ue = u[d_t]                              # (m,15,3)
        # one large GEMM instead of 2.5M tiny batched ones (cublas runs
        # tiny (135x15)@(15x3) batches at a few % of peak)
        ut = cp.ascontiguousarray(ue.transpose(1, 0, 2)).reshape(15, -1)
        W = (d_G @ ut).reshape(135, m, 3)
        W = cp.ascontiguousarray(W.transpose(1, 0, 2)).reshape(m, 15, 27)
        fe = -cp.matmul(W, d_D)
        f = cp.zeros((n, 3), cp.float32)
        cp.add.at(f.ravel(), d_scat, fe.reshape(-1, 3).ravel())
        f[d_sids] += d_wav[it] * d_sw
        v += dt * f * d_mi
        v *= d_damp
        u += dt * v
        u *= d_damp
        for r, (ids, Nr) in enumerate(rec_pack):
            rec_dev[it, r] = (u[ids] * Nr).sum()
        if progress and it % 500 == 0:
            progress(it / nt)
    return cp.asnumpy(rec_dev).astype(float)
