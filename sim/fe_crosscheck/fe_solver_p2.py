"""Quadratic (TET10) explicit FE - the arbiter upgrade.

Linear CST elements have a measured mesh-texture noise floor of -26.8 dB
(re direct) on our polycrystal boxes - 16 dB ABOVE the grain-scattering
signal they were meant to referee. Quadratic elements typically buy
20-30 dB. Implementation notes that matter:
  * straight-sided tets -> constant Jacobian -> the 4 Gauss-point B
    matrices (6 x 30) are exact and PRECOMPUTABLE per element;
  * mass lumping is HRZ (diagonal-of-consistent scaled to element mass):
    row-sum lumping gives NEGATIVE corner masses on TET10;
  * 4-point Gauss rule (degree-2 exact) matches the strain field order.
Same A-B/floor harness as the linear solver decides if it is clean enough.
"""
import os
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
import sys

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor"))))

import fdtd                                     # noqa: E402
from fe_solver import locate                    # noqa: E402  (P1 point-locate
#                                    on corner sub-tets is fine for placement)

# 4-point Gauss rule on the reference tet (barycentric), degree 2
_GA = 0.5854101966249685
_GB = 0.13819660112501053
_GPTS = np.array([[_GA, _GB, _GB, _GB], [_GB, _GA, _GB, _GB],
                  [_GB, _GB, _GA, _GB], [_GB, _GB, _GB, _GA]])
_GW = np.full(4, 0.25)


def _shape_grads_bary(L):
    """dN/dL (10 x 4) for TET10 at barycentric point L."""
    g = np.zeros((10, 4))
    for a in range(4):
        g[a, a] = 4 * L[a] - 1                     # corner nodes
    edges = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]  # GMSH tet10 order  # gmsh order
    for e, (a, b) in enumerate(edges):
        g[4 + e, a] = 4 * L[b]
        g[4 + e, b] = 4 * L[a]
    return g


def _tet10_consistent_mass():
    """EXACT reference consistent mass: M[a,b] = (1/V) * int Na Nb dV.

    Shape functions are quadratics in barycentric coords, so every product
    is a degree-4 monomial sum and the classic formula
        int L0^i L1^j L2^k L3^l dV = 6V * i!j!k!l! / (i+j+k+l+3)!
    integrates it exactly - no quadrature, no Monte Carlo noise.
    Partition of unity => M.sum() == 1, asserted."""
    from math import factorial

    def corner(a):
        e2 = [0] * 4; e2[a] = 2
        e1 = [0] * 4; e1[a] = 1
        return {tuple(e2): 2.0, tuple(e1): -1.0}   # La(2La-1)

    def edge(a, b):
        e = [0] * 4; e[a] += 1; e[b] += 1
        return {tuple(e): 4.0}                     # 4 La Lb

    edges = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]  # GMSH tet10
    Ns = [corner(a) for a in range(4)] + [edge(a, b) for a, b in edges]

    def integ(d1, d2):
        s = 0.0
        for e1, c1 in d1.items():
            for e2, c2 in d2.items():
                e = [e1[i] + e2[i] for i in range(4)]
                num = 1
                for q in e:
                    num *= factorial(q)
                s += c1 * c2 * 6.0 * num / factorial(sum(e) + 3)
        return s

    M = np.array([[integ(Ns[i], Ns[j]) for j in range(10)]
                  for i in range(10)])
    assert abs(M.sum() - 1.0) < 1e-14              # partition of unity
    return M


def _ref_mass_diag():
    """Diagonal of the exact reference consistent mass (for HRZ lumping)."""
    return np.diag(_tet10_consistent_mass())


def straighten(nodes, tets10):
    """Snap edge nodes to exact corner midpoints (in place, returns nodes).

    gmsh's high-order placement leaves edge nodes up to ~0.3 um off the
    midpoint (measured: 8497 of 278k), which violates the straight-element
    constant-Jacobian assumption and put a 6e-4 systematic error in the
    patch test. All elements sharing an edge node agree on its corner
    pair, so the snap is globally consistent."""
    edges = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]
    nodes = nodes.copy()
    for e, (a, b) in enumerate(edges):
        nodes[tets10[:, 4 + e]] = 0.5 * (nodes[tets10[:, a]]
                                         + nodes[tets10[:, b]])
    return nodes


def stable_dt_p2(nodes, tets10, vmax, safety=0.5):
    """P2 explicit bound from element HEIGHTS (edge-based bounds miss
    flat elements), halved for the quadratic node spacing."""
    p = nodes[tets10[:, :4]]
    a = p[:, 1] - p[:, 0]; b = p[:, 2] - p[:, 0]; c = p[:, 3] - p[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0

    def fa(q0, q1, q2):
        return 0.5 * np.linalg.norm(np.cross(q1 - q0, q2 - q0), axis=1)
    A = np.max(np.stack([fa(p[:, 0], p[:, 1], p[:, 2]),
                         fa(p[:, 0], p[:, 1], p[:, 3]),
                         fa(p[:, 0], p[:, 2], p[:, 3]),
                         fa(p[:, 1], p[:, 2], p[:, 3])]), axis=0)
    hmin = (3.0 * vol / A).min()
    return safety * 0.5 * hmin / vmax


def precompute_p2(nodes, tets10):
    """Per-element volume and Gauss-point B matrices (4, 6, 30)."""
    p = nodes[tets10[:, :4]]                       # corners define J (straight)
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    c = p[:, 3] - p[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    J = np.stack([a, b, c], axis=2)
    Jin = np.linalg.inv(J)                         # (m, 3, 3)
    m = len(tets10)
    Bg = np.zeros((m, 4, 6, 30), np.float32)
    # dL/dx: L0 = 1 - xi - eta - zeta in natural coords; rows of [dL/dx]:
    dLdx = np.zeros((m, 4, 3))
    dLdx[:, 1:, :] = Jin
    dLdx[:, 0, :] = -Jin.sum(axis=1)
    for gp in range(4):
        g_b = _shape_grads_bary(_GPTS[gp])         # (10, 4)
        gN = np.einsum("na,mak->mnk", g_b, dLdx)   # (m, 10, 3)
        for k in range(10):
            gx, gy, gz = gN[:, k, 0], gN[:, k, 1], gN[:, k, 2]
            Bg[:, gp, 0, 3*k+0] = gx
            Bg[:, gp, 1, 3*k+1] = gy
            Bg[:, gp, 2, 3*k+2] = gz
            Bg[:, gp, 3, 3*k+1] = gz; Bg[:, gp, 3, 3*k+2] = gy
            Bg[:, gp, 4, 3*k+0] = gz; Bg[:, gp, 4, 3*k+2] = gx
            Bg[:, gp, 5, 3*k+0] = gy; Bg[:, gp, 5, 3*k+1] = gx
    return vol.astype(np.float32), Bg


def forward_fe_p2(nodes, tets10, grain, axes, dt, nt, wavelet,
                  src_xyz, rec_xyz, rho=917.0, shell_frac=0.18,
                  shell_damp=0.35, mass_iter=0, progress=None):
    """mass_iter=0: HRZ lumped mass. Measured on the structured-bar probe
    (two-receiver xcorr, exact analytic reference): -3.11% at lambda/8,
    dt-INdependent (-3.13% at dt/2) - i.e. HRZ lumping degrades TET10 to
    the same 2nd-order dispersion as P1 (-2.48% same mesh). This is the
    known price of lumping quadratic elements.

    mass_iter=k>0 (KEPT AS A DOCUMENTED NEGATIVE RESULT - DO NOT USE):
    k Jacobi iterations towards the consistent-mass solve M_c a = f,
    preconditioned by the HRZ diagonal. DIVERGES: k=1 +93% spurious fast
    precursor, k=2 +24%, k=3 explodes at BOTH dt and dt/2 (so not CFL).
    Root cause is structural, not a bug: the TET10 consistent mass is
    not diagonally dominant (corner diagonal rho*V/70 vs corner row sum
    MINUS rho*V/20), so diag-preconditioned Jacobi has spectral radius
    > 1. The literature fix is the Mulder / Chin-Joe-Kong 15-node P2+
    element (face+interior bubbles admit true positive mass lumping)."""
    import cupy as cp
    vol, Bg = precompute_p2(nodes, tets10)
    Ct, _ = fdtd.material_tables(axes)
    C6 = Ct.reshape(-1, 6, 6)[grain + 1].astype(np.float32)

    d_tets = cp.asarray(tets10)
    d_B = cp.asarray(Bg.reshape(len(tets10), 4 * 6, 30))   # (m, 24, 30)
    d_C = cp.asarray(C6)

    n = len(nodes)
    ref = _ref_mass_diag()
    mel = rho * vol                                 # element masses
    md = np.zeros(n, np.float32)
    w10 = ref / ref.sum()
    np.add.at(md, tets10.ravel(),
              (mel[:, None] * w10[None, :]).ravel())
    d_m = cp.asarray(md)[:, None]
    # one scatter index for every element->node add (force and mass)
    d_scat = (cp.asarray(tets10).ravel()[:, None] * 3
              + cp.arange(3)[None, :]).ravel()
    if mass_iter:
        d_Mref = cp.asarray(_tet10_consistent_mass().astype(np.float32))
        d_mel = cp.asarray(mel.astype(np.float32))

    lo, hi = nodes.min(0), nodes.max(0)
    t = np.minimum(nodes - lo, hi - nodes).min(axis=1)
    w = (hi - lo).min() * shell_frac
    damp = np.exp(-shell_damp * np.clip(1.0 - t / w, 0, 1) ** 2)
    d_damp = cp.asarray(damp.astype(np.float32))[:, None]

    # corner-subtet locate: search P1 connectivity of the corner nodes
    sti, slam = locate(nodes, tets10[:, :4], np.asarray(src_xyz))
    # consistent nodal loads: evaluate TET10 shape functions at the point
    Lsrc = np.array([slam[0], slam[1], slam[2], slam[3]])
    Ns = np.zeros(10)
    for a in range(4):
        Ns[a] = Lsrc[a] * (2 * Lsrc[a] - 1)
    edges = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]  # GMSH tet10 order
    for e, (a, b) in enumerate(edges):
        Ns[4 + e] = 4 * Lsrc[a] * Lsrc[b]
    d_sids = cp.asarray(tets10[sti])
    d_sw = cp.asarray(Ns.astype(np.float32))

    rec_pack = []
    for x in rec_xyz:
        ti, lam = locate(nodes, tets10[:, :4], np.asarray(x))
        Nr = np.zeros(10)
        for a in range(4):
            Nr[a] = lam[a] * (2 * lam[a] - 1)
        for e, (a, b) in enumerate(edges):
            Nr[4 + e] = 4 * lam[a] * lam[b]
        rec_pack.append((cp.asarray(tets10[ti]),
                         cp.asarray(Nr.astype(np.float32))))

    u = cp.zeros((n, 3), cp.float32)
    v = cp.zeros((n, 3), cp.float32)
    rec = np.zeros((nt, len(rec_pack)))
    gw_vol = cp.asarray((vol[:, None] * _GW[None, :]).astype(np.float32))

    for it in range(nt):
        ue = u[d_tets].reshape(-1, 30, 1)
        eg = cp.matmul(d_B, ue).reshape(-1, 4, 6)            # strains @ gp
        sg = cp.einsum("mij,mgj->mgi", d_C, eg)              # stresses @ gp
        sg *= gw_vol[:, :, None]
        fe = -cp.matmul(d_B.transpose(0, 2, 1),
                        sg.reshape(-1, 24, 1))[:, :, 0]      # (m, 30)
        f = cp.zeros((n, 3), cp.float32)
        cp.add.at(f.ravel(), d_scat, fe.reshape(-1, 10, 3).reshape(-1))
        f[d_sids, 0] += wavelet[it] * d_sw
        acc = f / d_m
        for _ in range(mass_iter):                  # Jacobi: M_c acc = f
            Ma = cp.einsum("ab,mbc->mac", d_Mref, acc[d_tets])
            Ma *= d_mel[:, None, None]
            Mca = cp.zeros((n, 3), cp.float32)
            cp.add.at(Mca.ravel(), d_scat, Ma.reshape(-1))
            acc += (f - Mca) / d_m
        v += dt * acc
        v *= d_damp
        u += dt * v
        u *= d_damp
        for r, (ids, Nr) in enumerate(rec_pack):
            rec[it, r] = float((u[ids, 0] * Nr).sum())
        if progress and it % 200 == 0:
            progress(it / nt)
    return rec
