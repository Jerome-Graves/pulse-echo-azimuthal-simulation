"""Explicit anisotropic elastic FE on grain-conforming tets (CuPy).

The gold-standard cross-check partner for the FDTD: same per-grain
stiffness tensors (fdtd.material_tables, so the MATERIALS are identical by
construction), but the mesh CONFORMS to every grain boundary - zero
staircase. Deliberately minimal: linear (constant-strain) tets, lumped
mass, central differences, a damped shell standing in for the absorbing
boundary. Not a production solver; an arbiter.

Update per step (displacement form):
    strain_e = B_e u_e ;  sigma_e = C_g(e) strain_e
    f = -sum_e V_e B_e^T sigma_e   (scatter-add)
    v += dt f / m_lumped ;  u += dt v ;  damped shell on u, v
Stability: dt <= ~0.55 h_min / v_max (checked per mesh from edge lengths).
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")

import fdtd                                     # noqa: E402


def precompute(nodes, tets):
    """Per-tet volume and B (6 x 12) for constant-strain tets."""
    p = nodes[tets]                                # (m, 4, 3)
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    c = p[:, 3] - p[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    # shape-function gradients: rows of the inverse Jacobian
    J = np.stack([a, b, c], axis=2)                # (m, 3, 3)
    Jin = np.linalg.inv(J)                         # (m, 3, 3)
    g1, g2, g3 = Jin[:, 0], Jin[:, 1], Jin[:, 2]   # grads of N1..N3
    g0 = -(g1 + g2 + g3)
    G = np.stack([g0, g1, g2, g3], axis=1)         # (m, 4, 3)
    m = len(tets)
    B = np.zeros((m, 6, 12), np.float32)
    for k in range(4):
        gx, gy, gz = G[:, k, 0], G[:, k, 1], G[:, k, 2]
        B[:, 0, 3*k+0] = gx
        B[:, 1, 3*k+1] = gy
        B[:, 2, 3*k+2] = gz
        B[:, 3, 3*k+1] = gz; B[:, 3, 3*k+2] = gy   # yz
        B[:, 4, 3*k+0] = gz; B[:, 4, 3*k+2] = gx   # xz
        B[:, 5, 3*k+0] = gy; B[:, 5, 3*k+1] = gx   # xy
    return vol.astype(np.float32), B


def locate(nodes, tets, x):
    """(tet index, barycentric weights) of the tet containing point x.

    Nearest-node src/rec placement differs mesh-to-mesh by ~h/2 (~25 ns of
    path), which contaminated the remesh texture control; barycentric
    interpolation puts source and receiver at EXACT physical coordinates
    on any mesh.
    """
    x = np.asarray(x, float)
    d2 = np.sum((nodes - x) ** 2, axis=1)
    for nn in np.argsort(d2)[:30]:
        cand = np.where((tets == nn).any(axis=1))[0]
        for ti in cand:
            p = nodes[tets[ti]]
            M = np.column_stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]])
            try:
                lam123 = np.linalg.solve(M, x - p[0])
            except np.linalg.LinAlgError:
                continue
            lam = np.array([1.0 - lam123.sum(), *lam123])
            if (lam > -1e-9).all():
                return int(ti), np.clip(lam, 0, 1)
    raise ValueError(f"point {x} not inside any tet near its 30 NN")


def stable_dt(nodes, tets, vmax, safety=0.55):
    e = []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        e.append(np.linalg.norm(nodes[tets[:, i]] - nodes[tets[:, j]],
                                axis=1).min())
    return safety * min(e) / vmax


def forward_fe(nodes, tets, grain, axes, dt, nt, src_node, wavelet,
               rec_nodes, rho=917.0, shell_frac=0.12, shell_damp=0.04,
               src_xyz=None, rec_xyz=None, progress=None):
    """src_xyz/rec_xyz (physical coordinates) override the node-index
    src_node/rec_nodes with barycentric-exact placement."""
    import cupy as cp
    vol, B = precompute(nodes, tets)
    Ct, _ = fdtd.material_tables(axes)             # (n_mat, 36); mat 0=fluid
    C6 = Ct.reshape(-1, 6, 6)[grain + 1]           # per-tet 6x6 (ice grains)
    # engineering-strain convention: our B builds gamma (2*eps) for shears,
    # and material_tables stores the standard Voigt C, so C @ [eps,gamma]
    # is consistent with the FDTD's velocity-stress usage.

    d_tets = cp.asarray(tets)
    d_vol = cp.asarray(vol)
    d_B = cp.asarray(B)
    d_C = cp.asarray(C6.astype(np.float32))

    n = len(nodes)
    mlump = np.zeros(n, np.float32)
    np.add.at(mlump, tets.ravel(),
              np.repeat(rho * vol / 4.0, 4))
    d_m = cp.asarray(mlump)[:, None]

    # damped shell near the box boundary (crude absorber, matched windows
    # in the cross-check make its imperfection irrelevant)
    lo, hi = nodes.min(0), nodes.max(0)
    t = np.minimum(nodes - lo, hi - nodes).min(axis=1)
    w = (hi - lo).min() * shell_frac
    damp = np.exp(-shell_damp * np.clip(1.0 - t / w, 0, 1) ** 2)
    d_damp = cp.asarray(damp.astype(np.float32))[:, None]

    u = cp.zeros((n, 3), cp.float32)
    v = cp.zeros((n, 3), cp.float32)
    if src_xyz is not None:
        sti, slam = locate(nodes, tets, src_xyz)
        src_ids = tets[sti]
        src_w = slam
    else:
        src_ids, src_w = np.array([src_node]), np.array([1.0])
    d_src_ids = cp.asarray(src_ids)
    d_src_w = cp.asarray(src_w.astype(np.float32))
    if rec_xyz is not None:
        rec_pack = []
        for x in rec_xyz:
            ti, lam = locate(nodes, tets, x)
            rec_pack.append((cp.asarray(tets[ti]),
                             cp.asarray(lam.astype(np.float32))))
        nrec = len(rec_pack)
    else:
        rec_pack = [(cp.asarray(np.array([int(q)])),
                     cp.asarray(np.array([1.0], np.float32)))
                    for q in rec_nodes]
        nrec = len(rec_nodes)
    rec = np.zeros((nt, nrec))
    flat = d_tets.ravel()

    for it in range(nt):
        ue = u[d_tets].reshape(-1, 12, 1)                       # (m,12,1)
        eps = cp.matmul(d_B, ue)                                # (m,6,1)
        sig = cp.matmul(d_C, eps)                               # (m,6,1)
        fe = -cp.matmul(d_B.transpose(0, 2, 1), sig)[:, :, 0]   # (m,12)
        fe *= d_vol[:, None]
        f = cp.zeros((n, 3), cp.float32)
        cp.add.at(f.ravel(), (flat[:, None] * 3
                              + cp.arange(3)[None, :]).ravel(),
                  fe.reshape(-1, 4, 3).reshape(-1))
        f[d_src_ids, 0] += wavelet[it] * d_src_w
        v += dt * f / d_m
        v *= d_damp
        u += dt * v
        u *= d_damp
        for r, (ids, lam) in enumerate(rec_pack):
            rec[it, r] = float((u[ids, 0] * lam).sum())
        if progress and it % 200 == 0:
            progress(it / nt)
    return rec
