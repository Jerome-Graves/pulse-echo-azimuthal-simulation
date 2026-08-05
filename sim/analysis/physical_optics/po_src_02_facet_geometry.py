"""Step 2: EXACT grain-boundary facet polygons of the seed-11 girdle
specimen, from the Laguerre (power) tessellation.

CPU ONLY.  The GPU labeller in specimen.py is monkeypatched to an exact
all-seeds numpy argmin (identical arithmetic, no CUDA), so nothing here
touches the GPU while master_run.py is using it.

A power diagram is a convex polyhedral tessellation, so every grain
boundary is a convex POLYGON that can be constructed exactly by
half-space clipping - no voxel counting, no staircase correction.
The label volume is still built (coarsely, h = 1 mm) for one reason
only: `axes = sample_watson(n_real, ...)` is drawn AFTER empty cells are
dropped, so the c-axis assignment depends on how many cells survive
rasterisation.  We check that number is grid-independent.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim"))))
sys.path[:0] = [os.path.join(sys.path[0], _d)
                for _d in ('core', 'model', 'pipeline', 'fe_crosscheck', 'fw_checks')]
sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vendor"))))

import specimen as SP                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DIA, THK = 0.100, 0.035
R_DISC = DIA / 2


def _cpu_label_grid(x, z, R, seeds, weights):
    """Exact all-seeds power-diagram argmin, plane by plane, on the CPU.

    Same result as DiskSpecimen._label_grid_gpu, in numpy float64.
    """
    nx, nz = len(x), len(z)
    X, Y = np.meshgrid(x, x, indexing="ij")
    inside = (X ** 2 + Y ** 2) <= R ** 2
    Pxy = np.stack([X[inside], Y[inside]], axis=1)
    d_xy = (((Pxy[:, None, :] - seeds[None, :, :2]) ** 2).sum(-1)
            - weights[None, :])
    raw = np.full((nx, nx, nz), -1, np.int32)
    for iz in range(nz):
        d2 = d_xy + (z[iz] - seeds[:, 2])[None, :] ** 2
        plane = np.full((nx, nx), -1, np.int32)
        plane[inside] = d2.argmin(axis=1).astype(np.int32)
        raw[:, :, iz] = plane
    return raw


SP.DiskSpecimen._label_grid_gpu = staticmethod(_cpu_label_grid)


def build(h):
    sp = SP.DiskSpecimen(diameter_m=DIA, thickness_m=THK, n_grains=100,
                         size_cv=0.35, concentration=-8.0, spatial_corr=0.0,
                         fabric_axis=(1.0, 0.0, 0.0), seed=11)
    t0 = time.time()
    b = sp.build(h)
    b["_wall"] = time.time() - t0
    b["_sp"] = sp
    return b


# ─────────────────────────── polygon clipping ────────────────────────────
def clip(poly, a, b, c):
    """Keep the part of a 2-D convex polygon with a*u + b*v <= c."""
    if len(poly) == 0:
        return poly
    s = poly[:, 0] * a + poly[:, 1] * b - c
    keep = s <= 0
    if keep.all():
        return poly
    if not keep.any():
        return np.zeros((0, 2))
    out = []
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        if keep[i]:
            out.append(poly[i])
        if keep[i] != keep[j]:
            t = s[i] / (s[i] - s[j])
            out.append(poly[i] + t * (poly[j] - poly[i]))
    return np.array(out)


def poly_area_centroid(p):
    if len(p) < 3:
        return 0.0, np.zeros(2)
    x, y = p[:, 0], p[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = 0.5 * cr.sum()
    if abs(A) < 1e-18:
        return 0.0, p.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cr).sum() / (6 * A)
    cy = ((y + np.roll(y, -1)) * cr).sum() / (6 * A)
    return abs(A), np.array([cx, cy])


def facets(seeds, weights, n_rim=512, big=0.20):
    """Every grain-boundary polygon of the power diagram, clipped to the
    disc (rim approximated by n_rim tangent planes) and to |z| <= T/2."""
    ns = len(seeds)
    sq = (seeds ** 2).sum(1)
    th = np.arange(n_rim) * 2 * np.pi / n_rim
    rim_n = np.stack([np.cos(th), np.sin(th), np.zeros(n_rim)], axis=1)
    out = []
    for i in range(ns):
        for j in range(i + 1, ns):
            dv = seeds[j] - seeds[i]
            L = np.linalg.norm(dv)
            nrm = dv / L
            # plane 2 x.dv = |sj|^2-|si|^2 - wj + wi
            rhs = (sq[j] - sq[i] - weights[j] + weights[i]) / 2.0
            x0 = nrm * (rhs / L)                       # foot of the plane
            # in-plane basis
            t = np.array([0.0, 0.0, 1.0])
            if abs(nrm[2]) > 0.9:
                t = np.array([1.0, 0.0, 0.0])
            u = np.cross(nrm, t)
            u /= np.linalg.norm(u)
            v = np.cross(nrm, u)
            poly = np.array([[-big, -big], [big, -big], [big, big],
                             [-big, big]])
            # power half-spaces from every other seed k
            for k in range(ns):
                if k == i or k == j:
                    continue
                dk = seeds[k] - seeds[i]
                ck = (sq[k] - sq[i] - weights[k] + weights[i]) / 2.0
                # x.dk <= ck  with x = x0 + a u + b v
                a_, b_ = dk @ u, dk @ v
                c_ = ck - dk @ x0
                if abs(a_) < 1e-18 and abs(b_) < 1e-18:
                    if c_ < 0:
                        poly = np.zeros((0, 2))
                        break
                    continue
                poly = clip(poly, a_, b_, c_)
                if len(poly) < 3:
                    break
            if len(poly) < 3:
                continue
            # disc rim + flat faces
            for rn in rim_n:
                poly = clip(poly, rn @ u, rn @ v, R_DISC - rn @ x0)
                if len(poly) < 3:
                    break
            if len(poly) < 3:
                continue
            for sgn in (1.0, -1.0):
                zz = np.array([0.0, 0.0, sgn])
                poly = clip(poly, zz @ u, zz @ v, THK / 2 - zz @ x0)
                if len(poly) < 3:
                    break
            if len(poly) < 3:
                continue
            A, cxy = poly_area_centroid(poly)
            if A < 1e-10:
                continue
            out.append(dict(i=i, j=j, n=nrm, u=u, v=v, x0=x0, poly=poly,
                            area=A, cen=x0 + cxy[0] * u + cxy[1] * v))
    return out


if __name__ == "__main__":
    print("building seed-11 girdle specimen on the CPU (no CUDA)")
    b1 = build(1.0e-3)
    lab = b1["labels"]
    n1 = int((np.bincount(lab[lab >= 0].ravel()) > 0).sum())
    print(f"  h = 1.00 mm  grid {b1['shape']}  {b1['_wall']:.1f} s  "
          f"grains realised {len(b1['seeds'])} (nonempty {n1})")
    print(b1["_sp"].diagnose(b1))

    b2 = build(0.5e-3)
    n2 = len(b2["seeds"])
    print(f"\n  h = 0.50 mm  grains realised {n2}  "
          f"({b2['_wall']:.1f} s)  seeds identical: "
          f"{np.allclose(b1['seeds'], b2['seeds']) if n2 == len(b1['seeds']) else False}")
    print(f"  axes identical: "
          f"{np.allclose(b1['axes'], b2['axes']) if n2 == len(b1['seeds']) else False}")

    cnt = np.bincount(lab[lab >= 0].ravel(),
                      minlength=len(b1["seeds"])).astype(float)
    print(f"  smallest cell at h=1mm: {int(cnt.min())} voxels "
          f"(d_eq {2*(3*cnt.min()*1e-9/(4*np.pi))**(1/3)*1e3:.2f} mm)"
          f" -> no cell is anywhere near vanishing")

    t0 = time.time()
    F = facets(b1["seeds"], b1["weights"])
    A = np.array([f["area"] for f in F])
    print(f"\nEXACT polygon facets: {len(F)} boundaries, "
          f"total area {A.sum()*1e4:.1f} cm^2  ({time.time()-t0:.1f} s)")
    V = np.pi * R_DISC ** 2 * THK
    print(f"  S_v = A_tot/V = {A.sum()/V:.2f} 1/m")
    print(f"  facet area: mean {A.mean()*1e6:.1f} mm^2, median "
          f"{np.median(A)*1e6:.1f}, area-weighted mean "
          f"{(A*A).sum()/A.sum()*1e6:.1f} mm^2")
    print(f"  equivalent diameter sqrt(4A/pi): mean "
          f"{np.sqrt(4*A.mean()/np.pi)*1e3:.2f} mm, area-weighted "
          f"{np.sqrt(4*(A*A).sum()/A.sum()/np.pi)*1e3:.2f} mm")
    print(f"  faces per grain = 2*{len(F)}/{len(b1['seeds'])} = "
          f"{2*len(F)/len(b1['seeds']):.1f}")

    # cross-check against the voxel-counted table (staircase-corrected)
    import t3_facets as FT
    vt = FT.facet_table(dict(labels=lab, h=1.0e-3, seeds=b1["seeds"]))
    print(f"\n  voxel table (h=1mm, staircase corrected): {len(vt['area'])} "
          f"facets, total {vt['area'].sum()*1e4:.1f} cm^2  "
          f"(exact polygons: {len(F)}, {A.sum()*1e4:.1f} cm^2)")

    np.savez(os.path.join(HERE, "po_src_geom.npz"),
             seeds=b1["seeds"], weights=b1["weights"], axes=b1["axes"],
             fi=np.array([f["i"] for f in F]),
             fj=np.array([f["j"] for f in F]),
             fn=np.array([f["n"] for f in F]),
             fu=np.array([f["u"] for f in F]),
             fv=np.array([f["v"] for f in F]),
             fx0=np.array([f["x0"] for f in F]),
             farea=np.array([f["area"] for f in F]),
             fcen=np.array([f["cen"] for f in F]),
             npoly=np.array([len(f["poly"]) for f in F]),
             polys=np.concatenate([f["poly"] for f in F]))
    print("\nwrote po_src_geom.npz")
