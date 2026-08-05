"""Grain-CONFORMING tet mesh of a box polycrystal, without Neper.

Our specimens are Laguerre/Voronoi diagrams whose walls are exact planes,
so each grain is a convex polyhedron we can construct outright:
half-space intersection of the bisector planes against every other seed
plus the box walls (scipy.spatial.HalfspaceIntersection). Each polyhedron
goes to gmsh as points/faces/volume; coordinates are snapped to a fixed
grid so shared faces are bit-identical from both sides, and
removeAllDuplicates() welds them - the mesh then CONFORMS to every grain
boundary, which is the whole point: no staircase anywhere.

Output: nodes (n,3), tets (m,4), tet_grain (m,) - the inputs the explicit
FE solver needs, plus the .msh for inspection.
"""
import os
import sys

import numpy as np

sys.path.insert(0, (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor"))))
from scipy.spatial import HalfspaceIntersection  # noqa: E402

SNAP = 1e-6      # 1 um snap: also merges near-degenerate Voronoi
                 # vertices whose sliver facets break gmsh (measured:
                 # 'overlapping facets' at 1e-9 on the 30 mm box)


def voronoi_cells(seeds, box):
    """Convex cell polyhedra (vertices, faces) for each seed in a box.

    faces are lists of vertex indices, ordered around the outward normal.
    """
    (x0, x1), (y0, y1), (z0, z1) = box
    cells = []
    for i, si in enumerate(seeds):
        hs = []
        for j, sj in enumerate(seeds):
            if j == i:
                continue
            n = sj - si
            m = 0.5 * (si + sj)
            hs.append([*n, -np.dot(n, m)])         # n.x - n.m <= 0
        hs += [[-1, 0, 0, x0], [1, 0, 0, -x1],
               [0, -1, 0, y0], [0, 1, 0, -y1],
               [0, 0, -1, z0], [0, 0, 1, -z1]]
        hs = np.array(hs, float)
        hs /= np.linalg.norm(hs[:, :3], axis=1, keepdims=True)  # UNIT normals:
        # the on-face tolerance below is metric only if |n| = 1 (unnormalised
        # bisector normals carried |sj-si| ~ cm and broke vertex selection)
        hi = HalfspaceIntersection(hs, si.astype(float))
        v = np.round(hi.intersections / SNAP) * SNAP
        # faces: vertices lying on each active half-space, angularly sorted
        faces = []
        for k, h in enumerate(hs):
            n, d = h[:3], h[3]
            on = np.where(np.abs(v @ n + d) < 3 * SNAP)[0]
            if len(on) < 3:
                continue
            c = v[on].mean(axis=0)
            a = n / np.linalg.norm(n)
            t1 = np.cross(a, [1.0, 0.0, 0.0])
            if np.linalg.norm(t1) < 1e-6:
                t1 = np.cross(a, [0.0, 1.0, 0.0])
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(a, t1)
            ang = np.arctan2((v[on] - c) @ t2, (v[on] - c) @ t1)
            faces.append(on[np.argsort(ang)].tolist())
        # dedupe vertices within the cell
        uv, inv = np.unique(v, axis=0, return_inverse=True)
        faces = [[int(inv[ix]) for ix in f] for f in faces]
        faces = [[f[k] for k in range(len(f))
                  if f[k] != f[(k - 1) % len(f)]] for f in faces]
        # prune micro-faces (degenerate slivers break gmsh meshing)
        kept = []
        for f in faces:
            if len(f) < 3:
                continue
            pts = uv[f]
            ar = 0.5 * np.linalg.norm(
                sum(np.cross(pts[k] - pts[0], pts[(k + 1) % len(f)] - pts[0])
                    for k in range(1, len(f) - 1)))
            if ar > 1e-10:
                kept.append(f)
        cells.append((uv, kept))
    return cells


def mesh_polycrystal(seeds, box, h, out_npz, verbose=True, order=1):
    """Conforming tet mesh; returns nodes, tets, tet_grain."""
    import gmsh
    if gmsh.isInitialized():
        gmsh.finalize()          # clean slate after any earlier failure
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.6)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    # sliver control: floor2's worst tet forced dt=0.55 ns (26x cost);
    # the Netgen optimiser removes most such tets
    gmsh.option.setNumber("Mesh.Optimize", 1)
    try:
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    except Exception:
        pass
    cells = voronoi_cells(np.asarray(seeds, float), box)

    vol_tags = []
    for gi, (v, faces) in enumerate(cells):
        ptags = [gmsh.model.geo.addPoint(*p) for p in v]
        stags = []
        for f in faces:
            ltags = []
            for k in range(len(f)):
                a, b = ptags[f[k]], ptags[f[(k + 1) % len(f)]]
                ltags.append(gmsh.model.geo.addLine(a, b))
            cl = gmsh.model.geo.addCurveLoop(ltags)
            stags.append(gmsh.model.geo.addPlaneSurface([cl]))
        sl = gmsh.model.geo.addSurfaceLoop(stags)
        vol_tags.append((gmsh.model.geo.addVolume([sl]), gi))
    gmsh.model.geo.removeAllDuplicates()
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    if order == 2:
        gmsh.model.mesh.setOrder(2)        # TET10 for the quadratic solver

    nodes_t, nodes_xyz, _ = gmsh.model.mesh.getNodes()
    idmap = {int(t): k for k, t in enumerate(nodes_t)}
    nodes = np.asarray(nodes_xyz, float).reshape(-1, 3)

    tets, grain = [], []
    for dim, tag in gmsh.model.getEntities(3):
        gi = next(g for vt, g in vol_tags if vt == tag)
        etypes, etags, conn = gmsh.model.mesh.getElements(3, tag)
        want = 4 if order == 1 else 11     # 4-node vs 10-node tets
        npe = 4 if order == 1 else 10
        for et, cn in zip(etypes, conn):
            if et != want:
                continue
            cc = np.asarray(cn, np.int64).reshape(-1, npe)
            for row in cc:
                tets.append([idmap[int(q)] for q in row])
                grain.append(gi)
    gmsh.finalize()
    tets = np.asarray(tets, np.int64)
    grain = np.asarray(grain, np.int32)
    if verbose:
        print(f"mesh: {len(nodes)} nodes, {len(tets)} tets, "
              f"{grain.max()+1} grains")
    np.savez(out_npz, nodes=nodes, tets=tets, grain=grain)
    return nodes, tets, grain


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    L = 0.020                                  # 20 mm box
    n_g = 12
    seeds = rng.uniform(0.15 * L, 0.85 * L, (n_g, 3))
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8e-3
    mesh_polycrystal(seeds, ((0, L), (0, L), (0, L)), h,
                     "fe_mesh_test.npz")
