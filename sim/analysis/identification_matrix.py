"""TEST 4 step 3b: IDENTIFICATION MATRIX.

Seven sweeps, four distinct tessellations (seeds 11/17/23/41).  Score every
sweep's image against every candidate tessellation - its own and 43 wrong
ones.  If the coda images grain boundaries, each sweep must rank its OWN
tessellation top.

Pre-specified PRIMARY cell: gate=coda, compound=inc, z-support=slab,
blur=2 mm, Spearman.  Pre-specified SECONDARY: same with compound=max (the
specular-aware operator of item 4).  Everything else is exploratory and is
reported with its multiplicity.

Two nulls:
  wrong-tessellation   43 alternative Laguerre specimens, same statistics,
                       same code path.  p = (1+#{null>=true})/44, floor 0.023
  rotation             the specimen's OWN map rotated by 72 non-zero angles
                       (keeps its texture, destroys registration only)
"""
import os as _os
import sys as _sys
# helper modules shared with sibling directories
for _d in ('facet_model',):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

import t4_common as T

SCR = T.SCR
PIX_MM = 0.5
R_MAX = 42.0            # mm
MODES = ("inc", "coh", "max", "top3")
GATES = ("coda", "full")
ZS = ("mid", "slab", "full")
BLURS = (1.0, 2.0, 3.0, 4.0)
SWEEPS = [("girdle_perp_ppw8", 11), ("singlemax_ppw8", 11),
          ("rigid_seed11", 11), ("girdle_par", 11),
          ("kappa8_seed17", 17), ("oos_seed23", 23), ("iso_gcal", 41)]
REAL = (11, 17, 23, 41)


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    n = len(p)
    adj = np.empty(n)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (n - i) * p[k])
        adj[k] = min(run, 1.0)
    return adj


def main():
    Z = np.load(os.path.join(SCR, "t4_maps.npz"))
    x = Z["x_mm"]
    X, Y = np.meshgrid(x, x, indexing="ij")
    core = (X ** 2 + Y ** 2) <= R_MAX ** 2
    nulls = sorted({int(k.split("_")[0][4:]) for k in Z.files
                    if k.startswith("null")})
    # cache blurred maps
    B = {}
    for zk in ZS:
        for bl in BLURS:
            B[("true", zk, bl)] = ndimage.gaussian_filter(
                Z[f"true_{zk}"], bl / PIX_MM)
            for s in nulls:
                B[(s, zk, bl)] = ndimage.gaussian_filter(
                    Z[f"null{s}_{zk}"], bl / PIX_MM)

    def cand_map(seed, zk, bl):
        return B[("true", zk, bl)] if seed == 11 else B[(seed, zk, bl)]

    res = {}
    for nm, own in SWEEPS:
        for gate in GATES:
            D = np.load(os.path.join(SCR, f"t4stk_{nm}_{gate}.npz"))
            m = D["mask"] & core
            for mode in MODES:
                img = D[f"img_{mode}"][m]
                for zk in ZS:
                    for bl in BLURS:
                        sc = {s: spearman(img, cand_map(s, zk, bl)[m])
                              for s in ([11] + nulls)}
                        res[(nm, gate, mode, zk, bl)] = sc
    np.save(os.path.join(SCR, "t4_ident.npy"),
            np.array([res], dtype=object), allow_pickle=True)

    def report(gate, mode, zk, bl, title):
        print(f"\n=== {title}: gate={gate} compound={mode} z={zk} "
              f"blur={bl} mm  (Spearman, n_pix="
              f"{int(core.sum())}) ===")
        print(f"{'sweep':<18}{'own':>5}{'r_own':>8}{'rank':>6}"
              f"{'/N':>4}{'p':>8}   " + "".join(f"r(s{s}){'':<2}"
                                                for s in REAL))
        ps = []
        for nm, own in SWEEPS:
            sc = res[(nm, gate, mode, zk, bl)]
            keys = [11] + nulls
            vals = np.array([sc[k] for k in keys])
            # own tessellation's score under this sweep
            r_own = sc[own] if own in sc else np.nan
            rank = int(np.sum(vals >= r_own)) if np.isfinite(r_own) else -1
            p = (1 + np.sum(vals[[k != own for k in keys]] >= r_own)) \
                / len(vals)
            ps.append(p)
            row = f"{nm:<18}{own:>5}{r_own:>8.3f}{rank:>6}{len(vals):>4}" \
                  f"{p:>8.3f}   "
            row += "".join(f"{sc[s]:>7.3f}" for s in REAL)
            print(row)
        ph = holm(ps)
        print("Holm-adjusted p: " + "  ".join(f"{nm.split('_')[0][:9]}"
                                              f"={q:.3f}"
                                              for (nm, _), q in
                                              zip(SWEEPS, ph)))
        return np.array(ps)

    p1 = report("coda", "inc", "slab", 2.0, "PRIMARY")
    p2 = report("coda", "max", "slab", 2.0, "SECONDARY (specular-aware)")

    # exploratory grid: how often does the own tessellation win?
    print("\n=== EXPLORATORY GRID (all 2x4x3x4 = 96 cells x 7 sweeps) ===")
    hits = tot = 0
    best = []
    for k, sc in res.items():
        nm = k[0]
        own = dict(SWEEPS)[nm]
        keys = [11] + nulls
        vals = np.array([sc[j] for j in keys])
        p = (1 + np.sum(vals[[j != own for j in keys]] >= sc[own])) / len(vals)
        tot += 1
        hits += p <= 0.05
        best.append((p, k, sc[own]))
    print(f"cells with p<=0.05 : {hits} of {tot} "
          f"(expected by chance ~{0.05*tot:.0f}); the cells are strongly "
          "correlated with each other, so this is NOT 672 independent tests")
    best.sort()
    print("10 strongest cells:")
    for p, k, r in best[:10]:
        print(f"   p={p:.3f} r={r:+.3f}  {k}")

    # per-sweep: fraction of the 96 cells in which the own tessellation wins
    print("\nper-sweep fraction of cells with p<=0.05:")
    for nm, own in SWEEPS:
        c = [b for b in best if b[1][0] == nm]
        print(f"   {nm:<18}{sum(1 for b in c if b[0] <= 0.05):>3}/{len(c)}")

    # rotation null on the PRIMARY cell
    print("\n=== ROTATION NULL on the PRIMARY cell ===")
    for nm, own in SWEEPS:
        D = np.load(os.path.join(SCR, f"t4stk_{nm}_coda.npz"))
        m = D["mask"] & core
        img = D["img_inc"][m]
        base = cand_map(own, "slab", 2.0)
        r0 = spearman(img, base[m])
        rr = []
        for ang in np.arange(5, 360, 5):
            rot = ndimage.rotate(base, ang, reshape=False, order=1,
                                 mode="nearest")
            rr.append(spearman(img, rot[m]))
        rr = np.array(rr)
        p = (1 + np.sum(rr >= r0)) / (len(rr) + 1)
        print(f"{nm:<18} r0={r0:+.3f}  rot-null mean={rr.mean():+.3f} "
              f"sd={rr.std():.3f}  p={p:.3f}")


if __name__ == "__main__":
    main()
