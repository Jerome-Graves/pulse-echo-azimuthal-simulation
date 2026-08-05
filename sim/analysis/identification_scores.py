"""TEST 4 step 3: does the back-projected image match ITS OWN specimen's
grain boundaries better than 43 wrong tessellations?

Metrics, all computed identically for the true map and every null map:
  rS      Spearman rank correlation image vs blurred boundary map
  rP      Pearson on the dB image
  AUC     image value as a detector of "boundary within 1.5 mm"
  dpk     median distance from the 40 brightest image peaks to the nearest
          boundary pixel (lower = better)

Null = the 43 wrong tessellations (seeds 17/23/41 = the other real sweeps,
plus 40 fresh seeds), which have identical statistics and identical
processing.  Empirical p = (1 + #{null >= true}) / (M + 1).
A SECOND null rotates the true map through 359 non-zero angles: it keeps the
map's own texture and destroys only the registration.
"""
import os as _os
import sys as _sys
# helper modules shared with sibling directories
for _d in ('facet_model',):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import itertools
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

import range_domain_common as T

SCR = T.SCR
BLUR_MM = [1.0, 2.0, 3.0, 4.0, 6.0]
R_MAX = 42e-3            # analysis disc: keep clear of the rim
PIX_MM = 0.5
NPEAK = 40


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def auc(score, lab):
    r = rankdata(score)
    n1 = lab.sum()
    n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def peak_coords(img, mask, npk=NPEAK, min_sep_px=6):
    v = np.where(mask, img, -np.inf)
    mx = ndimage.maximum_filter(v, size=2 * min_sep_px + 1)
    cand = np.argwhere((v == mx) & mask)
    val = v[cand[:, 0], cand[:, 1]]
    o = np.argsort(-val)
    return cand[o[:npk]]


def score_pair(img, bmap, mask, blur_px, pk):
    b = ndimage.gaussian_filter(bmap.astype(float), blur_px)
    a = img[mask]
    c = b[mask]
    rS = spearman(a, c)
    db = 20 * np.log10(np.maximum(img, 1e-12))
    rP = float(np.corrcoef(db[mask], c)[0, 1])
    # binary label from the UNBLURRED map dilated to 1.5 mm
    lab = ndimage.binary_dilation(bmap > 0,
                                  iterations=int(round(1.5 / PIX_MM)))
    A = auc(a, lab[mask])
    dt = ndimage.distance_transform_edt(~(bmap > 0)) * PIX_MM
    dpk = float(np.median(dt[pk[:, 0], pk[:, 1]]))
    return rS, rP, A, dpk


def main():
    Z = np.load(os.path.join(SCR, "t4_maps.npz"))
    x_mm = Z["x_mm"]
    X, Y = np.meshgrid(x_mm, x_mm, indexing="ij")
    core = (X ** 2 + Y ** 2) <= (R_MAX * 1e3) ** 2
    nulls = sorted({int(k.split("_")[0][4:]) for k in Z.files
                    if k.startswith("null")})
    print(f"null ensemble: {len(nulls)} wrong tessellations  {nulls}\n")

    rows = []
    for sw in ("girdle_seed11_ppw8_dev", "singlemax_seed11_ppw8_twin"):
        for gate in ("coda", "full"):
            D = np.load(os.path.join(SCR, f"t4img_{sw}_{gate}.npz"))
            m = D["mask"] & core
            for mode in ("inc", "coh"):
                img = np.nan_to_num(D[f"img_{mode}"], nan=0.0)
                pk = peak_coords(img, m)
                for zk in ("mid", "slab", "full"):
                    for bl in BLUR_MM:
                        bpx = bl / PIX_MM
                        t = score_pair(img, Z[f"true_{zk}"], m, bpx, pk)
                        nl = np.array([score_pair(img, Z[f"null{s}_{zk}"],
                                                  m, bpx, pk)
                                       for s in nulls])
                        rows.append(dict(sw=sw, gate=gate, mode=mode, z=zk,
                                         blur=bl, true=t, null=nl))
                    print(f"  done {sw} {gate} {mode} {zk}", flush=True)
    np.save(os.path.join(SCR, "t4_scoobservable_panel_results.npy"),
            np.array(rows, dtype=object), allow_pickle=True)

    names = ["rS", "rP", "AUC", "dpk"]
    sign = [+1, +1, +1, -1]          # +1: bigger is better
    print(f"\n{'sweep':<17}{'gate':<6}{'mode':<5}{'z':<6}{'blur':>5}"
          + "".join(f"{n:>9}{'p':>7}" for n in names))
    for r in rows:
        s = f"{r['sw']:<17}{r['gate']:<6}{r['mode']:<5}{r['z']:<6}" \
            f"{r['blur']:>5.1f}"
        for i, n in enumerate(names):
            tv, nv = r["true"][i], r["null"][:, i]
            p = (1 + np.sum(sign[i] * nv >= sign[i] * tv)) / (len(nv) + 1)
            s += f"{tv:>9.3f}{p:>7.3f}"
        print(s)


if __name__ == "__main__":
    main()
