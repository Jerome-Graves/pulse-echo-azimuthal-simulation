"""TEST 4 item 4: the specular-visibility census, and the resolution limit.

Every Laguerre facet has an EXACT normal (seed_i - seed_j) and an exact
planar extent, so "can this facet ever backscatter to the probe?" is a
closed-form question.  A facet is counted VISIBLE from an azimuth when

  * its centroid lies inside the 8.9 deg beam cone (d_perp <= beam radius)
  * its two-way delay falls in the gate
  * the specular directivity |2J1(x)/x|^2 >= 0.25 (-6 dB), with
    x = k*D_eff*sin(angle between facet normal and the facet->probe ray)

D_eff is the coherent aperture of the facet: its own equivalent diameter,
capped at the first Fresnel-zone diameter 2*sqrt(lambda*r/2) at that range,
because a spherical incident wave cannot use more of a flat facet than that.
Both the capped and uncapped answers are reported - the cap widens the lobe
for the big facets and materially changes the census.
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
from scipy.special import j1

import t4_common as T

SCR = T.SCR
AZ = np.arange(0, 360, 6)          # the 60-azimuth production sweep
AZ_FINE = np.arange(0, 360, 0.25)  # for the angular-width measurement
GATES = dict(coda=(24e-6, 36e-6), wide=(12e-6, 48e-6))
C = T.C_REF


def facet_table(seed=11):
    b = T.build_spec(8.0, -8.0, (1.0, 0.0, 0.0), seed)
    lab, sd, h = b["labels"], b["seeds"], b["h"]
    pos, pair, nrm = T.facets_from_labels(lab, sd, h)
    key = pair[:, 0] * 10000 + pair[:, 1]
    uk, inv = np.unique(key, return_inverse=True)
    n = len(uk)
    cnt = np.bincount(inv, minlength=n)
    cen = np.stack([np.bincount(inv, pos[:, k], n) / cnt for k in range(3)], 1)
    nm = np.stack([np.bincount(inv, nrm[:, k], n) / cnt for k in range(3)], 1)
    nm /= np.linalg.norm(nm, axis=1, keepdims=True)
    # voxel-face count -> true planar area: sum of projected areas is
    # A*(|nx|+|ny|+|nz|)
    proj = np.abs(nm).sum(1)
    area = cnt * h ** 2 / proj
    deq = 2 * np.sqrt(area / np.pi)
    return cen, nm, area, deq, cnt


def directivity(ct, D):
    st = np.sqrt(np.clip(1 - ct ** 2, 0, 1))
    x = T.K * D * st
    d = np.where(x < 1e-9, 1.0, 2 * j1(np.where(x < 1e-9, 1.0, x))
                 / np.where(x < 1e-9, 1.0, x))
    return d ** 2


def census(cen, nm, area, deq, az, gate, fresnel=True):
    R = T.DIA / 2
    nv = np.zeros(len(cen), int)          # azimuths from which it is visible
    inbeam = np.zeros(len(cen), int)
    for a in az:
        th = np.radians(float(a))
        q = np.array([R * np.cos(th), R * np.sin(th), 0.0])
        b = -np.array([np.cos(th), np.sin(th), 0.0])
        v = cen - q
        r = np.linalg.norm(v, axis=1)
        dpar = v @ b
        dperp = np.linalg.norm(v - dpar[:, None] * b[None, :], axis=1)
        bw = T.beam_radius(dpar)
        t2 = 2 * r / C
        ok = (dpar > 0) & (dperp <= bw) & (t2 >= gate[0]) & (t2 <= gate[1])
        u = -v / r[:, None]
        ct = np.abs(np.einsum("ij,ij->i", nm, u))
        D = np.minimum(deq, 2 * np.sqrt(T.LAM * r / 2)) if fresnel else deq
        vis = ok & (directivity(ct, D) >= 0.25)
        nv += vis
        inbeam += ok
    return nv, inbeam


def main():
    cen, nm, area, deq, cnt = facet_table(11)
    print(f"seed-11 tessellation: {len(cen)} distinct grain-boundary facets")
    print(f"  facet equivalent diameter: median {np.median(deq)*1e3:.1f} mm, "
          f"10-90 pct {np.percentile(deq,10)*1e3:.1f}-"
          f"{np.percentile(deq,90)*1e3:.1f} mm")
    print(f"  total facet area {area.sum()*1e6:.0f} mm^2\n")

    for gname, g in GATES.items():
        for fres in (True, False):
            nv, ib = census(cen, nm, area, deq, AZ, g, fres)
            sel = ib > 0                     # facets the beam ever reaches
            tag = "Fresnel-capped" if fres else "full-facet"
            print(f"gate={gname:<5} {tag:<15} "
                  f"reachable {sel.sum():>4}/{len(cen)}  "
                  f"visible>=1 az: {int((nv>0).sum()):>4} "
                  f"({100*(nv>0).sum()/max(sel.sum(),1):.1f}% of reachable, "
                  f"{100*(nv>0).sum()/len(cen):.1f}% of all)  "
                  f"area-weighted {100*area[nv>0].sum()/max(area[sel].sum(),1e-30):.1f}%  "
                  f"mean az/visible facet {nv[nv>0].mean() if (nv>0).any() else 0:.2f}")
    print()

    # angular width of the specular window, measured on a fine azimuth grid
    nvf, ibf = census(cen, nm, area, deq, AZ_FINE, GATES["coda"], True)
    dth = 0.25
    w = nvf[nvf > 0] * dth
    print(f"specular window width (fine {dth} deg scan, coda gate): "
          f"median {np.median(w):.2f} deg, mean {w.mean():.2f} deg, "
          f"90th {np.percentile(w,90):.2f} deg  (n={len(w)} facets)")
    print(f"facets with a specular window at all: {len(w)} of "
          f"{int((ibf>0).sum())} reachable")
    p_hit = np.median(w) / 6.0
    print(f"-> with a 6 deg azimuth step, a median facet is caught by "
          f"{p_hit:.2f} azimuths; probability a given facet is missed "
          f"entirely ~ {max(0.0, 1-p_hit):.2f}")

    # per-azimuth glint count
    R = T.DIA / 2
    ng = []
    for a in AZ:
        th = np.radians(float(a))
        q = np.array([R * np.cos(th), R * np.sin(th), 0.0])
        b = -np.array([np.cos(th), np.sin(th), 0.0])
        v = cen - q
        r = np.linalg.norm(v, axis=1)
        dpar = v @ b
        dperp = np.linalg.norm(v - dpar[:, None] * b[None, :], axis=1)
        t2 = 2 * r / C
        ok = (dpar > 0) & (dperp <= T.beam_radius(dpar)) \
            & (t2 >= GATES["coda"][0]) & (t2 <= GATES["coda"][1])
        u = -v / r[:, None]
        ct = np.abs(np.einsum("ij,ij->i", nm, u))
        D = np.minimum(deq, 2 * np.sqrt(T.LAM * r / 2))
        ng.append(int((ok & (directivity(ct, D) >= 0.25)).sum()))
    ng = np.array(ng)
    print(f"\nfacets in the -6 dB lobe per azimuth (coda gate): "
          f"mean {ng.mean():.2f}, median {np.median(ng):.0f}, "
          f"max {ng.max()},  azimuths with ZERO specular facet: "
          f"{int((ng==0).sum())}/{len(AZ)}")

    # resolution budget
    bw = 3.0e6 - 0.8e6
    dr = C / (2 * bw)
    vspread = 0.069
    print(f"\nRESOLUTION BUDGET")
    print(f"  range resolution  c/2B  = {dr*1e3:.2f} mm "
          f"(B = 2.2 MHz after the 0.8-3.0 MHz band-pass)")
    print(f"  range ACCURACY with an unknown fabric: +-{vspread/2*57:.1f} mm "
          f"at 57 mm (qP spread {100*vspread:.1f}%), i.e. the dominant term")
    print(f"  cross-range, single view = beam diameter "
          f"{2*T.beam_radius(0.046)*1e3:.0f}-{2*T.beam_radius(0.069)*1e3:.0f} mm")
    ap = np.median(w)
    xr = T.LAM / (4 * np.sin(np.radians(ap) / 2)) if ap > 0 else np.inf
    print(f"  cross-range, synthetic aperture limited by the specular "
          f"window {ap:.2f} deg -> lambda/(4 sin(dtheta/2)) = {xr*1e3:.1f} mm")
    print(f"  grain diameter 17.4 mm; Fresnel zone at 57 mm 14.8 mm")


if __name__ == "__main__":
    main()
