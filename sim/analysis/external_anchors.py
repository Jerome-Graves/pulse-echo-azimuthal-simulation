"""External anchors: what published results can this simulation be checked
against WITHOUT new computation?

Three numbers are produced here, all from the cached Laguerre geometry in
physical_optics/po_src_geom.npz and the CPU physical-optics kernel of
po_src_03_kirchhoff_pred.py.  Nothing here touches CUDA and nothing here runs an FDTD
forward model.

  (1) FACET POWER CONCENTRATION.  How much of the gated coda power is
      carried by how few boundaries.  Each boundary is traced ALONE
      through the same Kirchhoff integral, band-limited by the same
      Ricker, gated by the same 24-36 us window, and its gated envelope
      energy recorded.  The concentration statistic is then the ordered
      cumulative share of that incoherent decomposition, pooled over the
      30 production azimuths.  This is the quantity the metals literature
      would have to match for the "a handful of boundaries carries the
      return" claim to be a shared property rather than an ice one.

  (2) MASON-McSKIMIN GEOMETRIC-LIMIT ATTENUATION.  Mason and McSkimin
      (J. Appl. Phys. 19, 940, 1948; doi 10.1063/1.1697900, corrected by
      doi 10.1063/1.1698343) showed that once the wavelength is a third
      of the grain size or less the transmission becomes a diffusion
      process, the grain size sets the mean free path, and the loss goes
      inversely with grain diameter and with the mean boundary reflection
      coefficient.  For a convex-cell tessellation the crossings per unit
      path length of a random line is S_v/2 (stereology), each crossing
      removing a fraction <R^2> of the intensity, so

          alpha_I = S_v <R^2> / 2      (intensity, per metre)

      which is frequency independent - the geometric-limit signature.
      Both ingredients, S_v and <R^2>, are already measured in this
      paper, so the law can be evaluated with no free parameter and
      compared with the specimen's two-way path.

  (3) THE SAME LAW AT A METALS CONTRAST.  Anderson et al. (2007,
      NUREG/CR-6933, doi 10.2172/921260) report grain-to-grain velocity
      differences of up to 14 per cent in cast austenitic stainless
      steel.  Substituting that contrast for ice's shows how far the
      per-boundary reflection coefficient, not D/lambda, separates the
      two materials.

Run:  python sim/analysis/external_anchors.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PO_DIR = os.path.join(HERE, "physical_optics")
if PO_DIR not in sys.path:
    sys.path.insert(0, PO_DIR)

import po_src_03_kirchhoff_pred as PO                                  # noqa: E402
from po_src_03_kirchhoff_pred import (C, F0, LAM, DIA, THK, GATE, NF, FAREA,   # noqa: E402
                        ricker_spec, trace_from, env_rms_gate)

AZ = np.arange(0, 360, 12)          # the 30 production azimuths
DS = LAM / 16                       # the reference quadrature of step 3


def per_facet_gate_energy(az_deg, ds, Wspec):
    """Gated envelope energy of every boundary traced on its own.

    Returns (facet_index, energy) for the boundaries that put any energy
    in the gate at this azimuth.  Facets are traced individually, so the
    decomposition is incoherent by construction; the coherent sum over
    all of them is what po_src_03_kirchhoff_pred reports as the azimuth level.
    """
    a = np.radians(az_deg)
    nb = np.array([np.cos(a), np.sin(a), 0.0])
    p0 = PO.R_DISC * nb
    axis = -nb
    dlo = (GATE[0] - 2e-6) * C / 2
    dhi = (GATE[1] + 2e-6) * C / 2

    idx, eng = [], []
    for k in range(NF):
        pk = PO.poly_of(k)
        rc = np.linalg.norm(pk[:, 0:1] * PO.FU[k][None, :]
                            + pk[:, 1:2] * PO.FV[k][None, :]
                            + PO.FX0[k][None, :] - PO.FCEN[k][None, :],
                            axis=1).max()
        vc = PO.FCEN[k] - p0
        dc = np.linalg.norm(vc)
        if dc + rc < dlo or dc - rc > dhi:
            continue
        cphi_c = (vc / dc) @ axis
        ang = np.arccos(np.clip(cphi_c, -1, 1)) - np.arcsin(min(1.0, rc / dc))
        if ang > np.arcsin(PO.S_MAX):
            continue
        X, dA = PO.quad_points(k, ds)
        if X is None:
            continue
        V = X - p0[None, :]
        d = np.linalg.norm(V, axis=1)
        U = V / d[:, None]
        cphi = U @ axis
        sphi = np.sqrt(np.clip(1 - cphi ** 2, 0, 1))
        cth = U @ PO.FN[k]
        vi = np.interp(np.arccos(np.clip(np.abs(U @ PO.AXES[PO.FI[k]]), 0, 1)),
                       PO.FW._PSI, PO.FW._VQP)
        vj = np.interp(np.arccos(np.clip(np.abs(U @ PO.AXES[PO.FJ[k]]), 0, 1)),
                       PO.FW._PSI, PO.FW._VQP)
        Rg = (vi - vj) / (vi + vj)
        tau = 2 * d / C
        m = ((tau > GATE[0] - 2e-6) & (tau < GATE[1] + 2e-6)
             & (sphi < PO.S_MAX) & (cphi > 0))
        if not m.any():
            continue
        Gh = np.zeros((PO.NT, PO.NS_BIN))
        wgt = (Rg * cth * dA / d ** 2)[m]
        ti = tau[m] / PO.DT
        i0 = np.floor(ti).astype(int)
        fr = ti - i0
        si = np.clip((sphi[m] / PO.S_MAX * PO.NS_BIN).astype(int),
                     0, PO.NS_BIN - 1)
        np.add.at(Gh, (i0, si), wgt * (1 - fr))
        np.add.at(Gh, (i0 + 1, si), wgt * fr)
        e = env_rms_gate(trace_from(Gh, Wspec)) ** 2
        if e > 0:
            idx.append(k)
            eng.append(e)
    return np.array(idx, int), np.array(eng, float)


def concentration(share):
    """Ordered cumulative share, and the counts that reach given levels."""
    s = np.sort(share)[::-1]
    cum = np.cumsum(s) / s.sum()
    return s, cum


def main():
    Wspec, _ = ricker_spec(PO.NT, PO.DT, F0)

    # ── geometry, recomputed from the cached tessellation ───────────────
    vol = np.pi * (DIA / 2) ** 2 * THK
    s_v = FAREA.sum() / vol
    print("SPECIMEN GEOMETRY (from po_src_geom.npz)")
    print(f"  boundaries                     {NF}")
    print(f"  total boundary area            {FAREA.sum()*1e6:.1f} mm^2")
    print(f"  mean facet area                {FAREA.mean()*1e6:.2f} mm^2")
    print(f"  disc volume                    {vol*1e6:.1f} cm^3"
          .replace("cm^3", "x1e-6 m^3"))
    print(f"  boundary area per unit volume  S_v = {s_v:.1f} /m")
    print(f"  lambda at {F0/1e6:.0f} MHz              {LAM*1e3:.3f} mm\n")

    # ── (1) facet power concentration ──────────────────────────────────
    print("(1) FACET POWER CONCENTRATION "
          f"({len(AZ)} azimuths, ds = lam/{LAM/DS:.0f})")
    t0 = time.time()
    pooled = np.zeros(NF)
    per_az_top1 = []
    for az in AZ:
        idx, eng = per_facet_gate_energy(az, DS, Wspec)
        pooled[idx] += eng
        s, cum = concentration(eng)
        per_az_top1.append(cum[0])
    print(f"  traced {NF} boundaries individually at each azimuth "
          f"({time.time()-t0:.0f} s)")

    live = pooled[pooled > 0]
    s, cum = concentration(live)
    n_live = len(live)
    print(f"  boundaries with any gated energy at any azimuth : "
          f"{n_live} of {NF}")
    for n in (1, 2, 3, 5, 6, 10, 27, 54):
        if n <= n_live:
            print(f"    brightest {n:3d} boundaries "
                  f"({100*n/NF:5.2f} % of all {NF}, "
                  f"{100*n/n_live:5.2f} % of the {n_live} live) "
                  f"carry {100*cum[n-1]:5.1f} % of the pooled gated power")
    for lvl in (0.50, 0.90, 0.92, 0.95, 0.99):
        n = int(np.searchsorted(cum, lvl) + 1)
        print(f"    {100*lvl:4.0f} % of the pooled power needs "
              f"{n:3d} boundaries ({100*n/NF:.2f} % of {NF})")
    print(f"  single-azimuth brightest facet share: mean "
          f"{100*np.mean(per_az_top1):.1f} %, "
          f"min {100*np.min(per_az_top1):.1f} %, "
          f"max {100*np.max(per_az_top1):.1f} %\n")

    # ── (2) Mason-McSkimin geometric-limit attenuation ─────────────────
    # area-weighted rms reflection coefficient over the insonified
    # population, recomputed here rather than quoted.
    print("(2) MASON-McSKIMIN GEOMETRIC-LIMIT ATTENUATION")
    r2_list, w_list = [], []
    for az in AZ[::3]:
        _, diag = PO.azimuth_response(az, DS, want_diag=True)
        for (_k, a_in, _a6, r_rms, _ct, _d) in diag:
            if a_in > 0:
                r2_list.append(r_rms ** 2)
                w_list.append(a_in)
    r2 = np.array(r2_list)
    w = np.array(w_list)
    r2_bar = float((r2 * w).sum() / w.sum())
    print(f"  area-weighted <R^2> over insonified boundaries "
          f"= {r2_bar:.4e}  (|R|_rms = {100*np.sqrt(r2_bar):.3f} %)")
    n_l = s_v / 2.0
    print(f"  boundary crossings per unit path  N_L = S_v/2 = {n_l:.2f} /m")
    alpha_i = n_l * r2_bar
    print(f"  alpha_I = N_L <R^2>               = {alpha_i:.4e} /m "
          f"= {4.342944819 * alpha_i:.4f} dB/m (intensity)")
    two_way = DIA
    print(f"  over the {two_way*1e3:.0f} mm two-way path: "
          f"{4.342944819 * alpha_i * two_way:.5f} dB")
    print("  frequency independent by construction: no f enters.")
    mfp = 1.0 / alpha_i
    print(f"  implied scattering mean free path 1/alpha_I = {mfp:.1f} m")
    print(f"  specimen diameter in mean free paths       = "
          f"{DIA / mfp:.2e}")
    print("  -> the specimen is three orders of magnitude thinner than one "
          "mean free\n     path, so no diffusion-limit anchor (equipartition, "
          "coda diffusivity,\n     energy-transport decay rate) can be brought "
          "to bear on it.\n")

    # ── (3) the same law at a cast-stainless-steel contrast ────────────
    print("(3) THE SAME LAW AT A CAST-STAINLESS CONTRAST")
    for dv in (0.05, 0.14):
        # R = (v_far - v_near)/(v_far + v_near) for a fractional velocity
        # difference dv between the two grains
        r_metal = dv / (2.0 + dv)
        a_metal = n_l * r_metal ** 2
        print(f"  velocity difference {100*dv:4.1f} %  ->  |R| = "
              f"{100*r_metal:.2f} %, ratio to ice {r_metal**2/r2_bar:7.1f}x "
              f"= {10*np.log10(r_metal**2/r2_bar):+5.1f} dB per boundary")
        print(f"      at this specimen's S_v that is "
              f"{4.342944819 * a_metal:.3f} dB/m, "
              f"{4.342944819 * a_metal * two_way:.3f} dB over 100 mm")
    print("\n  (S_v held at the ice value on purpose: the point is that the "
          "per-boundary\n   contrast, not D/lambda, is what separates the "
          "two materials.)")


if __name__ == "__main__":
    main()
