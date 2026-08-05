"""TEST 2 final: exhaustive 2-D circular-shift null + normal-scramble control.

The centring used ('harm') commutes with BOTH cyclic rolls, so the null over
every one of the n_az x n_t joint shifts can be evaluated exactly by FFT
cross-correlation instead of being sub-sampled.  p floor = 1/(60*600).
"""
import os as _os
import sys as _sys
# helper modules shared with sibling directories
for _d in ('..',):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import os

import numpy as np

import _t2_common as C
import shift_null_2d as S

MODE = "harm"
CASES = [
    ("girdle_seed11_ppw8_dev", "gp8", "CORRECT"),
    ("singlemax_seed11_ppw8_twin", "sm8", "CORRECT"),
    ("isotropic_seed41_ppw6_calibration", "iso6", "random-fabric sweep, own tessellation"),
    ("girdle_seed11_ppw8_dev", "sm8", "same tessellation, WRONG fabric"),
    ("singlemax_seed11_ppw8_twin", "gp8", "same tessellation, WRONG fabric"),
    ("girdle_seed11_ppw8_dev", "wrongseed8", "WRONG tessellation"),
    ("singlemax_seed11_ppw8_twin", "wrongseed8", "WRONG tessellation"),
    ("isotropic_seed41_ppw6_calibration", "wrongseed6", "WRONG tessellation"),
]


def all_shift_r(mc, pc):
    """r for every cyclic (az, time) shift, exactly, by FFT."""
    F = np.fft.rfft2(mc) * np.conj(np.fft.rfft2(pc))
    R = np.fft.irfft2(F, s=mc.shape)
    return R / np.sqrt((mc ** 2).sum() * (pc ** 2).sum())


def exact_p(M, P, mode=MODE):
    mc = S.centre(10 * np.log10(M + S.EPS), mode)
    pc = S.centre(10 * np.log10(P + S.EPS), mode)
    R = all_shift_r(mc, pc)
    r0 = R[0, 0]
    n = R.size
    p = float((np.abs(R) >= abs(r0)).sum() / n)
    # azimuth-only and time-only marginals of the same exhaustive null
    p_az = float((np.abs(R[:, 0]) >= abs(r0)).sum() / R.shape[0])
    p_t = float((np.abs(R[0]) >= abs(r0)).sum() / R.shape[1])
    return r0, p, p_az, p_t, n


print("=" * 90)
print("EXHAUSTIVE 2-D circular-shift null, harm-centring")
print("(azimuth harmonics k=0..4 removed at every time sample, then the")
print(" per-azimuth level removed: no texture template and no total-energy")
print(" information survives -- only WHERE in time the echoes sit.)")
print("=" * 90)
print(f"{'measurement':<18}{'predictor':<11}{'var':<8}{'r':>7}{'r2 %':>7}"
      f"{'p_2D':>10}{'p_az':>8}{'p_t':>8}   note")
allp = []
for sw, tg, lbl in CASES:
    az, tt, M, P = S.load(sw, tg)
    for v in ("full", "geom", "nodirec", "flat"):
        r0, p, p_az, p_t, n = exact_p(M, P[v])
        allp.append(p)
        print(f"{sw:<18}{tg:<11}{v:<8}{r0:>7.3f}{100*r0**2:>7.2f}"
              f"{p:>10.5f}{p_az:>8.4f}{p_t:>8.4f}   {lbl}")
    print()
print(f"n_shifts per test = {n};  p floor = {1/n:.2e}")
print(f"tests in this table = {len(allp)}")

print()
print("=" * 90)
print("FACET-NORMAL SCRAMBLE.  Same facets, same arrival times, same multiset")
print("of directivity weights -- but the weight/time pairing is destroyed.")
print("If the exact Laguerre normals are what carries the prediction, this")
print("must collapse to null.  20 independent scrambles per case.")
print("=" * 90)
TG = np.arange(20e-6, 42e-6, C.DT_C)
GATE = (TG >= C.CODA_W[0]) & (TG < C.CODA_W[1])
edges = np.concatenate([TG - C.DT_C / 2, [TG[-1] + C.DT_C / 2]])
kern = C.pulse_power_kernel(C.DT_C)
for sw, tg in (("girdle_seed11_ppw8_dev", "gp8"), ("singlemax_seed11_ppw8_twin", "sm8")):
    az0, tt, M, P = S.load(sw, tg)
    b = np.load(os.path.join(C.HERE, f"t2b_{tg}.npz"))
    lab, ax, sd, h = (b["labels"].astype(np.int32), b["axes"], b["seeds"],
                      float(b["h"]))
    ev = [C.facet_events(lab, ax, sd, h, a, use_dv=False, use_direc=True)
          for a in az0]
    rs = []
    for trial in range(20):
        rng = np.random.default_rng(1000 + trial)
        rows = []
        for t, w, _ in ev:
            rows.append(np.histogram(t, edges,
                                     weights=rng.permutation(w))[0])
        Q = np.apply_along_axis(lambda r: np.convolve(r, kern, "same"), 1,
                                np.array(rows))[:, GATE]
        rs.append(S.score(M, Q, MODE))
    rs = np.array(rs)
    r_true = S.score(M, P["geom"], MODE)
    z = (r_true - rs.mean()) / rs.std()
    print(f"{sw:<18} true r={r_true:.3f}   scrambled r = "
          f"{rs.mean():+.3f} +- {rs.std():.3f}  "
          f"(min {rs.min():+.3f}, max {rs.max():+.3f})   z={z:.1f}")

print()
print("=" * 90)
print("DEPTH DEPENDENCE, harm-centred, geom variant, exhaustive 2-D null")
print("=" * 90)
for sw, tg, lbl in CASES[:2] + [CASES[5], CASES[6]]:
    az, tt, M, P = S.load(sw, tg)
    print(f"\n{sw} <- {tg}   [{lbl}]")
    print(f"  {'gate us':<10}{'range mm':<12}{'r':>7}{'r2 %':>7}{'p_2D':>10}")
    for lo in np.arange(24, 36, 2) * 1e-6:
        sel = (tt >= lo) & (tt < lo + 2e-6)
        r0, p, _, _, n2 = exact_p(M[:, sel], P["geom"][:, sel])
        s0 = (lo - C.T0_SRC) / 2 * C.C_REF * 1e3
        s1 = (lo + 2e-6 - C.T0_SRC) / 2 * C.C_REF * 1e3
        print(f"  {lo*1e6:.0f}-{lo*1e6+2:<7.0f}{s0:.0f}-{s1:<9.0f}"
              f"{r0:>7.3f}{100*r0**2:>7.2f}{p:>10.5f}")
