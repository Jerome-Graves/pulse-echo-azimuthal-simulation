"""TEST 2 main run: does the facet model predict WHEN each echo arrives?"""
import numpy as np

import _t2_common as C
import shift_null_2d as S

CASES = [("girdle_perp_ppw8", "gp8", "correct"),
         ("singlemax_ppw8", "sm8", "correct"),
         ("girdle_perp_ppw8", "wrongseed8", "WRONG tessellation"),
         ("singlemax_ppw8", "wrongseed8", "WRONG tessellation"),
         ("iso_gcal", "iso6", "ISOTROPIC CONTROL")]

D = {}
for sw, tg, lbl in CASES:
    D[(sw, tg)] = S.load(sw, tg)

print("=" * 78)
print("A. POOLED azimuth x time correlation, log (dB) domain")
print("=" * 78)
print(f"{'measurement':<19}{'predictor':<12}{'variant':<9}"
      f"{'r raw':>8}{'r col':>8}{'r dbl':>8}   note")
for sw, tg, lbl in CASES:
    az, tt, M, P = D[(sw, tg)]
    for v in ("full", "geom", "nodirec", "flat"):
        rr = [S.score(M, P[v], m) for m in ("raw", "col", "double")]
        print(f"{sw:<19}{tg:<12}{v:<9}"
              f"{rr[0]:>8.3f}{rr[1]:>8.3f}{rr[2]:>8.3f}   {lbl}")
    print()

print("=" * 78)
print("B. LINEAR power domain (same fields, no log)")
print("=" * 78)
print(f"{'measurement':<19}{'predictor':<12}{'variant':<9}"
      f"{'r raw':>8}{'r col':>8}{'r dbl':>8}")
for sw, tg, lbl in CASES:
    az, tt, M, P = D[(sw, tg)]
    for v in ("full", "geom"):
        rr = [S.score(M, P[v], m, log=False) for m in ("raw", "col", "double")]
        print(f"{sw:<19}{tg:<12}{v:<9}"
              f"{rr[0]:>8.3f}{rr[1]:>8.3f}{rr[2]:>8.3f}")

print()
print("=" * 78)
print("C. NULLS.  p = fraction of circular shifts of the MEASURED matrix")
print("   scoring |r| >= |r0|.  col = depth profile removed;")
print("   double = depth profile AND per-azimuth level removed.")
print("=" * 78)
rows = []
for sw, tg, lbl in CASES:
    az, tt, M, P = D[(sw, tg)]
    for v in ("full", "geom"):
        for mode in ("col", "double"):
            r0, p_az, n_az = S.shift_null(M, P[v], mode, kind="az")
            _, p_t, n_t = S.shift_null(M, P[v], mode, kind="time")
            _, p_2d, n_2 = S.shift_null(M, P[v], mode, kind="2d")
            ne = S.n_eff(S.centre(10 * np.log10(M + S.EPS), mode),
                         S.centre(10 * np.log10(P[v] + S.EPS), mode))
            rows.append([sw, tg, v, mode, r0, p_az, p_t, p_2d, ne,
                         S.t_p(r0, ne), lbl])
print(f"{'measurement':<19}{'pred':<11}{'var':<6}{'ctr':<7}{'r':>7}"
      f"{'p_az':>8}{'p_time':>8}{'p_2D':>9}{'N_eff':>8}{'p_par':>10}")
for r in rows:
    print(f"{r[0]:<19}{r[1]:<11}{r[2]:<6}{r[3]:<7}{r[4]:>7.3f}"
          f"{r[5]:>8.4f}{r[6]:>8.4f}{r[7]:>9.5f}{r[8]:>8.1f}{r[9]:>10.2e}"
          f"   {r[10]}")
np.save("t2_rows.npy", np.array(rows, dtype=object), allow_pickle=True)
