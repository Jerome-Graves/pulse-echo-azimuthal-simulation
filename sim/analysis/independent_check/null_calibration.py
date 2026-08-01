"""SKEPTIC step 10: null calibration from my 64 wrong-tessellation tests, and
the meta-combination redone on the AZIMUTH-ONLY p-values (the only cyclic axis
that is unambiguously exchangeable)."""
import numpy as np
from scipy import stats

# p_2D from every wrong-tessellation test I ran (sk4: 16 seeds x 2 ppw8 sweeps,
# sk7: 8 seeds x 4 ppw6 sweeps).  Under a valid null these must be ~U(0,1).
p_wrong = [
    # girdle_perp_ppw8, 16 wrong seeds
    0.72950, 0.87164, 0.52331, 0.59119, 0.86553, 0.68000, 0.13689, 0.41944,
    0.47067, 0.40183, 0.95789, 0.83461, 0.60175, 0.67303, 0.38850, 0.93378,
    # singlemax_ppw8, 16 wrong seeds
    0.48792, 0.08697, 0.17694, 0.81761, 0.14592, 0.83039, 0.30700, 0.63711,
    0.05514, 0.92792, 0.64219, 0.93067, 0.04833, 0.62569, 0.24219, 0.99997,
    # iso_gcal, oos_seed23, kappa8_seed17, rigid_seed11: 8 wrong seeds each
    0.63457, 0.40932, 0.73792, 0.87346, 0.95516, 0.82528, 0.48602, 0.10013,
    0.06186, 0.36884, 0.84951, 0.49330, 0.85200, 0.30531, 0.12676, 0.08865,
    0.92978, 0.87717, 0.31911, 0.66633, 0.13583, 0.44087, 0.52472, 0.47828,
    0.88204, 0.98010, 0.23743, 0.24265, 0.26799, 0.02129, 0.60960, 0.56795,
]
p_wrong = np.array(p_wrong)
print(f"64 wrong-tessellation exhaustive-2D p-values")
print(f"  mean {p_wrong.mean():.3f} (U(0,1) -> 0.5), "
      f"# < 0.05: {(p_wrong < 0.05).sum()} (expect 3.2), "
      f"# < 0.10: {(p_wrong < 0.10).sum()} (expect 6.4)")
ks = stats.kstest(p_wrong, "uniform")
print(f"  KS vs uniform: D={ks.statistic:.3f}, p={ks.pvalue:.3f}  "
      f"-> the exhaustive 2-D shift null is empirically calibrated"
      f"{' (slightly conservative)' if p_wrong.mean() > 0.5 else ''}")

print()
print("META-COMBINATION over the FOUR INDEPENDENT TESSELLATIONS, redone on the")
print("AZIMUTH-ONLY randomisation p (azimuth is genuinely cyclic; time is not).")
# p_az measured by me
cases = {"seed11 (rigid_seed11, 360 az)": 0.0111,
         "seed23 (oos_seed23, 360 az)": 0.0056,
         "seed17 (kappa8_seed17, 90 az)": 0.0111,
         "seed41 (iso_gcal, 360 az)": 0.0111}
p = np.array(list(cases.values()))
X2 = -2 * np.log(p).sum()
print(f"  p_az = {p}  ->  Fisher X2 = {X2:.1f}, df = 8, "
      f"p = {stats.chi2.sf(X2, 8):.2e}")
alt = p.copy()
alt[0] = 0.0167          # use girdle_perp_ppw8 (floor 1/60) for seed 11
X2b = -2 * np.log(alt).sum()
print(f"  with the ppw8 sweep for seed 11: X2 = {X2b:.1f}, "
      f"p = {stats.chi2.sf(X2b, 8):.2e}")
print(f"  (their reported Fisher on the 2-D p: 2.6e-8 to 6.4e-10)")

print()
print("RANK EVIDENCE, four independent tessellations, my 8 fresh wrong")
print("tessellations each: correct beaten 0/8 every time -> exact combined")
print(f"  p = (1/9)^4 = {(1/9)**4:.2e} (assumes only that a wrong tessellation")
print("      is exchangeable with the right one under H0)")
