"""TEST 3, part 1: the measured 2 -> 5 MHz coda LEVEL ratio."""
import numpy as np

import freq_common as T


def boot_ci(x, nb=4000, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(x)
    m = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return np.percentile(m, [2.5, 97.5])


AZ5 = T.az_list("singlemax_seed7_ppw6_5mhz_production")
print(f"singlemax_seed7_ppw6_5mhz_production azimuths (n={len(AZ5)}): {AZ5[0]}..{AZ5[-1]} deg, "
      f"steps {sorted(set(np.diff(AZ5)))}")
print("  -> HALF CIRCLE ONLY, irregular spacing (5 gaps of 8 deg)\n")

print("=== A. legacy vs rigid2 convention, same specimen (seed 11, 2 MHz) ===")
for nm in ("singlemax_seed11_ppw6_rigid2", "singlemax_seed11_ppw6_inversion_cal"):
    m = T.measure(nm)
    print(f"  {nm:<16} n={len(m['rot']):3d}  mean coda {m['coda_db'].mean():7.2f} dB"
          f"  sd {m['coda_db'].std(ddof=1):5.2f}  E1 mean {m['e1'].mean():.4g}")
r1 = T.measure("singlemax_seed11_ppw6_rigid2")
v1 = T.measure("singlemax_seed11_ppw6_inversion_cal")
d = v1["coda_db"].mean() - r1["coda_db"].mean()
se = np.hypot(v1["coda_db"].std(ddof=1) / np.sqrt(len(v1["rot"])),
              r1["coda_db"].std(ddof=1) / np.sqrt(len(r1["rot"])))
print(f"  legacy - rigid2 = {d:+.2f} dB  (naive SE {se:.2f}, "
      f"but azimuths are autocorrelated so SE is optimistic)\n")

print("=== B. 2 MHz vs 5 MHz, seed 7, ppw6, double raster ===")
res = {}
for nm, lab in (("singlemax_seed7_ppw6_fittest_legacy", "2 MHz"), ("singlemax_seed7_ppw6_5mhz_production", "5 MHz")):
    azs = AZ5 if nm == "singlemax_seed7_ppw6_fittest_legacy" else None      # matched azimuth set
    for fe1 in (False, True):
        m = T.measure(nm, azs=azs, filt_e1=fe1)
        tag = "bp-E1" if fe1 else "raw-E1"
        res[(nm, fe1)] = m
        print(f"  {nm:<10}{lab}  {tag}: n={len(m['rot'])}  "
              f"coda/E1 {m['coda_db'].mean():7.2f} dB "
              f"(sd {m['coda_db'].std(ddof=1):.2f})  "
              f"E1mean {m['e1'].mean():.5g}")

for fe1 in (False, True):
    a = res[("singlemax_seed7_ppw6_fittest_legacy", fe1)]["coda_db"]
    b = res[("singlemax_seed7_ppw6_5mhz_production", fe1)]["coda_db"]
    dd = b.mean() - a.mean()
    ca, cb = boot_ci(a), boot_ci(b)
    tag = "bp-E1" if fe1 else "raw-E1"
    print(f"\n  [{tag}] 5 MHz - 2 MHz = {dd:+.2f} dB"
          f"   (bootstrap CI on each mean: 2MHz [{ca[0]:.2f},{ca[1]:.2f}], "
          f"5MHz [{cb[0]:.2f},{cb[1]:.2f}])")
    n_eff = 12.0     # ~40 az over 176 deg, integral length ~10 deg at 2 MHz
    se = np.hypot(a.std(ddof=1), b.std(ddof=1)) / np.sqrt(n_eff)
    print(f"        SE with n_eff~{n_eff:.0f} independent azimuths: "
          f"+/-{1.96*se:.2f} dB")
    print(f"        implied exponent  coda ~ f^q :  q = {dd/(20*np.log10(2.5)):+.3f}")

print("\n=== C. raw backwall (E1) level ratio, and coda without E1 division ===")
for fe1 in (False, True):
    a = res[("singlemax_seed7_ppw6_fittest_legacy", fe1)]
    b = res[("singlemax_seed7_ppw6_5mhz_production", fe1)]
    tag = "bp-E1" if fe1 else "raw-E1"
    print(f"  [{tag}] E1: 20log10(5MHz/2MHz) = "
          f"{20*np.log10(b['e1'].mean()/a['e1'].mean()):+.2f} dB")
print("  (E1 ratio also contains the source-injection normalisation, which "
      "differs\n   between grids: weights 1/N over N element points, N ~ h^-2. "
      "Only the\n   coda/E1 RATIO is grid-normalisation free.)")

np.savez(r"freq_level_ratio.npz",
         az5=np.array(AZ5),
         coda2=res[("singlemax_seed7_ppw6_fittest_legacy", False)]["coda_db"],
         coda5=res[("singlemax_seed7_ppw6_5mhz_production", False)]["coda_db"],
         lin2=res[("singlemax_seed7_ppw6_fittest_legacy", False)]["coda_lin"],
         lin5=res[("singlemax_seed7_ppw6_5mhz_production", False)]["coda_lin"])
