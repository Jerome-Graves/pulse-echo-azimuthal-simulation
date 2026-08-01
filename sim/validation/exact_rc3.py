"""Part 3: what the residual scatter actually is.

validate.py measures  Eb/E1  where
  Eb = envelope peak of the grain-boundary echo   (path: 2 x 40 mm in grain A)
  E1 = envelope peak of the BACKWALL echo         (path: 2 x 40 mm in A
                                                    PLUS 2 x 60 mm in grain B)
so the normaliser carries 120 mm of propagation through grain B, whose
anisotropic beam spreading depends on psi_b. That factor is absent from
both R_iso and R_exact.
"""
import os
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\Jerome\Documents\GitHub\openUSCT\simulation")
from ringfwi import anisotropy as an  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # siblings
from exact_rc import setup, exact_normal_RT, RHO, banner  # noqa: E402


def vqp(psi):
    return float(an.ice_qp_vs_caxis(np.radians(psi)))


# fine phase-speed curve and its derivatives w.r.t. the phase angle psi (rad)
PS = np.linspace(-0.5, np.pi / 2 + 0.5, 400001)
VS = np.array([an.ice_qp_vs_caxis(p) for p in PS])
DP = PS[1] - PS[0]
DV = np.gradient(VS, DP)
THG = PS + np.arctan(DV / VS)                 # group angle from the c-axis
FIN = np.gradient(THG, DP)                    # in-plane ray-tube magnification


def spread(psi_deg):
    """(F_in, F_out, skew_deg) for a qP beam whose PHASE direction is at
    psi to the c-axis of a TI ice crystal.

    F_in  = d(theta_group)/d(theta_phase)   : in-plane angular magnification
    F_out = sin(theta_group)/sin(theta_phase): out-of-plane (azimuthal) one
    Far-field beam amplitude scales as 1/sqrt(F_in * F_out).
    """
    p = np.radians(psi_deg)
    fin = float(np.interp(p, PS, FIN))
    tg = float(np.interp(p, PS, THG))
    # at psi -> 0 the azimuthal magnification degenerates to the in-plane one
    fout = fin if abs(np.sin(p)) < 1e-6 else np.sin(tg) / np.sin(p)
    return fin, fout, np.degrees(tg - p)


banner("6.  ANISOTROPIC BEAM SPREADING IN GRAIN B (what E1 carries)")
print(f"{'psi':>6}{'v_qP':>10}{'skew(deg)':>11}{'F_in':>9}{'F_out':>9}"
      f"{'sqrt(Fin*Fout)':>16}")
for psi in [0, 10, 20, 27, 35, 45, 51, 60, 70, 80, 90]:
    fi, fo, sk = spread(psi)
    print(f"{psi:>6}{vqp(psi):>10.1f}{sk:>11.3f}{fi:>9.4f}{fo:>9.4f}"
          f"{np.sqrt(fi*fo):>16.4f}")

banner("7.  DOES IT EXPLAIN THE FIVE MEASURED POINTS?")
val = np.load(r"C:\Users\Jerome\Documents\pulse-echo-analysis-scratch"
              r"\bicrystal_val.npz")['res']
rows = []
for pa, pb, riso, amp, va, vb in val[1:]:
    r = exact_normal_RT(*setup(pa, pb)[:2], RHO)
    fi, fo, sk = spread(pb)
    rows.append(dict(pa=pa, pb=pb, riso=abs(riso), rex=abs(r['R_by']['qP']),
                     amp=amp, S=np.sqrt(fi * fo), fi=fi, fo=fo))

print(f"{'psi_a/psi_b':>12}{'|R_iso|':>11}{'|R_exact|':>11}{'sqrtF':>9}"
      f"{'amp/Riso':>10}{'amp/Rex':>9}{'amp/(Rex*sqrtF)':>17}")
for d in rows:
    print(f"{int(d['pa'])}/{int(d['pb']):<9}{d['riso']:>11.6f}"
          f"{d['rex']:>11.6f}{d['S']:>9.4f}{d['amp']/d['riso']:>10.3f}"
          f"{d['amp']/d['rex']:>9.3f}"
          f"{d['amp']/(d['rex']*d['S']):>17.3f}")


def stats(name, k):
    k = np.asarray(k)
    print(f"  {name:<34} mean {k.mean():7.3f}  rel.sd "
          f"{100*k.std(ddof=1)/k.mean():5.1f}%  spread "
          f"{20*np.log10(k.max()/k.min()):5.2f} dB")


print()
stats("amp / |R_iso|  (paper's model)", [d['amp'] / d['riso'] for d in rows])
stats("amp / |R_exact|", [d['amp'] / d['rex'] for d in rows])
stats("amp / (|R_iso| * sqrtF)", [d['amp'] / (d['riso'] * d['S']) for d in rows])
stats("amp / (|R_exact| * sqrtF)", [d['amp'] / (d['rex'] * d['S']) for d in rows])

am = np.array([d['amp'] for d in rows])
for nm, pr in (("|R_iso|", [d['riso'] for d in rows]),
               ("|R_exact|", [d['rex'] for d in rows]),
               ("|R_exact|*sqrtF", [d['rex'] * d['S'] for d in rows]),
               ("|R_iso|*sqrtF", [d['riso'] * d['S'] for d in rows])):
    print(f"  r(model, measured) with {nm:<18} = "
          f"{np.corrcoef(np.array(pr), am)[0,1]:.5f}")

banner("8.  THE psi_a-INDEPENDENCE TEST (the decisive one)")
print("0/51 and 90/51 have completely different grain A but the same grain B.")
print("If the residual were an interface-physics error it would differ; if it")
print("is a grain-B normalisation effect it must be the same.")
a, b = rows[2], rows[4]
print(f"   0/51 : amp/|R_iso| = {a['amp']/a['riso']:.3f}")
print(f"  90/51 : amp/|R_iso| = {b['amp']/b['riso']:.3f}")
print(f"  difference {100*abs(a['amp']/a['riso'] - b['amp']/b['riso'])/(a['amp']/a['riso']):.2f} %"
      "   while the full ladder scatters by 86 %")
