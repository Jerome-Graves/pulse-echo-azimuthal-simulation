"""Step 7: sensitivity of the PO anchor to every input I had to choose,
so the uncertainty band is measured rather than asserted.
Each variant is the full brute-force calculation over 60 azimuths.
"""
import os

import numpy as np

import po_src_03_kirchhoff_pred as PO
from po_src_03_kirchhoff_pred import DT, F0, LAM, NT, ricker_spec

HERE = os.path.dirname(os.path.abspath(__file__))
Wspec, _ = ricker_spec(NT, DT, F0)
AZ = np.arange(0, 360, 6.0)
BASE_C, BASE_G, BASE_S, BASE_A, BASE_R = PO.C, PO.GATE, PO.S_MAX, PO.A_EL, PO.R_DISC
BASE_ST = PO.S_T


def _el(f):
    PO.A_EL = BASE_A * f
    PO.S_T = BASE_ST * f ** 2


def level(tag):
    lev = np.array([PO.env_rms_gate(PO.trace_from(
        PO.azimuth_response(a, LAM / 8)[0], Wspec)) for a in AZ])
    return 10 * np.log10((lev ** 2).mean())


ref = level("base")
print(f"reference (60 azimuths, ds = lam/8): {ref:.2f} dB re source\n")
print(f"{'variant':44}{'dB':>9}{'delta':>9}")
rows = []
for tag, setter in [
    ("delay->range speed 3735 m/s (-3%)", lambda: setattr(PO, "C", 3735.0)),
    ("delay->range speed 3965 m/s (+3%)", lambda: setattr(PO, "C", 3965.0)),
    ("gate 24.5-35.5 us", lambda: setattr(PO, "GATE", (24.5e-6, 35.5e-6))),
    ("gate 23.5-36.5 us", lambda: setattr(PO, "GATE", (23.5e-6, 36.5e-6))),
    ("beam cone kept to sin(phi) < 0.30", lambda: setattr(PO, "S_MAX", 0.30)),
    ("beam cone kept to sin(phi) < 0.55", lambda: setattr(PO, "S_MAX", 0.55)),
    ("element radius +3 % (area with it)", lambda: _el(1.03)),
    ("element radius -3 % (area with it)", lambda: _el(0.97)),
    ("probe recessed h/2 into the rim",
     lambda: setattr(PO, "R_DISC", BASE_R - PO.H / 2)),
]:
    setter()
    v = level(tag)
    # S_t is tied to A_EL in the transfer; keep them consistent
    print(f"{tag:44}{v:>9.2f}{v-ref:>+9.2f}")
    rows.append((tag, v - ref))
    PO.C, PO.GATE, PO.S_MAX, PO.A_EL, PO.R_DISC = (BASE_C, BASE_G, BASE_S,
                                                   BASE_A, BASE_R)
    PO.S_T = BASE_ST

print("\nanalytic levers (no re-run needed)")
print(f"{'|R| grain boundary +/- 10 %':44}{'':>9}{'+/-0.83':>9}")
print(f"{'S_v facet area density +/- 3 %':44}{'':>9}{'+/-0.13':>9}")
print(f"{'S_t element area (enters squared) +/- 3 %':44}{'':>9}{'+/-0.26':>9}")
print(f"{'source calibration from the zero-contrast rim':44}{'':>9}{'-0.82':>9}")

D = np.load(os.path.join(HERE, "po_src_diag.npz"))
G = np.load(os.path.join(HERE, "po_src_geom.npz"))
print("\nheadline numbers for the write-up")
print(f"  boundaries in gate & -6 dB two-way beam : {D['n6'].mean():.0f} "
      f"per azimuth")
print(f"  intercepted facet area, that subset      : mean "
      f"{D['a6'][D['a6']>0].mean()*1e6:.1f} mm^2 "
      f"(equiv diam {np.sqrt(4*D['a6'][D['a6']>0].mean()/np.pi)*1e3:.1f} mm)")
print(f"  total facet area in gate & beam          : "
      f"{D['a6'].sum()/30*1e6:.0f} mm^2 per azimuth")
print(f"  |R|_rms area weighted                    : "
      f"{np.sqrt((D['rr']**2*D['ain']).sum()/D['ain'].sum())*100:.3f} %")
print(f"  |R| range over insonified boundaries     : "
      f"{np.percentile(D['rr'],[5,50,95])*100}")
print(f"  monostatic Fresnel area lam d/2 at gate centre = "
      f"{LAM*0.0575/2*1e6:.1f} mm^2 (diam "
      f"{np.sqrt(4*LAM*0.0575/2/np.pi)*1e3:.1f} mm)")
print(f"  mean facet area of the whole tessellation = "
      f"{G['farea'].mean()*1e6:.1f} mm^2")
