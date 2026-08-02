"""Is the prediction actually REGISTERED in time?

r as a function of an imposed time lag on the prediction.  If the model only
predicted 'how much' and not 'when', the curve would be flat or peak
somewhere arbitrary.  A peak at zero lag, one pulse length wide, is the
signature of genuine depth registration.
"""
import os as _os
import sys as _sys
# helper modules shared with sibling directories
for _d in ('..',):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import numpy as np

import _t2_common as C
import shift_null_2d as S

CASES = [("girdle_perp_ppw8", "gp8", "CORRECT"),
         ("singlemax_ppw8", "sm8", "CORRECT"),
         ("girdle_perp_ppw8", "sm8", "same tess WRONG fabric"),
         ("singlemax_ppw8", "gp8", "same tess WRONG fabric"),
         ("girdle_perp_ppw8", "wrongseed8", "WRONG tessellation"),
         ("singlemax_ppw8", "wrongseed8", "WRONG tessellation")]
MODE = "harm"
LAGS = np.arange(-40, 41, 2)            # x 20 ns  -> +-0.8 us

print("r vs imposed time lag on the prediction (harm-centred, variant=geom)")
print("lag in us; 20 ns grid; band-limited pulse length ~0.45 us")
print()
hdr = "  ".join(f"{l*C.DT_C*1e6:+.2f}" for l in LAGS[::4])
print(f"{'case':<40}{hdr}")
for sw, tg, lbl in CASES:
    az, tt, M, P = S.load(sw, tg)
    m = S.centre(10 * np.log10(M + S.EPS), MODE)
    curve = []
    for L in LAGS:
        p = S.centre(10 * np.log10(np.roll(P["geom"], L, axis=1) + S.EPS),
                     MODE)
        curve.append(S.corr(m, p))
    curve = np.array(curve)
    row = "  ".join(f"{c:+.3f}" for c in curve[::4])
    kpk = int(np.argmax(curve))
    print(f"{sw[:16]:<17}{tg:<11}{lbl[:11]:<12}{row}")
    print(f"{'':<40}peak r={curve[kpk]:+.3f} at "
          f"{LAGS[kpk]*C.DT_C*1e6:+.2f} us")

print()
print("=" * 78)
print("Speckle ceiling.  A fully developed speckle envelope has")
print("var(10 log10 |E|^2) = (10/ln10)^2 * pi^2/6 = 31.0 dB^2 per sample,")
print("irreducible by ANY model of the local mean power.")
print("=" * 78)
for sw, tg, lbl in CASES[:2]:
    az, tt, M, P = S.load(sw, tg)
    md = 10 * np.log10(M + S.EPS)
    for mode in ("col", "double", "harm"):
        v = float(np.var(S.centre(md, mode)))
        ceil = np.sqrt(max(v - 31.02, 0) / v)
        r = S.score(M, P["geom"], mode)
        print(f"{sw:<18}{mode:<8}var={v:6.1f} dB^2  ceiling r<={ceil:.3f}"
              f"   observed r={r:.3f}   -> {100*(r/max(ceil,1e-9))**2:5.1f}% "
              f"of the predictable variance")
