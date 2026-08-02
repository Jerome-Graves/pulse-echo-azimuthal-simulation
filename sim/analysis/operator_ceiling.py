"""Which limit does the observed spectral pileup sit at?

Supports the dispersion subsection of Section 4. A pileup of spectral
content is observed near kh = 2.55, and two candidate explanations sit
almost on top of one another:

  the OPERATOR CEILING. A staggered finite-difference first derivative
  represents a wavenumber k by an effective wavenumber k_eff whose value
  is bounded. The bound is reached at kh = pi and equals 2*sum|c_n|. No
  input, however sharp, can produce an effective wavenumber above it, so
  content driven past the limit accumulates there.

  MITTET'S HALF-SPECTRAL RULE. If the medium is represented to Nyquist,
  the usable band is limited to half the spectrum, kh = pi/2 = 1.571,
  because the material and the field then alias into one another.

The two are distinguished by an experiment that needs no simulation at
all: the operator ceiling DEPENDS ON ORDER and pi/2 does not. If the
pileup is the ceiling it moves when the operator changes; if it is the
half-spectral rule it stays.

Reads:  nothing. Everything follows from the difference coefficients.
Writes: ../results/operator_ceiling.log
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dispersion_coeffs import keff_h, taylor_sg          # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, '..', 'results'))
ORDERS = (4, 6, 8, 10, 12, 16)
OBSERVED_PILEUP_KH = 2.55      # Section 4
HALF_SPECTRAL_KH = np.pi / 2   # Mittet, order independent


def ceiling(order):
    """Largest effective wavenumber the operator can represent, and where.

    Evaluated on a dense grid rather than assumed to lie at kh = pi,
    because for a strongly optimised operator the maximum need not sit at
    the Nyquist edge, and the claim being tested is precisely about where
    the maximum is.
    """
    c = taylor_sg(order // 2)
    kh = np.linspace(1e-6, np.pi, 200001)
    ke = keff_h(c, kh)
    i = int(np.argmax(ke))
    return float(ke[i]), float(kh[i]), 2.0 * float(np.abs(c).sum())


def main():
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def note(msg):
        print(msg, flush=True)
        lines.append(msg)

    note('=' * 68)
    note('Operator ceiling against order. Observed pileup at kh = %.2f;'
         % OBSERVED_PILEUP_KH)
    note('the half-spectral rule sits at pi/2 = %.4f for every order.'
         % HALF_SPECTRAL_KH)
    note('')
    note('%-7s %12s %10s %12s %12s' % ('order', 'max k_eff h', 'at kh',
                                       '2 sum|c_n|', 'vs observed'))
    got = []
    for order in ORDERS:
        ke_max, at_kh, sum_rule = ceiling(order)
        got.append((order, ke_max))
        note('%-7d %12.4f %10.4f %12.4f %+12.4f'
             % (order, ke_max, at_kh, sum_rule, ke_max - OBSERVED_PILEUP_KH))

    note('')
    spread = max(k for _o, k in got) - min(k for _o, k in got)
    note('The ceiling moves by %.3f in kh across orders 4 to 16, while the'
         % spread)
    note('half-spectral rule does not move at all. The observed pileup is')
    note('%.3f above pi/2, which is %.0f per cent of the way from pi/2 to'
         % (OBSERVED_PILEUP_KH - HALF_SPECTRAL_KH,
            100 * (OBSERVED_PILEUP_KH - HALF_SPECTRAL_KH)
            / (got[2][1] - HALF_SPECTRAL_KH)))
    note('the eighth-order ceiling, so the two hypotheses are far apart in')
    note('kh even though the numbers 2.573 and 1.571 look similar on a page.')
    note('')
    note('CONCLUSION. The pileup at kh = %.2f lies within %.3f of the'
         % (OBSERVED_PILEUP_KH, abs(got[2][1] - OBSERVED_PILEUP_KH)))
    note('eighth-order ceiling and %.3f above the half-spectral limit. It is'
         % (OBSERVED_PILEUP_KH - HALF_SPECTRAL_KH))
    note('the operator ceiling. A simulation at a different order would move')
    note('the pileup by the amounts tabulated above and is the confirming')
    note('experiment, but the arithmetic already separates the hypotheses:')
    note('pi/2 is not close to 2.55 by any measure.')

    with open(os.path.join(RESULTS, 'operator_ceiling.log'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
