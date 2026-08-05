r"""Figure `regime`: where this specimen sits on the scattering-regime map.

Supports the claim of Section~\ref{sec:theory} that the analytical
scattering theories used to validate every comparable simulation study
are derived in the Rayleigh and stochastic regimes, while this specimen
is geometric, at $ka \approx 28$ and $D/\lambda \approx 9$. That is why
the validation strategy of this paper differs from its predecessors, and
why no prior convergence criterion can be assumed to transfer.

What is drawn, and where each part comes from:

  the curve      Born single scattering with an exponential two-point
                 correlation, integrated in closed form here, capped at a
                 geometric plateau. Its asymptotic slopes, 4 then 2 then
                 0, are the physics; the saturation LEVEL is not
                 predicted by the Born integral and is anchored at the
                 stochastic-to-geometric boundary, so the curve is a
                 guide and the caption says so.
  the boundaries D/lambda = 0.1 and 1, that is ka = pi/10 and pi.
  this work      ka computed from the tessellation and the acquisition,
                 by the same code that draws Fig. `scales`.
  Bai 2018       the one prior study of Table~\ref{tab:regime} for which
                 a grain-size-to-wavelength ratio is quoted, 0.1 to 0.3.
  the bracket    the region occupied by the prior convergence studies.
                 Drawn as a region and not as points because three of the
                 four rows of Table~\ref{tab:regime} carry no D/lambda:
                 see PRIOR_STUDIES below to add them once they are.

Reads
    (through fig_scales) data/po_src_geom.npz and
    ../../out/sweeps/girdle_seed11_ppw8_dev/config.json
Writes
    <paper>/figures/regime.pdf

No GPU, no simulation call, no measured data: this is a map of where the
experiment sits, not a result.
"""
import numpy as np

import figstyle as S
from fig_scales import length_scales, load_geometry

# Regime boundaries as grain-diameter-to-wavelength ratios. ka = pi D /
# lambda, since k = 2 pi / lambda and a = D / 2, so these convert
# straight to ka. The conventional partition is lambda much larger than
# the grain (Rayleigh), comparable (stochastic) and much smaller
# (geometric); 0.1 and 1 are the usual round numbers, and they put
# Bai 2018 in the Rayleigh-to-stochastic region exactly as that study
# describes itself.
BOUNDARY_D_OVER_LAM = (0.1, 1.0)
REGIME_NAMES = ("Rayleigh", "stochastic", "geometric")

# Prior numerical studies, as (label, D/lambda low, D/lambda high), each
# on its own source's grain measure and with the same provenance as the
# cells of Table~\ref{tab:regime}. Huang 2020 reaches D/lambda about 2,
# which is ABOVE the geometric boundary this figure draws at 1, so the
# bracket below is taken from the studies themselves rather than from
# the boundary: the claim the caption makes is that the present specimen
# lies far beyond the prior work, not that the prior work stops at the
# regime line.
PRIOR_STUDIES = (("Van Pamel 2015", 0.03, 0.25),
                 ("Bai 2018", 0.1, 0.3),
                 ("Huang 2020", 0.08, 1.98))

XLIM_KA = (0.05, 100.0)
YLIM = (1e-6, 6.0)


# ────────────────────────────── computation ──────────────────────────────
def born_strength(ka):
    """Single-scattering strength against ka, up to a constant.

    For a medium with a normalised two-point correlation exp(-r/a) the
    spectral density is 8 pi a^3 / (1 + q^2 a^2)^2 with q = 2 k sin(t/2),
    and the scattering cross-section per unit volume is k^4 times its
    integral over solid angle. Substituting x = sin^2(t/2),

        integral = 32 pi^2 a^3 / (1 + 4 k^2 a^2),

    so the attenuation per grain radius goes as (ka)^4 / (1 + 4 (ka)^2):
    the fourth power of frequency while ka << 1/2, the second power
    beyond it. Both are the textbook Rayleigh and stochastic laws, and
    neither saturates, which is the point made in geometric_strength.
    """
    ka = np.asarray(ka, float)
    return ka ** 4 / (1.0 + 4.0 * ka ** 2)


def geometric_strength(ka, ka_saturate):
    """Born strength capped at the geometric limit, normalised to it.

    Single scattering has no geometric regime: its cross-section grows
    without bound, whereas a real grain cannot scatter more than its own
    cross-section intercepts, so the attenuation per grain radius becomes
    constant once the wavelength is small compared with the grain. The
    two limits are combined so that the smaller governs, which is the
    standard way of drawing this map. The plateau is set at the
    stochastic-to-geometric boundary; its height is a normalisation and
    carries no information.
    """
    ratio = born_strength(ka) / born_strength(ka_saturate)
    return ratio / (1.0 + ratio)


def ka_of(d_over_lam):
    """ka from the grain-diameter-to-wavelength ratio: ka = pi D/lambda."""
    return np.pi * np.asarray(d_over_lam, float)


# ─────────────────────────────── drawing ─────────────────────────────────
def draw_regimes(ax, ka_bounds):
    """Shade and name the three regions."""
    # Only the geometric region is shaded, because the figure exists to
    # show that this work is alone in it.
    ax.axvspan(ka_bounds[1], XLIM_KA[1], color=S.GREY_LIGHT, alpha=0.30,
               lw=0, zorder=0)
    for kb in ka_bounds:
        ax.axvline(kb, color=S.GREY, lw=S.LW["annotation"],
                   ls=(0, (1.0, 1.4)), zorder=1)
    edges = (XLIM_KA[0],) + tuple(ka_bounds) + (XLIM_KA[1],)
    for name, lo, hi in zip(REGIME_NAMES, edges[:-1], edges[1:]):
        S.direct_label(ax, np.sqrt(lo * hi), 3.2, name, colour=S.GREY,
                       ha="center", va="center")


def draw_curve(ax, ka, ka_saturate):
    """The scattering strength, with its two power-law asymptotes named."""
    ax.plot(ka, geometric_strength(ka, ka_saturate), color="black",
            lw=S.LW["emphasis"], zorder=4)
    # Asymptotes drawn a decade above the curve so that both the line and
    # the curve stay readable, the convention in this literature.
    S.reference_slope(ax, 0.075, 0.30, 12.0 * geometric_strength(
        0.075, ka_saturate), 4.0, "$(ka)^{4}$")
    S.reference_slope(ax, 0.80, 2.50, 1.8 * geometric_strength(
        0.80, ka_saturate), 2.0, "$(ka)^{2}$")
    S.direct_label(ax, 55.0, 0.45, "saturated", colour=S.GREY, ha="center",
                   va="top")


def draw_studies(ax, ka_saturate, ka_this, d_over_lam_this):
    """Prior work as a region and a point, this work as a marked point."""
    prior = S.series(1, reference=True)
    prior.update(linestyle="none", markersize=3.4)
    for name, lo, hi in PRIOR_STUDIES:
        span = ka_of([lo, hi])
        mid = np.sqrt(span[0] * span[1])
        ax.plot(span, geometric_strength(span, ka_saturate),
                color=S.BLUE, lw=2.2, solid_capstyle="butt", zorder=5)
        ax.plot([mid], geometric_strength([mid], ka_saturate), zorder=6,
                **prior)
        S.direct_label(ax, mid * 1.5, geometric_strength(mid, ka_saturate)
                       * 0.35, name, colour=S.BLUE, va="top")

    # The bracket spans the studies actually tabulated, not the regime
    # boundary. Huang 2020 crosses that boundary, so anchoring the right
    # end at D/lambda = 1 would draw a figure its own caption contradicts.
    d_hi = max(hi for _, _, hi in PRIOR_STUDIES)
    y_bracket = 4e-6
    ax.plot(ka_of([XLIM_KA[0] / np.pi, d_hi]),
            [y_bracket] * 2, color=S.GREY, lw=S.LW["reference"],
            solid_capstyle="butt", zorder=3)
    ax.plot([ka_of(d_hi)] * 2, [y_bracket * 0.55,
            y_bracket * 1.8], color=S.GREY, lw=S.LW["reference"], zorder=3)
    S.direct_label(ax, np.sqrt(XLIM_KA[0] * ka_of(d_hi)),
                   y_bracket * 2.4, "prior convergence studies",
                   colour=S.GREY, ha="center")

    this = S.series(0)
    this.update(linestyle="none", markersize=4.2, zorder=7)
    ax.plot([ka_this], geometric_strength([ka_this], ka_saturate), **this)
    S.arrow_label(ax, "this work\n$ka$ = %.0f, $D/\\lambda$ = %.1f"
                  % (ka_this, d_over_lam_this),
                  xy=(ka_this, geometric_strength(ka_this, ka_saturate)
                      * 0.75),
                  xytext=(ka_this * 0.75, 6e-3), ha="center", va="top",
                  colour="black")


# ──────────────────────────────── figure ─────────────────────────────────
def main():
    sc = length_scales(load_geometry())
    ka_bounds = ka_of(BOUNDARY_D_OVER_LAM)
    ka_saturate = ka_bounds[1]

    print("regime boundaries at D/lambda %s, that is ka %.2f and %.2f"
          % (list(BOUNDARY_D_OVER_LAM), ka_bounds[0], ka_bounds[1]))
    print("this work: ka %.1f, D/lambda %.2f, so %.0f times past the "
          "geometric boundary" % (sc["ka"], sc["d_over_lam"],
                                  sc["ka"] / ka_bounds[1]))
    for name, lo, hi in PRIOR_STUDIES:
        print("%-10s D/lambda %.2f to %.2f, ka %.2f to %.2f"
              % (name, lo, hi, ka_of(lo), ka_of(hi)))

    fig, ax = S.figure("regime", left=12.0, right=2.5, bottom=8.5,
                       top=9.5)
    ka = np.logspace(np.log10(XLIM_KA[0]), np.log10(XLIM_KA[1]), 600)
    draw_regimes(ax, ka_bounds)
    draw_curve(ax, ka, ka_saturate)
    draw_studies(ax, ka_saturate, sc["ka"], sc["d_over_lam"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM_KA)
    ax.set_ylim(*YLIM)
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
    S.log_minor_off(ax)
    ax.tick_params(top=False)
    S.plain_log_ticks(ax, "x", [0.1, 1.0, 10.0, 100.0])
    ax.set_xlabel("$ka$")
    ax.set_ylabel("normalised scattering strength")

    top = ax.secondary_xaxis("top", functions=(lambda k: k / np.pi,
                                               lambda d: d * np.pi))
    top.set_xlabel("grain diameter / wavelength", labelpad=4.0)
    top.tick_params(labelsize=S.FS["tick"], width=S.LW["spine"], size=2.6,
                    direction="in", pad=2.0)
    S.plain_log_ticks(top, "x", [0.1, 1.0, 10.0])

    S.save(fig, "regime", expect=("normalised scattering strength",
                                  "grain diameter / wavelength",
                                  "Rayleigh", "stochastic", "geometric",
                                  "this work", "prior convergence studies"))


if __name__ == "__main__":
    main()
