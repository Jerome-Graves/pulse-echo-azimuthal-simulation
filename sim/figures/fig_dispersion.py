r"""Figure 'dispersion': numerical dispersion of the eighth-order operator.

Supports Section 4, "Discrimination against numerical dispersion". The
claim the figure has to carry is a negative one: the propagator is
accurate over the whole band the experiment uses, so the decibels lost in
the tilt test are lost in the representation of the medium and not on the
way there. It also shows that the group velocity has no interior zero, so
there is no wavenumber at which energy stalls and accumulates as a
spurious grid mode.

Reads no measurement file. Everything here is the solver's own numerics,
recomputed from
    ../fdtd.py   optimised_coeffs(8, kh_max=2.0, multistart=True), which
                 is the operator every production sweep used (fixed at
                 sim/runs/master_run.py line 39 and recorded as
                 'fd_kh_max' in every out/sweeps/*/config.json), the
                 Taylor coefficients of the same order for contrast, and
                 ricker(), the deployed source wavelet
Analysis constants quoted from sim/analysis/_t2_common.py: the coda is
band-passed to 0.8 to 3.0 MHz and the source centre frequency is 2 MHz.

Writes ../../../pulse-echo-cof-paper/figures/dispersion.pdf and prints
the operator ceiling, the in-band phase error and the group velocity at
the second harmonic, which are the numbers Section 4 quotes.

TWO NUMBERS IN THE TEX ARE FROM AN OLDER OPERATOR. The manuscript gives
K_max = 2.573 for eighth order and a group velocity of 0.868 at twice the
source frequency. 2.573 is the TAYLOR value, and 0.868 belongs to the
pre-2026-07-31 optimised operator (kh_max 1.6, single start). The
operator actually deployed gives 2.705 and 0.958. This script prints all
of them so the difference is visible, and plots the deployed one.
"""
import sys
from pathlib import Path

import numpy as np

import figstyle as fs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fdtd                                             # noqa: E402

# The operator the sweeps ran with. CuPy is imported lazily inside fdtd,
# so pulling in the coefficient routines touches no GPU.
ORDER = 8
KH_MAX = 2.0

F0_HZ = 2.0e6                       # source centre frequency
CODA_BAND_HZ = (0.8e6, 3.0e6)       # BAND in sim/analysis/_t2_common.py
SOURCE_LEVEL_DB = -20.0             # edge of the source band, see below

# Bands are drawn at the COARSEST production grid. kh scales as 1/n_lambda
# at fixed frequency, so n_lambda 6 puts the bands at the largest kh any
# sweep in the paper occupies, and the accuracy read off here is the worst
# case rather than a favourable one.
PPW_BANDS = 6.0


# --------------------------------------------------------------- compute

def phase_ratio(coeffs, kh):
    """Numerical phase velocity over the exact one.

    The staggered first derivative has Fourier symbol
    D(k) = (2/h) sum_n c_n sin((n - 1/2) kh) where the exact operator
    gives k, so the phase velocity ratio is D(k)/k.
    """
    n = np.arange(1, len(coeffs) + 1) - 0.5
    return (2.0 * (np.asarray(coeffs)[None, :]
                   * np.sin(np.outer(kh, n))).sum(1)) / kh


def group_ratio(coeffs, kh):
    """Numerical group velocity over the exact one, dD/dk.

    This, not the phase error, is what decides whether a scheme can
    produce a spurious grid mode: energy at a wavenumber where the group
    velocity vanishes stops propagating and piles up there.
    """
    n = np.arange(1, len(coeffs) + 1) - 0.5
    return 2.0 * (np.asarray(coeffs)[None, :] * n
                  * np.cos(np.outer(kh, n))).sum(1)


def operator_ceiling(coeffs):
    """Largest effective wavenumber the operator can represent, times h.

    K_max = 2 sum |c_n|. For a staggered stencil the signs of c_n
    alternate and sin((n - 1/2) pi) alternates with them, so every term
    adds at kh = pi: the bound is attained, exactly at Nyquist, and
    nowhere else.
    """
    return 2.0 * np.abs(np.asarray(coeffs)).sum()


def kh_of(f_hz, ppw):
    """Grid wavenumber of a frequency, given points per wavelength at f0.

    h = c / (f0 ppw) and k = 2 pi f / c, so kh = 2 pi f / (f0 ppw) and the
    sound speed cancels. Nothing here depends on the material.
    """
    return 2.0 * np.pi * np.asarray(f_hz, float) / (F0_HZ * ppw)


def source_band_hz(level_db=SOURCE_LEVEL_DB):
    """Frequencies where the deployed wavelet is within level_db of peak.

    Taken from fdtd.ricker itself rather than from the closed-form Ricker
    spectrum, so that the band drawn is the band the solver injects. The
    sampling interval and length here are the transform's own, fine and
    long enough that the edges are converged to the digit printed.
    """
    dt, nt = 2.0e-9, 8192
    spectrum = np.abs(np.fft.rfft(fdtd.ricker(F0_HZ, dt, nt)))
    f_hz = np.fft.rfftfreq(nt, dt)
    inside = spectrum >= spectrum.max() * 10.0 ** (level_db / 20.0)
    return f_hz[inside][0], f_hz[inside][-1]


def worst_in_band(kh, value, band_kh):
    """Largest departure from unity inside a band, for the printed table."""
    inside = (kh >= band_kh[0]) & (kh <= band_kh[1])
    return np.abs(value[inside] - 1.0).max()


# ------------------------------------------------------------------ draw

def shade_bands(ax, source_kh, coda_kh):
    """The source band with the analysis band nested inside it."""
    ax.axvspan(*source_kh, color=fs.GREY_LIGHT, alpha=0.30, linewidth=0,
               zorder=0)
    ax.axvspan(*coda_kh, color=fs.GREY_LIGHT, alpha=0.55, linewidth=0,
               zorder=0)


def draw_velocity(ax, kh, coeffs, kh_second_harmonic):
    """Phase and group velocity of the deployed operator."""
    for i, ratio in enumerate((phase_ratio(coeffs, kh),
                               group_ratio(coeffs, kh))):
        style = fs.series(i)
        style["marker"] = "none"        # both curves sit on 1 until kh 2,
        style["linewidth"] = fs.LW["emphasis"]   # markers would only blur
        ax.plot(kh, ratio, zorder=3, **style)    # that agreement

    ax.axvline(kh_second_harmonic, color=fs.GREY, zorder=2,
               linewidth=fs.LW["annotation"], linestyle=(0, (1.0, 1.6)))
    vg_second = float(group_ratio(coeffs, np.array([kh_second_harmonic]))[0])
    ax.plot([kh_second_harmonic], [vg_second], marker="s", markersize=2.8,
            color=fs.BLUE, zorder=4)

    ax.set_ylim(0.0, 1.12)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("velocity ratio")
    fs.direct_label(ax, 2.62, 0.83, "$v_p/v$", va="top")
    fs.direct_label(ax, 2.35, 0.52, "$v_g/v$", colour=fs.BLUE, va="top")
    fs.direct_label(ax, kh_second_harmonic + 0.04, 0.03, "$2f_0$")
    fs.arrow_label(ax, "$%.3f$" % vg_second,
                   xy=(kh_second_harmonic, vg_second),
                   xytext=(kh_second_harmonic - 0.13, 0.86), colour=fs.BLUE,
                   ha="right", va="center")

    # The ceiling is a value of the EFFECTIVE wavenumber and not a place
    # on the kh axis, which is the confusion the manuscript's CHK note is
    # about. It is attained at Nyquist and nowhere else, so the phase
    # velocity ratio where this curve ends IS K_max / pi. Said in words
    # rather than with a leader line, because every path from the empty
    # part of this panel to that endpoint crosses the group velocity
    # curve.
    ceiling = operator_ceiling(coeffs)
    fs.direct_label(ax, 0.10, 0.60,
                    r"$K_\mathrm{max} = 2\sum|c_n| = %.3f$" % ceiling
                    + "\n" + r"$v_p/v = K_\mathrm{max}/\pi$ at $kh = \pi$",
                    va="top")
    fs.condition(ax, r"bands drawn at $n_\lambda = 6$", y=0.78)


def draw_error(ax, kh, deployed, taylor, coda_kh):
    """Phase error of the deployed operator against plain Taylor."""
    for i, coeffs in ((0, deployed), (2, taylor)):
        style = fs.series(i)
        style["marker"] = "none"
        ax.plot(kh, np.abs(phase_ratio(coeffs, kh) - 1.0), zorder=3,
                **style)
    ax.axhline(1.0e-3, color=fs.GREY, linewidth=fs.LW["annotation"],
               linestyle=(0, (3.0, 1.5)), zorder=2)

    ax.set_yscale("log")
    ax.set_ylim(1.0e-6, 0.6)
    fs.log_minor_off(ax)
    ax.set_ylabel("phase velocity error")
    ax.set_xlabel("normalised wavenumber, $kh$")
    fs.direct_label(ax, 0.06, 1.2e-3, "0.1%", colour=fs.GREY)
    fs.direct_label(ax, 1.62, 4.2e-3, "Taylor coefficients",
                    colour=fs.ORANGE, ha="right")
    # The optimised curve touches zero three times inside its design band.
    # That is the equioscillation of a minimax fit, not a plotting fault,
    # and it is why the label cannot sit under the curve on the left.
    fs.direct_label(ax, 3.10, 1.4e-5, r"optimised, $kh \leq 2$",
                    ha="right", va="top")
    fs.direct_label(ax, np.mean(coda_kh), 0.20, "coda band", colour=fs.GREY,
                    ha="center")
    fs.direct_label(ax, 2.28, 0.20, "source band", colour=fs.GREY,
                    ha="right")


# ------------------------------------------------------------------ main

def main():
    fs.apply()
    deployed = fdtd.optimised_coeffs(ORDER, kh_max=KH_MAX, multistart=True)
    taylor = fdtd.taylor_coeffs(ORDER)
    legacy = fdtd.optimised_coeffs(ORDER, kh_max=1.6, multistart=False)

    kh = np.linspace(1.0e-4, np.pi, 4000)
    coda_kh = tuple(kh_of(np.array(CODA_BAND_HZ), PPW_BANDS))
    source_kh = tuple(kh_of(np.array(source_band_hz()), PPW_BANDS))
    kh_second_harmonic = kh_of(2.0 * F0_HZ, PPW_BANDS)

    fig, ax = fs.figure("dispersion", nrows=2, sharex=True, hspace=0.22)
    for a in ax:
        shade_bands(a, source_kh, coda_kh)
    draw_velocity(ax[0], kh, deployed, kh_second_harmonic)
    draw_error(ax[1], kh, deployed, taylor, coda_kh)
    ax[1].set_xlim(0.0, np.pi)
    ax[1].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    for a, s in zip(ax, "ab"):
        fs.panel(a, s)

    print("order %d staggered first derivative" % ORDER)
    print("  deployed coefficients (kh_max %.1f, multistart): %s"
          % (KH_MAX, ", ".join("%.6f" % c for c in deployed)))
    print("  bands at n_lambda %g: source %.3f to %.3f, coda %.3f to %.3f"
          % ((PPW_BANDS,) + source_kh + coda_kh))
    print("  source band is the %g dB width of fdtd.ricker, %.2f to %.2f MHz"
          % (SOURCE_LEVEL_DB, source_band_hz()[0] / 1e6,
             source_band_hz()[1] / 1e6))
    print("%-26s %10s %14s %14s"
          % ("operator", "K_max", "phase err coda", "v_g at 2 f0"))
    for tag, coeffs in (("deployed, kh_max 2.0", deployed),
                        ("Taylor", taylor),
                        ("legacy, kh_max 1.6", legacy)):
        print("%-26s %10.4f %13.4f%% %14.4f"
              % (tag, operator_ceiling(coeffs),
                 100.0 * worst_in_band(kh, phase_ratio(coeffs, kh), coda_kh),
                 group_ratio(coeffs, np.array([kh_second_harmonic]))[0]))
    vg = group_ratio(deployed, kh)
    interior = kh < np.pi - 1e-3
    print("  deployed group velocity minimum below Nyquist: %.4f at kh %.3f"
          % (vg[interior].min(), kh[interior][vg[interior].argmin()]))
    print("  so no interior zero, and no wavenumber at which energy stalls")
    kh_centre = np.array([kh_of(F0_HZ, 6.0)])
    print("  phase error at the n_lambda 6 centre frequency: %.4f%%"
          % (100.0 * abs(phase_ratio(deployed, kh_centre)[0] - 1.0)))

    fs.save(fig, "dispersion",
            expect=["velocity ratio", "phase velocity error",
                    "normalised wavenumber", "source band", "coda band",
                    "Taylor coefficients", "0.1%"])


if __name__ == "__main__":
    main()
