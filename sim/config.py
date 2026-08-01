"""Every parameter of the pulse-echo azimuthal COF experiment, in one place.

PROVENANCE TAGS on each value:
  [T]  from the PhD thesis (App. B.4 reflection system, Ch.4 methodology)
  [D]  from the manufacturer datasheet
  [M]  measured or derived during design analysis
  [A]  ASSUMED - you should check or change this
Anything tagged [A] is a guess. Change it.
"""
from dataclasses import dataclass, field
import numpy as np


# ───────────────────────────── specimen ──────────────────────────────
@dataclass
class Specimen:
    diameter_m: float = 100e-3      # [T] Method A: 100 mm CNC-machined disk
    thickness_m: float = 35e-3      # [T/user] 35 mm
    rho: float = 917.0              # [D] ice density

    # grain structure
    n_grains: int = 100             # [A] CHECK: sets mean grain size below
    grain_size_m: float = None      # derived; set n_grains OR this

    # fabric: c-axis orientation distribution
    # 'random'        -> uniform on sphere (max disorder, max scattering)
    # 'single_max'    -> clustered about `fabric_axis`, spread `fabric_kappa`
    # 'girdle'        -> c-axes in a plane normal to `fabric_axis`
    fabric: str = "single_max"      # [A]
    fabric_axis: tuple = (0.0, 0.0, 1.0)   # [A] vertical single maximum
    fabric_kappa: float = 4.0       # [A] Watson concentration; 0=random, large=tight

    def mean_grain_size(self):
        """Equivalent spherical grain diameter for n_grains in the disk."""
        V = np.pi * (self.diameter_m / 2) ** 2 * self.thickness_m
        return 2.0 * (3.0 * V / (4.0 * np.pi * self.n_grains)) ** (1 / 3)


# ───────────────────────────── transducer ────────────────────────────
@dataclass
class Probe:
    f0: float = 5.0e6               # [user] 5 MHz
    element_d_m: float = 6.35e-3    # [D] 6.35 mm, Evident V3XX series [T]
    frac_bw: float = 0.6            # [A] broadband contact probe, ~60%
    # pulser (OPLabBox / OPTEL) [T]
    pamp: float = 50.0              # [T] @PAMP default; units undocumented
    pulse_width_s: float = 0.1e-6   # [M] half-period at 5 MHz. THESIS USED 0.2/0.5us
                                    #     which suit 2.5/1.0 MHz - wrong for this probe
    gain_db: float = 40.0           # [T] thesis used 40 dB

    def wavelength(self, c=3850.0):
        return c / self.f0

    def beam_half_angle(self, c=3850.0):
        """First-null half-angle of a circular piston."""
        return np.arcsin(min(1.0, 1.22 * self.wavelength(c) / self.element_d_m))

    def near_field_m(self, c=3850.0):
        return self.element_d_m ** 2 / (4 * self.wavelength(c))

    def range_res_m(self, c=3850.0):
        """c / 2B  with B = frac_bw * f0."""
        return c / (2 * self.frac_bw * self.f0)

    def cross_range_m(self, c=3850.0):
        """D/4.88 - set by aperture ONLY, independent of frequency."""
        return self.element_d_m / 4.88

    def beam_diameter_at(self, r_m, c=3850.0):
        return self.element_d_m + 2 * r_m * np.tan(self.beam_half_angle(c))


# ─────────────────────────── acquisition ─────────────────────────────
@dataclass
class Acquisition:
    az_step_deg: float = 2.0        # [T] thesis used 2 deg (180 angles)
    n_averages: int = 1024          # [T] DSOX3024A hardware averaging
    adc_bits: int = 8               # [D] DSOX3024A is 8-bit
    fs: float = 100e6               # [A] sample rate actually saved (scope does 4 GSa/s)
    record_end_s: float = 115e-6    # [M] must exceed E2 (~104 us) to capture it
    prf_hz: float = 1000.0          # [T] @PRF default
    lift_and_reseat: bool = True    # [T] Z retracts to clearance after every angle
    contact_trigger_g: float = 5.0  # [T] load-cell threshold ~5 g

    # reference-azimuth revisits for drift correction [M]
    ref_revisit_every: int = 20     # 0 disables

    # --- forward-model sampling (not physical, these are solver knobs) ---
    n_rays_inplane: int = 9         # rays across the beam cone, in the scan plane
    n_rays_outplane: int = 5        # rays out of plane (the 35 mm thickness)
    n_reverb: int = 2               # rim round trips to synthesise: E1, E2, ...
    include_multiples: bool = True  # rim-to-boundary multiples fill E1..E2
    offaxis_rim_suppression: float = 0.05   # [M] rays whose return misses the
                                    # element keep only diffuse content
    seek_time_s: float = 1.2        # [A] rotate + lift + reseat per angle
    window_frac: float = 0.18       # E1/E2 pick window, fraction of nominal

    # Out-of-plane probe tilt. With every ray in the scan plane, n_z never
    # multiplies anything in n.d, so the c-axis is only recoverable up to
    # reflection in that plane (ambiguity arccos|cos 2*theta|, worst at 45 deg
    # colatitude). Tilting makes n_z enter linearly and breaks it.
    # Geometric ceiling: from mid-height the ray rises D*tan(tilt) over the
    # diameter and must stay inside the half-thickness, so for a 100 x 35 mm
    # disk tilt <= atan(17.5/100) = 9.9 deg. Starting low doubles that.
    tilts_deg: tuple = (0.0,)

    def n_angles(self):
        return int(round(360.0 / self.az_step_deg))


# ──────────────────── error sources: REALISTIC LIMITS ────────────────
@dataclass
class Errors:
    """Every term here degrades the synthetic data. Set enabled=False to
    get the ideal case and isolate which error kills the reconstruction."""
    enabled: bool = True

    # --- angular ---------------------------------------------------------
    # The thesis quotes +/-2 deg backlash and U~+/-2.3 deg (k=2), but states
    # these are "specification-derived limits; no independently calibrated
    # uncertainty budget has been established". They are worst-case bounds
    # for the OLD open-loop rig, not measurements. Realistically:
    #   - scanning is UNIDIRECTIONAL, so backlash never unwinds -> ~0
    #   - toothed belt lost motion ~0; its error is elastic and systematic
    #   - 1/32 microstep = 0.0141 deg at specimen (4:1 belt) [T]
    #   - open-loop microstep ACCURACY ~5-10% of a full step -> ~0.05 deg
    unidirectional_scan: bool = True     # [T] 0,2,...,358 with no reversal
    angle_backlash_deg: float = 0.0      # zero while unidirectional
    microstep_deg: float = 0.0141        # [T] resolution at specimen
    microstep_accuracy_deg: float = 0.05 # [M] open-loop landing error

    # AS5600 fitted AT THE SPECIMEN: you measure true position, so the
    # drive errors above stop mattering. [D] datasheet figures.
    encoder_on_specimen: bool = True     # <-- CONFIRMED: this is the build
    encoder_res_deg: float = 0.088       # 12-bit over 360 deg
    encoder_rms_deg: float = 0.015       # RMS output noise, slow filter
    encoder_inl_deg: float = 1.0         # systematic, smooth -> calibratable
    index_reinsertion_deg: float = 0.5   # [A] with an index mark + encoder

    # --- couplant -------------------------------------------------------
    # [M] flat 6.35 mm face on a 50 mm radius rim -> 101 um sagitta,
    #     ~25 um beam-weighted mean gap, ~40 um azimuth-to-azimuth variation
    couplant_gap_mean_m: float = 25e-6
    couplant_gap_sigma_m: float = 40e-6
    couplant_c: float = 1440.0      # [D] light mineral oil at ~20C
    couplant_dcdT: float = -3.6     # [D] m/s per K -> ~1590 m/s at -10C
    rim_is_oiled: bool = True       # [user] far rim carries oil film too
    # The oil film is ~25 um against a 0.77 mm wavelength, so the far rim
    # behaves as ice/air and reflects almost everything. [M]
    rim_reflection: float = -0.97
    near_rim_internal: float = 0.55  # [M] partial: the probe loads this face

    # --- thermal --------------------------------------------------------
    # [T] operating temperature ~ -10 C
    # [M] dv/dT = -2.3 m/s/K for ice; chest freezer compressor cycles
    T_mean_C: float = -10.0
    T_cycle_amplitude_K: float = 1.5
    T_cycle_period_s: float = 1500.0
    dvdT: float = -2.3

    # --- specimen geometry ----------------------------------------------
    # [M] 107 um of diameter runout = 0.11% = the whole 4.1 m/s budget
    rim_runout_m: float = 107e-6
    rim_runout_harmonics: int = 3

    # --- mechanical ------------------------------------------------------
    z_position_sigma_m: float = 0.1e-3   # [T] CNC practical accuracy ~0.1 mm
    contact_force_sigma_g: float = 1.0   # [A]

    # --- electronic -------------------------------------------------------
    trigger_jitter_s: float = 1e-9       # [A] CHECK against the scope spec
    noise_rms_frac: float = 1e-3         # [A] receiver noise as fraction of full scale


# ───────────────── optical ground truth (the comparison) ─────────────
@dataclass
class OpticalTruth:
    """The thin section that provides grain geometry.

    THE VOLUME MISMATCH IS A FIRST-CLASS ERROR TERM:
    the section is a thin slice at mid-height, but the ultrasonic beam
    integrates over a much taller band, so grains outside the section
    plane scatter without appearing in the ground truth.
    """
    section_thickness_m: float = 2.0e-3   # [T] nominal 2.0 +/- 0.5 mm
    section_height_frac: float = 0.5      # [T] mid-height
    # [T] "the section plane is the same plane as the ultrasonic
    #      propagation plane for the reflection scan"
    coplanar_with_beam: bool = True
    boundary_pos_sigma_m: float = 0.2e-3  # [A] segmentation accuracy


# ─────────────────────────── forward model ───────────────────────────
@dataclass
class Solver:
    # VALIDATED production values (coda_convergence.py): amplitude work
    # needs ppw >= 6 with order 8 - the convergence series flattens at 6,
    # and ppw 10 just costs (10/6)^4 ~ 7x more for nothing. CFL 0.5 is
    # the amplitude-safe timestep (0.6-0.7 usable for timing-only).
    ppw: int = 6                    # points per wavelength (>= 6 validated)
    dim: int = 3                    # 3 = honest (out-of-plane scattering); 2 = fast
    cfl: float = 0.5
    use_gpu: bool = True
    c_ref: float = 3850.0           # reference speed for grid sizing

    def h(self, probe: Probe):
        return probe.wavelength(self.c_ref) / self.ppw

    def grid_cells(self, spec: Specimen, probe: Probe):
        h = self.h(probe)
        nx = int(np.ceil(spec.diameter_m / h))
        nz = int(np.ceil(spec.thickness_m / h))
        return (nx, nx, nz) if self.dim == 3 else (nx, nx)


@dataclass
class Config:
    specimen: Specimen = field(default_factory=Specimen)
    probe: Probe = field(default_factory=Probe)
    acq: Acquisition = field(default_factory=Acquisition)
    err: Errors = field(default_factory=Errors)
    optical: OpticalTruth = field(default_factory=OpticalTruth)
    solver: Solver = field(default_factory=Solver)
    seed: int = 7

    def summary(self):
        s, p, a, sv = self.specimen, self.probe, self.acq, self.solver
        c = sv.c_ref
        R = s.diameter_m / 2
        t_e1 = 2 * s.diameter_m / c
        L = []
        L.append("SPECIMEN")
        L.append(f"   disk {s.diameter_m*1e3:.0f} mm dia x {s.thickness_m*1e3:.0f} mm thick")
        L.append(f"   {s.n_grains} grains -> mean size {s.mean_grain_size()*1e3:.2f} mm")
        L.append(f"   fabric: {s.fabric} (kappa={s.fabric_kappa})")
        L.append("PROBE")
        L.append(f"   {p.f0/1e6:.1f} MHz, {p.element_d_m*1e3:.2f} mm element, lambda={p.wavelength(c)*1e3:.2f} mm")
        L.append(f"   range res {p.range_res_m(c)*1e3:.2f} mm | cross-range {p.cross_range_m()*1e3:.2f} mm (aperture-set)")
        L.append(f"   beam half-angle {np.degrees(p.beam_half_angle(c)):.1f} deg | near field {p.near_field_m(c)*1e3:.1f} mm")
        L.append(f"   beam dia at centre {p.beam_diameter_at(R, c)*1e3:.1f} mm  vs disk thickness {s.thickness_m*1e3:.0f} mm")
        L.append("ACQUISITION")
        L.append(f"   {a.n_angles()} angles at {a.az_step_deg} deg, {a.n_averages} averages")
        L.append(f"   E1 at {t_e1*1e6:.1f} us, E2 at {2*t_e1*1e6:.1f} us, record to {a.record_end_s*1e6:.0f} us")
        L.append(f"   ADC {a.adc_bits} bit = {6.02*a.adc_bits:.0f} dB; +{10*np.log10(a.n_averages):.0f} dB from averaging")
        L.append(f"   probe {'lifts and reseats each angle' if a.lift_and_reseat else 'stays in contact'}")
        L.append("DOMINANT ERRORS")
        e = self.err
        ang = e.encoder_rms_deg if e.encoder_on_specimen else e.microstep_accuracy_deg
        src = "AS5600 at specimen (RMS)" if e.encoder_on_specimen else "open-loop microstep"
        L.append(f"   angular {ang:.3f} deg RMS ({src}); backlash {e.angle_backlash_deg:.2f} deg (unidirectional={e.unidirectional_scan})")
        L.append(f"   couplant gap {e.couplant_gap_mean_m*1e6:.0f} +/- {e.couplant_gap_sigma_m*1e6:.0f} um")
        L.append(f"   thermal +/-{e.T_cycle_amplitude_K:.1f} K -> {abs(e.dvdT)*e.T_cycle_amplitude_K:.1f} m/s")
        L.append(f"   rim runout {e.rim_runout_m*1e6:.0f} um -> {c*e.rim_runout_m/s.diameter_m:.1f} m/s")
        L.append("GROUND TRUTH MISMATCH")
        o = self.optical
        band = p.beam_diameter_at(R, c)
        L.append(f"   optical section {o.section_thickness_m*1e3:.1f} mm vs beam band ~{band*1e3:.1f} mm")
        L.append(f"   -> ultrasound sees ~{band/o.section_thickness_m:.1f}x the volume the section describes")
        L.append("SOLVER")
        nc = sv.grid_cells(s, p)
        L.append(f"   {sv.dim}D, h={sv.h(p)*1e3:.3f} mm, grid {nc} = {np.prod(nc)/1e6:.1f} M cells")
        return "\n".join(L)


if __name__ == "__main__":
    print(Config().summary())
