using Printf
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.Grids: λnode, φnode, znode, Center
using Oceananigans.Architectures: on_architecture, architecture
using Oceananigans.Fields: CenterField, interior
using SeawaterPolynomials.TEOS10: Sᴬ_from_Sᴾ, Θ_from_T
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity,
                                       ConvectiveAdjustmentVerticalDiffusivity
using NumericalEarth.EarthSystemModels.InterfaceComputations: COARELogarithmicSimilarityProfile,
                                                              WindDependentWaveFormulation,
                                                              MomentumRoughnessLength,
                                                              TemperatureDependentAirViscosity,
                                                              ScalarRoughnessLength,
                                                              atmosphere_sea_ice_stability_functions,
                                                              MomentumBasedFrictionVelocity,
                                                              LargeYeagerTransferCoefficients,
                                                              FixedIterations,
                                                              large_yeager_stability_functions,
                                                              RelativeVelocity,
                                                              WindVelocity,
                                                              ConstantGustiness,
                                                              ShearAwareGustiness

#####
##### Flux configurations
#####

"""
    corrected_atmosphere_ocean_fluxes(FT = Float64)

COARE 3.6-consistent atmosphere-ocean flux formulation with:
- Wind-dependent Charnock parameter (Edson et al. 2013, eq. 13)
- COARE logarithmic similarity profile (no ψ(ℓ/L) term)
- Minimum gustiness = 0.5 m/s (CICE / NCAR CORE-II convention)
- `gustiness` kwarg accepts either a `ConstantGustiness(min_gust, β)` (default; constant floor)
  or a `ShearAwareGustiness(c, min_gust, β)` (Mahrt-Sun 1995 / Edson 2013 form)
- Temperature-dependent air viscosity
"""
function corrected_atmosphere_ocean_fluxes(FT = Float64;
                                           gustiness = ConstantGustiness(FT; minimum_gustiness = 0.5))
    air_kinematic_viscosity = TemperatureDependentAirViscosity(FT)
    return SimilarityTheoryFluxes(FT;
                                  similarity_form              = COARELogarithmicSimilarityProfile(),
                                  gustiness                    = gustiness,
                                  momentum_roughness_length    = MomentumRoughnessLength(FT;
                                  wave_formulation             = WindDependentWaveFormulation(FT),
                                  air_kinematic_viscosity      = TemperatureDependentAirViscosity(FT)),
                                  temperature_roughness_length = ScalarRoughnessLength(FT; air_kinematic_viscosity),
                                  water_vapor_roughness_length = ScalarRoughnessLength(FT; air_kinematic_viscosity))
end

"""
    corrected_atmosphere_sea_ice_fluxes(FT = Float64)

Atmosphere-sea ice flux formulation with:
- SHEBA/Paulson+Grachev stability functions (existing default, correct)
- Fixed momentum roughness z0 = 5e-4 m (CICE/SHEBA standard; Andreas et al. 2010)
- Fixed scalar roughness z0t = z0q = 5e-5 m (Andreas 1987: z0t ≈ z0/10 at R*≈7)
- COARE logarithmic similarity profile
- Minimum gustiness = 0.2 m/s
"""
corrected_atmosphere_sea_ice_fluxes(FT = Float64) = 
    SimilarityTheoryFluxes(FT;
                           stability_functions          = atmosphere_sea_ice_stability_functions(FT),
                           similarity_form              = COARELogarithmicSimilarityProfile(),
                           minimum_gustiness            = FT(0.2),
                           momentum_roughness_length    = FT(5e-4),
                           temperature_roughness_length = FT(5e-5),
                           water_vapor_roughness_length = FT(5e-5))

"""
    corrected_ice_ocean_heat_flux()

Three-equation ice-ocean heat flux with momentum-based friction velocity
computed from actual ice-ocean stress (McPhee 1992, 2008; SHEBA median u*≈0.01 m/s).
"""
corrected_ice_ocean_heat_flux() = ThreeEquationHeatFlux(; friction_velocity = MomentumBasedFrictionVelocity())

"""
    ncar_atmosphere_ocean_fluxes(FT = Float64)

OMIP-2 standard atmosphere-ocean flux formulation using the Large & Yeager
(2004, 2009) bulk algorithm. Iterates directly on transfer coefficients (Cd, Ch, Ce),
NOT on roughness lengths. Uses 5 fixed iterations with Paulson stability functions.
"""
ncar_atmosphere_ocean_fluxes(FT = Float64) =
    CoefficientBasedFluxes(FT;
                           transfer_coefficients = LargeYeagerTransferCoefficients(FT),
                           solver_stop_criteria = FixedIterations(5))

"""
    ncar_atmosphere_sea_ice_fluxes(FT = Float64)

NCAR/CORE atmosphere-sea ice flux formulation with full Monin-Obukhov
similarity theory and stability corrections:
- Paulson (1970) + linear stable (-5ζ) stability functions (same as NCAR ocean)
- Fixed z0 = z0t = z0q = 5e-4 m (CICE default; SHEBA standard)
- Wind speed floor at 0.5 m/s
- COARE logarithmic similarity profile (no ψ(ℓ/L) term)

Over ice the roughness lengths are fixed geometric constants (not wind-dependent),
so the standard MOST roughness-length iteration is consistent here (unlike the
ocean case where the NCAR polynomial Cd requires its own solver).
"""
ncar_atmosphere_sea_ice_fluxes(FT = Float64) =
    SimilarityTheoryFluxes(FT;
                           stability_functions          = large_yeager_stability_functions(FT),
                           similarity_form              = COARELogarithmicSimilarityProfile(),
                           gustiness_parameter          = FT(0),
                           minimum_gustiness            = FT(0.5),
                           momentum_roughness_length    = FT(5e-4),
                           temperature_roughness_length = FT(5e-4),
                           water_vapor_roughness_length = FT(5e-4))

"""
    corrected_radiation(sea_ice)

Radiation with OMIP-2 standard ocean parameters (emissivity = 1.0, albedo = 0.06)
and CCSM3 temperature/snow/thickness-dependent sea ice albedo.
"""
function corrected_radiation(sea_ice)
    hi = sea_ice.model.ice_thickness
    hs = sea_ice.model.snow_thickness

    # When snow is present, the snow layer owns the surface temperature;
    # otherwise the ice top surface temperature is the atmosphere interface.
    snow_thermo = sea_ice.model.snow_thermodynamics
    Ts = if isnothing(snow_thermo)
        sea_ice.model.ice_thermodynamics.top_surface_temperature
    else
        snow_thermo.top_surface_temperature
    end

    sea_ice_albedo = SeaIceAlbedo(hi, hs, Ts)

    return Radiation(; ocean_emissivity  = 1.00,
                       ocean_albedo      = 0.06,
                       sea_ice_albedo)
end

"""
    build_coupled_model(ocean, sea_ice, atmosphere, radiation, flux_configuration;
                        velocity_formulation = :relative)

Build the `OceanSeaIceModel` with the specified flux configuration.
Options for `flux_configuration`: `:default`, `:corrected`, `:shear_aware`, `:ncar`.
Options for `velocity_formulation`:  `:relative`, `:wind`
"""
function build_coupled_model(ocean, sea_ice, atmosphere, radiation, flux_configuration;
                             velocity_formulation::Symbol = :relative)
    if flux_configuration == :default
        return OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
    end

    FT = eltype(ocean.model.grid)
    radiation = corrected_radiation(sea_ice)

    velocity_difference_obj = velocity_formulation == :relative ? RelativeVelocity() :
                              velocity_formulation == :wind     ? WindVelocity()     :
                              error("Unknown velocity_formulation: $velocity_formulation. Options: :relative, :wind")

    if flux_configuration == :corrected || flux_configuration == :shear_aware
        gustiness = flux_configuration == :shear_aware ?
                    ShearAwareGustiness(FT; shear_wind_scale = 0.04, minimum_gustiness = 0.5) :
                    ConstantGustiness(FT;   minimum_gustiness = 0.5)
        interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                         radiation,
                                         atmosphere_ocean_fluxes   = corrected_atmosphere_ocean_fluxes(FT; gustiness),
                                         atmosphere_sea_ice_fluxes = corrected_atmosphere_sea_ice_fluxes(FT),
                                         sea_ice_ocean_heat_flux   = corrected_ice_ocean_heat_flux(),
                                         atmosphere_ocean_velocity_difference   = velocity_difference_obj,
                                         atmosphere_sea_ice_velocity_difference = velocity_difference_obj)
    elseif flux_configuration == :ncar
        interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                         radiation,
                                         atmosphere_ocean_fluxes   = ncar_atmosphere_ocean_fluxes(FT),
                                         atmosphere_sea_ice_fluxes = ncar_atmosphere_sea_ice_fluxes(FT),
                                         sea_ice_ocean_heat_flux   = corrected_ice_ocean_heat_flux(),
                                         atmosphere_ocean_velocity_difference   = velocity_difference_obj,
                                         atmosphere_sea_ice_velocity_difference = velocity_difference_obj)
    else
        error("Unknown flux_configuration: $flux_configuration. Options: :default, :corrected, :shear_aware, :ncar")
    end

    return OceanSeaIceModel(ocean, sea_ice; atmosphere, interfaces)
end

#####
##### Main simulation builder
#####

"""
    omip_simulation(config::Symbol = :halfdegree; kwargs...)

Create a fully coupled ocean--sea-ice--atmosphere OMIP simulation.

The single positional argument selects the grid configuration:

- `:halfdegree`  -- 720x360   `TripolarGrid`
- `:tenthdegree` -- 3600x1800 `TripolarGrid`
- `:orca`        -- NEMO eORCA mesh

Returns a `Simulation` wrapping an `OceanSeaIceModel`. The simulation
already has a progress callback attached, and (when `diagnostics=true`)
the OMIP-protocol output writers from [`add_omip_diagnostics!`](@ref).

To restart from a previous run, simply call

    run!(sim; pickup = true)

which uses Oceananigans' built-in `Checkpointer` machinery — no extra
plumbing is needed because `NumericalEarth.EarthSystemModels` provides
`prognostic_state` / `restore_prognostic_state!` for the coupled model.

# Keyword arguments

- `arch`: architecture (`CPU()` or `GPU()`). Default: `CPU()`.
- `Nz::Int`: number of vertical levels. Default: `100`.
- `depth`: maximum ocean depth in metres. Default: `5500`.
- `κ_skew`, `κ_symmetric`: GM/Redi diffusivities. Defaults: `500`, `100`.
- `forcing_dir`: directory for JRA55 forcing data. Default: `"forcing_data"`.
- `restoring_dir`: directory for restoring/IC climatology. Default: `"climatology"`.
- `piston_velocity`: surface salinity restoring piston velocity in m/day. Default: `1/6`.
  Restoring is automatically masked by sea ice concentration (no restoring under ice).
- `start_date`, `end_date`: bracket for forcing/restoring metadata. Defaults: 1958-01-01 .. 2018-01-01.
- `Δt`: simulation time step. Default: `30minutes`.
- `stop_time`: stop time for the wrapping `Simulation`. Default: `Inf`.
- `flux_configuration`: surface flux formulation. Options:
   * `:default` — current defaults (Edson/COARE with constant Charnock 0.02)
   * `:corrected` — COARE 3.6 with wind-dependent Charnock, fixed ice roughness, momentum-based u*
   * `:shear_aware` — `:corrected` plus the Mahrt–Sun (1995) / Edson (2013)
                      shear-aware gustiness form (`ShearAwareGustiness`),
                      Uᴳ² = (β·w★)² + (c·|Δu|)² + Uᴳ₀². Designed to inject
                      additional gustiness at moderate winds where convective
                      gustiness is weak (e.g., the equator).
   * `:ncar` — OMIP-2 standard Large & Yeager (2004) bulk formulae
- `vertical_closure::Symbol`: ocean vertical-mixing closure. Options:
   * `:catke` — CATKE TKE-based scheme (default).
   * `:simple` — `ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1)` plus a
     depth-step background `VerticalScalarDiffusivity` (κ=10⁻², ν=10⁻² in upper
     100 m; κ=10⁻⁵, ν=10⁻⁴ below). For diagnostic A/B tests vs CATKE.
   * `:nori` — NORi base Richardson-number closure
     (xkykai/NORiOceanParameterization.jl, vendored as
     `nori_base_closure.jl`). Calibrated defaults; no `Cᵇ` parameter.
   * `:rbvd` — Oceananigans' built-in `RiBasedVerticalDiffusivity`
     (Richardson-number-based, with κ-clip and time-averaged smoothing).
     A battle-tested comparison point for `:nori`; no `Cᵇ` parameter.
- `velocity_formulation::Symbol`: Δu used by the bulk formula. Options:
   * `:relative` — `Δu = u_atm − u_ocean` (OMIP-2 α=1, default).
   * `:wind` — `Δu = u_atm` (ignores ocean current). For isolating bulk-formula
     response from current feedback (e.g. when an over-strong ACC self-reinforces).
- `diagnostics::Bool`: whether to attach OMIP diagnostics. Default: `true`.
- `surface_averaging_interval`, `field_averaging_interval`: averaging windows.
- `checkpoint_interval`: interval between checkpoint writes.
- `output_dir`, `filename_prefix`, `file_splitting_interval`: output configuration.
"""
function omip_simulation(config::Symbol = :halfdegree;
                         arch = CPU(),
                         Nz = 100,
                         depth = 5500,
                         Δz_top = nothing,
                         κ_skew = 250,
                         κ_symmetric = 100,
                         Cᵇ = 0.28,
                         biharmonic_timescale = 40days,
                         biharmonic_viscosity = nothing,
                         forcing_dir = joinpath(get(ENV, "DATA", ""), "forcing_data"),
                         staging_dir = nothing,
                         backend_size = 50,
                         restoring_dir = "climatology",
                         piston_velocity = 1 / 6, # m / day
                         start_date = DateTime(1958, 1, 1),
                         end_date = DateTime(2018, 1, 1),
                         Δt = 30minutes,
                         stop_time = Inf,
                         flux_configuration = :default,
                         vertical_closure = :catke,
                         velocity_formulation = :relative,
                         Cᵂu★ = nothing,
                         with_snow = false,
                         diagnostics = true,
                         field_mean_interval = 5days,
                         surface_averaging_interval = 5days,
                         field_averaging_interval = 15days,
                         checkpoint_interval = 360days,
                         output_dir = ".",
                         filename_prefix = string(config),
                         file_splitting_interval = 360days)

    cfg = Val(config)

    grid = build_grid(cfg, arch, Nz, depth; Δz_top)

    ocean = build_ocean(cfg, grid;
                        κ_skew, κ_symmetric, Cᵇ,
                        biharmonic_timescale,
                        biharmonic_viscosity,
                        vertical_closure,
                        Cᵂu★,
                        restoring_dir, piston_velocity,
                        start_date, end_date)

    snow_thermodynamics = with_snow ? NumericalEarth.SeaIces.default_snow_thermodynamics(grid) : nothing
    sea_ice = build_sea_ice(cfg, grid, ocean; restoring_dir, snow_thermodynamics)

    # When staging_dir is provided, JRA55 data is read from fast scratch
    # with symlink fallback to the slow source directory.
    if !isnothing(staging_dir)
        setup_staging_directory(forcing_dir, staging_dir)
        atmosphere_dir = staging_dir
    else
        atmosphere_dir = forcing_dir
    end

    atmosphere, radiation = omip_atmosphere(arch;
                                            forcing_dir = atmosphere_dir,
                                            start_date,
                                            end_date,
                                            backend_size)

    coupled = build_coupled_model(ocean, sea_ice, atmosphere, radiation, flux_configuration;
                                  velocity_formulation)

    simulation = Simulation(coupled; Δt, stop_time)

    for dir in [forcing_dir, restoring_dir, output_dir]
        if !isdir(dir)
            mkdir(dir)
        end
    end

    # Stage JRA55 data from slow disk to fast scratch
    if !isnothing(staging_dir)
        staging_callback = JRA55DataStagingCallback(; source_dir = forcing_dir,
                                                      staging_dir,
                                                      start_date)
        # Run monthly (≈1440 iterations at Δt=30min) — well ahead of year boundaries.
        # The callback only copies files at year transitions; otherwise it returns immediately.
        add_callback!(simulation, staging_callback, IterationInterval(1440))
    end

    wall_time = Ref(time_ns())
    add_callback!(simulation, omip_progress_callback(wall_time), IterationInterval(10))

    if diagnostics
        add_omip_diagnostics!(simulation;
                              surface_averaging_interval,
                              field_averaging_interval,
                              field_mean_interval,
                              checkpoint_interval,
                              output_dir,
                              filename_prefix,
                              file_splitting_interval)
    end

    return simulation
end

#####
##### WOA → TEOS-10 conversion utilities
#####
##### WOA's `t_an` is sea_water_temperature (in-situ, °C) and `s_an` is
##### sea_water_practical_salinity (PSS-78). Oceananigans' default
##### `TEOS10EquationOfState` expects Conservative Temperature (Θ) and
##### Absolute Salinity (S_A). The functions below convert WOA fields to the
##### TEOS-10 conventions in place, using SeawaterPolynomials (CPU only —
##### the SAAR atlas read is host-resident and the loop body is scalar).
#####

# Approximate hydrostatic pressure in dbar from depth z [m] (cell-center, negative for ocean).
@inline approx_pressure_dbar(z) = max(zero(z), -z)

"""
    woa_to_teos10!(T_field, S_field)

Convert WOA in-situ temperature `t [°C]` and Practical Salinity `S_P` to TEOS-10 Conservative Temperature `Θ`
and Absolute Salinity `S_A`, in place. Both fields must live on the same grid. The conversion runs on the host;
data is copied to/from the device automatically.
"""
function woa_to_teos10!(T_field, S_field)
    grid = T_field.grid
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(grid)
    T_h = Array(interior(T_field))
    S_h = Array(interior(S_field))
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        t  = T_h[i, j, k]
        SP = S_h[i, j, k]
        (isnan(t) || isnan(SP)) && continue
        λ = λnode(i, j, k, cpu_grid, Center(), Center(), Center())
        φ = φnode(i, j, k, cpu_grid, Center(), Center(), Center())
        z = znode(i, j, k, cpu_grid, Center(), Center(), Center())
        p = approx_pressure_dbar(z)
        SA = Sᴬ_from_Sᴾ(SP, p, λ, φ)
        Θ  = Θ_from_T(SA, t, p)
        T_h[i, j, k] = Θ
        S_h[i, j, k] = SA
    end
    copyto!(interior(T_field), T_h)
    copyto!(interior(S_field), S_h)
    return T_field, S_field
end

"""
    woa_salinity_fts_to_teos10!(fts)

Convert each time slice of a WOA Practical Salinity `FieldTimeSeries` to TEOS-10
Absolute Salinity, in place. Requires that all time indices be in memory
(use `time_indices_in_memory = length(metadata)`).
"""
function woa_salinity_fts_to_teos10!(fts)
    grid = fts.grid
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(grid)
    Nt = length(fts.times)
    for t_idx in 1:Nt
        S_int = interior(fts[t_idx])
        S_h   = Array(S_int)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            SP = S_h[i, j, k]
            isnan(SP) && continue
            λ = λnode(i, j, k, cpu_grid, Center(), Center(), Center())
            φ = φnode(i, j, k, cpu_grid, Center(), Center(), Center())
            z = znode(i, j, k, cpu_grid, Center(), Center(), Center())
            p = approx_pressure_dbar(z)
            S_h[i, j, k] = Sᴬ_from_Sᴾ(SP, p, λ, φ)
        end
        copyto!(S_int, S_h)
    end
    return fts
end

#####
##### Shared closure utilities
#####

@inline νhb(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, λ) = Oceananigans.Operators.Az(i, j, k, grid, ℓx, ℓy, ℓz)^2 / λ

# Background tracer diffusivity following Henyey et al. (1986).
@inline henyey_diffusivity(x, y, z, t) = max(1e-6, 5e-6 * abs(sind(y)))

# Step-function background diffusivity for the :simple closure.
# Strong mixing in the upper 100 m, weak interior diffusivity below.
@inline ν_step_simple(x, y, z, t) = ifelse(z >= -100, 1e-2, 1e-4)
@inline κ_step_simple(x, y, z, t) =
      z >= -10  ? 5e-2 :       # mimic BL mixing
      z >= -100 ? 1e-2 :
                  1e-5

# Build a vertical-mixing closure tuple. The eddy and horizontal
# components are common to every option; the primary vertical closure
# and any background κ/ν are selected by `vertical_closure`.
function omip_closure(vertical_closure::Symbol;
                      κ_skew, κ_symmetric, Cᵇ = 0.28,
                      biharmonic_timescale,
                      biharmonic_viscosity = nothing,
                      Cᵂu★ = nothing)

    primary, background = if vertical_closure == :catke
        mixing_length = CATKEMixingLength(; Cᵇ)
        tke_eq = isnothing(Cᵂu★) ? CATKEEquation() : CATKEEquation(; Cᵂu★)
        catke = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization();
                                         mixing_length,
                                         turbulent_kinetic_energy_equation = tke_eq)
        catke, VerticalScalarDiffusivity(κ=henyey_diffusivity, ν=3e-5)
    elseif vertical_closure == :simple
        convective = ConvectiveAdjustmentVerticalDiffusivity(VerticallyImplicitTimeDiscretization();
                                                             convective_κz = 1.0,
                                                             convective_νz = 1.0)
        background = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ=κ_step_simple, ν=ν_step_simple)
        convective, background
    elseif vertical_closure == :nori
        NORiBaseVerticalDiffusivity(), nothing
    elseif vertical_closure == :rbvd
        RiBasedVerticalDiffusivity(; horizontal_Ri_filter = Oceananigans.TurbulenceClosures.FivePointHorizontalFilter()), nothing
    else
        error("Unknown vertical_closure: $vertical_closure. Options: :catke, :simple, :nori, :rbvd")
    end

    eddy  = if isnothing(κ_skew) | isnothing(κ_symmetric)
        nothing
    else
        IsopycnalSkewSymmetricDiffusivity(; κ_skew, κ_symmetric)
    end

    horizontal_viscosity = if !isnothing(biharmonic_viscosity)
        HorizontalScalarBiharmonicDiffusivity(ν=biharmonic_viscosity)
    elseif !isnothing(biharmonic_timescale)
        HorizontalScalarBiharmonicDiffusivity(ν=νhb,
                                              discrete_form=true,
                                              parameters=biharmonic_timescale)
    else
        nothing
    end

    return filter(!isnothing, (primary, eddy, horizontal_viscosity, background))
end

#####
##### Salinity restoring (shared by both configurations)
#####

# Surface-only restoring, applied uniformly in space (no ice mask).
# Wrapped as a `SurfaceFluxRestoring` so it rides on the ocean's top-flux BC
# via the `additional_surface_fluxes` kwarg of `ocean_simulation`.
# WOA Practical Salinity is converted to TEOS-10 Absolute Salinity at setup so
# the restoring target matches the ocean prognostic-S convention.
function salinity_surface_restoring(grid, dataset;
                                    restoring_dir,
                                    piston_velocity)

    Nz = size(grid, 3)
    Δz_surface = CUDA.@allowscalar Δzᶜᶜᶜ(1, 1, Nz, grid)

    rate = piston_velocity / (Δz_surface * days)

    Smetadata = Metadata(:salinity; dir = restoring_dir, dataset)

    restoring = DatasetRestoring(Smetadata, Oceananigans.Architectures.architecture(grid);
                                 rate,
                                 time_indices_in_memory = length(Smetadata))

    woa_salinity_fts_to_teos10!(restoring.field_time_series)

    return SurfaceFluxRestoring(restoring)
end

#####
##### Grid builder
#####

function find_exponential_scale(Nz, depth, Δzᵀ; tolerance = 1e-7, maxiter = 200)
    Δzᵁ = depth / Nz
    Δzᵀ < Δzᵁ || throw(ArgumentError("Δzᵀ = $Δzᵀ must be < depth/Nz = $Δzᵁ"))
    Δzᵀ > 0   || throw(ArgumentError("Δzᵀ = $Δzᵀ must be positive"))

    Δz_at_scale(h) = depth * expm1(Δzᵁ / h) / expm1(depth / h)

    h⁻ = Δzᵁ / 1000
    h⁺ = 1000 * depth

    for _ in 1:maxiter
        h = (h⁻ + h⁺) / 2
        Δz = Δz_at_scale(h)
        abs(Δz - Δzᵀ) <= tolerance * Δzᵀ && return h
        Δz < Δzᵀ ? (h⁻ = h) : (h⁺ = h)
    end
    error("Could not converge to scale matching Δz_top = $Δz_top within relative tolerance $tolerance")
end

exponential_scale(Nz, depth, ::Nothing) = 1300
exponential_scale(Nz, depth, Δz_top)    = find_exponential_scale(Nz, depth, Δz_top)

function build_grid(config, arch, Nz, depth; Δz_top = nothing)

    Nx = config == Val(:halfdegree)  ? 720 :
         config == Val(:tenthdegree) ? 3600 :
         throw("Configuration $(config) does not exist")

    Ny = Nx ÷ 2

    scale = exponential_scale(Nz, depth, Δz_top)
    z_faces = ExponentialDiscretization(Nz, -depth, 0; scale, mutable=true)

    base_grid = TripolarGrid(arch;
                             size = (Nx, Ny, Nz),
                             z = z_faces,
                             halo = (7, 7, 7))

    bottom_height = regrid_bathymetry(base_grid;
                                    minimum_depth = 20,
                                    major_basins = 1,
                                    interpolation_passes = 25)

    return ImmersedBoundaryGrid(base_grid, GridFittedBottom(bottom_height); active_cells_map = true)
end

function build_grid(::Val{:orca}, arch, Nz, depth; Δz_top = nothing)

    scale = exponential_scale(Nz, depth, Δz_top)
    z_faces = ExponentialDiscretization(Nz, -depth, 0; scale, mutable=true)

    return ORCAGrid(arch;
                    dataset = ORCA1(),
                    Nz,
                    z = z_faces,
                    halo = (7, 7, 7),
                    with_bathymetry = true,
                    active_cells_map = true)
end

#####
##### ORCA builder
#####

config_momentum_advection(::Val{:orca})        = WENOVectorInvariant(order=5) 
config_momentum_advection(::Val{:halfdegree})  = WENOVectorInvariant(order=5)
config_momentum_advection(::Val{:tenthdegree}) = WENOVectorInvariant()

function build_ocean(config, grid;
                     κ_skew, κ_symmetric, Cᵇ = 0.28,
                     restoring_dir, piston_velocity,
                     biharmonic_timescale,
                     biharmonic_viscosity = nothing,
                     vertical_closure = :catke,
                     Cᵂu★ = nothing,
                     start_date, end_date)

    salt_restoring = salinity_surface_restoring(grid, WOAMonthly(); restoring_dir, piston_velocity)
    closure = omip_closure(vertical_closure;
                           κ_skew, κ_symmetric, Cᵇ,
                           biharmonic_timescale, biharmonic_viscosity,
                           Cᵂu★)
    coriolis = HydrostaticSphericalCoriolis(scheme = Oceananigans.Coriolis.EnstrophyConserving())
    momentum_advection = config_momentum_advection(config)

    ocean = ocean_simulation(grid;
                             Δt = 1minutes,
                             momentum_advection,
                             tracer_advection = WENO(order=7; minimum_buffer_upwind_order=3),
                             coriolis,
                             timestepper = :SplitRungeKutta3,
                             free_surface = SplitExplicitFreeSurface(grid; substeps=70),
                             additional_surface_fluxes = (; S = salt_restoring),
                             closure)

    # Load WOA Annual T (in-situ, °C) and S (Practical) onto the model grid,
    # convert to TEOS-10 Conservative T and Absolute Salinity in place, then
    # initialize the prognostic ocean state from the converted fields.
    T_init = CenterField(grid)
    S_init = CenterField(grid)
    set!(T_init, Metadatum(:temperature; dir=restoring_dir, dataset=WOAAnnual()))
    set!(S_init, Metadatum(:salinity;    dir=restoring_dir, dataset=WOAAnnual()))
    woa_to_teos10!(T_init, S_init)
    set!(ocean.model, T=T_init, S=S_init)

    return ocean
end

#####
##### Sea Ice builder
#####

function build_sea_ice(config, grid, ocean; restoring_dir, snow_thermodynamics = nothing)
    sea_ice = sea_ice_simulation(grid, ocean;
                                 advection = WENO(order=7, minimum_buffer_upwind_order=1),
                                 snow_thermodynamics)

    set!(sea_ice.model,
         h = Metadatum(:sea_ice_thickness;     dir=restoring_dir, dataset=ECCO4Monthly()),
         ℵ = Metadatum(:sea_ice_concentration; dir=restoring_dir, dataset=ECCO4Monthly()))

    return sea_ice
end

#####
##### Progress callback
#####

function omip_progress_callback(wall_time)
    function progress(sim)
        sea_ice = sim.model.sea_ice
        ocean   = sim.model.ocean

        hmax = maximum(sea_ice.model.ice_thickness)
        ℵmax = maximum(sea_ice.model.ice_concentration)
        Tmax = maximum(ocean.model.tracers.T)
        Tmin = minimum(ocean.model.tracers.T)
        Smax = maximum(ocean.model.tracers.S)
        Smin = minimum(ocean.model.tracers.S)
        umax = maximum(ocean.model.velocities.u)
        vmax = maximum(ocean.model.velocities.v)
        wmax = maximum(ocean.model.velocities.w)

        step_time = 1e-9 * (time_ns() - wall_time[])

        msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ",
                        prettytime(sim), iteration(sim), prettytime(sim.Δt))
        msg2 = @sprintf("max(h): %.2e m, max(ℵ): %.2e ", hmax, ℵmax)
        msg3 = @sprintf("extrema(T, S): (%.2f, %.2f) ᵒC, (%.2f, %.2f) psu ",
                        Tmin, Tmax, Smin, Smax)
        msg4 = @sprintf("maximum(u): (%.2e, %.2e, %.2e) m/s, ", umax, vmax, wmax)
        msg5 = @sprintf("wall time: %s", prettytime(step_time))

        @info msg1 * msg2 * msg3 * msg4 * msg5

        wall_time[] = time_ns()

        return nothing
    end

    return progress
end
