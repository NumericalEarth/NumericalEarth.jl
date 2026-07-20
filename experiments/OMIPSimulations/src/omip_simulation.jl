using Printf
using KernelAbstractions: @index, @kernel
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.Grids: λnode, φnode, znode, Center
using Oceananigans.Architectures: on_architecture, architecture
using Oceananigans.DistributedComputations: @root
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction, getbc, fill_halo_regions!
using Oceananigans.Fields: CenterField, interior
using Oceananigans.ImmersedBoundaries: bottom_height_field
using Oceananigans.Utils: launch!
using NumericalEarth.Bathymetry: remove_minor_basins!
using NumericalEarth.Oceans: MultipleFluxes, FreshwaterExchange
using SeawaterPolynomials.TEOS10: Sᴬ_from_Sᴾ, Θ_from_T
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity,
                                       ConvectiveAdjustmentVerticalDiffusivity
using Oceananigans.Utils: NormalDivision
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
                                                              ConvectiveGustiness

#####
##### Flux configurations
#####

"""
    corrected_atmosphere_ocean_fluxes(FT = Float64)

COARE 3.6-consistent atmosphere-ocean flux formulation with:
- Wind-dependent Charnock parameter (Edson et al. 2013, eq. 13)
- COARE logarithmic similarity profile (no ψ(ℓ/L) term)
- Minimum gustiness = 0.5 m/s (CICE / NCAR CORE-II convention)
- Temperature-dependent air viscosity
"""
function corrected_atmosphere_ocean_fluxes(FT = Float64;
                                           subgrid_velocities = ConvectiveGustiness{FT}(minimum_gustiness = FT(0.5)))
    air_kinematic_viscosity = TemperatureDependentAirViscosity(FT)
    return SimilarityTheoryFluxes(FT;
                                  similarity_form              = COARELogarithmicSimilarityProfile(),
                                  subgrid_velocities           = subgrid_velocities,
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
                           subgrid_velocities           = ConvectiveGustiness{FT}(minimum_gustiness = FT(0.2)),
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
                           subgrid_velocities           = ConvectiveGustiness{FT}(gustiness_parameter = FT(0),
                                                                                  minimum_gustiness   = FT(0.5)),
                           momentum_roughness_length    = FT(5e-4),
                           temperature_roughness_length = FT(5e-4),
                           water_vapor_roughness_length = FT(5e-4))

"""
    build_coupled_model(ocean, sea_ice, atmosphere, radiation, land, flux_configuration;
                        velocity_formulation = :relative)

Build the `OceanSeaIceModel` with the specified flux configuration.
Options for `flux_configuration`: `:default`, `:corrected`, `:ncar`.
Options for `velocity_formulation`:  `:relative`, `:wind`
"""
function build_coupled_model(ocean, sea_ice, atmosphere, radiation, land, flux_configuration;
                             velocity_formulation::Symbol = :relative)
    FT = eltype(ocean.model.grid)
    if flux_configuration == :default
        interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice; radiation, land)
        return OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation, land, interfaces)
    end

    velocity_difference_obj = velocity_formulation == :relative ? RelativeVelocity() :
                              velocity_formulation == :wind     ? WindVelocity()     :
                              error("Unknown velocity_formulation: $velocity_formulation. Options: :relative, :wind")

    if flux_configuration == :corrected
        interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                         radiation,
                                         land,
                                         atmosphere_ocean_fluxes   = corrected_atmosphere_ocean_fluxes(FT),
                                         atmosphere_sea_ice_fluxes = corrected_atmosphere_sea_ice_fluxes(FT),
                                         sea_ice_ocean_heat_flux   = corrected_ice_ocean_heat_flux(),
                                         atmosphere_ocean_velocity_difference   = velocity_difference_obj,
                                         atmosphere_sea_ice_velocity_difference = velocity_difference_obj)
    elseif flux_configuration == :ncar
        interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                         radiation,
                                         land,
                                         atmosphere_ocean_fluxes   = ncar_atmosphere_ocean_fluxes(FT),
                                         atmosphere_sea_ice_fluxes = ncar_atmosphere_sea_ice_fluxes(FT),
                                         sea_ice_ocean_heat_flux   = corrected_ice_ocean_heat_flux(),
                                         atmosphere_ocean_velocity_difference   = velocity_difference_obj,
                                         atmosphere_sea_ice_velocity_difference = velocity_difference_obj)
    else
        error("Unknown flux_configuration: $flux_configuration. Options: :default, :corrected, :ncar")
    end

    return OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation, land, interfaces)
end

#####
##### Salinity flux normalization
#####
#
# At each callback, subtract the global mean of the *restoring* flux alone from the
# bulk-flux Field, so the restoring contributes zero net global salt while the physical
# coupled and freshwater fluxes keep their real global means. Standard OMIP practice
# (e.g. NorESM/BLOM) normalizes the restoring flux to a zero global mean.
#
# The restoring rides inside the salinity BC's `FreshwaterExchange` (which also carries
# the freshwater volume flux); `restoring_flux` extracts only the restoring term. A bare
# 2D `Field` BC has no restoring, so its normalizer is a no-op.

@kernel function _materialize_top_flux!(buffer, additional, grid, clock, fields)
    i, j = @index(Global, NTuple)
    @inbounds buffer[i, j, 1] = getbc(additional, i, j, grid, clock, fields)
end

struct NormalizeSalinity{F, A, B, M}
    flux_field       :: F   # bulk flux field (gets corrected each call)
    restoring        :: A   # restoring flux callable (or `nothing`)
    restoring_buffer :: B   # 2D scratch field for the materialized restoring flux
    mean_restoring   :: M   # Field(Average(restoring_buffer, dims=(1,2)))
end

# The restoring rides inside a `FreshwaterExchange` alongside the freshwater volume flux;
# extract only the restoring term so the freshwater flux keeps its real global mean.
restoring_flux(fe::FreshwaterExchange) = fe.additional
restoring_flux(additional) = additional

salinity_normalizer(bc::DiscreteBoundaryFunction) = salinity_normalizer(bc.func)

function salinity_normalizer(mf::MultipleFluxes)
    flux_field       = mf.flux_field
    restoring        = restoring_flux(mf.additional_fluxes)
    restoring_buffer = similar(flux_field)
    fill!(parent(restoring_buffer), 0)
    mean_restoring   = Field(Average(restoring_buffer, dims=(1, 2)))
    return NormalizeSalinity(flux_field, restoring, restoring_buffer, mean_restoring)
end

salinity_normalizer(f::Field) = NormalizeSalinity(f, nothing, nothing, nothing)

function (n::NormalizeSalinity)(sim)
    isnothing(n.restoring) && return nothing
    model  = sim.model.ocean.model
    grid   = model.grid
    arch   = architecture(grid)
    fields = merge(model.velocities, model.tracers)
    launch!(arch, grid, :xy, _materialize_top_flux!,
            n.restoring_buffer, n.restoring, grid, model.clock, fields)
    compute!(n.mean_restoring)
    parent(n.flux_field) .-= n.mean_restoring
    return nothing
end

#####
##### Main simulation builder
#####

"""
    omip_simulation(config::Symbol = :halfdegree; kwargs...)

Create a fully coupled ocean--sea-ice--atmosphere OMIP simulation.

The single positional argument selects the grid configuration:

- `:halfdegree`    -- 720x360   `TripolarGrid`
- `:quarterdegree` -- NEMO eORCA025 (1/4ᵒ) mesh
- `:twelfthdegree` -- NEMO eORCA12 (1/12ᵒ) mesh
- `:orca`          -- NEMO eORCA1 (~1ᵒ) mesh
- `:test`          -- NEMO eORCA1 (~1ᵒ) mesh, locally-runnable preset for reproducing the
                      quarter-degree spurious high-latitude ice + surface salinity drift.
                      Overrides `Nz = 15`, `Δz_top = 1.5` m, `Δt = 45minutes`, and a `10days`
                      biharmonic-viscosity timescale. GM/Redi is disabled (`κ_skew = κ_symmetric =
                      nothing`) to keep short test runs cheap; momentum advection follows `:orca`.

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
- `Nz::Int`: number of vertical levels. Per-config default: `15` for `:test`, `100` otherwise.
- `depth`: maximum ocean depth in metres. Default: `5500`.
- `Δz_top`: target surface-cell thickness in metres (sets the exponential vertical scale). Per-config
  default: `1.5` for `:quarterdegree`/`:twelfthdegree`/`:test`, `nothing` (scale derived from
  `depth`/`Nz`) otherwise.
- `κ_skew`, `κ_symmetric`: GM/Redi diffusivities. Per-config defaults: `nothing` (no isopycnal
  diffusivity) for the eddy-resolving `:quarterdegree`/`:twelfthdegree` and for `:test`, `800` for
  `:orca`, `250` for `:halfdegree`.
- `biharmonic_timescale`: horizontal biharmonic-viscosity timescale. Per-config default: `nothing`
  (no biharmonic viscosity) for `:quarterdegree`/`:twelfthdegree`, `10days` for `:test`, `50days`
  otherwise.
- `forcing_dir`: directory for JRA55 forcing data. Default: `"forcing_data"`.
- `restoring_dir`: directory for restoring/IC climatology. Default: `"climatology"`.
- `piston_velocity`: surface salinity restoring piston velocity in m/day. Default: `1/6`.
  Restoring is automatically masked by sea ice concentration (no restoring under ice).
- `start_date`, `end_date`: bracket for forcing/restoring metadata. Defaults: 1958-01-01 .. 2018-01-01.
- `Δt`: simulation time step. Per-config default: `5minutes` for `:twelfthdegree`, `20minutes` for
  `:quarterdegree`, `30minutes` otherwise.
- `stop_time`: stop time for the wrapping `Simulation`. Default: `Inf`.
- `flux_configuration`: surface flux formulation. Options:
   * `:default` — current defaults (Edson/COARE with constant Charnock 0.02)
   * `:corrected` — COARE 3.6 with wind-dependent Charnock, fixed ice roughness, momentum-based u*
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
   * `:kpp` — KPP boundary-layer scheme (Large 1994 / MITgcm), vendored
     in `KPP/`. Includes nonlocal tracer flux + SW-aware Bf. No `Cᵇ`.
   * `:nemo_tke` — NEMO 3.6 TKE scheme (Blanke & Delecluse 1993; Gaspar et al.
     1990; Madec et al. 2017), vendored in `NEMOTKE/`. OMIP-2 ORCAOne preset:
     prognostic e, gradient-limited length scale, Langmuir + Mellor-Blumberg
     wave penetration + EVD on static instability. No `Cᵇ`.
- `implicit_vertical_advection::Bool`: if `true` (default), tracer and momentum vertical advection use
  `AdaptiveVerticallyImplicitDiscretization(cfl=0.5)` (switches the vertical advective flux to implicit
  where the vertical Courant number is large — e.g. in thin near-surface cells). If `false`, fully
  explicit `WENO`/`WENOVectorInvariant`. Use `false` to isolate adaptive-implicit advection effects.
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
                         Nz = ConfigDefault(),
                         depth = 5500,
                         Δz_top = ConfigDefault(),
                         κ_skew = ConfigDefault(),
                         κ_symmetric = ConfigDefault(),
                         Cᵇ = 0.28,
                         biharmonic_timescale = ConfigDefault(),
                         biharmonic_viscosity = nothing,
                         forcing_dir = joinpath(get(ENV, "DATA", ""), "forcing_data"),
                         staging_dir = nothing,
                         backend_size = 50,
                         restoring_dir = "climatology",
                         piston_velocity = 1 / 6, # m / day
                         start_date = DateTime(1958, 1, 1),
                         end_date = DateTime(2018, 1, 1),
                         Δt = ConfigDefault(),
                         stop_time = Inf,
                         flux_configuration = :default,
                         vertical_closure = :catke,
                         implicit_vertical_advection = true,
                         velocity_formulation = :relative,
                         Cᵂu★ = nothing,
                         with_snow = false,
                         with_ice_dynamics = true,
                         normalize_salinity = true,
                         river_mixing = true,
                         river_mixing_κ = 0.1,
                         river_mixing_depth = 10,
                         diagnostics = true,
                         field_mean_interval = 5days,
                         surface_averaging_interval = 5days,
                         field_averaging_interval = 15days,
                         checkpoint_interval = 360days,
                         output_dir = ".",
                         filename_prefix = string(config),
                         file_splitting_interval = 360days)

    cfg = Val(config)

    # Resolve resolution-sensitive parameters to their per-configuration defaults unless the
    # user passed an explicit value (see `config_*` below).
    Nz                   = resolve_config_default(Nz,                   config_Nz(cfg))
    Δz_top               = resolve_config_default(Δz_top,               config_Δz_top(cfg))
    κ_skew               = resolve_config_default(κ_skew,               config_κ_skew(cfg))
    κ_symmetric          = resolve_config_default(κ_symmetric,          config_κ_symmetric(cfg))
    biharmonic_timescale = resolve_config_default(biharmonic_timescale, config_biharmonic_timescale(cfg))
    Δt                   = resolve_config_default(Δt,                   config_Δt(cfg))

    grid = build_grid(cfg, arch, Nz, depth; Δz_top)

    # When staging_dir is provided, JRA55 data is read from fast scratch
    # with symlink fallback to the slow source directory.
    if !isnothing(staging_dir)
        setup_staging_directory(forcing_dir, staging_dir)
        atmosphere_dir = staging_dir
    else
        atmosphere_dir = forcing_dir
    end

    # Build the land before the ocean so its river routing can seed enhanced vertical mixing at
    # river mouths — an extra closure that keeps concentrated runoff from freshening a cell to zero.
    # Closing the shallow Ob/Yenisei gulfs relocates their mouths ~2–3° onto the deeper Kara Sea shelf,
    # so the routing search must reach that far. A fixed geographic reach keeps it resolution-independent.
    Nx, Ny, _ = size(grid)
    maximum_search_radius = max(5, ceil(Int, 3 / ((360 / Nx + 180 / Ny) / 2)))
    land = JRA55PrescribedLand(grid; dir = atmosphere_dir, dataset = MultiYearJRA55(),
                               start_date, end_date, time_indices_in_memory = backend_size, prefetch = true,
                               maximum_search_radius)

    river_κ = river_mixing ?
        river_mouth_vertical_diffusivity(grid, land.river_routing; κ = river_mixing_κ, mixing_depth = river_mixing_depth) :
        nothing

    ocean = build_ocean(cfg, grid;
                        κ_skew, κ_symmetric, Cᵇ,
                        biharmonic_timescale,
                        biharmonic_viscosity,
                        vertical_closure,
                        implicit_vertical_advection,
                        Cᵂu★,
                        restoring_dir, piston_velocity,
                        additional_tracer_closure = river_κ,
                        start_date, end_date)

    snow_thermodynamics = with_snow ? NumericalEarth.SeaIces.default_snow_thermodynamics(grid) : nothing
    sea_ice = build_sea_ice(cfg, grid, ocean; restoring_dir, snow_thermodynamics, with_ice_dynamics)

    atmosphere, radiation = omip_forcing(arch, sea_ice;
                                         forcing_dir = atmosphere_dir,
                                         start_date,
                                         end_date,
                                         backend_size)

    coupled = build_coupled_model(ocean, sea_ice, atmosphere, radiation, land, flux_configuration;
                                  velocity_formulation)

    simulation = Simulation(coupled; Δt, stop_time)

    # Only rank 0 creates dirs; others barrier inside @root and proceed once
    # the dirs exist. mkpath is idempotent so a race-free retry would also
    # work, but @root keeps the pattern symmetric with the staging code.
    @root for dir in [forcing_dir, restoring_dir, output_dir]
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

    if normalize_salinity
        NS = salinity_normalizer(ocean.model.tracers.S.boundary_conditions.top.condition)
        add_callback!(simulation, NS, IterationInterval(1))
    end


    wall_time = Ref(time_ns())
    add_callback!(simulation, omip_progress_callback(wall_time), IterationInterval(1))

    if diagnostics
        add_omip_diagnostics!(simulation;
                              surface_averaging_interval,
                              field_averaging_interval,
                              field_mean_interval,
                              checkpoint_interval,
                              output_dir,
                              filename_prefix,
                              file_splitting_interval)

        # Dispatches to the active method only for cfg == Val(:twelfthdegree);
        # other configurations get the no-op fallback.
        add_ke_spectrum_diagnostic!(simulation, cfg;
                                     output_dir,
                                     filename_prefix,
                                     flush_interval = field_averaging_interval)
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
    cpu_arch = Oceananigans.DistributedComputations.cpu_architecture(architecture(grid))
    cpu_grid = on_architecture(cpu_arch, grid)
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
    cpu_arch = Oceananigans.DistributedComputations.cpu_architecture(architecture(grid))
    cpu_grid = on_architecture(cpu_arch, grid)
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
@inline henyey_diffusivity(x, y, z, t) = max(2e-6, 1e-5 * abs(sind(y)))

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
                                         maximum_viscosity=3,
                                         maximum_tracer_diffusivity=3,
                                         maximum_tke_diffusivity=3,
                                         negative_tke_damping_time_scale=10, # (seconds)
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
        convective = RiBasedVerticalDiffusivity(; horizontal_Ri_filter = Oceananigans.TurbulenceClosures.FivePointHorizontalFilter())
        background = VerticalScalarDiffusivity(κ=henyey_diffusivity, ν=1e-4)
        convective, background
    elseif vertical_closure == :kpp
        KPPVerticalDiffusivity(), nothing
    elseif vertical_closure == :nemo_tke
        NEMOTKEVerticalDiffusivity(), nothing
    else
        error("Unknown vertical_closure: $vertical_closure. Options: :catke, :simple, :nori, :rbvd, :kpp, :nemo_tke")
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

# Enhanced vertical mixing at river mouths (cf. NEMO `rn_avt_rnf` over `rn_hrnf`): an extra tracer
# diffusivity `κ` over the top `mixing_depth` metres at the routed river-mouth cells, mixing the
# fresh plume downward so a coastal surface cell cannot be freshened to zero. Added to the closure.
@inline river_mouth_κ(i, j, k, grid, clock, fields, mask) = @inbounds mask[i, j, k]

function river_mouth_vertical_diffusivity(grid, river_routing; κ = 0.1, mixing_depth = 10)
    zc = Array(znodes(grid, Center()))
    Nz = size(grid, 3)
    mask_data = zeros(eltype(grid), size(grid)...)

    for routing in values(river_routing)
        ti = Array(routing.target_i)
        tj = Array(routing.target_j)
        for n in eachindex(ti), k in 1:Nz
            zc[k] > -mixing_depth && (mask_data[ti[n], tj[n], k] = κ)
        end
    end

    mask = CenterField(grid)
    set!(mask, mask_data)

    return VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization();
                                     κ = river_mouth_κ, discrete_form = true,
                                     loc = (Center, Center, Center), parameters = mask)
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
    error("Could not converge to scale matching Δz_top = $Δzᵀ within relative tolerance $tolerance")
end

exponential_scale(Nz, depth, ::Nothing) = 1300
exponential_scale(Nz, depth, Δz_top)    = find_exponential_scale(Nz, depth, Δz_top)

function build_grid(config, arch, Nz, depth; Δz_top = nothing)

    Nx = config == Val(:halfdegree) ? 720 : throw("Configuration $(config) does not exist")

    Ny = Nx ÷ 2

    scale = exponential_scale(Nz, depth, Δz_top)
    z_faces = ExponentialDiscretization(Nz, -depth, 0; scale, mutable=true)

    base_grid = TripolarGrid(arch;
                             size = (Nx, Ny, Nz),
                             z = z_faces,
                             halo = (8, 8, 8))

    bottom_height = regrid_bathymetry(base_grid;
                                    minimum_depth = 20,
                                    major_basins = 1,
                                    interpolation_passes = 25)

    return ImmersedBoundaryGrid(base_grid, GridFittedBottom(bottom_height); active_cells_map = true)
end

build_grid(::Val{:orca}, arch, Nz, depth; Δz_top = nothing)          = build_grid(ORCAOne(),     arch, Nz, depth; Δz_top)
build_grid(::Val{:quarterdegree}, arch, Nz, depth; Δz_top = nothing) = build_grid(ORCAQuarter(), arch, Nz, depth; Δz_top)
build_grid(::Val{:twelfthdegree}, arch, Nz, depth; Δz_top = nothing) = build_grid(ORCATwelfth(), arch, Nz, depth; Δz_top)

# The Gulf of Ob and the Yenisei Gulf are ~5 m deep for hundreds of kilometres, so their full river
# discharge lands in a single 1.5 m top cell with no water column to mix into and the salinity collapses.
# Closing the sub-`minimum_depth` cells of each gulf turns them to land; the river routing then relocates
# the discharge onto the deeper Kara Sea shelf just north, where the top ~10 m spans five cells.
# Boxes are (λ_min, λ_max, φ_min, φ_max) in degrees.
const kara_river_closures = ((68.0, 77.0, 66.0, 72.6),   # Gulf of Ob
                             (77.0, 85.0, 70.0, 73.8))    # Yenisei Gulf

@kernel function _close_shallow_regions!(bottom_height, grid, regions, minimum_depth)
    i, j = @index(Global, NTuple)
    λ = λnode(i, j, 1, grid, Center(), Center(), Center())
    φ = φnode(i, j, 1, grid, Center(), Center(), Center())
    @inbounds z = bottom_height[i, j, 1]
    shallow = (z < 0) & (z > -minimum_depth)
    closed = false
    for (λ₀, λ₁, φ₀, φ₁) in regions
        closed = closed | (shallow & (λ ≥ λ₀) & (λ ≤ λ₁) & (φ ≥ φ₀) & (φ ≤ φ₁))
    end
    @inbounds bottom_height[i, j, 1] = ifelse(closed, oftype(z, 100), z)
end

function close_shallow_river_regions(grid; regions = kara_river_closures, minimum_depth = 10)
    arch      = architecture(grid)
    underlying = grid.underlying_grid
    bottom    = bottom_height_field(grid)
    launch!(arch, underlying, :xy, _close_shallow_regions!, bottom, underlying, regions,
            convert(eltype(grid), minimum_depth))
    fill_halo_regions!(bottom)
    remove_minor_basins!(bottom, 1)
    return ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom); active_cells_map = true)
end

function build_grid(dataset::ORCADataset, arch, Nz, depth; Δz_top = nothing)

    scale = exponential_scale(Nz, depth, Δz_top)
    z_faces = ExponentialDiscretization(Nz, -depth, 0; scale, mutable=true)

    grid = ORCAGrid(arch;
                    dataset,
                    Nz,
                    z = z_faces,
                    halo = (8, 8, 8),
                    with_bathymetry = true,
                    major_basins = 1,
                    active_cells_map = true)

    return grid # close_shallow_river_regions(grid)
end

# Locally-runnable testing configuration: the NEMO eORCA1 (~1ᵒ) mesh, used to reproduce the
# quarter-degree spurious high-latitude ice + surface salinity drift at a fraction of the cost.
build_grid(::Val{:test}, arch, Nz, depth; Δz_top = nothing) = build_grid(Val(:orca), arch, Nz, depth; Δz_top)

#####
##### ORCA builder
#####

using Oceananigans.TimeSteppers: AdaptiveVerticallyImplicitDiscretization, ExplicitTimeDiscretization
using Oceananigans.Utils: NormalDivision

# `time_discretization` selects explicit vs. adaptive-implicit vertical advection (see `build_ocean`).
config_momentum_advection(::Val{:orca},          td) = WENOVectorInvariant(order=5, time_discretization=td)
config_momentum_advection(::Val{:test},          td) = WENOVectorInvariant(order=5, time_discretization=td)
config_momentum_advection(::Val{:halfdegree},    td) = WENOVectorInvariant(order=5, time_discretization=td)
config_momentum_advection(::Val{:quarterdegree}, td) = WENOVectorInvariant(time_discretization=td)
config_momentum_advection(::Val{:twelfthdegree}, td) = WENOVectorInvariant(time_discretization=td)

struct ConfigDefault end

@inline resolve_config_default(value, default)           = value
@inline resolve_config_default(::ConfigDefault, default) = default

config_Nz(::Val)        = 100
config_Nz(::Val{:test}) = 15

config_κ_skew(::Val{:orca})          = 800
config_κ_skew(::Val{:halfdegree})    = 250
config_κ_skew(::Val{:quarterdegree}) = nothing
config_κ_skew(::Val{:twelfthdegree}) = nothing
config_κ_skew(::Val{:test})          = nothing

config_κ_symmetric(::Val{:orca})          = 800
config_κ_symmetric(::Val{:halfdegree})    = 250
config_κ_symmetric(::Val{:quarterdegree}) = nothing
config_κ_symmetric(::Val{:twelfthdegree}) = nothing
config_κ_symmetric(::Val{:test})          = nothing

config_biharmonic_timescale(::Val)                 = 50days
config_biharmonic_timescale(::Val{:quarterdegree}) = nothing
config_biharmonic_timescale(::Val{:twelfthdegree}) = nothing
config_biharmonic_timescale(::Val{:test})          = 10days

config_Δt(::Val)                 = 30minutes
config_Δt(::Val{:quarterdegree}) = 20minutes
config_Δt(::Val{:twelfthdegree}) = 5minutes
config_Δt(::Val{:test})          = 45minutes

config_Δz_top(::Val)                 = nothing
config_Δz_top(::Val{:quarterdegree}) = 1.5
config_Δz_top(::Val{:twelfthdegree}) = 1.5
config_Δz_top(::Val{:test})          = 1.5

# Buoyancy gradients are only needed by the GM/Redi closures; the eddy-resolving
# configurations run without isopycnal diffusivities, so skip materializing them.
config_materialize_buoyancy_gradients(::Val)                 = true
config_materialize_buoyancy_gradients(::Val{:quarterdegree}) = false
config_materialize_buoyancy_gradients(::Val{:twelfthdegree}) = false
config_materialize_buoyancy_gradients(::Val{:test})          = false

function build_ocean(config, grid;
                     κ_skew, κ_symmetric, Cᵇ = 0.28,
                     restoring_dir, piston_velocity,
                     biharmonic_timescale,
                     biharmonic_viscosity = nothing,
                     vertical_closure = :catke,
                     implicit_vertical_advection = true,
                     Cᵂu★ = nothing,
                     additional_tracer_closure = nothing,
                     start_date, end_date)

    salt_restoring = salinity_surface_restoring(grid, WOAMonthly(); restoring_dir, piston_velocity)
    closure = omip_closure(vertical_closure;
                           κ_skew, κ_symmetric, Cᵇ,
                           biharmonic_timescale, biharmonic_viscosity,
                           Cᵂu★)
    closure = isnothing(additional_tracer_closure) ? closure : (closure..., additional_tracer_closure)
    coriolis = HydrostaticSphericalCoriolis(scheme = Oceananigans.Coriolis.EnstrophyConserving())

    time_discretization = implicit_vertical_advection ?
        AdaptiveVerticallyImplicitDiscretization(cfl=0.5) : ExplicitTimeDiscretization()
    momentum_advection = config_momentum_advection(config, time_discretization)

    ocean = ocean_simulation(grid;
                             Δt = 1minutes,
                             momentum_advection,
                             tracer_advection = WENO(order=7; minimum_buffer_upwind_order=3, time_discretization),
                             coriolis,
                             timestepper = :SplitRungeKutta3,
                             materialize_buoyancy_gradients = config_materialize_buoyancy_gradients(config),
                             free_surface = SplitExplicitFreeSurface(grid; substeps=100),
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

function build_sea_ice(config, grid, ocean; restoring_dir, snow_thermodynamics = nothing, with_ice_dynamics = true)
    dynamics = with_ice_dynamics ? NumericalEarth.SeaIces.sea_ice_dynamics(grid, ocean) : nothing
    sea_ice = sea_ice_simulation(grid, ocean;
                                 advection = WENO(order=7, minimum_buffer_upwind_order=1, weight_computation=NormalDivision),
                                 dynamics,
                                 snow_thermodynamics)

    set!(sea_ice.model,
         h = Metadatum(:sea_ice_thickness;     dir=restoring_dir, dataset=ECCO4Monthly(), date = DateTime(1993, 1, 1)),
         ℵ = Metadatum(:sea_ice_concentration; dir=restoring_dir, dataset=ECCO4Monthly(), date = DateTime(1993, 1, 1)))

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

        # Determinism probe: hash the prognostic state at a few iterations.
        # Compare these hashes between two pickup-from-same-checkpoint runs:
        # the first iteration whose hashes differ pinpoints when divergence
        # is introduced. Cheap (~ms; one host copy of each field).
        iter = iteration(sim)
        if iter in (1, 5, 100, 1000)
            T  = Array(parent(interior(ocean.model.tracers.T)))
            S  = Array(parent(interior(ocean.model.tracers.S)))
            u  = Array(parent(interior(ocean.model.velocities.u)))
            h  = Array(parent(interior(sea_ice.model.ice_thickness)))
            @info @sprintf("STATE_HASH iter=%d  T=%016x  S=%016x  u=%016x  h=%016x",
                           iter, hash(T), hash(S), hash(u), hash(h))
        end

        wall_time[] = time_ns()

        return nothing
    end

    return progress
end
