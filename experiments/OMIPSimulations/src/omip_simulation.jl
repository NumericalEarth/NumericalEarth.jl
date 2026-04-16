using Printf
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity

#####
##### Flux configurations
#####

"""
    corrected_atmosphere_ocean_fluxes(FT = Float64)

COARE 3.6-consistent atmosphere-ocean flux formulation with:
- Wind-dependent Charnock parameter (Edson et al. 2013, eq. 13)
- COARE logarithmic similarity profile (no ψ(ℓ/L) term)
- Minimum gustiness = 0.2 m/s (Fairall et al. 2003)
- Temperature-dependent air viscosity
"""
function corrected_atmosphere_ocean_fluxes(FT = Float64) 
    air_kinematic_viscosity = TemperatureDependentAirViscosity(FT)
    return SimilarityTheoryFluxes(FT;
                                  similarity_form              = COARELogarithmicSimilarityProfile(),
                                  minimum_gustiness            = FT(0.2),
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

OMIP-2 standard atmosphere-ocean flux formulation using the NCAR/Large & Yeager
(2004, 2009) bulk algorithm. Iterates directly on transfer coefficients (Cd, Ch, Ce),
NOT on roughness lengths. Uses 5 fixed iterations with Paulson stability functions.
"""
ncar_atmosphere_ocean_fluxes(FT = Float64) = NCARBulkFluxes(FT)

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
                           stability_functions          = ncar_stability_functions(FT),
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

    return Radiation(; ocean_emissivity  = 1.0,
                       ocean_albedo      = 0.06,
                       sea_ice_albedo)
end

"""
    build_coupled_model(ocean, sea_ice, atmosphere, radiation, flux_configuration)

Build the `OceanSeaIceModel` with the specified flux configuration.
Options: `:default`, `:corrected`, `:ncar`.
"""
function build_coupled_model(ocean, sea_ice, atmosphere, radiation, flux_configuration)
    if flux_configuration == :default
        return OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
    end

    FT = eltype(ocean.model.grid)
    radiation = corrected_radiation(sea_ice)

    if flux_configuration == :corrected
        interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                         radiation,
                                         atmosphere_ocean_fluxes   = corrected_atmosphere_ocean_fluxes(FT),
                                         atmosphere_sea_ice_fluxes = corrected_atmosphere_sea_ice_fluxes(FT),
                                         sea_ice_ocean_heat_flux   = corrected_ice_ocean_heat_flux())
    elseif flux_configuration == :ncar
        interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                         radiation,
                                         atmosphere_ocean_fluxes   = ncar_atmosphere_ocean_fluxes(FT),
                                         atmosphere_sea_ice_fluxes = ncar_atmosphere_sea_ice_fluxes(FT),
                                         sea_ice_ocean_heat_flux   = corrected_ice_ocean_heat_flux())
    else
        error("Unknown flux_configuration: $flux_configuration. Options: :default, :corrected, :ncar")
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
   * `:ncar` — OMIP-2 standard Large & Yeager (2004) bulk formulae
- `diagnostics::Bool`: whether to attach OMIP diagnostics. Default: `true`.
- `surface_averaging_interval`, `field_averaging_interval`: averaging windows.
- `checkpoint_interval`: interval between checkpoint writes.
- `output_dir`, `filename_prefix`, `file_splitting_interval`: output configuration.
"""
function omip_simulation(config::Symbol = :halfdegree;
                         arch = CPU(),
                         Nz = 100,
                         depth = 5500,
                         κ_skew = 250,
                         κ_symmetric = 100,
                         biharmonic_timescale = 40days,
                         forcing_dir = "forcing_data",
                         restoring_dir = "climatology",
                         piston_velocity = 1 / 6, # m / day
                         start_date = DateTime(1958, 1, 1),
                         end_date = DateTime(2018, 1, 1),
                         Δt = 30minutes,
                         stop_time = Inf,
                         flux_configuration = :default,
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

    # Build the grid first so we can allocate the restoring mask
    grid = build_grid(cfg, arch, Nz, depth)

    # Pre-allocate restoring mask (1 = open water); updated each step from sea ice concentration
    ice_free_fraction = Field{Center, Center, Nothing}(grid)
    set!(ice_free_fraction, 1)

    ocean = build_ocean(cfg, grid;
                        κ_skew, κ_symmetric,
                        biharmonic_timescale,
                        restoring_dir, piston_velocity,
                        restoring_mask = ice_free_fraction,
                        start_date, end_date)

    snow_thermodynamics = with_snow ? NumericalEarth.SeaIces.default_snow_thermodynamics(grid) : nothing
    sea_ice = build_sea_ice(cfg, grid, ocean; restoring_dir, snow_thermodynamics)

    atmosphere, radiation = omip_atmosphere(arch;
                                            forcing_dir,
                                            start_date,
                                            end_date)

    coupled = build_coupled_model(ocean, sea_ice, atmosphere, radiation, flux_configuration)

    simulation = Simulation(coupled; Δt, stop_time)

    for dir in [forcing_dir, restoring_dir, output_dir]
        if !isdir(dir)
            mkdir(dir)
        end
    end

    # Callback to sync the restoring mask with sea ice concentration each coupled step
    ℵ = sea_ice.model.ice_concentration
    update_restoring_mask!(sim) = parent(ice_free_fraction) .= 1 .- parent(ℵ)
    add_callback!(simulation, update_restoring_mask!, IterationInterval(1))

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
##### Shared closure utilities
#####

@inline νhb(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, λ) = Oceananigans.Operators.Az(i, j, k, grid, ℓx, ℓy, ℓz)^2 / λ

# Background tracer diffusivity following Henyey et al. (1986).
@inline henyey_diffusivity(x, y, z, t) = max(1e-6, 5e-6 * abs(sind(y)))

function omip_closure(; κ_skew, κ_symmetric, biharmonic_timescale)
    catke = default_ocean_closure()

    eddy  = if isnothing(κ_skew) | isnothing(κ_symmetric)
        nothing
    else
        IsopycnalSkewSymmetricDiffusivity(; κ_skew, κ_symmetric)
    end

    horizontal_viscosity = if isnothing(biharmonic_timescale)
        nothing
    else
        HorizontalScalarBiharmonicDiffusivity(ν=νhb,
                                              discrete_form=true,
                                              parameters=biharmonic_timescale)
    end

    vertical_diffusivity = VerticalScalarDiffusivity(κ=henyey_diffusivity, ν=3e-5)

    return filter(!isnothing, (catke, eddy, horizontal_viscosity, vertical_diffusivity))
end

#####
##### Salinity restoring (shared by both configurations)
#####

function salinity_restoring_forcing(grid, dataset;
                                    restoring_dir,
                                    piston_velocity,
                                    mask = 1)

    Nz = size(grid, 3)
    Δz_surface = CUDA.@allowscalar Δzᶜᶜᶜ(1, 1, Nz, grid)

    rate = piston_velocity / (Δz_surface * days)

    Smetadata = Metadata(:salinity;
                         dir = restoring_dir,
                         dataset)

    return DatasetRestoring(Smetadata, Oceananigans.Architectures.architecture(grid);
                            rate, mask,
                            time_indices_in_memory = 12)
end

#####
##### Grid builder
#####

function build_grid(config, arch, Nz, depth)
    
    Nx = config == Val(:halfdegree)  ? 720 :
         config == Val(:tenthdegree) ? 3600 :
         throw("Configuration $(config) does not exist") 

    Ny = Nx ÷ 2

    z_faces = ExponentialDiscretization(Nz, -depth, 0; scale=1300, mutable=true)

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

function build_grid(::Val{:orca}, arch, Nz, depth)

    z_faces = ExponentialDiscretization(Nz, -depth, 0; scale=1300, mutable=true)

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

config_momentum_advection(::Val{:orca})        = VectorInvariant()
config_momentum_advection(::Val{:halfdegree})  = WENOVectorInvariant(order=5)
config_momentum_advection(::Val{:tenthdegree}) = WENOVectorInvariant()

function build_ocean(config, grid;
                     κ_skew, κ_symmetric,
                     restoring_dir, piston_velocity,
                     biharmonic_timescale,
                     restoring_mask = 1,
                     start_date, end_date)

    FS = salinity_restoring_forcing(grid, WOAMonthly(); restoring_dir, piston_velocity, mask = restoring_mask)

    closure = omip_closure(; κ_skew, κ_symmetric, biharmonic_timescale)
    coriolis = HydrostaticSphericalCoriolis(scheme = Oceananigans.Coriolis.EnstrophyConserving())
    momentum_advection = config_momentum_advection(config)

    ocean = ocean_simulation(grid;
                             Δt = 1minutes,
                             momentum_advection,
                             tracer_advection = WENO(order=7; minimum_buffer_upwind_order=3),
                             coriolis,
                             timestepper = :SplitRungeKutta3,
                             free_surface = SplitExplicitFreeSurface(grid; substeps=70),
                             surface_restoring = (; S = FS),
                             closure)

    set!(ocean.model,
         T = Metadatum(:temperature; dir=restoring_dir, dataset=WOAAnnual(), date=start_date),
         S = Metadatum(:salinity;    dir=restoring_dir, dataset=WOAAnnual(), date=start_date))

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
