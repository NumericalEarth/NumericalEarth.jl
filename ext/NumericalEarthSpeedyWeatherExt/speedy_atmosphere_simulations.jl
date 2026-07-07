const SpeedySimulation = SpeedyWeather.Simulation
const SpeedyEarthSystemModel = NumericalEarth.EarthSystemModel{<:Any, <:SpeedySimulation}
const SpeedyNoSeaIceEarthSystemModel = NumericalEarth.EarthSystemModel{<:Union{Nothing, NumericalEarth.SeaIces.FreezingLimitedOceanTemperature}, <:SpeedySimulation}

Base.summary(::SpeedySimulation) = "SpeedyWeather.Simulation"

# Take one time-step or more depending on the global timestep
function Oceananigans.TimeSteppers.time_step!(atmos::SpeedySimulation, Δt)
    Δt_atmos = atmos.model.time_stepping.Δt_sec
    nsteps = ceil(Int, Δt / Δt_atmos)

    if (Δt / Δt_atmos) % 1 != 0
        @warn "NumericalEarth only supports atmosphere timesteps that are integer divisors of the ESM timesteps"
    end

    for _ in 1:nsteps
        SpeedyWeather.timestep!(atmos)
    end
end

## Make sure the atmospheric parameters from SpeedyWeather can be used in the compute fluxes function

# The height of near-surface variables used in the turbulent flux solver
function NumericalEarth.EarthSystemModels.surface_layer_height(sim::SpeedySimulation)
    T = sim.model.atmosphere.reference_temperature
    g = sim.model.planet.gravity
    Φ = sim.model.geopotential.Δp_geopot_full
    Φₙ = @allowscalar Φ[end]
    return Φₙ * T / g
end

# This is a parameter that is used in the computation of the fluxes,
# It probably should not be here but in the similarity theory type.
NumericalEarth.EarthSystemModels.boundary_layer_height(::SpeedySimulation) = 600

# This is a _hack_!! The parameters should be consistent with what is specified in SpeedyWeather
NumericalEarth.EarthSystemModels.thermodynamics_parameters(::SpeedySimulation) = AtmosphereThermodynamicsParameters(Float32)

function initialize_atmospheric_state!(simulation::SpeedyWeather.Simulation)
    vars, model = SpeedyWeather.unpack(simulation)
    (; time) = vars.prognostic.clock  # current time

    # set the tendencies back to zero for accumulation
    SpeedyWeather.reset_tendencies!(vars)

    if !model.dynamics_only
        SpeedyWeather.parameterization_tendencies!(vars, model)
    end

    return nothing
end

"""
    atmosphere_simulation(spectral_grid::SpeedyWeather.SpectralGrid;
                          output_interval=nothing,
                          stop_time=nothing,
                          time_stepping=SpeedyWeather.Leapfrog(spectral_grid))

Return an atmosphere simulation using `SpeedyWeather.PrimitiveWetModel` on `spectral_grid`.
Output is written when `output_interval` is provided. `time_stepping` controls
SpeedyWeather's internal timestep (e.g. via its `Δt_at_T31` field); the resulting
`Δt_sec` must be an integer divisor of the coupled `EarthSystemModel` timestep.

`stop_time` should match the `stop_time` later passed to the coupled
`Oceananigans.Simulation`. It only synchronizes SpeedyWeather's internal clock
(used for, e.g., `progress.txt` reporting); the actual run length is controlled
by the coupled `Simulation`, not by SpeedyWeather's clock.
"""
function NumericalEarth.Atmospheres.atmosphere_simulation(spectral_grid::SpeedyWeather.SpectralGrid;
                                                          output_interval=nothing,
                                                          stop_time=nothing,
                                                          time_stepping=SpeedyWeather.Leapfrog(spectral_grid))
    # Surface fluxes
    humidity_flux_ocean = SpeedyWeather.PrescribedOceanHumidityFlux(spectral_grid)
    humidity_flux_land = SpeedyWeather.SurfaceLandHumidityFlux(spectral_grid)
    surface_humidity_flux = SpeedyWeather.SurfaceHumidityFlux(ocean=humidity_flux_ocean, land=humidity_flux_land)

    ocean_heat_flux = SpeedyWeather.PrescribedOceanHeatFlux(spectral_grid)
    land_heat_flux = SpeedyWeather.SurfaceLandHeatFlux(spectral_grid)
    surface_heat_flux = SpeedyWeather.SurfaceHeatFlux(ocean=ocean_heat_flux, land=land_heat_flux)

    # The atmospheric model
    atmosphere_model = SpeedyWeather.PrimitiveWetModel(
        spectral_grid;
        time_stepping,
        surface_heat_flux,
        surface_humidity_flux,
        ocean = SpeedyWeather.PrescribedOcean(),
        sea_ice = nothing # provided by ClimaSeaIce
    )

    output = !isnothing(output_interval)
    output && (atmosphere_model.output.interval = SpeedyWeather.Second(output_interval))

    # Construct the simulation
    atmosphere = SpeedyWeather.initialize!(atmosphere_model)

    # Initialize the simulation, syncing SpeedyWeather's internal clock with the
    # coupled simulation's intended stop time so progress.txt reports the correct
    # run duration.
    period_kw = isnothing(stop_time) ? NamedTuple() : (; period=SpeedyWeather.Second(stop_time))
    SpeedyWeather.initialize!(atmosphere; output, period_kw...)

    # Fill in prognostic fields
    initialize_atmospheric_state!(atmosphere)

    return atmosphere
end

Oceananigans.Simulations.reset_clock!(atmos::SpeedyWeather.Simulation) =
    SpeedyWeather.initialize!(atmos.prognostic_variables.clock)
