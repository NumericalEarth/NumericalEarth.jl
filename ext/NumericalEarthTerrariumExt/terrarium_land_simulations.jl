import NumericalEarth.Land: land_simulation

const LandSimulation = Terrarium.ModelIntegrator
const LandEarthSystemModel = NumericalEarth.EarthSystemModel{<:Any, <:Any, <:LandSimulation} #TODO: fix when updated

Base.summary(::LandSimulation) = "Terrarium.ModelIntegrator"

# Take one time-step or more depending on the global timestep
function Oceananigans.TimeSteppers.time_step!(integrator::Terrarium.ModelIntegrator, Δt)
    Δt_land = Terrarium.default_dt(integrator.timestepper)
    nsteps = ceil(Int, Δt / Δt_land)

    if (Δt / Δt_land) % 1 != 0
        @warn "NumericalEarth only supports land timesteps that are integer divisors of the ESM timesteps"
    end

    for _ in 1:nsteps
        Terrarium.timestep!(integrator, Δt_land)
    end
    return
end

"""
    land_simulation(grid::Terrarium.AbstractLandGrid)

Return a land simulation based on the given `AbstractLandGrid`.

TODO: add more kwarg config options
"""
function land_simulation(grid::Terrarium.AbstractLandGrid)
    # The land model
    land_model = Terrarium.LandModel(grid)

    # Construct the Terrarium integrator
    integrator = Terrarium.initialize(land_model)

    return integrator
end
