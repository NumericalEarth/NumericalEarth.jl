"""
    land_model(grid::Terrarium.AbstractLandGrid; kwargs...)

Return a `Terrarium.LandModel` on the given land `grid` with components listed in `kwargs`. 
"""
function NumericalEarth.Lands.land_model(
        grid::Terrarium.AbstractLandGrid{NF};
        surface_energy_balance = deferred_surface_energy_balance(NF),
        kwargs...
    ) where {NF}
    land_model = Terrarium.LandModel(grid; surface_energy_balance, kwargs...)
    return land_model
end

"""
    land_simulation(grid::Terrarium.AbstractLandGrid; Δt = 300, initializers = (;), inputs, kwargs...)

Build a Terrarium `LandModel` on `grid` in the deferred-flux configuration (surface
turbulent fluxes computed by NumericalEarth), initialize it, and wrap it in an Oceananigans
`Simulation` ready to pass as the `land` component of an `AtmosphereLandModel` /
`EarthSystemModel`. Extra `kwargs` are forwarded to `Terrarium.LandModel` (e.g. `soil`,
`vegetation`, `snow`); `initializers` and `inputs` are forwarded to `Terrarium.initialize`.
"""
function NumericalEarth.Lands.land_simulation(
        grid::Terrarium.AbstractLandGrid{NF};
        Δt = 300.0,
        initializers = (;),
        inputs = Terrarium.InputSources(NF),
        kwargs...
    ) where {NF}
    model = land_model(grid; kwargs...)
    integrator = Terrarium.initialize(model; initializers, inputs)
    simulation = Simulation(integrator; Δt, verbose = false)
    # Adaptive diffusive time step for the soil column (inert while the coupler forces `Δt`,
    # active when the land `Simulation` is stepped on its own).
    conjure_time_step_wizard!(simulation; show_progress = false)
    return simulation
end

"""
    deferred_surface_energy_balance(NF)

Surface energy balance in which the skin temperature and the turbulent (sensible/latent)
fluxes are prescribed by the coupler while Terrarium owns albedo and radiation. The ground
heat flux is calculated internal from the prescribed fluxes (see `solve_surface_energy_balance!`
for `PrescribedSkinTemperature`).
"""
deferred_surface_energy_balance(NF) =
    Terrarium.SurfaceEnergyBalance(NF;
        skin_temperature = Terrarium.PrescribedSkinTemperature(NF),
        turbulent_fluxes = Terrarium.PrescribedTurbulentFluxes(NF),
        radiative_fluxes = Terrarium.DiagnosedRadiativeFluxes(NF),
        albedo           = Terrarium.DiagnosticAlbedo(NF)
    )
