"""
    land_model(grid::AbstractGrid; kwargs...)

Return a `Terrarium.LandModel` on the given land `grid` with components listed in `kwargs`. 
"""
function NumericalEarth.Lands.land_model(
        grid::AbstractGrid;
        surface_energy_balance = deferred_surface_energy_balance(eltype(grid)),
        kwargs...
    )
    # `Terrarium.LandModel` accepts either a plain Oceananigans grid (from which it builds its
    # multi-domain `LandGrid` via `create_land_grid`) or an `AbstractLandGrid` directly, so this
    # dispatches on the Oceananigans supertype rather than on `Terrarium.AbstractLandGrid`.
    land_model = Terrarium.LandModel(grid; surface_energy_balance, kwargs...)
    return land_model
end

"""
    land_simulation(grid::AbstractGrid; Δt = 300, initializers = (;), inputs, kwargs...)

Build a Terrarium `LandModel` on `grid` in the deferred-flux configuration (surface
turbulent fluxes computed by NumericalEarth), initialize it, and wrap it in an Oceananigans
`Simulation` ready to pass as the `land` component of an `AtmosphereLandModel` /
`EarthSystemModel`. Extra `kwargs` are forwarded to `Terrarium.LandModel` (e.g. `soil`,
`vegetation`, `snow`); `initializers` and `inputs` are forwarded to `Terrarium.initialize`.
"""
function NumericalEarth.Lands.land_simulation(
        grid::AbstractGrid;
        Δt = 300,
        initializers = (;),
        inputs = Terrarium.InputSources(eltype(grid)),
        kwargs...
    )
    model = land_model(grid; kwargs...)
    integrator = Terrarium.initialize(model; initializers, inputs)
    simulation = Simulation(integrator; Δt, verbose = false)
    # Adaptive diffusive time step for the soil column (inert while the coupler forces `Δt`,
    # active only when the land `Simulation` is stepped on its own).
    conjure_time_step_wizard!(simulation; show_progress = false)
    return simulation
end

"""
    deferred_surface_energy_balance(NF)

Surface energy balance with all four flux groups supplied by the coupler: skin temperature,
turbulent fluxes, radiative fluxes, and ground heat flux.

The ground heat flux carries the budget. NumericalEarth assembles the net surface energy flux
into `ground_heat_flux`, which `PrescribedGroundHeatFlux` declares as an input, so Terrarium does
not re-close the surface energy balance and the coupler's value reaches the soil-top boundary
condition unmodified.

Prescribing the radiative fluxes keeps the upwelling radiation Terrarium reports identical to the
one that drove that budget; diagnosing it locally would use a different albedo and emissivity.
The albedo is inert here, since it only feeds a prescribed computation.

!!! note
    `G = R_net + H_s + H_l` is not enforced by Terrarium. It holds because NumericalEarth
    assembles every term at one interface temperature within a single `update_state!`.
"""
deferred_surface_energy_balance(NF) =
    Terrarium.SurfaceEnergyBalance(NF;
        skin_temperature = Terrarium.PrescribedSkinTemperature(NF),
        turbulent_fluxes = Terrarium.PrescribedTurbulentFluxes(NF),
        radiative_fluxes = Terrarium.PrescribedRadiativeFluxes(NF),
        ground_heat_flux = Terrarium.PrescribedGroundHeatFlux(NF),
        albedo           = Terrarium.DiagnosticAlbedo(NF)
    )
