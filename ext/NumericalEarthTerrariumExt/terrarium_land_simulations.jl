#####
##### `TerrariumLand` — adapter that plugs a Terrarium land model into
##### NumericalEarth's `EarthSystemModel` as the land component.
#####
##### The turbulent (and, optionally, radiative) surface fluxes are computed by
##### NumericalEarth's `InterfaceComputations` (Monin-Obukhov similarity theory);
##### the Terrarium land model receives them as prescribed inputs and closes the
##### ground heat flux internally (`PrescribedSkinTemperature` residual). Terrarium
##### owns the water budget (infiltration, runoff, evapotranspiration) and its own
##### albedo/radiation.
#####

"""
    TerrariumLand{I, G, F}

Wrap a `Terrarium.ModelIntegrator` so it satisfies NumericalEarth's land-component
contract. `grid` is the (Oceananigans) exchange grid the integrator's fields live on;
`fluxes` is a (possibly empty) `NamedTuple` of coupler-written flux accumulators, present
so `apply_air_land_radiative_fluxes!` can query it (Terrarium computes radiation locally,
so it is empty in the default configuration).
"""
struct TerrariumLand{I, G, F}
    integrator :: I
    grid       :: G
    fluxes     :: F
end

TerrariumLand(integrator::Terrarium.ModelIntegrator) =
    TerrariumLand(integrator, integrator.grid, NamedTuple())

"""
    deferred_surface_energy_balance(NF)

Surface energy balance in which the skin temperature and the turbulent (sensible/latent)
fluxes are prescribed by the coupler while Terrarium owns albedo and radiation. The ground
heat flux is closed as the residual `R_net + H_s + H_l` (see Terrarium's
`solve_surface_energy_balance!` for `PrescribedSkinTemperature`).
"""
deferred_surface_energy_balance(NF) = Terrarium.SurfaceEnergyBalance(NF;
    skin_temperature = Terrarium.PrescribedSkinTemperature(NF),
    turbulent_fluxes = Terrarium.PrescribedTurbulentFluxes(NF),
    radiative_fluxes = Terrarium.DiagnosedRadiativeFluxes(NF),
    albedo           = Terrarium.DiagnosticAlbedo(NF))

"""
    NumericalEarth.land_simulation(grid::Terrarium.AbstractLandGrid; initializers = (;), kwargs...)

Build a Terrarium `LandModel` on `grid` in the deferred-flux configuration (surface
turbulent fluxes computed by NumericalEarth), initialize it, and wrap it as a
`TerrariumLand` ready to pass as the `land` component of an `AtmosphereLandModel` /
`EarthSystemModel`. Extra `kwargs` are forwarded to `Terrarium.LandModel` (e.g. `soil`,
`vegetation`, `snow`); `initializers` are forwarded to `Terrarium.initialize`.
"""
function NumericalEarth.Lands.land_simulation(grid::Terrarium.AbstractLandGrid; initializers = (;), kwargs...)
    NF = eltype(grid)
    land_model = Terrarium.LandModel(grid;
                                     surface_energy_balance = deferred_surface_energy_balance(NF),
                                     kwargs...)
    integrator = Terrarium.initialize(land_model; initializers)
    return TerrariumLand(integrator)
end

Base.summary(::TerrariumLand) = "TerrariumLand (Terrarium.ModelIntegrator)"

Base.show(io::IO, land::TerrariumLand) = print(io, summary(land))

# Advance the Terrarium land by exactly the ESM time step `Δt`, sub-stepping so that no
# sub-step exceeds Terrarium's own `default_dt`. The prescribed coupling inputs are held
# fixed across the sub-steps (Terrarium's `InputSources` are empty, so `update_inputs!`
# does not overwrite the coupler-written fields).
function Oceananigans.TimeSteppers.time_step!(land::TerrariumLand, Δt)
    integrator = land.integrator
    Δt_land = min(Terrarium.default_dt(integrator), Δt)
    nsteps = max(1, round(Int, Δt / Δt_land))
    Δt_sub = Δt / nsteps
    for _ in 1:nsteps
        Terrarium.timestep!(integrator, Δt_sub)
    end
    return nothing
end
