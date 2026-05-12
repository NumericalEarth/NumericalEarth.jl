#####
##### `SlabLand` — composable slab land-surface component for
##### `EarthSystemModel` (and direct atmosphere-side coupling).
#####
##### A `SlabLand` is a 2D land model on an Oceananigans grid composed of
##### three closures along independent axes:
#####
#####   - `energy    :: AbstractEnergyBalance`     -- skin temperature
#####   - `hydrology :: AbstractHydrology`         -- water + wetness factor β
#####   - `surface   :: AbstractSurfaceProperties` -- roughness length
#####
##### State and flux fields are not allocated by the container: each
##### closure declares `prognostic_variables` / `flux_variables`, and the
##### constructor allocates exactly the union, keyed by `Symbol`. A pure
##### thermal slab with no hydrology has `state = (T = ...,)` only; a
##### Manabe-bucket slab adds `(:W, :moisture_availability)`.
#####

#####
##### Helpers — assemble state and flux NamedTuples from closure declarations
#####

@inline merge_unique(a::NTuple, b::NTuple) = (a..., (s for s in b if !(s in a))...)

function all_prognostic_keys(energy, hydrology, surface)
    return merge_unique(merge_unique(prognostic_variables(energy),
                                     prognostic_variables(hydrology)),
                        prognostic_variables(surface))
end

function all_flux_keys(energy, hydrology, surface)
    return merge_unique(merge_unique(flux_variables(energy),
                                     flux_variables(hydrology)),
                        flux_variables(surface))
end

"""
    build_state(grid, energy, hydrology, surface)

Allocate the prognostic-state `NamedTuple` for a `SlabLand`. Each key is
contributed by exactly one closure (or shared); duplicates are merged.
The per-symbol `Field` is built by `initial_state(closure, name, grid)`,
giving each closure a hook to install non-zero defaults.
"""
function build_state(grid, energy, hydrology, surface)
    keys = all_prognostic_keys(energy, hydrology, surface)
    fields = map(keys) do name
        if name in prognostic_variables(energy)
            return initial_state(energy, name, grid)
        elseif name in prognostic_variables(hydrology)
            return initial_state(hydrology, name, grid)
        else
            return initial_state(surface, name, grid)
        end
    end
    return NamedTuple{keys}(fields)
end

"""
    build_flux_accumulators(grid, energy, hydrology, surface)

Allocate the flux/forcing accumulator `NamedTuple` for a `SlabLand`.
Same shape as `build_state`. The coupler writes into these every step;
closures only read.
"""
function build_flux_accumulators(grid, energy, hydrology, surface)
    keys = all_flux_keys(energy, hydrology, surface)
    fields = map(keys) do name
        if name in flux_variables(energy)
            return initial_flux(energy, name, grid)
        elseif name in flux_variables(hydrology)
            return initial_flux(hydrology, name, grid)
        else
            return initial_flux(surface, name, grid)
        end
    end
    return NamedTuple{keys}(fields)
end

#####
##### Top-level struct
#####

"""
    SlabLand{FT, G, Clk, S, F, E, H, Sfc}

A composable slab land-surface component. The default configuration —
`SlabEnergy + BucketHydrology + ConstantSurfaceProperties` — is the
classic Manabe-bucket slab. Replace any axis to swap in a different
energy, hydrology, or surface-property closure.

# Fields
- `grid`      : Oceananigans grid (typically `(Nx, Ny, Flat)`).
- `clock`     : `Oceananigans.TimeSteppers.Clock`.
- `state`     : `NamedTuple` of prognostic `Field`s (keys per closure).
- `fluxes`    : `NamedTuple` of flux/forcing `Field`s the coupler writes.
- `energy`    : an `AbstractEnergyBalance`.
- `hydrology` : an `AbstractHydrology`.
- `surface`   : an `AbstractSurfaceProperties`.
"""
struct SlabLand{FT, G, Clk, S, F, E, H, Sfc}
    grid      :: G
    clock     :: Clk
    state     :: S
    fluxes    :: F
    energy    :: E
    hydrology :: H
    surface   :: Sfc
end

# Inner-style typed constructor capturing FT.
SlabLand{FT}(grid, clock, state, fluxes, energy, hydrology, surface) where FT =
    SlabLand{FT, typeof(grid), typeof(clock),
             typeof(state), typeof(fluxes),
             typeof(energy), typeof(hydrology), typeof(surface)}(grid, clock, state, fluxes,
                                                                 energy, hydrology, surface)

"""
    SlabLand(grid;
             energy    = SlabEnergy(eltype(grid)),
             hydrology = BucketHydrology(eltype(grid)),
             surface   = ConstantSurfaceProperties(eltype(grid)),
             clock     = Clock{eltype(grid)}(time = 0))

Construct a `SlabLand` with the chosen closures. State and flux fields
are sized from each closure's declarations, then any closure-specific
initial state is applied through `initial_state(closure, name, grid)`.
"""
function SlabLand(grid;
                  energy    = SlabEnergy(eltype(grid)),
                  hydrology = BucketHydrology(eltype(grid)),
                  surface   = ConstantSurfaceProperties(eltype(grid)),
                  clock     = Clock{eltype(grid)}(time = 0))

    state  = build_state(grid, energy, hydrology, surface)
    fluxes = build_flux_accumulators(grid, energy, hydrology, surface)
    FT     = eltype(grid)
    return SlabLand{FT}(grid, clock, state, fluxes,
                        energy, hydrology, surface)
end

Base.eltype(::SlabLand{FT}) where FT = FT

function Base.summary(land::SlabLand{FT}) where FT
    A = nameof(typeof(architecture(land.grid)))
    G = nameof(typeof(land.grid))
    return string("SlabLand{$FT, $A, $G}",
                  "(time = ", prettytime(land.clock.time),
                  ", iteration = ", land.clock.iteration, ")")
end

function Base.show(io::IO, land::SlabLand)
    print(io, summary(land), '\n',
              "├── grid:       ", summary(land.grid), '\n',
              "├── energy:     ", summary(land.energy), '\n',
              "├── hydrology:  ", summary(land.hydrology), '\n',
              "├── surface:    ", summary(land.surface), '\n',
              "├── state:      ", keys(land.state), '\n',
              "└── fluxes:     ", keys(land.fluxes))
end

#####
##### Time stepping
#####

"""
    time_step!(land::SlabLand, Δt)

Advance the slab by `Δt`. Each closure runs its own `step!`, then
`update_state!` refreshes any cached diagnostics (e.g. snow-cover
fraction, surface albedo, Jarvis resistance). State halos are filled at
the end so atmosphere kernels reading `state.T` see consistent values.

Order of closure invocations is fixed: `energy → hydrology`. Hydrology
runs after energy so closures that close the energy budget through
phase change (snow melt, soil freeze/thaw) see the freshly updated `T`.
"""
function Oceananigans.TimeSteppers.time_step!(land::SlabLand, Δt)
    tick!(land.clock, Δt)

    step!(land.energy,    land.state, land.fluxes, land.surface, land.grid, Δt)
    step!(land.hydrology, land.state, land.fluxes, land.surface, land.grid, Δt)

    Oceananigans.TimeSteppers.update_state!(land)

    foreach(fill_halo_regions!, values(land.state))
    return nothing
end

"""
    update_state!(land::SlabLand)

Refresh closure-owned diagnostics, in the order
`hydrology → surface → energy`.
"""
function Oceananigans.TimeSteppers.update_state!(land::SlabLand)
    update_diagnostics!(land.hydrology, land.state, land.fluxes, land.surface, land.grid)
    update_diagnostics!(land.surface,   land.state, land.fluxes, land.surface, land.grid)
    update_diagnostics!(land.energy,    land.state, land.fluxes, land.surface, land.grid)
    return nothing
end

#####
##### Container-level atmosphere-facing accessors
#####

surface_temperature(land::SlabLand)       = surface_temperature(land.energy, land.state)
surface_wetness(land::SlabLand)           = wetness(land.hydrology, land.state)
momentum_roughness_length(land::SlabLand) = momentum_roughness_length(land.surface, land.state)
scalar_roughness_length(land::SlabLand)   = scalar_roughness_length(land.surface, land.state)

#####
##### EarthSystemModel interface — generic SlabLand coupling.
#####

"""
    update_net_fluxes!(coupled_model, land::SlabLand)

Consume atmosphere-land turbulent fluxes and populate the
`net_energy_flux`, `precipitation`, and `evaporation` accumulators
declared by the land closures. `net_energy_flux` is positive into the
slab.
"""
function update_net_fluxes!(coupled_model, land::SlabLand)
    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    interface_fluxes = al_interface.fluxes
    fluxes = land.fluxes
    grid = land.grid
    arch = architecture(grid)

    if hasproperty(fluxes, :net_energy_flux)
        launch!(arch, grid, :xy, _assemble_slab_land_net_energy_flux!,
                fluxes.net_energy_flux, interface_fluxes)
    end

    if hasproperty(fluxes, :precipitation)
        launch!(arch, grid, :xy, _assemble_slab_land_precipitation!,
                fluxes.precipitation, interface_fluxes)
    end

    if hasproperty(fluxes, :evaporation)
        launch!(arch, grid, :xy, _assemble_slab_land_evaporation!,
                fluxes.evaporation, interface_fluxes)
    end

    return nothing
end

@kernel function _assemble_slab_land_net_energy_flux!(Q, interface_fluxes)
    i, j = @index(Global, NTuple)
    @inbounds begin
        𝒬ᵀ = interface_fluxes.sensible_heat[i, j, 1]
        𝒬ᵛ = interface_fluxes.latent_heat[i, j, 1]
        Q[i, j, 1] = -(𝒬ᵀ + 𝒬ᵛ)
    end
end

@kernel function _assemble_slab_land_evaporation!(E, interface_fluxes)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(E)
        Jᵛ = interface_fluxes.water_vapor[i, j, 1]
        E[i, j, 1] = max(zero(FT), -Jᵛ)
    end
end

@kernel function _assemble_slab_land_precipitation!(P, interface_fluxes)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(P)
        Jᵛ = interface_fluxes.water_vapor[i, j, 1]
        P[i, j, 1] = max(zero(FT), Jᵛ)
    end
end

interpolate_state!(exchanger, grid, ::SlabLand, coupled_model) = nothing

"""
    ComponentExchanger(land::SlabLand, grid)

Expose the generic atmosphere-facing SlabLand state through accessors:
skin temperature `T`, `moisture_availability`, and `roughness_length`.
"""
function ComponentExchanger(land::SlabLand, grid)
    state = (T                     = surface_temperature(land),
             moisture_availability = surface_wetness(land),
             roughness_length      = momentum_roughness_length(land))
    return ComponentExchanger(state, nothing)
end

initialize!(::ComponentExchanger, grid, ::SlabLand) = nothing
