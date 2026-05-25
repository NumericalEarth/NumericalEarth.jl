#####
##### `SlabLand` — composable slab land-surface component for
##### `EarthSystemModel` (and direct atmosphere-side coupling).
#####
##### A `SlabLand` is a 2D land model on an Oceananigans grid composed of
##### three closures along independent axes:
#####
#####   - `energy    :: AbstractEnergyBalance`     -- skin temperature
#####   - `hydrology :: AbstractHydrology`         -- water + moisture availability β ∈ [0, 1]
#####   - `surface   :: AbstractSurfaceProperties` -- roughness length
#####
##### State and flux fields are not allocated by the container: each
##### closure declares `prognostic_variables` / `flux_variables`, and the
##### constructor allocates exactly the union, keyed by `Symbol`. A pure
##### thermal slab with no hydrology has `state = (T = ...,)` only; a
##### Manabe-bucket slab adds `(:W, :moisture_availability)`.
#####
##### Some `state` entries are diagnostic in nature (e.g. β =
##### `state.moisture_availability` is a function of `state.W`). These
##### are still declared in `prognostic_variables` so the container
##### allocates a field for them, but are recomputed inside
##### `update_diagnostics!` (called at the end of `time_step!`) rather
##### than inside `step!`. The single-namedtuple convention keeps the
##### atmosphere-facing accessors simple.
#####

#####
##### Helpers — assemble state and flux NamedTuples from closure declarations
#####

@inline merge_unique(a::NTuple, b::NTuple) = (a..., (s for s in b if !(s in a))...)

# Generic field-tuple builder. Each closure declares its own keys via
# `keyfn(closure)` and constructs each field via `initfn(closure, name, grid)`.
function _build_closure_fields(keyfn, initfn, grid, energy, hydrology, surface)
    keys = merge_unique(merge_unique(keyfn(energy), keyfn(hydrology)), keyfn(surface))
    fields = map(keys) do name
        if name in keyfn(energy)
            initfn(energy, name, grid)
        elseif name in keyfn(hydrology)
            initfn(hydrology, name, grid)
        else
            initfn(surface, name, grid)
        end
    end
    return NamedTuple{keys}(fields)
end

"""
    build_state(grid, energy, hydrology, surface)

Allocate the prognostic-state `NamedTuple` for a `SlabLand`. Each key is
contributed by exactly one closure (or shared); duplicates are merged.
The per-symbol `Field` is built by `initial_state(closure, name, grid)`,
giving each closure a hook to install non-zero defaults.
"""
build_state(grid, energy, hydrology, surface) =
    _build_closure_fields(prognostic_variables, initial_state, grid, energy, hydrology, surface)

"""
    build_flux_accumulators(grid, energy, hydrology, surface)

Allocate the flux/forcing accumulator `NamedTuple` for a `SlabLand`.
Same shape as `build_state`. The coupler writes into these every step;
closures only read.
"""
build_flux_accumulators(grid, energy, hydrology, surface) =
    _build_closure_fields(flux_variables, initial_flux, grid, energy, hydrology, surface)

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
struct SlabLand{FT, G, Clk, S, F, E, H, Sfc} <: AbstractLand
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
`update_state!` refreshes diagnostics. State halos are filled at
the end so atmosphere kernels reading `state.T` see consistent values.

Closure-invocation order: `energy → hydrology`. Hydrology runs after
energy so future closures that close the energy budget through phase
change (snow melt, soil freeze/thaw) see the freshly updated `T`.

The clock is ticked first; subsequent closures see `land.clock.time =
t + Δt`. Time-dependent property providers should therefore evaluate
at the post-step time (consistent with `Atmospheres` and `Radiations`).
"""
function Oceananigans.TimeSteppers.time_step!(land::SlabLand, Δt)
    tick!(land.clock, Δt)
    time = land.clock.time

    step!(land.energy,    land.state, land.fluxes, land.surface, land.grid, Δt, time)
    step!(land.hydrology, land.state, land.fluxes, land.surface, land.grid, Δt, time)

    Oceananigans.TimeSteppers.update_state!(land)

    foreach(fill_halo_regions!, values(land.state))
    return nothing
end

"""
    update_state!(land::SlabLand)

Refresh closure-owned diagnostics. Order is `hydrology → surface →
energy`: hydrology produces wetness/β, the surface closure may consume
β (e.g. wetness-dependent albedo, LAI-aware roughness) before the
energy closure assembles an effective heat capacity diagnostic from
the freshly updated water storage.
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
declared by the land closures.

* `net_energy_flux` ← `-(𝒬ᵀ + 𝒬ᵛ)`, positive into the slab.
* `precipitation`   ← atmospheric rainfall flux + condensation (dew),
                      both positive into the slab.
* `evaporation`     ← net upward vapor flux, positive out of the slab.

Radiative contributions are added on top in
`apply_air_land_radiative_fluxes!`.
"""
function update_net_fluxes!(coupled_model, land::SlabLand)
    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    interface_fluxes = al_interface.fluxes
    fluxes = land.fluxes
    grid = land.grid
    arch = architecture(grid)

    Q = hasproperty(fluxes, :net_energy_flux) ? fluxes.net_energy_flux : nothing
    P = hasproperty(fluxes, :precipitation)   ? fluxes.precipitation   : nothing
    E = hasproperty(fluxes, :evaporation)     ? fluxes.evaporation     : nothing

    (isnothing(Q) && isnothing(P) && isnothing(E)) && return nothing

    # Prescribed atmospheric rainfall reaches the land via the atmosphere
    # exchanger (`Jʳⁿ` is allocated by `PrescribedAtmosphere`'s exchanger
    # state). When absent (e.g. radiatively decoupled or atmosphere
    # without precipitation), fall back to a `ZeroField`.
    atmos_state = coupled_model.interfaces.exchanger.atmosphere.state
    Jʳⁿ = hasproperty(atmos_state, :Jʳⁿ) ? atmos_state.Jʳⁿ : ZeroField()

    launch!(arch, grid, :xy, _assemble_slab_land_fluxes!, Q, P, E, interface_fluxes, Jʳⁿ)
    return nothing
end

@inline _maybe_write!(::Nothing, i, j, value) = nothing
@inline _maybe_write!(field, i, j, value) = @inbounds field[i, j, 1] = value

@kernel function _assemble_slab_land_fluxes!(Q, P, E, interface_fluxes, Jʳⁿ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        𝒬ᵀ = interface_fluxes.sensible_heat[i, j, 1]
        𝒬ᵛ = interface_fluxes.latent_heat[i, j, 1]
        Jᵛ = interface_fluxes.water_vapor[i, j, 1]
        rain = Jʳⁿ[i, j, 1]
    end
    _maybe_write!(Q, i, j, -(𝒬ᵀ + 𝒬ᵛ))
    _maybe_write!(P, i, j, rain + max(zero(Jᵛ), -Jᵛ))
    _maybe_write!(E, i, j, max(zero(Jᵛ),  Jᵛ))
end

interpolate_state!(exchanger, grid, ::SlabLand, coupled_model) = nothing

"""
    ComponentExchanger(land::SlabLand, grid)

Expose the generic atmosphere-facing SlabLand state through accessors:
skin temperature `T`, `moisture_availability`, and the two roughness lengths.
"""
function EarthSystemModels.InterfaceComputations.ComponentExchanger(land::SlabLand, grid)
    state = (T                         = surface_temperature(land),
             moisture_availability     = surface_wetness(land),
             momentum_roughness_length = momentum_roughness_length(land),
             scalar_roughness_length   = scalar_roughness_length(land))
    return ComponentExchanger(state, nothing)
end

EarthSystemModels.InterfaceComputations.initialize!(::ComponentExchanger, grid, ::SlabLand) = nothing
