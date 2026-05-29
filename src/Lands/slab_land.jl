#####
##### `SlabLand` — composable slab land-surface component for
##### `EarthSystemModel` (and direct atmosphere-side coupling).
#####
##### A `SlabLand` is a 2D land model on an Oceananigans grid composed of
##### two closures along independent axes:
#####
#####   - `energy    :: AbstractEnergyBalance`     -- skin temperature
#####   - `hydrology :: AbstractHydrology`         -- water + moisture availability β ∈ [0, 1]
#####
##### Aerodynamic roughness lengths are a property of the atmosphere-land flux
##### closure (`atmosphere_land_fluxes`), not of the land model.
#####
##### The prognostic `temperature` and `water_storage` fields and the
##### diagnostic `saturation` field live directly on the
##### container (top-level access: `land.temperature`, `land.water_storage`,
##### `land.saturation`). The flux accumulators the coupler
##### writes are sized from each closure's `flux_variables` declaration and
##### grouped in the `fluxes` NamedTuple.
#####
##### `saturation` (𝒮) is diagnostic — recomputed from
##### `water_storage` inside `update_diagnostics!` (called at the end of
##### `time_step!`) rather than inside `step!`.
#####

#####
##### Helpers — assemble the flux-accumulator NamedTuple from closure declarations
#####

@inline merge_unique(a::NTuple, b::NTuple) = (a..., (s for s in b if !(s in a))...)

# Generic field-tuple builder. Each closure declares its own keys via
# `keyfn(closure)` and constructs each field via `initfn(closure, name, grid)`.
function build_closure_fields(keyfn, initfn, grid, energy, hydrology)
    keys = merge_unique(keyfn(energy), keyfn(hydrology))
    fields = map(keys) do name
        if name in keyfn(energy)
            initfn(energy, name, grid)
        else
            initfn(hydrology, name, grid)
        end
    end
    return NamedTuple{keys}(fields)
end

"""
    build_flux_accumulators(grid, energy, hydrology)

Allocate the flux/forcing accumulator `NamedTuple` for a `SlabLand`.
The coupler writes into these every step; closures only read.
"""
build_flux_accumulators(grid, energy, hydrology) =
    build_closure_fields(flux_variables, initial_flux, grid, energy, hydrology)

#####
##### Top-level struct
#####

"""
    SlabLand{FT, G, Clk, T, W, B, F, E, H}

A composable slab land-surface component. The default configuration —
`SlabEnergy + BucketHydrology` — is the classic Manabe-bucket slab. Replace
either axis to swap in a different energy or hydrology closure. Aerodynamic
roughness lengths are a property of the atmosphere-land flux closure
(`atmosphere_land_fluxes`), not of the land model.

# Fields
- `grid`                  : Oceananigans grid (typically `(Nx, Ny, Flat)`).
- `clock`                 : `Oceananigans.TimeSteppers.Clock`.
- `temperature`           : prognostic bulk land temperature `T` (K).
- `water_storage`         : prognostic land water mass per area `Mˡᵃ` (kg m⁻²).
- `saturation`            : diagnostic surface saturation `𝒮 = Mˡᵃ/Mˡᵃ⁺ ∈ [0, 1]` (–).
- `fluxes`                : `NamedTuple` of flux/forcing `Field`s the coupler writes.
- `energy`                : an `AbstractEnergyBalance` (parameters).
- `hydrology`             : an `AbstractHydrology` (parameters).
"""
struct SlabLand{FT, G, Clk, T, W, B, F, E, H} <: AbstractLand
    grid                  :: G
    clock                 :: Clk
    temperature           :: T
    water_storage         :: W
    saturation :: B
    fluxes                :: F
    energy                :: E
    hydrology             :: H
end

# Inner-style typed constructor capturing FT.
SlabLand{FT}(grid, clock, temperature, water_storage, saturation,
             fluxes, energy, hydrology) where FT =
    SlabLand{FT, typeof(grid), typeof(clock),
             typeof(temperature), typeof(water_storage), typeof(saturation),
             typeof(fluxes), typeof(energy), typeof(hydrology)}(
                 grid, clock, temperature, water_storage, saturation,
                 fluxes, energy, hydrology)

"""
    SlabLand(grid;
             energy    = SlabEnergy(eltype(grid)),
             hydrology = BucketHydrology(eltype(grid)),
             clock     = Clock{eltype(grid)}(time = 0))

Construct a `SlabLand` with the chosen closures. The prognostic
`temperature` and `water_storage` fields and the diagnostic
`saturation` field are allocated on `grid`; the flux
accumulators the coupler writes are sized from each closure's
`flux_variables` declaration.
"""
function SlabLand(grid;
                  energy    = SlabEnergy(eltype(grid)),
                  hydrology = BucketHydrology(eltype(grid)),
                  clock     = Clock{eltype(grid)}(time = 0))

    temperature           = CenterField(grid)
    water_storage         = CenterField(grid)
    saturation = CenterField(grid)
    fluxes                = build_flux_accumulators(grid, energy, hydrology)
    FT                    = eltype(grid)
    return SlabLand{FT}(grid, clock, temperature, water_storage, saturation,
                        fluxes, energy, hydrology)
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
              "├── grid:                  ", summary(land.grid), '\n',
              "├── energy:                ", summary(land.energy), '\n',
              "├── hydrology:             ", summary(land.hydrology), '\n',
              "├── temperature:           ", summary(land.temperature), '\n',
              "├── water_storage:         ", summary(land.water_storage), '\n',
              "├── saturation: ", summary(land.saturation), '\n',
              "└── fluxes:                ", keys(land.fluxes))
end

#####
##### Time stepping
#####

"""
    time_step!(land::SlabLand, Δt)

Advance the slab by `Δt`. Each closure runs its own `step!`, then
`update_state!` refreshes diagnostics. Prognostic halos are filled at
the end so atmosphere kernels reading the surface state see consistent
values.

Closure-invocation order: `energy → hydrology`. Hydrology runs after
energy so future closures that close the energy budget through phase
change (snow melt, soil freeze/thaw) see the freshly updated `temperature`.

The clock is ticked first; subsequent closures see `land.clock.time =
t + Δt`. Time-dependent property providers should therefore evaluate
at the post-step time (consistent with `Atmospheres` and `Radiations`).
"""
function Oceananigans.TimeSteppers.time_step!(land::SlabLand, Δt)
    tick!(land.clock, Δt)
    time = land.clock.time

    step!(land.energy,    land, Δt, time)
    step!(land.hydrology, land, Δt, time)

    Oceananigans.TimeSteppers.update_state!(land)

    fill_halo_regions!(land.temperature)
    fill_halo_regions!(land.water_storage)
    fill_halo_regions!(land.saturation)
    return nothing
end

"""
    update_state!(land::SlabLand)

Refresh closure-owned diagnostics. Order is `hydrology → energy`: hydrology
produces `saturation` (𝒮) before the energy closure assembles any
heat-capacity diagnostic from the freshly updated water storage.
"""
function Oceananigans.TimeSteppers.update_state!(land::SlabLand)
    update_diagnostics!(land.hydrology, land)
    update_diagnostics!(land.energy,    land)
    return nothing
end

"""
    set!(land::SlabLand; T=nothing, M=nothing)

Set the slab's prognostic skin temperature `T` and water storage `M` and refresh
diagnostics in one call (so `saturation` is consistent with `M` afterward).
Either keyword can be omitted to leave that field untouched. Each value is
anything `Oceananigans.set!` accepts — a `Number`, `Field`, `AbstractOperation`,
function `(λ, φ, z)`, or array — so the initial state can stay abstract, e.g.
`set!(land; T = ERA5_T2m[1] - Γ * (z_land - z_era5), M = 75)`.
"""
function Oceananigans.set!(land::SlabLand; T=nothing, M=nothing)
    isnothing(T) || Oceananigans.set!(land.temperature,   T)
    isnothing(M) || Oceananigans.set!(land.water_storage, M)
    Oceananigans.TimeSteppers.update_state!(land)
    return land
end

#####
##### Container-level atmosphere-facing accessors
#####

surface_temperature(land::SlabLand)       = surface_temperature(land.energy, land)
surface_saturation(land::SlabLand)           = saturation(land.hydrology, land)

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

Expose the generic atmosphere-facing SlabLand state: skin temperature `T` and
surface `saturation`. Aerodynamic roughness lengths belong to the atmosphere-land
flux closure (`atmosphere_land_fluxes`), not the land state.
"""
function EarthSystemModels.InterfaceComputations.ComponentExchanger(land::SlabLand, grid)
    state = (T          = surface_temperature(land),
             saturation = surface_saturation(land))
    return ComponentExchanger(state, nothing)
end

EarthSystemModels.InterfaceComputations.initialize!(::ComponentExchanger, grid, ::SlabLand) = nothing
