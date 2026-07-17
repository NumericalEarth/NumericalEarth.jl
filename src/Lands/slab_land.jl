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
##### `time_step!`) rather than inside the per-closure update.
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

"""
    build_diagnostic_accumulators(grid, energy, hydrology)

Allocate the closure-owned diagnostics `NamedTuple` (e.g. `deep_liquid_flux`,
`surface_runoff`, `water_storage_tendency`). One closure can write a key the
other reads (hydrology publishes `water_storage_tendency`; energy reads it for
the variable-heat-capacity correction). Closures that need no diagnostics
declare an empty tuple via `diagnostic_variables(closure) = ()`.
"""
build_diagnostic_accumulators(grid, energy, hydrology) =
    build_closure_fields(diagnostic_variables, initial_diagnostic, grid, energy, hydrology)

"""
    build_extra_prognostics(grid, energy, hydrology)

Allocate the `NamedTuple` of *extra* prognostic `Field`s a closure declares beyond
the container's hardcoded `temperature`/`water_storage` (e.g. an
[`InterceptingHydrology`](@ref)'s `canopy_water_storage`), sized from each
closure's `prognostic_variables`. Empty (`(;)`) when no closure declares any.
"""
build_extra_prognostics(grid, energy, hydrology) =
    build_closure_fields(prognostic_variables, initial_prognostic, grid, energy, hydrology)

#####
##### Top-level struct
#####

"""
    SlabLand{FT, G, Clk, T, W, B, F, D, E, H}

A composable slab land-surface component. The default configuration —
`SlabEnergy + BucketHydrology` — is the classic bucket slab introduced by
[Manabe (1969)](@cite manabe1969climate). Replace
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
- `diagnostics`           : `NamedTuple` of closure-owned diagnostic `Field`s.
- `prognostic`            : `NamedTuple` of *extra* prognostic `Field`s a closure declares
                            beyond `temperature`/`water_storage` (e.g. a canopy water store);
                            `(;)` when none.
- `energy`                : an `AbstractEnergyBalance` (parameters).
- `hydrology`             : an `AbstractHydrology` (parameters).
"""
struct SlabLand{FT, G, Clk, T, W, B, F, D, P, E, H} <: AbstractLand
    grid          :: G
    clock         :: Clk
    temperature   :: T
    water_storage :: W
    saturation    :: B
    fluxes        :: F
    diagnostics   :: D
    prognostic    :: P
    energy        :: E
    hydrology     :: H
end

# Inner-style typed constructor capturing FT.
SlabLand{FT}(grid, clock, temperature, water_storage, saturation,
             fluxes, diagnostics, prognostic, energy, hydrology) where FT =
    SlabLand{FT, typeof(grid), typeof(clock),
             typeof(temperature), typeof(water_storage), typeof(saturation),
             typeof(fluxes), typeof(diagnostics), typeof(prognostic),
             typeof(energy), typeof(hydrology)}(
                 grid, clock, temperature, water_storage, saturation,
                 fluxes, diagnostics, prognostic, energy, hydrology)

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

    temperature   = CenterField(grid)
    water_storage = CenterField(grid)
    saturation    = CenterField(grid)
    fluxes        = build_flux_accumulators(grid, energy, hydrology)
    diagnostics   = build_diagnostic_accumulators(grid, energy, hydrology)
    prognostic    = build_extra_prognostics(grid, energy, hydrology)
    FT            = eltype(grid)
    return SlabLand{FT}(grid, clock, temperature, water_storage, saturation,
                        fluxes, diagnostics, prognostic, energy, hydrology)
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
              "├── saturation:            ", summary(land.saturation), '\n',
              "├── fluxes:                ", keys(land.fluxes), '\n',
              "├── diagnostics:           ", keys(land.diagnostics), '\n',
              "└── prognostic:            ", keys(land.prognostic))
end

#####
##### Time stepping
#####

"""
    time_step!(land::SlabLand, Δt)

Advance the slab by `Δt`. Each closure runs its own `time_step!`, then
`update_state!` refreshes diagnostics. Prognostic halos are filled at
the end so atmosphere kernels reading the surface state see consistent
values.

Closure-invocation order: `hydrology → energy`. Hydrology runs first so the
energy step reads the *same* step's `water_storage_tendency` (`dMˡᵃ/dt`) and
updated `Mˡᵃ`. The conservative `WaterCoupledEnergy` closure pairs the advective
energy carried by each mass flux against `cˡ(Tˡᵃ − Tᵣ) dMˡᵃ/dt`; that
cancellation — and hence energy conservation under water exchange at the slab
temperature — is exact only when both use the mass flux hydrology actually
applied this step.

The clock is ticked first; subsequent closures see `land.clock.time =
t + Δt`. Time-dependent property providers should therefore evaluate
at the post-step time (consistent with `Atmospheres` and `Radiations`).
"""
function Oceananigans.TimeSteppers.time_step!(land::SlabLand, Δt)
    tick!(land.clock, Δt)
    time = land.clock.time

    time_step!(land.hydrology, land, Δt, time)
    time_step!(land.energy,    land, Δt, time)

    Oceananigans.TimeSteppers.update_state!(land)

    fill_halo_regions!(land.temperature)
    fill_halo_regions!(land.water_storage)
    fill_halo_regions!(land.saturation)
    map(fill_halo_regions!, values(land.prognostic))
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
    set!(land::SlabLand; T=nothing, M=nothing, extra_prognostics...)

Set the slab's prognostic skin temperature `T` and water storage `M` and refresh
diagnostics in one call (so `saturation` is consistent with `M` afterward).
Either keyword can be omitted to leave that field untouched. Extra
closure-declared prognostics are set by name, e.g. `set!(land; canopy_water_storage = 0.2)`.
Each value is anything `Oceananigans.set!` accepts — a `Number`, `Field`,
`AbstractOperation`, function `(λ, φ, z)`, or array — so the initial state can stay
abstract, e.g. `set!(land; T = ERA5_T2m[1] - Γ * (z_land - z_era5), M = 75)`.
"""
function Oceananigans.set!(land::SlabLand; T=nothing, M=nothing, extra_prognostics...)
    isnothing(T) || Oceananigans.set!(land.temperature,   T)
    isnothing(M) || Oceananigans.set!(land.water_storage, M)
    for (name, value) in extra_prognostics
        Oceananigans.set!(getproperty(land.prognostic, name), value)
    end
    Oceananigans.TimeSteppers.update_state!(land)
    return land
end

#####
##### Prognostic-field accessors and checkpointing
#####

# Math-named NamedTuple of the prognostic fields (`saturation` is diagnostic).
# Bare symbols within the land namespace; the `ˡᵃ` superscript is reserved for
# cross-component contexts (coupling, the all-components notation table). Extra
# closure-declared prognostics (e.g. `canopy_water_storage`) are appended verbatim.
Oceananigans.prognostic_fields(land::SlabLand) =
    merge((; T = land.temperature, M = land.water_storage), land.prognostic)

function Oceananigans.prognostic_state(land::SlabLand)
    return (; clock         = prognostic_state(land.clock),
              temperature   = prognostic_state(land.temperature),
              water_storage = prognostic_state(land.water_storage),
              prognostic    = map(prognostic_state, land.prognostic))
end

function Oceananigans.restore_prognostic_state!(land::SlabLand, state)
    restore_prognostic_state!(land.clock,         state.clock)
    restore_prognostic_state!(land.temperature,   state.temperature)
    restore_prognostic_state!(land.water_storage, state.water_storage)
    extra = hasproperty(state, :prognostic) ? state.prognostic : (;)
    map(restore_prognostic_state!, values(land.prognostic), values(extra))
    update_state!(land)
    return land
end

Oceananigans.restore_prognostic_state!(land::SlabLand, ::Nothing) = land

#####
##### Container-level atmosphere-facing accessors
#####

EarthSystemModels.surface_temperature(land::SlabLand) = surface_temperature(land.energy, land)
surface_saturation(land::SlabLand) = saturation(land.hydrology, land)

# Prognostic canopy water store `Wᶜ` the interface reads to form the wet fraction
# `f_wet` (a `CanopyAirSpace` with interception). A `ZeroField` when no closure
# declares a store, so a dry canopy reads `Wᶜ = 0` and the interface reduces to the
# ordinary CAS.
surface_canopy_water_storage(land::SlabLand) =
    hasproperty(land.prognostic, :canopy_water_storage) ? land.prognostic.canopy_water_storage : ZeroField()

#####
##### EarthSystemModel interface — generic SlabLand coupling.
#####

"""
    update_net_fluxes!(coupled_model, land::SlabLand)

Consume atmosphere-land turbulent fluxes and populate the
`precipitation`, `evaporation`, `vapor_flux`, `surface_energy_flux`,
and `liquid_precipitation_flux` accumulators declared by the land closures.

* `surface_energy_flux`        ← `𝒬ᵀ + 𝒬ᵛ`, positive upward (out of the slab).
* `precipitation`              ← rainfall + condensation, positive into the slab.
* `evaporation`                ← positive part of upward vapor flux.
* `vapor_flux`                 ← signed `Jᵛ`, positive upward (consumed by
                                 `VariablySaturatedHydrology` and `WaterCoupledEnergy`).
* `liquid_precipitation_flux`  ← rainfall as Pˡ, positive downward (consumed by
                                 `VariablySaturatedHydrology`).

Radiative contributions are added on top in
`apply_air_land_radiative_fluxes!`.
"""
function EarthSystemModels.update_net_fluxes!(coupled_model, land::SlabLand)
    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    interface_fluxes = al_interface.fluxes
    fluxes = land.fluxes
    grid = land.grid
    arch = architecture(grid)

    P  = hasproperty(fluxes, :precipitation)               ? fluxes.precipitation               : nothing
    E  = hasproperty(fluxes, :evaporation)                 ? fluxes.evaporation                 : nothing
    Jv = hasproperty(fluxes, :vapor_flux)                  ? fluxes.vapor_flux                  : nothing
    Es = hasproperty(fluxes, :surface_energy_flux)         ? fluxes.surface_energy_flux         : nothing
    Pl = hasproperty(fluxes, :liquid_precipitation_flux)   ? fluxes.liquid_precipitation_flux   : nothing
    Cev = hasproperty(fluxes, :canopy_evaporation)         ? fluxes.canopy_evaporation          : nothing

    (isnothing(P) && isnothing(E) &&
     isnothing(Jv) && isnothing(Es) && isnothing(Pl)) && return nothing

    # Prescribed atmospheric rainfall reaches the land via the atmosphere
    # exchanger (`Jʳⁿ` is allocated by `PrescribedAtmosphere`'s exchanger
    # state). When absent (e.g. radiatively decoupled or atmosphere
    # without precipitation), fall back to a `ZeroField`.
    atmos_state = coupled_model.interfaces.exchanger.atmosphere.state
    Jʳⁿ = hasproperty(atmos_state, :Jʳⁿ) ? atmos_state.Jʳⁿ : ZeroField()

    # A CanopyAirSpace interface carries the skin→bulk ground heat flux `Gcond`; the
    # slab is then driven by conduction (`Jᴱs = −Gcond`) rather than by the total
    # turbulent flux, and radiation is internalized (no separate radiative add). Other
    # closures pass `nothing` and keep the turbulent `𝒬ᵀ + 𝒬ᵛ` budget. A CAS with
    # interception also carries the wet-canopy evaporation `E_wet`, which is split off
    # from the soil vapor sink (`Jᵛ → Jᵛ − E_wet`) and routed to the canopy store.
    Gᶜ = ground_heat_flux_field(al_interface.temperature)
    Ew = canopy_evaporation_field(al_interface.temperature)

    launch!(arch, grid, :xy, _assemble_slab_land_fluxes!,
            P, E, Jv, Es, Pl, Cev, interface_fluxes, Jʳⁿ, Gᶜ, Ew)
    return nothing
end

@inline ground_heat_flux_field(temperature) = nothing
@inline ground_heat_flux_field(temperature::NamedTuple) = temperature.ground_heat_flux

@inline canopy_evaporation_field(temperature) = nothing
@inline canopy_evaporation_field(temperature::NamedTuple) = temperature.canopy_evaporation

# Additive identity for the no-interception path so `Jᵛ − E_wet` stays `Jᵛ` bit-for-bit.
@inline canopy_evaporation_value(::Nothing, i, j) = false
@inline canopy_evaporation_value(Ew, i, j) = @inbounds Ew[i, j, 1]

@inline slab_energy_flux(::Nothing, 𝒬ᵀ, 𝒬ᵛ, i, j) = 𝒬ᵀ + 𝒬ᵛ
@inline slab_energy_flux(Gᶜ, 𝒬ᵀ, 𝒬ᵛ, i, j) = @inbounds -Gᶜ[i, j, 1]

@inline _maybe_write!(::Nothing, i, j, value) = nothing
@inline _maybe_write!(field, i, j, value) = @inbounds field[i, j, 1] = value

@kernel function _assemble_slab_land_fluxes!(P, E, Jv, Es, Pl, Cev, interface_fluxes, Jʳⁿ, Gᶜ, Ew)
    i, j = @index(Global, NTuple)
    @inbounds begin
        𝒬ᵀ = interface_fluxes.sensible_heat[i, j, 1]
        𝒬ᵛ = interface_fluxes.latent_heat[i, j, 1]
        Jᵛ = interface_fluxes.water_vapor[i, j, 1]
        rain = Jʳⁿ[i, j, 1]
    end
    E_wet = canopy_evaporation_value(Ew, i, j)
    _maybe_write!(Es,  i, j, slab_energy_flux(Gᶜ, 𝒬ᵀ, 𝒬ᵛ, i, j))
    _maybe_write!(P,   i, j, rain + max(zero(Jᵛ), -Jᵛ))
    _maybe_write!(E,   i, j, max(zero(Jᵛ),  Jᵛ))
    _maybe_write!(Jv,  i, j, Jᵛ - E_wet)   # soil evaporation + transpiration → Mˡᵃ sink
    _maybe_write!(Cev, i, j, E_wet)         # wet-canopy evaporation → Wᶜ sink (interception step)
    _maybe_write!(Pl,  i, j, rain)          # raw rain; interception step overwrites with throughfall
end

EarthSystemModels.interpolate_state!(exchanger, grid, ::SlabLand, coupled_model) = nothing

"""
    ComponentExchanger(land::SlabLand, grid)

Expose the generic atmosphere-facing SlabLand state: skin temperature `T` and
surface `saturation`. Aerodynamic roughness lengths belong to the atmosphere-land
flux closure (`atmosphere_land_fluxes`), not the land state.
"""
function EarthSystemModels.InterfaceComputations.ComponentExchanger(land::SlabLand, grid)
    state = (T                    = surface_temperature(land),
             saturation           = surface_saturation(land),
             canopy_water_storage = surface_canopy_water_storage(land))
    return ComponentExchanger(state, nothing)
end

EarthSystemModels.InterfaceComputations.initialize!(::ComponentExchanger, grid, ::SlabLand) = nothing
