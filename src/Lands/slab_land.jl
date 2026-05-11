#####
##### `SlabLand` — composable slab land-surface component for
##### `EarthSystemModel` (and direct atmosphere-side coupling).
#####
##### A `SlabLand` is a 2D land model on an Oceananigans grid composed of
##### three closures along independent axes:
#####
#####   - `energy    :: AbstractEnergyBalance`     -- skin temperature
#####   - `hydrology :: AbstractHydrology`         -- water + wetness factor β
#####   - `surface   :: AbstractSurfaceProperties` -- albedo / emissivity / z₀
#####
##### State and flux fields are not allocated by the container: each
##### closure declares `prognostic_variables` / `flux_variables`, and the
##### constructor allocates exactly the union, keyed by `Symbol`. A pure
##### thermal slab with no hydrology has `state = (T = ...,)` only; a
##### Manabe-bucket slab adds `(:W,)`; the RUC composite adds the snow,
##### canopy, and soil-moisture/ice fields the RUC kernels need.
#####

#####
##### Helpers — assemble state and flux NamedTuples from closure declarations
#####

@inline _merge_unique(a::NTuple, b::NTuple) = (a..., (s for s in b if !(s in a))...)

function _all_prognostic_keys(energy, hydrology, surface)
    return _merge_unique(_merge_unique(prognostic_variables(energy),
                                       prognostic_variables(hydrology)),
                         prognostic_variables(surface))
end

function _all_flux_keys(energy, hydrology, surface)
    return _merge_unique(_merge_unique(flux_variables(energy),
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
    keys = _all_prognostic_keys(energy, hydrology, surface)
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
    keys = _all_flux_keys(energy, hydrology, surface)
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
    SlabLand{FT, G, Clk, S, F, E, H, Sfc, P}

A composable slab land-surface component. The default configuration —
`SlabEnergy + ManabeBucket + ConstantSurfaceProperties` — is the
classic Manabe-bucket slab. Replace any axis to upgrade fidelity:
`ForceRestoreEnergy` for a deep-soil thermal companion,
`BucketWithSnow` to add SWE, `SnowModifiedSurface` to make snow
modify albedo, etc.

# Fields
- `grid`       : Oceananigans grid (typically `(Nx, Ny, Flat)`).
- `clock`      : `Oceananigans.TimeSteppers.Clock`.
- `state`      : `NamedTuple` of prognostic `Field`s (keys per closure).
- `fluxes`     : `NamedTuple` of flux/forcing `Field`s the coupler writes.
- `energy`     : an `AbstractEnergyBalance`.
- `hydrology`  : an `AbstractHydrology`.
- `surface`    : an `AbstractSurfaceProperties`.
- `parameters` : container-level scalar physical constants.
"""
struct SlabLand{FT, G, Clk, S, F, E, H, Sfc, P}
    grid       :: G
    clock      :: Clk
    state      :: S
    fluxes     :: F
    energy     :: E
    hydrology  :: H
    surface    :: Sfc
    parameters :: P
end

# Inner-style typed constructor capturing FT.
SlabLand{FT}(grid, clock, state, fluxes, energy, hydrology, surface, parameters) where FT =
    SlabLand{FT, typeof(grid), typeof(clock),
             typeof(state), typeof(fluxes),
             typeof(energy), typeof(hydrology), typeof(surface),
             typeof(parameters)}(grid, clock, state, fluxes,
                                 energy, hydrology, surface, parameters)

"""
    SlabLand(grid;
             energy     = SlabEnergy(eltype(grid)),
             hydrology  = ManabeBucket(eltype(grid)),
             surface    = ConstantSurfaceProperties(eltype(grid)),
             parameters = SlabLandParameters(eltype(grid)),
             clock      = Clock{eltype(grid)}(time = 0))

Construct a `SlabLand` with the chosen closures. State and flux fields
are sized from each closure's declarations, then any closure-specific
initial state is applied through `initial_state(closure, name, grid)`.
"""
function SlabLand(grid;
                  energy     = SlabEnergy(eltype(grid)),
                  hydrology  = ManabeBucket(eltype(grid)),
                  surface    = ConstantSurfaceProperties(eltype(grid)),
                  parameters = SlabLandParameters(eltype(grid)),
                  clock      = Clock{eltype(grid)}(time = 0))

    state  = build_state(grid, energy, hydrology, surface)
    fluxes = build_flux_accumulators(grid, energy, hydrology, surface)
    _assert_surface_state_compatible(surface, state)
    FT     = eltype(grid)
    return SlabLand{FT}(grid, clock, state, fluxes,
                        energy, hydrology, surface, parameters)
end

# Surface closures that read cross-axis state Fields must assert their
# requirements are present in the assembled state. Default is a no-op;
# `SnowModifiedSurface` specialises this in its own file.
_assert_surface_state_compatible(::AbstractSurfaceProperties, ::Any) = nothing

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

    step!(land.energy,    land.state, land.fluxes, land.surface, land.parameters, land.grid, Δt)
    step!(land.hydrology, land.state, land.fluxes, land.surface, land.parameters, land.grid, Δt)

    Oceananigans.TimeSteppers.update_state!(land)

    foreach(fill_halo_regions!, values(land.state))
    return nothing
end

"""
    update_state!(land::SlabLand)

Refresh closure-owned diagnostics. The order is fixed
`hydrology → surface → energy`: surface-property blends (e.g. snow vs.
veg) read the just-updated `snowfrac` from hydrology; the energy
closure refreshes any cached fields last.
"""
function Oceananigans.TimeSteppers.update_state!(land::SlabLand)
    update_diagnostics!(land.hydrology, land.state, land.fluxes, land.surface, land.parameters, land.grid)
    update_diagnostics!(land.surface,   land.state, land.fluxes, land.surface, land.parameters, land.grid)
    update_diagnostics!(land.energy,    land.state, land.fluxes, land.surface, land.parameters, land.grid)
    return nothing
end

#####
##### Container-level atmosphere-facing accessors
#####

surface_temperature(land::SlabLand)       = surface_temperature(land.energy, land.state)
surface_wetness(land::SlabLand)           = wetness(land.hydrology, land.state, land.parameters)
albedo(land::SlabLand)                    = albedo(land.surface, land.state, land.parameters)
emissivity(land::SlabLand)                = emissivity(land.surface, land.state, land.parameters)
momentum_roughness_length(land::SlabLand) = momentum_roughness_length(land.surface, land.state, land.parameters)
scalar_roughness_length(land::SlabLand)   = scalar_roughness_length(land.surface, land.state, land.parameters)

#####
##### EarthSystemModel interface — generic SlabLand coupling.
#####

"""
    update_net_fluxes!(coupled_model, land::SlabLand)

Consume atmosphere-land turbulent fluxes and populate whichever generic
flux accumulators are declared by the land closures. `net_energy_flux`
is positive into the slab. `evaporation` and `precipitation` are split
from the vapor-flux sign convention used by the atmosphere-land interface.
"""
function update_net_fluxes!(coupled_model, land::SlabLand)
    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    interface_fluxes = al_interface.fluxes
    fluxes = land.fluxes
    grid = land.grid
    arch = architecture(grid)
    atmos_state = coupled_model.interfaces.exchanger.atmosphere.state

    if hasproperty(fluxes, :net_energy_flux)
        launch!(arch, grid, :xy, _assemble_slab_land_net_energy_flux!,
                fluxes.net_energy_flux, interface_fluxes)
    end

    # Legacy single-bucket precipitation accumulator (positive water_vapor → down).
    if hasproperty(fluxes, :precipitation)
        launch!(arch, grid, :xy, _assemble_slab_land_precipitation!,
                fluxes.precipitation, interface_fluxes)
    end

    # Legacy single bare-soil evaporation, only when the BucketWithSnow
    # three-way split is not in play.
    if hasproperty(fluxes, :evaporation) && !hasproperty(fluxes, :sublimation)
        launch!(arch, grid, :xy, _assemble_slab_land_evaporation!,
                fluxes.evaporation, interface_fluxes)
    end

    # BucketWithSnow path: split precipitation by air-temperature threshold.
    if hasproperty(fluxes, :rainfall_rate) && hasproperty(fluxes, :snowfall_rate) &&
       hasproperty(atmos_state, :T)
        launch!(arch, grid, :xy, _split_precip_by_temperature!,
                fluxes.rainfall_rate, fluxes.snowfall_rate,
                interface_fluxes.water_vapor, atmos_state.T)
    end

    # BucketWithSnow path: split vapor flux into evaporation + sublimation
    # + transpiration using snow_fraction and surface.vegfrac.
    if hasproperty(fluxes, :sublimation) && hasproperty(fluxes, :transpiration) &&
       hasproperty(fluxes, :evaporation)
        launch!(arch, grid, :xy, _split_vapor_flux!,
                fluxes.evaporation, fluxes.sublimation, fluxes.transpiration,
                interface_fluxes.water_vapor,
                land.state.snow_fraction, land.surface.vegfrac)
    end

    # Atmospheric forcing copies for the Jarvis stress functions.
    if hasproperty(fluxes, :air_temperature) && hasproperty(atmos_state, :T)
        launch!(arch, grid, :xy, _copy_atmos_field!,
                fluxes.air_temperature, atmos_state.T)
    end
    if hasproperty(fluxes, :air_humidity) && hasproperty(atmos_state, :q)
        launch!(arch, grid, :xy, _copy_atmos_field!,
                fluxes.air_humidity, atmos_state.q)
    end
    if hasproperty(fluxes, :surface_pressure) && hasproperty(atmos_state, :p)
        launch!(arch, grid, :xy, _copy_atmos_field!,
                fluxes.surface_pressure, atmos_state.p)
    end
    if hasproperty(fluxes, :solar_irradiance) && hasproperty(atmos_state, :ℐꜜˢʷ)
        launch!(arch, grid, :xy, _copy_atmos_field!,
                fluxes.solar_irradiance, atmos_state.ℐꜜˢʷ)
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

@kernel function _split_vapor_flux!(E, Sub, Tt, water_vapor, snow_fraction, vegfrac)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(E)
        E_total = max(zero(FT), -water_vapor[i, j, 1])    # upward positive
        f_sn = snow_fraction[i, j, 1]
        vf   = vegfrac[i, j, 1]
        Sub[i, j, 1] = E_total * f_sn
        E[i, j, 1]   = E_total * (one(FT) - f_sn) * (one(FT) - vf)
        Tt[i, j, 1]  = E_total * (one(FT) - f_sn) * vf
    end
end

@kernel function _split_precip_by_temperature!(P_rain, P_snow, water_vapor, T_air)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(P_rain)
        # Atmosphere convention: positive `water_vapor` is downward (precipitation).
        P = max(zero(FT), water_vapor[i, j, 1])
        if T_air[i, j, 1] ≤ FT(273.15)
            P_snow[i, j, 1] = P
            P_rain[i, j, 1] = zero(FT)
        else
            P_snow[i, j, 1] = zero(FT)
            P_rain[i, j, 1] = P
        end
    end
end

@kernel function _copy_atmos_field!(dst, src)
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j, 1] = src[i, j, 1]
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
