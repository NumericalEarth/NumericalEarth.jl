#####
##### `RucSlabLand` — convenience constructor + RUC-specific glue.
#####
##### Builds a `SlabLand` with the RUC trio of closures
##### (`RucEnergy`, `RucHydrology`, `RucSurfaceProperties`) using a
##### single `RucSlabLandParameters` bag. The result is a `SlabLand`,
##### so the `time_step!` / `update_state!` / accessor machinery
##### defined on the container applies unchanged. The handful of
##### RUC-specific glue routines (`set!`, `ComponentExchanger`,
##### `update_net_fluxes!`, `prognostic_state` /
##### `restore_prognostic_state!`) dispatch on the concrete closure
##### types so they don't pollute the generic `SlabLand` path.
#####
##### Implements the slab-compatible subset of the RUC LSM
##### (Smirnova et al. 1997, 2016) plus complementary parameterizations
##### from ISBA, Noilhan-Planton (1989), and Mahfouf-Noilhan (1991). See
##### the closure files for the kernels and references.
#####

#####
##### Convenience constructor
#####

"""
    RucSlabLand(grid;
                FT         = eltype(grid),
                parameters = RucSlabLandParameters(FT),
                clock      = Clock{FT}(time = 0))

Build a `SlabLand` composed of `RucEnergy`, `RucHydrology`, and
`RucSurfaceProperties` — the slab-compatible subset of the RUC LSM.
The atmosphere reads `state.T` (skin temperature),
`state.moisture_availability` (β), and
`surface.{albedo, emissivity, roughness_length}` through the
`ComponentExchanger`-aliased fields in `EarthSystemModels`.

Coupler-supplied flux fields (each a `Field`):

- `fluxes.temperature_flux`         ≡ `Jᵀ_g` [K m s⁻¹]
- `fluxes.canopy_temperature_flux`  ≡ `Jᵀ_c` [W m⁻²]
- `fluxes.snowfall_rate`            [m s⁻¹] LWE
- `fluxes.rainfall_rate`            [m s⁻¹]
- `fluxes.moisture_flux`            [kg m⁻² s⁻¹], positive up
- `fluxes.canopy_evaporation`       [kg m⁻² s⁻¹]
- `fluxes.transpiration`            [kg m⁻² s⁻¹]
- `fluxes.solar_irradiance`         [W m⁻²]
- `fluxes.air_temperature`          [K]
- `fluxes.air_humidity`             [kg kg⁻¹]
- `fluxes.surface_pressure`         [hPa]

Initialise with `set!(land; T=..., θ=..., snwe=..., …)` after
construction.
"""
function RucSlabLand(grid;
                     FT = eltype(grid),
                     parameters = RucSlabLandParameters(FT),
                     clock = Clock{FT}(time = 0))

    energy    = RucEnergy(parameters)
    hydrology = RucHydrology(parameters)
    surface   = RucSurfaceProperties(grid, parameters)

    return SlabLand(grid; energy, hydrology, surface, parameters, clock)
end

#####
##### `set!` for the RUC composition (and any SlabLand that exposes
##### those state / surface keys).
#####

"""
    set!(land::SlabLand; T=…, Tc=…, θ=…, θ_ice=…, snwe=…, snhei=…, rhosn=…, swl=…, vegfrac=…, lai=…)

Initialise prognostic state and per-cell vegetation-class fields. State
keys map to `land.state.<name>`; `vegfrac` and `lai` map to
`land.surface.<name>` because they are owned by the surface-property
closure.

`Tc` defaults to `T` when only `T` is supplied — preserves the legacy
`RucSlabLand` ergonomics.
"""
function Oceananigans.set!(land::SlabLand;
                           T = nothing,
                           Tc = nothing,
                           θ = nothing,
                           θ_ice = nothing,
                           snwe = nothing,
                           snhei = nothing,
                           rhosn = nothing,
                           swl = nothing,
                           vegfrac = nothing,
                           lai = nothing)
    state   = land.state
    surface = land.surface

    !isnothing(T)              && haskey(state, :T)   && set!(state.T,   T)
    !isnothing(Tc)             && haskey(state, :Tc)  && set!(state.Tc,  Tc)
    isnothing(Tc) && !isnothing(T) && haskey(state, :Tc) && set!(state.Tc, T)
    !isnothing(θ)              && haskey(state, :θ)     && set!(state.θ,     θ)
    !isnothing(θ_ice)          && haskey(state, :θ_ice) && set!(state.θ_ice, θ_ice)
    !isnothing(snwe)           && haskey(state, :snwe)  && set!(state.snwe,  snwe)
    !isnothing(snhei)          && haskey(state, :snhei) && set!(state.snhei, snhei)
    !isnothing(rhosn)          && haskey(state, :rhosn) && set!(state.rhosn, rhosn)
    !isnothing(swl)            && haskey(state, :swl)   && set!(state.swl,   swl)
    if !isnothing(vegfrac) && hasproperty(surface, :vegfrac)
        set!(surface.vegfrac, vegfrac)
    end
    if !isnothing(lai) && hasproperty(surface, :lai)
        set!(surface.lai, lai)
    end
    return nothing
end

#####
##### EarthSystemModel interface — RUC-composition specialisations.
#####
##### These dispatch on the concrete closure types so they don't pollute
##### the generic SlabLand path.
#####

"""
    update_net_fluxes!(coupled_model, land::SlabLand{...,<:RucEnergy,...})

Consume atmosphere-land turbulent fluxes from
`compute_atmosphere_land_fluxes!` and write them into the slab's flux
fields.

Sign convention for `interface_fluxes`:

  𝒬ᵀ, 𝒬ᵛ : atmospheric net energy gain (W m⁻²); negative when the
            atmosphere is cooled-by-surface, positive when atmosphere
            loses energy to surface.
  Jᵛ      : atmospheric vapor flux; negative when vapor flows from
            atmosphere to surface (condensation).

Slab-side (surface positive upward):

  Q_net (into ground)     = -(𝒬ᵀ + 𝒬ᵛ)
  Jᵀ (slab boundary)       = -Q_net / (ρ · c)
  F_v (slab evap, upward)  = -Jᵛ
"""
function update_net_fluxes!(coupled_model,
                            land::SlabLand{FT, G, Clk, S, F,
                                           <:RucEnergy, <:RucHydrology,
                                           <:RucSurfaceProperties, P}) where {FT, G, Clk, S, F, P}
    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    fluxes = al_interface.fluxes
    grid   = land.grid
    arch   = architecture(grid)
    e      = land.energy
    ρcₛ    = e.density * e.heat_capacity   # J m⁻³ K⁻¹

    launch!(arch, grid, :xy, _assemble_slab_land_fluxes!,
            land.fluxes.temperature_flux,
            land.fluxes.moisture_flux,
            fluxes, ρcₛ)
    return nothing
end

@kernel function _assemble_slab_land_fluxes!(Jᵀ, M, fluxes, ρcₛ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        𝒬ᵀ = fluxes.sensible_heat[i, j, 1]
        𝒬ᵛ = fluxes.latent_heat[i, j, 1]
        Jᵛ = fluxes.water_vapor[i, j, 1]

        Q_net_into_ground = -(𝒬ᵀ + 𝒬ᵛ)      # W m⁻²
        Jᵀ[i, j, 1] = -Q_net_into_ground / ρcₛ
        M[i, j, 1]  = -Jᵛ                    # kg m⁻² s⁻¹ upward
    end
end

"""
    ComponentExchanger(land::SlabLand{...,<:RucEnergy,...,<:RucSurfaceProperties}, grid)

Expose RUC slab-side state as a `NamedTuple` with the fixed key set
`(T, Tc, θ, θ_ice, albedo, emissivity, roughness_length,
moisture_availability, stomatal_resistance)`. These keys are what
`compute_atmosphere_land_fluxes!` reads to drive Monin-Obukhov
similarity (`T`, `moisture_availability`, `roughness_length`).
"""
function ComponentExchanger(land::SlabLand{FT, G, Clk, S, F,
                                           <:RucEnergy, <:RucHydrology,
                                           <:RucSurfaceProperties, P}, grid) where {FT, G, Clk, S, F, P}
    state = (T                     = land.state.T,
             Tc                    = land.state.Tc,
             θ                     = land.state.θ,
             θ_ice                 = land.state.θ_ice,
             albedo                = land.surface.albedo,
             emissivity            = land.surface.emissivity,
             roughness_length      = land.surface.roughness_length,
             moisture_availability = land.state.moisture_availability,
             stomatal_resistance   = land.state.stomatal_resistance)
    return ComponentExchanger(state, nothing)
end

#####
##### Checkpointing — generic over state NamedTuple keys.
#####

import Oceananigans: prognostic_state, restore_prognostic_state!

function prognostic_state(land::SlabLand)
    state_arrays   = map(f -> Array(interior(f)), land.state)
    surface_arrays = _surface_state_arrays(land.surface)
    return (; clock   = prognostic_state(land.clock),
              state   = state_arrays,
              surface = surface_arrays)
end

function _surface_state_arrays(s::RucSurfaceProperties)
    return (; vegfrac                     = Array(interior(s.vegfrac)),
              lai                         = Array(interior(s.lai)),
              albedo_vegetation           = Array(interior(s.albedo_vegetation)),
              emissivity_vegetation       = Array(interior(s.emissivity_vegetation)),
              roughness_length_vegetation = Array(interior(s.roughness_length_vegetation)),
              stomatal_resistance_min     = Array(interior(s.stomatal_resistance_min)),
              is_urban                    = Array(interior(s.is_urban)),
              albedo                      = Array(interior(s.albedo)),
              emissivity                  = Array(interior(s.emissivity)),
              roughness_length            = Array(interior(s.roughness_length)))
end

_surface_state_arrays(::AbstractSurfaceProperties) = (;)

function restore_prognostic_state!(land::SlabLand, state)
    restore_prognostic_state!(land.clock, state.clock)
    for k in keys(land.state)
        if hasproperty(state.state, k)
            interior(getproperty(land.state, k)) .= getproperty(state.state, k)
        end
    end
    _restore_surface!(land.surface, state.surface)
    Oceananigans.TimeSteppers.update_state!(land)
    return land
end

function _restore_surface!(s::RucSurfaceProperties, surface_state)
    for k in (:vegfrac, :lai,
              :albedo_vegetation, :emissivity_vegetation, :roughness_length_vegetation,
              :stomatal_resistance_min, :is_urban,
              :albedo, :emissivity, :roughness_length)
        if hasproperty(surface_state, k)
            interior(getproperty(s, k)) .= getproperty(surface_state, k)
        end
    end
    return nothing
end

_restore_surface!(::AbstractSurfaceProperties, ::Any) = nothing

restore_prognostic_state!(land::SlabLand, ::Nothing) = land
