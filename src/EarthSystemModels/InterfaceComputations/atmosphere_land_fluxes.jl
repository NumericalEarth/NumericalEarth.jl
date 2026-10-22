land_surface_state(land, temperature) = (; state=(;), next=(;))
surface_hydrology(land, humidity, grid) = (;)
thermal_exchange_depth(temperature) = Inf
thermal_exchange_depth(temperature::EnergyBalanceTemperature) = temperature.coupling.thickness
thermal_exchange_depth(temperature::CanopyAirSpace) = temperature.ground_heat_flux.thickness
thermal_exchange_depth(temperature::SkinTemperature{<:DiffusiveFlux}) = temperature.internal_flux.δ

using Oceananigans.Grids: inactive_node

#####
##### Atmosphere-Land interface constructor

atmosphere_land_interface(grid, ::Nothing,    land;     kw...) = nothing
atmosphere_land_interface(grid, atmosphere, ::Nothing; kw...) = nothing
atmosphere_land_interface(grid, ::Nothing,  ::Nothing; kw...) = nothing

"""
    atmosphere_land_interface(grid, atmosphere, land;
                              fluxes               = default_atmosphere_land_fluxes(land, eltype(grid)),
                              temperature          = BulkTemperature(),
                              velocity_difference  = RelativeVelocity(),
                              specific_humidity    = default_al_specific_humidity(temperature, land))

Build the atmosphere--land interface on `grid` from `atmosphere` and `land` with
the given turbulent-flux closure, interface-temperature model, atmosphere-relative
velocity model, and specific-humidity formulation. A [`CanopyAirSpace`](@ref)
`temperature` closes the humidity slot as well. Pass the result as
`atmosphere_land_interface = ...` to `ComponentInterfaces` /
`AtmosphereLandModel` to override the default.

The flux closure's roughness lengths and zero-plane displacement may be per-cell
`Field`s at `(Center, Center, Nothing)` on `grid` — for example from
`urban_roughness` or a canopy roughness closure — localized to each cell before the
Monin--Obukhov solve.
"""
function atmosphere_land_interface(grid, atmosphere, land;
                                   fluxes              = default_atmosphere_land_fluxes(land, eltype(grid)),
                                   temperature         = default_al_temperature(land),
                                   velocity_difference = RelativeVelocity(),
                                   specific_humidity   = default_al_specific_humidity(temperature, land))
    return atmosphere_land_interface(grid, atmosphere, land, temperature;
                                     fluxes, velocity_difference, specific_humidity)
end

function atmosphere_land_interface(grid, atmosphere, land, temperature;
                                   fluxes, velocity_difference, specific_humidity)
    validate_flux_formulation(fluxes, grid)

    if temperature isa CanopyAirSpace && specific_humidity !== temperature
        throw(ArgumentError("a CanopyAirSpace closes both interface slots; pass it as `temperature` only"))
    end

    surface = land_surface_state(land, temperature)
    if (temperature isa PrognosticCanopyAirSpace || temperature isa PrognosticEnergyBalanceTemperature) && isempty(surface.state)
        throw(ArgumentError("Configure prognostic surface physics on LandSurface when constructing the land."))
    end
    boundary = merge(surface, (; hydrology=surface_hydrology(land, specific_humidity, grid)))

    al_fluxes = AtmosphereSurfaceFluxes(grid)
    al_properties = InterfaceProperties(specific_humidity, temperature, velocity_difference)
    interface_temperature = build_interface_temperature(temperature, grid)
    interface_specific_humidity = Field{Center, Center, Nothing}(grid)
    return AtmosphereInterface(al_fluxes, fluxes, interface_temperature,
                               interface_specific_humidity, al_properties, boundary)
end

# The atmosphere-facing interface temperature: a single field, or the
# `CanopyAirSpaceDiagnostics` of a canopy air space.
@inline build_interface_temperature(temperature_formulation, grid) = Field{Center, Center, Nothing}(grid)
@inline build_interface_temperature(cas::CanopyAirSpace, grid) = CanopyAirSpaceDiagnostics(grid)

# Initial interface values for the fixed point: the bulk land temperature and its
# saturation humidity, or the stored state of a prognostic formulation once the model
# is stepping (`Δt > 0`; the initializing call has `Δt = 0`).
@inline initial_interface_values(formulation, state, i, j, T₀, q₀, Δt) = (T₀, q₀)

@inline function initial_interface_values(::PrognosticCanopyAirSpace, state,
                                          i, j, T₀, q₀, Δt)
    stepped = Δt > 0
    @inbounds T = ifelse(stepped, state.temperature[i, j, 1], T₀)
    @inbounds q = ifelse(stepped, state.specific_humidity[i, j, 1], q₀)
    return T, q
end

@inline function initial_interface_values(::PrognosticEnergyBalanceTemperature, state,
                                          i, j, T₀, q₀, Δt)
    @inbounds T = ifelse(Δt > 0, state.temperature[i, j, 1], T₀)
    return T, q₀
end

# Store diagnostics and candidate land updates from the converged boundary solve.
@inline function finalize_interface_state!(Ts, next, i, j, formulation, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, Δt)
    @inbounds Ts[i, j, 1] = Ψₛ.temperature
    return Ψₛ
end

@inline function finalize_interface_state!(Ts, next, i, j, c::CanopyAirSpace, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, Δt)
    sol = canopy_air_space_solve(c, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    @inbounds begin
        Ts.interface[i, j, 1]              = sol.Tᵃᶜ
        Ts.canopy[i, j, 1]                 = sol.Tˡᵉᵃᶠ
        Ts.ground[i, j, 1]                 = sol.Tᵍ
        Ts.effective[i, j, 1]              = sol.effective_temperature
        Ts.effective_albedo[i, j, 1]       = sol.effective_albedo
        Ts.ground_heat_flux[i, j, 1]       = sol.𝒬ᵍ
        Ts.canopy_latent_heat[i, j, 1]     = sol.LEˡᵉᵃᶠ
        Ts.soil_latent_heat[i, j, 1]       = sol.LEᵍ
        Ts.canopy_sensible_heat[i, j, 1]   = sol.Hˡᵉᵃᶠ
        Ts.soil_sensible_heat[i, j, 1]     = sol.Hᵍ
        Ts.canopy_evaporation[i, j, 1]     = sol.Eʷᵉᵗ
        Ts.canopy_wet_latent_heat[i, j, 1] = sol.ℒ * sol.Eʷᵉᵗ
        Ts.land_vapor_flux[i, j, 1]        = (sol.LEˡᵉᵃᶠ + sol.LEᵍ) / sol.ℒ - sol.Eʷᵉᵗ
        Ts.transpiration[i, j, 1]          = sol.LEˡᵉᵃᶠ / sol.ℒ - sol.Eʷᵉᵗ
    end
    store_canopy_air_update!(next, i, j, sol, Ψₛ)

    # The exported scales are evaluated at the node the fluxes above were partitioned at, so
    # the atmosphere receives exactly the Kirchhoff supply (minus the storage tendency of a
    # prognostic node) and the stores are debited by the same solve.
    FT  = eltype(Ψₛ)
    θᵃᵗ = convert(FT, surface_atmosphere_temperature(Ψₐ, ℙₐ))
    fluxes = InterfaceFluxScales(Ψₛ.fluxes.u★,
                                 convert(FT, Ψₛ.fluxes.χθ * (θᵃᵗ - sol.Tᵃᶜ)),
                                 convert(FT, Ψₛ.fluxes.χq * (Ψₐ.q - sol.qᵃᶜ)),
                                 Ψₛ.fluxes.χθ, Ψₛ.fluxes.χq)
    return rebuild_interface_state(Ψₛ, fluxes, sol.Tᵃᶜ, sol.qᵃᶜ)
end

@inline store_canopy_air_update!(::NamedTuple{()}, i, j, sol, Ψₛ) = nothing

# The node advances by its exact relaxation toward the equilibrium the skins were solved against.
@inline function store_canopy_air_update!(state, i, j, sol, Ψₛ)
    Δt = Ψₛ.energy.time_step
    T⁺ = advance_canopy_air(Ψₛ.temperature, sol.equilibrium_temperature, sol.Σgᵀ, sol.Cᵀ, Δt)
    q⁺ = advance_canopy_air(Ψₛ.specific_humidity, sol.equilibrium_humidity, sol.Σgᵛ, sol.Cᵛ, Δt)
    @inbounds begin
        state.temperature[i, j, 1]       = T⁺
        state.specific_humidity[i, j, 1] = q⁺
    end
    return nothing
end

# A prognostic energy-balance skin is frozen through the fixed point and advanced once per
# step by a backward-Euler update of C dTₛ/dt = Rₙ + G − H − LE, Newton-refined so the
# radiative and vapor curvature is re-linearized each iterate. Δt⁻¹ = 0 (the initializing
# call) lands the update on the equilibrium root.
@inline function finalize_interface_state!(Ts, next, i, j, t::PrognosticEnergyBalanceTemperature,
                                          Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, Δt)
    FT = eltype(Ψₛ)
    Tₛ = Ψₛ.temperature
    C  = convert(FT, state2dindex(t.storage.heat_capacity, i, j))
    Δt⁻¹ = ifelse(Δt > 0, 1 / Δt, zero(FT))

    T⁺ = Tₛ
    for _ in 1:3
        F, Σλ = skin_energy_imbalance(T⁺, t, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ)
        R  = C * (T⁺ - Tₛ) * Δt⁻¹ - F
        dR = C * Δt⁻¹ + Σλ
        T⁺ = ifelse(dR > 0, T⁺ - R / dR, T⁺)
    end

    @inbounds Ts[i, j, 1] = next.temperature[i, j, 1] = T⁺

    Tᵃᵗ = convert(FT, surface_atmosphere_temperature(Ψₐ, ℙₐ))
    q⁺  = compute_interface_humidity(ℙₛ.specific_humidity_formulation, T⁺, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    fluxes = InterfaceFluxScales(Ψₛ.fluxes.u★,
                                 convert(FT, Ψₛ.fluxes.χθ * (Tᵃᵗ - T⁺)),
                                 convert(FT, Ψₛ.fluxes.χq * (Ψₐ.q - q⁺)),
                                 Ψₛ.fluxes.χθ, Ψₛ.fluxes.χq)
    return rebuild_interface_state(Ψₛ, fluxes, convert(FT, T⁺), convert(FT, q⁺))
end

# Fallback: assume land has a `fluxes` property like SlabLand
land_surface_energy_flux(land) = land.fluxes.surface_energy_flux

"""
    land_surface_shortwave_up(land)
    land_surface_longwave_up(land)

Fields into which `apply_air_land_radiative_fluxes!` writes the upwelling radiation (positive
upward), for land components that take it as a prescribed input. The upwelling fluxes are only
known in the radiative phase, after `update_net_fluxes!`, so they are pushed rather than pulled.
Returns `nothing` by default, in which case nothing is written.
"""
land_surface_shortwave_up(land) = nothing

@doc (@doc land_surface_shortwave_up)
land_surface_longwave_up(land) = nothing

#####
##### Flux compute driver
#####

"""
$(TYPEDSIGNATURES)

Compute atmosphere–land fluxes and candidate surface updates over `Δt`.
The land state is held fixed; `Δt = 0` computes diagnostic equilibrium.
"""
compute_atmosphere_land_fluxes!(coupled_model, Δt) =
    compute_atmosphere_land_fluxes!(coupled_model, coupled_model.interfaces.atmosphere_land_interface, Δt)

compute_atmosphere_land_fluxes!(coupled_model, ::Nothing, Δt) = nothing

function compute_atmosphere_land_fluxes!(coupled_model, atmosphere_land_interface, Δt)
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    arch = architecture(grid)
    clock = coupled_model.clock
    atmosphere_fields = exchanger.atmosphere.state

    # See compute_atmosphere_ocean_fluxes! for rationale.
    atmosphere_data = merge(atmosphere_fields,
                            (; h_bℓ = boundary_layer_height(coupled_model.atmosphere)))

    flux_formulation = atmosphere_land_interface.flux_formulation
    interface_fluxes = atmosphere_land_interface.fluxes
    interface_temperature = atmosphere_land_interface.temperature
    interface_specific_humidity = atmosphere_land_interface.specific_humidity
    interface_properties = atmosphere_land_interface.properties
    atmosphere_properties = (thermodynamics_parameters = thermodynamics_parameters(coupled_model.atmosphere),
                             surface_layer_height = coupled_model.interfaces.properties.surface_layer_height,
                             gravitational_acceleration = coupled_model.interfaces.properties.gravitational_acceleration)

    # Land surface state from the exchanger. `interface_energy_state` /
    # `interface_hydrology_state` read these per cell to build the land
    # interface state; the surface models derive `β`, the reservoir
    # temperature, etc. from them.
    land_exchanger_state = exchanger.land.state
    FT = eltype(grid)
    land_state = merge((T = land_exchanger_state.T,
                  saturation = land_exchanger_state.saturation,
                  root_zone_saturation = land_exchanger_state.root_zone_saturation,
                  canopy_water_storage = land_exchanger_state.canopy_water_storage,
                  canopy_water_capacity = land_exchanger_state.canopy_water_capacity,
                  time_step = convert(FT, Δt),
                  thermal_exchange_depth = thermal_exchange_depth(interface_properties.temperature_formulation)),
                  atmosphere_land_interface.boundary.hydrology)

    # Prescribed leaf area index off the canopy formulation (or `nothing`),
    # reduced to a kernel-friendly value plus its host-side time interpolator.
    leaf_area_index = canopy_leaf_area_index(interface_properties.specific_humidity_formulation)
    vegetation, leaf_area_index_time_interpolator = kernel_surface_field(leaf_area_index, arch, clock.time)

    radiation = coupled_model.radiation
    radiation_exchanger    = exchanger.radiation
    radiation_kernel_props = kernel_radiation_properties(radiation, radiation_exchanger)
    radiation_state        = isnothing(radiation_exchanger) ? nothing : radiation_exchanger.state

    # Interior cells only: halo cells of the atmosphere exchanger state are uninitialized when the
    # atmosphere grid is a regional cutout matching the exchange-grid interior exactly.
    launch!(arch, grid, :xy,
            _compute_atmosphere_land_interface_state!,
            interface_fluxes,
            interface_temperature,
            interface_specific_humidity,
            grid,
            clock,
            flux_formulation,
            land_state,
            vegetation,
            leaf_area_index_time_interpolator,
            atmosphere_data,
            interface_properties,
            atmosphere_land_interface.boundary,
            atmosphere_properties,
            radiation_kernel_props,
            radiation_state)

    return nothing
end

#####
##### Prescribed, possibly time-varying surface inputs.
##### `surface_field_value` reads the per-cell value from a `Number`, a static
##### `Field`, or a `FieldTimeSeries` interpolated to the model clock.
#####

@inline surface_field_value(x, i, j, time_interpolator) = state2dindex(x, i, j)

# `i, j` name an exact cell, so the value is interpolated in time only. Passing them to
# `interpolate` as spatial fractional indices would also read the neighbor at
# `(i + 1, j + 1)`, which does not exist in a `Flat` direction or at the last cell.
#
# The indices are converted because `time_interpolator` reaches the kernel as an argument,
# where its scalar fields can arrive as `CuTracedRNumber` rather than `Int`
# (CliMA/Oceananigans.jl#4230).
#
# TODO: drop this blend once Oceananigans accepts a `TimeInterpolator` in `getindex`
# (CliMA/Oceananigans.jl#5886), leaving `x[i, j, 1, time_interpolator]`. The flux kernels take
# an interpolator precomputed on the host, as the prescribed-atmosphere path does, because
# `fts[i, j, k, Time(t)]` recomputes the time indices in every thread; today Oceananigans
# accepts one only through `interpolate`, which interpolates in space as well.
@inline function surface_field_value(x::FlavorOfFTS, i, j, time_interpolator)
    ñ  = time_interpolator.fractional_index
    n₁ = convert(Int, time_interpolator.first_index)
    n₂ = convert(Int, time_interpolator.second_index)

    @inbounds ψ₁ = x[i, j, 1, n₁]
    @inbounds ψ₂ = x[i, j, 1, n₂]

    return ifelse(n₁ == n₂, ψ₁, ψ₂ * ñ + ψ₁ * (1 - ñ))
end

# Host-side: pair a prescribed-surface spec (LAI, vegetation fraction, …) with the time
# index used to interpolate it. Constants and static fields pass through untouched
# (`nothing` interpolator); a `FieldTimeSeries` is brought into memory at `time` and keeps
# its time index precomputed on the host, so the kernel does not recompute it.
@inline kernel_surface_field(surface_field, arch, time) = (surface_field, nothing)
@inline function kernel_surface_field(surface_field::FieldTimeSeries, arch, time)
    update_field_time_series!(surface_field, Time(time))
    time_interpolator = cpu_interpolating_time_indices(arch, surface_field.times,
                                                       surface_field.time_indexing, time)
    return surface_field, time_interpolator
end

# The LAI spec lives on the canopy humidity formulation; other formulations carry
# none. The canopy / composite methods are defined alongside those formulations.
@inline canopy_leaf_area_index(q_formulation) = nothing

#####
##### Land surface state materialized into the interface state.
#####
##### The surface model (`interface_model`, here the specific-humidity
##### formulation) dispatches these helpers to pull *exactly* the per-cell land
##### state it consumes — saturation for the moisture-availability models, the
##### bulk temperature for the reservoir model — and nothing otherwise. The
##### model then derives `β`, the reservoir temperature, etc. from what it pulled.
#####

@inline land_saturation(i, j, grid, land_state) = state2dindex(land_state.saturation, i, j)

# Hydrology state, per humidity formulation. The caller adds the `saturation` key.
@inline interface_hydrology_state(i, j, grid, ::BulkHumidity, land_state) = (; saturation = land_saturation(i, j, grid, land_state))
@inline interface_hydrology_state(i, j, grid, q::FractionalHumidity, land_state) = interface_hydrology_state(i, j, grid, q.efficiency, land_state)
@inline interface_hydrology_state(i, j, grid, ::CriticalSaturation, land_state) = (; saturation = land_saturation(i, j, grid, land_state))
@inline function interface_hydrology_state(i, j, grid, ::PlantAvailableWaterStress, land_state)
    return (root_zone_saturation = state2dindex(land_state.root_zone_saturation, i, j),
            field_capacity_saturation = state2dindex(land_state.field_capacity_saturation, i, j),
            wilting_saturation = state2dindex(land_state.wilting_saturation, i, j))
end

@inline interface_hydrology_state(i, j, grid, ::DryLayerHumidity, land_state) =
    (saturation = land_saturation(i, j, grid, land_state),
     vapor_diffusivity = state2dindex(land_state.vapor_diffusivity, i, j))
@inline interface_hydrology_state(i, j, grid, interface_model, land_state) = (;) # default: pulls nothing

# Energy state: humidity formulations that need the bulk land temperature
# (the SkinHumidity reservoir and the DryLayerHumidity dry-layer model)
# pull it from the materialized land state.
@inline interface_energy_state(i, j, grid, ::SkinHumidity, land_state) =
    (temperature = state2dindex(land_state.T, i, j),)
@inline interface_energy_state(i, j, grid, ::DryLayerHumidity, land_state) =
    (temperature = state2dindex(land_state.T, i, j),
     thermal_exchange_depth = state2dindex(land_state.thermal_exchange_depth, i, j))
@inline interface_energy_state(i, j, grid, interface_model, land_state) = (;) # default: pulls nothing

# Vegetation state, per humidity formulation. Only the canopy formulations
# (defined in their own files) pull a leaf area index; everything else is empty.
@inline interface_vegetation_state(i, j, grid, interface_model, vegetation, time_interpolator) = (;)

@kernel function _compute_atmosphere_land_interface_state!(interface_fluxes,
                                                           interface_temperature,
                                                           interface_specific_humidity,
                                                           grid,
                                                           clock,
                                                           turbulent_flux_formulation,
                                                           land_state,
                                                           vegetation,
                                                           leaf_area_index_time_interpolator,
                                                           atmosphere_state,
                                                           interface_properties,
                                                           boundary,
                                                           atmosphere_properties,
                                                           radiation_kernel_props,
                                                           radiation_exchanger_state)

    i, j = @index(Global, NTuple)
    time = Time(clock.time)
    Δt = land_state.time_step

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    Ψₐ  = local_atmosphere_state(i, j, atmosphere_state, atmosphere_properties)

    # `CanopyAirSpace` optics slots may be per-cell `Field`s; collapse them to this cell's
    # values before the index-free solve.
    temperature_formulation = local_interface_formulation(interface_properties.temperature_formulation, i, j)
    q_formulation           = local_interface_formulation(interface_properties.specific_humidity_formulation, i, j)

    local_interface_properties = InterfaceProperties(q_formulation, temperature_formulation,
                                                     interface_properties.velocity_formulation)

    # Bulk land temperature serves as the initial skin-temperature guess.
    Tₛ = state2dindex(land_state.T, i, j)
    FT = typeof(Tₛ)

    # Collapse Field-valued roughness lengths and displacement to this cell's values.
    local_turbulent_flux_formulation = local_flux_formulation(turbulent_flux_formulation, i, j)

    # Surface velocities are zero for land.
    uₛ = zero(FT)
    vₛ = zero(FT)

    local_interior_state = (u = uₛ, v = vₛ, T = Tₛ)

    radiation_state = air_land_interface_radiation_state(radiation_kernel_props,
                                                         radiation_exchanger_state,
                                                         i, j, 1, grid, time)

    # Estimate initial interface state. Use the saturated value as the initial
    # surface humidity guess (the solver recomputes it via the formulation);
    # prognostic formulations read their stored state back instead.
    u★ = convert(FT, 1e-4)
    qₛ = convert(FT, saturation_specific_humidity(ℂᵃᵗ, Tₛ, Ψₐ.p, interface_phase(q_formulation)))
    Tₛ, qₛ = initial_interface_values(temperature_formulation,
                                      boundary.state, i, j, Tₛ, qₛ, Δt)
    initial_interface_state = AirLandInterfaceState(i, j, grid,
                                                    InterfaceFluxScales(u★, u★, u★),
                                                    InterfaceVelocities(uₛ, vₛ),
                                                    q_formulation, land_state,
                                                    vegetation, leaf_area_index_time_interpolator,
                                                    Tₛ, qₛ)

    Ψₛ = compute_interface_state(local_turbulent_flux_formulation,
                                 initial_interface_state,
                                 Ψₐ,
                                 local_interior_state,
                                 radiation_state,
                                 local_interface_properties,
                                 atmosphere_properties,
                                 (;))

    Ψₛ = finalize_interface_state!(interface_temperature, boundary.next, i, j,
                                  temperature_formulation, Ψₛ, Ψₐ,
                                  local_interior_state, radiation_state,
                                  local_interface_properties, atmosphere_properties, Δt)

    ℒˡ = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Ψₐ.T)

    @inbounds interface_specific_humidity[i, j, 1] = Ψₛ.specific_humidity
    store_interface_fluxes!(interface_fluxes, i, j, Ψₛ, Ψₐ, ℂᵃᵗ, ℒˡ, local_interface_properties)
end

"""
$(TYPEDSIGNATURES)

Surface energy loss and water fluxes consumed by land, positive upward (W m⁻² and kg m⁻² s⁻¹).
"""
function land_surface_fluxes(interface::AtmosphereInterface, bulk_temperature)
    temperature = interface.temperature
    if temperature isa CanopyAirSpaceDiagnostics
        return land_surface_fluxes(temperature)
    end
    fluxes = interface.fluxes
    conductance = skin_conductance(interface)
    energy = isnothing(conductance) ? fluxes.sensible_heat + fluxes.latent_heat :
                                     -conductance * (temperature - bulk_temperature)
    transpiration = interface.properties.specific_humidity_formulation isa CanopyConductanceHumidity ?
                    fluxes.water_vapor : ZeroField()
    return (; energy, vapor=fluxes.water_vapor, transpiration, canopy_evaporation=ZeroField())
end

land_surface_fluxes(temperature::CanopyAirSpaceDiagnostics) =
    (; energy=-temperature.ground_heat_flux, vapor=temperature.land_vapor_flux,
       transpiration=temperature.transpiration, canopy_evaporation=temperature.canopy_evaporation)

"""
$(TYPEDSIGNATURES)

Surface temperature, albedo, and emissivity for outgoing radiation.
"""
radiative_surface(interface::AtmosphereInterface, albedo, emissivity) =
    radiative_surface(interface.temperature, albedo, emissivity)
radiative_surface(temperature, albedo, emissivity) = (; temperature, albedo, emissivity)
radiative_surface(temperature::CanopyAirSpaceDiagnostics, albedo, emissivity) =
    (; temperature=temperature.effective, albedo=temperature.effective_albedo, emissivity=1)
