using Oceananigans.Grids: inactive_node

#####
##### Land surface state the interface reads
#####

land_surface_state(land) = (; state=nothing, next=(;))
validate_surface_storage(land, state) = nothing

# Distance over which a skin conducts to the bulk temperature.
thermal_exchange_depth(temperature) = Inf
thermal_exchange_depth(temperature::EnergyBalanceTemperature) = temperature.coupling.thickness
thermal_exchange_depth(temperature::CanopyInterface) = temperature.ground_heat_flux.thickness
thermal_exchange_depth(temperature::SkinTemperature{<:DiffusiveFlux}) = temperature.internal_flux.δ

#####
##### Atmosphere-Land interface constructor

atmosphere_land_interface(grid, ::Nothing,    land;     kw...) = nothing
atmosphere_land_interface(grid, atmosphere, ::Nothing; kw...) = nothing
atmosphere_land_interface(grid, ::Nothing,  ::Nothing; kw...) = nothing

"""
$(TYPEDSIGNATURES)

Build the atmosphere–land interface. `state` specifies the interface-state formulation:
a named tuple of `temperature` and `specific_humidity` models, or a `CanopyInterface`
that solves the coupled canopy and ground balances. `fluxes` and `velocity_difference`
specify turbulent transport and relative motion.

The flux closure's roughness lengths and zero-plane displacement may be per-cell
`Field`s at `(Center, Center, Nothing)` on `grid` — for example from
`urban_roughness` or a canopy roughness closure — localized to each cell before the
Monin--Obukhov solve.
"""
function atmosphere_land_interface(grid, atmosphere, land;
                                   fluxes              = default_atmosphere_land_fluxes(land, eltype(grid)),
                                   temperature         = BulkTemperature(),
                                   velocity_difference = RelativeVelocity(),
                                   specific_humidity   = default_al_specific_humidity(land),
                                   state               = (; temperature, specific_humidity))
    validate_surface_storage(land, state)
    return build_atmosphere_land_interface(grid, fluxes, state, velocity_difference)
end

function build_atmosphere_land_interface(grid, fluxes, state, velocity_difference)
    validate_flux_formulation(fluxes, grid)
    al_fluxes = AtmosphereSurfaceFluxes(grid)
    formulation = InterfaceFormulation(fluxes, state, velocity_difference)
    interface_temperature = build_interface_temperature(temperature_formulation(state), grid)
    interface_specific_humidity = Field{Center, Center, Nothing}(grid)
    return AtmosphereInterface(al_fluxes, interface_temperature, interface_specific_humidity, formulation)
end

# The atmosphere-facing interface temperature: a single field, or the
# `CanopyInterfaceDiagnostics` of a canopy air space.
@inline build_interface_temperature(temperature_formulation, grid) = Field{Center, Center, Nothing}(grid)
@inline build_interface_temperature(surface::CanopyInterface, grid) = CanopyInterfaceDiagnostics(grid)

function initialize_land_interface!(coupled_model)
    interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(interface) && return nothing
    exchanger = coupled_model.interfaces.exchanger
    EarthSystemModels.interpolate_state!(exchanger.land, exchanger.grid, coupled_model.land, coupled_model)
    initialize_land_interface!(coupled_model, interface)
    return nothing
end

function initialize_land_interface!(coupled_model, interface, surface_state = land_surface_state(coupled_model.land))
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    arch = architecture(grid)
    clock = coupled_model.clock
    state = interface.formulation.state
    leaf_area_index = canopy_leaf_area_index(specific_humidity_formulation(state))
    vegetation, time_interpolator = kernel_surface_field(leaf_area_index, arch, clock.time)
    launch!(arch, grid, :xy, _initialize_land_interface!, interface.temperature,
            exchanger.land.state.T, surface_state.state, grid, clock, state, vegetation, time_interpolator)
    return nothing
end

@kernel function _initialize_land_interface!(temperature, bulk_temperature, storage, grid, clock, state,
                                             vegetation, time_interpolator)
    i, j = @index(Global, NTuple)
    formulation = local_interface_formulation(state, i, j)
    vegetation_state = interface_vegetation_state(i, j, grid, formulation, vegetation, time_interpolator)
    T = state2dindex(bulk_temperature, i, j, grid, Time(clock.time))
    local_storage = state2dindex(storage, i, j, grid, Time(clock.time))
    T, _ = initial_interface_values(temperature_formulation(formulation), local_storage, T, zero(T), zero(T))
    initialize_interface_temperature!(temperature, i, j, T, formulation, vegetation_state)
end

@inline function initialize_interface_temperature!(temperature, i, j, T, formulation, vegetation)
    @inbounds temperature[i, j, 1] = T
    return nothing
end

@inline function initialize_interface_temperature!(temperature::CanopyInterfaceDiagnostics, i, j, T, c, vegetation)
    transmittance = canopy_transmittance(c.extinction, c.clumping, vegetation.leaf_area_index)
    @inbounds begin
        temperature.interface[i, j, 1] = T
        temperature.canopy[i, j, 1] = T
        temperature.ground[i, j, 1] = T
        temperature.effective[i, j, 1] = T
        temperature.effective_albedo[i, j, 1] = (1 - transmittance) * c.leaf_albedo + transmittance * c.ground_albedo
    end
    return nothing
end

# Initial interface values for the fixed point: the bulk land temperature and its
# saturation humidity, or the stored state of a prognostic formulation once the model
# is stepping (`Δt > 0`; the initializing call has `Δt = 0`).
@inline initial_interface_values(formulation, storage::Nothing, T₀, q₀, Δt) = (T₀, q₀)

@inline function initial_interface_values(::CanopyInterface, storage::NamedTuple, T₀, q₀, Δt)
    return ifelse(Δt > 0, storage.temperature, T₀), ifelse(Δt > 0, storage.specific_humidity, q₀)
end

@inline function initial_interface_values(::EnergyBalanceTemperature, storage::NamedTuple, T₀, q₀, Δt)
    return ifelse(Δt > 0, storage.temperature, T₀), q₀
end

# Store diagnostics and candidate land updates from the converged boundary solve.
@inline function finalize_interface_state!(Ts, next, i, j, formulation, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₛ, ℙₐ, Δt)
    @inbounds Ts[i, j, 1] = Ψₛ.temperature
    return Ψₛ
end

@inline function finalize_interface_state!(Ts, next, i, j, c::CanopyInterface, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₛ, ℙₐ, Δt)
    solution = canopy_air_space_solve(c, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, Δt)
    @inbounds begin
        Ts.interface[i, j, 1]              = solution.Tᵃᶜᵃⁿ
        Ts.canopy[i, j, 1]                 = solution.Tˡᵉᵃ
        Ts.ground[i, j, 1]                 = solution.Tᵍʳᵒ
        Ts.effective[i, j, 1]              = solution.effective_temperature
        Ts.effective_albedo[i, j, 1]       = solution.effective_albedo
        Ts.ground_heat_flux[i, j, 1]       = solution.𝒬ᵍʳᵒ
        Ts.canopy_latent_heat[i, j, 1]     = solution.𝒬ᵛˡᵉᵃ
        Ts.ground_latent_heat[i, j, 1]     = solution.𝒬ᵛᵍʳᵒ
        Ts.canopy_sensible_heat[i, j, 1]   = solution.𝒬ᵀˡᵉᵃ
        Ts.ground_sensible_heat[i, j, 1]   = solution.𝒬ᵀᵍʳᵒ
        Ts.canopy_evaporation[i, j, 1]     = solution.Jʷᵉᵗ
        Ts.wet_canopy_latent_heat[i, j, 1] = solution.ℒ * solution.Jʷᵉᵗ
        Ts.land_vapor_flux[i, j, 1]        = (solution.𝒬ᵛˡᵉᵃ + solution.𝒬ᵛᵍʳᵒ) / solution.ℒ - solution.Jʷᵉᵗ
        Ts.transpiration[i, j, 1]          = solution.𝒬ᵛˡᵉᵃ / solution.ℒ - solution.Jʷᵉᵗ
    end
    store_canopy_air_update!(next, i, j, solution, Ψₛ, Δt)

    # The exported scales are evaluated at the node the fluxes above were partitioned at.
    return rebuild_interface_state(Ψₛ, Ψₐ, ℙₐ, solution.Tᵃᶜᵃⁿ, solution.qᵃᶜᵃⁿ)
end

@inline store_canopy_air_update!(::NamedTuple{()}, i, j, solution, Ψₛ, Δt) = nothing

# The node advances by its exact relaxation toward the equilibrium the skins were solved against.
@inline function store_canopy_air_update!(state, i, j, solution, Ψₛ, Δt)
    Tⁿ⁺¹ = advance_canopy_air(Ψₛ.temperature, solution.equilibrium_temperature, solution.Σgᵀ, solution.Cᵀ, Δt)
    qⁿ⁺¹ = advance_canopy_air(Ψₛ.specific_humidity, solution.equilibrium_humidity, solution.Σgᵛ, solution.Cᵛ, Δt)
    @inbounds begin
        state.temperature[i, j, 1]       = Tⁿ⁺¹
        state.specific_humidity[i, j, 1] = qⁿ⁺¹
    end
    return nothing
end

# Backward Euler for skin heat storage, with Newton refinement of radiation and vapor fluxes.
@inline function finalize_interface_state!(Ts, next::NamedTuple{(:temperature,)}, i, j, t::EnergyBalanceTemperature,
                                          Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₛ, ℙₐ, Δt)
    FT = eltype(Ψₛ)
    Tₛ = Ψₛ.temperature
    C  = convert(FT, Ψˡᵃ.surface_storage.heat_capacity)
    Δt⁻¹ = ifelse(Δt > 0, 1 / Δt, zero(FT))

    Tⁿ⁺¹ = Tₛ
    for _ in 1:3
        F, Σg = skin_energy_imbalance(Tⁿ⁺¹, t, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₛ, ℙₐ)
        R    = C * (Tⁿ⁺¹ - Tₛ) * Δt⁻¹ - F
        dR   = C * Δt⁻¹ + Σg
        Tⁿ⁺¹ = ifelse(dR > 0, Tⁿ⁺¹ - R / dR, Tⁿ⁺¹)
    end

    @inbounds Ts[i, j, 1] = next.temperature[i, j, 1] = Tⁿ⁺¹

    qⁿ⁺¹ = compute_interface_humidity(specific_humidity_formulation(ℙₛ.state), Tⁿ⁺¹, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)
    return rebuild_interface_state(Ψₛ, Ψₐ, ℙₐ, Tⁿ⁺¹, qⁿ⁺¹)
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

function compute_atmosphere_land_fluxes!(coupled_model, atmosphere_land_interface, Δt,
                                         surface_state = land_surface_state(coupled_model.land))
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    arch = architecture(grid)
    clock = coupled_model.clock
    atmosphere_fields = exchanger.atmosphere.state

    atmosphere_data = merge(atmosphere_fields,
                            (; h_bℓ = boundary_layer_height(coupled_model.atmosphere)))

    interface_fluxes = atmosphere_land_interface.fluxes
    interface_temperature = atmosphere_land_interface.temperature
    interface_specific_humidity = atmosphere_land_interface.specific_humidity
    interface_formulation = atmosphere_land_interface.formulation
    atmosphere_properties = (thermodynamics_parameters = thermodynamics_parameters(coupled_model.atmosphere),
                             surface_layer_height = coupled_model.interfaces.properties.surface_layer_height,
                             gravitational_acceleration = coupled_model.interfaces.properties.gravitational_acceleration)

    FT = eltype(grid)
    land_state = merge(exchanger.land.state,
                       (surface_storage = surface_state.state,
                        thermal_exchange_depth = thermal_exchange_depth(temperature_formulation(interface_formulation.state))))

    # Prescribed leaf area index off the canopy formulation (or `nothing`),
    # reduced to a kernel-friendly value plus its host-side time interpolator.
    leaf_area_index = canopy_leaf_area_index(specific_humidity_formulation(interface_formulation.state))
    vegetation, time_interpolator = kernel_surface_field(leaf_area_index, arch, clock.time)

    radiation = coupled_model.radiation
    radiation_exchanger    = exchanger.radiation
    radiation_kernel_props = kernel_radiation_properties(radiation, radiation_exchanger)
    radiation_state        = isnothing(radiation_exchanger) ? nothing : radiation_exchanger.state

    # Interior cells only: halo cells of the atmosphere exchanger state are uninitialized when the
    # atmosphere grid is a regional cutout matching the exchange-grid interior exactly.
    launch!(arch, grid, :xy, _compute_atmosphere_land_interface_state!,
            interface_fluxes, interface_temperature, interface_specific_humidity, grid, clock,
            interface_formulation, land_state, vegetation, time_interpolator, atmosphere_data,
            surface_state.next, atmosphere_properties, radiation_kernel_props, radiation_state, convert(FT, Δt))

    return nothing
end

#####
##### Prescribed, possibly time-varying surface inputs (λˡᵉᵃᶠ, vegetation fraction, …): a
##### `Number`, a static `Field`, or a `FieldTimeSeries`.
#####

# A `FieldTimeSeries` is brought into memory at `time` and paired with the time interpolator
# it is read at; anything else passes through with none.
@inline kernel_surface_field(surface_field, arch, time) = (surface_field, nothing)
@inline function kernel_surface_field(surface_field::FieldTimeSeries, arch, time)
    update_field_time_series!(surface_field, Time(time))
    time_interpolator = cpu_interpolating_time_indices(arch, surface_field.times, surface_field.time_indexing, time)
    return surface_field, time_interpolator
end

# The λˡᵉᵃᶠ spec lives on the canopy humidity formulation; other formulations carry
# none. The canopy / composite methods are defined alongside those formulations.
@inline canopy_leaf_area_index(q_formulation) = nothing

#####
##### Land state read by the interface solver
#####

"""
$(TYPEDSIGNATURES)

The land state that the atmosphere--land interface described by `formulation` reads at
each cell, as a `NamedTuple` of accessor functions such as `(T = land_temperature, 𝒮 = land_saturation)`.
The land component publishes `accessor(land)` for each entry, on the exchange grid, through
its `ComponentExchanger`; the flux kernel reads them pointwise as the land state `Ψˡᵃ`, under
the same names. The bulk land temperature `T` is always included: it is the initial guess for
the interface temperature.
"""
land_state_accessors(formulation::InterfaceFormulation) = land_state_accessors(specific_humidity_formulation(formulation.state))

land_state_accessors(specific_humidity_formulation) = (T = land_temperature, 𝒮 = land_saturation)
land_state_accessors(::SkinHumidity) = (T = land_temperature,)
land_state_accessors(::FractionalHumidity{<:Number}) = (T = land_temperature,)
land_state_accessors(q::FractionalHumidity) = land_state_accessors(q.efficiency)

"""
    land_temperature(land)

Return the bulk land temperature on the land grid.
"""
function land_temperature end

"""
    land_saturation(land)

Return the surface soil saturation `𝒮` on the land grid.
"""
function land_saturation end

"""
    land_canopy_water_storage(land)

Return the canopy interception store `Mᶜᵃⁿ` (kg m⁻²) on the land grid.
"""
function land_canopy_water_storage end

"""
    land_canopy_water_capacity(land)

Return the canopy interception capacity `Mᶜᵃⁿ⁺` (kg m⁻²) on the land grid.
"""
function land_canopy_water_capacity end

# Vegetation state, per humidity formulation. Only the canopy formulations
# (defined in their own files) pull a leaf area index; everything else is empty.
@inline interface_vegetation_state(i, j, grid, interface_model, vegetation, time_interpolator) = (;)

@kernel function _compute_atmosphere_land_interface_state!(interface_fluxes,
                                                           interface_temperature,
                                                           interface_specific_humidity,
                                                           grid,
                                                           clock,
                                                           interface_formulation,
                                                           land_state,
                                                           vegetation,
                                                           time_interpolator,
                                                           atmosphere_state,
                                                           next_surface_state,
                                                           atmosphere_properties,
                                                           radiation_kernel_props,
                                                           radiation_exchanger_state,
                                                           Δt)

    i, j = @index(Global, NTuple)
    time = Time(clock.time)

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    Ψᵃᵗ = local_atmosphere_state(i, j, atmosphere_state, atmosphere_properties)

    # The land state at this cell, with the canopy's prescribed vegetation (leaf area index).
    vegetation_state = interface_vegetation_state(i, j, grid, specific_humidity_formulation(interface_formulation.state),
                                                  vegetation, time_interpolator)
    Ψˡᵃ = merge(state2dindex(land_state, i, j, grid, time), vegetation_state)
    # `typeof(Ψˡᵃ.T)`, not `eltype(grid)`, which is a traced type inside a Reactant kernel.
    FT  = typeof(Ψˡᵃ.T)

    formulation = local_interface_formulation(interface_formulation, i, j)

    radiation_state = air_land_interface_radiation_state(radiation_kernel_props,
                                                         radiation_exchanger_state,
                                                         i, j, 1, grid, time)

    u★  = convert(FT, 1e-4)
    qⁱⁿ = convert(FT, saturation_specific_humidity(ℂᵃᵗ, Ψˡᵃ.T, Ψᵃᵗ.p, interface_phase(specific_humidity_formulation(formulation.state))))
    Tⁱⁿ, qⁱⁿ = initial_interface_values(temperature_formulation(formulation.state), Ψˡᵃ.surface_storage, Ψˡᵃ.T, qⁱⁿ, Δt)
    Ψⁱⁿ⁰ = AirLandInterfaceState(u★, u★, u★, zero(FT), zero(FT), convert(FT, Tⁱⁿ), convert(FT, qⁱⁿ))

    Ψⁱⁿ = compute_interface_state(formulation, Ψⁱⁿ⁰, Ψᵃᵗ, Ψˡᵃ, radiation_state, atmosphere_properties, (;), Δt)

    Ψⁱⁿ = finalize_interface_state!(interface_temperature, next_surface_state, i, j, temperature_formulation(formulation.state),
                                    Ψⁱⁿ, Ψᵃᵗ, Ψˡᵃ, radiation_state, formulation, atmosphere_properties, Δt)

    ℒˡ = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Ψᵃᵗ.T)

    @inbounds interface_specific_humidity[i, j, 1] = Ψⁱⁿ.specific_humidity
    store_interface_fluxes!(interface_fluxes, i, j, Ψⁱⁿ, Ψᵃᵗ, ℂᵃᵗ, ℒˡ, formulation)
end

"""
$(TYPEDSIGNATURES)

Surface energy loss and water fluxes consumed by land, positive upward (W m⁻² and kg m⁻² s⁻¹).
"""
land_surface_fluxes(interface::AtmosphereInterface, bulk_temperature) =
    land_surface_fluxes(interface.temperature, interface, bulk_temperature)

# TODO: Use KernelFunctionOperation after Oceananigans reconciles its Reactant traced
# scalar type with the unwrapped element type produced by GPU adaptation.
struct LandSurfaceEnergyFlux{T, F, B, C}
    temperature :: T
    fluxes :: F
    bulk_temperature :: B
    conductance :: C
end

Adapt.@adapt_structure LandSurfaceEnergyFlux

@inline Base.getindex(energy::LandSurfaceEnergyFlux, i, j, k) =
    surface_energy_loss(i, j, energy.temperature, energy.fluxes, energy.bulk_temperature, energy.conductance)

function land_surface_fluxes(temperature, interface, bulk_temperature)
    fluxes = interface.fluxes
    conductance = ground_conductance(interface)
    energy = LandSurfaceEnergyFlux(temperature, fluxes, bulk_temperature, conductance)
    return (; energy, vapor=fluxes.water_vapor, canopy_evaporation=ZeroField())
end

land_surface_fluxes(temperature::CanopyInterfaceDiagnostics, interface, bulk_temperature) =
    (; energy=-temperature.ground_heat_flux, vapor=temperature.land_vapor_flux,
       canopy_evaporation=temperature.canopy_evaporation)

@inline surface_energy_loss(i, j, temperature, fluxes, bulk_temperature, ::Nothing) =
    @inbounds fluxes.sensible_heat[i, j, 1] + fluxes.latent_heat[i, j, 1]

@inline surface_energy_loss(i, j, temperature, fluxes, bulk_temperature, conductance) =
    @inbounds -conductance * (temperature[i, j, 1] - bulk_temperature[i, j, 1])

"""
$(TYPEDSIGNATURES)

Surface temperature, albedo, and emissivity for outgoing radiation.
"""
radiative_surface(interface::AtmosphereInterface, albedo, emissivity) =
    radiative_surface(interface.temperature, albedo, emissivity)
radiative_surface(temperature, albedo, emissivity) = (; temperature, albedo, emissivity)
radiative_surface(temperature::CanopyInterfaceDiagnostics, albedo, emissivity) =
    (; temperature=temperature.effective, albedo=temperature.effective_albedo, emissivity=1)
