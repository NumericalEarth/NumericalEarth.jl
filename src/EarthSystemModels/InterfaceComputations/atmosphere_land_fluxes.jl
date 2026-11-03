using Oceananigans.Grids: inactive_node

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
that solves the coupled canopy and ground balances. A `CanopyInterface` with a
`canopy_air_depth`, or an [`EnergyBalanceTemperature`](@ref) with a `heat_capacity`, makes
the interface state prognostic. `fluxes` and `velocity_difference` specify turbulent
transport and relative motion.

The flux closure's roughness lengths and zero-plane displacement may be per-cell
`Field`s at `(Center, Center, Nothing)` on `grid` — for example from
`urban_roughness` or a canopy roughness closure — localized to each cell before the
Monin--Obukhov solve.
"""
function atmosphere_land_interface(grid, atmosphere, land;
                                   fluxes              = default_atmosphere_land_fluxes(land, eltype(grid)),
                                   state               = default_al_interface_state(land),
                                   velocity_difference = RelativeVelocity())
    return build_atmosphere_land_interface(grid, fluxes, state, velocity_difference)
end

function build_atmosphere_land_interface(grid, fluxes, state, velocity_difference)
    validate_flux_formulation(fluxes, grid)
    al_fluxes = AtmosphereSurfaceFluxes(grid)
    formulation = InterfaceFormulation(fluxes, state, velocity_difference)
    return AtmosphereInterface(al_fluxes, build_interface_state(temperature_formulation(state), grid), formulation)
end

@inline build_interface_state(temperature_formulation, grid) = TemperatureHumidityState(grid)
@inline build_interface_state(surface::CanopyInterface, grid) = CanopyState(grid)

function initialize_land_interface!(coupled_model)
    interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(interface) && return nothing
    exchanger = coupled_model.interfaces.exchanger
    EarthSystemModels.interpolate_state!(exchanger.land, exchanger.grid, coupled_model.land, coupled_model)
    initialize_land_interface!(coupled_model, interface)
    return nothing
end

function initialize_land_interface!(coupled_model, interface)
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    arch = architecture(grid)
    clock = coupled_model.clock
    state = interface.formulation.state
    leaf_area_index = canopy_leaf_area_index(specific_humidity_formulation(state))
    vegetation, time_interpolator = kernel_surface_field(leaf_area_index, arch, clock.time)
    launch!(arch, grid, :xy, _initialize_land_interface!, interface.state,
            exchanger.land.state.T, grid, clock, state, vegetation, time_interpolator)
    return nothing
end

@kernel function _initialize_land_interface!(state, bulk_temperature, grid, clock, state_formulation,
                                             vegetation, time_interpolator)
    i, j = @index(Global, NTuple)
    formulation = local_interface_formulation(state_formulation, i, j)
    vegetation_state = interface_vegetation_state(i, j, grid, formulation, vegetation, time_interpolator)
    T = state2dindex(bulk_temperature, i, j, grid, Time(clock.time))
    initialize_interface_state!(state, i, j, T, formulation, vegetation_state)
end

@inline function initialize_interface_state!(state, i, j, T, formulation, vegetation)
    @inbounds state.temperature[i, j, 1] = T
    return nothing
end

@inline function initialize_interface_state!(state::CanopyState, i, j, T, c, vegetation)
    transmittance = canopy_transmittance(c.extinction, c.clumping, vegetation.leaf_area_index)
    @inbounds begin
        state.temperature[i, j, 1] = T
        state.canopy[i, j, 1] = T
        state.ground[i, j, 1] = T
        state.effective[i, j, 1] = T
        state.effective_albedo[i, j, 1] = (1 - transmittance) * c.leaf_albedo + transmittance * c.ground_albedo
    end
    return nothing
end

# Initial interface values for the fixed point: the bulk land temperature and its
# saturation humidity, or the state of a prognostic interface.
@inline initial_interface_values(formulation, state, T₀, q₀) = (T₀, q₀)
@inline initial_interface_values(::CanopyInterface{<:Number}, state, T₀, q₀) = (state.temperature, state.specific_humidity)
@inline initial_interface_values(::EnergyBalanceTemperature{<:Number}, state, T₀, q₀) = (state.temperature, q₀)

# Store the diagnostics of the converged boundary solve.
@inline function store_interface_diagnostics!(Ts, i, j, formulation, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙⁱⁿ, ℙₐ, Δt)
    @inbounds begin
        Ts.temperature[i, j, 1]       = Ψⁱⁿ.temperature
        Ts.specific_humidity[i, j, 1] = Ψⁱⁿ.specific_humidity
    end
    return Ψⁱⁿ
end

@inline function store_interface_diagnostics!(Ts, i, j, c::CanopyInterface{Nothing}, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙⁱⁿ, ℙₐ, Δt)
    solution = canopy_air_space_solve(c, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, Δt)
    @inbounds begin
        Ts.temperature[i, j, 1]       = solution.Tᵃᶜᵃⁿ
        Ts.specific_humidity[i, j, 1] = solution.qᵃᶜᵃⁿ
    end
    return store_canopy_solution!(Ts, i, j, solution, Ψⁱⁿ, Ψₐ, ℙₐ)
end

# A prognostic node is held fixed.
@inline function store_interface_diagnostics!(Ts, i, j, c::CanopyInterface, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙⁱⁿ, ℙₐ, Δt)
    solution = canopy_air_space_solve(c, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, Δt)
    return store_canopy_solution!(Ts, i, j, solution, Ψⁱⁿ, Ψₐ, ℙₐ)
end

@inline function store_canopy_solution!(Ts, i, j, solution, Ψⁱⁿ, Ψₐ, ℙₐ)
    @inbounds begin
        Ts.canopy[i, j, 1]                 = solution.Tˡᵉᵃᶠ
        Ts.ground[i, j, 1]                 = solution.Tᵍʳⁿᵈ
        Ts.effective[i, j, 1]              = solution.effective_temperature
        Ts.effective_albedo[i, j, 1]       = solution.effective_albedo
        Ts.ground_heat_flux[i, j, 1]       = solution.𝒬ᵍʳⁿᵈ
        Ts.canopy_latent_heat[i, j, 1]     = solution.𝒬ᵛˡᵉᵃᶠ
        Ts.ground_latent_heat[i, j, 1]     = solution.𝒬ᵛᵍʳⁿᵈ
        Ts.canopy_sensible_heat[i, j, 1]   = solution.𝒬ᵀˡᵉᵃᶠ
        Ts.ground_sensible_heat[i, j, 1]   = solution.𝒬ᵀᵍʳⁿᵈ
        Ts.canopy_evaporation[i, j, 1]     = solution.Jʷᵉᵗ
        Ts.wet_canopy_latent_heat[i, j, 1] = solution.ℒ * solution.Jʷᵉᵗ
        Ts.land_vapor_flux[i, j, 1]        = (solution.𝒬ᵛˡᵉᵃᶠ + solution.𝒬ᵛᵍʳⁿᵈ) / solution.ℒ - solution.Jʷᵉᵗ
        Ts.transpiration[i, j, 1]          = solution.𝒬ᵛˡᵉᵃᶠ / solution.ℒ - solution.Jʷᵉᵗ
    end

    # The exported scales are evaluated at the node the fluxes above were partitioned at.
    return rebuild_interface_state(Ψⁱⁿ, Ψₐ, ℙₐ, solution.Tᵃᶜᵃⁿ, solution.qᵃᶜᵃⁿ)
end

@inline step_interface_state!(Ts, i, j, formulation, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙⁱⁿ, ℙₐ, Δt) =
    store_interface_diagnostics!(Ts, i, j, formulation, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙⁱⁿ, ℙₐ, Δt)

# The node advances by its exact relaxation toward the equilibrium the skins were solved against.
# The returned state carries the step-mean node of the fluxes.
@inline function step_interface_state!(Ts, i, j, c::CanopyInterface{<:Number}, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙⁱⁿ, ℙₐ, Δt)
    solution = canopy_air_space_solve(c, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, Δt)
    @inbounds begin
        Ts.temperature[i, j, 1]       = advance_canopy_air(Ψⁱⁿ.temperature, solution.equilibrium_temperature, solution.Σgᵀ, solution.Cᵀ, Δt)
        Ts.specific_humidity[i, j, 1] = advance_canopy_air(Ψⁱⁿ.specific_humidity, solution.equilibrium_humidity, solution.Σgᵛ, solution.Cᵛ, Δt)
    end
    return store_canopy_solution!(Ts, i, j, solution, Ψⁱⁿ, Ψₐ, ℙₐ)
end

# Backward Euler for skin heat capacity, with Newton refinement of radiation and vapor fluxes.
@inline function step_interface_state!(Ts, i, j, t::EnergyBalanceTemperature{<:Number}, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙⁱⁿ, ℙₐ, Δt)
    FT = eltype(Ψⁱⁿ)
    Tⁱⁿ = Ψⁱⁿ.temperature
    C  = convert(FT, t.heat_capacity)
    Δt⁻¹ = ifelse(Δt > 0, 1 / Δt, zero(FT))

    Tⁿ⁺¹ = Tⁱⁿ
    for _ in 1:3
        F, Σg = skin_energy_imbalance(Tⁿ⁺¹, t, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙⁱⁿ, ℙₐ)
        R    = C * (Tⁿ⁺¹ - Tⁱⁿ) * Δt⁻¹ - F
        dR   = C * Δt⁻¹ + Σg
        Tⁿ⁺¹ = ifelse(dR > 0, Tⁿ⁺¹ - R / dR, Tⁿ⁺¹)
    end

    qⁿ⁺¹ = compute_interface_humidity(specific_humidity_formulation(ℙⁱⁿ.state), Tⁿ⁺¹, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)
    Ψⁿ⁺¹ = rebuild_interface_state(Ψⁱⁿ, Ψₐ, ℙₐ, Tⁿ⁺¹, qⁿ⁺¹)
    return store_interface_diagnostics!(Ts, i, j, t, Ψⁿ⁺¹, Ψₐ, Ψˡᵃ, Ψᵣ, ℙⁱⁿ, ℙₐ, Δt)
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

Diagnose atmosphere–land fluxes from the current land and interface state.
Prognostic interface state is held fixed.
"""
compute_atmosphere_land_fluxes!(coupled_model) =
    compute_atmosphere_land_fluxes!(coupled_model, coupled_model.interfaces.atmosphere_land_interface)

compute_atmosphere_land_fluxes!(coupled_model, ::Nothing) = nothing

function compute_atmosphere_land_fluxes!(coupled_model, interface, formulation = interface.formulation)
    grid = coupled_model.interfaces.exchanger.grid
    solve_inputs = atmosphere_land_solve_inputs(coupled_model, formulation, interface.state)

    # Interior cells only: halo cells of the atmosphere exchanger state are uninitialized when the
    # atmosphere grid is a regional cutout matching the exchange-grid interior exactly.
    launch!(architecture(grid), grid, :xy, _compute_atmosphere_land_interface_state!,
            interface.fluxes, interface.state, solve_inputs)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Advance prognostic interface state over `Δt`, and store the step-mean atmosphere–land fluxes.
"""
function step_atmosphere_land_interface!(coupled_model, interface, Δt)
    grid = coupled_model.interfaces.exchanger.grid
    solve_inputs = atmosphere_land_solve_inputs(coupled_model, interface.formulation, interface.state)
    launch!(architecture(grid), grid, :xy, _step_atmosphere_land_interface!,
            interface.fluxes, interface.state, solve_inputs, convert(eltype(grid), Δt))
    return nothing
end

"""
$(TYPEDSIGNATURES)

Set prognostic interface state to the solution of a massless interface from the bulk land state.
"""
initialize_atmosphere_land_state!(coupled_model, interface) =
    compute_atmosphere_land_fluxes!(coupled_model, interface, massless(interface.formulation))

function atmosphere_land_solve_inputs(coupled_model, formulation, interface_state)
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    arch = architecture(grid)
    clock = coupled_model.clock

    atmosphere_state = merge(exchanger.atmosphere.state,
                             (; h_bℓ = boundary_layer_height(coupled_model.atmosphere)))

    atmosphere_properties = (thermodynamics_parameters = thermodynamics_parameters(coupled_model.atmosphere),
                             surface_layer_height = coupled_model.interfaces.properties.surface_layer_height,
                             gravitational_acceleration = coupled_model.interfaces.properties.gravitational_acceleration)

    land_state = merge(exchanger.land.state,
                       (; last_Δt = clock.last_Δt,
                          thermal_exchange_depth = thermal_exchange_depth(temperature_formulation(formulation.state))))

    # Prescribed leaf area index off the canopy formulation (or `nothing`),
    # reduced to a kernel-friendly value plus its host-side time interpolator.
    leaf_area_index = canopy_leaf_area_index(specific_humidity_formulation(formulation.state))
    vegetation, time_interpolator = kernel_surface_field(leaf_area_index, arch, clock.time)

    radiation_exchanger = exchanger.radiation

    return (; grid, clock, formulation, land_state, vegetation, time_interpolator,
              stored_state = (; interface_state.temperature, interface_state.specific_humidity),
              atmosphere_state, atmosphere_properties,
              radiation_properties = kernel_radiation_properties(coupled_model.radiation, radiation_exchanger),
              radiation_state = isnothing(radiation_exchanger) ? nothing : radiation_exchanger.state)
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

Return the canopy interception store `Mᶜⁿᵖʸ` (kg m⁻²) on the land grid.
"""
function land_canopy_water_storage end

"""
    land_canopy_water_capacity(land)

Return the canopy interception capacity `Mᶜⁿᵖʸ⁺` (kg m⁻²) on the land grid.
"""
function land_canopy_water_capacity end

# Vegetation state, per humidity formulation. Only the canopy formulations
# (defined in their own files) pull a leaf area index; everything else is empty.
@inline interface_vegetation_state(i, j, grid, interface_model, vegetation, time_interpolator) = (;)

@kernel function _compute_atmosphere_land_interface_state!(interface_fluxes, interface_state, solve_inputs)
    i, j = @index(Global, NTuple)
    Ψⁱⁿ, Ψᵃᵗ, Ψˡᵃ, Ψᵣ, formulation = solve_atmosphere_land_interface(i, j, solve_inputs, 0)
    ℙₐ = solve_inputs.atmosphere_properties
    Ψⁱⁿ = store_interface_diagnostics!(interface_state, i, j, temperature_formulation(formulation.state),
                                       Ψⁱⁿ, Ψᵃᵗ, Ψˡᵃ, Ψᵣ, formulation, ℙₐ, 0)
    store_atmosphere_land_fluxes!(interface_fluxes, i, j, Ψⁱⁿ, Ψᵃᵗ, ℙₐ, formulation)
end

@kernel function _step_atmosphere_land_interface!(interface_fluxes, interface_state, solve_inputs, Δt)
    i, j = @index(Global, NTuple)
    Ψⁱⁿ, Ψᵃᵗ, Ψˡᵃ, Ψᵣ, formulation = solve_atmosphere_land_interface(i, j, solve_inputs, Δt)
    ℙₐ = solve_inputs.atmosphere_properties
    Ψⁱⁿ = step_interface_state!(interface_state, i, j, temperature_formulation(formulation.state),
                                Ψⁱⁿ, Ψᵃᵗ, Ψˡᵃ, Ψᵣ, formulation, ℙₐ, Δt)
    store_atmosphere_land_fluxes!(interface_fluxes, i, j, Ψⁱⁿ, Ψᵃᵗ, ℙₐ, formulation)
end

# The similarity fixed point at one cell. `Δt` sets the canopy-air step mean.
@inline function solve_atmosphere_land_interface(i, j, inputs, Δt)
    grid = inputs.grid
    time = Time(inputs.clock.time)
    interface_formulation = inputs.formulation
    ℙₐ = inputs.atmosphere_properties

    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    Ψᵃᵗ = local_atmosphere_state(i, j, inputs.atmosphere_state, ℙₐ)

    # The land state at this cell, with the canopy's prescribed vegetation (leaf area index).
    vegetation_state = interface_vegetation_state(i, j, grid, specific_humidity_formulation(interface_formulation.state),
                                                  inputs.vegetation, inputs.time_interpolator)
    Ψˡᵃ = merge(state2dindex(inputs.land_state, i, j, grid, time), vegetation_state)
    # `typeof(Ψˡᵃ.T)`, not `eltype(grid)`, which is a traced type inside a Reactant kernel.
    FT  = typeof(Ψˡᵃ.T)

    formulation = local_interface_formulation(interface_formulation, i, j)

    radiation_state = air_land_interface_radiation_state(inputs.radiation_properties,
                                                         inputs.radiation_state,
                                                         i, j, 1, grid, time)

    u★  = convert(FT, 1e-4)
    qⁱⁿ = convert(FT, saturation_specific_humidity(ℂᵃᵗ, Ψˡᵃ.T, Ψᵃᵗ.p, interface_phase(specific_humidity_formulation(formulation.state))))
    Ψˢ = state2dindex(inputs.stored_state, i, j, grid, time)
    Tⁱⁿ, qⁱⁿ = initial_interface_values(temperature_formulation(formulation.state), Ψˢ, Ψˡᵃ.T, qⁱⁿ)
    Ψⁱⁿ⁰ = AirLandInterfaceState(u★, u★, u★, zero(FT), zero(FT), convert(FT, Tⁱⁿ), convert(FT, qⁱⁿ))

    Ψⁱⁿ = compute_interface_state(formulation, Ψⁱⁿ⁰, Ψᵃᵗ, Ψˡᵃ, radiation_state, ℙₐ, (;), convert(FT, Δt))

    return Ψⁱⁿ, Ψᵃᵗ, Ψˡᵃ, radiation_state, formulation
end

@inline function store_atmosphere_land_fluxes!(interface_fluxes, i, j, Ψⁱⁿ, Ψᵃᵗ, ℙₐ, formulation)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    ℒˡ = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Ψᵃᵗ.T)
    store_interface_fluxes!(interface_fluxes, i, j, Ψⁱⁿ, Ψᵃᵗ, ℂᵃᵗ, ℒˡ, formulation)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Surface energy loss and water fluxes consumed by land, positive upward (W m⁻² and kg m⁻² s⁻¹).
"""
land_surface_fluxes(interface::AtmosphereInterface, Tˡᵃ) = land_surface_fluxes(interface.state, interface, Tˡᵃ)

# TODO: Use KernelFunctionOperation after Oceananigans reconciles its Reactant traced
# scalar type with the unwrapped element type produced by GPU adaptation.
struct LandSurfaceEnergyFlux{G, T, F, L, C}
    grid :: G
    interface_temperature :: T
    fluxes :: F
    land_temperature :: L
    ground_conductance :: C
end

Adapt.@adapt_structure LandSurfaceEnergyFlux

@inline Base.getindex(𝒬::LandSurfaceEnergyFlux, i, j, k) =
    surface_energy_loss(i, j, 𝒬.grid, 𝒬.interface_temperature, 𝒬.fluxes, 𝒬.land_temperature, 𝒬.ground_conductance)

function land_surface_fluxes(state, interface, Tˡᵃ)
    Tⁱⁿ = state.temperature
    fluxes = interface.fluxes
    Λᵍʳⁿᵈ = ground_conductance(interface)
    energy = LandSurfaceEnergyFlux(Tⁱⁿ.grid, Tⁱⁿ, fluxes, Tˡᵃ, Λᵍʳⁿᵈ)
    return (; energy, vapor=fluxes.water_vapor, canopy_evaporation=ZeroField())
end

land_surface_fluxes(canopy::CanopyState, interface, Tˡᵃ) =
    (; energy=-canopy.ground_heat_flux, vapor=canopy.land_vapor_flux,
       canopy_evaporation=canopy.canopy_evaporation)

@inline surface_energy_loss(i, j, grid, Tⁱⁿ, fluxes, Tˡᵃ, ::Nothing) =
    @inbounds fluxes.sensible_heat[i, j, 1] + fluxes.latent_heat[i, j, 1]

@inline surface_energy_loss(i, j, grid, Tⁱⁿ, fluxes, Tˡᵃ, Λᵍʳⁿᵈ) =
    @inbounds -Λᵍʳⁿᵈ * (Tⁱⁿ[i, j, 1] - Tˡᵃ[i, j, 1])

"""
$(TYPEDSIGNATURES)

Surface temperature, albedo, and emissivity for outgoing radiation.
"""
radiative_surface(interface::AtmosphereInterface, albedo, emissivity) =
    radiative_surface(interface.state, albedo, emissivity)
radiative_surface(state, albedo, emissivity) = (; state.temperature, albedo, emissivity)
radiative_surface(canopy::CanopyState, albedo, emissivity) =
    (; temperature=canopy.effective, albedo=canopy.effective_albedo, emissivity=1)

#####
##### Which interfaces are prognostic
#####

# Names of the interface state variables that are stepped rather than diagnosed.
prognostic_state_names(formulation) = ()
prognostic_state_names(c::CanopyInterface) = isnothing(c.canopy_air_depth) ? () : (:T, :q)
prognostic_state_names(t::EnergyBalanceTemperature) = isnothing(t.heat_capacity) ? () : (:T,)
prognostic_state_names(interface::AtmosphereInterface) =
    prognostic_state_names(temperature_formulation(interface.formulation.state))

has_prognostic_state(interface) = false
has_prognostic_state(interface::AtmosphereInterface) = !isempty(prognostic_state_names(interface))

#####
##### Setting prognostic interface state
#####

"""
$(TYPEDSIGNATURES)

Set the prognostic state of the atmosphere--land interface, e.g.
`set!(model.interfaces; atmosphere_land_interface = (; T = 288, q = 0.01))`.
Every prognostic variable must be given: `T` and `q` (the canopy-air node) for a
[`CanopyInterface`](@ref) with a `canopy_air_depth`, and `T` for an
[`EnergyBalanceTemperature`](@ref) with a `heat_capacity`. A [`TiledLandInterface`](@ref)
takes one entry per tile, `(; vegetated = (; T, q), bare = (; T))`. Each value is anything
`Oceananigans.set!` accepts. The set state replaces the massless initial state, and the
fluxes follow it at the next `update_state!`.
"""
function Oceananigans.set!(interfaces::ComponentInterfaces; atmosphere_land_interface)
    set_interface_state!(interfaces.atmosphere_land_interface; atmosphere_land_interface...)
    interfaces.state_is_set = true
    return interfaces
end

function set_interface_state!(interface::AtmosphereInterface; values...)
    names = prognostic_state_names(interface)
    issetequal(keys(values), names) ||
        throw(ArgumentError("the interface's prognostic state is $names; got $(Tuple(keys(values)))"))
    haskey(values, :T) && Oceananigans.set!(interface.state.temperature, values[:T])
    haskey(values, :q) && Oceananigans.set!(interface.state.specific_humidity, values[:q])
    return nothing
end
