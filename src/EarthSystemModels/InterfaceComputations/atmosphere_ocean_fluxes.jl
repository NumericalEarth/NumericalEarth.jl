using Oceananigans.Grids: inactive_node

atmosphere_ocean_fields(coupled_model) = coupled_model.interfaces.exchanger.atmosphere.state

atmosphere_ocean_data(coupled_model) = merge(atmosphere_ocean_fields(coupled_model),
                                             (; h_bℓ = boundary_layer_height(coupled_model.atmosphere)))

atmosphere_ocean_properties(coupled_model) = (; thermodynamics_parameters = thermodynamics_parameters(coupled_model.atmosphere),
                                                surface_layer_height = coupled_model.interfaces.properties.surface_layer_height,
                                                gravitational_acceleration = coupled_model.interfaces.properties.gravitational_acceleration)

atmosphere_ocean_radiation_state(coupled_model) = begin
    radiation_exchanger = coupled_model.interfaces.exchanger.radiation
    return isnothing(radiation_exchanger) ? nothing : radiation_exchanger.state
end

function compute_atmosphere_ocean_fluxes!(coupled_model)
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    arch = architecture(grid)
    clock = coupled_model.clock
    # Simplify NamedTuple to reduce parameter space consumption.
    # See https://github.com/CliMA/NumericalEarth.jl/issues/116.
    atmosphere_data = atmosphere_ocean_data(coupled_model)

    flux_formulation = coupled_model.interfaces.atmosphere_ocean_interface.flux_formulation
    interface_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
    interface_temperature = coupled_model.interfaces.atmosphere_ocean_interface.temperature
    interface_properties = coupled_model.interfaces.atmosphere_ocean_interface.properties
    ocean_properties = coupled_model.interfaces.ocean_properties
    atmosphere_properties = atmosphere_ocean_properties(coupled_model)
    ocean_state = assemble_interior_fields(exchanger.ocean.state, interface_properties.temperature_formulation)

    # Radiation state for the interface solve (used by SkinTemperature).
    # When `radiation === nothing` these are `nothing`s and the getter
    # returns zero-valued radiative state, so SkinTemperature degrades to
    # a turbulent-only flux balance.
    radiation = coupled_model.radiation
    radiation_kernel_props = kernel_radiation_properties(radiation)
    radiation_state = atmosphere_ocean_radiation_state(coupled_model)

    kernel_parameters = interface_kernel_parameters(grid)

    launch!(arch, grid, kernel_parameters,
            _compute_atmosphere_ocean_interface_state!,
            interface_fluxes,
            interface_temperature,
            grid,
            clock,
            flux_formulation,
            ocean_state,
            atmosphere_data,
            interface_properties,
            atmosphere_properties,
            ocean_properties,
            radiation_kernel_props,
            radiation_state)

    return nothing
end

@inline function assemble_interior_state(i, j, kᴺ, grid, interior_state, ocean_properties, temperature_formulation)
    @inbounds begin
        uᵒᶜ = ℑxᶜᵃᵃ(i, j, kᴺ, grid, interior_state.u)
        vᵒᶜ = ℑyᵃᶜᵃ(i, j, kᴺ, grid, interior_state.v)
        Tᵒᶜ = interior_state.T[i, j, kᴺ]
        Tᵒᶜ = convert_to_kelvin(ocean_properties.temperature_units, Tᵒᶜ)
        Sᵒᶜ = interior_state.S[i, j, kᴺ]
    end
    return (u=uᵒᶜ, v=vᵒᶜ, T=Tᵒᶜ, S=Sᵒᶜ)
end

@inline function assemble_interior_state(i, j, kᴺ, grid, interior_state, ocean_properties, ::IDST)
    Ψᵢ = assemble_interior_state(i, j, kᴺ, grid, interior_state, ocean_properties, nothing)
    κ  = @inbounds interior_state.κ[i, j, 1]
    return merge(Ψᵢ, (; κ))
end

""" Compute turbulent fluxes between an atmosphere and an interface state using similarity theory """
@kernel function _compute_atmosphere_ocean_interface_state!(interface_fluxes,
                                                            interface_temperature,
                                                            grid,
                                                            clock,
                                                            turbulent_flux_formulation,
                                                            interior_state,
                                                            atmosphere_state,
                                                            interface_properties,
                                                            atmosphere_properties,
                                                            ocean_properties,
                                                            radiation_kernel_props,
                                                            radiation_exchanger_state)

    i, j = @index(Global, NTuple)
    kᴺ   = size(grid, 3) # index of the top ocean cell
    time = Time(clock.time)

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    Ψₐ  = local_atmosphere_state(i, j, atmosphere_state, atmosphere_properties)

    local_interior_state = assemble_interior_state(i, j, kᴺ, grid, interior_state, ocean_properties, interface_properties.temperature_formulation)
    Tᵒᶜ = local_interior_state.T
    Sᵒᶜ = local_interior_state.S

    radiation_state = air_sea_interface_radiation_state(radiation_kernel_props,
                                                        radiation_exchanger_state,
                                                        i, j, kᴺ, grid, time)

    FT = typeof(Tᵒᶜ)
    u★ = convert(FT, 1e-4)

    # Estimate interface specific humidity using interior temperature
    q_formulation = interface_properties.specific_humidity_formulation
    qₛ = surface_specific_humidity(q_formulation, ℂᵃᵗ, Ψₐ.p, Tᵒᶜ, Sᵒᶜ)
    initial_interface_state = AirSeaInterfaceState(u★, u★, u★, local_interior_state.u, local_interior_state.v, Tᵒᶜ, Sᵒᶜ, qₛ)

    # Don't use convergence criteria in an inactive cell
    stop_criteria = turbulent_flux_formulation.solver_stop_criteria
    needs_to_converge = stop_criteria isa ConvergenceStopCriteria
    not_water = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    if needs_to_converge && not_water
        interface_state = zero_interface_state(FT)
    else
        interface_state = compute_interface_state(turbulent_flux_formulation,
                                                  initial_interface_state,
                                                  Ψₐ,
                                                  local_interior_state,
                                                  radiation_state,
                                                  interface_properties,
                                                  atmosphere_properties,
                                                  ocean_properties)
    end

    # In the case of FixedIterations, make sure interface state is zero'd
    Ψₛ = ifelse(not_water, zero_interface_state(FT), interface_state)
    ℒˡ = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Ψₐ.T)
    Tₛ = convert_from_kelvin(ocean_properties.temperature_units, Ψₛ.temperature)

    store_interface_fluxes!(interface_fluxes, interface_temperature, i, j, Ψₛ, Ψₐ, ℂᵃᵗ, ℒˡ, Tₛ, interface_properties)
end
