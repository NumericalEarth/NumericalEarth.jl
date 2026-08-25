using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: inactive_node

atmosphere_sea_ice_fields(coupled_model) = coupled_model.interfaces.exchanger.atmosphere.state

atmosphere_sea_ice_data(coupled_model) = merge(atmosphere_sea_ice_fields(coupled_model),
                                               (; h_bℓ = boundary_layer_height(coupled_model.atmosphere)))

atmosphere_sea_ice_properties(coupled_model) = (; thermodynamics_parameters = thermodynamics_parameters(coupled_model.atmosphere),
                                                  surface_layer_height = coupled_model.interfaces.properties.surface_layer_height,
                                                  gravitational_acceleration = coupled_model.interfaces.properties.gravitational_acceleration)

atmosphere_sea_ice_radiation_state(coupled_model) = begin
    radiation_exchanger = coupled_model.interfaces.exchanger.radiation
    return isnothing(radiation_exchanger) ? nothing : radiation_exchanger.state
end

function compute_atmosphere_sea_ice_fluxes!(coupled_model)
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    arch = architecture(grid)
    clock = coupled_model.clock

    interior_state = merge(exchanger.sea_ice.state,
                           (; Tᵒᶜ = exchanger.ocean.state.T,
                              Sᵒᶜ = exchanger.ocean.state.S))

    atmosphere_data = atmosphere_sea_ice_data(coupled_model)

    flux_formulation = coupled_model.interfaces.atmosphere_sea_ice_interface.flux_formulation
    interface_fluxes = coupled_model.interfaces.atmosphere_sea_ice_interface.fluxes
    interface_temperature = coupled_model.interfaces.atmosphere_sea_ice_interface.temperature
    interface_properties = coupled_model.interfaces.atmosphere_sea_ice_interface.properties
    sea_ice_properties = coupled_model.interfaces.sea_ice_properties
    ocean_properties = coupled_model.interfaces.ocean_properties

    atmosphere_properties = atmosphere_sea_ice_properties(coupled_model)

    radiation = coupled_model.radiation
    radiation_kernel_props = kernel_radiation_properties(radiation)
    radiation_state = atmosphere_sea_ice_radiation_state(coupled_model)

    kernel_parameters = interface_kernel_parameters(grid)

    launch!(arch, grid, kernel_parameters,
            _compute_atmosphere_sea_ice_interface_state!,
            interface_fluxes,
            interface_temperature,
            grid,
            clock,
            flux_formulation,
            interior_state,
            atmosphere_data,
            interface_properties,
            atmosphere_properties,
            sea_ice_properties,
            ocean_properties,
            radiation_kernel_props,
            radiation_state)

    return nothing
end

""" Compute turbulent fluxes between an atmosphere and an interface state using similarity theory """
@kernel function _compute_atmosphere_sea_ice_interface_state!(interface_fluxes,
                                                              interface_temperature,
                                                              grid,
                                                              clock,
                                                              turbulent_flux_formulation,
                                                              interior_state,
                                                              atmosphere_state,
                                                              interface_properties,
                                                              atmosphere_properties,
                                                              sea_ice_properties,
                                                              ocean_properties,
                                                              radiation_kernel_props,
                                                              radiation_exchanger_state)

    i, j = @index(Global, NTuple)
    kᴺ   = size(grid, 3) # index of the top ocean cell
    FT   = eltype(grid)
    time = Time(clock.time)

    @inbounds begin
        # Ocean properties below sea ice
        Tᵒᶜ = convert_to_kelvin(ocean_properties.temperature_units, interior_state.Tᵒᶜ[i, j, kᴺ])
        Sᵒᶜ = interior_state.Sᵒᶜ[i, j, kᴺ]

        # Sea ice properties
        uˢⁱ = zero(FT) # ℑxᶜᵃᵃ(i, j, 1, grid, interior_state.u)
        vˢⁱ = zero(FT) # ℑyᵃᶜᵃ(i, j, 1, grid, interior_state.v)
        hˢⁱ = interior_state.hi[i, j, 1]
        hˢⁿ = interior_state.hs[i, j, 1]
        hc  = interior_state.hc[i, j, 1]
        ℵᵢ  = interior_state.ℵ[i, j, 1]
        Tₛ  = convert_to_kelvin(sea_ice_properties.temperature_units, interface_temperature[i, j, 1])
    end

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    Ψₐ  = local_atmosphere_state(i, j, atmosphere_state, atmosphere_properties)

    interior = (u=uˢⁱ, v=vˢⁱ, T=Tᵒᶜ, S=Sᵒᶜ, hi=hˢⁱ, hs=hˢⁿ, hc=hc)

    radiation_state = air_sea_ice_interface_radiation_state(radiation_kernel_props,
                                                            radiation_exchanger_state,
                                                            i, j, kᴺ, grid, time)

    # Estimate initial interface state (FP32 compatible)
    u★ = convert(FT, 1f-4)

    q_formulation = interface_properties.specific_humidity_formulation
    qₛ = surface_specific_humidity(q_formulation, ℂᵃᵗ, Ψₐ.p, Tₛ, Sᵒᶜ)

    # Air–ice sublimation is over fresh ice — no interface salinity.
    initial_interface_state = AirIceInterfaceState(u★, u★, u★, uˢⁱ, vˢⁱ, Tₛ, convert(FT, qₛ))
    not_water = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())
    ice_free = ℵᵢ == 0

    stop_criteria = turbulent_flux_formulation.solver_stop_criteria
    needs_to_converge = stop_criteria isa ConvergenceStopCriteria

    if (needs_to_converge && not_water) || ice_free
        Ψₛ = AirIceInterfaceState(zero(FT), zero(FT), zero(FT), uˢⁱ, vˢⁱ, Tᵒᶜ, zero(FT))
    else
        Ψₛ = compute_interface_state(turbulent_flux_formulation,
                                     initial_interface_state,
                                     Ψₐ,
                                     interior,
                                     radiation_state,
                                     interface_properties,
                                     atmosphere_properties,
                                     sea_ice_properties)
    end

    ℒⁱ = AtmosphericThermodynamics.latent_heat_sublim(ℂᵃᵗ, Ψₐ.T)
    Tᵢ = convert_from_kelvin(sea_ice_properties.temperature_units, Ψₛ.temperature)

    store_interface_fluxes!(interface_fluxes, interface_temperature, i, j, Ψₛ, Ψₐ, ℂᵃᵗ, ℒⁱ, Tᵢ, interface_properties)
    store_interface_scales!(interface_fluxes, i, j, Ψₛ)
end
