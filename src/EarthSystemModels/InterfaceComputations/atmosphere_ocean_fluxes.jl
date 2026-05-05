using Oceananigans.Operators: intrinsic_vector
using Oceananigans.Grids: inactive_node

function compute_atmosphere_ocean_fluxes!(coupled_model)
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    arch = architecture(grid)
    clock = coupled_model.clock
    ocean_state = exchanger.ocean.state
    atmosphere_fields = exchanger.atmosphere.state

    # Simplify NamedTuple to reduce parameter space consumption.
    # See https://github.com/CliMA/NumericalEarth.jl/issues/116.
    atmosphere_data = merge(atmosphere_fields,
                            (; h_bℓ = boundary_layer_height(coupled_model.atmosphere)))

    flux_formulation = coupled_model.interfaces.atmosphere_ocean_interface.flux_formulation
    interface_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
    interface_temperature = coupled_model.interfaces.atmosphere_ocean_interface.temperature
    interface_properties = coupled_model.interfaces.atmosphere_ocean_interface.properties
    ocean_properties = coupled_model.interfaces.ocean_properties
    atmosphere_properties = (thermodynamics_parameters = thermodynamics_parameters(coupled_model.atmosphere),
                             surface_layer_height = surface_layer_height(coupled_model.atmosphere),
                             gravitational_acceleration = coupled_model.interfaces.properties.gravitational_acceleration)

    # Radiation state for the interface solve (used by SkinTemperature).
    # When `radiation === nothing` these are `nothing`s and the getter
    # returns zero-valued radiative state, so SkinTemperature degrades to
    # a turbulent-only flux balance.
    radiation = coupled_model.radiation
    radiation_kernel_props = kernel_radiation_properties(radiation)
    radiation_exchanger    = exchanger.radiation
    radiation_state        = isnothing(radiation_exchanger) ? nothing : radiation_exchanger.state

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

    @inbounds begin
        uᵃᵗ = atmosphere_state.u[i, j, 1]
        vᵃᵗ = atmosphere_state.v[i, j, 1]
        Tᵃᵗ = atmosphere_state.T[i, j, 1]
        pᵃᵗ = atmosphere_state.p[i, j, 1]
        qᵃᵗ = atmosphere_state.q[i, j, 1]

        # Ocean state at cell centers
        uᵒᶜ = ℑxᶜᵃᵃ(i, j, kᴺ, grid, interior_state.u)
        vᵒᶜ = ℑyᵃᶜᵃ(i, j, kᴺ, grid, interior_state.v)
        Tᵒᶜ = interior_state.T[i, j, kᴺ]
        Tᵒᶜ = convert_to_kelvin(ocean_properties.temperature_units, Tᵒᶜ)
        Sᵒᶜ = interior_state.S[i, j, kᴺ]
    end

    # Build thermodynamic and dynamic states in the atmosphere and interface.
    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    zᵃᵗ = atmosphere_properties.surface_layer_height # elevation of atmos variables relative to interface

    local_atmosphere_state = (z = zᵃᵗ,
                              u = uᵃᵗ,
                              v = vᵃᵗ,
                              T = Tᵃᵗ,
                              p = pᵃᵗ,
                              q = qᵃᵗ,
                              h_bℓ = atmosphere_state.h_bℓ)

    local_interior_state = (u=uᵒᶜ, v=vᵒᶜ, T=Tᵒᶜ, S=Sᵒᶜ)

    # Local radiative state at this cell. Returns zero-valued state when
    # radiation is off.
    radiation_state = air_sea_interface_radiation_state(radiation_kernel_props,
                                                        radiation_exchanger_state,
                                                        i, j, kᴺ, grid, time)

    # Estimate initial interface state
    FT = typeof(Tᵒᶜ)
    u★ = convert(FT, 1e-4)

    # Estimate interface specific humidity using interior temperature
    q_formulation = interface_properties.specific_humidity_formulation
    qₛ = surface_specific_humidity(q_formulation, ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ, Tᵒᶜ, Sᵒᶜ)
    initial_interface_state = InterfaceState(u★, u★, u★, uᵒᶜ, vᵒᶜ, Tᵒᶜ, Sᵒᶜ, qₛ)

    # Don't use convergence criteria in an inactive cell
    stop_criteria = turbulent_flux_formulation.solver_stop_criteria
    needs_to_converge = stop_criteria isa ConvergenceStopCriteria
    not_water = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    if needs_to_converge && not_water
        interface_state = zero_interface_state(FT)
    else
        interface_state = compute_interface_state(turbulent_flux_formulation,
                                                  initial_interface_state,
                                                  local_atmosphere_state,
                                                  local_interior_state,
                                                  radiation_state,
                                                  interface_properties,
                                                  atmosphere_properties,
                                                  ocean_properties)
    end

    # In the case of FixedIterations, make sure interface state is zero'd
    interface_state = ifelse(not_water, zero_interface_state(FT), interface_state)

    u★ = interface_state.u★
    θ★ = interface_state.θ★
    q★ = interface_state.q★

    Ψₛ = interface_state
    Ψₐ = local_atmosphere_state
    Δu, Δv = velocity_difference(interface_properties.velocity_formulation, Ψₐ, Ψₛ)
    ΔU = sqrt(Δu^2 + Δv^2)

    τˣ = ifelse(ΔU == 0, zero(grid), - u★^2 * Δu / ΔU)
    τʸ = ifelse(ΔU == 0, zero(grid), - u★^2 * Δv / ΔU)

    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    cᵖᵐ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ) # moist heat capacity
    ℒˡ = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Tᵃᵗ)

    # Store fluxes
    𝒬ᵛ  = interface_fluxes.latent_heat
    𝒬ᵀ  = interface_fluxes.sensible_heat
    Jᵛ  = interface_fluxes.water_vapor
    ρτˣ = interface_fluxes.x_momentum
    ρτʸ = interface_fluxes.y_momentum
    Ts  = interface_temperature

    @inbounds begin
        # +0: cooling, -0: heating
        𝒬ᵛ[i, j, 1]  = - ρᵃᵗ * ℒˡ * u★ * q★
        𝒬ᵀ[i, j, 1]  = - ρᵃᵗ * cᵖᵐ * u★ * θ★
        Jᵛ[i, j, 1]  = - ρᵃᵗ * u★ * q★
        ρτˣ[i, j, 1] = + ρᵃᵗ * τˣ
        ρτʸ[i, j, 1] = + ρᵃᵗ * τʸ
        Ts[i, j, 1]  = convert_from_kelvin(ocean_properties.temperature_units, Ψₛ.T)

        interface_fluxes.friction_velocity[i, j, 1] = u★
        interface_fluxes.temperature_scale[i, j, 1] = θ★
        interface_fluxes.water_vapor_scale[i, j, 1] = q★
    end
end
