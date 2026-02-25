using Oceananigans.Operators: intrinsic_vector
using Oceananigans.Grids: inactive_node
using Oceananigans.Fields: ZeroField

function compute_atmosphere_sea_ice_fluxes!(coupled_model)
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    arch = architecture(grid)
    clock = coupled_model.clock

    interior_state = merge(exchanger.sea_ice.state,
                           (; Tᵒᶜ = exchanger.ocean.state.T,
                              Sᵒᶜ = exchanger.ocean.state.S))

    atmosphere_fields = exchanger.atmosphere.state

    # Simplify NamedTuple to reduce parameter space consumption.
    # See https://github.com/CliMA/NumericalEarth.jl/issues/116.
    atmosphere_data = merge(atmosphere_fields, 
                            (; h_bℓ = boundary_layer_height(coupled_model.atmosphere)))

    flux_formulation = coupled_model.interfaces.atmosphere_sea_ice_interface.flux_formulation
    interface_fluxes = coupled_model.interfaces.atmosphere_sea_ice_interface.fluxes
    interface_temperature = coupled_model.interfaces.atmosphere_sea_ice_interface.temperature
    interface_properties = coupled_model.interfaces.atmosphere_sea_ice_interface.properties
    sea_ice_properties = coupled_model.interfaces.sea_ice_properties
    ocean_properties = coupled_model.interfaces.ocean_properties

    atmosphere_properties = (thermodynamics_parameters = thermodynamics_parameters(coupled_model.atmosphere),
                             surface_layer_height = surface_layer_height(coupled_model.atmosphere),
                             gravitational_acceleration = coupled_model.interfaces.properties.gravitational_acceleration)

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
            ocean_properties)

    return nothing
end

""" Compute turbulent fluxes between an atmosphere and a interface state using similarity theory """
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
                                                              ocean_properties)

    i, j = @index(Global, NTuple)
    kᴺ   = size(grid, 3) # index of the top ocean cell
    FT   = eltype(grid)

    @inbounds begin
        uᵃᵗ = atmosphere_state.u[i, j, 1]
        vᵃᵗ = atmosphere_state.v[i, j, 1]
        Tᵃᵗ = atmosphere_state.T[i, j, 1]
        pᵃᵗ = atmosphere_state.p[i, j, 1]
        qᵃᵗ = atmosphere_state.q[i, j, 1]
        ℐꜜˢʷ = atmosphere_state.ℐꜜˢʷ[i, j, 1]
        ℐꜜˡʷ = atmosphere_state.ℐꜜˡʷ[i, j, 1]

        # Extract state variables at cell centers
        # Ocean properties below sea ice
        Tᵒᶜ = interior_state.Tᵒᶜ[i, j, kᴺ]
        Tᵒᶜ = convert_to_kelvin(ocean_properties.temperature_units, Tᵒᶜ)
        Sᵒᶜ = interior_state.Sᵒᶜ[i, j, kᴺ]

        # Sea ice properties
        uˢⁱ = zero(FT) # ℑxᶜᵃᵃ(i, j, 1, grid, interior_state.u)
        vˢⁱ = zero(FT) # ℑyᵃᶜᵃ(i, j, 1, grid, interior_state.v)
        hˢⁱ = interior_state.h[i, j, 1]
        hc = interior_state.hc[i, j, 1]
        ℵᵢ = interior_state.ℵ[i, j, 1]
        Tₛ = interface_temperature[i, j, 1]
        Tₛ = convert_to_kelvin(sea_ice_properties.temperature_units, Tₛ)
    end

    # Build thermodynamic and dynamic states in the atmosphere and interface.
    # Notation:
    #   ⋅ 𝒰 ≡ "dynamic" state vector (thermodynamics + reference height + velocity)
    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    zᵃᵗ = atmosphere_properties.surface_layer_height # elevation of atmos variables relative to interface

    local_atmosphere_state = (z = zᵃᵗ,
                              u = uᵃᵗ,
                              v = vᵃᵗ,
                              T = Tᵃᵗ,
                              p = pᵃᵗ,
                              q = qᵃᵗ,
                              h_bℓ = atmosphere_state.h_bℓ)

    downwelling_radiation = (; ℐꜜˢʷ, ℐꜜˡʷ)
    local_interior_state = (u=uˢⁱ, v=vˢⁱ, T=Tᵒᶜ, S=Sᵒᶜ, h=hˢⁱ, hc=hc)
    
    # Estimate initial interface state (FP32 compatible)
    u★ = convert(FT, 1f-4)

    # Estimate interface specific humidity using interior temperature
    q_formulation = interface_properties.specific_humidity_formulation
    qₛ = surface_specific_humidity(q_formulation, ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ, Tₛ, Sᵒᶜ)

    # Guess
    Sₛ = zero(FT) # what should we use for interface salinity?
    initial_interface_state = InterfaceState(u★, u★, u★, uˢⁱ, vˢⁱ, Tₛ, Sₛ, convert(FT, qₛ))
    not_water = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())
    ice_free = ℵᵢ == 0

    stop_criteria = turbulent_flux_formulation.solver_stop_criteria
    needs_to_converge = stop_criteria isa ConvergenceStopCriteria

    if (needs_to_converge && not_water) || ice_free
        interface_state = InterfaceState(zero(FT), zero(FT), zero(FT), uˢⁱ, vˢⁱ, Tᵒᶜ, Sₛ, zero(FT))
    else
        interface_state = compute_interface_state(turbulent_flux_formulation,
                                                  initial_interface_state,
                                                  local_atmosphere_state,
                                                  local_interior_state,
                                                  downwelling_radiation,
                                                  interface_properties,
                                                  atmosphere_properties,
                                                  sea_ice_properties)
    end

    u★ = interface_state.u★
    θ★ = interface_state.θ★
    q★ = interface_state.q★
    Ψₛ = interface_state
    Ψₐ = local_atmosphere_state
    Δu, Δv = velocity_difference(interface_properties.velocity_formulation, Ψₐ, Ψₛ)
    ΔU = sqrt(Δu^2 + Δv^2)
    τˣ = ifelse(ΔU == 0, zero(ΔU), - u★^2 * Δu / ΔU)
    τʸ = ifelse(ΔU == 0, zero(ΔU), - u★^2 * Δv / ΔU)

    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    cᵖᵐ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ) # moist heat capacity
    ℒⁱ = AtmosphericThermodynamics.latent_heat_sublim(ℂᵃᵗ, Tᵃᵗ)

    # Store fluxes
    𝒬ᵛ = interface_fluxes.latent_heat
    𝒬ᵀ = interface_fluxes.sensible_heat
    Jᵛ = interface_fluxes.water_vapor
    ρτˣ = interface_fluxes.x_momentum
    ρτʸ = interface_fluxes.y_momentum
    Ts = interface_temperature

    @inbounds begin
        # +0: cooling, -0: heating
        𝒬ᵛ[i, j, 1]  = - ρᵃᵗ * u★ * q★ * ℒⁱ
        𝒬ᵀ[i, j, 1]  = - ρᵃᵗ * cᵖᵐ * u★ * θ★
        Jᵛ[i, j, 1]  = - ρᵃᵗ * u★ * q★
        ρτˣ[i, j, 1] = + ρᵃᵗ * τˣ
        ρτʸ[i, j, 1] = + ρᵃᵗ * τʸ
        Ts[i, j, 1]  = convert_from_kelvin(sea_ice_properties.temperature_units, Ψₛ.T)
    end
end
