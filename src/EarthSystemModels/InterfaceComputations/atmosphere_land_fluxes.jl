using Oceananigans.Grids: inactive_node

#####
##### Atmosphere-Land interface constructor
#####
##### The atmosphere–land turbulent fluxes share their container type
##### with atmosphere–ocean ([`AtmosphereSurfaceFluxes`](@ref)); only
##### the compute kernel differs.
#####

atmosphere_land_interface(grid, ::Nothing,    land,     args...) = nothing
atmosphere_land_interface(grid, atmosphere, ::Nothing, args...) = nothing
atmosphere_land_interface(grid, ::Nothing,  ::Nothing, args...) = nothing

function atmosphere_land_interface(grid,
                                   atmosphere,
                                   land,
                                   al_flux_formulation,
                                   temperature_formulation,
                                   velocity_formulation,
                                   specific_humidity_formulation)

    al_fluxes = AtmosphereSurfaceFluxes(grid)

    al_properties = InterfaceProperties(specific_humidity_formulation,
                                        temperature_formulation,
                                        velocity_formulation)

    interface_temperature = Field{Center, Center, Nothing}(grid)

    return AtmosphereInterface(al_fluxes, al_flux_formulation,
                               interface_temperature, al_properties)
end

#####
##### Flux compute driver
#####

compute_atmosphere_land_fluxes!(coupled_model) =
    compute_atmosphere_land_fluxes!(coupled_model, coupled_model.interfaces.atmosphere_land_interface)

compute_atmosphere_land_fluxes!(coupled_model, ::Nothing) = nothing

function compute_atmosphere_land_fluxes!(coupled_model, atmosphere_land_interface)
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
    interface_properties = atmosphere_land_interface.properties
    atmosphere_properties = (thermodynamics_parameters = thermodynamics_parameters(coupled_model.atmosphere),
                             surface_layer_height = surface_layer_height(coupled_model.atmosphere),
                             gravitational_acceleration = coupled_model.interfaces.properties.gravitational_acceleration)

    # Land surface state from the exchanger (`Tₛ` in the interface solver,
    # `β = moisture_availability` in the land hydrology closure).
    # The generic SlabLand `ComponentExchanger` exposes these directly.
    land_state = exchanger.land.state
    Tₛ = land_state.T
    βₛ = land_state.moisture_availability

    land_properties = atmosphere_land_surface_properties(land_state)

    radiation = coupled_model.radiation
    radiation_kernel_props = kernel_radiation_properties(radiation)
    radiation_exchanger    = exchanger.radiation
    radiation_state        = isnothing(radiation_exchanger) ? nothing : radiation_exchanger.state

    # Land turbulent fluxes are evaluated only over interior cells; the
    # downstream SlabLand step uses `:xy` (interior-only), and halo
    # cells of the atmosphere exchanger state may not be initialized
    # when the atmosphere grid is a regional cutout matching the
    # exchange-grid interior exactly (`interface_kernel_parameters`
    # iterates 0:Nx+1 for the ocean's benefit; we do not need that
    # here).
    launch!(arch, grid, :xy,
            _compute_atmosphere_land_interface_state!,
            interface_fluxes,
            interface_temperature,
            grid,
            clock,
            flux_formulation,
            Tₛ,
            βₛ,
            atmosphere_data,
            interface_properties,
            atmosphere_properties,
            land_properties,
            radiation_kernel_props,
            radiation_state)

    return nothing
end

function atmosphere_land_surface_properties(land_state::NamedTuple{names}) where names
    momentum_roughness = _atmosphere_land_roughness_field(land_state,
                                                           :momentum_roughness_length)
    scalar_roughness   = _atmosphere_land_roughness_field(land_state,
                                                           :scalar_roughness_length)
    return _assemble_atmosphere_land_surface_properties(momentum_roughness, scalar_roughness)
end

@inline function _atmosphere_land_roughness_field(land_state::NamedTuple{names}, name::Symbol) where names
    if name in names
        return getproperty(land_state, name)
    elseif :roughness_length in names
        return land_state.roughness_length
    else
        return nothing
    end
end

@inline function _assemble_atmosphere_land_surface_properties(ℓm, ℓs)
    if isnothing(ℓm) && isnothing(ℓs)
        return (;)
    elseif isnothing(ℓs)
        return (; momentum_roughness_length = ℓm)
    elseif isnothing(ℓm)
        return (; scalar_roughness_length = ℓs)
    else
        return (; momentum_roughness_length = ℓm,
                 scalar_roughness_length   = ℓs)
    end
end

@inline local_atmosphere_land_surface_properties(land_properties, i, j) = (;)

@inline _roughness_value(::Nothing, i, j, k=1) = nothing
@inline _roughness_value(ℓ::Number, i, j, k=1) = ℓ
@inline _roughness_value(field, i, j, k=1) = @inbounds field[i, j, k]

@inline _moisture_availability(β::Number, i, j, k=1) = β
@inline _moisture_availability(β, i, j, k=1) = @inbounds β[i, j, k]

@inline function local_atmosphere_land_surface_properties(land_properties::NamedTuple{names}, i, j) where names
    ℓm = _roughness_value(_atmosphere_land_roughness_field(land_properties,
                                                           :momentum_roughness_length),
                           i, j, 1)
    ℓs = _roughness_value(_atmosphere_land_roughness_field(land_properties,
                                                           :scalar_roughness_length),
                           i, j, 1)

    return _assemble_atmosphere_land_surface_properties(ℓm, ℓs)
end

@kernel function _compute_atmosphere_land_interface_state!(interface_fluxes,
                                                           interface_temperature,
                                                           grid,
                                                           clock,
                                                           turbulent_flux_formulation,
                                                           Tₛ_field,
                                                           βₛ_field,
                                                           atmosphere_state,
                                                           interface_properties,
                                                           atmosphere_properties,
                                                           land_properties,
                                                           radiation_kernel_props,
                                                           radiation_exchanger_state)

    i, j = @index(Global, NTuple)
    time = Time(clock.time)

    @inbounds begin
        uᵃᵗ = atmosphere_state.u[i, j, 1]
        vᵃᵗ = atmosphere_state.v[i, j, 1]
        Tᵃᵗ = atmosphere_state.T[i, j, 1]
        pᵃᵗ = atmosphere_state.p[i, j, 1]
        qᵃᵗ = atmosphere_state.q[i, j, 1]

        Tₛ = Tₛ_field[i, j, 1]      # surface temperature [K]
        βₛ = convert(typeof(Tₛ), _moisture_availability(βₛ_field, i, j, 1)) # moisture availability ∈ [0, 1]
    end

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    zᵃᵗ = atmosphere_properties.surface_layer_height

    local_atmosphere_state = (z = zᵃᵗ,
                              u = uᵃᵗ,
                              v = vᵃᵗ,
                              T = Tᵃᵗ,
                              p = pᵃᵗ,
                              q = qᵃᵗ,
                              h_bℓ = atmosphere_state.h_bℓ)

    # Surface velocities are zero for land. β is threaded through the
    # `S` slot of `InterfaceState` so the existing fixed-point solver can
    # propagate it across iterations without API changes.
    FT = typeof(Tₛ)
    uₛ = zero(FT)
    vₛ = zero(FT)

    local_interior_state = (u = uₛ, v = vₛ, T = Tₛ, S = βₛ)
    local_land_properties = local_atmosphere_land_surface_properties(land_properties, i, j)

    radiation_state = air_land_interface_radiation_state(radiation_kernel_props,
                                                         radiation_exchanger_state,
                                                         i, j, 1, grid, time)

    # Estimate initial interface state.
    u★ = convert(FT, 1e-4)

    q_formulation = interface_properties.specific_humidity_formulation
    qₛ = surface_specific_humidity(q_formulation, ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ, Tₛ, βₛ)
    initial_interface_state = InterfaceState(u★, u★, u★, uₛ, vₛ, Tₛ, βₛ, qₛ)

    interface_state = compute_interface_state(turbulent_flux_formulation,
                                              initial_interface_state,
                                              local_atmosphere_state,
                                              local_interior_state,
                                              radiation_state,
                                              interface_properties,
                                              atmosphere_properties,
                                              local_land_properties)

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
    cᵖᵐ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ)
    ℒˡ = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Tᵃᵗ)

    𝒬ᵛ  = interface_fluxes.latent_heat
    𝒬ᵀ  = interface_fluxes.sensible_heat
    Jᵛ  = interface_fluxes.water_vapor
    ρτˣ = interface_fluxes.x_momentum
    ρτʸ = interface_fluxes.y_momentum
    Ts  = interface_temperature

    @inbounds begin
        𝒬ᵛ[i, j, 1]  = - ρᵃᵗ * ℒˡ * u★ * q★
        𝒬ᵀ[i, j, 1]  = - ρᵃᵗ * cᵖᵐ * u★ * θ★
        Jᵛ[i, j, 1]  = - ρᵃᵗ * u★ * q★
        ρτˣ[i, j, 1] = + ρᵃᵗ * τˣ
        ρτʸ[i, j, 1] = + ρᵃᵗ * τʸ
        Ts[i, j, 1]  = Ψₛ.T

        interface_fluxes.friction_velocity[i, j, 1] = u★
        interface_fluxes.temperature_scale[i, j, 1] = θ★
        interface_fluxes.water_vapor_scale[i, j, 1] = q★
    end
end
