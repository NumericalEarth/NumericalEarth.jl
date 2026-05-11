using Oceananigans.Grids: inactive_node

#####
##### Atmosphere-Land flux container
#####

struct AtmosphereLandFluxes{F}
    latent_heat       :: F
    sensible_heat     :: F
    water_vapor       :: F
    x_momentum        :: F
    y_momentum        :: F
    friction_velocity :: F
    temperature_scale :: F
    water_vapor_scale :: F
end

function AtmosphereLandFluxes(grid)
    F = Field{Center, Center, Nothing}
    return AtmosphereLandFluxes(F(grid), F(grid), F(grid),
                                F(grid), F(grid), F(grid),
                                F(grid), F(grid))
end

AtmosphereLandFluxes(::Nothing) = AtmosphereLandFluxes(ntuple(_ -> ZeroField(), 8)...)

Adapt.adapt_structure(to, fluxes::AtmosphereLandFluxes) =
    AtmosphereLandFluxes(Adapt.adapt(to, fluxes.latent_heat),
                         Adapt.adapt(to, fluxes.sensible_heat),
                         Adapt.adapt(to, fluxes.water_vapor),
                         Adapt.adapt(to, fluxes.x_momentum),
                         Adapt.adapt(to, fluxes.y_momentum),
                         Adapt.adapt(to, fluxes.friction_velocity),
                         Adapt.adapt(to, fluxes.temperature_scale),
                         Adapt.adapt(to, fluxes.water_vapor_scale))

on_architecture(arch, fluxes::AtmosphereLandFluxes) =
    AtmosphereLandFluxes(on_architecture(arch, fluxes.latent_heat),
                         on_architecture(arch, fluxes.sensible_heat),
                         on_architecture(arch, fluxes.water_vapor),
                         on_architecture(arch, fluxes.x_momentum),
                         on_architecture(arch, fluxes.y_momentum),
                         on_architecture(arch, fluxes.friction_velocity),
                         on_architecture(arch, fluxes.temperature_scale),
                         on_architecture(arch, fluxes.water_vapor_scale))

#####
##### Atmosphere-Land interface constructor
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

    al_fluxes = AtmosphereLandFluxes(grid)

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

    # Land surface state from the exchanger (T_g [K], β = moisture_availability).
    # The RucSlabLand `ComponentExchanger` aliases these to the slab fields,
    # so no copy is needed.
    land_state = exchanger.land.state
    Tₛ = land_state.T
    βₛ = land_state.moisture_availability

    land_properties = atmosphere_land_surface_properties(land_state)

    kernel_parameters = interface_kernel_parameters(grid)

    launch!(arch, grid, kernel_parameters,
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
            land_properties)

    return nothing
end

function atmosphere_land_surface_properties(land_state::NamedTuple{names}) where names
    if :roughness_length in names
        return (; roughness_length = land_state.roughness_length)
    else
        return (;)
    end
end

@inline local_atmosphere_land_surface_properties(land_properties, i, j) = (;)

@inline function local_atmosphere_land_surface_properties(land_properties::NamedTuple{(:roughness_length,), T}, i, j) where T
    roughness_length = @inbounds land_properties.roughness_length[i, j, 1]
    return (; roughness_length)
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
                                                           land_properties)

    i, j = @index(Global, NTuple)

    @inbounds begin
        uᵃᵗ = atmosphere_state.u[i, j, 1]
        vᵃᵗ = atmosphere_state.v[i, j, 1]
        Tᵃᵗ = atmosphere_state.T[i, j, 1]
        pᵃᵗ = atmosphere_state.p[i, j, 1]
        qᵃᵗ = atmosphere_state.q[i, j, 1]

        Tₛ = Tₛ_field[i, j, 1]      # surface temperature [K]
        βₛ = βₛ_field[i, j, 1]      # moisture availability ∈ [0, 1]
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

    radiation_state = air_sea_interface_radiation_state(nothing, nothing, i, j, 1, grid, Time(clock.time))

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
