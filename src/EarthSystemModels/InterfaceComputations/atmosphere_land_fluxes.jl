using Oceananigans.Grids: inactive_node

#####
##### Atmosphere-Land interface constructor
#####
##### The atmosphere–land turbulent fluxes share their container type
##### with atmosphere–ocean ([`AtmosphereSurfaceFluxes`](@ref)); only
##### the compute kernel differs.
#####

atmosphere_land_interface(grid, ::Nothing,    land;     kw...) = nothing
atmosphere_land_interface(grid, atmosphere, ::Nothing; kw...) = nothing
atmosphere_land_interface(grid, ::Nothing,  ::Nothing; kw...) = nothing

"""
    atmosphere_land_interface(grid, atmosphere, land;
                              fluxes               = default_atmosphere_land_fluxes(land, eltype(grid)),
                              temperature          = BulkTemperature(),
                              velocity_difference  = RelativeVelocity(),
                              specific_humidity    = default_al_specific_humidity(land))

Build the atmosphere--land interface on `grid` from `atmosphere` and `land` with
the given turbulent-flux closure, interface-temperature model, atmosphere-relative
velocity model, and specific-humidity formulation. Pass the result as
`atmosphere_land_interface = ...` to `ComponentInterfaces` /
`AtmosphereLandModel` to override the default.
"""
function atmosphere_land_interface(grid, atmosphere, land;
                                   fluxes              = default_atmosphere_land_fluxes(land, eltype(grid)),
                                   temperature         = BulkTemperature(),
                                   velocity_difference = RelativeVelocity(),
                                   specific_humidity   = default_al_specific_humidity(land))

    al_fluxes = AtmosphereSurfaceFluxes(grid)

    al_properties = InterfaceProperties(specific_humidity, temperature, velocity_difference)

    interface_temperature = Field{Center, Center, Nothing}(grid)

    return AtmosphereInterface(al_fluxes, fluxes, interface_temperature, al_properties)
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

    # Land surface state from the exchanger. `interface_energy_state` /
    # `interface_hydrology_state` read these per cell to build the land
    # interface state; the surface models derive `β`, the reservoir
    # temperature, etc. from them.
    land_exchanger_state = exchanger.land.state
    land_state = (T = land_exchanger_state.T,
                  saturation = land_exchanger_state.saturation)

    land_properties = atmosphere_land_surface_properties(land_exchanger_state)

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
            land_state,
            atmosphere_data,
            interface_properties,
            atmosphere_properties,
            land_properties,
            radiation_kernel_props,
            radiation_state)

    return nothing
end

function atmosphere_land_surface_properties(land_state::NamedTuple)
    ℓᵐ = _atmosphere_land_roughness_field(land_state, :momentum_roughness_length)
    ℓˢ = _atmosphere_land_roughness_field(land_state, :scalar_roughness_length)
    return _assemble_atmosphere_land_surface_properties(ℓᵐ, ℓˢ)
end

@inline function _atmosphere_land_roughness_field(land_state::NamedTuple, name::Symbol)
    if hasproperty(land_state, name)
        return getproperty(land_state, name)
    elseif hasproperty(land_state, :roughness_length)
        return land_state.roughness_length
    else
        return nothing
    end
end

@inline function _assemble_atmosphere_land_surface_properties(ℓᵐ, ℓˢ)
    if isnothing(ℓᵐ) && isnothing(ℓˢ)
        return (;)
    elseif isnothing(ℓˢ)
        return (; momentum_roughness_length = ℓᵐ)
    elseif isnothing(ℓᵐ)
        return (; scalar_roughness_length = ℓˢ)
    else
        return (; momentum_roughness_length = ℓᵐ,
                 scalar_roughness_length   = ℓˢ)
    end
end

@inline local_atmosphere_land_surface_properties(land_properties, i, j) = (;)

@inline _roughness_value(::Nothing, i, j, k=1) = nothing
@inline _roughness_value(ℓ::Number, i, j, k=1) = ℓ
@inline _roughness_value(field, i, j, k=1) = @inbounds field[i, j, k]

# Per-cell scalar from a constant or a `Field`.
@inline land_field_value(x::Number, i, j) = x
@inline land_field_value(x, i, j) = @inbounds x[i, j, 1]

#####
##### Land surface state materialized into the interface state.
#####
##### The surface model (`interface_model`, here the specific-humidity
##### formulation) dispatches these helpers to pull *exactly* the per-cell land
##### state it consumes — saturation for the moisture-availability models, the
##### bulk temperature for the reservoir model — and nothing otherwise. The
##### model then derives `β`, the reservoir temperature, etc. from what it pulled.
#####

@inline land_saturation(i, j, grid, land_state) =
    (saturation = convert(eltype(grid), land_field_value(land_state.saturation, i, j)),)

# Hydrology state, per humidity formulation.
@inline interface_hydrology_state(i, j, grid, ::BulkHumidity, land_state) = land_saturation(i, j, grid, land_state)
@inline interface_hydrology_state(i, j, grid, q::FractionalHumidity, land_state) =
    interface_hydrology_state(i, j, grid, q.efficiency, land_state)
@inline interface_hydrology_state(i, j, grid, ::CriticalSaturation, land_state) = land_saturation(i, j, grid, land_state)
@inline interface_hydrology_state(i, j, grid, interface_model, land_state) = (;) # default: pulls nothing

# Energy state: only the reservoir (skin-humidity) model needs the bulk temperature.
@inline interface_energy_state(i, j, grid, ::SkinHumidity, land_state) =
    (temperature = convert(eltype(grid), land_field_value(land_state.T, i, j)),)
@inline interface_energy_state(i, j, grid, interface_model, land_state) = (;) # default: pulls nothing

@inline function local_atmosphere_land_surface_properties(land_properties::NamedTuple, i, j)
    ℓᵐ = _roughness_value(_atmosphere_land_roughness_field(land_properties, :momentum_roughness_length),
                         i, j, 1)
    ℓˢ = _roughness_value(_atmosphere_land_roughness_field(land_properties, :scalar_roughness_length),
                         i, j, 1)

    return _assemble_atmosphere_land_surface_properties(ℓᵐ, ℓˢ)
end

@kernel function _compute_atmosphere_land_interface_state!(interface_fluxes,
                                                           interface_temperature,
                                                           grid,
                                                           clock,
                                                           turbulent_flux_formulation,
                                                           land_state,
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
    end

    q_formulation = interface_properties.specific_humidity_formulation

    # Bulk land temperature (the initial skin-temperature guess). The surface
    # models pull only the sub-state they consume from `land_state`.
    Tₛ = convert(eltype(grid), land_field_value(land_state.T, i, j))
    energy    = interface_energy_state(i, j, grid, q_formulation, land_state)
    hydrology = interface_hydrology_state(i, j, grid, q_formulation, land_state)

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    zᵃᵗ = atmosphere_properties.surface_layer_height

    local_atmosphere_state = (z = zᵃᵗ,
                              u = uᵃᵗ,
                              v = vᵃᵗ,
                              T = Tᵃᵗ,
                              p = pᵃᵗ,
                              q = qᵃᵗ,
                              h_bℓ = atmosphere_state.h_bℓ)

    # Surface velocities are zero for land.
    FT = typeof(Tₛ)
    uₛ = zero(FT)
    vₛ = zero(FT)

    local_interior_state = (u = uₛ, v = vₛ, T = Tₛ)
    local_land_properties = local_atmosphere_land_surface_properties(land_properties, i, j)

    radiation_state = air_land_interface_radiation_state(radiation_kernel_props,
                                                         radiation_exchanger_state,
                                                         i, j, 1, grid, time)

    # Estimate initial interface state. Use the saturated value as the initial
    # surface humidity guess (the solver recomputes it via the formulation).
    u★ = convert(FT, 1e-4)
    qₛ = convert(FT, saturation_specific_humidity(ℂᵃᵗ, Tₛ, pᵃᵗ, q_formulation.phase))
    initial_interface_state = AirLandInterfaceState(u★, u★, u★, uₛ, vₛ, Tₛ, qₛ, hydrology, energy)

    interface_state = compute_interface_state(turbulent_flux_formulation,
                                              initial_interface_state,
                                              local_atmosphere_state,
                                              local_interior_state,
                                              radiation_state,
                                              interface_properties,
                                              atmosphere_properties,
                                              local_land_properties)

    u★ = interface_state.fluxes.u★
    θ★ = interface_state.fluxes.θ★
    q★ = interface_state.fluxes.q★

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
        Ts[i, j, 1]  = Ψₛ.temperature

        interface_fluxes.friction_velocity[i, j, 1] = u★
        interface_fluxes.temperature_scale[i, j, 1] = θ★
        interface_fluxes.water_vapor_scale[i, j, 1] = q★
    end
end
