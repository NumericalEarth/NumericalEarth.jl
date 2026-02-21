using NumericalEarth.EarthSystemModels.InterfaceComputations: computed_fluxes, 
                                                          get_possibly_zero_flux,
                                                          interface_kernel_parameters,
                                                          convert_to_kelvin,
                                                          emitted_longwave_radiation,
                                                          absorbed_longwave_radiation,
                                                          transmitted_shortwave_radiation

update_net_fluxes!(coupled_model, ::FreezingLimitedOceanTemperature) = nothing

function update_net_fluxes!(coupled_model, sea_ice::Simulation{<:SeaIceModel})
    ocean = coupled_model.ocean
    grid  = sea_ice.model.grid
    arch  = architecture(grid)
    clock = coupled_model.clock

    top_fluxes = coupled_model.interfaces.net_fluxes.sea_ice.top
    bottom_heat_flux = coupled_model.interfaces.net_fluxes.sea_ice.bottom.heat
    sea_ice_ocean_fluxes = computed_fluxes(coupled_model.interfaces.sea_ice_ocean_interface)
    atmosphere_sea_ice_fluxes = computed_fluxes(coupled_model.interfaces.atmosphere_sea_ice_interface)

    # Simplify NamedTuple to reduce parameter space consumption.
    # See https://github.com/CliMA/NumericalEarth.jl/issues/116.
    atmosphere_fields = coupled_model.interfaces.exchanger.atmosphere.state

    downwelling_radiation = (Qs = atmosphere_fields.Qs.data,
                             Qℓ = atmosphere_fields.Qℓ.data)

    freshwater_flux = atmosphere_fields.Jᶜ.data

    atmos_sea_ice_properties = coupled_model.interfaces.atmosphere_sea_ice_interface.properties
    sea_ice_properties = coupled_model.interfaces.sea_ice_properties

    kernel_parameters = interface_kernel_parameters(grid)

    sea_ice_surface_temperature = coupled_model.interfaces.atmosphere_sea_ice_interface.temperature
    ice_concentration = sea_ice_concentration(sea_ice)
    
    launch!(arch, grid, kernel_parameters,
            _assemble_net_sea_ice_fluxes!,
            top_fluxes,
            bottom_heat_flux,
            grid,
            clock,
            atmosphere_sea_ice_fluxes,
            sea_ice_ocean_fluxes,
            freshwater_flux,
            ice_concentration,
            sea_ice_surface_temperature,
            downwelling_radiation,
            sea_ice_properties,
            atmos_sea_ice_properties)

    return nothing
end

@kernel function _assemble_net_sea_ice_fluxes!(top_fluxes,
                                               bottom_heat_flux,
                                               grid,
                                               clock,
                                               atmosphere_sea_ice_fluxes,
                                               sea_ice_ocean_fluxes,
                                               freshwater_flux, # Where do we add this one?
                                               ice_concentration,
                                               surface_temperature,
                                               downwelling_radiation,
                                               sea_ice_properties,
                                               atmos_sea_ice_properties)

    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    time = Time(clock.time)

    @inbounds begin
        Ts = surface_temperature[i, j, kᴺ]
        Ts = convert_to_kelvin(sea_ice_properties.temperature_units, Ts)
        ℵi = ice_concentration[i, j, 1]
        
        Qs = downwelling_radiation.Qs[i, j, 1]
        Qℓ = downwelling_radiation.Qℓ[i, j, 1]
        𝒬ᵀ = get_possibly_zero_flux(atmosphere_sea_ice_fluxes, :sensible_heat)[i, j, 1] # sensible or "conductive" heat flux
        𝒬ᵛ = get_possibly_zero_flux(atmosphere_sea_ice_fluxes, :latent_heat)[i, j, 1]   # latent heat flux
        Qf = get_possibly_zero_flux(sea_ice_ocean_fluxes, :frazil_heat)[i, j, 1]        # frazil heat flux
        Qi = get_possibly_zero_flux(sea_ice_ocean_fluxes, :interface_heat)[i, j, 1]   # interfacial heat flux
    end

    ρτx = get_possibly_zero_flux(atmosphere_sea_ice_fluxes, :x_momentum) # zonal momentum flux
    ρτy = get_possibly_zero_flux(atmosphere_sea_ice_fluxes, :y_momentum) # meridional momentum flux

    # Compute radiation fluxes
    σ = atmos_sea_ice_properties.radiation.σ
    α = atmos_sea_ice_properties.radiation.α
    ϵ = atmos_sea_ice_properties.radiation.ϵ
    Qu = emitted_longwave_radiation(i, j, kᴺ, grid, time, Ts, σ, ϵ) 
    Qs = transmitted_shortwave_radiation(i, j, kᴺ, grid, time, α, Qs)
    Qℓ = absorbed_longwave_radiation(i, j, kᴺ, grid, time, ϵ, Qℓ)

    ΣQt = (Qs + Qℓ + Qu + 𝒬ᵀ + 𝒬ᵛ) * (ℵi > 0) # If ℵi == 0 there is no heat flux from the top!
    ΣQb = Qf + Qi

    # Mask fluxes over land for convenience
    inactive = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    @inbounds top_fluxes.heat[i, j, 1]  = ifelse(inactive, zero(grid), ΣQt)
    @inbounds top_fluxes.u[i, j, 1]     = ifelse(inactive, zero(grid), ℑxᶠᵃᵃ(i, j, 1, grid, ρτx))
    @inbounds top_fluxes.v[i, j, 1]     = ifelse(inactive, zero(grid), ℑyᵃᶠᵃ(i, j, 1, grid, ρτy))
    @inbounds bottom_heat_flux[i, j, 1] = ifelse(inactive, zero(grid), ΣQb)
end
