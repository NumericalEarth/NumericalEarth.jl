using NumericalEarth.EarthSystemModels.InterfaceComputations: computed_fluxes,
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

    downwelling_radiation = (ℐꜜˢʷ = atmosphere_fields.ℐꜜˢʷ.data,
                             ℐꜜˡʷ = atmosphere_fields.ℐꜜˡʷ.data)

    freshwater_flux = atmosphere_fields.Jᶜ.data

    atmos_sea_ice_properties = coupled_model.interfaces.atmosphere_sea_ice_interface.properties
    sea_ice_properties = coupled_model.interfaces.sea_ice_properties

    sea_ice_surface_temperature = coupled_model.interfaces.atmosphere_sea_ice_interface.temperature
    ice_concentration = sea_ice_concentration(sea_ice)
    
    launch!(arch, grid, :xy,
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

        ℐꜜˢʷ = downwelling_radiation.ℐꜜˢʷ[i, j, 1]
        ℐꜜˡʷ = downwelling_radiation.ℐꜜˡʷ[i, j, 1]
        𝒬ᵀ   = atmosphere_sea_ice_fluxes.sensible_heat[i, j, 1]   # sensible heat flux
        𝒬ᵛ   = atmosphere_sea_ice_fluxes.latent_heat[i, j, 1]     # latent heat flux
        𝒬ᶠʳᶻ = sea_ice_ocean_fluxes.frazil_heat[i, j, 1]          # frazil heat flux
        𝒬ⁱⁿᵗ = sea_ice_ocean_fluxes.interface_heat[i, j, 1]       # interfacial heat flux
    end

    ρτˣ = atmosphere_sea_ice_fluxes.x_momentum # zonal momentum flux
    ρτʸ = atmosphere_sea_ice_fluxes.y_momentum # meridional momentum flux

    # Compute radiation fluxes
    σ = atmos_sea_ice_properties.radiation.σ
    α = atmos_sea_ice_properties.radiation.α
    ϵ = atmos_sea_ice_properties.radiation.ϵ
    ℐꜛˡʷ = emitted_longwave_radiation(i, j, kᴺ, grid, time, Ts, σ, ϵ)
    ℐₜˢʷ = transmitted_shortwave_radiation(i, j, kᴺ, grid, time, α, ℐꜜˢʷ)
    ℐₐˡʷ = absorbed_longwave_radiation(i, j, kᴺ, grid, time, ϵ, ℐꜜˡʷ)

    ΣQt = (ℐₜˢʷ + ℐₐˡʷ + ℐꜛˡʷ + 𝒬ᵀ + 𝒬ᵛ) * (ℵi > 0) # If ℵi == 0 there is no heat flux from the top!
    ΣQb = 𝒬ᶠʳᶻ + 𝒬ⁱⁿᵗ

    # Mask fluxes over land for convenience
    inactive = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    @inbounds top_fluxes.heat[i, j, 1]  = ifelse(inactive, zero(grid), ΣQt)
    @inbounds top_fluxes.u[i, j, 1]     = ifelse(inactive, zero(grid), ℑxᶠᵃᵃ(i, j, 1, grid, ρτˣ))
    @inbounds top_fluxes.v[i, j, 1]     = ifelse(inactive, zero(grid), ℑyᵃᶠᵃ(i, j, 1, grid, ρτʸ))
    @inbounds bottom_heat_flux[i, j, 1] = ifelse(inactive, zero(grid), ΣQb)
end
