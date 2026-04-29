using NumericalEarth.EarthSystemModels.InterfaceComputations: computed_fluxes,
                                                          interface_kernel_parameters,
                                                          convert_to_kelvin

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

    atmosphere_fields = coupled_model.interfaces.exchanger.atmosphere.state
    freshwater_flux = atmosphere_fields.Jᶜ.data

    sea_ice_properties = coupled_model.interfaces.sea_ice_properties
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
            sea_ice_properties)

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
                                               sea_ice_properties)

    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)

    @inbounds begin
        ℵi = ice_concentration[i, j, 1]
        𝒬ᵀ   = atmosphere_sea_ice_fluxes.sensible_heat[i, j, 1] # sensible heat flux
        𝒬ᵛ   = atmosphere_sea_ice_fluxes.latent_heat[i, j, 1]   # latent heat flux
        𝒬ᶠʳᶻ = sea_ice_ocean_fluxes.frazil_heat[i, j, 1]          # frazil heat flux
        𝒬ⁱⁿᵗ = sea_ice_ocean_fluxes.interface_heat[i, j, 1]       # interfacial heat flux
    end

    ρτˣ = atmosphere_sea_ice_fluxes.x_momentum # zonal momentum flux
    ρτʸ = atmosphere_sea_ice_fluxes.y_momentum # meridional momentum flux

    # Turbulent contributions only (radiation added later by apply_air_sea_ice_radiative_fluxes!)
    ΣQt = (𝒬ᵀ + 𝒬ᵛ) * (ℵi > 0)
    ΣQb = 𝒬ᶠʳᶻ + 𝒬ⁱⁿᵗ

    # Mask fluxes over land for convenience
    inactive = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    @inbounds top_fluxes.heat[i, j, 1]  = ifelse(inactive, zero(grid), ΣQt)
    @inbounds top_fluxes.u[i, j, 1]     = ifelse(inactive, zero(grid), ℑxᶠᵃᵃ(i, j, 1, grid, ρτˣ))
    @inbounds top_fluxes.v[i, j, 1]     = ifelse(inactive, zero(grid), ℑyᵃᶠᵃ(i, j, 1, grid, ρτʸ))
    @inbounds bottom_heat_flux[i, j, 1] = ifelse(inactive, zero(grid), ΣQb)
end
