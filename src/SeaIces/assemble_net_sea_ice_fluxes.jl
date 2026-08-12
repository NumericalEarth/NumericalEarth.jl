using Oceananigans.Fields: ZeroField

using ..EarthSystemModels: sea_ice_concentration, NoAtmosInterfaceModel
using ..EarthSystemModels.InterfaceComputations: computed_fluxes

EarthSystemModels.update_net_fluxes!(coupled_model, ::FreezingLimitedOceanTemperature) = nothing

snowfall_flux(coupled_model::NoAtmosInterfaceModel) = ZeroField(eltype(coupled_model))
snowfall_flux(coupled_model) = coupled_model.interfaces.exchanger.atmosphere.state.Jˢⁿ.data

function EarthSystemModels.update_net_fluxes!(coupled_model, sea_ice::Simulation{<:SeaIceModel})
    ocean = coupled_model.ocean
    grid  = sea_ice.model.grid
    arch  = architecture(grid)
    clock = coupled_model.clock

    top_fluxes = coupled_model.interfaces.net_fluxes.sea_ice.top
    bottom_heat_flux = coupled_model.interfaces.net_fluxes.sea_ice.bottom.heat
    sea_ice_ocean_fluxes = computed_fluxes(coupled_model.interfaces.sea_ice_ocean_interface)
    atmosphere_sea_ice_fluxes = computed_fluxes(coupled_model.interfaces.atmosphere_sea_ice_interface)

    snowfall = snowfall_flux(coupled_model)

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
            snowfall,
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
                                               snowfall_flux,
                                               ice_concentration,
                                               sea_ice_properties)

    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)

    @inbounds begin
        ℵi = ice_concentration[i, j, 1]
        𝒬ᵀ   = atmosphere_sea_ice_fluxes.sensible_heat[i, j, 1]   # sensible heat flux
        𝒬ᵛ   = atmosphere_sea_ice_fluxes.latent_heat[i, j, 1]     # latent heat flux
        𝒬ᶠʳᶻ = sea_ice_ocean_fluxes.frazil_heat[i, j, 1]          # frazil heat flux
        𝒬ⁱⁿ  = sea_ice_ocean_fluxes.interface_heat[i, j, 1]       # interfacial heat flux
        Jˢⁿ  = snowfall_flux[i, j, 1]
    end

    ρτˣ = atmosphere_sea_ice_fluxes.x_momentum # zonal momentum flux
    ρτʸ = atmosphere_sea_ice_fluxes.y_momentum # meridional momentum flux

    ΣQt = (𝒬ᵀ + 𝒬ᵛ) * ℵi
    
    # Frazil ice does not depend on the ice concentration (it is already per-cell)
    # While interface heat is pre-multiplied by concentration
    ΣQb = 𝒬ᶠʳᶻ + 𝒬ⁱⁿ

    # Mask fluxes over land for convenience
    inactive = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    @inbounds top_fluxes.heat[i, j, 1]     = ifelse(inactive, zero(grid), ΣQt)
    @inbounds top_fluxes.snowfall[i, j, 1] = ifelse(inactive, zero(grid), Jˢⁿ)
    @inbounds top_fluxes.u[i, j, 1]        = ifelse(inactive, zero(grid), ℑxᶠᵃᵃ(i, j, 1, grid, ρτˣ))
    @inbounds top_fluxes.v[i, j, 1]        = ifelse(inactive, zero(grid), ℑyᵃᶠᵃ(i, j, 1, grid, ρτʸ))
    @inbounds bottom_heat_flux[i, j, 1]    = ifelse(inactive, zero(grid), ΣQb)
end
