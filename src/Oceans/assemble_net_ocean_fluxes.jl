using Oceananigans.Fields: ZeroField

using ..EarthSystemModels: NoAtmosInterfaceModel, NoOceanInterfaceModel, NoInterfaceModel, sea_ice_concentration
using ..EarthSystemModels.InterfaceComputations: computed_fluxes

@inline τᶜᶜᶜ(i, j, k, grid, ρᵒᶜ⁻¹, ℵ, ρτᶜᶜᶜ) = @inbounds ρᵒᶜ⁻¹ * (1 - ℵ[i, j, k]) * ρτᶜᶜᶜ[i, j, k]

#####
##### Generic flux assembler — turbulent + sea-ice contributions only.
##### Radiative contributions are added later by `apply_air_sea_radiative_fluxes!`.
#####

# Fallback for an ocean-only model (it has no interfaces!)
EarthSystemModels.update_net_fluxes!(coupled_model::Union{NoOceanInterfaceModel, NoInterfaceModel}, ocean::OceananigansModelSimulations) = nothing

EarthSystemModels.update_net_fluxes!(coupled_model, ocean::OceananigansModelSimulations) =
    update_net_ocean_fluxes!(coupled_model, ocean, ocean.model.grid)

rainfall_flux(::NoAtmosInterfaceModel) = ZeroField()
rainfall_flux(coupled_model) = coupled_model.interfaces.exchanger.atmosphere.state.Jʳⁿ.data

snowfall_flux(::NoAtmosInterfaceModel) = ZeroField()
snowfall_flux(coupled_model) = coupled_model.interfaces.exchanger.atmosphere.state.Jˢⁿ.data

atmos_ocean_flux(coupled_model) = computed_fluxes(coupled_model.interfaces.atmosphere_ocean_interface)

land_freshwater_flux(::Nothing) = ZeroField()
land_freshwater_flux(land_exchanger) = land_exchanger.state.freshwater_flux.data

function update_net_ocean_fluxes!(coupled_model, ocean_model, grid)
    sea_ice = coupled_model.sea_ice
    arch = architecture(grid)
    clock = coupled_model.clock

    net_ocean_fluxes = coupled_model.interfaces.net_fluxes.ocean
    sea_ice_ocean_fluxes = computed_fluxes(coupled_model.interfaces.sea_ice_ocean_interface)

    atmos_ocean_fluxes = atmos_ocean_flux(coupled_model)
    rainfall = rainfall_flux(coupled_model)
    snowfall = snowfall_flux(coupled_model)

    land_exchanger = coupled_model.interfaces.exchanger.land
    freshwater_flux = land_freshwater_flux(land_exchanger)

    ice_concentration = sea_ice_concentration(sea_ice)
    ocean_surface_salinity = EarthSystemModels.ocean_surface_salinity(ocean_model)
    ocean_properties = coupled_model.interfaces.ocean_properties

    launch!(arch, grid, :xy,
            _assemble_net_ocean_fluxes!,
            net_ocean_fluxes,
            grid,
            clock,
            atmos_ocean_fluxes,
            sea_ice_ocean_fluxes,
            ocean_surface_salinity,
            ice_concentration,
            rainfall,
            snowfall,
            freshwater_flux,
            ocean_properties)
    return nothing
end

Base.@propagate_inbounds get_land_freshwater_flux(i, j, flux) = flux[i, j, 1]

@kernel function _assemble_net_ocean_fluxes!(net_ocean_fluxes,
                                             grid,
                                             clock,
                                             atmos_ocean_fluxes,
                                             sea_ice_ocean_fluxes,
                                             ocean_surface_salinity,
                                             sea_ice_concentration,
                                             rainfall_flux,
                                             snowfall_flux,
                                             land_freshwater_flux,
                                             ocean_properties)

    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    ρτˣᵃᵒ = atmos_ocean_fluxes.x_momentum   # atmosphere - ocean zonal momentum flux
    ρτʸᵃᵒ = atmos_ocean_fluxes.y_momentum   # atmosphere - ocean meridional momentum flux
    ρτˣⁱᵒ = sea_ice_ocean_fluxes.x_momentum             # sea_ice - ocean zonal momentum flux (explicit part)
    ρτʸⁱᵒ = sea_ice_ocean_fluxes.y_momentum             # sea_ice - ocean meridional momentum flux (explicit part)
    ρλˣⁱᵒ = sea_ice_ocean_fluxes.x_momentum_coefficient # ice-ocean implicit drag coefficient (zonal)
    ρλʸⁱᵒ = sea_ice_ocean_fluxes.y_momentum_coefficient # ice-ocean implicit drag coefficient (meridional)

    @inbounds begin
        ℵᵢ = sea_ice_concentration[i, j, 1]
        Sᵒᶜ = ocean_surface_salinity[i, j, 1]

        Jʳⁿ = rainfall_flux[i, j, 1]
        Jˢⁿ = snowfall_flux[i, j, 1]
        Jˡⁿ = get_land_freshwater_flux(i, j, land_freshwater_flux)
        𝒬ᵀ = atmos_ocean_fluxes.sensible_heat[i, j, 1]
        𝒬ᵛ = atmos_ocean_fluxes.latent_heat[i, j, 1]
        Jᵛ = atmos_ocean_fluxes.water_vapor[i, j, 1]
    end

    # Turbulent contributions to surface heat flux (radiation added later)
    ΣQao = (𝒬ᵀ + 𝒬ᵛ) * (1 - ℵᵢ)

    # Freshwater flux to the ocean per unit cell area (volume flux, positive up = leaving ocean):
    # - rain and land runoff reach the ocean everywhere (rain runs through cracks in ice)
    # - snow only reaches the ocean through the open-water fraction (1 - ℵᵢ);
    # - evaporation acts only over the open-water fraction (1 - ℵᵢ)
    # The atmospheric mass-flux convention is positive down; Jᵛ is positive up.
    ρᵒᶜ⁻¹ = 1 / ocean_properties.reference_density
    ΣFao = - (Jʳⁿ + Jˡⁿ + (1 - ℵᵢ) * Jˢⁿ) * ρᵒᶜ⁻¹ + (1 - ℵᵢ) * Jᵛ * ρᵒᶜ⁻¹


    τˣ = net_ocean_fluxes.u
    τʸ = net_ocean_fluxes.v
    λˣ = net_ocean_fluxes.u_coefficient
    λʸ = net_ocean_fluxes.v_coefficient
    Jᵀ = net_ocean_fluxes.T
    Jˢ = net_ocean_fluxes.S
    ℵ  = sea_ice_concentration
    cᵒᶜ⁻¹ = 1 / ocean_properties.heat_capacity
    inactive = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    @inbounds begin
        𝒬ⁱⁿᵗ = sea_ice_ocean_fluxes.interface_heat[i, j, 1]
        Jˢio = sea_ice_ocean_fluxes.salt[i, j, 1]
        Jᵀao = ΣQao * ρᵒᶜ⁻¹ * cᵒᶜ⁻¹
        Jᵀio = 𝒬ⁱⁿᵗ * ρᵒᶜ⁻¹ * cᵒᶜ⁻¹

        # salinity flux > 0 extracts salinity (opposite of water vapor flux sign)
        Jˢao = - Sᵒᶜ * ΣFao

        τˣᵃᵒ = ℑxᶠᵃᵃ(i, j, 1, grid, τᶜᶜᶜ, ρᵒᶜ⁻¹, ℵ, ρτˣᵃᵒ)
        τʸᵃᵒ = ℑyᵃᶠᵃ(i, j, 1, grid, τᶜᶜᶜ, ρᵒᶜ⁻¹, ℵ, ρτʸᵃᵒ)
        # Ice-ocean drag, split for the ocean's semi-implicit momentum BC: the explicit part joins
        # the atmosphere-ocean stress, the ℵ-weighted coefficient feeds the BC's implicit drag term.
        τˣⁱᵒ = ρτˣⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑxᶠᵃᵃ(i, j, 1, grid, ℵ)
        τʸⁱᵒ = ρτʸⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑyᵃᶠᵃ(i, j, 1, grid, ℵ)
        λˣⁱᵒ = ρλˣⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑxᶠᵃᵃ(i, j, 1, grid, ℵ)
        λʸⁱᵒ = ρλʸⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑyᵃᶠᵃ(i, j, 1, grid, ℵ)

        τˣ[i, j, 1] = ifelse(inactive, zero(grid), τˣᵃᵒ + τˣⁱᵒ)
        τʸ[i, j, 1] = ifelse(inactive, zero(grid), τʸᵃᵒ + τʸⁱᵒ)
        λˣ[i, j, 1] = ifelse(inactive, zero(grid), λˣⁱᵒ)
        λʸ[i, j, 1] = ifelse(inactive, zero(grid), λʸⁱᵒ)

        # Tracer fluxes — radiative contributions added later by apply_air_sea_radiative_fluxes!
        Jᵀ[i, j, 1] = ifelse(inactive, zero(grid), Jᵀao + Jᵀio)

        # Suppress freshening when surface salinity is at/below `minimum_salinity`.
        Jˢo  = Jˢao + Jˢio
        Smin = ocean_properties.minimum_salinity
        clip = (Sᵒᶜ < Smin) & (Jˢo > zero(Jˢo))
        Jˢo  = ifelse(clip, zero(Jˢo), Jˢo)

        Jˢ[i, j, 1] = ifelse(inactive, zero(grid), Jˢo)
    end
end
