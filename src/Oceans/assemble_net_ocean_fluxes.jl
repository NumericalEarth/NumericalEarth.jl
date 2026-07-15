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

rainfall_flux(coupled_model::NoAtmosInterfaceModel) = ZeroField(eltype(coupled_model))
rainfall_flux(coupled_model) = coupled_model.interfaces.exchanger.atmosphere.state.Jʳⁿ.data

snowfall_flux(coupled_model::NoAtmosInterfaceModel) = ZeroField(eltype(coupled_model))
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
    ocean_surface_temperature = EarthSystemModels.ocean_surface_temperature(ocean_model)
    ocean_properties = coupled_model.interfaces.ocean_properties

    launch!(arch, grid, :xy,
            _assemble_net_ocean_fluxes!,
            net_ocean_fluxes,
            grid,
            clock,
            atmos_ocean_fluxes,
            sea_ice_ocean_fluxes,
            ocean_surface_temperature,
            ice_concentration,
            rainfall,
            snowfall,
            freshwater_flux,
            ocean_properties)

    if grid isa MutableGridOfSomeKind
        fill_halo_regions!(net_ocean_fluxes.η)
    end

    return nothing
end

Base.@propagate_inbounds get_land_freshwater_flux(i, j, flux) = flux[i, j, 1]

@kernel function _assemble_net_ocean_fluxes!(net_ocean_fluxes,
                                             grid,
                                             clock,
                                             atmos_ocean_fluxes,
                                             sea_ice_ocean_fluxes,
                                             ocean_surface_temperature,
                                             sea_ice_concentration,
                                             rainfall_flux,
                                             snowfall_flux,
                                             land_freshwater_flux,
                                             ocean_properties)

    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    ρτˣᵃᵒ = atmos_ocean_fluxes.x_momentum   # atmosphere - ocean zonal momentum flux
    ρτʸᵃᵒ = atmos_ocean_fluxes.y_momentum   # atmosphere - ocean meridional momentum flux
    ρτˣⁱᵒ = sea_ice_ocean_fluxes.x_momentum # sea_ice - ocean zonal momentum flux
    ρτʸⁱᵒ = sea_ice_ocean_fluxes.y_momentum # sea_ice - ocean meridional momentum flux

    @inbounds begin
        ℵᵢ = sea_ice_concentration[i, j, 1]
        Tᵒᶜ = ocean_surface_temperature[i, j, 1]

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
    ΣFao  = - (Jʳⁿ + Jˡⁿ + (1 - ℵᵢ) * Jˢⁿ) * ρᵒᶜ⁻¹ + (1 - ℵᵢ) * Jᵛ * ρᵒᶜ⁻¹
    Jʷao  = - ΣFao # Freshwater flux (positive increases the volume)

    τˣ = net_ocean_fluxes.u
    τʸ = net_ocean_fluxes.v
    Jᵀ = net_ocean_fluxes.T
    Jˢ = net_ocean_fluxes.S
    Jʷ = net_ocean_fluxes.η
    Jᴴ = net_ocean_fluxes.freshwater_heat_content # Σᵢ Tᵢ Jʷᵢ — atmosphere freshwater enters at SST
    ℵ  = sea_ice_concentration
    cᵒᶜ⁻¹ = 1 / ocean_properties.heat_capacity
    inactive = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    @inbounds begin
        𝒬ⁱⁿ = sea_ice_ocean_fluxes.interface_heat[i, j, 1]
        Jˢio = sea_ice_ocean_fluxes.salt[i, j, 1]
        Jʷio = sea_ice_ocean_fluxes.freshwater[i, j, 1]
        Jᵀao = ΣQao * ρᵒᶜ⁻¹ * cᵒᶜ⁻¹
        Jᵀio =  𝒬ⁱⁿ * ρᵒᶜ⁻¹ * cᵒᶜ⁻¹

        τˣᵃᵒ = ℑxᶠᵃᵃ(i, j, 1, grid, τᶜᶜᶜ, ρᵒᶜ⁻¹, ℵ, ρτˣᵃᵒ)
        τʸᵃᵒ = ℑyᵃᶠᵃ(i, j, 1, grid, τᶜᶜᶜ, ρᵒᶜ⁻¹, ℵ, ρτʸᵃᵒ)
        τˣⁱᵒ = ρτˣⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑxᶠᵃᵃ(i, j, 1, grid, ℵ)
        τʸⁱᵒ = ρτʸⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑyᵃᶠᵃ(i, j, 1, grid, ℵ)

        τˣ[i, j, 1] = ifelse(inactive, zero(grid), τˣᵃᵒ + τˣⁱᵒ)
        τʸ[i, j, 1] = ifelse(inactive, zero(grid), τʸᵃᵒ + τʸⁱᵒ)

        # Tracer fluxes — radiative contributions added later by apply_air_sea_radiative_fluxes!.
        # The atmosphere-ocean virtual salt flux (Sᴺ Jʷ) and the surface-value heat exchange
        # (Tᴺ Jʷ) are applied live in the salinity/temperature top BCs, so Jˢ holds only the
        # sea-ice contribution and Jᴴ the freshwater enthalpy (rain − evaporation at SST).
        Jᵀ[i, j, 1]  = ifelse(inactive, zero(grid), Jᵀao + Jᵀio)
        Jˢ[i, j, 1]  = ifelse(inactive, zero(grid), Jˢio)
        Jʷ[i, j, 1]  = ifelse(inactive, zero(grid), Jʷao + Jʷio)
        Jᴴ[i, j, 1] = ifelse(inactive, zero(grid), Tᵒᶜ * Jʷao)
    end
end
