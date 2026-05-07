using Printf
using Oceananigans.Grids: inactive_node
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Forcings: MultipleForcings
using NumericalEarth.EarthSystemModels: EarthSystemModel, NoOceanInterfaceModel, NoInterfaceModel

using NumericalEarth.EarthSystemModels.InterfaceComputations: interface_kernel_parameters,
                                                              computed_fluxes,
                                                              sea_ice_concentration

@inline τᶜᶜᶜ(i, j, k, grid, ρᵒᶜ⁻¹, ℵ, ρτᶜᶜᶜ) = @inbounds ρᵒᶜ⁻¹ * (1 - ℵ[i, j, k]) * ρτᶜᶜᶜ[i, j, k]

#####
##### Generic flux assembler — turbulent + sea-ice contributions only.
##### Radiative contributions are added later by `apply_air_sea_radiative_fluxes!`.
#####

# Fallback for an ocean-only model (it has no interfaces!)
update_net_fluxes!(coupled_model::Union{NoOceanInterfaceModel, NoInterfaceModel}, ocean::Simulation{<:HydrostaticFreeSurfaceModel}) = nothing

update_net_fluxes!(coupled_model, ocean::Simulation{<:HydrostaticFreeSurfaceModel}) =
    update_net_ocean_fluxes!(coupled_model, ocean, ocean.model.grid)

function update_net_ocean_fluxes!(coupled_model, ocean_model, grid)
    sea_ice = coupled_model.sea_ice
    arch = architecture(grid)
    clock = coupled_model.clock

    net_ocean_fluxes = coupled_model.interfaces.net_fluxes.ocean
    atmos_ocean_fluxes = computed_fluxes(coupled_model.interfaces.atmosphere_ocean_interface)
    sea_ice_ocean_fluxes = computed_fluxes(coupled_model.interfaces.sea_ice_ocean_interface)

    atmosphere_fields = coupled_model.interfaces.exchanger.atmosphere.state
    freshwater_flux = atmosphere_fields.Jᶜ.data

    # Extract land freshwater flux if land component is present
    land_exchanger = coupled_model.interfaces.exchanger.land
    land_freshwater_flux = isnothing(land_exchanger) ? nothing : land_exchanger.state.freshwater_flux.data

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
            freshwater_flux,
            land_freshwater_flux,
            ocean_properties)

    return nothing
end

@inline get_land_freshwater_flux(i, j, ::Nothing) = 0
Base.@propagate_inbounds get_land_freshwater_flux(i, j, flux) = flux[i, j, 1]

@kernel function _assemble_net_ocean_fluxes!(net_ocean_fluxes,
                                             grid,
                                             clock,
                                             atmos_ocean_fluxes,
                                             sea_ice_ocean_fluxes,
                                             ocean_surface_salinity,
                                             sea_ice_concentration,
                                             freshwater_flux,
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
        Sᵒᶜ = ocean_surface_salinity[i, j, 1]

        Jᶜ = freshwater_flux[i, j, 1] + get_land_freshwater_flux(i, j, land_freshwater_flux)
        𝒬ᵀ = atmos_ocean_fluxes.sensible_heat[i, j, 1]
        𝒬ᵛ = atmos_ocean_fluxes.latent_heat[i, j, 1]
        Jᵛ = atmos_ocean_fluxes.water_vapor[i, j, 1]
    end

    # Turbulent contributions to surface heat flux (radiation added later)
    ΣQao = (𝒬ᵀ + 𝒬ᵛ) * (1 - ℵᵢ)

    # Convert mass flux to volume flux; sign-flip (prescribed flux is positive down)
    ρᵒᶜ⁻¹ = 1 / ocean_properties.reference_density
    ΣFao = - Jᶜ * ρᵒᶜ⁻¹

    # Add turbulent water vapor flux (positive upward sign convention)
    Jᵛᵒᶜ = Jᵛ * ρᵒᶜ⁻¹
    ΣFao += Jᵛᵒᶜ

    τˣ = net_ocean_fluxes.u
    τʸ = net_ocean_fluxes.v
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
        τˣⁱᵒ = ρτˣⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑxᶠᵃᵃ(i, j, 1, grid, ℵ)
        τʸⁱᵒ = ρτʸⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑyᵃᶠᵃ(i, j, 1, grid, ℵ)

        τˣ[i, j, 1] = ifelse(inactive, zero(grid), τˣᵃᵒ + τˣⁱᵒ)
        τʸ[i, j, 1] = ifelse(inactive, zero(grid), τʸᵃᵒ + τʸⁱᵒ)

        # Tracer fluxes — radiative contributions added later by apply_air_sea_radiative_fluxes!
        Jᵀ[i, j, 1] = ifelse(inactive, zero(grid), Jᵀao + Jᵀio)
        Jˢ[i, j, 1] = ifelse(inactive, zero(grid), (1 - ℵᵢ) * Jˢao + Jˢio)
    end
end
