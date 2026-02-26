using Printf
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Forcings: MultipleForcings
using NumericalEarth.EarthSystemModels: EarthSystemModel, NoOceanInterfaceModel, NoInterfaceModel

using NumericalEarth.EarthSystemModels.InterfaceComputations: interface_kernel_parameters, 
                                                          computed_fluxes, 
                                                          get_possibly_zero_flux, 
                                                          sea_ice_concentration,
                                                          convert_to_kelvin,
                                                          emitted_longwave_radiation,
                                                          absorbed_longwave_radiation,
                                                          transmitted_shortwave_radiation
                                                          

@inline τᶜᶜᶜ(i, j, k, grid, ρᵒᶜ⁻¹, ℵ, ρτᶜᶜᶜ) = @inbounds ρᵒᶜ⁻¹ * (1 - ℵ[i, j, k]) * ρτᶜᶜᶜ[i, j, k]

#####
##### Generic flux assembler
#####

# Fallback for an ocean-only model (it has no interfaces!)
update_net_fluxes!(coupled_model::Union{NoOceanInterfaceModel, NoInterfaceModel}, ocean::Simulation{<:HydrostaticFreeSurfaceModel}) = nothing

update_net_fluxes!(coupled_model, ocean::Simulation{<:HydrostaticFreeSurfaceModel}) = 
    update_net_ocean_fluxes!(coupled_model, ocean, ocean.model.grid)

# A generic ocean flux assembler for a coupled model with both an atmosphere and sea ice
function update_net_ocean_fluxes!(coupled_model, ocean_model, grid)
    sea_ice = coupled_model.sea_ice
    arch = architecture(grid)
    clock = coupled_model.clock

    net_ocean_fluxes = coupled_model.interfaces.net_fluxes.ocean
    atmos_ocean_fluxes = computed_fluxes(coupled_model.interfaces.atmosphere_ocean_interface)
    sea_ice_ocean_fluxes = computed_fluxes(coupled_model.interfaces.sea_ice_ocean_interface)

    # Simplify NamedTuple to reduce parameter space consumption.
    # See https://github.com/CliMA/NumericalEarth.jl/issues/116.
    atmosphere_fields = coupled_model.interfaces.exchanger.atmosphere.state

    downwelling_radiation = (ℐꜜˢʷ = atmosphere_fields.ℐꜜˢʷ.data,
                             ℐꜜˡʷ = atmosphere_fields.ℐꜜˡʷ.data)

    freshwater_flux = atmosphere_fields.Jᶜ.data

    ice_concentration = sea_ice_concentration(sea_ice)
    ocean_surface_salinity = EarthSystemModels.ocean_surface_salinity(ocean_model)
    atmos_ocean_properties = coupled_model.interfaces.atmosphere_ocean_interface.properties
    ocean_properties = coupled_model.interfaces.ocean_properties

    ocean_surface_temperature = coupled_model.interfaces.atmosphere_ocean_interface.temperature
    penetrating_radiation = get_radiative_forcing(ocean_model)

    launch!(arch, grid, :xy,
            _assemble_net_ocean_fluxes!,
            net_ocean_fluxes,
            penetrating_radiation,
            grid,
            clock,
            atmos_ocean_fluxes,
            sea_ice_ocean_fluxes,
            ocean_surface_salinity,
            ocean_surface_temperature,
            ice_concentration,
            downwelling_radiation,
            freshwater_flux,
            atmos_ocean_properties,
            ocean_properties)

    return nothing
end

@kernel function _assemble_net_ocean_fluxes!(net_ocean_fluxes,
                                             penetrating_radiation,
                                             grid,
                                             clock,
                                             atmos_ocean_fluxes,
                                             sea_ice_ocean_fluxes,
                                             ocean_surface_salinity,
                                             ocean_surface_temperature,
                                             sea_ice_concentration,
                                             downwelling_radiation,
                                             freshwater_flux,
                                             atmos_ocean_properties,
                                             ocean_properties)

    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    time = Time(clock.time)
    ρτˣᵃᵒ = get_possibly_zero_flux(atmos_ocean_fluxes,   :x_momentum) # atmosphere - ocean zonal momentum flux
    ρτʸᵃᵒ = get_possibly_zero_flux(atmos_ocean_fluxes,   :y_momentum) # atmosphere - ocean meridional momentum flux
    ρτˣⁱᵒ = get_possibly_zero_flux(sea_ice_ocean_fluxes, :x_momentum) # sea_ice - ocean zonal momentum flux
    ρτʸⁱᵒ = get_possibly_zero_flux(sea_ice_ocean_fluxes, :y_momentum) # sea_ice - ocean meridional momentum flux

    @inbounds begin
        ℵᵢ = sea_ice_concentration[i, j, 1]
        Sᵒᶜ = ocean_surface_salinity[i, j, 1]
        Tₛ = ocean_surface_temperature[i, j, 1]
        Tₛ = convert_to_kelvin(ocean_properties.temperature_units, Tₛ)

        Jᶜ  = freshwater_flux[i, j, 1] # Prescribed freshwater (condensate) flux
        ℐꜜˢʷ = downwelling_radiation.ℐꜜˢʷ[i, j, 1] # Downwelling shortwave radiation
        ℐꜜˡʷ = downwelling_radiation.ℐꜜˡʷ[i, j, 1] # Downwelling longwave radiation
        𝒬ᵀ  = get_possibly_zero_flux(atmos_ocean_fluxes, :sensible_heat)[i, j, 1] # sensible or "conductive" heat flux
        𝒬ᵛ  = get_possibly_zero_flux(atmos_ocean_fluxes, :latent_heat)[i, j, 1] # latent heat flux
        Jᵛ  = get_possibly_zero_flux(atmos_ocean_fluxes, :water_vapor)[i, j, 1] # mass flux of water vapor
    end

    # Compute radiation fluxes (radiation is multiplied by the fraction of ocean, 1 - sea ice concentration)
    σ = atmos_ocean_properties.radiation.σ
    α = atmos_ocean_properties.radiation.α
    ϵ = atmos_ocean_properties.radiation.ϵ
    ℐꜛˡʷ = emitted_longwave_radiation(i, j, kᴺ, grid, time, Tₛ, σ, ϵ)
    ℐₐˡʷ = absorbed_longwave_radiation(i, j, kᴺ, grid, time, ϵ, ℐꜜˡʷ)

    # Compute the interior + surface absorbed shortwave radiation
    ℐₜˢʷ = transmitted_shortwave_radiation(i, j, kᴺ, grid, time, α, ℐꜜˢʷ)

    ℐₐˡʷ *= (1 - ℵᵢ)
    ℐₜˢʷ *= (1 - ℵᵢ)

    Qss = shortwave_radiative_forcing(i, j, grid, penetrating_radiation, ℐₜˢʷ, ocean_properties)

    # Compute the total heat flux
    ΣQao = (ℐꜛˡʷ + 𝒬ᵀ + 𝒬ᵛ) * (1 - ℵᵢ) + ℐₐˡʷ + Qss

    @inbounds begin
        # Write radiative components of the heat flux for diagnostic purposes
        atmos_ocean_fluxes.upwelling_longwave[i, j, 1] = ℐꜛˡʷ
        atmos_ocean_fluxes.downwelling_longwave[i, j, 1] = - ℐₐˡʷ
        atmos_ocean_fluxes.downwelling_shortwave[i, j, 1] = - ℐₜˢʷ
    end

    # Convert from a mass flux to a volume flux (aka velocity)
    # by dividing with the ocean reference density.
    # Also switch the sign, for some reason we are given freshwater flux as positive down.
    ρᵒᶜ⁻¹ = 1 / ocean_properties.reference_density
    ΣFao = - Jᶜ * ρᵒᶜ⁻¹

    # Add the contribution from the turbulent water vapor flux, which has
    # a different sign convention as the prescribed water mass fluxes (positive upwards)
    Jᵛᵒᶜ = Jᵛ * ρᵒᶜ⁻¹
    ΣFao += Jᵛᵒᶜ

    # Compute fluxes for u, v, T, and S from momentum, heat, and freshwater fluxes
    τˣ = net_ocean_fluxes.u
    τʸ = net_ocean_fluxes.v
    Jᵀ = net_ocean_fluxes.T
    Jˢ = net_ocean_fluxes.S
    ℵ  = sea_ice_concentration
    cᵒᶜ = ocean_properties.heat_capacity

    @inbounds begin
        𝒬ⁱⁿᵗ = get_possibly_zero_flux(sea_ice_ocean_fluxes, :interface_heat)[i, j, 1]
        Jˢio = get_possibly_zero_flux(sea_ice_ocean_fluxes, :salt)[i, j, 1]
        Jᵀao = ΣQao  * ρᵒᶜ⁻¹ / cᵒᶜ
        Jᵀio = 𝒬ⁱⁿᵗ * ρᵒᶜ⁻¹ / cᵒᶜ
    
        # salinity flux > 0 extracts salinity from the ocean --- the opposite of a water vapor flux
        Jˢao = - Sᵒᶜ * ΣFao

        τˣᵃᵒ = ℑxᶠᵃᵃ(i, j, 1, grid, τᶜᶜᶜ, ρᵒᶜ⁻¹, ℵ, ρτˣᵃᵒ)
        τʸᵃᵒ = ℑyᵃᶠᵃ(i, j, 1, grid, τᶜᶜᶜ, ρᵒᶜ⁻¹, ℵ, ρτʸᵃᵒ)
        τˣⁱᵒ = ρτˣⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑxᶠᵃᵃ(i, j, 1, grid, ℵ)
        τʸⁱᵒ = ρτʸⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑyᵃᶠᵃ(i, j, 1, grid, ℵ)

        # Stresses
        τˣ[i, j, 1] = τˣᵃᵒ + τˣⁱᵒ
        τʸ[i, j, 1] = τʸᵃᵒ + τʸⁱᵒ

        # Tracer fluxes
        Jᵀ[i, j, 1] = Jᵀao + Jᵀio # Jᵀao is already multiplied by the sea ice concentration
        Jˢ[i, j, 1] = (1 - ℵᵢ) * Jˢao + Jˢio
    end
end
