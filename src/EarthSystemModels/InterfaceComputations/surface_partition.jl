using KernelAbstractions: @kernel, @index
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ZeroField, OneField, set!
using Oceananigans.Utils: launch!

using ..EarthSystemModels: sea_ice_concentration, DegreesCelsius, convert_to_kelvin

"""
    SurfacePartition{F, T, J}

Fractional partition of each exchange-grid cell among the surface types the atmosphere
couples to. `ocean_fraction` (þ) is the fraction of the cell covered by ocean, so the
land fraction is `1 - þ`, and the sea ice concentration ℵ further splits the ocean
fraction into an open-ocean part `þ (1 - ℵ)` and an ice-covered part `þ ℵ`.
`surface_temperature` is the partition-weighted skin temperature (K) and `fluxes` the
partition-weighted turbulent fluxes: the single surface the atmosphere and radiation see.
Both are `nothing` for an atmosphere that does not feel the surface.
"""
struct SurfacePartition{F, T, J}
    ocean_fraction :: F
    surface_temperature :: T
    fluxes :: J
end

Base.summary(partition::SurfacePartition) =
    string("SurfacePartition(ocean_fraction = ", summary(partition.ocean_fraction), ")")

Base.show(io::IO, partition::SurfacePartition) = print(io, summary(partition))

"""
    atmosphere_feels_surface(atmosphere)

Whether `atmosphere` responds to the surface it is coupled to, and so needs the partition-weighted
fluxes and skin temperature. A prescribed atmosphere (`false`, method added in `Atmospheres`) or no
atmosphere at all does not: its surfaces only receive fluxes, each per unit of its own area.
"""
atmosphere_feels_surface(atmosphere) = true
atmosphere_feels_surface(::Nothing) = false

function surface_flux_fields(grid)
    F = Field{Center, Center, Nothing}
    velocity_bcs = vector_component_boundary_conditions(grid, (Center(), Center(), nothing))
    return (; sensible_heat = F(grid),
              water_vapor = F(grid),
              x_momentum = F(grid; boundary_conditions = velocity_bcs),
              y_momentum = F(grid; boundary_conditions = velocity_bcs))
end

function SurfacePartition(grid, atmosphere, ocean_fraction, ao_interface, atmosphere_land_interface)
    feels_surface = atmosphere_feels_surface(atmosphere)
    fluxes = feels_surface ? surface_flux_fields(grid) : nothing

    isnothing(ao_interface) && return SurfacePartition(ZeroField(), nothing, fluxes)

    Tⁱⁿ = feels_surface ? Field{Center, Center, Nothing}(grid) : nothing

    isnothing(atmosphere_land_interface) && return SurfacePartition(OneField(), Tⁱⁿ, fluxes)

    if isnothing(ocean_fraction)
        feels_surface && throw(ArgumentError("a model with both ocean and land components " *
                                             "requires an explicit `ocean_fraction`"))
        return SurfacePartition(nothing, nothing, nothing)
    end

    þ = Field{Center, Center, Nothing}(grid)
    set!(þ, ocean_fraction)
    fill_halo_regions!(þ)

    return SurfacePartition(þ, Tⁱⁿ, fluxes)
end

#####
##### Partition-weighted surface (skin) temperature
#####

update_surface_temperature!(coupled_model) =
    update_surface_temperature!(coupled_model.interfaces.surface_partition, coupled_model)

update_surface_temperature!(::SurfacePartition{<:Any, Nothing}, coupled_model) = nothing

function update_surface_temperature!(partition::SurfacePartition, coupled_model)
    interfaces = coupled_model.interfaces
    grid = interfaces.exchanger.grid
    arch = architecture(grid)

    ℵ = sea_ice_concentration(coupled_model.sea_ice)
    ai_interface = interfaces.atmosphere_sea_ice_interface
    Tᵃⁱ = isnothing(ai_interface) ? ZeroField(eltype(grid)) : ai_interface.temperature
    sea_ice_properties = interfaces.sea_ice_properties
    sea_ice_units = isnothing(sea_ice_properties) ? DegreesCelsius() : sea_ice_properties.temperature_units
    al_interface = interfaces.atmosphere_land_interface
    Tᵃˡ = isnothing(al_interface) ? ZeroField(eltype(grid)) : EarthSystemModels.surface_temperature(al_interface)

    launch!(arch, grid, :xy, _update_surface_temperature!,
            partition.surface_temperature,
            partition.ocean_fraction,
            ℵ,
            interfaces.atmosphere_ocean_interface.temperature,
            Tᵃⁱ,
            Tᵃˡ,
            interfaces.ocean_properties.temperature_units,
            sea_ice_units)

    return nothing
end

@kernel function _update_surface_temperature!(Tⁱⁿ, þ, ℵ, Tᵃᵒ, Tᵃⁱ, Tᵃˡ, ocean_units, sea_ice_units)
    i, j = @index(Global, NTuple)
    @inbounds begin
        þᵢ = þ[i, j, 1]
        ℵᵢ = ℵ[i, j, 1]
        Tᵒ = convert_to_kelvin(ocean_units, Tᵃᵒ[i, j, 1])
        Tⁱ = convert_to_kelvin(sea_ice_units, Tᵃⁱ[i, j, 1])
        Tˡ = Tᵃˡ[i, j, 1]
        Tⁱⁿ[i, j, 1] = (1 - þᵢ) * Tˡ + þᵢ * ((1 - ℵᵢ) * Tᵒ + ℵᵢ * Tⁱ)
    end
end

#####
##### Partition-weighted turbulent fluxes
#####

update_surface_fluxes!(coupled_model) =
    update_surface_fluxes!(coupled_model.interfaces.surface_partition, coupled_model)

update_surface_fluxes!(::SurfacePartition{<:Any, <:Any, Nothing}, coupled_model) = nothing

function update_surface_fluxes!(partition::SurfacePartition, coupled_model)
    interfaces = coupled_model.interfaces
    grid = interfaces.exchanger.grid

    launch!(architecture(grid), grid, interface_kernel_parameters(grid), _update_surface_fluxes!,
            partition.fluxes,
            computed_fluxes(interfaces.atmosphere_ocean_interface),
            computed_fluxes(interfaces.atmosphere_sea_ice_interface),
            computed_fluxes(interfaces.atmosphere_land_interface),
            partition.ocean_fraction,
            sea_ice_concentration(coupled_model.sea_ice))

    return nothing
end

@kernel function _update_surface_fluxes!(fluxes, ao_fluxes, ai_fluxes, al_fluxes, þ, ℵ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        open_ocean = þ[i, j, 1] * (1 - ℵ[i, j, 1])
        ice_covered = þ[i, j, 1] * ℵ[i, j, 1]
        land = 1 - þ[i, j, 1]

        fluxes.sensible_heat[i, j, 1] = open_ocean * ao_fluxes.sensible_heat[i, j, 1] + ice_covered * ai_fluxes.sensible_heat[i, j, 1] + land * al_fluxes.sensible_heat[i, j, 1]
        fluxes.water_vapor[i, j, 1]   = open_ocean * ao_fluxes.water_vapor[i, j, 1]   + ice_covered * ai_fluxes.water_vapor[i, j, 1]   + land * al_fluxes.water_vapor[i, j, 1]
        fluxes.x_momentum[i, j, 1]    = open_ocean * ao_fluxes.x_momentum[i, j, 1]    + ice_covered * ai_fluxes.x_momentum[i, j, 1]    + land * al_fluxes.x_momentum[i, j, 1]
        fluxes.y_momentum[i, j, 1]    = open_ocean * ao_fluxes.y_momentum[i, j, 1]    + ice_covered * ai_fluxes.y_momentum[i, j, 1]    + land * al_fluxes.y_momentum[i, j, 1]
    end
end
