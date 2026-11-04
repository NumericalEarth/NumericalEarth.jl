using KernelAbstractions: @kernel, @index
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ZeroField, OneField, set!
using Oceananigans.Utils: launch!

using ..EarthSystemModels: sea_ice_concentration, DegreesCelsius, convert_to_kelvin

"""
    SurfacePartition{F, T}

Fractional partition of each exchange-grid cell among the surface types the atmosphere
couples to. `ocean_fraction` (þ) is the fraction of the cell covered by ocean, so the
land fraction is `1 - þ`, and the sea ice concentration ℵ further splits the ocean
fraction into an open-ocean part `þ (1 - ℵ)` and an ice-covered part `þ ℵ`.
`surface_temperature` is the partition-weighted skin temperature (K) — the single
surface the atmosphere and radiation see; it is `nothing` unless the model carries
both a land and an ocean interface and an atmosphere that feels the surface.
"""
struct SurfacePartition{F, T}
    ocean_fraction :: F
    surface_temperature :: T
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

function SurfacePartition(grid, atmosphere, ocean_fraction, ao_interface, atmosphere_land_interface)
    isnothing(ao_interface) && return SurfacePartition(ZeroField(), nothing)
    isnothing(atmosphere_land_interface) && return SurfacePartition(OneField(), nothing)

    feels_surface = atmosphere_feels_surface(atmosphere)

    if isnothing(ocean_fraction)
        feels_surface && throw(ArgumentError("a model with both ocean and land components " *
                                             "requires an explicit `ocean_fraction`"))
        return SurfacePartition(nothing, nothing)
    end

    þ = Field{Center, Center, Nothing}(grid)
    set!(þ, ocean_fraction)
    fill_halo_regions!(þ)
    Tⁱⁿ = feels_surface ? Field{Center, Center, Nothing}(grid) : nothing

    return SurfacePartition(þ, Tⁱⁿ)
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

    launch!(arch, grid, :xy, _update_surface_temperature!,
            partition.surface_temperature,
            partition.ocean_fraction,
            ℵ,
            interfaces.atmosphere_ocean_interface.temperature,
            Tᵃⁱ,
            EarthSystemModels.surface_temperature(interfaces.atmosphere_land_interface),
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
