using KernelAbstractions: @kernel, @index
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ZeroField, OneField, set!
using Oceananigans.Utils: launch!

using ..EarthSystemModels: sea_ice_concentration, DegreesCelsius, convert_to_kelvin

"""
    SurfacePartition{F, T}

Fractional partition of each exchange-grid cell among the surface types the atmosphere
couples to. `ocean_fraction` (θ) is the fraction of the cell covered by ocean, so the
land fraction is `1 - θ`, and the sea ice concentration ℵ further splits the ocean
fraction into an open-ocean part `θ (1 - ℵ)` and an ice-covered part `θ ℵ`.
`surface_temperature` is the partition-weighted skin temperature (K) — the single
surface the atmosphere and radiation see; it is `nothing` unless the model carries
both a land and an ocean interface.
"""
struct SurfacePartition{F, T}
    ocean_fraction :: F
    surface_temperature :: T
end

Base.summary(partition::SurfacePartition) =
    string("SurfacePartition(ocean_fraction = ", summary(partition.ocean_fraction), ")")

Base.show(io::IO, partition::SurfacePartition) = print(io, summary(partition))

function SurfacePartition(grid, ocean_fraction, ao_interface, al_interface)
    isnothing(ao_interface) && return SurfacePartition(ZeroField(), nothing)
    isnothing(al_interface) && return SurfacePartition(OneField(), nothing)

    isnothing(ocean_fraction) &&
        throw(ArgumentError("a model with both ocean and land components requires an explicit `ocean_fraction`"))

    θ = Field{Center, Center, Nothing}(grid)
    set!(θ, ocean_fraction)
    fill_halo_regions!(θ)
    Tₛ = Field{Center, Center, Nothing}(grid)

    return SurfacePartition(θ, Tₛ)
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
            interfaces.atmosphere_land_interface.temperature,
            interfaces.ocean_properties.temperature_units,
            sea_ice_units)

    return nothing
end

@kernel function _update_surface_temperature!(Tₛ, θ, ℵ, Tᵃᵒ, Tᵃⁱ, Tᵃˡ, ocean_units, sea_ice_units)
    i, j = @index(Global, NTuple)
    @inbounds begin
        θᵢ = θ[i, j, 1]
        ℵᵢ = ℵ[i, j, 1]
        Tᵒ = convert_to_kelvin(ocean_units, Tᵃᵒ[i, j, 1])
        Tⁱ = convert_to_kelvin(sea_ice_units, Tᵃⁱ[i, j, 1])
        Tˡ = Tᵃˡ[i, j, 1]
        Tₛ[i, j, 1] = (1 - θᵢ) * Tˡ + θᵢ * ((1 - ℵᵢ) * Tᵒ + ℵᵢ * Tⁱ)
    end
end
