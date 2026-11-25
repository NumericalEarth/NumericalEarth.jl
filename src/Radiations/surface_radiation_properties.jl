struct SurfaceRadiationProperties{A, E}
    albedo :: A
    emissivity :: E
end

"""
    SurfaceRadiationProperties(albedo, emissivity)

Bundle the radiative properties of a single surface (ocean, sea ice, snow, land)
that participate in radiative flux computation: shortwave reflectivity (`albedo`)
and longwave emissivity (`emissivity`).

`albedo` may be a `Number`, a `LatitudeDependentAlbedo`, a `TabulatedAlbedo`, or
any other object for which `stateindex` is defined. `emissivity` may be a `Number`
or any other `stateindex`-able object.
"""
SurfaceRadiationProperties(; albedo, emissivity = default_water_emissivity) =
    SurfaceRadiationProperties(albedo, emissivity)

Adapt.adapt_structure(to, s::SurfaceRadiationProperties) =
    SurfaceRadiationProperties(Adapt.adapt(to, s.albedo),
                               Adapt.adapt(to, s.emissivity))

# The radiative properties of a surface with its numbers in the float type `FT`. An albedo or emissivity that is
# not a number (a `Field`, a `SeaIceAlbedo`, a `LatitudeDependentAlbedo`) carries its own float type and is kept.
# The interface kernels compute with these: a `Float64` albedo on a `Float32` grid promotes the skin temperature
# solve to `Float64` and makes it type-unstable.
convert_surface_radiation_properties(FT, ::Nothing) = nothing
convert_surface_radiation_properties(FT, s::SurfaceRadiationProperties) =
    SurfaceRadiationProperties(convert_if_number(FT, s.albedo), convert_if_number(FT, s.emissivity))

Base.summary(::SurfaceRadiationProperties) = "SurfaceRadiationProperties"

function Base.show(io::IO, s::SurfaceRadiationProperties)
    print(io, summary(s), ":", '\n')
    print(io, "├── albedo: ",     prettysummary(s.albedo), '\n')
    print(io, "└── emissivity: ", prettysummary(s.emissivity))
end

#####
##### Default properties of the ocean and sea ice surfaces, for radiations that do not carry them
#####

# Albedo of open water under an overhead sun (the plateau of `TabulatedAlbedo`'s Payne (1972) table)
const default_ocean_albedo = 0.06

# The `PrescribedRadiation` default for sea ice
const default_sea_ice_albedo = 0.7
const default_sea_ice_emissivity = 1.0

"""
$(SIGNATURES)

Radiative properties of open water on `grid`: emissivity `default_water_emissivity` and, on a
`LatitudeLongitudeGrid`, the [`LatitudeDependentAlbedo`](@ref) of
[Large and Yeager (2009)](@cite large2009global), 0.069 - 0.011 cos(2φ). Where the grid has no
latitude the albedo is `default_ocean_albedo`.
"""
default_ocean_surface_radiation(grid) =
    SurfaceRadiationProperties(convert(eltype(grid), default_ocean_albedo),
                               convert(eltype(grid), default_water_emissivity))

default_ocean_surface_radiation(grid::LatitudeLongitudeGrid) =
    SurfaceRadiationProperties(LatitudeDependentAlbedo(eltype(grid)),
                               convert(eltype(grid), default_water_emissivity))

"""
$(SIGNATURES)

Radiative properties of the surface of `sea_ice` on `grid`: albedo `default_sea_ice_albedo` and
emissivity `default_sea_ice_emissivity`, unless the sea ice component provides its own.
"""
default_sea_ice_surface_radiation(sea_ice, grid) =
    SurfaceRadiationProperties(convert(eltype(grid), default_sea_ice_albedo),
                               convert(eltype(grid), default_sea_ice_emissivity))

"""
    interface_radiation_fluxes(radiation, radiation_exchanger)

Named tuple of the [`InterfaceRadiationFlux`](@ref) diagnostics of each surface, or `nothing` for a
radiation that keeps none. `PrescribedRadiation` holds its own; a radiation without storage for them
keeps them in its exchanger.
"""
interface_radiation_fluxes(radiation, radiation_exchanger) = nothing
