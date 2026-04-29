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
SurfaceRadiationProperties(; albedo, emissivity) = SurfaceRadiationProperties(albedo, emissivity)

Adapt.adapt_structure(to, s::SurfaceRadiationProperties) =
    SurfaceRadiationProperties(Adapt.adapt(to, s.albedo),
                               Adapt.adapt(to, s.emissivity))

Base.summary(::SurfaceRadiationProperties) = "SurfaceRadiationProperties"

function Base.show(io::IO, s::SurfaceRadiationProperties)
    print(io, summary(s), ":", '\n')
    print(io, "├── albedo: ",     prettysummary(s.albedo), '\n')
    print(io, "└── emissivity: ", prettysummary(s.emissivity))
end
