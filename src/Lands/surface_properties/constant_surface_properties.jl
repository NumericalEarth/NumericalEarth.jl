#####
##### `ConstantSurfaceProperties` — scalar α, ε, and roughness lengths uniform over the grid.
#####
##### The simplest surface-property closure. Returns `ConstantField`s
##### so callers can treat them like any other `AbstractField` without
##### allocating per-cell storage.
#####

"""
    ConstantSurfaceProperties(FT = Float64;
                              albedo = 0.2,
                              emissivity = 0.97,
                              momentum_roughness_length = 0.1,
                              scalar_roughness_length = 0.01)

Spatially uniform surface optical and aerodynamic properties.
"""
struct ConstantSurfaceProperties{FT} <: AbstractSurfaceProperties
    albedo                    :: FT
    emissivity                :: FT
    momentum_roughness_length :: FT
    scalar_roughness_length   :: FT
end

function ConstantSurfaceProperties(FT::Type = Float64;
                                   albedo = 0.2,
                                   emissivity = 0.97,
                                   momentum_roughness_length = 0.1,
                                   scalar_roughness_length = 0.01)
    return ConstantSurfaceProperties{FT}(convert(FT, albedo),
                                          convert(FT, emissivity),
                                          convert(FT, momentum_roughness_length),
                                          convert(FT, scalar_roughness_length))
end

albedo(s::ConstantSurfaceProperties, state)                    = ConstantField(s.albedo)
emissivity(s::ConstantSurfaceProperties, state)                = ConstantField(s.emissivity)
momentum_roughness_length(s::ConstantSurfaceProperties, state) = ConstantField(s.momentum_roughness_length)
scalar_roughness_length(s::ConstantSurfaceProperties, state)   = ConstantField(s.scalar_roughness_length)

Base.summary(s::ConstantSurfaceProperties{FT}) where FT =
    string("ConstantSurfaceProperties{$FT}(α=", s.albedo,
           ", ε=", s.emissivity,
           ", momentum_roughness_length=", s.momentum_roughness_length,
           ", scalar_roughness_length=", s.scalar_roughness_length, ")")
