#####
##### `ConstantSurfaceProperties` — scalar α, ε, z₀ uniform over the grid.
#####
##### The simplest surface-property closure. Returns `ConstantField`s
##### so callers can treat them like any other `AbstractField` without
##### allocating per-cell storage.
#####

"""
    ConstantSurfaceProperties(FT = Float64;
                              albedo = 0.2,
                              emissivity = 0.97,
                              z0_m = 0.1,
                              z0_h = 0.01)

Spatially uniform surface optical and aerodynamic properties.
"""
struct ConstantSurfaceProperties{FT} <: AbstractSurfaceProperties
    albedo     :: FT
    emissivity :: FT
    z0_m       :: FT
    z0_h       :: FT
end

function ConstantSurfaceProperties(FT::Type = Float64;
                                   albedo = 0.2,
                                   emissivity = 0.97,
                                   z0_m = 0.1,
                                   z0_h = 0.01)
    return ConstantSurfaceProperties{FT}(convert(FT, albedo),
                                          convert(FT, emissivity),
                                          convert(FT, z0_m),
                                          convert(FT, z0_h))
end

albedo(s::ConstantSurfaceProperties, state, parameters)                    = ConstantField(s.albedo)
emissivity(s::ConstantSurfaceProperties, state, parameters)                = ConstantField(s.emissivity)
momentum_roughness_length(s::ConstantSurfaceProperties, state, parameters) = ConstantField(s.z0_m)
scalar_roughness_length(s::ConstantSurfaceProperties, state, parameters)   = ConstantField(s.z0_h)

Base.summary(s::ConstantSurfaceProperties{FT}) where FT =
    string("ConstantSurfaceProperties{$FT}(α=", s.albedo,
           ", ε=", s.emissivity, ", z₀ₘ=", s.z0_m, ", z₀ₕ=", s.z0_h, ")")
