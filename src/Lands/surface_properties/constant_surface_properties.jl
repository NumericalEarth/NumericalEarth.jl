#####
##### `ConstantSurfaceProperties` — scalar roughness lengths uniform over the grid.
#####
##### The simplest aerodynamic surface-property closure. Returns
##### `ConstantField`s so callers can treat them like any other
##### `AbstractField` without allocating per-cell storage.
#####
##### Radiative properties (albedo, emissivity) live on the top-level
##### `radiation` component, not here.
#####

"""
    ConstantSurfaceProperties(FT = Float64;
                              momentum_roughness_length = 0.1,
                              scalar_roughness_length = 0.01)

Spatially uniform aerodynamic surface properties.
"""
struct ConstantSurfaceProperties{FT} <: AbstractSurfaceProperties
    momentum_roughness_length :: FT
    scalar_roughness_length   :: FT
end

function ConstantSurfaceProperties(FT::Type = Float64;
                                   momentum_roughness_length = 0.1,
                                   scalar_roughness_length = 0.01)
    return ConstantSurfaceProperties{FT}(convert(FT, momentum_roughness_length),
                                          convert(FT, scalar_roughness_length))
end

momentum_roughness_length(s::ConstantSurfaceProperties, state) = ConstantField(s.momentum_roughness_length)
scalar_roughness_length(s::ConstantSurfaceProperties, state)   = ConstantField(s.scalar_roughness_length)

Base.summary(s::ConstantSurfaceProperties{FT}) where FT =
    string("ConstantSurfaceProperties{$FT}(momentum_roughness_length=", s.momentum_roughness_length,
           ", scalar_roughness_length=", s.scalar_roughness_length, ")")
