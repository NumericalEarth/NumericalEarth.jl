#####
##### `ConstantSurfaceProperties` — minimal aerodynamic surface-property closure.
#####
##### The simplest aerodynamic surface-property closure. It accepts either
##### uniform scalars or per-cell Fields and materializes scalars to
##### constant fields the coupler can index uniformly.
#####
##### Radiative properties (albedo, emissivity) live on the top-level
##### `radiation` component, not here.
#####

"""
    ConstantSurfaceProperties(FT = Float64;
                              momentum_roughness_length = 0.1,
                              scalar_roughness_length = 0.01)

Minimal aerodynamic surface properties. Roughness lengths may be scalars
or per-cell `Field`s.
"""
struct ConstantSurfaceProperties{M, S} <: AbstractSurfaceProperties
    momentum_roughness_length :: M
    scalar_roughness_length   :: S
end

function ConstantSurfaceProperties(FT::Type = Float64;
                                   momentum_roughness_length = 0.1,
                                   scalar_roughness_length = 0.01)
    momentum_roughness_length = normalize_property(FT, momentum_roughness_length)
    scalar_roughness_length   = normalize_property(FT, scalar_roughness_length)
    return ConstantSurfaceProperties(momentum_roughness_length, scalar_roughness_length)
end

momentum_roughness_length(s::ConstantSurfaceProperties, state) = s.momentum_roughness_length
scalar_roughness_length(s::ConstantSurfaceProperties, state)   = s.scalar_roughness_length

Base.summary(s::ConstantSurfaceProperties) =
    string("ConstantSurfaceProperties(momentum_roughness_length=", prettysummary(s.momentum_roughness_length),
           ", scalar_roughness_length=", prettysummary(s.scalar_roughness_length), ")")
