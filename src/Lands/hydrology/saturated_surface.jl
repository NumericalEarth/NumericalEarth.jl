#####
##### `SaturatedSurface` — β ≡ 1. Wet-swamp / shallow-pond surface.
#####
##### Useful for testing that the land path matches the ocean path when
##### forced identically.
#####

struct SaturatedSurface <: AbstractHydrology end

prognostic_variables(::SaturatedSurface) = ()
flux_variables(::SaturatedSurface)       = ()

wetness(::SaturatedSurface, state, parameters) = ConstantField(1)

Base.summary(::SaturatedSurface) = "SaturatedSurface (β ≡ 1)"
