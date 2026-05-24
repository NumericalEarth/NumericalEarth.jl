#####
##### `SaturatedSurface` — moisture availability β ≡ 1. Wet-swamp / shallow-pond surface.
#####
##### Useful for testing that the land path matches the ocean path when
##### forced identically.
#####

struct SaturatedSurface <: AbstractHydrology end

prognostic_variables(::SaturatedSurface) = ()
flux_variables(::SaturatedSurface)       = ()

wetness(::SaturatedSurface, state) = 1.0

Base.summary(::SaturatedSurface) = "SaturatedSurface (moisture availability β ≡ 1)"
