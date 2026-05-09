#####
##### `SlabLandParameters` — container-level scalar physical constants.
#####
##### These are the universal constants shared across closures. Closure-
##### specific tunables (heat capacities, bucket capacities, snow
##### thresholds, …) live on the closure structs themselves, not here.
#####

"""
    SlabLandParameters(FT = Float64;
                       gravitational_acceleration = 9.81,
                       density_of_water = 1000,
                       latent_heat_fusion = 3.337e5)

Universal scalar constants for `SlabLand`. Defaults are fresh-water /
standard-gravity values.
"""
struct SlabLandParameters{FT}
    gravitational_acceleration :: FT
    density_of_water           :: FT
    latent_heat_fusion         :: FT
end

function SlabLandParameters(FT::Type = Float64;
                            gravitational_acceleration = 9.81,
                            density_of_water = 1000,
                            latent_heat_fusion = 3.337e5)
    return SlabLandParameters{FT}(convert(FT, gravitational_acceleration),
                                  convert(FT, density_of_water),
                                  convert(FT, latent_heat_fusion))
end

Base.summary(::SlabLandParameters{FT}) where FT = "SlabLandParameters{$FT}"
