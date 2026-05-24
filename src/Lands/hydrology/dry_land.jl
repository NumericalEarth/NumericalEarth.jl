#####
##### `DryLand` — moisture availability β ≡ 0. Latent heat collapses to zero.
#####
##### Useful for desert idealisations and unit-test isolation of the
##### energy path.
#####

struct DryLand <: AbstractHydrology end

prognostic_variables(::DryLand) = ()
flux_variables(::DryLand)       = ()

wetness(::DryLand, state) = 0.0

Base.summary(::DryLand) = "DryLand (moisture availability β ≡ 0)"
