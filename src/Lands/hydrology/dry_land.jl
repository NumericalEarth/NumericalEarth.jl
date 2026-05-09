#####
##### `DryLand` — β ≡ 0. Latent heat collapses to zero.
#####
##### Useful for desert idealisations and unit-test isolation of the
##### energy path.
#####

struct DryLand <: AbstractHydrology end

prognostic_variables(::DryLand) = ()
flux_variables(::DryLand)       = ()

wetness(::DryLand, state, parameters) = ZeroField()

Base.summary(::DryLand) = "DryLand (β ≡ 0)"
