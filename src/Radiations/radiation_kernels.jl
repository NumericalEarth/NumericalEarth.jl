@inline hack_cosd(φ) = cos(π * φ / 180)

@inline emitted_longwave_radiation(T, σ, ϵ) = σ * ϵ * T^4
@inline absorbed_longwave_radiation(ϵ, ℐꜜˡʷ) = - ϵ * ℐꜜˡʷ
@inline transmitted_shortwave_radiation(α, ℐꜜˢʷ) = - (1 - α) * ℐꜜˢʷ
