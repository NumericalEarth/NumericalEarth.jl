@inline hack_cosd(φ) = cos(π * φ / 180)
@inline hack_sind(φ) = sin(π * φ / 180)

const CCC = (Center, Center, Center)

@inline function emitted_longwave_radiation(i, j, k, grid, time, T, σ, ϵ)
    ϵi = stateindex(ϵ, i, j, k, grid, time, CCC)
    return σ * ϵi * T^4
end

@inline function absorbed_longwave_radiation(i, j, k, grid, time, ϵ, ℐꜜˡʷ)
    ϵi = stateindex(ϵ, i, j, k, grid, time, CCC)
    return - ϵi * ℐꜜˡʷ
end

@inline function transmitted_shortwave_radiation(i, j, k, grid, time, α, ℐꜜˢʷ)
    αi = stateindex(α, i, j, k, grid, time, CCC, ℐꜜˢʷ)
    return - (1 - αi) * ℐꜜˢʷ
end

# Inside the solver we lose both spatial and temporal information, but the
# radiative properties have already been computed correctly
@inline net_absorbed_interface_radiation(ℐꜜˢʷ, ℐꜜˡʷ, α, ϵ) = - (1 - α) * ℐꜜˢʷ - ϵ * ℐꜜˡʷ
@inline emitted_longwave_radiation(T, σ, ϵ) = σ * ϵ * T^4
