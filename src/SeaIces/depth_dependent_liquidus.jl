using ClimaSeaIce.SeaIceThermodynamics: SeaIceThermodynamics

struct DepthDependentLiquidus{FT}
    freshwater_melting_temperature :: FT
    slope :: FT
    depth_slope :: FT
end

"""
$(TYPEDSIGNATURES)

Return a liquidus that depends linearly on salinity `S` and on the vertical coordinate `z`,

```math
Tₘ(S, z) = T₀ - m S + λ z ,
```

where ``T₀`` is the `freshwater_melting_temperature`, ``m`` is the `slope`, and ``λ`` is the `depth_slope`, so that
the melting temperature decreases with salinity and with depth (``z < 0`` below the surface).
`melting_temperature(liquidus, S)` returns the surface value, `melting_temperature(liquidus, S, z)` the value at `z`.

The defaults fit the TEOS-10 freezing point of air-saturated seawater, in Conservative Temperature, within 0.02 K
over `S = 28-36 g kg⁻¹` and the top 1000 m.
"""
function DepthDependentLiquidus(FT::DataType=Oceananigans.defaults.FloatType;
                                freshwater_melting_temperature = 0, # ᵒC
                                slope = 0.0542,                     # ᵒC kg g⁻¹
                                depth_slope = 7.89e-4)              # ᵒC m⁻¹

    return DepthDependentLiquidus(convert(FT, freshwater_melting_temperature),
                                  convert(FT, slope),
                                  convert(FT, depth_slope))
end

@inline SeaIceThermodynamics.melting_temperature(liquidus::DepthDependentLiquidus, S, z=0) =
    liquidus.freshwater_melting_temperature - liquidus.slope * S + liquidus.depth_slope * z
