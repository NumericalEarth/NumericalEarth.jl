using ClimaSeaIce.SeaIceThermodynamics: LinearLiquidus
using DocStringExtensions: TYPEDEF, TYPEDFIELDS

import ClimaSeaIce.SeaIceThermodynamics: melting_temperature

"""
$(TYPEDEF)

A linear liquidus with a pressure (depth) correction, appropriate for the base
of an ice shelf (Jenkins et al. 2001; ISOMIP/ISOMIP+, Asay-Davis et al. 2016):

```math
Tₘ(S, z) = T₀ - m S + λ z ,
```

where ``T₀`` is the freshwater melting temperature at the surface, ``m > 0``
is the salinity slope, ``z ≤ 0`` is the depth of the interface (the ice
draft), and ``λ > 0`` is the depth slope expressed in °C/m, so that the
melting temperature *decreases* with depth.

The default coefficients are the ISOMIP+ values ``T₀ = 0.0832`` °C,
``m = 0.0573`` °C/psu, and ``λ₃ = -7.53 × 10⁻⁸`` °C/Pa converted to depth
units with ``p = -ρ₀ g z`` (``ρ₀ = 1028`` kg/m³, ``g = 9.81`` m/s²):
``λ = 7.53 × 10⁻⁸ ρ₀ g ≈ 7.59 × 10⁻⁴`` °C/m.

$(TYPEDFIELDS)
"""
struct PressureDependentLiquidus{FT}
    "freshwater melting temperature at the surface ``T₀`` (°C)"
    freshwater_melting_temperature :: FT
    "salinity slope ``m`` (°C/psu)"
    slope :: FT
    "depth slope ``λ`` (°C/m)"
    depth_slope :: FT
end

function PressureDependentLiquidus(FT::DataType = Oceananigans.defaults.FloatType;
                                   freshwater_melting_temperature = 0.0832,
                                   slope = 0.0573,
                                   depth_slope = 7.53e-8 * 1028 * 9.81)
    return PressureDependentLiquidus(convert(FT, freshwater_melting_temperature),
                                     convert(FT, slope),
                                     convert(FT, depth_slope))
end

@inline melting_temperature(liquidus::PressureDependentLiquidus, salinity, z) =
    liquidus.freshwater_melting_temperature - liquidus.slope * salinity + liquidus.depth_slope * z

# At fixed depth z the liquidus is linear with freshwater melting temperature T₀ + λ z.
@inline at_depth(liquidus::PressureDependentLiquidus{FT}, z) where FT =
    LinearLiquidus(liquidus.freshwater_melting_temperature + liquidus.depth_slope * convert(FT, z),
                   liquidus.slope)

@inline at_depth(liquidus::LinearLiquidus, z) = liquidus

Base.summary(::PressureDependentLiquidus{FT}) where FT = "PressureDependentLiquidus{$FT}"

function Base.show(io::IO, liq::PressureDependentLiquidus)
    print(io, summary(liq), '\n')
    print(io, "├── freshwater_melting_temperature: ", liq.freshwater_melting_temperature, " °C", '\n')
    print(io, "├── slope: ", liq.slope, " °C/psu", '\n')
    print(io, "└── depth_slope: ", liq.depth_slope, " °C/m")
end
