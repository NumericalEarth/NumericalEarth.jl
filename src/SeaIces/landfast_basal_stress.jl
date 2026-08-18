using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES

using ClimaSeaIce: SeaIceDynamics
using ClimaSeaIce.SeaIceDynamics: StressBalanceFreeDrift

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: znode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, static_column_depthᶜᶜᵃ

"""
    LandfastBasalStress{FT, W}

Basal stress exerted by the seabed on grounded sea-ice keels, following
[Lemieux et al. (2015)](@cite Lemieux2015). Where the ice is thick enough that its keels reach the
sea floor, the seabed arrests the ice, producing the landfast belt observed along shallow Arctic
shelves and through the narrow channels of the Canadian Arctic Archipelago.

The stress is

```math
τᵇ = k₂ \\max(0, h - hᶜ) \\exp[-C (1 - ℵ)] \\frac{𝐮}{|𝐮| + u₀}, \\qquad hᶜ = \\frac{H ℵ}{k₁}
```

where ``H`` is the still-water column depth, ``h`` the ice thickness, ``ℵ`` the ice concentration,
and ``u₀`` a small velocity that keeps the stress finite as ``𝐮 → 0``. The stress acts only where
``H`` is smaller than `maximum_water_depth`.

Wrap it together with the ice-ocean stress in a `SeaIceBottomStress` so that both are treated
semi-implicitly by the momentum substepping.
"""
struct LandfastBasalStress{FT, W}
    critical_thickness_parameter :: FT
    stress_parameter :: FT
    concentration_hardening :: FT
    minimum_speed :: FT
    maximum_water_depth :: FT
    water_depth :: W
end

"""
$(TYPEDSIGNATURES)

Construct a `LandfastBasalStress`. The returned object is a skeleton whose `water_depth` is
`nothing`; the depth field is built by `materialize_stress` once the velocity grid is known.

Keyword Arguments
=================

- `critical_thickness_parameter`: ``k₁``, setting the keel thickness that grounds in a column of
                                  depth ``H``. Default: `8`.
- `stress_parameter`: ``k₂`` (N m⁻³). Default: `15`.
- `concentration_hardening`: ``C``, suppressing the stress in unconsolidated ice. Default: `20`.
- `minimum_speed`: ``u₀`` (m s⁻¹), regularizing the stress at vanishing velocity. Default: `5e-5`.
- `maximum_water_depth`: depth (m) beyond which no grounding is possible. Default: `30`.

```jldoctest
using NumericalEarth.SeaIces: LandfastBasalStress

LandfastBasalStress()

# output
LandfastBasalStress{Float64}
├── critical_thickness_parameter: 8.0
├── stress_parameter: 15.0
├── concentration_hardening: 20.0
├── minimum_speed: 5.0e-5
└── maximum_water_depth: 30.0
```
"""
function LandfastBasalStress(FT::DataType = Oceananigans.defaults.FloatType;
                             critical_thickness_parameter = 8,
                             stress_parameter = 15,
                             concentration_hardening = 20,
                             minimum_speed = 5e-5,
                             maximum_water_depth = 30)

    return LandfastBasalStress(convert(FT, critical_thickness_parameter),
                               convert(FT, stress_parameter),
                               convert(FT, concentration_hardening),
                               convert(FT, minimum_speed),
                               convert(FT, maximum_water_depth),
                               nothing)
end

Base.summary(::LandfastBasalStress{FT}) where FT = "LandfastBasalStress{$FT}"

function Base.show(io::IO, b::LandfastBasalStress)
    print(io, summary(b), '\n')
    print(io, "├── critical_thickness_parameter: ", b.critical_thickness_parameter, '\n')
    print(io, "├── stress_parameter: ", b.stress_parameter, '\n')
    print(io, "├── concentration_hardening: ", b.concentration_hardening, '\n')
    print(io, "├── minimum_speed: ", b.minimum_speed, '\n')
    print(io, "└── maximum_water_depth: ", b.maximum_water_depth)
end

Adapt.adapt_structure(to, b::LandfastBasalStress) =
    LandfastBasalStress(b.critical_thickness_parameter,
                        b.stress_parameter,
                        b.concentration_hardening,
                        b.minimum_speed,
                        b.maximum_water_depth,
                        Adapt.adapt(to, b.water_depth))

@inline column_depth(i, j, grid) = @inbounds - znode(i, j, 1, grid, Center(), Center(), Face())
@inline column_depth(i, j, grid::ImmersedBoundaryGrid) = static_column_depthᶜᶜᵃ(i, j, grid)

@kernel function _compute_water_depth!(H, grid)
    i, j = @index(Global, NTuple)
    @inbounds H[i, j, 1] = column_depth(i, j, grid)
end

materialize_basal_stress(::Nothing, grid) = nothing

function materialize_basal_stress(b::LandfastBasalStress, grid)
    H = Field{Center, Center, Nothing}(grid)
    launch!(architecture(grid), grid, :xy, _compute_water_depth!, H, grid)
    fill_halo_regions!(H)

    return LandfastBasalStress(b.critical_thickness_parameter,
                               b.stress_parameter,
                               b.concentration_hardening,
                               b.minimum_speed,
                               b.maximum_water_depth,
                               H)
end

# Grounded keel thickness in excess of what the column can accommodate, hardened by concentration.
# Grounding is a property of the cell, so the magnitude is formed at centres and only then
# interpolated: averaging h, ℵ and H to a velocity point first halves both the keel and the critical
# thickness against a dry neighbour, and the criterion silently cancels itself along every coastline.
@inline function basal_stress_magnitude(i, j, k, grid, b, fields)
    h = @inbounds fields.h[i, j, 1]
    ℵ = @inbounds fields.ℵ[i, j, 1]
    H = @inbounds b.water_depth[i, j, 1]
    δh = max(0, h - H * ℵ / b.critical_thickness_parameter)
    kᵇ = b.stress_parameter * δh * exp(- b.concentration_hardening * (1 - ℵ))
    return ifelse(H < b.maximum_water_depth, kᵇ, zero(grid))
end

@inline function basal_τx_coefficient(i, j, k, grid, b::LandfastBasalStress, fields)
    kᵇ = ℑxᶠᵃᵃ(i, j, 1, grid, basal_stress_magnitude, b, fields)
    u = @inbounds fields.u[i, j, k]
    v = ℑxyᶠᶜᵃ(i, j, k, grid, fields.v)
    return kᵇ / (sqrt(u^2 + v^2) + b.minimum_speed)
end

@inline function basal_τy_coefficient(i, j, k, grid, b::LandfastBasalStress, fields)
    kᵇ = ℑyᵃᶠᵃ(i, j, 1, grid, basal_stress_magnitude, b, fields)
    u = ℑxyᶜᶠᵃ(i, j, k, grid, fields.u)
    v = @inbounds fields.v[i, j, k]
    return kᵇ / (sqrt(u^2 + v^2) + b.minimum_speed)
end

@inline basal_τx_coefficient(i, j, k, grid, ::Nothing, fields) = zero(grid)
@inline basal_τy_coefficient(i, j, k, grid, ::Nothing, fields) = zero(grid)

"""
    SeaIceBottomStress{O, B}

The total stress on the underside of the sea ice: the `ocean` drag plus an optional `basal` stress
from grounded keels. Both are velocity-dependent, so both enter the implicit part of the momentum
substep. Only the `ocean` component is transmitted to the ocean — the basal stress is carried by the
sea floor — so the coupler reads it through `ice_ocean_momentum_stress`.
"""
struct SeaIceBottomStress{O, B}
    ocean :: O
    basal :: B
end

Base.summary(::SeaIceBottomStress) = "SeaIceBottomStress"

function Base.show(io::IO, τ::SeaIceBottomStress)
    print(io, summary(τ), '\n')
    print(io, "├── ocean: ", summary(τ.ocean), '\n')
    print(io, "└── basal: ", summary(τ.basal))
end

Adapt.adapt_structure(to, τ::SeaIceBottomStress) =
    SeaIceBottomStress(Adapt.adapt(to, τ.ocean), Adapt.adapt(to, τ.basal))

@inline InterfaceComputations.ice_ocean_momentum_stress(τ::SeaIceBottomStress) = τ.ocean

SeaIceDynamics.materialize_stress(τ::SeaIceBottomStress, grid) =
    SeaIceBottomStress(SeaIceDynamics.materialize_stress(τ.ocean, grid),
                       materialize_basal_stress(τ.basal, grid))

SeaIceDynamics.update_external_stress!(τ::SeaIceBottomStress, grid) =
    SeaIceDynamics.update_external_stress!(τ.ocean, grid)

# Free drift is a marginal-ice limit, where grounding is suppressed by the concentration hardening,
# so it balances against the ocean drag alone.
SeaIceDynamics.materialize_free_drift(::StressBalanceFreeDrift, top, bottom::SeaIceBottomStress) =
    StressBalanceFreeDrift(top, bottom.ocean)

@inline SeaIceDynamics.explicit_τx(i, j, k, grid, τ::SeaIceBottomStress, clock, fields) =
    SeaIceDynamics.explicit_τx(i, j, k, grid, τ.ocean, clock, fields)

@inline SeaIceDynamics.explicit_τy(i, j, k, grid, τ::SeaIceBottomStress, clock, fields) =
    SeaIceDynamics.explicit_τy(i, j, k, grid, τ.ocean, clock, fields)

@inline SeaIceDynamics.implicit_τx_coefficient(i, j, k, grid, τ::SeaIceBottomStress, clock, fields) =
    SeaIceDynamics.implicit_τx_coefficient(i, j, k, grid, τ.ocean, clock, fields) +
    basal_τx_coefficient(i, j, k, grid, τ.basal, fields)

@inline SeaIceDynamics.implicit_τy_coefficient(i, j, k, grid, τ::SeaIceBottomStress, clock, fields) =
    SeaIceDynamics.implicit_τy_coefficient(i, j, k, grid, τ.ocean, clock, fields) +
    basal_τy_coefficient(i, j, k, grid, τ.basal, fields)
