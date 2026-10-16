#####
##### Friction velocity from ocean velocities under the ice
#####

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES

"""
$(TYPEDEF)

A friction velocity formulation that computes the friction velocity from the
ocean velocity averaged over the boundary layer beneath the ice shelf,

```math
u_*^2 = C_d \\, (u^2 + v^2 + u_\\mathrm{tidal}^2) ,
```

where ``C_d`` is a quadratic drag coefficient (default ``C_d = 0.0015``) and ``u_\\mathrm{tidal}``
is an RMS tidal velocity that keeps the friction velocity (and thus the melt
rate) from vanishing when the resolved flow is at rest (ISOMIP+ uses
``u_\\mathrm{tidal} = 0.01`` m/s; default 0). This is the ice-shelf analog of
`MomentumBasedFrictionVelocity`, which is built from ice-ocean stress fields
that do not exist for a static ice shelf.

$(TYPEDFIELDS)
"""
struct VelocityBasedFrictionVelocity{FT}
    "quadratic drag coefficient ``C_d``"
    drag_coefficient :: FT
    "RMS tidal velocity, which keeps ``u_*`` from vanishing at rest"
    tidal_velocity :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a [`VelocityBasedFrictionVelocity`](@ref).
"""
function VelocityBasedFrictionVelocity(FT::DataType = Oceananigans.defaults.FloatType;
                                       drag_coefficient = 0.0015,
                                       tidal_velocity = 0)
    return VelocityBasedFrictionVelocity(convert(FT, drag_coefficient),
                                         convert(FT, tidal_velocity))
end

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2
@inline U²ᶜᶜᶜ(i, j, k, grid, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², v)

"""
$(TYPEDSIGNATURES)

Return the friction velocity at the ice-shelf base for column `(i, j)`.
A `Number` is returned as-is (constant-``u_*`` formulation);
`VelocityBasedFrictionVelocity` computes
``u_* = \\sqrt{C_d (u² + v² + u_tidal²)}`` from `U²` averaged over the top `H_TBL` meters of the wet column.
"""
@inline ice_shelf_friction_velocity(u★::Number, i, j, kd, grid, U², H_TBL) = u★

@inline function ice_shelf_friction_velocity(fv::VelocityBasedFrictionVelocity, i, j, kd, grid, U², H_TBL)
    Ū² = surface_layer_average(i, j, kd, grid, Center(), Center(), U², H_TBL)
    return sqrt(fv.drag_coefficient * (Ū² + fv.tidal_velocity^2))
end

Base.summary(::VelocityBasedFrictionVelocity) = "VelocityBasedFrictionVelocity"

function Base.show(io::IO, fv::VelocityBasedFrictionVelocity)
    print(io, "VelocityBasedFrictionVelocity(drag_coefficient = ", fv.drag_coefficient,
              ", tidal_velocity = ", fv.tidal_velocity, ")")
end

#####
##### The ice shelf-ocean interface
#####

"""
$(TYPEDEF)

Container for the melt-rate computation at the base of a static ice shelf,
following the sea ice-ocean interface pattern (`SeaIceOceanInterface`).

$(TYPEDFIELDS)
"""
mutable struct IceShelfOceanInterface{K, A, J, F, L, T, S, U, P}
    "topmost wet cell of each ice-covered column, zero in open-ocean and dry columns"
    k_draft :: K
    "fraction of each column's immersed top that is ice, by which the heat, salt and melt fluxes are scaled"
    ice_concentration :: A
    "2D `interface_heat`, `temperature`, `salt` and `melt_rate` fluxes, positive out of the ocean"
    fluxes :: J
    "`ThreeEquationHeatFlux` or `IceBathHeatFlux`"
    flux_formulation :: F
    "`TEOS10Liquidus`, `LinearLiquidus`, or `nothing` for a `TEOS10Liquidus` consistent with the ocean"
    liquidus :: L
    "interface temperature"
    temperature :: T
    "interface salinity"
    salinity :: S
    "friction velocity at the ice base"
    friction_velocity :: U
    "reference density, heat capacity, latent heat, ice salinity and boundary-layer thickness"
    properties :: P
end

"""
$(TYPEDSIGNATURES)

Build an `IceShelfOceanInterface` on `grid`, which is expected to be an
`ImmersedBoundaryGrid` with an immersed top (e.g. `GridFittedBottomAndTop`).
The draft index map `k_draft` is computed from
the grid's `immersed_cell` at construction; columns whose topmost cell
`k = Nz` is wet (no ice above) get `k_draft = 0` and are excluded from the
melt computation.

`ice_concentration` is the fraction of the immersed top of each column that is ice,
between 0 and 1: a number (default 1), a function of `(x, y)`, an array or a field.
The heat, salt and melt fluxes are scaled by it, so an immersed top with
`ice_concentration = 0` is inert, like rock. It may be changed during a simulation
with `set!(interface.ice_concentration, ...)`.

The default `flux_formulation` is a `ThreeEquationHeatFlux` with
friction-velocity-based transfer coefficients (αₕ = 0.0095, αₛ = αₕ/35) and
``u_*`` computed from the sub-shelf ocean velocities with drag coefficient
``C_d = 0.0015``.

The default `liquidus = nothing` is a [`TEOS10Liquidus`](@ref) (see [`ice_shelf_liquidus`](@ref)),
with the reference density of a `TEOS10EquationOfState` and the buoyancy's gravitational
acceleration when the ocean provides them.

`boundary_layer_thickness` sets the fixed physical thickness over which the
ocean state feeding the melt-rate solver is averaged, default 30 m
(Mathiot et al., 2017).
"""
function IceShelfOceanInterface(grid;
                                flux_formulation = ThreeEquationHeatFlux(eltype(grid);
                                                                         friction_velocity = VelocityBasedFrictionVelocity(eltype(grid))),
                                liquidus = nothing,
                                ice_concentration = 1,
                                reference_density = 1020,
                                heat_capacity = 3991,
                                latent_heat = 334e3,
                                ice_salinity = 0,
                                boundary_layer_thickness = 30)

    FT = eltype(grid)

    k_draft = Field{Center, Center, Nothing}(grid, Int)
    launch!(architecture(grid), grid, :xy, _compute_draft_index!, k_draft, grid)

    F = Field{Center, Center, Nothing}
    ℵ = F(grid)
    set!(ℵ, ice_concentration)

    fluxes = (interface_heat = F(grid),
              temperature    = F(grid),
              salt           = F(grid),
              melt_rate      = F(grid))

    temperature = F(grid)
    salinity = F(grid)
    friction_velocity = F(grid)

    properties = (reference_density           = convert(FT, reference_density),
                  heat_capacity               = convert(FT, heat_capacity),
                  latent_heat                 = convert(FT, latent_heat),
                  ice_salinity                = convert(FT, ice_salinity),
                  boundary_layer_thickness    = convert(FT, boundary_layer_thickness))

    return IceShelfOceanInterface(k_draft, ℵ, fluxes, flux_formulation, liquidus,
                                  temperature, salinity, friction_velocity, properties)
end

# k_draft = 0 marks open-ocean and fully immersed columns, which have no ice base.
@kernel function _compute_draft_index!(k_draft, grid)
    i, j = @index(Global, NTuple)

    kd = 0
    for k in grid.Nz : -1 : 1
        if !immersed_cell(i, j, k, grid)
            kd = k
            break
        end
    end

    kd = ifelse(kd == grid.Nz, 0, kd)
    @inbounds k_draft[i, j, 1] = kd
end

Base.summary(::IceShelfOceanInterface) = "IceShelfOceanInterface"

function Base.show(io::IO, interface::IceShelfOceanInterface)
    print(io, summary(interface), '\n')
    print(io, "├── flux_formulation: ", summary(interface.flux_formulation), '\n')
    print(io, "├── liquidus: ", liquidus_summary(interface.liquidus), '\n')
    print(io, "└── properties: ", interface.properties)
end

liquidus_summary(liquidus) = summary(liquidus)
liquidus_summary(::Nothing) = "from the ocean's equation of state"
