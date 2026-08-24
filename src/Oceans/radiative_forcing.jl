using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans.Grids: inactive_cell
using Oceananigans.Operators: ∂zᶜᶜᶜ

using ..NumericalEarth: stateindex

"""
    ChlorophyllOptics(FT = Oceananigans.defaults.FloatType;
                      clear_water_attenuation = 0.0232,
                      chlorophyll_scaling = 0.074,
                      chlorophyll_exponent = 0.674)

Parameters relating the blue-green absorption coefficient of sea water to its chlorophyll
concentration, following [Manizza et al. (2005)](@cite manizza2005bio). Defaults are the values
given there. See [`absorption_coefficient`](@ref).
"""
struct ChlorophyllOptics{FT}
    clear_water_attenuation :: FT
    chlorophyll_scaling :: FT
    chlorophyll_exponent :: FT
end

function ChlorophyllOptics(FT = Oceananigans.defaults.FloatType;
                           clear_water_attenuation = 0.0232,
                           chlorophyll_scaling = 0.074,
                           chlorophyll_exponent = 0.674)
    return ChlorophyllOptics(convert(FT, clear_water_attenuation),
                             convert(FT, chlorophyll_scaling),
                             convert(FT, chlorophyll_exponent))
end

Adapt.adapt_structure(to, optics::ChlorophyllOptics) =
    ChlorophyllOptics(adapt(to, optics.clear_water_attenuation),
                      adapt(to, optics.chlorophyll_scaling),
                      adapt(to, optics.chlorophyll_exponent))

"""
    absorption_coefficient(optics::ChlorophyllOptics, chlorophyll)

Return the blue-green absorption coefficient in m⁻¹ implied by a `chlorophyll` concentration in
mg m⁻³,

```math
κ = a + b \\, C^{n}
```

with ``a`` the clear-water attenuation and ``b``, ``n`` the chlorophyll scaling and exponent of
`optics`. Clear subtropical water, ``C ≈ 0.05``, decays over about 30 m; a subpolar bloom,
``C ≈ 1``, over about 10 m. [`equivalent_chlorophyll`](@ref) inverts the relation.

```jldoctest
using NumericalEarth

round(1 / absorption_coefficient(ChlorophyllOptics(), 1.0), digits=1)

# output

10.3
```
"""
@inline function absorption_coefficient(optics::ChlorophyllOptics, chlorophyll)
    κw = optics.clear_water_attenuation
    Cs = optics.chlorophyll_scaling
    Ce = optics.chlorophyll_exponent
    return κw + Cs * max(0, chlorophyll)^Ce
end

"""
    equivalent_chlorophyll(optics::ChlorophyllOptics, absorption_coefficient)

Return the chlorophyll concentration in mg m⁻³ whose blue-green absorption coefficient is
`absorption_coefficient`, the inverse of [`absorption_coefficient`](@ref). Use it to express a
Jerlov water type, whose optics are quoted as a decay scale, as a chlorophyll.

```jldoctest
using NumericalEarth

round(equivalent_chlorophyll(ChlorophyllOptics(), 1 / 23), digits=3)

# output

0.147
```
"""
@inline function equivalent_chlorophyll(optics::ChlorophyllOptics, absorption_coefficient)
    κ  = absorption_coefficient
    κw = optics.clear_water_attenuation
    Cs = optics.chlorophyll_scaling
    Ce = 1 / optics.chlorophyll_exponent 
    return ((κ - κw) / Cs)^Ce 
end

struct TwoColorRadiation{E, K, A, O, C, J}
    first_color_fraction :: E
    first_absorption_coefficient :: K
    second_absorption_coefficient :: A
    chlorophyll_optics :: O
    chlorophyll :: C
    surface_flux :: J
end

Adapt.adapt_structure(to, R::TwoColorRadiation) =
    TwoColorRadiation(adapt(to, R.first_color_fraction),
                      adapt(to, R.first_absorption_coefficient),
                      adapt(to, R.second_absorption_coefficient),
                      adapt(to, R.chlorophyll_optics),
                      adapt(to, R.chlorophyll),
                      adapt(to, R.surface_flux))

"""
    TwoColorRadiation(grid; first_color_fraction = 0.58,
                            first_decay_scale = 0.35,
                            chlorophyll_optics = ChlorophyllOptics(eltype(grid)),
                            chlorophyll = equivalent_chlorophyll(chlorophyll_optics, 1 / 23))

Return `TwoColorRadiation` that computes the radiative flux divergence associated with
a two-color radiation flux that decays according to Beer's law,

```math
I(z) = ϵ₁ I₀ \\exp(κ₁ z) + (1 - ϵ₁) I₀ \\exp(κ₂ z)
```

where ``I₀`` is the surface flux, ``ϵ₁`` is the `first_color_fraction`, and ``κ₁``, ``κ₂`` are the
absorption coefficients of the two colors. The first color is red and is absorbed within
`first_decay_scale` of the surface; the second is blue-green and its absorption coefficient ``κ₂``
follows the water's `chlorophyll` through [`absorption_coefficient`](@ref).

`chlorophyll` is anything `stateindex` resolves: a number for globally uniform optics, a surface
`Field` or array for a fixed spatial pattern, a function of `(λ, φ, z, t)`, or a `FieldTimeSeries`,
which is interpolated in time and so carries a seasonal cycle. `Cyclical` time indexing turns a
twelve-month climatology, such as
[`SeaWiFSMonthly`](@ref NumericalEarth.DataWrangling.SeaWiFS.SeaWiFSMonthly), into optics
that repeat every year.

The defaults are the red band of Manizza et al. (2005) and the chlorophyll whose decay scale is
23 m, which is the Jerlov Type I value of
[Paulson and Simpson (1977)](@cite paulson1977irradiance) for the clearest open-ocean water.

Uniform `chlorophyll` gives a scalar ``κ₂``; anything else gives a surface field that
`compute_absorption_coefficient!` refreshes once per column each step.

```jldoctest
using NumericalEarth
using Oceananigans

grid = LatitudeLongitudeGrid(size=(4, 4, 4), longitude=(0, 360), latitude=(-60, 60), z=(-100, 0))

TwoColorRadiation(grid)

# output

TwoColorRadiation with red fraction 0.58 decaying over 0.35 m
└── blue-green chlorophyll: 0.147 mg m⁻³, decaying over 23.0 m
```
"""
function TwoColorRadiation(grid;
                           first_color_fraction = 0.58,
                           first_decay_scale = 0.35,
                           chlorophyll_optics = ChlorophyllOptics(eltype(grid)),
                           chlorophyll = equivalent_chlorophyll(chlorophyll_optics, 1 / 23))
    FT = eltype(grid)
    surface_flux = Field{Center, Center, Nothing}(grid)
    chlorophyll = chlorophyll isa Number ? convert(FT, chlorophyll) : chlorophyll

    second_absorption_coefficient = chlorophyll isa Number ?
        absorption_coefficient(chlorophyll_optics, chlorophyll) :
        Field{Center, Center, Nothing}(grid)

    return TwoColorRadiation(convert(FT, first_color_fraction),
                             convert(FT, 1 / first_decay_scale),
                             second_absorption_coefficient,
                             chlorophyll_optics,
                             chlorophyll,
                             surface_flux)
end

function Base.show(io::IO, R::TwoColorRadiation)
    blue_green = R.chlorophyll isa Number ?
        string(round(R.chlorophyll, digits=3), " mg m⁻³, decaying over ",
               round(1 / absorption_coefficient(R.chlorophyll_optics, R.chlorophyll), digits=1), " m") :
        summary(R.chlorophyll)

    print(io, "TwoColorRadiation with red fraction ", R.first_color_fraction,
              " decaying over ", round(1 / R.first_absorption_coefficient, digits=2), " m", '\n',
              "└── blue-green chlorophyll: ", blue_green)
end

const c = Center()
const f = Face()

# In zstar we might have positive z, so  `exp(κ * z)` is not correct
# Radiation that reaches the bottom is dumped on the last cell
@inline function beers_law_radiation(i, j, k, grid, J₀ , κ)
    Nz = size(grid, 3)
    z  = Oceananigans.Grids.znode(i, j, k,    grid, c, c, f)
    η  = Oceananigans.Grids.znode(i, j, Nz+1, grid, c, c, f)
    J  = J₀ * exp(κ * (z - η))
    return ifelse(inactive_cell(i, j, k - 1, grid), zero(J), J)
end

@inline blue_green_absorption_coefficient(κ₂::Number, i, j) = κ₂
@inline blue_green_absorption_coefficient(κ₂, i, j) = @inbounds κ₂[i, j, 1]

"""
$(TYPEDSIGNATURES)

Refresh the blue-green absorption coefficient of `radiation` from its chlorophyll at `time`, and
return `nothing`. Chlorophyll reaches the optics through a power law and, for a `FieldTimeSeries`,
an interpolation in time; both depend on the column alone, so evaluating them once per column here
keeps them out of the flux divergence, which is evaluated at every level. Uniform chlorophyll
carries its coefficient as a scalar and there is nothing to refresh.
"""
compute_absorption_coefficient!(radiation, time) = nothing

compute_absorption_coefficient!(R::TwoColorRadiation, time) =
    compute_absorption_coefficient!(R.second_absorption_coefficient, R.chlorophyll, R.chlorophyll_optics, time)

compute_absorption_coefficient!(κ₂::Number, chlorophyll, optics, time) = nothing

function compute_absorption_coefficient!(κ₂, chlorophyll, optics, time)
    grid = κ₂.grid
    launch!(architecture(grid), grid, :xy, _compute_absorption_coefficient!, κ₂, grid, chlorophyll, optics, time)
    return nothing
end

@inline surface_stateindex(a, i, j, grid, time) = stateindex(a, i, j, 1, grid, time, (Center, Center, Nothing))

# A `Nothing` vertical location leaves `_node` a two-tuple, which a function of `(λ, φ, z, t)` cannot
# destructure, so functions are resolved at the topmost center instead.
@inline surface_stateindex(a::Function, i, j, grid, time) =
    stateindex(a, i, j, size(grid, 3), grid, time, (Center, Center, Center))

@kernel function _compute_absorption_coefficient!(κ₂, grid, chlorophyll, optics, time)
    i, j = @index(Global, NTuple)
    C = surface_stateindex(chlorophyll, i, j, grid, time)
    @inbounds κ₂[i, j, 1] = absorption_coefficient(optics, C)
end

@inline function (R::TwoColorRadiation)(i, j, k, grid, clock, fields)
    J₀ = @inbounds R.surface_flux[i, j, 1]
    κ₁ = R.first_absorption_coefficient
    κ₂ = blue_green_absorption_coefficient(R.second_absorption_coefficient, i, j)
    ϵ₁ = R.first_color_fraction

    # Radiation flux divergences
    dJ₁dz = ∂zᶜᶜᶜ(i, j, k, grid, beers_law_radiation, J₀, κ₁)
    dJ₂dz = ∂zᶜᶜᶜ(i, j, k, grid, beers_law_radiation, J₀, κ₂)

    # Net radiation flux divergence
    return ϵ₁ * dJ₁dz + (1 - ϵ₁) * dJ₂dz
end

@inline shortwave_radiative_forcing(i, j, grid, Fᵀ, ℐₜˢʷ, ocean_properties) = ℐₜˢʷ

@inline function shortwave_radiative_forcing(i, j, grid, tcr::TwoColorRadiation, Iˢʷ, ocean_properties)
    ρᵒᶜ = ocean_properties.reference_density
    cᵒᶜ = ocean_properties.heat_capacity
    J₀ = tcr.surface_flux
    @inbounds J₀[i, j,  1] = - Iˢʷ / (ρᵒᶜ * cᵒᶜ)
    return zero(Iˢʷ)
end

get_radiative_forcing(something) = nothing
get_radiative_forcing(tcr::TwoColorRadiation) = tcr

function get_radiative_forcing(FT::MultipleForcings)
    for forcing in FT.forcings
        forcing isa TwoColorRadiation && return forcing
    end
    return nothing
end

get_radiative_forcing(sim::Simulation) = get_radiative_forcing(sim.model)
get_radiative_forcing(model::HydrostaticFreeSurfaceModel) = get_radiative_forcing(model.forcing.T)
get_radiative_forcing(model::NonhydrostaticModel) = get_radiative_forcing(model.forcing.T)
