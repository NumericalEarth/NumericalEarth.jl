#####
##### Tidal astronomy
#####

const astronomical_epoch = DateTime(1900, 1, 1)

# Mean solar angle and mean longitudes of the Moon, the Sun and the lunar perigee at
# `astronomical_epoch` [°], followed by a right angle, and their rates [° per Julian century].
const astronomical_angles = (0, 277.0248, 280.1895, 334.3853, 90)
const astronomical_rates = (360 * 36525, 481267.8906, 36000.7689, 4069.0340, 0)

# Longitude of the Moon's ascending node [°] and its rate [° per Julian century]
const lunar_node_angle = 259.1568
const lunar_node_rate = -1934.1420

"""
Properties of the tidal constituents: the equilibrium `amplitude` [m], which is the
Cartwright–Tayler amplitude times the Love number factor ``1 + k - h`` of an elastic Earth; the
multiples of the angles in `astronomical_angles` that sum to the equilibrium argument ``V``; and the
`nodal` coefficients, with which the 18.6-year cycle of the lunar node longitude ``N`` modulates
amplitude as ``f = f₀ + f₁ \\cos N`` and phase as ``u = u₁ \\sin N`` [°].

The first multiple is the constituent's species: 2 semidiurnal, 1 diurnal, 0 long-period.

References: Schureman (1958), [Kowalik and Luick (2019)](@cite kowalik2019modern).
"""
const tidal_constituents = (
    M2 = (amplitude = 0.242334 * 0.693, argument = ( 2, -2,  2,  0,  0), nodal = (f₀ = 1.000, f₁ = -0.037, u₁ =  -2.1)),
    S2 = (amplitude = 0.112743 * 0.693, argument = ( 2,  0,  0,  0,  0), nodal = (f₀ = 1.000, f₁ =  0.000, u₁ =   0.0)),
    N2 = (amplitude = 0.046397 * 0.693, argument = ( 2, -3,  2,  1,  0), nodal = (f₀ = 1.000, f₁ = -0.037, u₁ =  -2.1)),
    K2 = (amplitude = 0.030684 * 0.693, argument = ( 2,  0,  2,  0,  0), nodal = (f₀ = 1.024, f₁ =  0.286, u₁ = -17.7)),
    K1 = (amplitude = 0.141565 * 0.736, argument = ( 1,  0,  1,  0,  1), nodal = (f₀ = 1.006, f₁ =  0.115, u₁ =  -8.9)),
    O1 = (amplitude = 0.100661 * 0.695, argument = ( 1, -2,  1,  0, -1), nodal = (f₀ = 1.009, f₁ =  0.187, u₁ =  10.8)),
    P1 = (amplitude = 0.046848 * 0.706, argument = ( 1,  0, -1,  0, -1), nodal = (f₀ = 1.000, f₁ =  0.000, u₁ =   0.0)),
    Q1 = (amplitude = 0.019273 * 0.695, argument = ( 1, -3,  1,  1, -1), nodal = (f₀ = 1.009, f₁ =  0.187, u₁ =  10.8)),
    Mf = (amplitude = 0.042041 * 0.693, argument = ( 0,  2,  0,  0,  0), nodal = (f₀ = 1.043, f₁ =  0.414, u₁ = -23.7)),
    Mm = (amplitude = 0.022191 * 0.693, argument = ( 0,  1,  0, -1,  0), nodal = (f₀ = 1.000, f₁ = -0.130, u₁ =   0.0)),
)

struct TidalHarmonics{N, FT, C}
    constituents :: C
    reference_date :: DateTime
    frequencies :: NTuple{N, FT}
    phases :: NTuple{N, FT}
    nodal_factors :: NTuple{N, FT}
    equilibrium_amplitudes :: NTuple{N, FT}
    species :: NTuple{N, Int}
    ramp_time :: FT
end

"""
$(TYPEDSIGNATURES)

The astronomical tide of `constituents` at `reference_date`: their frequencies ``ω`` and the phases
``V + u`` that the equilibrium arguments and the nodal corrections reach at that date. Model time is
seconds from `reference_date`, so a constituent of amplitude ``A`` and phase lag ``G`` contributes
``f A \\cos(ω t + V + u - G)``.

The available constituents are the eight primary astronomical ones, `:M2`, `:S2`, `:N2`, `:K2`,
`:K1`, `:O1`, `:P1` and `:Q1`, and the two dominant long-period ones, `:Mf` and `:Mm`. A shelf
generates the compound and overtides internally, so those are not offered.

`ramp_time` eases the tide in from rest as ``\\tanh(t / T)``, which a basin started impulsively needs
to avoid ringing its gravest gravity mode.

A single `TidalHarmonics` feeds both `tidal_forcing` and `tidal_boundary_conditions`, which keeps
the astronomical tide and the boundary tide in phase with each other.

```jldoctest
using NumericalEarth
using Dates

TidalHarmonics(DateTime(2019, 4, 1); constituents = (:M2, :K1))

# output
TidalHarmonics at 2019-04-01T00:00:00 with M2, K1
```
"""
function TidalHarmonics(reference_date::DateTime;
                        constituents = keys(tidal_constituents),
                        ramp_time = 0)

    names = Tuple(Symbol(constituent) for constituent in constituents)
    properties = Tuple(tidal_constituents[name] for name in names)

    centuries = Dates.value(reference_date - astronomical_epoch) / (86_400_000 * 36_525)
    angles = astronomical_angles .+ astronomical_rates .* centuries
    node = lunar_node_angle + lunar_node_rate * centuries

    FT = Oceananigans.defaults.FloatType
    frequencies = Tuple(FT(deg2rad(sum(p.argument .* astronomical_rates)) / (36_525 * 86_400)) for p in properties)
    phases = Tuple(FT(deg2rad(mod(sum(p.argument .* angles) + p.nodal.u₁ * sind(node), 360))) for p in properties)
    nodal_factors = Tuple(FT(p.nodal.f₀ + p.nodal.f₁ * cosd(node)) for p in properties)
    equilibrium_amplitudes = Tuple(FT(p.amplitude) for p in properties)
    species = Tuple(first(p.argument) for p in properties)

    return TidalHarmonics(names, reference_date, frequencies, phases, nodal_factors,
                          equilibrium_amplitudes, species, FT(ramp_time))
end

Adapt.adapt_structure(to, harmonics::TidalHarmonics) =
    TidalHarmonics(nothing,
                   harmonics.reference_date,
                   harmonics.frequencies,
                   harmonics.phases,
                   harmonics.nodal_factors,
                   harmonics.equilibrium_amplitudes,
                   harmonics.species,
                   harmonics.ramp_time)

Base.summary(harmonics::TidalHarmonics) =
    string("TidalHarmonics at ", harmonics.reference_date, " with ", join(harmonics.constituents, ", "))

Base.show(io::IO, harmonics::TidalHarmonics) = print(io, summary(harmonics))

@inline tidal_ramp(harmonics, t) = ifelse(harmonics.ramp_time > 0, tanh(t / harmonics.ramp_time), one(t))

#####
##### The equilibrium tide and its body force
#####

# Elevation [m] of the equilibrium tide at a cell center. The latitude structure of a constituent
# follows from its species: cos²φ semidiurnal, sin2φ diurnal, and ½ - 3/2 sin²φ long-period.
@inline function equilibrium_tide(i, j, k, grid, clock, harmonics)
    λ = deg2rad(λnode(i, j, k, grid, Center(), Center(), Center()))
    φ = deg2rad(φnode(i, j, k, grid, Center(), Center(), Center()))
    t = clock.time

    η = zero(grid)

    @inbounds for n in eachindex(harmonics.frequencies)
        s = harmonics.species[n]
        structure = ifelse(s == 2, cos(φ)^2, ifelse(s == 1, sin(2φ), (1 - 3 * sin(φ)^2) / 2))
        η += harmonics.nodal_factors[n] * harmonics.equilibrium_amplitudes[n] * structure *
             cos(harmonics.frequencies[n] * t + harmonics.phases[n] + s * λ)
    end

    return η
end

@inline x_tidal_forcing(i, j, k, grid, clock, fields, p) =
    p.gravitational_acceleration * ∂xᶠᶜᶜ(i, j, k, grid, equilibrium_tide, clock, p.harmonics) *
    tidal_ramp(p.harmonics, clock.time)

@inline y_tidal_forcing(i, j, k, grid, clock, fields, p) =
    p.gravitational_acceleration * ∂yᶜᶠᶜ(i, j, k, grid, equilibrium_tide, clock, p.harmonics) *
    tidal_ramp(p.harmonics, clock.time)

"""
$(TYPEDSIGNATURES)

Forcing `(u = Fᵘ, v = Fᵛ)` of the astronomical tide of `harmonics` on a spherical grid, which enters
the momentum equations as ``+g ∇η_{eq}`` for the equilibrium elevation ``η_{eq}``. The gradient is
taken over the same cells as the model's pressure gradient, so a resting ocean with
``η = η_{eq}`` stays at rest.

The forcing is barotropic: the astronomical tide pulls uniformly over the column.

```jldoctest
using NumericalEarth
using Dates

harmonics = TidalHarmonics(DateTime(2019, 4, 1))
keys(tidal_forcing(harmonics))

# output
(:u, :v)
```
"""
function tidal_forcing(harmonics::TidalHarmonics;
                       gravitational_acceleration = default_gravitational_acceleration)

    parameters = (; harmonics, gravitational_acceleration)
    u = Forcing(x_tidal_forcing; discrete_form = true, parameters)
    v = Forcing(y_tidal_forcing; discrete_form = true, parameters)

    return (; u, v)
end

#####
##### Boundary conditions from a tidal atlas
#####

"""
$(TYPEDSIGNATURES)

Complex harmonic constants of `constituent` over a window of `dataset` covering `grid`, as `Field`s
on the atlas's own staggered grid: sea surface height [m] at `(Center, Center)`, eastward transport
[m² s⁻¹] at `(Face, Center)` and northward transport [m² s⁻¹] at `(Center, Face)`. A constant pairs
with its Greenwich phase lag ``G`` as ``A e^{-i G}``.
"""
function tidal_atlas_constants end

# Greenwich phase lags enter as complex constants, so a constituent contributes
# real(f A e^{-iG} e^{i(ωt + V + u)}) to the transport and to the elevation alike.
@inline function tidal_transport_and_elevation(i, j, grid, clock, fields, p)
    t = clock.time
    harmonics = p.harmonics

    U = zero(grid)
    η = zero(grid)

    @inbounds for n in eachindex(harmonics.frequencies)
        rotation = harmonics.nodal_factors[n] * cis(harmonics.frequencies[n] * t + harmonics.phases[n])
        U += real(p.transport[i, n] * rotation)
        η += real(p.elevation[i, n] * rotation)
    end

    ramp = tidal_ramp(harmonics, t)

    return (ramp * U, ramp * η)
end

boundary_constants(constants, nodes) =
    [interpolate((mod(λ, 360), φ), constant) for (λ, φ) in nodes, constant in constants]

function flather_condition(harmonics, transport, elevation, architecture, transport_nodes, elevation_nodes)
    parameters = (; harmonics,
                  transport = on_architecture(architecture, boundary_constants(transport, transport_nodes)),
                  elevation = on_architecture(architecture, boundary_constants(elevation, elevation_nodes)))

    return GravityWaveRadiationBoundaryCondition(tidal_transport_and_elevation; discrete_form = true, parameters)
end

"""
$(TYPEDSIGNATURES)

Boundary conditions `(U, V, η)` that drive the barotropic tide of `harmonics` through every lateral
boundary of `grid`, with the tidal transports and elevations of `dataset`: Flather conditions
[Flather (1976)](@cite flather1976tidal) on the transports `U` and `V`, which carry the tide in, and
radiation conditions [Chapman (1985)](@cite chapman1985numerical) on the free surface `η`.

The atlas is sampled at the boundary nodes themselves, the transports where the model holds `U` and
`V` and the elevations at the adjacent cell centers, which the Flather condition compares with its
own.

The transports enter unconverted, so the tidal mass flux is preserved where the atlas and the model
disagree about depth. Keyword arguments reach `dataset`, such as the `dir` it reads.
"""
function tidal_boundary_conditions(grid, harmonics::TidalHarmonics, dataset; kw...)

    constants = map(constituent -> tidal_atlas_constants(dataset, grid, constituent; kw...), harmonics.constituents)
    elevation = map(constant -> constant.sea_surface_height, constants)
    eastward_transport = map(constant -> constant.eastward_transport, constants)
    northward_transport = map(constant -> constant.northward_transport, constants)

    arch = architecture(grid)
    λᶜ, λᶠ = λnodes(grid, Center()), λnodes(grid, Face())
    φᶜ, φᶠ = φnodes(grid, Center()), φnodes(grid, Face())

    west = flather_condition(harmonics, eastward_transport, elevation, arch,
                             [(first(λᶠ), φ) for φ in φᶜ], [(first(λᶜ), φ) for φ in φᶜ])

    east = flather_condition(harmonics, eastward_transport, elevation, arch,
                             [(last(λᶠ), φ) for φ in φᶜ], [(last(λᶜ), φ) for φ in φᶜ])

    south = flather_condition(harmonics, northward_transport, elevation, arch,
                              [(λ, first(φᶠ)) for λ in λᶜ], [(λ, first(φᶜ)) for λ in λᶜ])

    north = flather_condition(harmonics, northward_transport, elevation, arch,
                              [(λ, last(φᶠ)) for λ in λᶜ], [(λ, last(φᶜ)) for λ in λᶜ])

    U = FieldBoundaryConditions(grid, (Face(), Center(), nothing); west, east)
    V = FieldBoundaryConditions(grid, (Center(), Face(), nothing); south, north)

    radiation = SurfaceWaveRadiationBoundaryCondition()
    η = FieldBoundaryConditions(grid, (Center(), Center(), Face());
                                west = radiation, east = radiation, south = radiation, north = radiation)

    return (; U, V, η)
end
