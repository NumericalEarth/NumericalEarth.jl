#####
##### Earth's tidal astronomy
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
Properties of Earth's tidal constituents: the equilibrium `amplitude` [m], which is the
Cartwright–Tayler amplitude times the Love number factor ``1 + k - h`` of an elastic Earth; the
multiples of the angles in `astronomical_angles` that sum to the equilibrium argument ``V``; and the
`nodal` coefficients, with which the 18.6-year cycle of the lunar node longitude ``N`` modulates
amplitude as ``f = f₀ + f₁ \\cos N`` and phase as ``u = u₁ \\sin N`` [°].

The first multiple is the constituent's species: 2 semidiurnal, 1 diurnal, 0 long-period.

References: Schureman (1958), [Kowalik and Luick (2019)](@cite kowalik2019modern).
"""
const earth_tidal_constituents = (
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

"""
$(TYPEDSIGNATURES)

`TidalHarmonics` for Earth's astronomical tide of `constituents` at `reference_date`: their
frequencies ``ω`` and the phases ``V + u`` that the equilibrium arguments and the nodal corrections
reach at that date. Model time is seconds from `reference_date`, so a constituent of amplitude ``A``
and phase lag ``G`` contributes ``f A \\cos(ω t + V + u - G)``.

The available constituents are the eight primary astronomical ones, `:M2`, `:S2`, `:N2`, `:K2`,
`:K1`, `:O1`, `:P1` and `:Q1`, and the two dominant long-period ones, `:Mf` and `:Mm`. A shelf
generates the compound and overtides internally, so those are not offered.

`ramp_time` eases the tide in from rest as ``\\tanh(t / T)``, which a basin started impulsively needs
to avoid ringing its gravest gravity mode.

```jldoctest
using NumericalEarth
using Dates

earth_tidal_harmonics(DateTime(2019, 4, 1); constituents = (:M2, :K1))

# output
TidalHarmonics with M2, K1
```
"""
function earth_tidal_harmonics(reference_date::DateTime;
                               constituents = keys(earth_tidal_constituents),
                               ramp_time = 0)

    names = Tuple(Symbol(constituent) for constituent in constituents)
    properties = Tuple(earth_tidal_constituents[name] for name in names)

    centuries = Dates.value(reference_date - astronomical_epoch) / (86_400_000 * 36_525)
    angles = astronomical_angles .+ astronomical_rates .* centuries
    node = lunar_node_angle + lunar_node_rate * centuries

    return TidalHarmonics(constituents = names,
                          frequencies = Tuple(deg2rad(sum(p.argument .* astronomical_rates)) / (36_525 * 86_400) for p in properties),
                          phases = Tuple(deg2rad(mod(sum(p.argument .* angles) + p.nodal.u₁ * sind(node), 360)) for p in properties),
                          nodal_factors = Tuple(p.nodal.f₀ + p.nodal.f₁ * cosd(node) for p in properties),
                          equilibrium_amplitudes = Tuple(p.amplitude for p in properties),
                          species = Tuple(first(p.argument) for p in properties),
                          ramp_time = ramp_time)
end
