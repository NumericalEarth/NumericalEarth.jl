#####
##### Scalars at cell centers
#####

# Regrid the interior of `source` onto the interior of `destination`: no correction, no halo fill
function regrid_interior!(destination::LatitudeLongitudeField, regridder, source::YinYangField, operator)
    tables = tables_to_latitude_longitude(regridder)
    for r in 1:2
        apply_regridding_table!(destination.data, destination.grid, getregion(source.data, r), tables[r];
                                accumulate = r == 2)
    end
    return nothing
end

function regrid_interior!(destination::YinYangField, regridder, source::LatitudeLongitudeField, operator)
    tables = tables_to_yin_yang(regridder, operator)
    for r in 1:2
        apply_regridding_table!(getregion(destination.data, r), getregion(destination.grid, r), source.data, tables[r])
    end
    return nothing
end

# Fill the halos of a scalar, then clamp them to the range of its interior. Oceananigans fills the halos of a
# Yin-Yang panel by cubic interpolation from the other panel, which is not convex: a 0/1 mask overshoots to
# [-0.13, 1.13] there. Across the poles of a latitude-longitude grid the halos hold `2 c̄ - c`, which leaves the
# range of the field as well. Physics that reads halo columns must not see a fraction outside [0, 1].
#
# The clamp holds until the next ordinary `fill_halo_regions!` of the field, which overshoots again. A kernel
# that weights with a fraction in halo cells therefore clamps it where it reads it: the net fluxes of a Breeze
# atmosphere do (`bounded_fraction` in `NumericalEarthBreezeExt`), and the radiation mosaic and the blended
# skin temperature read the interior only.
function fill_bounded_halos!(field, regridder)
    fill_halo_regions!(field)
    regions = eachindex(region_geometries(regridder, field))
    lowest = minimum(r -> minimum(regional_interior(field, r)), regions)
    highest = maximum(r -> maximum(regional_interior(field, r)), regions)
    for r in regions
        data = parent(getregion(field, r))
        data .= clamp.(data, lowest, highest)
    end
    return nothing
end

"""
$(SIGNATURES)

Interpolate `source`, state or boundary data on the latitude-longitude grid of `regridder`, bilinearly
onto its Yin-Yang grid, and fill the halos of `destination`. This is the operator for intensive
quantities, such as sea surface temperature, and for masks and fractions sampled at points. It conserves
nothing: fluxes take `transfer_flux!`.

`source` and `destination` are fields at (Center, Center) with one vertical level, or pairs `(u, v)` of
horizontal vector components: eastward and northward at (Center, Center) on the latitude-longitude grid,
and in each panel's basis on the Yin-Yang grid, either both at (Center, Center) or on the C-grid. Vectors
are interpolated as Cartesian 3-vectors, as in `transfer_flux!`.

Bounds
======

- Interior: every value is a convex combination of source values, so it lies within the range of the
  source. A fraction in `[0, 1]` stays in `[0, 1]`. Destination nodes that coincide with source nodes
  take the source values bit for bit.
- Halos of a scalar: filled by Oceananigans' cubic interpolation between panels, which overshoots, and
  then clamped to the range of the interior, so the same bounds hold there until the halos are filled
  again: a later `fill_halo_regions!(destination)` overshoots. Code that reads a fraction in halo cells
  must clamp it there.
- Vector components are not bounded.
"""
function interpolate_to_yin_yang!(destination::YinYangField, regridder::YinYangRegridder, source::LatitudeLongitudeField)
    validate_centered(regridder, destination)
    validate_centered(regridder, source)
    regrid_interior!(destination, regridder, source, BilinearInterpolation())
    fill_bounded_halos!(destination, regridder)
    return destination
end

"""
$(SIGNATURES)

Transfer the flux `source` onto `destination` with the area-weighted averages of `regridder`, and fill
the halos of `destination`. One of the two is on the Yin-Yang grid of `regridder` and the other on its
latitude-longitude grid. Both are fields at (Center, Center) with one vertical level, or both are pairs
`(u, v)` of horizontal vector components. State is not a flux: it takes `interpolate_to_yin_yang!`, or
Oceananigans' `interpolate!` from the Yin-Yang grid.

- Latitude-longitude → Yin-Yang: each cell of both panels takes the average of the source over the cell.
- Yin-Yang → latitude-longitude: each cell takes the average of the source over the part of the cell
  that each panel owns.

The averages are piecewise constant: first-order accurate.

Vectors
=======

`(u, v)` on the latitude-longitude grid are the eastward and northward components at (Center, Center).
On the Yin-Yang grid they are the components in each panel's basis, either both at (Center, Center) or on
the C-grid, at (Face, Center) and (Center, Face). The components are converted to a Cartesian 3-vector
at cell centers, each Cartesian component is averaged as a scalar, and the result is projected onto the
destination basis. C-grid sources are averaged to cell centers as a pair first, so their halos must be
filled; C-grid destinations are interpolated from the centers and then filled with
`fill_halo_regions!((u, v))`. The halos of a (Center, Center) pair on the Yin-Yang grid are not filled.

TODO: Oceananigans fills Yin-Yang halos of the C-grid pair only; a (Center, Center) pair needs
`fill_halo_regions!((u, v))` for both components at cell centers upstream.

Correction
==========

The averages are row-normalized and only approximately conservative. `correction` closes the global
integral of the result on that of `source`, in the ownership quadrature of `surface_integral`. It closes
that budget only: the transfer is not locally conservative with or without it, and the flux through the
seam between the panels is not conserved.

- `nothing`: no correction. The integral differs by the conservation residual of the averages.
- `GlobalRescaling()`: one multiplicative factor, for finite, non-negative scalars. A source with negative,
  NaN or infinite values is an error, raised before `destination` is written.
- `UniformRedistribution()`: a uniform additive shift, for scalars of either sign and for vectors, whose
  net force and net torque are both closed. For a C-grid pair they are measured and closed at the faces,
  after the conversion from the centers; see `surface_force_and_torque`.

Bounds
======

- Interior, `correction = nothing`: convex averages, within the range of the source.
- Interior, `GlobalRescaling()`: the sign is kept; the upper bound is scaled by the correction factor.
- Interior, `UniformRedistribution()`: no bounds; a non-negative flux can become slightly negative.
- Halos of a scalar: clamped to the range of the interior, whatever the correction, until the halos are
  filled again.
- Vector components are not bounded.
"""
function transfer_flux!(destination::Field, regridder::YinYangRegridder, source::Field; correction = nothing)
    validate_centered(regridder, destination)
    validate_centered(regridder, source)
    validate_correctable(regridder, source, correction)
    regrid_interior!(destination, regridder, source, AreaWeightedAverage())
    correct_integral!(destination, regridder, source, correction)
    fill_bounded_halos!(destination, regridder)
    return destination
end
