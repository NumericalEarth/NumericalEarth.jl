# The two kinds of operator. They are kept apart in the interface: `interpolate_to_yin_yang!` interpolates
# state and boundary data, `transfer_flux!` transfers fluxes.
struct BilinearInterpolation end # point values, interpolated bilinearly in longitude and latitude
struct AreaWeightedAverage end   # averages over each destination cell, weighted by the cell intersections

"""
    cell_intersection_areas(destination_grid, source_grid)

Return `(destinations, sources, areas)`: the linear indices `i + (j - 1) Nx` of the pairs of cells of
two single-region grids that intersect, and the areas of their intersections.

The method lives in `NumericalEarthConservativeRegriddingExt` and requires `ConservativeRegridding`
to be loaded.
"""
function cell_intersection_areas end

struct YinYangRegridder{G, L, C, I, Y, E, S, M, N}
    yin_yang_grid :: G
    latitude_longitude_grid :: L
    to_latitude_longitude :: C       # area-weighted tables (Yin, Yang) → latitude-longitude, or `nothing`
    to_yin_yang :: C                 # area-weighted tables latitude-longitude → (Yin, Yang), or `nothing`
    interpolation :: I               # bilinear tables latitude-longitude → (Yin, Yang)
    yin_yang_geometry :: Y           # `RegionGeometry` of (Yin, Yang)
    latitude_longitude_geometry :: E # 1-tuple with the `RegionGeometry` of the latitude-longitude grid
    scratch :: S                     # Cartesian vector components on both grids
    rigid_shift_operators :: M       # 6 × 6 matrices of the force and torque of a rigid shift, per representation
    grid_signatures :: N             # the geometry the tables were built for
end

function Base.summary(regridder::YinYangRegridder)
    Nx, Ny, _ = size(regridder.latitude_longitude_grid)
    operators = isnothing(regridder.to_yin_yang) ? "state interpolation" : "state interpolation and flux transfer"
    return string("YinYangRegridder between a ", summary(regridder.yin_yang_grid), " and a ",
                  Nx, "×", Ny, " LatitudeLongitudeGrid (", operators, ")")
end

Base.show(io::IO, regridder::YinYangRegridder) = print(io, summary(regridder))

"""
$(SIGNATURES)

Return a regridder between `yin_yang_grid` and the global `latitude_longitude_grid`, for fields at
cell centers and for horizontal vectors. The operators are built on the CPU and moved to the
architecture of the grids. Fields passed to the regridder must be on these grids, or on grids with the
same coordinates and radius.

The regridder serves two functions, which are kept apart:

- `interpolate_to_yin_yang!(destination, regridder, source)` interpolates state and boundary data from
  the latitude-longitude grid onto the panels, bilinearly. It conserves nothing.
- `transfer_flux!(destination, regridder, source; correction)` transfers fluxes in either direction with
  area-weighted averages: over the cell intersections, and from the Yin-Yang grid over the part of each
  cell that its panel owns, so that every point of the sphere is counted once.

What the flux transfer does and does not conserve
=================================================

The area-weighted tables are normalized by the sum of their own weights. They are convex averages: they
preserve constants and bounds. They are *not* conservative operators: the global integral, which on the
Yin-Yang grid is the ownership quadrature `Σ w Az c` with `w = owned_weights(yin_yang_grid)`, changes by
a residual, because the owned fraction of a cell cut by the ownership boundary is not uniform over the
cell and because the cell intersections are computed between great-circle polygons, whose areas differ
from the cell areas.

`correction = GlobalRescaling()` or `UniformRedistribution()` closes that global, ownership-integrated
budget to rounding, and nothing else: the flux leaving a cell is not the flux entering the cells it
overlaps, so neither local conservation nor conservation of the flux through the seam between the panels
holds.

Keyword arguments
=================

- `flux_transfer`: build the area-weighted tables, which requires `ConservativeRegridding` to be
                   loaded. Default: `true`.
"""
function YinYangRegridder(yin_yang_grid::YinYangGrid, latitude_longitude_grid::LatitudeLongitudeGrid;
                          flux_transfer = true)

    arch = architecture(latitude_longitude_grid)
    FT = eltype(yin_yang_grid)
    validate_global_grid(latitude_longitude_grid)

    # The host-side geometry (cell areas, owned weights, unit vectors) and the tables are built from Float64 copies of both grids, whatever
    # the model's FT: the metrics of a Float32 grid differ from the sphere by ~4e-8 (job 19948 failed the 1e-9 area check at 1.4e-9 even
    # with Float64 sums of the Float32 areas). Only the transferred field data are in FT.
    yin_yang = float64_cpu_copy(yin_yang_grid)
    latitude_longitude = float64_cpu_copy(latitude_longitude_grid)
    panels = map(r -> getregion(yin_yang, r), (1, 2))

    owned = owned_weights(yin_yang)
    owned_fractions = map(r -> Float64.(interior(getregion(owned, r), :, :, 1)), (1, 2))

    yin_yang_geometry = map((panel, w) -> RegionGeometry(panel, (i, j) -> w[i, j]; face_vectors=true), panels, owned_fractions)
    latitude_longitude_geometry = tuple(RegionGeometry(latitude_longitude, (i, j) -> 1))
    equalize_total_area!(yin_yang_geometry, latitude_longitude_geometry)

    interpolation = map(panel -> bilinear_table(FT, panel, latitude_longitude), panels)

    if flux_transfer
        applicable(cell_intersection_areas, latitude_longitude, first(panels)) ||
            throw(ArgumentError("The flux transfer of a YinYangRegridder requires ConservativeRegridding: " *
                                "load it with `using ConservativeRegridding`, or pass `flux_transfer = false`."))

        to_latitude_longitude, to_yin_yang = area_weighted_tables(FT, panels, latitude_longitude, owned_fractions)
    else
        to_latitude_longitude, to_yin_yang = nothing, nothing
    end

    rigid_shift_operators = (yin_yang = rigid_shift_operator(yin_yang_geometry),
                             yin_yang_staggered = staggered_rigid_shift_operator(yin_yang_geometry),
                             latitude_longitude = rigid_shift_operator(latitude_longitude_geometry))

    # The signatures are those of the grids the fields live on (FT), not of the Float64 copies the geometry was built from
    grid_signatures = (yin_yang = grid_signature(on_architecture(CPU(), yin_yang_grid)),
                       latitude_longitude = grid_signature(on_architecture(CPU(), latitude_longitude_grid)),
                       yin_yang_id = objectid(yin_yang_grid),
                       latitude_longitude_id = objectid(latitude_longitude_grid))

    cartesian_fields(grid) = ntuple(d -> Field{Center, Center, Nothing}(grid), 3)
    scratch = (yin_yang = cartesian_fields(yin_yang_grid),
               latitude_longitude = cartesian_fields(latitude_longitude_grid))

    return YinYangRegridder(yin_yang_grid,
                            latitude_longitude_grid,
                            move_to_architecture(arch, to_latitude_longitude),
                            move_to_architecture(arch, to_yin_yang),
                            move_to_architecture(arch, interpolation),
                            move_to_architecture(arch, yin_yang_geometry),
                            move_to_architecture(arch, latitude_longitude_geometry),
                            scratch,
                            rigid_shift_operators,
                            grid_signatures)
end

# Float64 CPU copies of the grids of a regridder (see `YinYangRegridder`)
function float64_cpu_copy(grid::YinYangGrid)
    c = grid.connectivity
    yin = first(on_architecture(CPU(), grid))
    return YinYangGrid(CPU(), Float64; size = (c.core_size..., size(grid, 3)), halo = halo_size(yin),
                       z = Oceananigans.Grids.cpu_face_constructor_z(yin), stencil_width = c.stencil_width, overlap = c.overlap,
                       radius = Float64(yin.radius))
end

float64_cpu_copy(grid::LatitudeLongitudeGrid) = Oceananigans.Grids.with_number_type(Float64, on_architecture(CPU(), grid))

# The owned areas of the panels sum to the area of the sphere only to about 1e-13, the rounding error of the
# closed-form owned fractions. Scale them so that both grids measure the same sphere: otherwise a constant
# has different integrals on the two grids, and no correction can both close the integral and preserve
# the constant to rounding.
function equalize_total_area!(yin_yang_geometry, latitude_longitude_geometry)
    scaling = total_area(latitude_longitude_geometry) / total_area(yin_yang_geometry)
    abs(scaling - 1) < 1e-9 ||
        throw(ArgumentError("The owned areas of the Yin-Yang grid sum to $(1 / scaling) times the area of the " *
                            "latitude-longitude grid: the grids do not cover the same sphere."))
    for geometry in yin_yang_geometry
        geometry.area .*= scaling
        update_face_areas!(geometry)
    end
    return nothing
end

# What the tables depend on: a field on a grid with another signature would silently get the wrong weights
grid_signature(grid::LatitudeLongitudeGrid) =
    (size(grid)[1:2], Float64(grid.radius), Float64.(λnodes(grid, Face())), Float64.(φnodes(grid, Face())))

grid_signature(grid::YinYangGrid) =
    (size(grid)[1:2], Float64(getregion(grid, 1).radius), grid.connectivity.core_size, grid.connectivity.overlap)

function validate_global_grid(grid::LatitudeLongitudeGrid)
    φ = Array(φnodes(grid, Face()))   # on the host: first/last of a GPU array is scalar indexing
    global_grid = topology(grid, 1) === Periodic && first(φ) ≈ -90 && last(φ) ≈ 90
    global_grid || throw(ArgumentError("A YinYangRegridder requires a global LatitudeLongitudeGrid: periodic in " *
                                       "longitude and spanning latitudes -90 to 90."))
    return nothing
end

#####
##### Bilinear interpolation of a global latitude-longitude field onto one panel
#####

# Fractional positions within `tolerance` of a source node are snapped onto it, so that destination nodes
# that coincide with source nodes take the source values bit for bit.
snap_fraction(ξ; tolerance = 1e-9) = ξ < tolerance ? zero(ξ) : ξ > 1 - tolerance ? one(ξ) : ξ

# The cells `i⁻`, `i⁺` that bracket the longitude `λ` among the periodic cell centers `λᶜ`, and the fractional
# distance `ξ` from `i⁻`
function longitude_stencil(λ, λᶜ)
    N = length(λᶜ)
    λ = λᶜ[1] + mod(λ - λᶜ[1], 360)
    i⁻ = clamp(searchsortedlast(λᶜ, λ), 1, N)
    i⁺ = i⁻ == N ? 1 : i⁻ + 1
    λ⁺ = i⁻ == N ? λᶜ[1] + 360 : λᶜ[i⁺]
    ξ = snap_fraction((λ - λᶜ[i⁻]) / (λ⁺ - λᶜ[i⁻]))
    return i⁻, i⁺, ξ
end

"""
$(SIGNATURES)

Return the `RegriddingTable` that interpolates a field at the cell centers of the global `source_grid`
bilinearly onto the cell centers of `panel`, a `RotatedLatitudeLongitudeGrid`.

Poleward of the last row of cell centers the interpolation runs between that row and the pole, whose
value is the zonal mean of the row. A destination node on a pole therefore takes one value whatever
its longitude.

This replaces `interpolate!(panel_field, source_field)` for two reasons. On a region of a `YinYangGrid`
that call ends in a halo fill that a single region cannot do. And in Oceananigans 0.117.2 the halo
north of the last row of a latitude-longitude field holds `2 c̄ˢ - c` with `c̄ˢ` the zonal mean of the
*southernmost* row, so anything interpolated poleward of the northernmost cell centers is wrong: a field
equal to 290 at the north pole and 270 at the south pole interpolates to 270 at the north pole.

TODO: fix upstream in `Oceananigans.BoundaryConditions`. `Adapt.adapt_structure(to, ::PolarValue)` drops
`side`, and `fill_halo_regions!(::Field)` adapts the boundary conditions before `update_pole_value!`
reads `side` to choose the row to average, so the north pole is always averaged over row 1. Once fixed,
add `interpolate!(to::MultiRegionField, from)` upstream, which interpolates region by region and fills
the halos once.
"""
function bilinear_table(FT, panel, source_grid)
    Nx, Ny, _ = size(panel)
    Nλ, Nφ, _ = size(source_grid)
    c = Center()

    λᶜ = Float64.(λnodes(source_grid, c))
    φᶜ = Float64.(φnodes(source_grid, c))
    λᶠ = Float64.(λnodes(source_grid, Face()))
    zonal_mean_weights = [((i == Nλ ? λᶠ[1] + 360 : λᶠ[i + 1]) - λᶠ[i]) / 360 for i in 1:Nλ]

    destinations, sources, weights = Int[], Int[], Float64[]

    function add_entry!(n, i, j, weight)
        push!(destinations, n)
        push!(sources, i + (j - 1) * Nλ)
        push!(weights, weight)
        return nothing
    end

    for j in 1:Ny, i in 1:Nx
        n = i + (j - 1) * Nx
        λ = Float64(λnode(i, j, 1, panel, c, c, c))
        φ = clamp(Float64(φnode(i, j, 1, panel, c, c, c)), -90, 90)
        i⁻, i⁺, ξ = longitude_stencil(λ, λᶜ)

        function add_row!(row, weight)
            add_entry!(n, i⁻, row, weight * (1 - ξ))
            add_entry!(n, i⁺, row, weight * ξ)
            return nothing
        end

        function add_polar_cap!(row, pole, φ)
            η = snap_fraction((pole - φ) / (pole - φᶜ[row]))
            add_row!(row, η)
            for i′ in 1:Nλ
                add_entry!(n, i′, row, (1 - η) * zonal_mean_weights[i′])
            end
            return nothing
        end

        if φ ≥ φᶜ[Nφ]
            add_polar_cap!(Nφ, 90, φ)
        elseif φ ≤ φᶜ[1]
            add_polar_cap!(1, -90, φ)
        else
            j⁻ = searchsortedlast(φᶜ, φ)
            η = snap_fraction((φ - φᶜ[j⁻]) / (φᶜ[j⁻ + 1] - φᶜ[j⁻]))
            add_row!(j⁻, 1 - η)
            add_row!(j⁻ + 1, η)
        end
    end

    return RegriddingTable(FT, Nx * Ny, Nλ, destinations, sources, weights)
end

#####
##### Area-weighted operators for flux transfer
#####

"""
$(SIGNATURES)

Return `(to_latitude_longitude, to_yin_yang)`: the area-weighted tables of each panel in both directions.

With `A` the areas of the intersections of the panel cells `c` with the latitude-longitude cells `l`, and
`w` the owned fractions, the operators are

- Yin-Yang → latitude-longitude: `F_l = Σ_c w_c A_lc F_c / Σ_c w_c A_lc`, summed over both panels, and
- latitude-longitude → Yin-Yang: `F_c = Σ_l A_lc F_l / Σ_l A_lc`, for every cell of both panels.

Both are normalized by the sum of their own weights (row-normalized), which preserves constants in the
cells cut by the ownership boundary at the price of a conservation residual: these are approximate
averages, not conservative operators. Dividing by the cell area instead would conserve the global integral
but not preserve constants in the cut cells.

TODO: a supermesh that clips the panel cells by the ownership boundary, whose closed form is in
`Oceananigans.MultiRegion.owned_cell_fraction`, would conserve and preserve constants at once. Scaling
these weights by iterative proportional fitting does not: the exact solution has zeros where these weights
are positive, so the iteration stalls (residual 5e-4 after 2000 sweeps) and distorts the seam stencils.
"""
function area_weighted_tables(FT, panels, latitude_longitude, owned_fractions)
    Nλ, Nφ, _ = size(latitude_longitude)
    Nx, Ny, _ = size(first(panels))
    panel_cells = Nx * Ny

    # Entries of both panels, with the cells of Yang numbered after those of Yin
    rows, columns, areas = Int[], Int[], Float64[]
    entries = map((1, 2)) do r
        l, c, A = cell_intersection_areas(latitude_longitude, panels[r])
        first_entry = length(rows) + 1
        append!(rows, l)
        append!(columns, c .+ (r - 1) * panel_cells)
        append!(areas, A)
        return first_entry:length(rows)
    end

    owned = vcat(vec.(owned_fractions)...)
    owned_areas = owned[columns] .* areas

    owned_cover = zeros(Nλ * Nφ)
    cover = zeros(2 * panel_cells)
    for p in eachindex(rows)
        owned_cover[rows[p]] += owned_areas[p]
        cover[columns[p]] += areas[p]
    end

    all(>(0), owned_cover) && all(>(0), cover) ||
        throw(ArgumentError("The Yin-Yang panels and the latitude-longitude grid do not cover each other."))

    to_latitude_longitude = map((1, 2)) do r
        p = entries[r]
        RegriddingTable(FT, Nλ * Nφ, Nx, rows[p], columns[p] .- (r - 1) * panel_cells,
                        owned_areas[p] ./ owned_cover[rows[p]])
    end

    to_yin_yang = map((1, 2)) do r
        p = entries[r]
        RegriddingTable(FT, panel_cells, Nλ, columns[p] .- (r - 1) * panel_cells, rows[p],
                        areas[p] ./ cover[columns[p]])
    end

    return to_latitude_longitude, to_yin_yang
end

tables_to_yin_yang(regridder, ::BilinearInterpolation) = regridder.interpolation

function tables_to_yin_yang(regridder, ::AreaWeightedAverage)
    isnothing(regridder.to_yin_yang) &&
        throw(ArgumentError("This YinYangRegridder was built with `flux_transfer = false`."))
    return regridder.to_yin_yang
end

function tables_to_latitude_longitude(regridder)
    isnothing(regridder.to_latitude_longitude) &&
        throw(ArgumentError("This YinYangRegridder was built with `flux_transfer = false`."))
    return regridder.to_latitude_longitude
end

#####
##### Fields on the two grids
#####

const YinYangField = Field{<:Any, <:Any, <:Any, <:Any, <:YinYangGrid}
const LatitudeLongitudeField = Field{<:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid}

region_geometries(regridder, ::YinYangField) = regridder.yin_yang_geometry
region_geometries(regridder, ::LatitudeLongitudeField) = regridder.latitude_longitude_geometry

cartesian_scratch(regridder, ::YinYangField) = regridder.scratch.yin_yang
cartesian_scratch(regridder, ::LatitudeLongitudeField) = regridder.scratch.latitude_longitude

rigid_shift_operator(regridder, ::YinYangField) = regridder.rigid_shift_operators.yin_yang
rigid_shift_operator(regridder, ::LatitudeLongitudeField) = regridder.rigid_shift_operators.latitude_longitude

regridder_grid_id(regridder, ::YinYangField) = regridder.grid_signatures.yin_yang_id
regridder_grid_id(regridder, ::LatitudeLongitudeField) = regridder.grid_signatures.latitude_longitude_id

regridder_signature(regridder, ::YinYangField) = regridder.grid_signatures.yin_yang
regridder_signature(regridder, ::LatitudeLongitudeField) = regridder.grid_signatures.latitude_longitude

horizontal_location(field) = instantiated_location(field)[1:2]

# The tables hold the weights and indices of the regridder's grids. A field on another grid would be read
# and written out of bounds, or, on a grid of the same size with other coordinates or another radius, would
# silently get the wrong weights.
#
# The regridder's own grids are recognized by `objectid`, and any other grid is compared by its signature.
# `field.grid === grid` is not used: with Julia 1.13.2 that comparison of two grid structs gave different
# answers in the same call when the function had first been called with a Yin-Yang field, and wrong grids
# passed. The grid regression test guards this.
function validate_grid(regridder, field)
    size(field, 3) == 1 || throw(ArgumentError("A YinYangRegridder regrids fields with one vertical level."))

    if objectid(field.grid) != regridder_grid_id(regridder, field)
        if grid_signature(on_architecture(CPU(), field.grid)) != regridder_signature(regridder, field)
            throw(ArgumentError("The field's grid, $(summary(field.grid)), is not the grid this YinYangRegridder " *
                                "was built for: its size, coordinates or radius differ."))
        end
    end

    return nothing
end

function validate_centered(regridder, field)
    horizontal_location(field) == (Center(), Center()) ||
        throw(ArgumentError("A YinYangRegridder regrids scalar fields at (Center, Center); got a field at " *
                            "$(horizontal_location(field))."))
    return validate_grid(regridder, field)
end
