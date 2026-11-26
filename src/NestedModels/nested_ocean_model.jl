#####
##### nested_ocean_model: a regional ocean nested in a parent ocean
#####

using Oceananigans.Fields: CenterField, XFaceField, YFaceField, interior, interpolate!
using Oceananigans.BoundaryConditions: ObliqueRadiation, TracerReservoir
using Oceananigans.Grids: LatitudeLongitudeGrid, OrthogonalSphericalShellGrid, Open, TwoSided, constructor_arguments,
                          λnodes, φnodes
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Units: days, hours, kilometers
using Oceananigans.DistributedComputations: all_reduce, reconstruct_global_grid
using Oceananigans.ImmersedBoundaries: has_active_cells_map, has_active_z_columns, underlying_grid

using ..Bathymetry: regrid_bathymetry
using ..Oceans: hydrostatic_ocean_model, merge_boundary_conditions, estimate_maximum_Δt
using ..DataWrangling: BoundingBox, Metadatum, default_download_directory, default_horizontal_padding,
                       matching_bathymetry, sea_level_includes_inverse_barometer

default_velocity_scheme() = ObliqueRadiation(inflow_timescale = 9hours, outflow_timescale = 360days)
default_tracer_scheme() = TracerReservoir(inflow_length_scale = 20kilometers, outflow_length_scale = 0)

# Sponge and blend widths of 8 and 4 cells at 1/12°, held fixed in degrees across resolutions
const default_relaxation_degrees = 8 / 12
const default_bathymetry_blend_degrees = 4 / 12

# The number of cells of `grid` that span `degrees` along its finest horizontal direction.
function cells_spanning(grid, degrees)
    Δ = all_reduce(min, finest_spacing(underlying_grid(grid)), architecture(grid))
    return max(1, round(Int, degrees / Δ))
end

# In degrees of longitude or latitude, or of arc along a curvilinear grid's own axes
function finest_spacing(grid::LatitudeLongitudeGrid)
    cpu_grid = on_architecture(CPU(), grid)
    return min(minimum(diff(λnodes(cpu_grid, Face()))), minimum(diff(φnodes(cpu_grid, Face()))))
end

function finest_spacing(grid::OrthogonalSphericalShellGrid)
    Nx, Ny, _ = size(grid)
    Δx = view(on_architecture(CPU(), grid.Δxᶜᶜᵃ), 1:Nx, 1:Ny)
    Δy = view(on_architecture(CPU(), grid.Δyᶜᶜᵃ), 1:Nx, 1:Ny)
    return rad2deg(min(minimum(Δx), minimum(Δy)) / grid.radius)
end

const OceanParent = Union{PrescribedOcean, BoundaryPrescribedComponent}

function validate_nested_ocean_grid(grid)
    underlying_grid(grid) isa Union{LatitudeLongitudeGrid, OrthogonalSphericalShellGrid} ||
        throw(ArgumentError("a nested ocean requires a grid on the sphere; got $(summary(grid))."))
    return nothing
end

# Every rank takes part, so that on a distributed grid all of them throw together.
function validate_open_sides(grid, open_sides)
    Nopen = all_reduce(+, length(open_sides), architecture(grid))
    Nopen > 0 || throw(ArgumentError("$(summary(grid)) has no open sides to nest through."))
    return nothing
end

# A band must fit within the grid, or the rank of a distributed grid, along each open side.
function validate_band_width(grid, sides, width, name)
    fits = all(side -> width ≤ size(grid, boundary_dimension(side)), sides)
    Nmisfits = all_reduce(+, Int(!fits), architecture(grid))
    Nmisfits == 0 ||
        throw(ArgumentError("$name = $width is wider than the grid, or a rank of it, normal to an open side."))
    return nothing
end

# A copy of `grid` with `sides` `Open`; a side any rank holds opens for the whole domain
function with_open_sides(grid::LatitudeLongitudeGrid, sides)
    sides = global_sides(grid, sides)
    args, kwargs = constructor_arguments(reconstruct_global_grid(grid))
    TX, TY, TZ = kwargs[:topology]
    kwargs[:topology] = (open_topology(TX, :west ∈ sides, :east ∈ sides),
                         open_topology(TY, :south ∈ sides, :north ∈ sides), TZ)
    return LatitudeLongitudeGrid(architecture(grid), args[:number_type]; kwargs...)
end

function with_open_sides(grid::OrthogonalSphericalShellGrid, sides)
    TX, TY, TZ = topology(grid)
    TXᵒ = open_topology(TX, :west ∈ sides, :east ∈ sides)
    TYᵒ = open_topology(TY, :south ∈ sides, :north ∈ sides)
    properties = ntuple(n -> getfield(grid, n), fieldcount(typeof(grid)))
    return OrthogonalSphericalShellGrid{eltype(grid), TXᵒ, TYᵒ, TZ}(properties...)
end

function with_open_sides(grid::GridFittedBottomGrid, sides)
    open_grid = with_open_sides(grid.underlying_grid, sides)
    immersed_boundary = with_bottom_height(grid.immersed_boundary, grid.immersed_boundary.bottom_height)
    return ImmersedBoundaryGrid(open_grid, immersed_boundary;
                                active_cells_map = has_active_cells_map(grid),
                                active_z_columns = has_active_z_columns(grid))
end

open_topology(::Type{TwoSided{L, R}}, left, right) where {L, R} = TwoSided{ifelse(left, Open, L), ifelse(right, Open, R)}
open_topology(T, left, right) = T

# Names both specify are combined as `combine(user[name], nested[name])`; names only one of them specifies pass through.
function merge_by_name(combine, user, nested)
    names = union(keys(nested), keys(user))
    merged(name) = haskey(user, name) && haskey(nested, name) ? combine(user[name], nested[name]) :
                   haskey(user, name) ? user[name] : nested[name]
    return NamedTuple(name => merged(name) for name in names)
end

# A strip parent knows only the boundaries, so it cannot initialize the interior.
default_initial_state(parent) = parent
default_initial_state(::BoundaryPrescribedComponent) = nothing

"""
$(TYPEDSIGNATURES)

Build a regional hydrostatic ocean on `grid` nested one-way in `parent`, wrapped in a [`NestedModel`](@ref). The parent
is a volumetric [`PrescribedOcean`](@ref), or a
[`BoundaryPrescribedComponent`](@ref NumericalEarth.EarthSystemModels.BoundaryPrescribedComponent) of them that knows the
parent only on strips along the boundaries. Each time step steps the parent first, samples it onto bands along the open
sides of `grid`, then steps the child.

The `open_sides` (default: the walls of `grid` that touch water) are made `Open` and get oblique radiation on the
boundary-normal and tangential velocities (`normal_velocity_scheme`, `tangential_velocity_scheme`; nudged toward the
parent over 9 hours on inflow and 360 days on outflow), tracer reservoirs on the tracers (`tracer_scheme`; relaxing
toward the parent over 20 km on inflow), Flather on the barotropic transport, and Chapman on the free surface. Each
scheme may also be a `NamedTuple` keyed by side. Over the outermost `relaxation_width` cells of each open side (default:
2/3°), `T`, `S` and the baroclinic parts of `u` and `v` relax toward the parent at `relaxation_rate` (s⁻¹; default
1/6 hours), leaving the barotropic flow to Flather. `relaxation_rate = nothing` disables the relaxation, and
`relaxation_rate = (inflow = 1/6hours, outflow = 0)` relaxes `T` and `S` only where the flow enters through the side.
On a curvilinear `grid`, the parent's eastward and northward velocities are rotated onto the grid's own axes.

When the parent's seafloor `parent_bottom_height` (a bottom height `Field`) is known, the Flather exterior transport is
the parent's depth-mean velocity over its own column times the child's wet depth, the boundary-normal velocity is
corrected to integrate to it, and the child's bottom is blended toward the parent's within `bathymetry_blend_width`
cells of each open side (default: 1/3°), but no shallower than `minimum_depth`. Unless
`parent_includes_inverse_barometer = true`, the parent's sea level is taken to lack the inverse-barometer response to the
atmospheric pressure, as in an ocean reanalysis, and the Flather exterior sea level adds the child's own response to the
pressure it is forced with (zero when the child has no atmospheric pressure).

The child's `clock` starts at the parent's time; a `clock` passed in must already be there. The child starts from
`initial_state` (default: the parent, unless the parent knows only the boundaries) interpolated onto `grid`, and
`nothing` leaves it as built. A caller's `boundary_conditions` replace the nested ones side by side, and a caller's
`forcing` is added to the relaxation. The remaining keyword arguments `kw` go to the ocean model, as in
[`ocean_simulation`](@ref).
"""
function nested_ocean_model(parent::OceanParent, grid::AbstractGrid;
                            open_sides = open_boundary_sides(grid),
                            normal_velocity_scheme = default_velocity_scheme(),
                            tangential_velocity_scheme = normal_velocity_scheme,
                            tracer_scheme = default_tracer_scheme(),
                            relaxation_rate = 1/6hours,
                            relaxation_width = cells_spanning(grid, default_relaxation_degrees),
                            bathymetry_blend_width = cells_spanning(grid, default_bathymetry_blend_degrees),
                            minimum_depth = 10,
                            parent_bottom_height = nothing,
                            parent_includes_inverse_barometer = false,
                            initial_state = default_initial_state(parent),
                            clock = Clock{Float64}(time = parent.clock.time, kernel_time_type = eltype(grid)),
                            boundary_conditions = NamedTuple(),
                            forcing = NamedTuple(),
                            kw...)

    validate_nested_ocean_grid(grid)
    open_sides = domain_boundary_sides(grid, open_sides)
    validate_open_sides(grid, open_sides)

    time = parent.clock.time
    clock.time == time ||
        throw(ArgumentError("the child clock (time $(clock.time)) must start at the parent's time, $time."))

    grid = with_open_sides(grid, open_sides)
    grid = blend_parent_bathymetry(grid, parent_bottom_height, open_sides; width = bathymetry_blend_width, minimum_depth)
    width = isnothing(relaxation_rate) ? 1 : relaxation_width
    validate_band_width(grid, open_sides, width, "relaxation_width")
    exchanger = OceanBoundaryExchanger(parent, grid, open_sides; width, time, parent_bottom_height)

    inverse_barometer_sea_level = parent_includes_inverse_barometer ? nothing : Field{Center, Center, Nothing}(grid)

    nested_bcs = ocean_open_boundary_conditions(grid, exchanger.bands; depth_means = exchanger.depth_means,
                                                inverse_barometer = inverse_barometer_sea_level,
                                                normal_velocity_scheme, tangential_velocity_scheme, tracer_scheme)

    relaxation = isnothing(relaxation_rate) ? NamedTuple() :
                 ocean_boundary_relaxation(grid, exchanger.bands; rate = relaxation_rate, width)

    child = hydrostatic_ocean_model(grid;
                                    clock,
                                    boundary_conditions = merge_by_name(merge_boundary_conditions, NamedTuple(boundary_conditions), nested_bcs),
                                    forcing = merge_by_name((user, nested) -> (nested, user), NamedTuple(forcing), relaxation),
                                    kw...)

    isnothing(initial_state) || set_from_parent!(child, initial_state, time)

    child_callbacks = inverse_barometer_callbacks(child, inverse_barometer_sea_level)

    return NestedModel(parent, child, exchanger, child_callbacks)
end

"""
$(TYPEDSIGNATURES)

Build a regional ocean on `grid` nested one-way in `parent_dataset` (e.g. `GLORYSDaily()`) over `dates`, a
`(start_date, end_date)` tuple; time `0` is the start of `first(dates)`. The parent is `parent_dataset` on its native
grid, on strips `relaxation_width` cells wide along the open sides padded by `parent_padding` degrees (default: the
dataset's `default_horizontal_padding`), streaming `parent_time_indices_in_memory` samples at a time. The child starts
from a snapshot of `parent_dataset` at `first(dates)`, and its bathymetry blends toward the dataset's own seafloor or,
for a dataset without one, the seafloor its temperature marks with missing values. Whether the parent's sea level
includes the inverse-barometer response comes from `sea_level_includes_inverse_barometer(parent_dataset)`. The remaining
keyword arguments `kw` go to `nested_ocean_model(parent, grid; kw...)`.
"""
function nested_ocean_model(grid::AbstractGrid, parent_dataset; dates,
                            dir = default_download_directory(parent_dataset),
                            parent_padding = default_horizontal_padding(parent_dataset),
                            parent_time_indices_in_memory = 2,
                            relaxation_width = cells_spanning(grid, default_relaxation_degrees),
                            bathymetry_blend_width = cells_spanning(grid, default_bathymetry_blend_degrees),
                            parent_bottom_height = nothing,
                            open_sides = open_boundary_sides(grid),
                            parent_includes_inverse_barometer = sea_level_includes_inverse_barometer(parent_dataset),
                            kw...)

    validate_nested_ocean_grid(grid)
    open_sides = domain_boundary_sides(grid, open_sides)
    validate_open_sides(grid, open_sides)
    region = BoundingBox(underlying_grid(grid); padding = parent_padding)

    # The seafloor the parent state was produced over: the dataset's own, or where its temperature is missing
    if isnothing(parent_bottom_height)
        metadatum = something(matching_bathymetry(parent_dataset; region, dir),
                              Metadatum(:temperature; dataset = parent_dataset, date = first(dates), region, dir))
        parent_bottom_height = regrid_bathymetry(underlying_grid(grid), metadatum; major_basins = Inf)
    end

    parent = BoundaryPrescribedComponent(PrescribedOcean, underlying_grid(grid), dates, parent_dataset;
                                         width = relaxation_width, padding = parent_padding, sides = open_sides,
                                         dir, time_indices_in_memory = parent_time_indices_in_memory)

    snapshot = PrescribedOcean(region, [first(dates)], parent_dataset; architecture = architecture(grid), dir)

    return nested_ocean_model(parent, grid; open_sides, relaxation_width, bathymetry_blend_width,
                              parent_bottom_height, parent_includes_inverse_barometer, initial_state = snapshot, kw...)
end

"""
$(TYPEDSIGNATURES)

Return a `Simulation` of [`nested_ocean_model`](@ref)`(grid, parent_dataset; dates, verbose, kw...)`, the nested
counterpart of [`ocean_simulation`](@ref), stepping with `Δt` (default: the maximum stable time step estimated
from `grid`) until `stop_time`. The ocean starts from the parent at `first(dates)`.
"""
function nested_ocean_simulation(grid::AbstractGrid, parent_dataset; dates,
                                 Δt = estimate_maximum_Δt(grid),
                                 stop_time = Inf,
                                 verbose = false,
                                 kw...)

    model = nested_ocean_model(grid, parent_dataset; dates, verbose, kw...)
    return Simulation(model; Δt, stop_time, verbose)
end

# The parent interpolated onto the whole child grid, so the interior starts consistent with the boundaries
function set_from_parent!(child, parent, time)
    grid = child.grid
    sources = child_frame_sources(grid, parent_ocean_sources(parent))

    u₀ = XFaceField(grid)
    v₀ = YFaceField(grid)
    T₀ = CenterField(grid)
    S₀ = CenterField(grid)
    η₀ = Field{Center, Center, Nothing}(grid)

    sample_parent!(u₀, boundary_location(:u), sources.u, time)
    sample_parent!(v₀, boundary_location(:v), sources.v, time)
    sample_parent!(T₀, boundary_location(:T), sources.T, time)
    sample_parent!(S₀, boundary_location(:S), sources.S, time)
    sample_parent!(η₀, boundary_location(:η), sources.η, time)

    set!(child; u = u₀, v = v₀, T = T₀, S = S₀, η = interior(η₀), intrinsic_velocities = true)

    return child
end
