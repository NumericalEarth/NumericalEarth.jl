#####
##### nested_ocean_model: a regional ocean nested in a parent ocean
#####

using Oceananigans.Fields: CenterField, XFaceField, YFaceField, interior, interpolate!
using Oceananigans.BoundaryConditions: ObliqueRadiation, TracerReservoir
using Oceananigans.Grids: LatitudeLongitudeGrid, λnodes, φnodes
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Units: days, hours, kilometers
using Oceananigans.DistributedComputations: all_reduce

using ..Bathymetry: regrid_bathymetry
using ..Oceans: hydrostatic_ocean_model, merge_boundary_conditions
using ..DataWrangling: BoundingBox, Metadatum, default_download_directory, default_horizontal_padding,
                       matching_bathymetry

# The open-boundary defaults of the reference Mid-Atlantic Bight configuration (NumericalEarthSandbox, script 05):
# oblique radiation on the velocities and tracer reservoirs on the tracers, a 6-hour sponge 8 cells wide and a
# bathymetry blend 4 cells wide at 1/12°, the widths held fixed in degrees across resolutions.
default_velocity_scheme() = ObliqueRadiation(inflow_timescale = 3days, outflow_timescale = 360days)
default_tracer_scheme() = TracerReservoir(inflow_length_scale = 20kilometers, outflow_length_scale = 0)

const default_relaxation_degrees = 8 / 12
const default_bathymetry_blend_degrees = 4 / 12

# The number of cells of `grid` that span `degrees` along its finest horizontal direction.
function cells_spanning(grid, degrees)
    cpu_grid = on_architecture(CPU(), underlying_grid(grid))
    Δ = min(minimum(diff(λnodes(cpu_grid, Face()))), minimum(diff(φnodes(cpu_grid, Face()))))
    return max(1, round(Int, degrees / Δ))
end

const OceanParent = Union{PrescribedOcean, BoundaryPrescribedComponent}

underlying_grid(grid) = grid
underlying_grid(grid::ImmersedBoundaryGrid) = grid.underlying_grid

function validate_nested_ocean_grid(grid)
    underlying_grid(grid) isa LatitudeLongitudeGrid ||
        throw(ArgumentError("a nested ocean requires a LatitudeLongitudeGrid (possibly immersed), whose velocity " *
                            "components are eastward and northward like the parent's; got $(summary(grid))."))
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

# Sides the user left unspecified keep the open-boundary conditions; fields only one of the two
# specifies pass through.
function merge_field_boundary_conditions(user, nested)
    names = union(keys(nested), keys(user))
    merged(name) = haskey(user, name) && haskey(nested, name) ? merge_boundary_conditions(user[name], nested[name]) :
                   haskey(user, name) ? user[name] : nested[name]
    return NamedTuple(name => merged(name) for name in names)
end

function combine_forcings(relaxation, user)
    names = union(keys(relaxation), keys(user))
    combined(name) = haskey(user, name) && haskey(relaxation, name) ? (relaxation[name], user[name]) :
                     haskey(user, name) ? user[name] : relaxation[name]
    return NamedTuple(name => combined(name) for name in names)
end

# The child steps from the parent's time.
nested_clock(grid, time::Number) = Clock{Float64}(; time, kernel_time_type = eltype(grid))
nested_clock(grid, time) = Clock{typeof(time)}(; time, kernel_time_type = eltype(grid))

# A strip parent knows only the boundaries, so it cannot initialize the interior.
default_initial_state(parent) = parent
default_initial_state(::BoundaryPrescribedComponent) = nothing

"""
$(TYPEDSIGNATURES)

Build a regional hydrostatic ocean on `grid` nested one-way in `parent`, wrapped in a [`NestedModel`](@ref).
The parent is a volumetric [`PrescribedOcean`](@ref), or a
[`BoundaryPrescribedComponent`](@ref NumericalEarth.EarthSystemModels.BoundaryPrescribedComponent) of them
that knows the parent only on strips along the boundaries. Time stepping the `NestedModel` steps the parent
first, samples it onto the boundary bands of `grid` (an [`OceanBoundaryExchanger`](@ref)), then steps the child.

Each side in `open_sides` (default: the `Bounded` sides of the child grid that touch water; see
[`open_boundary_sides`](@ref)) gets the open boundary conditions of [`ocean_open_boundary_conditions`](@ref): oblique
radiation on the boundary-normal and tangential velocities (`normal_velocity_scheme`, `tangential_velocity_scheme`;
nudged toward the parent over 3 days on inflow and 360 days on outflow), tracer reservoirs on the tracers
(`tracer_scheme`; relaxing toward the parent over 20 km on inflow), Flather on the barotropic transport, and Chapman
on the free surface. Over the outermost `relaxation_width` cells of each open side (default: 2/3°), `T` and `S` and
the baroclinic parts of `u` and `v` relax toward the parent at `relaxation_rate` (s⁻¹; default 1/6 hours), leaving
the barotropic flow to Flather (see [`ocean_boundary_relaxation`](@ref)); `relaxation_rate = nothing` disables it.
These defaults follow the reference Mid-Atlantic Bight configuration of NumericalEarthSandbox.

When the parent's seafloor `parent_bottom_height` (a bottom height `Field`) is known, the Flather exterior transport
is the parent's depth mean velocity over its own column times the child's wet depth, and the boundary-normal
velocity is corrected to integrate to it ([`BarotropicBoundaryTransport`](@ref), [`ConsistentBoundaryVelocity`](@ref)),
and the child is built on a copy of `grid` whose immersed bottom is blended toward it within `bathymetry_blend_width` cells of each open side (default: 1/3°; see
[`blend_parent_bathymetry`](@ref)), so that the geometry at the open boundaries matches the one the parent state
was produced over, but never shallower than `minimum_depth`.

The child's `clock` starts at the parent's time; a `clock` passed in must already be there. The child is
initialized from `initial_state` (default: the
parent, unless the parent knows only the boundaries), a parent-like ocean interpolated onto `grid` at the
parent's time; `nothing` leaves the child as built. `boundary_conditions` and `forcing` the caller passes
are merged with the nested ones: a caller's boundary condition replaces the nested one on its side, and a
caller's forcing is added to the relaxation. The remaining keyword arguments `kw` go to
[`hydrostatic_ocean_model`](@ref).
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
                            initial_state = default_initial_state(parent),
                            clock = nested_clock(grid, parent.clock.time),
                            boundary_conditions = NamedTuple(),
                            forcing = NamedTuple(),
                            kw...)

    validate_nested_ocean_grid(grid)
    isempty(open_sides) && throw(ArgumentError("$(summary(grid)) has no open sides to nest through."))

    time = parent.clock.time
    clock.time == time ||
        throw(ArgumentError("the child clock (time $(clock.time)) must start at the parent's time, $time."))

    grid = blend_parent_bathymetry(grid, parent_bottom_height, open_sides; width = bathymetry_blend_width, minimum_depth)
    width = isnothing(relaxation_rate) ? 1 : relaxation_width
    exchanger = OceanBoundaryExchanger(parent, grid, open_sides; width, time, parent_bottom_height)

    nested_bcs = ocean_open_boundary_conditions(grid, exchanger.bands; depth_means = exchanger.depth_means,
                                                normal_velocity_scheme, tangential_velocity_scheme, tracer_scheme)

    relaxation = isnothing(relaxation_rate) ? NamedTuple() :
                 ocean_boundary_relaxation(grid, exchanger.bands; rate = relaxation_rate, width)

    child = hydrostatic_ocean_model(grid;
                                    clock,
                                    boundary_conditions = merge_field_boundary_conditions(NamedTuple(boundary_conditions), nested_bcs),
                                    forcing = combine_forcings(relaxation, NamedTuple(forcing)),
                                    kw...)

    isnothing(initial_state) || set_from_parent!(child, initial_state, time)

    return NestedModel(parent, child, exchanger)
end

"""
$(TYPEDSIGNATURES)

Build a regional ocean on `grid` nested one-way in `parent_dataset` (e.g. `GLORYSDaily()`) over `dates` — a
`(start_date, end_date)` tuple expanded to the dataset's native cadence — and initialize it from the dataset
at `first(dates)`; time `0` is the start of `first(dates)`. The parent is a
[`BoundaryPrescribedComponent`](@ref NumericalEarth.EarthSystemModels.BoundaryPrescribedComponent) of
volumetric [`PrescribedOcean`](@ref)s on strips along the open sides: `parent_dataset` on its native grid,
along each side and `relaxation_width` cells inward, padded by `parent_padding` degrees (default the dataset's
`default_horizontal_padding`), streaming `parent_time_indices_in_memory` samples at a time. A full-domain
snapshot of `parent_dataset` at `first(dates)`, with the same padding, initializes the child. The child bathymetry
blends toward `parent_dataset`'s own seafloor (see [`matching_bathymetry`](@ref)), or, for a dataset without one,
the seafloor its temperature marks with missing values at `first(dates)` (see [`regrid_bathymetry`](@ref)). The remaining keyword arguments `kw` go to
`nested_ocean_model(parent, grid; kw...)`.
"""
function nested_ocean_model(grid::AbstractGrid, parent_dataset; dates,
                            dir = default_download_directory(parent_dataset),
                            parent_padding = default_horizontal_padding(parent_dataset),
                            parent_time_indices_in_memory = 2,
                            relaxation_width = cells_spanning(grid, default_relaxation_degrees),
                            bathymetry_blend_width = cells_spanning(grid, default_bathymetry_blend_degrees),
                            parent_bottom_height = nothing,
                            open_sides = open_boundary_sides(grid),
                            kw...)

    validate_nested_ocean_grid(grid)
    isempty(open_sides) && throw(ArgumentError("$(summary(grid)) has no open sides to nest through."))
    region = BoundingBox(grid; padding = parent_padding)

    # The seafloor the parent state was produced over, every basin of it: the dataset's own, or where its
    # temperature is missing. The Flather exterior takes the parent's depth mean over it.
    if isnothing(parent_bottom_height)
        metadatum = something(matching_bathymetry(parent_dataset; region, dir),
                              Metadatum(:temperature; dataset = parent_dataset, date = first(dates), region, dir))
        parent_bottom_height = regrid_bathymetry(underlying_grid(grid), metadatum; major_basins = Inf)
    end

    parent = BoundaryPrescribedComponent(PrescribedOcean, grid, dates, parent_dataset;
                                         width = relaxation_width, padding = parent_padding, sides = open_sides,
                                         dir, time_indices_in_memory = parent_time_indices_in_memory)

    snapshot = PrescribedOcean(region, [first(dates)], parent_dataset; architecture = architecture(grid), dir)

    return nested_ocean_model(parent, grid; open_sides, relaxation_width, bathymetry_blend_width,
                              parent_bottom_height, initial_state = snapshot, kw...)
end

# Interpolate the parent onto the whole child grid at `time`, so the interior starts consistent with the
# boundaries.
function set_from_parent!(child, parent, time)
    grid = child.grid
    sources = parent_ocean_sources(parent)

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

    set!(child; u = u₀, v = v₀, T = T₀, S = S₀, η = interior(η₀))

    return child
end
