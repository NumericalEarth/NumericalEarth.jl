#####
##### nested_ocean_model: a regional ocean nested in a parent ocean
#####

using Oceananigans.Fields: AbstractField, CenterField, XFaceField, YFaceField, interpolate!
using Oceananigans.Grids: LatitudeLongitudeGrid
using Oceananigans.ImmersedBoundaries: GridFittedBottom, ImmersedBoundaryGrid
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Units: hours

using ..Bathymetry: bathymetry_from_missing_values, regrid_bathymetry
using ..Oceans: hydrostatic_ocean_model, merge_boundary_conditions
using ..DataWrangling: BoundingBox, Metadatum, default_download_directory, default_horizontal_padding

const default_ocean_relaxation_width = 8

const OceanParent = Union{PrescribedOcean, BoundaryPrescribedComponent}

underlying_grid(grid) = grid
underlying_grid(grid::ImmersedBoundaryGrid) = grid.underlying_grid

function validate_nested_ocean_grid(grid)
    underlying_grid(grid) isa LatitudeLongitudeGrid ||
        throw(ArgumentError("a nested ocean requires a LatitudeLongitudeGrid (possibly immersed), whose velocity " *
                            "components are eastward and northward like the parent's; got $(summary(grid))."))
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

# Child bathymetry: an immersed grid's own bottom, or `bathymetry` — a bottom height `Field`, or a dataset
# (e.g. `ETOPO2022()`) regridded onto the grid — which replaces it. Blending toward the parent's seafloor
# keeps the geometry at the open boundaries consistent with the transport the parent imposes there.
child_bottom_height(grid, ::Nothing) = nothing
function child_bottom_height(grid::ImmersedBoundaryGrid, ::Nothing)
    bottom_height = Field{Center, Center, Nothing}(grid.underlying_grid)
    Base.parent(bottom_height) .= Base.parent(grid.immersed_boundary.bottom_height)
    return bottom_height
end

child_bottom_height(grid, bathymetry::AbstractField) = bathymetry
child_bottom_height(grid, bathymetry) = regrid_bathymetry(underlying_grid(grid); dataset = bathymetry)

function materialize_nested_bathymetry(grid, bathymetry, parent_bottom_height, blend_width)
    blending = !isnothing(parent_bottom_height) && blend_width > 0
    isnothing(bathymetry) && !blending && return grid

    bottom_height = child_bottom_height(grid, bathymetry)
    isnothing(bottom_height) && return grid

    if blending
        parent_height = Field{Center, Center, Nothing}(underlying_grid(grid))
        interpolate!(parent_height, parent_bottom_height)
        blend_parent_terrain!(bottom_height, parent_height; width = blend_width)
    end

    return ImmersedBoundaryGrid(underlying_grid(grid), GridFittedBottom(bottom_height))
end

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
[`open_boundary_sides`](@ref)) gets the open boundary conditions of [`ocean_open_boundary_conditions`](@ref): radiation conditions on the
boundary-normal and tangential velocities and on the tracers (`normal_velocity_scheme`,
`tangential_velocity_scheme`, `tracer_scheme`, which relax toward the parent on inflow over
`boundary_inflow_timescale`), Flather on the barotropic transport, and Chapman on the free surface. Over the
outermost `relaxation_width` cells of each open side, `u`, `v`, `T` and `S` relax toward the parent at
`relaxation_rate` (s⁻¹; see [`ocean_boundary_relaxation`](@ref)); `relaxation_rate = nothing` disables it.

The child's bathymetry is `grid`'s immersed bottom or, when given, `bathymetry`: a bottom height `Field` or a
dataset (e.g. `ETOPO2022()`) regridded onto `grid` (see [`regrid_bathymetry`](@ref)). When the parent's seafloor
`parent_bottom_height` is known, the child bathymetry is blended toward it over an outer frame of physical width
`bathymetry_blend_length` (meters; converted to a resolution-invariant cell count, or overridden directly with
`bathymetry_blend_width`), as [`nested_atmosphere_model`](@ref) blends its terrain, so the geometry at the open
boundaries matches the one the parent state was produced over.

The child's `clock` starts at the parent's time. It is initialized from `initial_state` (default: the
parent, unless the parent knows only the boundaries), a parent-like ocean interpolated onto `grid` at the
parent's time; `nothing` leaves the child as built. `boundary_conditions` and `forcing` the caller passes
are merged with the nested ones: a caller's boundary condition replaces the nested one on its side, and a
caller's forcing is added to the relaxation. The remaining keyword arguments `kw` go to
[`hydrostatic_ocean_model`](@ref).
"""
function nested_ocean_model(parent::OceanParent, grid::AbstractGrid;
                            open_sides = nothing,
                            boundary_inflow_timescale = 6hours,
                            normal_velocity_scheme = NormalRadiation(inflow_timescale = boundary_inflow_timescale,
                                                                     outflow_timescale = Inf),
                            tangential_velocity_scheme = normal_velocity_scheme,
                            tracer_scheme = normal_velocity_scheme,
                            relaxation_rate = 1/12hours,
                            relaxation_width = default_ocean_relaxation_width,
                            bathymetry = nothing,
                            bathymetry_blend_length = 60_000,
                            bathymetry_blend_width = nothing,
                            parent_bottom_height = nothing,
                            initial_state = default_initial_state(parent),
                            clock = Clock(grid),
                            boundary_conditions = NamedTuple(),
                            forcing = NamedTuple(),
                            kw...)

    validate_nested_ocean_grid(grid)

    blend_width = something(bathymetry_blend_width, default_terrain_blend_width(grid, bathymetry_blend_length))
    grid = materialize_nested_bathymetry(grid, bathymetry, parent_bottom_height, blend_width)

    open_sides = something(open_sides, open_boundary_sides(grid))
    isempty(open_sides) && throw(ArgumentError("$(summary(grid)) has no open sides to nest through."))

    # The child starts at the parent's time, and steps the parent along from there.
    time = parent.clock.time
    clock.time = time
    width = isnothing(relaxation_rate) ? 1 : relaxation_width
    exchanger = OceanBoundaryExchanger(parent, grid, open_sides; width, time)

    nested_bcs = ocean_open_boundary_conditions(grid, exchanger.bands;
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
snapshot of `parent_dataset` at `first(dates)`, with the same padding, initializes the child, and the seafloor
`parent_dataset` marks with missing values there (see [`bathymetry_from_missing_values`](@ref)) is the
`parent_bottom_height` the child bathymetry (`bathymetry`, blended over `bathymetry_blend_length` or
`bathymetry_blend_width`) blends toward. The strips follow the open sides of the blended bathymetry. The
remaining keyword arguments `kw` go to `nested_ocean_model(parent, grid; kw...)`.
"""
function nested_ocean_model(grid::AbstractGrid, parent_dataset; dates,
                            dir = default_download_directory(parent_dataset),
                            parent_padding = default_horizontal_padding(parent_dataset),
                            parent_time_indices_in_memory = 2,
                            relaxation_width = default_ocean_relaxation_width,
                            bathymetry = nothing,
                            bathymetry_blend_length = 60_000,
                            bathymetry_blend_width = nothing,
                            parent_bottom_height = nothing,
                            open_sides = nothing,
                            kw...)

    validate_nested_ocean_grid(grid)
    region = BoundingBox(grid; padding = parent_padding)

    if isnothing(parent_bottom_height)
        metadatum = Metadatum(:temperature; dataset = parent_dataset, date = first(dates), region, dir)
        parent_bottom_height = bathymetry_from_missing_values(underlying_grid(grid), metadatum)
    end

    # The strips follow the open sides of the blended child bathymetry.
    blend_width = something(bathymetry_blend_width, default_terrain_blend_width(grid, bathymetry_blend_length))
    grid = materialize_nested_bathymetry(grid, bathymetry, parent_bottom_height, blend_width)

    open_sides = something(open_sides, open_boundary_sides(grid))
    isempty(open_sides) && throw(ArgumentError("$(summary(grid)) has no open sides to nest through."))

    parent = BoundaryPrescribedComponent(PrescribedOcean, grid, dates, parent_dataset;
                                         width = relaxation_width, padding = parent_padding, sides = open_sides,
                                         dir, time_indices_in_memory = parent_time_indices_in_memory)

    snapshot = PrescribedOcean(region, [first(dates)], parent_dataset; architecture = architecture(grid), dir)

    return nested_ocean_model(parent, grid; open_sides, relaxation_width, initial_state = snapshot, kw...)
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
