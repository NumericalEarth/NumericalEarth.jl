#####
##### Nested-ocean model: an Oceananigans child driven by a parent ocean state.
#####
#
# The parent carries the child's own variables (u, v, T, S, η), so the lateral boundary conditions and
# the interior Davies relaxation interpolate the parent series directly — no combine, unlike the
# atmosphere nest. What the parent does not carry is the barotropic transport the Flather condition
# needs; the `OceanStateExchanger` derives it over the child's own column depth (see
# `ocean_state_exchanger.jl`).

const OCEAN_SIDES = (:west, :east, :south, :north)

# The parent velocity that crosses each side, and hence drives its normal-flow boundary condition.
side_normal_velocity(side) = side in (:west, :east) ? :u : :v

"""
$(TYPEDSIGNATURES)

Return the `(Uᵉˣᵗ, ηᵉˣᵗ)` Flather boundary conditions for the barotropic transports `U` and `V`,
reading the exchanger slabs that [`exchange_state!`](@ref) refreshes each step.
"""
function barotropic_boundary_conditions(exchanger; sides = OCEAN_SIDES, gravitational_acceleration)
    zonal = filter(side -> side in (:west, :east), sides)
    meridional = filter(side -> side in (:south, :north), sides)

    flather(side) = (slabs = getproperty(exchanger.boundaries, side);
                     GravityWaveRadiationBoundaryCondition((slabs.U, slabs.η); gravitational_acceleration))

    boundary_conditions = NamedTuple()

    if !isempty(zonal)
        U = FieldBoundaryConditions(; NamedTuple(side => flather(side) for side in zonal)...)
        boundary_conditions = merge(boundary_conditions, (; U))
    end

    if !isempty(meridional)
        V = FieldBoundaryConditions(; NamedTuple(side => flather(side) for side in meridional)...)
        boundary_conditions = merge(boundary_conditions, (; V))
    end

    return boundary_conditions
end

# Momentum radiates as a `NormalFlowBoundaryCondition` on the side its own face coincides with, and as a
# `ValueBoundaryCondition` on the tangential sides, where a normal-flow fill would leave it
# under-constrained — the same per-side split the atmosphere nest uses for `ρu`/`ρv`.
function baroclinic_boundary_condition_types(sides)
    momentum_type(component, side) =
        side_normal_velocity(side) === component ? NormalFlowBoundaryCondition : ValueBoundaryCondition

    u = NamedTuple(side => momentum_type(:u, side) for side in sides)
    v = NamedTuple(side => momentum_type(:v, side) for side in sides)

    return (; u, v, T = ValueBoundaryCondition, S = ValueBoundaryCondition)
end

"""
$(TYPEDSIGNATURES)

Build a child ocean over `child_grid` nested in `parent_ocean` — a volumetric [`PrescribedOcean`](@ref)
carrying the parent's `u`, `v`, `T`, `S` and `η` — wrapped in a `NestedModel`.

The child's baroclinic velocities and tracers radiate through `NormalRadiation` (Orlanski, with
Marchesiello et al. adaptive nudging toward the parent at `inflow_timescale`/`outflow_timescale`), and
its barotropic transport through `GravityWaveRadiation` (Flather) against the exterior transport the
exchanger integrates over the child's own column. Oceananigans pairs the free surface's Chapman
companion condition automatically on the Flather sides.

When `relaxation_rate` (s⁻¹) is given, the child interior is additionally relaxed toward the parent
over `relaxation_mask` — by default a cosine ramp over the outermost `relaxation_width` cells.

Any `boundary_conditions`/`forcing` the caller passes are merged with the parent-derived ones (caller
wins), and remaining keyword arguments flow to `ocean_model`.
"""
function NestedModels.nested_ocean_model(parent_ocean::PrescribedOcean, child_grid::AbstractGrid;
                                         sides = OCEAN_SIDES,
                                         inflow_timescale = 1days,
                                         outflow_timescale = 360days,
                                         relaxation_rate = nothing,
                                         relaxation_width = 5,
                                         relaxation_mask = davies_relaxation_mask(child_grid, relaxation_width),
                                         gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
                                         free_surface = default_free_surface(child_grid),
                                         boundary_conditions = NamedTuple(),
                                         forcing = NamedTuple(),
                                         kw...)

    is_three_dimensional(parent_ocean.grid) || throw(ArgumentError(
        "a nested ocean needs a volumetric parent; `PrescribedOcean` over a single-level grid carries " *
        "surface fields only, which cannot drive the child's baroclinic boundaries"))

    exchanger = ocean_state_exchanger(parent_ocean, child_grid; sides)

    radiation = NormalRadiation(eltype(child_grid); inflow_timescale, outflow_timescale)
    schemes = (u = radiation, v = radiation, T = radiation, S = radiation)
    bc_types = baroclinic_boundary_condition_types(sides)
    variables = (u = exchanger.variables.u, v = exchanger.variables.v,
                 T = exchanger.variables.T, S = exchanger.variables.S)

    baroclinic_bcs = parent_boundary_conditions(child_grid; variables, sides, schemes, bc_types)
    barotropic_bcs = barotropic_boundary_conditions(exchanger; sides, gravitational_acceleration)
    nested_bcs = merge(baroclinic_bcs, barotropic_bcs)

    davies = if isnothing(relaxation_rate)
        NamedTuple()
    else
        mask = relaxation_mask isa Number ? Returns(relaxation_mask) : relaxation_mask
        parent_forcings(; variables, rate = relaxation_rate, mask)
    end

    child = ocean_model(child_grid;
                        free_surface,
                        gravitational_acceleration,
                        boundary_conditions = merge(nested_bcs, NamedTuple(boundary_conditions)),
                        forcing = merge(davies, NamedTuple(forcing)),
                        kw...)

    return NestedModel(parent_ocean, child, exchanger)
end

# The parent a nested ocean child is driven by, on the dataset's native grid over `region`.
function prescribed_parent_ocean(dataset, region, dates, architecture, FT, dir)
    series(name) = FieldTimeSeries(Metadata(name; dataset, dates, region, dir), architecture;
                                   time_indexing = LinearTimeIndexing())

    temperature = series(:temperature)

    return PrescribedOcean(temperature.grid, temperature.times;
                           FT,
                           clock = Clock{FT}(time = 0),
                           temperature,
                           salinity = series(:salinity),
                           velocities = (u = series(:u_velocity), v = series(:v_velocity)),
                           free_surface = series(:free_surface))
end

"""
$(TYPEDSIGNATURES)

Build the parent ocean state from `parent_dataset`, nest a child ocean in it over `child_grid`, and
initialize the child from the same dataset at `first(dates)` — the returned model is ready to step.

The parent spans `child_grid`'s bounding box padded by `parent_padding` (default `parent_dataset`'s
`default_horizontal_padding`, margin for the lateral-boundary interpolation stencils) at `dates`, on
`parent_dataset`'s native grid. The child is initialized through the same interpolation that drives its
boundaries, so interior and wall agree at `first(dates)` and no standing jump forces the walls.

Remaining keyword arguments flow to `nested_ocean_model(parent_ocean, child_grid; kw...)`.
"""
function NestedModels.nested_ocean_model(child_grid::AbstractGrid, parent_dataset;
                                         dates,
                                         dir = default_download_directory(parent_dataset),
                                         parent_padding = default_horizontal_padding(parent_dataset),
                                         kw...)

    architecture = Architectures.architecture(child_grid)
    parent_region = BoundingBox(child_grid; padding = parent_padding)
    parent_ocean = prescribed_parent_ocean(parent_dataset, parent_region, dates, architecture,
                                           eltype(child_grid), dir)

    nested_model = NestedModels.nested_ocean_model(parent_ocean, child_grid; kw...)
    initialize_nested_child!(nested_model, parent_dataset, first(dates), parent_region, dir)

    return nested_model
end

# Initialize the child from the SAME parent-derived state that drives its lateral boundaries, so the
# interior initial condition and the prescribed boundary agree at the walls.
function initialize_nested_child!(nested_model, dataset, date, region, dir)
    child = nested_model.child
    variables = (:temperature, :salinity, :u_velocity, :v_velocity, :free_surface)
    metadata_set = MetadataSet(variables; dataset, date, region, dir)

    set!(child, metadata_set)
    update_state!(nested_model)

    return nested_model
end
