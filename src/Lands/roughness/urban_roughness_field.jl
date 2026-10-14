#####
##### On-grid evaluation of the urban roughness closures: (λᵖ, h) → (ℓᵐ, d) fields.
#####

@kernel function _compute_urban_aerodynamic_roughness!(ℓᵐ, d, closure, plan_area_index, mean_building_height, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    λᵖ = convert(FT, stateindex(plan_area_index, i, j, 1))
    h  = convert(FT, stateindex(mean_building_height, i, j, 1))
    φ  = φnode(i, j, 1, grid, Center(), Center(), Center())
    cell = (plan_area_index = λᵖ, mean_building_height = h, latitude = φ)
    ℓᵐᵢⱼ, dᵢⱼ = aerodynamic_parameters(closure, cell)
    @inbounds ℓᵐ[i, j, 1] = ℓᵐᵢⱼ
    @inbounds d[i, j, 1] = dᵢⱼ
end

"""
$(TYPEDSIGNATURES)

Fill the momentum roughness length `ℓᵐ` and zero-plane displacement `d` (meters) in place
by applying an urban `closure` ([`AbstractUrbanRoughness`](@ref)) over every cell of `grid`.
`properties` is a NamedTuple of the closure's per-cell inputs — scalars, arrays or `Field`s;
the urban closures expect `plan_area_index` and `mean_building_height`.
"""
function compute_aerodynamic_roughness!(ℓᵐ, d, closure::AbstractUrbanRoughness, properties, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_urban_aerodynamic_roughness!,
            ℓᵐ, d, closure, properties.plan_area_index, properties.mean_building_height, grid)
    return ℓᵐ, d
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` and zero-plane displacement `d` (as `Field`s on the grid
of `h`) from a mean building-height field `h` and a plan-area index field `λᵖ`,
under `closure` (default [`MorphometricRoughness`](@ref)). Returns `(ℓᵐ, d)`. Where `λᵖ → 0`
the result reduces to a bare-soil roughness, and cells of invalid morphometry are `NaN`; fill them
with [`fill_aerodynamic_roughness_gaps!`](@ref) before passing the pair to a flux closure.
"""
function urban_roughness(h::AbstractField, λᵖ; closure = MorphometricRoughness(eltype(h.grid)))
    grid = h.grid
    ℓᵐ = Field{Center, Center, Nothing}(grid)
    d  = Field{Center, Center, Nothing}(grid)
    compute_aerodynamic_roughness!(ℓᵐ, d, closure, (; plan_area_index = λᵖ, mean_building_height = h), grid)
    return ℓᵐ, d
end

#####
##### Gap filling
#####

# A similarity profile needs a finite, positive roughness length.
@inline roughness_gap(ℓᵐ, d) = !(isfinite(ℓᵐ) & (ℓᵐ > 0)) | !isfinite(d)

@kernel function _fill_aerodynamic_roughness_gaps!(ℓᵐ, d, ℓᵘ, dᵘ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ℓᵐᵢⱼ = ℓᵐ[i, j, 1]
        dᵢⱼ  = d[i, j, 1]
        gap  = roughness_gap(ℓᵐᵢⱼ, dᵢⱼ)
        ℓᵐ[i, j, 1] = ifelse(gap, ℓᵘ, ℓᵐᵢⱼ)
        d[i, j, 1]  = ifelse(gap, dᵘ, dᵢⱼ)
    end
end

"""
$(TYPEDSIGNATURES)

Replace, in place, the gaps left where `closure` could not evaluate a cell with the
`unbuilt` pair `(ℓ, d)`, by default the closure's own bare-soil endpoint
`aerodynamic_parameters(closure, 0, 0)`. A gap is a non-finite `ℓᵐ` or `d`, or a
non-positive `ℓᵐ`, which a similarity profile cannot take either. The roughness length `ℓᵐ`
and zero-plane displacement `d` are filled together, so a gap in either becomes unbuilt
surface in both.

GHSL publishes no tile for the all-ocean cells of its Mollweide grid, so the offshore part
of a coastal window comes back as gaps, as does any no-data pixel inside a published tile.
The fill makes those cells evaluable and no more: bare soil (`ℓᵐ = 0.03` m by default) is
two orders of magnitude rougher than a water surface, and the atmosphere-land flux closure
applies land similarity theory wherever the land component runs. Where the gaps are known
to be open water, pass a water roughness such as `unbuilt = (1e-4, 0)`, or keep the land
component off those cells. For gaps that fall inside built-up land, inpaint `λᵖ` and `h`
with `DataWrangling.inpaint_mask!` before evaluating the closure instead, so the fill comes
from horizontal neighbors rather than from a constant.
"""
function fill_aerodynamic_roughness_gaps!(ℓᵐ, d, closure::AbstractUrbanRoughness; unbuilt = aerodynamic_parameters(closure, 0, 0))
    ℓᵘ, dᵘ = unbuilt
    ℓᵘ > 0 || throw(ArgumentError("the unbuilt roughness length must be positive, got $ℓᵘ"))
    dᵘ >= 0 || throw(ArgumentError("the unbuilt zero-plane displacement must be non-negative, got $dᵘ"))

    grid = ℓᵐ.grid
    FT = eltype(ℓᵐ)
    launch!(architecture(grid), grid, :xy, _fill_aerodynamic_roughness_gaps!, ℓᵐ, d, convert(FT, ℓᵘ), convert(FT, dᵘ))
    return ℓᵐ, d
end

#####
##### Measured-morphometry builder: per-cell σʰ, hᵐᵃˣ and λᶠ from a footprint-level dataset
##### instead of the closure's regressions and frontal-area estimator.
#####

@kernel function _compute_measured_urban_aerodynamic_roughness!(ℓᵐ, d, closure,
                                                                plan_area_index, mean_building_height,
                                                                building_height_deviation, maximum_building_height,
                                                                frontal_area, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    λᵖ   = convert(FT, stateindex(plan_area_index, i, j, 1))
    h    = convert(FT, stateindex(mean_building_height, i, j, 1))
    σʰ   = convert(FT, stateindex(building_height_deviation, i, j, 1))
    hᵐᵃˣ = convert(FT, stateindex(maximum_building_height, i, j, 1))
    λᶠ   = convert(FT, stateindex(frontal_area, i, j, 1))
    ℓᵐᵢⱼ, dᵢⱼ = aerodynamic_parameters(closure, λᵖ, h, σʰ, hᵐᵃˣ, λᶠ)
    @inbounds ℓᵐ[i, j, 1] = ℓᵐᵢⱼ
    @inbounds d[i, j, 1] = dᵢⱼ
end

const MeasuredMorphometryProperties = NamedTuple{(:plan_area_index, :mean_building_height,
                                                  :building_height_deviation, :maximum_building_height,
                                                  :frontal_area_index)}

# Selected over the estimator-based method by the richer `properties` NamedTuple.
function compute_aerodynamic_roughness!(ℓᵐ, d, closure::AbstractUrbanRoughness,
                                        properties::MeasuredMorphometryProperties, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_measured_urban_aerodynamic_roughness!,
            ℓᵐ, d, closure, properties.plan_area_index, properties.mean_building_height,
            properties.building_height_deviation, properties.maximum_building_height,
            properties.frontal_area_index, grid)
    return ℓᵐ, d
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` and zero-plane displacement `d` (as `Field`s on the grid of
`h`) from **measured** per-cell morphometry: the mean building height `h`, plan-area index
`λᵖ`, height standard deviation `σʰ`, maximum height `hᵐᵃˣ` and frontal-area index `λᶠ` —
the fields a footprint-level dataset such as `GlobalBuildingFootprints3D` aggregates. The
closure (default [`MorphometricRoughness`](@ref)) is fed the measured height heterogeneity
in place of its frontal-area estimator and `σʰ`/`hᵐᵃˣ` regressions. Where `λᵖ → 0` the
result reduces to a bare-soil roughness. Cells the closure cannot evaluate are `NaN`; fill
them with [`fill_aerodynamic_roughness_gaps!`](@ref) before passing the pair to a flux closure.
"""
function urban_roughness(h::AbstractField, λᵖ, σʰ, hᵐᵃˣ, λᶠ; closure = MorphometricRoughness(eltype(h.grid)))
    grid = h.grid
    ℓᵐ = Field{Center, Center, Nothing}(grid)
    d  = Field{Center, Center, Nothing}(grid)
    properties = (; plan_area_index = λᵖ, mean_building_height = h,
                    building_height_deviation = σʰ, maximum_building_height = hᵐᵃˣ,
                    frontal_area_index = λᶠ)
    compute_aerodynamic_roughness!(ℓᵐ, d, closure, properties, grid)
    return ℓᵐ, d
end

#####
##### Neighborhood-scale evaluation from building datasets: dataset → lattice morphometry →
##### neighborhood morphometry → closure → drag-preserving reduction to the grid.
#####

"""
$(TYPEDSIGNATURES)

Per-cell building morphometry on `grid` from a building `dataset`, as a NamedTuple of `Field`s
named after the closure inputs: `plan_area_index` and `mean_building_height` (m), and, where
the dataset measures them, `building_height_deviation` (m), `maximum_building_height` (m) and
`frontal_area_index`. Dataset modules add methods for their products.
"""
function building_morphometry end

# Neighborhood statistics use spherical overlap areas and built-area height moments.
@kernel function _neighborhood_morphometry!(pooled, grid, longitude_faces, latitude_faces, properties, neighborhood)
    i, j = @index(Global, NTuple)
    longitude_center = (longitude_faces[i] + longitude_faces[i + 1]) / 2
    latitude_center = (latitude_faces[j] + latitude_faces[j + 1]) / 2
    longitude_spacing = longitude_faces[i + 1] - longitude_faces[i]
    latitude_spacing = latitude_faces[j + 1] - latitude_faces[j]
    longitude_radius = max(0, round(Int, (neighborhood / (grid.radius * cosd(latitude_center) * deg2rad(longitude_spacing)) - 1) / 2))
    latitude_radius = max(0, round(Int, (neighborhood / (grid.radius * deg2rad(latitude_spacing)) - 1) / 2))
    west = ifelse(longitude_radius == 0, longitude_faces[i], longitude_center - (longitude_radius + 1//2) * longitude_spacing)
    east = ifelse(longitude_radius == 0, longitude_faces[i + 1], longitude_center + (longitude_radius + 1//2) * longitude_spacing)
    south = max(latitude_faces[1], ifelse(latitude_radius == 0, latitude_faces[j], latitude_center - (latitude_radius + 1//2) * latitude_spacing))
    north = min(latitude_faces[end], ifelse(latitude_radius == 0, latitude_faces[j + 1], latitude_center + (latitude_radius + 1//2) * latitude_spacing))
    order = ForwardOrdering()
    first_column = max(1, searchsortedlast(longitude_faces, west, 1, length(longitude_faces), order))
    last_column = min(size(grid, 1), searchsortedfirst(longitude_faces, east, 1, length(longitude_faces), order) - 1)
    first_row = max(1, searchsortedlast(latitude_faces, south, 1, length(latitude_faces), order))
    last_row = min(size(grid, 2), searchsortedfirst(latitude_faces, north, 1, length(latitude_faces), order) - 1)
    ground_area = built_area = volume = second_moment = maximum_height = frontal_area = convert(eltype(grid), 0)
    for row in first_row:last_row, column in first_column:last_column
        overlap_south = max(south, latitude_faces[row])
        overlap_north = min(north, latitude_faces[row + 1])
        # sin(north) - sin(south) as 2 cos(midpoint) sin(half-width), which does not cancel in Float32 for thin cells
        area = max(0, min(east, longitude_faces[column + 1]) - max(west, longitude_faces[column])) *
               2cosd((overlap_north + overlap_south) / 2) * max(0, sind((overlap_north - overlap_south) / 2))
        fraction = stateindex(properties.plan_area_index, column, row, 1)
        height = stateindex(properties.mean_building_height, column, row, 1)
        deviation = stateindex(properties.building_height_deviation, column, row, 1)
        tallest = stateindex(properties.maximum_building_height, column, row, 1)
        frontal = stateindex(properties.frontal_area_index, column, row, 1)
        frontal = ifelse((fraction == 0) & isnan(frontal), 0, frontal)
        built = area * fraction
        ground_area += area
        built_area += built
        volume += ifelse(built > 0, built * height, 0)
        second_moment += ifelse(built > 0, built * (height^2 + deviation^2), 0)
        maximum_height = max(maximum_height, ifelse(built > 0, tallest, 0))
        frontal_area += ifelse(area > 0, area * frontal, 0)
    end
    mean_height = ifelse(built_area > 0, volume / built_area, 0)
    variance = ifelse(built_area > 0, second_moment / built_area - mean_height^2, 0)
    @inbounds begin
        pooled[1][i, j, 1] = built_area / ground_area
        pooled[2][i, j, 1] = mean_height
        pooled[3][i, j, 1] = sqrt(max(0, variance))
        pooled[4][i, j, 1] = maximum_height
        pooled[5][i, j, 1] = frontal_area / ground_area
    end
end

function neighborhood_morphometry(properties, grid, longitude_faces, latitude_faces, neighborhood)
    measured = hasproperty(properties, :frontal_area_index)
    inputs = (; properties.plan_area_index, properties.mean_building_height,
                building_height_deviation = get(properties, :building_height_deviation, 0),
                maximum_building_height = get(properties, :maximum_building_height, 0),
                frontal_area_index = get(properties, :frontal_area_index, 0))
    pooled = ntuple(_ -> Field{Center, Center, Nothing}(grid), 5)
    launch!(architecture(grid), grid, :xy, _neighborhood_morphometry!, pooled, grid, longitude_faces, latitude_faces,
            inputs, neighborhood)
    morphometry = (; plan_area_index = pooled[1], mean_building_height = pooled[2])
    measured || return morphometry
    return (; morphometry..., building_height_deviation = pooled[3],
              maximum_building_height = pooled[4], frontal_area_index = pooled[5])
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length and zero-plane displacement (m) of the built-up surface on
`grid` (a `LatitudeLongitudeGrid`) from a building `dataset`, as the NamedTuple of `Field`s
`(; momentum_roughness_length, zero_plane_displacement, urban_fraction, building_height)`.

The closure is evaluated on neighborhoods about `neighborhood` (m) wide, from the morphometry
that [`building_morphometry`](@ref) supplies on a lattice: a grid cell wider than `neighborhood`
is subdivided into neighborhood-sized lattice cells, and on a finer grid each cell pools the
cells within half a neighborhood of it, clipped at the grid edges. Neighborhoods with plan-area
index at least the closure's `minimum_built_fraction` are urban, and `urban_fraction` is their
share of the grid-cell area. Each grid cell takes the displacement and roughness length whose
neutral drag at `surface_layer_height` (m) is the area-mean drag of its urban neighborhoods
[Mason (1988)](@cite Mason1988), the displacement being their drag-weighted mean, the mean
level of momentum absorption [Jackson (1981)](@cite Jackson1981). `building_height` is their
mean height weighted by built area. A grid cell without urban neighborhoods takes the closure's
bare-soil limit. An urban neighborhood with non-finite roughness or displacement, or
non-positive roughness, raises an `ArgumentError`. Remaining keyword arguments pass to
`building_morphometry`.
"""
function urban_roughness(grid::LatitudeLongitudeGrid, dataset; surface_layer_height, closure = MorphometricRoughness(eltype(grid)),
                         neighborhood = 1000, kw...)
    FT = eltype(grid)
    arch = architecture(grid)
    longitude_cells, latitude_cells, _ = size(grid)
    cpu_grid = on_architecture(CPU(), grid)
    longitude_faces = collect(λnodes(cpu_grid, Face()))
    latitude_faces = collect(φnodes(cpu_grid, Face()))
    reduction_grid = LatitudeLongitudeGrid(arch, FT; size = (longitude_cells, latitude_cells),
                                           longitude = longitude_faces, latitude = latitude_faces, radius = grid.radius,
                                           topology = (Bounded, Bounded, Flat))
    longitude_scale = grid.radius * deg2rad(1) * cosd(clamp(0, first(latitude_faces), last(latitude_faces)))
    latitude_scale = grid.radius * deg2rad(1)
    function subdivide(faces, scale)
        subdivisions = max.(1, round.(Int, diff(faces) .* scale ./ neighborhood))
        return [[faces[i] + (faces[i + 1] - faces[i]) * k / subdivisions[i]
                 for i in eachindex(subdivisions) for k in 0:subdivisions[i]-1]; faces[end]]
    end
    longitude_lattice = subdivide(longitude_faces, longitude_scale)
    latitude_lattice = subdivide(latitude_faces, latitude_scale)
    lattice = LatitudeLongitudeGrid(arch, FT; size = (length(longitude_lattice) - 1, length(latitude_lattice) - 1),
                                    longitude = longitude_lattice, latitude = latitude_lattice, radius = grid.radius,
                                    topology = (Bounded, Bounded, Flat))

    properties = neighborhood_morphometry(building_morphometry(lattice, dataset; kw...), lattice,
                                          on_architecture(arch, longitude_lattice), on_architecture(arch, latitude_lattice), neighborhood)
    lattice_roughness    = Field{Center, Center, Nothing}(lattice)
    lattice_displacement = Field{Center, Center, Nothing}(lattice)
    compute_aerodynamic_roughness!(lattice_roughness, lattice_displacement, closure, properties, lattice)

    # Neutral drag coefficients over ϰ² of the urban neighborhoods at the surface-layer height
    roughness = interior(lattice_roughness, :, :, 1)
    displacement = interior(lattice_displacement, :, :, 1)
    fraction = interior(properties.plan_area_index, :, :, 1)
    height = interior(properties.mean_building_height, :, :, 1)
    urban = fraction .>= closure.minimum_built_fraction
    any(urban .& roughness_gap.(roughness, displacement)) &&
        throw(ArgumentError("urban neighborhoods must have finite, positive roughness lengths and finite displacements"))
    any(urban .& (surface_layer_height .<= displacement .+ roughness)) &&
        throw(ArgumentError("surface_layer_height = $surface_layer_height m must exceed d + ℓᵐ in every urban neighborhood"))
    scaled_drag = inv.(log.((surface_layer_height .- displacement) ./ roughness)) .^ 2
    scaled_drag .= ifelse.(urban, scaled_drag, 0)
    contributions = (urban, scaled_drag, ifelse.(urban, scaled_drag .* displacement, 0),
                     ifelse.(urban, fraction, 0), ifelse.(urban, fraction .* height, 0))
    urban_fraction, mean_scaled_drag, scaled_drag_moment, built_fraction, building_volume = map(contributions) do data
        source = Field{Center, Center, Nothing}(lattice)
        target = Field{Center, Center, Nothing}(reduction_grid)
        set!(source, data)
        if longitude_lattice == longitude_faces && latitude_lattice == latitude_faces
            set!(target, source)
        else
            regrid!(target, source)
        end
        return interior(target, :, :, 1)
    end
    mean_displacement = scaled_drag_moment ./ mean_scaled_drag
    has_urban = urban_fraction .> 0
    bare_roughness, bare_displacement = aerodynamic_parameters(closure, 0, 0)
    roughness = ifelse.(has_urban, (surface_layer_height .- mean_displacement) .* exp.(-sqrt.(urban_fraction ./ mean_scaled_drag)), bare_roughness)
    displacement = ifelse.(has_urban, mean_displacement, bare_displacement)
    mean_height = ifelse.(has_urban, building_volume ./ built_fraction, 0)

    fields = map((roughness, displacement, urban_fraction, mean_height)) do data
        field = Field{Center, Center, Nothing}(grid)
        set!(field, FT.(data))
        return field
    end
    return NamedTuple{(:momentum_roughness_length, :zero_plane_displacement, :urban_fraction, :building_height)}(fields)
end
