using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.Fields: interior, set!, regrid!
using Oceananigans.Grids: LatitudeLongitudeGrid, Bounded, Flat, λnodes, φnodes

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
under `closure` (default [`MorphometricRoughness`](@ref)). Where `λᵖ → 0` the result
reduces to a bare-soil roughness, and cells of invalid morphometry are `NaN`; fill them
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
    FT = eltype(grid)
    longitude_center = (longitude_faces[i] + longitude_faces[i + 1]) / 2
    latitude_center = (latitude_faces[j] + latitude_faces[j + 1]) / 2
    Δλ = longitude_faces[i + 1] - longitude_faces[i]
    Δφ = latitude_faces[j + 1] - latitude_faces[j]
    nx = max(0, round(Int, (neighborhood / (grid.radius * cosd(latitude_center) * deg2rad(Δλ)) - 1) / 2))
    ny = max(0, round(Int, (neighborhood / (grid.radius * deg2rad(Δφ)) - 1) / 2))
    west  = ifelse(nx == 0, longitude_faces[i], longitude_center - (nx + 1//2) * Δλ)
    east  = ifelse(nx == 0, longitude_faces[i + 1], longitude_center + (nx + 1//2) * Δλ)
    south = max(latitude_faces[1], ifelse(ny == 0, latitude_faces[j], latitude_center - (ny + 1//2) * Δφ))
    north = min(latitude_faces[end], ifelse(ny == 0, latitude_faces[j + 1], latitude_center + (ny + 1//2) * Δφ))
    i₁ = max(1, searchsortedlast(longitude_faces, west))
    i₂ = min(size(grid, 1), searchsortedfirst(longitude_faces, east) - 1)
    j₁ = max(1, searchsortedlast(latitude_faces, south))
    j₂ = min(size(grid, 2), searchsortedfirst(latitude_faces, north) - 1)
    ground_area = built_area = volume = second_moment = maximum_height = frontal_area = zero(FT)
    for b in j₁:j₂, a in i₁:i₂
        overlap_south = max(south, latitude_faces[b])
        overlap_north = min(north, latitude_faces[b + 1])
        # sin(north) - sin(south) as 2 cos(midpoint) sin(half-width), which does not cancel in Float32 for thin cells
        area = max(0, min(east, longitude_faces[a + 1]) - max(west, longitude_faces[a])) *
               2cosd((overlap_north + overlap_south) / 2) * max(0, sind((overlap_north - overlap_south) / 2))
        λᵖ = stateindex(properties.plan_area_index, a, b, 1)
        h = stateindex(properties.mean_building_height, a, b, 1)
        σʰ = stateindex(properties.building_height_deviation, a, b, 1)
        hᵐᵃˣ = stateindex(properties.maximum_building_height, a, b, 1)
        λᶠ = stateindex(properties.frontal_area_index, a, b, 1)
        λᶠ = ifelse((λᵖ == 0) & isnan(λᶠ), 0, λᶠ)
        built = area * λᵖ
        ground_area += area
        built_area += built
        volume += ifelse(built > 0, built * h, 0)
        second_moment += ifelse(built > 0, built * (h^2 + σʰ^2), 0)
        maximum_height = max(maximum_height, ifelse(built > 0, hᵐᵃˣ, 0))
        frontal_area += ifelse(area > 0, area * λᶠ, 0)
    end
    h̄ = ifelse(built_area > 0, volume / built_area, 0)
    variance = ifelse(built_area > 0, second_moment / built_area - h̄^2, 0)
    @inbounds begin
        pooled[1][i, j, 1] = built_area / ground_area
        pooled[2][i, j, 1] = h̄
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
    launch!(CPU(), grid, :xy, _neighborhood_morphometry!, pooled, grid, longitude_faces, latitude_faces,
            inputs, neighborhood)
    morphometry = (; plan_area_index = pooled[1], mean_building_height = pooled[2])
    measured || return morphometry
    return (; morphometry..., building_height_deviation = pooled[3],
              maximum_building_height = pooled[4], frontal_area_index = pooled[5])
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length and zero-plane displacement (m) of the built-up surface on `grid`
from a building `dataset`, as the NamedTuple of `Field`s
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
function urban_roughness(grid, dataset; surface_layer_height, closure = MorphometricRoughness(eltype(grid)),
                         neighborhood = 1000, kw...)
    FT = eltype(grid)
    Nx, Ny, _ = size(grid)
    cpu_grid = on_architecture(CPU(), grid)
    longitude_faces = collect(λnodes(cpu_grid, Face()))
    latitude_faces = collect(φnodes(cpu_grid, Face()))
    reduction_grid = LatitudeLongitudeGrid(CPU(), FT; size = (Nx, Ny),
                                           longitude = longitude_faces, latitude = latitude_faces, radius = cpu_grid.radius,
                                           topology = (Bounded, Bounded, Flat))
    longitude_scale = cpu_grid.radius * deg2rad(cosd(clamp(0, first(latitude_faces), last(latitude_faces))))
    latitude_scale = cpu_grid.radius * deg2rad(1)
    function subdivide(faces, scale)
        subdivisions = max.(1, round.(Int, diff(faces) .* scale ./ neighborhood))
        return [[faces[i] + (faces[i + 1] - faces[i]) * k / subdivisions[i]
                 for i in eachindex(subdivisions) for k in 0:subdivisions[i]-1]; faces[end]]
    end
    longitude_lattice = subdivide(longitude_faces, longitude_scale)
    latitude_lattice = subdivide(latitude_faces, latitude_scale)
    lattice = LatitudeLongitudeGrid(CPU(), FT; size = (length(longitude_lattice) - 1, length(latitude_lattice) - 1),
                                    longitude = longitude_lattice, latitude = latitude_lattice, radius = cpu_grid.radius,
                                    topology = (Bounded, Bounded, Flat))

    properties = neighborhood_morphometry(building_morphometry(lattice, dataset; kw...), lattice,
                                          longitude_lattice, latitude_lattice, neighborhood)
    lattice_roughness    = Field{Center, Center, Nothing}(lattice)
    lattice_displacement = Field{Center, Center, Nothing}(lattice)
    compute_aerodynamic_roughness!(lattice_roughness, lattice_displacement, closure, properties, lattice)

    # Neutral drag coefficients over ϰ² of the urban neighborhoods at the surface-layer height
    z  = surface_layer_height
    ℓᵐ = interior(lattice_roughness, :, :, 1)
    d  = interior(lattice_displacement, :, :, 1)
    λᵖ = interior(properties.plan_area_index, :, :, 1)
    h = interior(properties.mean_building_height, :, :, 1)
    urban = λᵖ .>= closure.minimum_built_fraction
    any(urban .& roughness_gap.(ℓᵐ, d)) &&
        throw(ArgumentError("urban neighborhoods must have finite, positive roughness lengths and finite displacements"))
    any(urban .& (z .<= d .+ ℓᵐ)) &&
        throw(ArgumentError("surface_layer_height = $z m must exceed d + ℓᵐ in every urban neighborhood"))
    scaled_drag = inv.(log.((z .- d) ./ ℓᵐ)) .^ 2
    scaled_drag .= ifelse.(urban, scaled_drag, 0)
    contributions = (urban, scaled_drag, ifelse.(urban, scaled_drag .* d, zero(eltype(scaled_drag))),
                     ifelse.(urban, λᵖ, zero(FT)), ifelse.(urban, λᵖ .* h, zero(FT)))
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
    d̄ = scaled_drag_moment ./ mean_scaled_drag
    has_urban = urban_fraction .> 0
    bare_roughness, bare_displacement = aerodynamic_parameters(closure, 0, 0)
    roughness = ifelse.(has_urban, (z .- d̄) .* exp.(-sqrt.(urban_fraction ./ mean_scaled_drag)), bare_roughness)
    displacement = ifelse.(has_urban, d̄, bare_displacement)
    mean_height = ifelse.(has_urban, building_volume ./ built_fraction, zero(FT))

    fields = map((roughness, displacement, urban_fraction, mean_height)) do data
        field = Field{Center, Center, Nothing}(grid)
        set!(field, FT.(data))
        return field
    end
    return NamedTuple{(:momentum_roughness_length, :zero_plane_displacement, :urban_fraction, :building_height)}(fields)
end
