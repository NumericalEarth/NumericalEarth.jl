using Oceananigans.Architectures: CPU
using Oceananigans.Fields: interior, set!
using Oceananigans.Grids: LatitudeLongitudeGrid, Bounded, Flat, λnodes, φnodes
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ

#####
##### On-grid evaluation of the urban roughness closures: (λᵖ, h) → (ℓᵐ, d) fields.
#####

@kernel function _compute_urban_aerodynamic_roughness!(ℓᵐ, d, closure, plan_area_index, mean_building_height, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    λᵖ = convert(FT, property_value(plan_area_index, i, j))
    h  = convert(FT, property_value(mean_building_height, i, j))
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
reduces to a bare-soil roughness.
"""
function urban_roughness(h::AbstractField, λᵖ; closure = MorphometricRoughness(eltype(h.grid)))
    grid = h.grid
    ℓᵐ = Field{Center, Center, Nothing}(grid)
    d  = Field{Center, Center, Nothing}(grid)
    compute_aerodynamic_roughness!(ℓᵐ, d, closure, (; plan_area_index = λᵖ, mean_building_height = h), grid)
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
    λᵖ   = convert(FT, property_value(plan_area_index, i, j))
    h    = convert(FT, property_value(mean_building_height, i, j))
    σʰ   = convert(FT, property_value(building_height_deviation, i, j))
    hᵐᵃˣ = convert(FT, property_value(maximum_building_height, i, j))
    λᶠ   = convert(FT, property_value(frontal_area, i, j))
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
result reduces to a bare-soil roughness.
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
##### closure → built-area-weighted reduction to the grid.
#####

"""
$(TYPEDSIGNATURES)

Per-cell building morphometry on `grid` from a building `dataset`, as a NamedTuple of `Field`s
named after the closure inputs: `plan_area_index` and `mean_building_height` (m), and, where
the dataset measures them, `building_height_deviation` (m), `maximum_building_height` (m) and
`frontal_area_index`. Dataset modules add methods for their products.
"""
function building_morphometry end

# Sums over the `rx × ry` blocks of a lattice array.
block_sum(a, rx, ry) =
    dropdims(sum(reshape(a, rx, size(a, 1) ÷ rx, ry, size(a, 2) ÷ ry); dims = (1, 3)); dims = (1, 3))

"""
$(TYPEDSIGNATURES)

Momentum roughness length and zero-plane displacement (m) of the built-up surface on `grid`
from a building `dataset`, as the NamedTuple of `Field`s
`(; momentum_roughness_length, zero_plane_displacement, urban_fraction, building_height)`.

The closure is evaluated on a lattice of cells about `neighborhood` (m) wide subdividing each
grid cell, on the morphometry that [`building_morphometry`](@ref) supplies there. Each grid cell
then takes the built-area-weighted log-mean roughness length, mean displacement and mean
height of its urban lattice cells, those with plan-area index at least the closure's
`minimum_built_fraction`; `urban_fraction` is their share of the cell, and a cell without urban
lattice cells takes the closure's bare-soil limit. A grid cell narrower than `neighborhood` is
its own lattice cell. Remaining keyword arguments pass to `building_morphometry`.
"""
function urban_roughness(grid, dataset; closure = MorphometricRoughness(eltype(grid)),
                         neighborhood = 1000, kw...)
    FT = eltype(grid)
    Nx, Ny, _ = size(grid)
    center = (Nx ÷ 2 + 1, Ny ÷ 2 + 1, 1)
    rx = max(1, round(Int, Δxᶜᶜᶜ(center..., grid) / neighborhood))
    ry = max(1, round(Int, Δyᶜᶜᶜ(center..., grid) / neighborhood))

    lattice_faces(faces, r) = [[faces[i] + (faces[i + 1] - faces[i]) * k / r
                                for i in 1:length(faces) - 1 for k in 0:r - 1]; faces[end]]
    lattice = LatitudeLongitudeGrid(CPU(), FT; size = (rx * Nx, ry * Ny),
                                    longitude = lattice_faces(λnodes(grid, Face()), rx),
                                    latitude  = lattice_faces(φnodes(grid, Face()), ry),
                                    topology = (Bounded, Bounded, Flat))

    properties = building_morphometry(lattice, dataset; kw...)
    measured = (:plan_area_index, :mean_building_height,
                :building_height_deviation, :maximum_building_height, :frontal_area_index)
    closure_inputs = hasproperty(properties, :frontal_area_index) ? properties[measured] : properties[measured[1:2]]
    ℓᵐ = Field{Center, Center, Nothing}(lattice)
    d  = Field{Center, Center, Nothing}(lattice)
    compute_aerodynamic_roughness!(ℓᵐ, d, closure, closure_inputs, lattice)

    # Built-area weights of the urban lattice cells
    λᵖ = interior(properties.plan_area_index, :, :, 1)
    w  = ifelse.(λᵖ .>= closure.minimum_built_fraction, λᵖ, 0)
    Σw = block_sum(w, rx, ry)
    urban = Σw .> 0
    bare_roughness, bare_displacement = aerodynamic_parameters(closure, 0, 0)

    roughness      = ifelse.(urban, exp.(block_sum(w .* log.(interior(ℓᵐ, :, :, 1)), rx, ry) ./ Σw), bare_roughness)
    displacement   = ifelse.(urban, block_sum(w .* interior(d, :, :, 1), rx, ry) ./ Σw, bare_displacement)
    mean_height    = ifelse.(urban, block_sum(w .* interior(properties.mean_building_height, :, :, 1), rx, ry) ./ Σw, 0)
    urban_fraction = block_sum(w .> 0, rx, ry) ./ (rx * ry)

    fields = map((roughness, displacement, urban_fraction, mean_height)) do data
        field = Field{Center, Center, Nothing}(grid)
        set!(field, data)
        return field
    end
    return NamedTuple{(:momentum_roughness_length, :zero_plane_displacement, :urban_fraction, :building_height)}(fields)
end
