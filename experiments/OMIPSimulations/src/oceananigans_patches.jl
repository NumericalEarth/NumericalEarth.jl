#####
##### Oceananigans monkey-patches and JLD2 split-file readers
#####
#
# Anything in this file is a deliberate override of upstream Oceananigans
# behaviour. Keep new patches here so they are easy to find, audit, and
# eventually upstream / remove. Each patch should carry a comment that says
# what it overrides, the upstream version it was written against, and why
# the override is needed.

using JLD2
using Oceananigans
using Oceananigans.Solvers
using Oceananigans.Solvers: ZDirection
using Oceananigans.Operators
using Oceananigans.Utils: worksize
using Oceananigans.Architectures: architecture
using Oceananigans.OutputReaders: SplitFilePath, InMemoryFTS, InMemory, time_indices, file_and_local_index
import Oceananigans.Fields: set!

#####
##### BatchedTridiagonalSolver: fix size -> worksize
#####

FTGU = TripolarGrid
FTG  = Union{ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:FTGU}, FTGU}

function Solvers.BatchedTridiagonalSolver(grid::FTG;
                                          lower_diagonal,
                                          diagonal,
                                          upper_diagonal,
                                          scratch = zeros(architecture(grid), eltype(grid), worksize(grid)...),
                                          parameters = nothing,
                                          tridiagonal_direction = ZDirection())

    return Solvers.BatchedTridiagonalSolver(lower_diagonal, diagonal, upper_diagonal,
                                            scratch, grid, parameters, tridiagonal_direction)
end

#####
##### Tupled closures with six or more members
#####
#
# Oceananigans `src/TurbulenceClosures/closure_tuples.jl` (rev `ss/for-omip`, ~line 63) unrolls the
# tupled-closure dispatch explicitly for one to five closures and handles longer tuples by induction.
# The induction step interpolates `$f` where it means `$outer_f`, and `f` is the `const f = Face()`
# defined earlier in the module, so a tuple of six or more closures lowers to `Face()(i, j, k, grid, ...)`
# and the tendency kernels die at GPU compile time with
# `unsupported call to an unknown function (call to jl_f_throw_methoderror)`.
#
# The OMIP closure is `omip_closure` (four members) plus the optional river-mouth and ice-melt vertical
# diffusivities, so enabling both crosses the boundary. The methods below re-state the induction step
# correctly; they are strictly more specific than the upstream `closures::Tuple` fallback, so no upstream
# method is overwritten. Remove once the typo is fixed upstream.

const outer_closure_functions = [:∂ⱼ_τ₁ⱼ, :∂ⱼ_τ₂ⱼ, :∂ⱼ_τ₃ⱼ, :∇_dot_qᶜ,
                                 :_ivd_upper_diagonal, :_ivd_lower_diagonal, :_implicit_linear_coefficient,
                                 :diffusive_flux_x, :diffusive_flux_y, :diffusive_flux_z,
                                 :viscous_flux_ux, :viscous_flux_uy, :viscous_flux_uz,
                                 :viscous_flux_vx, :viscous_flux_vy, :viscous_flux_vz,
                                 :viscous_flux_wx, :viscous_flux_wy, :viscous_flux_wz]

const inner_closure_functions = [:∂ⱼ_τ₁ⱼ, :∂ⱼ_τ₂ⱼ, :∂ⱼ_τ₃ⱼ, :∇_dot_qᶜ,
                                 :ivd_upper_diagonal, :ivd_lower_diagonal, :implicit_linear_coefficient,
                                 :diffusive_flux_x, :diffusive_flux_y, :diffusive_flux_z,
                                 :viscous_flux_ux, :viscous_flux_uy, :viscous_flux_uz,
                                 :viscous_flux_vx, :viscous_flux_vy, :viscous_flux_vz,
                                 :viscous_flux_wx, :viscous_flux_wy, :viscous_flux_wz]

const SixOrMoreClosures = Tuple{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Vararg{Any}}

for (outer_function, inner_function) in zip(outer_closure_functions, inner_closure_functions)
    @eval @inline Oceananigans.TurbulenceClosures.$outer_function(i, j, k, grid, closures::SixOrMoreClosures, Ks, args...) = (
              Oceananigans.TurbulenceClosures.$inner_function(i, j, k, grid, closures[1], Ks[1], args...)
            + Oceananigans.TurbulenceClosures.$outer_function(i, j, k, grid, closures[2:end], Ks[2:end], args...))
end

#####
##### ShavedCellBottom: land kept out of the corner reconstruction, and re-materialization on a grid with a different halo
#####
#
# Overrides `materialize_immersed_boundary(grid, ::ShavedCellBottom)` of Oceananigans `src/ImmersedBoundaries/shaved_cell_bottom.jl`
# (rev `ss/for-omip` 3d16d88, slug 5CEyo) for two defects (see `shaved_cell_halo_fix.md`):
# 1. The centre-to-corner average (line 101) includes land at its placeholder height (+100 from `ORCAGrid`, clamped to the
#    surface), and the corner-to-centre average (line 173) spreads it back: on eORCA1 Nz 70, 2 661 land columns become wet with
#    0.30–10 m of water and coastal columns shoal (Malacca 38 → 0.6 m). The run blows up at iteration 8. Here only wet centres
#    enter a corner, and the centre heights reach the corner-to-centre average, which keeps the dry columns dry.
# 2. Rebuilding a materialized boundary wrapped `corner_bottom_height` directly into a `Field` on the new grid (line 108), which
#    fails when `with_halo` extends the halo of the `SplitExplicitFreeSurface` grid (Hy = 218). Here corners are copied through
#    their interior.
# The method is strictly more specific than the upstream one (`grid::AbstractGrid` against an untyped `grid`), so no upstream
# method is overwritten. Remove once fixed upstream.

using KernelAbstractions: @index, @kernel
using Oceananigans.Grids: AbstractGrid, rnode, topology
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: ShavedCellBottom, staggered_bottom_parameters, set_bottom_height!, x_index, y_index,
                                       _clamp_bottom_height_to_domain!, _average_corners_to_centers!,
                                       _compute_shaved_face_bottom_heights!, _average_faces_to_centers!

@kernel function _interpolate_wet_bottom_height_to_corners!(corner_field, center_field, grid)
    i, j = @index(Global, NTuple)
    iᵂ = x_index(i, grid, -1)
    jˢ = y_index(j, grid, -1)
    rᵗ = rnode(i, j, grid.Nz+1, grid, Center(), Center(), Face())

    @inbounds begin
        r₁ = center_field[iᵂ, jˢ, 1]
        r₂ = center_field[i,  jˢ, 1]
        r₃ = center_field[iᵂ, j,  1]
        r₄ = center_field[i,  j,  1]
    end

    w₁ = r₁ < rᵗ
    w₂ = r₂ < rᵗ
    w₃ = r₃ < rᵗ
    w₄ = r₄ < rᵗ
    n = w₁ + w₂ + w₃ + w₄
    Σr = ifelse(w₁, r₁, zero(rᵗ)) + ifelse(w₂, r₂, zero(rᵗ)) + ifelse(w₃, r₃, zero(rᵗ)) + ifelse(w₄, r₄, zero(rᵗ))

    @inbounds corner_field[i, j, 1] = ifelse(n > 0, Σr / max(n, 1), rᵗ)
end

# The centre heights that define the wet/dry mask: the input bathymetry on first materialization, the materialized
# centre heights (dry columns sit at the surface) when a grid is rebuilt with another halo.
function mask_bottom_height(ib::ShavedCellBottom, grid)
    center_field = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(center_field, ib.bottom_height)
    fill_halo_regions!(center_field)
    return center_field
end

# Corner heights given by the user (a `Field` at `(Face, Face)` or a function of position) carry no centre mask.
const CornerShavedCellBottom = Union{ShavedCellBottom{<:Field{Face, Face, Nothing}}, ShavedCellBottom{<:Function}}

function mask_bottom_height(ib::CornerShavedCellBottom, grid)
    center_field = Field{Center, Center, Nothing}(grid)
    set!(center_field, -Inf)
    return center_field
end

wet_corner_bottom_height!(corner_field, grid, ib::CornerShavedCellBottom, center_field, parameters) =
    set_bottom_height!(corner_field, ib.bottom_height)

wet_corner_bottom_height!(corner_field, grid, ib::ShavedCellBottom{<:Any, <:AbstractArray}, center_field, parameters) =
    set_bottom_height!(corner_field, ib.corner_bottom_height)

wet_corner_bottom_height!(corner_field, grid, ib::ShavedCellBottom, center_field, parameters) =
    launch!(architecture(grid), grid, parameters, _interpolate_wet_bottom_height_to_corners!, corner_field, center_field, grid)

function Oceananigans.ImmersedBoundaries.materialize_immersed_boundary(grid::AbstractGrid, ib::ShavedCellBottom)
    FT = eltype(grid)
    ϵ = max(convert(FT, ib.minimum_fractional_cell_height), sqrt(eps(FT)))
    arch = architecture(grid)
    parameters = staggered_bottom_parameters(grid)

    center_field = mask_bottom_height(ib, grid)

    corner_field = Field{Face, Face, Nothing}(grid)
    wet_corner_bottom_height!(corner_field, grid, ib, center_field, parameters)
    launch!(arch, grid, parameters, _clamp_bottom_height_to_domain!, corner_field, grid)
    fill_halo_regions!(corner_field)

    bottom_field = Field{Center, Center, Nothing}(grid)
    launch!(arch, grid, :xy, _average_corners_to_centers!, bottom_field, corner_field, center_field, grid, ϵ)
    fill_halo_regions!(bottom_field)

    west_field = Field{Face, Center, Nothing}(grid)
    south_field = Field{Center, Face, Nothing}(grid)
    previous_bottom_field = Field{Center, Center, Nothing}(grid)

    compute_ib = ShavedCellBottom(bottom_field, nothing, nothing, nothing, ϵ)
    TX, TY, _ = topology(grid)
    wˣ = TX === Flat && TY !== Flat ? 0 : 1
    wʸ = TY === Flat && TX !== Flat ? 0 : 1

    # Immersing a cell can close the faces of its neighbors, so faces and cells are recomputed until no level changes.
    converged = false
    while !converged
        set!(previous_bottom_field, bottom_field)
        launch!(arch, grid, parameters, _compute_shaved_face_bottom_heights!, west_field, south_field, corner_field, grid, compute_ib)
        fill_halo_regions!(west_field)
        fill_halo_regions!(south_field)
        launch!(arch, grid, :xy, _average_faces_to_centers!, bottom_field, west_field, south_field, grid, compute_ib, wˣ, wʸ)
        fill_halo_regions!(bottom_field)
        converged = maximum(abs, bottom_field - previous_bottom_field) == 0
    end

    return ShavedCellBottom(bottom_field.data, west_field.data, south_field.data, corner_field.data, ϵ)
end

#####
##### Materialized buoyancy gradients: horizontal components change sign across the tripolar fold
#####
#
# Overrides `BuoyancyForce(grid, formulation; materialize_gradients)` of Oceananigans
# `src/BuoyancyFormulations/buoyancy_force.jl` (rev `ss/for-omip`, slug pJPsy, line 54) on tripolar grids.
# Upstream stores ∂xᵣb and ∂yᵣb in `XFaceField`/`YFaceField`s with the auxiliary default north boundary condition,
# a zipper with sign +1. Horizontal derivatives reverse across the fold, like u and v, so the halo fill makes ∂yᵣb
# on the fold face equal between partner columns i and Nx+1-i instead of opposite. The triad cross-term
# (κ_symmetric − κ_skew) Sy ∂z c then passes two different fluxes through the one fold face and loses tracer
# whenever κ_skew ≠ κ_symmetric (−8.4 Gt/yr of salt at κ_skew = 500, κ_symmetric = 800). The method is strictly
# more specific than the upstream one (`grid::FTG` against an untyped `grid`). Remove once fixed upstream.

using Oceananigans.BuoyancyFormulations: BuoyancyForce, AbstractBuoyancyFormulation
using Oceananigans.Grids: NegativeZDirection, validate_unit_vector
using Oceananigans.OrthogonalSphericalShellGrids: north_fold_boundary_condition

function Oceananigans.BuoyancyFormulations.BuoyancyForce(grid::FTG, formulation::AbstractBuoyancyFormulation;
                                                          gravity_unit_vector = NegativeZDirection(),
                                                          materialize_gradients = false)

    gravity_unit_vector = validate_unit_vector(gravity_unit_vector)
    materialize_gradients || return BuoyancyForce(formulation, gravity_unit_vector, nothing)

    sign_flipping_boundary_conditions(loc) = FieldBoundaryConditions(grid, loc; north = north_fold_boundary_condition(grid)(-1))

    ∂xᵣ_b = XFaceField(grid; boundary_conditions = sign_flipping_boundary_conditions((Face(), Center(), Center())))
    ∂yᵣ_b = YFaceField(grid; boundary_conditions = sign_flipping_boundary_conditions((Center(), Face(), Center())))
    ∂z_b  = ZFaceField(grid)

    return BuoyancyForce(formulation, gravity_unit_vector, (; ∂xᵣ_b, ∂yᵣ_b, ∂z_b))
end

#####
##### InMemory FieldTimeSeries split-file (`..._partN.jld2`) support
#####
#
# Oceananigans 0.107.x bug (`OutputReaders/field_time_series.jl`, ~line
# 924-930): the inner `FieldTimeSeries` constructor only builds a
# `SplitFilePath` when `backend isa OnDisk`. With an `InMemory` backend on
# split output (`..._part1.jld2`, `..._part2.jld2`, …), `fts.path` collapses
# to a single part file, producing "No data found for time …" warnings (and
# stale/zero data) on later snapshot reads.
#
# The patch below overrides the `FieldTimeSeries(path, name; …)` constructor so that an `InMemory`
# backend whose `fts.path` is a single file is detected as a split set and re-wrapped with the
# correct `SplitFilePath`. (`set!(::InMemoryFTS, ::SplitFilePath)` is now provided by Oceananigans,
# so the former local definition was removed.)
#
# The helpers (`jld2_output_part_paths`, etc.) are factored out so this file
# and `scripts/visualize/common.jl` (and any future call site) can share a
# single source of truth for reading split Oceananigans JLD2 outputs.

"""
    jld2_output_part_paths(path)

Resolve `path` (Oceananigans-style JLD2 stem, with or without `.jld2`) to
the list of on-disk part files that hold its time series: either a
single-element vector `[abspath(path)]` if that file exists, or sorted
`…_partN.jld2` paths in `dirname(path)`. Returns an empty vector if nothing
matches (same convention as split detection for `FieldTimeSeries`).
"""
function jld2_output_part_paths(path::AbstractString)
    ap = abspath(path)
    isfile(ap) && return String[ap]
    base = endswith(ap, ".jld2") ? ap[1:end-5] : ap
    dir  = isempty(dirname(base)) ? "." : dirname(base)
    pat  = Regex("^" * Base.escape_string(basename(base)) * "_part(\\d+)\\.jld2\$")
    files = filter(f -> occursin(pat, f), readdir(dir))
    isempty(files) && return String[]
    sort!(files, by = f -> parse(Int, match(pat, f).captures[1]))
    return String[joinpath(dir, f) for f in files]
end

"""
    with_jld2(fn, path; reader_kw = NamedTuple())

Open `path` with JLD2, run `fn(file)`, guarantee close.
"""
function with_jld2(fn, path::AbstractString; reader_kw = NamedTuple())
    jf = JLD2.jldopen(path; reader_kw...)
    try
        return fn(jf)
    finally
        close(jf)
    end
end

"""
    jld2_parts(path)

Resolve `path` to its on-disk part files (single or split), erroring if
nothing matches. Used by every metadata reader below.
"""
function jld2_parts(path::AbstractString)
    parts = jld2_output_part_paths(path)
    isempty(parts) && error("No JLD2 output at path '$path' (single file or split parts).")
    return parts
end

# Per-part memo: `fn(file)` runs at most once per absolute part path, its
# return value cached in `memo`. Each derived diagnostic that validates a
# disk cache hits this memo and is therefore O(1) after the first read.
const JLD2_NT_PER_PART    = Dict{String, Int}()
const JLD2_TIMES_PER_PART = Dict{String, Vector{Float64}}()

memoize_jld2_part(fn, memo, path::AbstractString; reader_kw = NamedTuple()) =
    get!(memo, path) do
        with_jld2(fn, path; reader_kw)
    end

"""
    total_jld2_timeseries_snapshot_count(path; reader_kw = NamedTuple())

Total number of time indices for an Oceananigans JLD2 output stem `path`
(single file or split `_partN.jld2` parts), summed across parts. Reads
JLD2 metadata only — no `FieldTimeSeries` is built. Per-part counts are
memoized for the session.
"""
function total_jld2_timeseries_snapshot_count(path::AbstractString; reader_kw = NamedTuple())
    n = 0
    for p in jld2_parts(path)
        n += memoize_jld2_part(JLD2_NT_PER_PART, p; reader_kw) do jf
            length(keys(jf["timeseries/t"]))
        end
    end
    return n
end

"""
    total_jld2_timeseries_times(path; reader_kw = NamedTuple())

Concatenated time-coordinate vector for an Oceananigans JLD2 output stem
`path`. Reads `timeseries/t/<iter>` per part, sorted by iteration, and
concatenates across parts. No `FieldTimeSeries` construction. Per-part
time vectors are memoized for the session.
"""
function total_jld2_timeseries_times(path::AbstractString; reader_kw = NamedTuple())
    times = Float64[]
    for p in jld2_parts(path)
        ts = memoize_jld2_part(JLD2_TIMES_PER_PART, p; reader_kw) do jf
            iterations = sort!(parse.(Int, collect(keys(jf["timeseries/t"]))))
            return Float64[jf["timeseries/t/$it"] for it in iterations]
        end
        append!(times, ts)
    end
    return times
end

"""
    last_jld2_timeseries_time(path; reader_kw = NamedTuple())

Latest snapshot time for an Oceananigans JLD2 output stem `path`. Opens
only the last part file and reads only the highest-iteration entry of
`timeseries/t`. Avoids the per-iteration JLD2 lookup loop in
`total_jld2_timeseries_times`, which is O(N) and slow on long runs over
network filesystems.
"""
function last_jld2_timeseries_time(path::AbstractString; reader_kw = NamedTuple())
    last_part = last(jld2_parts(path))
    return with_jld2(last_part; reader_kw) do jf
        ks = keys(jf["timeseries/t"])
        max_iter = maximum(parse(Int, k) for k in ks if !isnothing(tryparse(Int, k)))
        return Float64(jf["timeseries/t/$max_iter"])
    end
end

"""
    total_jld2_serialized_grid(path, name; reader_kw = NamedTuple())

Read the grid serialized inside an Oceananigans JLD2 output stem `path`
for variable `name`, without instantiating a `FieldTimeSeries`. Defers to
`Oceananigans.OutputReaders.load_serialized_grid`, which handles
single-grid (`serialized/grid`) and multi-grid output. The grid is
invariant across split parts, so only the first part is opened.
"""
total_jld2_serialized_grid(path::AbstractString, name::AbstractString; reader_kw = NamedTuple()) =
    with_jld2(first(jld2_parts(path)); reader_kw) do jf
        Oceananigans.OutputReaders.load_serialized_grid(jf, name)
    end

"""
    total_jld2_scalar_timeseries(path, name; reader_kw = NamedTuple())

Read a scalar (1×1×1) variable's full time series for the JLD2 output stem
`path` directly from `timeseries/<name>/<iter>` metadata, sorted by
iteration and concatenated across split parts. Avoids building a
`FieldTimeSeries` and the per-snapshot buffer slides that imposes — ~100×
faster than `[interior(fts[n])[1] for n in ...]` for long time series of
`tosga`/`soga`-style global means.
"""
function total_jld2_scalar_timeseries(path::AbstractString, name::AbstractString;
                                      reader_kw = NamedTuple())
    values = Float64[]
    for p in jld2_parts(path)
        chunk = with_jld2(p; reader_kw) do jf
            ks = collect(keys(jf["timeseries/$name"]))
            iterations = sort!([parse(Int, k) for k in ks if !isnothing(tryparse(Int, k))])
            return Float64[jf["timeseries/$name/$it"][1] for it in iterations]
        end
        append!(values, chunk)
    end
    return values
end

function detect_split_file_path(path::AbstractString, reader_kw)
    isfile(path) && return nothing
    parts = jld2_output_part_paths(path)
    isempty(parts) && return nothing
    nper = Int[total_jld2_timeseries_snapshot_count(p; reader_kw) for p in parts]
    return SplitFilePath(parts, cumsum(nper))
end

location_types(::Oceananigans.OutputReaders.FieldTimeSeries{LX, LY, LZ}) where {LX, LY, LZ} = (LX, LY, LZ)

rebuild_backend_with_path(backend, new_path) = backend

# function rebuild_backend_with_path(backend::Prefetched, new_path)
#     old_buf = getfield(backend, :buffer_fts)
#     BLX, BLY, BLZ = location_types(old_buf)
#     new_buf = Oceananigans.OutputReaders.FieldTimeSeries{BLX, BLY, BLZ}(
#         old_buf.data, old_buf.grid, old_buf.backend, old_buf.boundary_conditions,
#         old_buf.indices, old_buf.times, new_path, old_buf.name,
#         old_buf.time_indexing, old_buf.reader_kw)
#     return Prefetched(backend.base_backend, backend.pending, new_buf, backend.next_start)
# end

function rebuild_fts_with_path(fts, new_path)
    LX, LY, LZ = location_types(fts)
    new_backend = rebuild_backend_with_path(fts.backend, new_path)
    return Oceananigans.OutputReaders.FieldTimeSeries{LX, LY, LZ}(
        fts.data, fts.grid, new_backend, fts.boundary_conditions,
        fts.indices, fts.times, new_path, fts.name,
        fts.time_indexing, fts.reader_kw)
end

# Patch 1 removed: `set!(::InMemoryFTS, ::SplitFilePath)` is now provided by Oceananigans
# (OutputReaders/set_field_time_series.jl) with identical per-part iteration. Re-defining it here
# overwrites the upstream method, which is an error during precompilation.

# Patch 2: detect split sets when the user passes a single stem path with an
# `InMemory` backend, and rewrap the FTS so its `path` is a `SplitFilePath`.
function Oceananigans.OutputReaders.FieldTimeSeries(path::String, name::String;
                                                    backend = InMemory(),
                                                    reader_kw = NamedTuple(),
                                                    kwargs...)
    fts = invoke(Oceananigans.OutputReaders.FieldTimeSeries,
                 Tuple{String, Vararg{Any}},
                 path, name; backend, reader_kw, kwargs...)
    if backend isa InMemory && !(fts.path isa SplitFilePath)
        sfp = detect_split_file_path(path, reader_kw)
        sfp === nothing || (fts = rebuild_fts_with_path(fts, sfp))
    end
    return fts
end
