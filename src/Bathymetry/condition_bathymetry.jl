#####
##### Offline bathymetry conditioning
#####

"""
    BathymetryPatchRecord(index, old_depth, new_depth, delta, reason, iteration)

One cell change made by [`smooth_flagged_bathymetry`](@ref).
"""
Base.@kwdef struct BathymetryPatchRecord
    index
    old_depth
    new_depth
    delta
    reason::Symbol = :conditioned
    iteration::Int = 1
end

"""
    BathymetryPatchLog(records)

Structured log of bathymetry changes. This is intentionally plain Julia data so
offline preprocessing scripts can write it to JLD2, CSV, JSON, or NetCDF sidecars.
"""
struct BathymetryPatchLog
    records::Vector{BathymetryPatchRecord}
end

BathymetryPatchLog() = BathymetryPatchLog(BathymetryPatchRecord[])

Base.length(log::BathymetryPatchLog) = length(log.records)
Base.iterate(log::BathymetryPatchLog, state...) = iterate(log.records, state...)
Base.getindex(log::BathymetryPatchLog, i::Integer) = log.records[i]

"""
    BathymetryPatchSummary

Summary returned by [`summarize_bathymetry_patch`](@ref).
"""
Base.@kwdef struct BathymetryPatchSummary
    changed_cells::Int
    max_abs_depth_change
    mean_abs_depth_change
    total_depth_change
    area_weighted_depth_change = nothing
end

_isvalid_depth(h) = !ismissing(h) && h isa Number && isfinite(h)
_iswet(h) = _isvalid_depth(h) && h > 0
_iswet(h, wetmask, index) = !isnothing(wetmask) ? Bool(wetmask[index]) && _isvalid_depth(h) : _iswet(h)

function _neighbor_offsets(ndims_h, neighbors)
    neighbors === :cardinal || throw(ArgumentError("Only neighbors = :cardinal is supported."))
    offsets = CartesianIndex[]
    for dim in 1:ndims_h
        plus = ntuple(d -> d == dim ? 1 : 0, ndims_h)
        minus = ntuple(d -> d == dim ? -1 : 0, ndims_h)
        push!(offsets, CartesianIndex(plus))
        push!(offsets, CartesianIndex(minus))
    end
    return offsets
end

_inside(index::CartesianIndex, axes_h) = all(dim -> index[dim] in axes_h[dim], 1:length(axes_h))

_as_wetmask(grid_or_mask, h) = grid_or_mask isa AbstractArray ? grid_or_mask : nothing

"""
    compute_bathymetry_roughness(h, grid_or_mask=nothing; neighbors=:cardinal)

Compute an rx0-like roughness metric for bathymetric depth `h`,
``abs(h_i - h_j) / (h_i + h_j)``, and return the maximum neighboring wet-cell
roughness for each cell.

This offline preprocessing utility ignores land, `missing`, `NaN`, and `Inf`
cells and avoids divide-by-zero. If `grid_or_mask` is a Boolean array, it is
used as a wet mask. Only `neighbors = :cardinal` is currently supported.
"""
function compute_bathymetry_roughness(h::AbstractArray, grid_or_mask=nothing; neighbors=:cardinal)
    wetmask = _as_wetmask(grid_or_mask, h)
    roughness = zeros(float(nonmissingtype(eltype(h))), axes(h))
    offsets = _neighbor_offsets(ndims(h), neighbors)
    axes_h = axes(h)

    for index in CartesianIndices(h)
        hi = h[index]
        _iswet(hi, wetmask, index) || continue

        max_roughness = zero(eltype(roughness))
        for offset in offsets
            neighbor = index + offset
            _inside(neighbor, axes_h) || continue

            hj = h[neighbor]
            _iswet(hj, wetmask, neighbor) || continue

            denominator = hi + hj
            denominator > 0 || continue
            max_roughness = max(max_roughness, abs(hi - hj) / denominator)
        end

        roughness[index] = max_roughness
    end

    return roughness
end

function _flag_diagnostic!(mask, diagnostic, threshold, wetmask)
    isnothing(diagnostic) && return mask
    size(diagnostic) == size(mask) || throw(DimensionMismatch("diagnostic size $(size(diagnostic)) does not match mask size $(size(mask))"))

    for index in CartesianIndices(mask)
        !isnothing(wetmask) && !Bool(wetmask[index]) && continue
        value = diagnostic[index]
        unstable = ismissing(value) || !(value isa Number) || !isfinite(value) || value > threshold
        mask[index] |= unstable
    end

    return mask
end

"""
    flag_unstable_columns(; cfl=nothing, u=nothing, v=nothing, h=nothing, eta=nothing,
                            wetmask=nothing, cfl_threshold=0.8,
                            velocity_threshold=nothing, min_depth=nothing,
                            roughness=nothing, roughness_threshold=nothing)

Return a Boolean mask of wet columns that should be considered for offline
bathymetry repair. CFL values above `cfl_threshold`, non-finite diagnostics,
excessive velocity magnitude, shallow depths below `min_depth`, and roughness
above `roughness_threshold` can all be flagged.

This function is deterministic and array-only. It does not run or inspect a
simulation.
"""
function flag_unstable_columns(; cfl=nothing, u=nothing, v=nothing, h=nothing, eta=nothing,
                                 wetmask=nothing, cfl_threshold=0.8,
                                 velocity_threshold=nothing, min_depth=nothing,
                                 roughness=nothing, roughness_threshold=nothing)
    reference = nothing
    for candidate in (cfl, u, v, h, eta, roughness, wetmask)
        if !isnothing(candidate)
            reference = candidate
            break
        end
    end
    isnothing(reference) && throw(ArgumentError("At least one diagnostic, h, roughness, or wetmask must be provided."))

    mask = falses(size(reference))
    !isnothing(wetmask) && size(wetmask) != size(mask) && throw(DimensionMismatch("wetmask size $(size(wetmask)) does not match diagnostic size $(size(mask))"))

    _flag_diagnostic!(mask, cfl, cfl_threshold, wetmask)
    _flag_diagnostic!(mask, eta, Inf, wetmask)

    if !isnothing(velocity_threshold)
        isnothing(u) && isnothing(v) && throw(ArgumentError("u or v must be provided when velocity_threshold is set."))
        velocity_reference = something(u, v)
        size(velocity_reference) == size(mask) || throw(DimensionMismatch("velocity size $(size(velocity_reference)) does not match mask size $(size(mask))"))

        for index in CartesianIndices(mask)
            !isnothing(wetmask) && !Bool(wetmask[index]) && continue
            ui = isnothing(u) ? zero(velocity_threshold) : u[index]
            vi = isnothing(v) ? zero(velocity_threshold) : v[index]
            unstable = ismissing(ui) || ismissing(vi) ||
                       !(ui isa Number) || !(vi isa Number) ||
                       !isfinite(ui) || !isfinite(vi) ||
                       sqrt(abs2(ui) + abs2(vi)) > velocity_threshold
            mask[index] |= unstable
        end
    else
        _flag_diagnostic!(mask, u, Inf, wetmask)
        _flag_diagnostic!(mask, v, Inf, wetmask)
    end

    if !isnothing(min_depth)
        isnothing(h) && throw(ArgumentError("h must be provided when min_depth is set."))
        size(h) == size(mask) || throw(DimensionMismatch("h size $(size(h)) does not match mask size $(size(mask))"))
        for index in CartesianIndices(mask)
            !isnothing(wetmask) && !Bool(wetmask[index]) && continue
            hi = h[index]
            unstable = ismissing(hi) || !(hi isa Number) || !isfinite(hi) || hi < min_depth
            mask[index] |= unstable
        end
    end

    if !isnothing(roughness_threshold)
        isnothing(roughness) && throw(ArgumentError("roughness must be provided when roughness_threshold is set."))
        _flag_diagnostic!(mask, roughness, roughness_threshold, wetmask)
    end

    if !isnothing(wetmask)
        mask .&= Bool.(wetmask)
    end

    return mask
end

"""
    dilate_mask(mask; radius=2, wetmask=nothing, neighbors=:cardinal)

Expand `mask` by `radius` cardinal-neighbor steps. When `wetmask` is provided,
the expansion is constrained to wet cells.
"""
function dilate_mask(mask::AbstractArray{Bool}; radius=2, wetmask=nothing, neighbors=:cardinal)
    radius < 0 && throw(ArgumentError("radius must be non-negative."))
    isnothing(wetmask) || size(wetmask) == size(mask) || throw(DimensionMismatch("wetmask size $(size(wetmask)) does not match mask size $(size(mask))"))

    offsets = _neighbor_offsets(ndims(mask), neighbors)
    axes_mask = axes(mask)
    dilated = copy(mask)
    !isnothing(wetmask) && (dilated .&= Bool.(wetmask))

    for _ in 1:radius
        expanded = copy(dilated)
        for index in CartesianIndices(mask)
            dilated[index] || continue
            for offset in offsets
                neighbor = index + offset
                _inside(neighbor, axes_mask) || continue
                !isnothing(wetmask) && !Bool(wetmask[neighbor]) && continue
                expanded[neighbor] = true
            end
        end
        dilated = expanded
    end

    return dilated
end

function _neighbor_values(h, index, wetmask, offsets, axes_h)
    values = Float64[]
    for offset in offsets
        neighbor = index + offset
        _inside(neighbor, axes_h) || continue
        hn = h[neighbor]
        _iswet(hn, wetmask, neighbor) || continue
        push!(values, Float64(hn))
    end
    return values
end

function _neighbor_mean(h, index, wetmask, offsets, axes_h)
    total = zero(float(nonmissingtype(eltype(h))))
    count = 0

    for offset in offsets
        neighbor = index + offset
        _inside(neighbor, axes_h) || continue
        hn = h[neighbor]
        _iswet(hn, wetmask, neighbor) || continue
        total += hn
        count += 1
    end

    return count == 0 ? nothing : total / count
end

function _limit_change(old, proposed, max_change)
    isnothing(max_change) && return proposed
    delta = clamp(proposed - old, -max_change, max_change)
    return old + delta
end

function _roughness_limited_depth(old, proposed, neighbor_depths, roughness_limit, mode)
    isnothing(roughness_limit) && return proposed
    roughness_limit < 0 && throw(ArgumentError("roughness_limit must be non-negative."))
    roughness_limit >= 1 && return proposed

    lower_bound = maximum(depth -> depth * (1 - roughness_limit) / (1 + roughness_limit), neighbor_depths; init=0.0)
    upper_bound = minimum(depth -> depth * (1 + roughness_limit) / (1 - roughness_limit), neighbor_depths; init=Inf)

    if mode === :deepen_only
        return max(proposed, lower_bound)
    else
        return clamp(proposed, lower_bound, upper_bound)
    end
end

"""
    smooth_flagged_bathymetry(h, mask; wetmask=nothing, iterations=1,
                              max_change=nothing, mode=:deepen_only,
                              roughness_limit=nothing, preserve_mask=nothing)

Return `(new_h, patch_log)` after local, deterministic conditioning of
bathymetric depth `h` inside `mask`.

This is an offline preprocessing/repair utility for reproducible bathymetry
patches around unstable estuaries, straits, shallow cells, or steep local jumps.
It is not hidden runtime model behavior and does not run the model internally.
Users should inspect bathymetry diffs and transport-sensitive passages before
production use.

`mode = :deepen_only` only increases positive depth. `mode = :smooth` moves each
flagged cell toward its wet-neighbor average. Land, `missing`, `NaN`, and `Inf`
cells are preserved. `preserve_mask` cells are never altered.
"""
function smooth_flagged_bathymetry(h::AbstractArray, mask::AbstractArray{Bool};
                                   wetmask=nothing, iterations=1, max_change=nothing,
                                   mode=:deepen_only, roughness_limit=nothing,
                                   preserve_mask=nothing)
    size(h) == size(mask) || throw(DimensionMismatch("mask size $(size(mask)) does not match h size $(size(h))"))
    isnothing(wetmask) || size(wetmask) == size(h) || throw(DimensionMismatch("wetmask size $(size(wetmask)) does not match h size $(size(h))"))
    isnothing(preserve_mask) || size(preserve_mask) == size(h) || throw(DimensionMismatch("preserve_mask size $(size(preserve_mask)) does not match h size $(size(h))"))
    iterations < 1 && throw(ArgumentError("iterations must be at least 1."))
    mode in (:deepen_only, :smooth) || throw(ArgumentError("mode must be :deepen_only or :smooth."))

    new_h = copy(h)
    offsets = _neighbor_offsets(ndims(h), :cardinal)
    axes_h = axes(h)
    records = BathymetryPatchRecord[]

    for iteration in 1:iterations
        previous_h = copy(new_h)

        for index in CartesianIndices(h)
            mask[index] || continue
            !isnothing(preserve_mask) && Bool(preserve_mask[index]) && continue

            old = previous_h[index]
            _iswet(old, wetmask, index) || continue

            neighbor_mean = _neighbor_mean(previous_h, index, wetmask, offsets, axes_h)
            isnothing(neighbor_mean) && continue

            proposed = mode === :deepen_only ? max(old, neighbor_mean) : neighbor_mean
            if !isnothing(roughness_limit)
                neighbor_depths = _neighbor_values(previous_h, index, wetmask, offsets, axes_h)
                proposed = _roughness_limited_depth(old, proposed, neighbor_depths, roughness_limit, mode)
            end
            proposed = _limit_change(old, proposed, max_change)

            if mode === :deepen_only
                proposed = max(old, proposed)
            end

            proposed == old && continue
            new_h[index] = proposed

            push!(records, BathymetryPatchRecord(index = Tuple(index),
                                                 old_depth = old,
                                                 new_depth = proposed,
                                                 delta = proposed - old,
                                                 reason = isnothing(roughness_limit) ? :neighbor_smoothing : :roughness_limit,
                                                 iteration = iteration))
        end
    end

    return new_h, BathymetryPatchLog(records)
end

"""
    summarize_bathymetry_patch(old_h, new_h; wetmask=nothing)

Summarize an offline bathymetry patch. The summary includes changed-cell count,
maximum and mean absolute depth change over changed cells, and the unweighted
sum of depth changes as a volume-like diagnostic when grid-cell areas are not
available.
"""
function summarize_bathymetry_patch(old_h::AbstractArray, new_h::AbstractArray; wetmask=nothing)
    size(old_h) == size(new_h) || throw(DimensionMismatch("old_h and new_h must have the same size."))
    isnothing(wetmask) || size(wetmask) == size(old_h) || throw(DimensionMismatch("wetmask size $(size(wetmask)) does not match h size $(size(old_h))"))

    deltas = Float64[]
    for index in CartesianIndices(old_h)
        !isnothing(wetmask) && !Bool(wetmask[index]) && continue
        old = old_h[index]
        new = new_h[index]
        _isvalid_depth(old) && _isvalid_depth(new) || continue
        new == old && continue
        push!(deltas, Float64(new - old))
    end

    if isempty(deltas)
        return BathymetryPatchSummary(changed_cells = 0,
                                      max_abs_depth_change = 0.0,
                                      mean_abs_depth_change = 0.0,
                                      total_depth_change = 0.0)
    end

    abs_deltas = abs.(deltas)
    return BathymetryPatchSummary(changed_cells = length(deltas),
                                  max_abs_depth_change = maximum(abs_deltas),
                                  mean_abs_depth_change = sum(abs_deltas) / length(abs_deltas),
                                  total_depth_change = sum(deltas))
end

#####
##### Checkpoint workflow
#####

function _checkpoint_iteration(path)
    m = match(r"iteration(\d+)", basename(path))
    return isnothing(m) ? nothing : parse(Int, m.captures[1])
end

function _discover_checkpoint_files(prefix; run="all")
    if isfile(prefix)
        files = [prefix]
    else
        dir = dirname(prefix)
        isempty(dir) && (dir = ".")
        base = basename(prefix)
        isdir(dir) || throw(ArgumentError("Checkpoint directory does not exist: $dir"))
        files = [joinpath(dir, file) for file in readdir(dir)
                 if startswith(file, base) && endswith(file, ".jld2")]
    end

    files = filter(path -> !isnothing(_checkpoint_iteration(path)), files)
    isempty(files) && throw(ArgumentError("No checkpoint files matching prefix $prefix were found."))

    files = sort(files, by=_checkpoint_iteration)

    if run in (:all, "all")
        return files
    elseif run isa Integer
        selected = filter(path -> _checkpoint_iteration(path) == run, files)
    else
        wanted = Set(Int.(collect(run)))
        selected = filter(path -> _checkpoint_iteration(path) in wanted, files)
    end

    isempty(selected) && throw(ArgumentError("No checkpoint files matching run = $run were found."))
    return selected
end

function _read_field_data(file, path)
    haskey(file, path) || return nothing
    group = file[path]
    group isa JLD2.Group || return group
    return haskey(group, "data") ? group["data"] : nothing
end

function _interior_slice(A::AbstractArray, target_size::Tuple{Int, Int}; halo=nothing)
    Nx, Ny = target_size
    size(A, 1) >= Nx && size(A, 2) >= Ny || throw(DimensionMismatch("Cannot extract interior size $target_size from array of size $(size(A))."))

    Hx, Hy = if isnothing(halo)
        extra_x = size(A, 1) - Nx
        extra_y = size(A, 2) - Ny
        iseven(extra_x) && iseven(extra_y) || throw(DimensionMismatch("Cannot infer symmetric halos from array size $(size(A)) and target interior size $target_size."))
        extra_x ÷ 2, extra_y ÷ 2
    else
        halo[1], halo[2]
    end

    Hx + Nx <= size(A, 1) && Hy + Ny <= size(A, 2) ||
        throw(DimensionMismatch("Grid halo ($Hx, $Hy) and interior size $target_size exceed array size $(size(A))."))

    x = Hx+1:Hx+Nx
    y = Hy+1:Hy+Ny
    return ndims(A) == 2 ? view(A, x, y) : view(A, x, y, :)
end

function _column_nonfinite_mask(A::AbstractArray, target_size::Tuple{Int, Int}; halo=nothing)
    interior = _interior_slice(A, target_size; halo)
    if ndims(interior) == 2
        return .!isfinite.(interior)
    else
        return dropdims(any(!isfinite, interior; dims=3), dims=3)
    end
end

function _column_max_abs(A::AbstractArray, target_size::Tuple{Int, Int}; halo=nothing)
    interior = _interior_slice(A, target_size; halo)
    if ndims(interior) == 2
        return abs.(Float64.(interior))
    else
        return dropdims(maximum(abs, interior; dims=3), dims=3)
    end
end

function _positive_depth(bottom_height)
    h = Array(bottom_height)
    if any(<(0), h)
        h = max.(-h, 0)
    else
        h = max.(h, 0)
    end
    return h
end

function _bathymetry_from_output_file(path)
    jldopen(path, "r") do file
        haskey(file, "serialized/grid") || return nothing
        grid = file["serialized/grid"]
        hasproperty(grid, :immersed_boundary) || return nothing
        bottom_height = grid.immersed_boundary.bottom_height
        Nx, Ny = grid.Nx, grid.Ny
        Hx, Hy = grid.Hx, grid.Hy
        h = _positive_depth(Array(bottom_height[Hx+1:Hx+Nx, Hy+1:Hy+Ny, 1]))
        return (; h, source = path, halo = (Hx, Hy))
    end
end

function _bathymetry_from_checkpoint(path)
    jldopen(path, "r") do file
        candidate_paths = (
            "simulation/model/ocean/model/grid/immersed_boundary/bottom_height",
            "simulation/model/ocean/model/grid/underlying_grid/immersed_boundary/bottom_height",
            "simulation/model/ocean/model/grid/bottom_height",
        )

        for candidate in candidate_paths
            haskey(file, candidate) || continue
            return (; h = _positive_depth(file[candidate]),
                      source = path,
                      halo = nothing)
        end
    end
    return nothing
end

function _run_tag_from_checkpoint_prefix(prefix)
    base = basename(prefix)
    base = replace(base, r"_checkpoint.*$" => "")
    parts = split(base, "_")
    return length(parts) >= 2 ? parts[2] : base
end

function _discover_bathymetry_from_sibling_outputs(checkpoint_prefix)
    dir = dirname(checkpoint_prefix)
    isempty(dir) && (dir = ".")
    tag = _run_tag_from_checkpoint_prefix(checkpoint_prefix)

    candidates = [joinpath(dir, file) for file in readdir(dir)
                  if endswith(file, ".jld2") && occursin(tag, file) && !occursin("checkpoint", file)]
    sort!(candidates)

    for candidate in candidates
        bathymetry = try
            _bathymetry_from_output_file(candidate)
        catch
            nothing
        end
        isnothing(bathymetry) || return bathymetry
    end

    return nothing
end

function _load_bathymetry_source(source)
    source isa AbstractArray && return (; h = Array(source), source = "array", halo = nothing)
    source isa AbstractString || throw(ArgumentError("bathymetry must be an array or a JLD2 path."))

    jldopen(source, "r") do file
        if haskey(file, "h")
            return (; h = Array(file["h"]), source, halo = nothing)
        elseif haskey(file, "bottom_height")
            return (; h = _positive_depth(file["bottom_height"]), source, halo = nothing)
        else
            throw(ArgumentError("Bathymetry file $source must contain key \"h\" or \"bottom_height\"."))
        end
    end
end

function _checkpoint_instability(path, target_size; velocity_threshold=nothing, halo=nothing)
    nan_mask = falses(target_size)
    velocity_mask = falses(target_size)

    jldopen(path, "r") do file
        field_paths = (
            "simulation/model/ocean/model/velocities/u",
            "simulation/model/ocean/model/velocities/v",
            "simulation/model/ocean/model/velocities/w",
            "simulation/model/ocean/model/tracers/T",
            "simulation/model/ocean/model/tracers/S",
            "simulation/model/ocean/model/tracers/e",
            "simulation/model/ocean/model/free_surface/displacement",
            "simulation/model/ocean/model/vertical_coordinate/ηⁿ",
        )

        for field_path in field_paths
            data = _read_field_data(file, field_path)
            isnothing(data) && continue
            ndims(data) >= 2 || continue
            nan_mask .|= _column_nonfinite_mask(data, target_size; halo)
        end

        if !isnothing(velocity_threshold)
            for field_path in ("simulation/model/ocean/model/velocities/u",
                               "simulation/model/ocean/model/velocities/v")
                data = _read_field_data(file, field_path)
                isnothing(data) && continue
                velocity_mask .|= _column_max_abs(data, target_size; halo) .> velocity_threshold
            end
        end
    end

    return nan_mask, velocity_mask
end

"""
    condition_bathymetry(checkpoint_prefix; run="all", bathymetry=nothing, output_path=nothing, kw...)

Ingest one or more saved checkpoints, union their unstable horizontal columns,
and write one conditioned bathymetry artifact. This is an offline preprocessing
workflow: it reads checkpoint arrays and bathymetry metadata, but never reruns a
model internally.

`checkpoint_prefix` may be a full checkpoint file or a prefix such as
`"../outputs/RYF_sxtdeg_checkpoint"`. With `run = "all"` all matching
`*iteration*.jld2` checkpoint files are used. With `run = 340` or
`run = [68, 136]`, only those checkpoint iterations are used.

The utility first tries to read bathymetry from the checkpoint. If the
checkpoint does not store the grid, it searches sibling output files containing
`serialized/grid` and uses the grid's `Hx` and `Hy` to skip halos while reading
checkpoint fields. A bathymetry array or JLD2 path can also be supplied via
`bathymetry`; in that case `field_halo = (Hx, Hy)` can be supplied explicitly.
"""
function condition_bathymetry(checkpoint_prefix;
                              run="all",
                              bathymetry=nothing,
                              output_path=nothing,
                              cfl_threshold=0.8,
                              velocity_threshold=nothing,
                              min_depth=nothing,
                              roughness_threshold=nothing,
                              dilation_radius=2,
                              smoothing_iterations=1,
                              max_change=nothing,
                              mode=:deepen_only,
                              roughness_limit=nothing,
                              preserve_mask=nothing,
                              field_halo=nothing,
                              write=true)

    checkpoint_files = _discover_checkpoint_files(checkpoint_prefix; run)
    @info "Discovered checkpoint files for bathymetry conditioning" checkpoint_prefix run checkpoints=length(checkpoint_files)

    bathymetry_record = if isnothing(bathymetry)
        @info "Looking for bathymetry in checkpoint files"
        checkpoint_h = _bathymetry_from_checkpoint(first(checkpoint_files))
        if isnothing(checkpoint_h)
            @info "Bathymetry not found in checkpoint; searching sibling output files with serialized/grid"
            sibling_bathymetry = _discover_bathymetry_from_sibling_outputs(checkpoint_prefix)
            if isnothing(sibling_bathymetry)
                throw(ArgumentError("Could not find bathymetry in checkpoints or sibling output files. Pass bathymetry=<array or JLD2 path>."))
            end
            sibling_bathymetry
        else
            checkpoint_h
        end
    else
        @info "Loading user-supplied bathymetry source"
        _load_bathymetry_source(bathymetry)
    end

    h = bathymetry_record.h
    bathymetry_source = bathymetry_record.source
    halo = isnothing(field_halo) ? bathymetry_record.halo : field_halo
    @info "Using bathymetry source" bathymetry_source size=size(h) halo

    wetmask = h .> 0
    target_size = size(h)
    @info "Computing bathymetry roughness" wet_cells=count(wetmask)
    roughness = compute_bathymetry_roughness(h, wetmask)

    @info "Building initial unstable mask from bathymetry criteria" min_depth roughness_threshold
    unstable_mask = flag_unstable_columns(h = h,
                                          wetmask = wetmask,
                                          cfl_threshold = cfl_threshold,
                                          min_depth = min_depth,
                                          roughness = roughness,
                                          roughness_threshold = roughness_threshold)
    @info "Initial unstable mask complete" flagged_cells=count(unstable_mask)

    nan_mask = falses(target_size)
    velocity_mask = falses(target_size)

    for (n, checkpoint) in enumerate(checkpoint_files)
        @info "Scanning checkpoint for unstable columns" checkpoint_index=n total_checkpoints=length(checkpoint_files) checkpoint
        checkpoint_nan_mask, checkpoint_velocity_mask = _checkpoint_instability(checkpoint, target_size; velocity_threshold, halo)
        nan_mask .|= checkpoint_nan_mask
        velocity_mask .|= checkpoint_velocity_mask
        @info "Finished checkpoint scan" checkpoint_index=n nan_columns=count(checkpoint_nan_mask) velocity_columns=count(checkpoint_velocity_mask)
    end

    unstable_mask .|= nan_mask
    unstable_mask .|= velocity_mask
    unstable_mask .&= wetmask
    @info "Combined unstable mask" flagged_cells=count(unstable_mask) nan_columns=count(nan_mask) velocity_columns=count(velocity_mask)

    @info "Dilating repair mask" dilation_radius
    repair_mask = dilate_mask(unstable_mask; radius = dilation_radius, wetmask)
    @info "Repair mask ready" repair_cells=count(repair_mask)

    @info "Smoothing flagged bathymetry" mode smoothing_iterations max_change roughness_limit
    patched_h, patch_log = smooth_flagged_bathymetry(h, repair_mask;
                                                     wetmask,
                                                     iterations = smoothing_iterations,
                                                     max_change,
                                                     mode,
                                                     roughness_limit,
                                                     preserve_mask)

    summary = summarize_bathymetry_patch(h, patched_h; wetmask)
    @info "Bathymetry conditioning summary" changed_cells=summary.changed_cells max_abs_depth_change=summary.max_abs_depth_change mean_abs_depth_change=summary.mean_abs_depth_change total_depth_change=summary.total_depth_change

    if isnothing(output_path)
        dir = dirname(checkpoint_prefix)
        isempty(dir) && (dir = ".")
        base = basename(checkpoint_prefix)
        output_path = joinpath(dir, base * "_conditioned_bathymetry.jld2")
    end

    result = (; h = patched_h,
                original_h = h,
                wetmask,
                roughness,
                unstable_mask,
                nan_mask,
                velocity_mask,
                repair_mask,
                patch_log,
                summary,
                checkpoint_files,
                bathymetry_source,
                halo,
                output_path)

    if write
        @info "Writing conditioned bathymetry" output_path
        jldopen(output_path, "w") do file
            file["h"] = patched_h
            file["original_h"] = h
            file["wetmask"] = wetmask
            file["roughness"] = roughness
            file["unstable_mask"] = unstable_mask
            file["nan_mask"] = nan_mask
            file["velocity_mask"] = velocity_mask
            file["repair_mask"] = repair_mask
            file["patch_records"] = patch_log.records
            file["patch_summary"] = summary
            file["checkpoint_files"] = checkpoint_files
            file["bathymetry_source"] = bathymetry_source
            file["halo"] = isnothing(halo) ? Int[] : collect(halo)
        end
        @info "Finished writing conditioned bathymetry" output_path
    else
        @info "Skipping conditioned bathymetry write because write=false"
    end

    return result
end
