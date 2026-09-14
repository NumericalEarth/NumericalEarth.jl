
# The bathymetry cache shares DataWrangling's keyed field cache, carrying the
# processing parameters in the key's keyword slot. Parameters are normalized so
# that, e.g., `minimum_depth = 0` and `minimum_depth = 0.0` key identically.
function bathymetry_regridding_key(grid, metadata;
                                   height_above_water, minimum_depth,
                                   interpolation_passes, major_basins)
    parameters = (; height_above_water = isnothing(height_above_water) ? nothing : Float64(height_above_water),
                    minimum_depth = Float64(minimum_depth),
                    interpolation_passes = Int(interpolation_passes),
                    major_basins = Float64(major_basins))
    return FieldRegridding(grid, metadata, parameters)
end

# methods specific to bathymetric datasets live within dataset modules

"""
    regrid_bathymetry(target_grid, metadata;
                      height_above_water = nothing,
                      minimum_depth = 0,
                      major_basins = 1,
                      interpolation_passes = 1,
                      cache = true,
                      overwrite_cache = false)

Return bathymetry that corresponds to  `metadata` onto `target_grid`.

Arguments
=========

- `target_grid`: grid to interpolate the bathymetry onto.

Keyword Arguments
=================

- `height_above_water`: limits the maximum height of above-water topography (where ``h > 0``) before interpolating.
                        Default: `nothing`, which implies that the original topography is retained.

- `minimum_depth`: minimum depth for the shallow regions, defined as a positive value.
                   `h > - minimum_depth` is considered land. Default: 0.

- `interpolation_passes`: regridding/interpolation passes. The bathymetry is interpolated in
                          `interpolation_passes - 1` intermediate steps. The more the interpolation
                          steps the smoother the final bathymetry becomes.

  Example
  =======

  Interpolating from a 400 x 200 grid to a 100 x 100 grid in 4 passes involves:

  * 400 x 200 → 325 x 175
  * 325 x 175 → 250 x 150
  * 250 x 150 → 175 x 125
  * 175 x 125 → 100 x 100

  If _coarsening_ the original grid, linear interpolation in passes is equivalent to
  applying a smoothing filter, with more passes increasing the strength of the filter.
  If _refining_ the original grid, additional passes do not help and no intermediate
  steps are performed.

- `major_basins`: Number of "independent major basins", or fluid regions fully encompassed by land,
                  that are retained by [`remove_minor_basins!`](@ref). Basins are removed by order of size:
                  the smallest basins are removed first. `major_basins = 1` retains only the largest basin.
                  If `Inf` then no basins are removed. Default: 1.

- `cache`: If `true` (default), caches the regridded bathymetry to disk and reuses it on subsequent
           calls with the same grid, parameters, and dataset file; a re-download of the dataset
           invalidates the entry. If `false`, the cache is disabled entirely: nothing is read
           or written.

- `overwrite_cache`: If `true`, skip the cache lookup and overwrite the entry with a freshly
                     regridded result. Default: `false`.
"""
function regrid_bathymetry(target_grid, metadata;
                           height_above_water = nothing,
                           minimum_depth = 0,
                           interpolation_passes = 1,
                           major_basins = 1,
                           cache = true,
                           overwrite_cache = false)

    validate_dataset_coverage(target_grid, metadata)

    if cache && !overwrite_cache
        config = bathymetry_regridding_key(target_grid, metadata;
                                           height_above_water, minimum_depth,
                                           interpolation_passes, major_basins)
        cached_data = load_field_cache(config)
        if !isnothing(cached_data)
            target_z = Field{Center, Center, Nothing}(target_grid)
            set!(target_z, cached_data)
            fill_halo_regions!(target_z)
            return target_z
        end
    end

    download(metadata)

    target_z = _regrid_bathymetry(target_grid, metadata;
                                  height_above_water,
                                  minimum_depth,
                                  interpolation_passes,
                                  major_basins)

    if cache
        # rebuild the key: `download` may have just fetched the dataset file it stamps
        config = bathymetry_regridding_key(target_grid, metadata;
                                           height_above_water, minimum_depth,
                                           interpolation_passes, major_basins)
        save_field_cache(config, Array(interior(target_z, :, :, 1)))
    end

    return target_z
end

# regrid the bathymetry assuming the data is already downloaded
function _regrid_bathymetry(target_grid, metadata;
                            height_above_water,
                            minimum_depth,
                            interpolation_passes,
                            major_basins)
    if isinteger(interpolation_passes)
        interpolation_passes = convert(Int, interpolation_passes)
    end

    if interpolation_passes isa Nothing || !isa(interpolation_passes, Int) || interpolation_passes ≤ 0
        return throw(ArgumentError("interpolation_passes has to be an integer ≥ 1"))
    end

    arch = architecture(target_grid)

    bathymetry_native_grid = native_grid(metadata, arch; halo = (10, 10, 1))
    FT = eltype(target_grid)

    filepath = metadata_path(metadata)
    dataset = Dataset(filepath, "r")

    z_data = convert(Array{FT}, dataset[dataset_variable_name(metadata)][:, :])
    close(dataset)

    if !isnothing(height_above_water)
        # Overwrite the height of cells above water.
        # This has an impact on reconstruction. Greater height_above_water reduces total
        # wet area by biasing coastal regions to land during bathymetry regridding.
        land = z_data .> 0
        z_data[land] .= height_above_water
    end

    native_z = Field{Center, Center, Nothing}(bathymetry_native_grid)
    set!(native_z, z_data)
    fill_halo_regions!(native_z)

    target_z = interpolate_bathymetry_in_passes(native_z, target_grid;
                                                passes = interpolation_passes)

    if minimum_depth > 0
        launch!(arch, target_grid, :xy, _enforce_minimum_depth!, target_z, minimum_depth)
    end

    if major_basins < Inf
        remove_minor_basins!(target_z, major_basins)
    end

    fill_halo_regions!(target_z)

    return target_z
end

"""
    regrid_bathymetry(target_grid; dataset=ETOPO2022(), cache=true, kw...)

Regrid bathymetry from `dataset` onto `target_grid`. Default: `dataset = ETOPO2022()`.
"""
function regrid_bathymetry(target_grid; dataset = ETOPO2022(), cache = true, kw...)
    metadatum = Metadatum(:bottom_height; dataset)
    return regrid_bathymetry(target_grid, metadatum; cache, kw...)
end

"""
    regrid_topography(target_grid; dataset = ETOPO2022(), kw...)

Land surface elevation (m, ≥ 0) regridded onto `target_grid` — the topographic
counterpart of [`regrid_bathymetry`](@ref) for land applications. Returns the
positive part of the dataset's bottom height (the elevation over land), with
ocean clamped to sea level (0). Accepts the same regridding keywords
(`interpolation_passes`, etc.); there is no depth/`minimum_depth` notion.
"""
function regrid_topography(target_grid; dataset = ETOPO2022(), kw...)
    elevation = regrid_bathymetry(target_grid; dataset, kw...)
    parent(elevation) .= max.(parent(elevation), 0) # land elevation; ocean → 0
    return elevation
end

"""
    regrid_topography(target_grid, metadata; kw...)

Land surface elevation regridded onto `target_grid` from `metadata`, the positive
counterpart of [`regrid_bathymetry`](@ref). Use this form for region-windowed
datasets such as `GLO30()`, whose `metadata` carries a `BoundingBox` region.
"""
function regrid_topography(target_grid, metadata; kw...)
    elevation = regrid_bathymetry(target_grid, metadata; kw...)
    parent(elevation) .= max.(parent(elevation), 0) # land elevation; ocean → 0
    return elevation
end

# Regridding bathymetry for distributed grids, we handle the whole process
# on just one rank, and share the results with the other processors.
function regrid_bathymetry(target_grid::DistributedGrid, metadata;
                           height_above_water = nothing,
                           minimum_depth = 0,
                           interpolation_passes = 1,
                           major_basins = 1,
                           cache = true,
                           overwrite_cache = false)

    global_grid = reconstruct_global_grid(target_grid)
    global_grid = on_architecture(CPU(), global_grid)
    arch = architecture(target_grid)
    Nx, Ny, _ = size(global_grid)

    # download uses @root internally; all ranks must call it
    download(metadata)

    config = cache ? bathymetry_regridding_key(global_grid, metadata;
                                               height_above_water, minimum_depth,
                                               interpolation_passes, major_basins) : nothing

    # Only rank 0 performs cache lookup and computation to avoid OOM.
    # Every rank must contribute the same element type to the shared reduction:
    # mismatched MPI datatypes across ranks corrupt the collective.
    FT = eltype(global_grid)
    bottom_height = if arch.local_rank == 0
        cached_data = cache && !overwrite_cache ? load_field_cache(config) : nothing
        rank_zero_data = if !isnothing(cached_data)
            cached_data
        else
            bottom_field = _regrid_bathymetry(global_grid, metadata;
                                              height_above_water, minimum_depth,
                                              interpolation_passes, major_basins)
            bh = Array(bottom_field.data[1:Nx, 1:Ny, 1])
            if cache
                save_field_cache(config, bh)
            end
            bh
        end
        convert(Matrix{FT}, rank_zero_data)
    else
        zeros(FT, Nx, Ny)
    end

    # Synchronize
    Oceananigans.DistributedComputations.barrier(arch.communicator)

    # Share the result (can we share SubArrays?)
    bottom_height = all_reduce(+, bottom_height, arch)

    # Partition the result. Distributed `set!` only auto-partitions global-size *host*
    # arrays (`Array`/`OffsetArray`), so stage the shared result through the CPU.
    local_bottom_height = Field{Center, Center, Nothing}(target_grid)
    set!(local_bottom_height, on_architecture(CPU(), bottom_height))
    fill_halo_regions!(local_bottom_height)

    return local_bottom_height
end

@kernel function _enforce_minimum_depth!(target_z, minimum_depth)
    i, j = @index(Global, NTuple)
    z = @inbounds target_z[i, j, 1]

    # Fix active cells to be at least `-minimum_depth`.
    active = z < 0 # it's a wet cell
    z = ifelse(active, min(z, -minimum_depth), z)

    @inbounds target_z[i, j, 1] = z
end

# Here we can either use `regrid!` (three dimensional version) or `interpolate!`.
function interpolate_bathymetry_in_passes(native_z, target_grid;
                                          passes = 10)

    Nλt, Nφt = Nt = size(target_grid)
    Nλn, Nφn = Nn = size(native_z)

    # Interpolate in passes
    latitude  = y_domain(native_z.grid)
    longitude = x_domain(native_z.grid)

    ΔNλ = floor((Nλn - Nλt) / passes)
    ΔNφ = floor((Nφn - Nφt) / passes)

    Nλ = [Nλn - ΔNλ * pass for pass in 1:passes-1]
    Nφ = [Nφn - ΔNφ * pass for pass in 1:passes-1]

    Nλ = Int[Nλ..., Nλt]
    Nφ = Int[Nφ..., Nφt]

    old_z  = native_z
    TXt, _, _ = topology(target_grid)
    _, TYn, _ = topology(native_z.grid)

    Hx, Hy, Hz = Oceananigans.halo_size(native_z.grid)

    # Intermediate grids are regular LatitudeLongitudeGrids — they have poles,
    # not a fold. Inheriting TY from a TripolarGrid target (RightCenterFolded)
    # is invalid: Field construction would default to PolarValue BCs at the
    # pole, which folded topology rejects (only Zipper / distributed_comm /
    # nothing are allowed). Use Bounded; the actual fold is applied at the
    # final pass when interpolating onto target_grid.
    @info "Interpolation passes of bathymetry size $(size(old_z)) onto a $(typeof(target_grid).name.wrapper) target grid of size $Nt:"
    for pass = 1:passes - 1
        new_size = (Nλ[pass], Nφ[pass], 1)
        @info "    pass $pass to size $new_size"

        new_grid = LatitudeLongitudeGrid(architecture(target_grid), Float32,
                                         size = new_size,
                                         latitude = (latitude[1],  latitude[2]),
                                         longitude = (longitude[1], longitude[2]),
                                         z = (0, 1),
                                         topology = (TXt, TYn, Bounded),
                                         halo = (Hx, Hy, Hz))

        new_z = Field{Center, Center, Nothing}(new_grid)

        interpolate!(new_z, old_z)
        old_z = new_z
    end

    new_size = (Nλ[passes], Nφ[passes], 1)
    @info "    pass $passes to size $new_size"
    target_z = Field{Center, Center, Nothing}(target_grid)
    interpolate!(target_z, old_z)

    return target_z
end

"""
    remove_minor_basins!(z_data, keep_major_basins)

Remove independent basins from the bathymetry data stored in `z_data` by identifying connected regions
below sea level. Basins are removed from smallest to largest until only `keep_major_basins` remain.

Arguments
=========

- `z_data`: A 2D array representing the bathymetry data.
- `keep_major_basins`: The maximum number of connected regions to keep.
                       If `Inf` is provided then all connected regions are kept.

"""
function remove_minor_basins!(zb::Field, keep_major_basins)
    if !isfinite(keep_major_basins)
        throw(ArgumentError("`keep_major_basins` must be a finite number!"))
    end

    if keep_major_basins < 1
        throw(ArgumentError("keep_major_basins must be larger than 0."))
    end

    cpu_arch  = Oceananigans.DistributedComputations.cpu_architecture(architecture(zb))
    zb_cpu    = on_architecture(cpu_arch, zb)
    TX, TY, _ = topology(zb_cpu.grid)

    Nx = Base.length(Center(), TX(), zb_cpu.grid.Nx)
    Ny = Base.length(Center(), TY(), zb_cpu.grid.Ny)

    # Get labels for the core region (extension is handled internally by label_ocean_basins)
    labels = label_ocean_basins(zb_cpu, TX, (Nx, Ny))
    nlabels = maximum(labels)

    if nlabels == 0
        return zb  # No basins found
    end

    # Rank labels by the number of elements they occupy
    total_elements = zeros(nlabels)
    label_elements = zeros(Int, nlabels)

    for e in 1:nlabels
        cnt = count(==(e), labels)
        total_elements[e] = cnt
        label_elements[e] = e
    end

    # Find valid basins (those with at least one cell)
    valid = findall(>(0), total_elements)

    major_basins = Int[]  # indices of major basins to keep
    m = 1

    # We add basin indexes until we reach the specified number (m == keep_major_basins) or
    # we run out of basins to keep -> isempty(valid)
    while (m ≤ keep_major_basins) && !isempty(valid)
        # Among the remaining valid labels, find the one with the largest core area.
        _, idx = findmax(total_elements[valid])
        next_label = label_elements[valid[idx]]
        push!(major_basins, next_label)
        deleteat!(valid, idx)
        m += 1
    end

    # Modify the bathymetry: set minor basin cells to 0 (land)
    # Work on interior view which directly modifies the underlying data
    zb_data = interior(zb_cpu, :, :, 1)

    for j in 1:Ny, i in 1:Nx
        label = labels[i, j]
        if label > 0 && !(label in major_basins)
            zb_data[i, j] = 0  # Flatten this cell (make it land)
        end
    end

    # If original field was on a different architecture, copy back
    if zb !== zb_cpu
        set!(zb, zb_cpu)
    end

    return zb
end
