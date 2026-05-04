# visualize/common.jl
#
# Imports, monkey-patches, FTS_BACKEND, plotting helpers, time-averaging
# utilities, land/ocean masks, climatology readers (ECCO, dBM, NCEP).
#
# Expects the orchestrator to have already defined:
#   - `cases`        :: Vector{NamedTuple}   (the case list)
#   - `output_dir`   :: String                (where PNGs go)
#
# Defines `obs_cache_dir`, `FTS_BACKEND`, `case_colors`, `savefig`, etc.

# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

const years    = 365 * 24 * 3600
const ρ_ocean  = 1026.0
const cp_ocean = 3991.86795711963

# ══════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════

using CairoMakie
using Statistics
using Dates
using Downloads
using DelimitedFiles
using JLD2
using NCDatasets
using WorldOceanAtlasTools
using Oceananigans
using Oceananigans.Grids: znodes, φnodes, φnode
using Oceananigans.Fields: interpolate!
using ConservativeRegridding
using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum
using NumericalEarth.DataWrangling.WOA: WOAAnnual
using NumericalEarth: ECCO4Monthly
using OMIPSimulations: strait_transports

# ══════════════════════════════════════════════════════════════
# Monkey-patch: InMemory FieldTimeSeries split-file support
# ══════════════════════════════════════════════════════════════
# Oceananigans 0.107.x bug (field_time_series.jl, ~line 924-930):
# the inner FieldTimeSeries constructor only builds a SplitFilePath
# when `backend isa OnDisk`. With an InMemory backend on split
# output (..._part1.jld2, ..._part2.jld2, ...), `fts.path` collapses
# to a single part file, producing "No data found for time ..." warnings
# (and stale/zero data) on later snapshot reads.

import Oceananigans.Fields: set!
using Oceananigans.OutputReaders: SplitFilePath, InMemoryFTS,
                                  InMemory, time_indices,
                                  file_and_local_index, Prefetched

function set!(fts::InMemoryFTS, sfp::SplitFilePath, name::String = fts.name;
              warn_missing_data = false, kwargs...)
    idxs = time_indices(fts)
    Ntot = last(sfp.cumulative_length)
    needed = String[]
    for n in idxs
        (n < 1 || n > Ntot) && continue
        file_path, _ = file_and_local_index(sfp, n)
        file_path ∉ needed && push!(needed, file_path)
    end
    for p in needed
        set!(fts, p, name; warn_missing_data, kwargs...)
    end
    return nothing
end

function _detect_split_file_path(path::AbstractString, reader_kw)
    isfile(path) && return nothing
    base = endswith(path, ".jld2") ? path[1:end-5] : path
    dir  = isempty(dirname(base)) ? "." : dirname(base)
    pat  = Regex("^" * Base.escape_string(basename(base)) * "_part(\\d+)\\.jld2\$")
    files = filter(f -> occursin(pat, f), readdir(dir))
    isempty(files) && return nothing
    sort!(files, by = f -> parse(Int, match(pat, f).captures[1]))
    part_paths = [joinpath(dir, f) for f in files]
    nper = Int[]
    for p in part_paths
        jf = JLD2.jldopen(p; reader_kw...)
        push!(nper, length(keys(jf["timeseries/t"])))
        close(jf)
    end
    return SplitFilePath(part_paths, cumsum(nper))
end

_location_types(::Oceananigans.OutputReaders.FieldTimeSeries{LX, LY, LZ}) where {LX, LY, LZ} = (LX, LY, LZ)

_rebuild_backend_with_path(backend, new_path) = backend

function _rebuild_backend_with_path(backend::Prefetched, new_path)
    old_buf = getfield(backend, :buffer_fts)
    BLX, BLY, BLZ = _location_types(old_buf)
    new_buf = Oceananigans.OutputReaders.FieldTimeSeries{BLX, BLY, BLZ}(
        old_buf.data, old_buf.grid, old_buf.backend, old_buf.boundary_conditions,
        old_buf.indices, old_buf.times, new_path, old_buf.name,
        old_buf.time_indexing, old_buf.reader_kw)
    return Prefetched(backend.base_backend, backend.pending, new_buf, backend.next_start)
end

function _rebuild_fts_with_path(fts, new_path)
    LX, LY, LZ = _location_types(fts)
    new_backend = _rebuild_backend_with_path(fts.backend, new_path)
    return Oceananigans.OutputReaders.FieldTimeSeries{LX, LY, LZ}(
        fts.data, fts.grid, new_backend, fts.boundary_conditions,
        fts.indices, fts.times, new_path, fts.name,
        fts.time_indexing, fts.reader_kw)
end

function Oceananigans.OutputReaders.FieldTimeSeries(path::String, name::String;
                                                    backend = InMemory(),
                                                    reader_kw = NamedTuple(),
                                                    kwargs...)
    fts = invoke(Oceananigans.OutputReaders.FieldTimeSeries,
                 Tuple{String, Vararg{Any}},
                 path, name; backend, reader_kw, kwargs...)
    if backend isa InMemory && !(fts.path isa SplitFilePath)
        sfp = _detect_split_file_path(path, reader_kw)
        sfp === nothing || (fts = _rebuild_fts_with_path(fts, sfp))
    end
    return fts
end

# ══════════════════════════════════════════════════════════════
# Output directory + obs cache
# ══════════════════════════════════════════════════════════════

mkpath(output_dir)
@info "Figures will be saved to: $output_dir"

const obs_cache_dir = joinpath(output_dir, "obs_cache")
mkpath(obs_cache_dir)

# Shared backend template — `deepcopy(FTS_BACKEND)` for every FieldTimeSeries
# so each one gets its own independent buffer state. `prefetch = false`
# because multiple FTS share the same JLD2 file and `Prefetched` assumes
# sole-reader access.
const FTS_BACKEND = InMemory(10; prefetch = false)

savefig(fig, name) = save(joinpath(output_dir, name), fig)

# Per-case line colors, cycling if more cases than base palette entries.
const _BASE_CASE_COLORS = [:firebrick, :royalblue, :seagreen, :darkorange,
                           :purple, :teal, :goldenrod, :saddlebrown,
                           :magenta, :olive]
case_colors = [_BASE_CASE_COLORS[mod1(i, length(_BASE_CASE_COLORS))]
               for i in 1:length(cases)]

run_dir_for(prefix) = "$(prefix)_run"

# ══════════════════════════════════════════════════════════════
# File / time helpers
# ══════════════════════════════════════════════════════════════

function find_first_file(run_dir, prefix, group)
    tag = "$(prefix)_$(group)"
    candidates = filter(f -> startswith(f, tag) && endswith(f, ".jld2") &&
                             !contains(f, "checkpoint"), readdir(run_dir))
    isempty(candidates) && error("No $group files for prefix '$prefix' in $run_dir")
    filename = first(sort(candidates))
    basename_no_part = replace(filename, r"_part\d+" => "")
    return joinpath(run_dir, basename_no_part)
end

function in_window(fts; start_time = 0, stop_time = Inf)
    return findall(t -> start_time <= t <= stop_time, fts.times)
end

function compute_time_mean(fts; start_time = 0, stop_time = Inf)
    idx = in_window(fts; start_time, stop_time)
    isempty(idx) && error("No snapshots in [$start_time, $stop_time]")
    sz  = size(interior(fts[first(idx)]))
    avg = zeros(sz)
    for n in idx
        avg .+= interior(fts[n])
    end
    return avg ./ length(idx)
end

function compute_monthly_mean(fts, target_months;
                              start_time = 0, stop_time = Inf,
                              reference_date = DateTime(1958, 1, 1))
    dates = [reference_date + Second(round(Int, t)) for t in fts.times]
    idx   = findall(i -> month(dates[i]) in target_months &&
                         start_time <= fts.times[i] <= stop_time,
                    eachindex(dates))
    isempty(idx) && return nothing
    sz  = size(interior(fts[first(idx)]))
    avg = zeros(sz)
    for n in idx
        avg .+= interior(fts[n])
    end
    return avg ./ length(idx)
end

function compute_monthly_means(fts; start_time = 0, stop_time = Inf,
                                reference_date = DateTime(1958, 1, 1))
    idx = in_window(fts; start_time, stop_time)
    isempty(idx) && error("No snapshots in [$start_time, $stop_time]")
    sz     = size(interior(fts[first(idx)]))
    sums   = [zeros(sz) for _ in 1:12]
    counts = zeros(Int, 12)
    for n in idx
        m = month(reference_date + Second(round(Int, fts.times[n])))
        sums[m]  .+= interior(fts[n])
        counts[m] += 1
    end
    return [counts[m] > 0 ? sums[m] ./ counts[m] : nothing for m in 1:12]
end

function compute_mean_and_monthly(fts; start_time = 0, stop_time = Inf,
                                   reference_date = DateTime(1958, 1, 1))
    idx = in_window(fts; start_time, stop_time)
    isempty(idx) && error("No snapshots in [$start_time, $stop_time]")
    sz      = size(interior(fts[first(idx)]))
    total   = zeros(sz)
    monthly = [zeros(sz) for _ in 1:12]
    counts  = zeros(Int, 12)
    for n in idx
        total .+= interior(fts[n])
        m = month(reference_date + Second(round(Int, fts.times[n])))
        monthly[m] .+= interior(fts[n])
        counts[m]   += 1
    end
    mean_out    = total ./ length(idx)
    monthly_out = [counts[m] > 0 ? monthly[m] ./ counts[m] : nothing for m in 1:12]
    return mean_out, monthly_out
end

function cached_download(url; cache_dir = obs_cache_dir,
                              retries = 3,
                              timeout = Inf)
    mkpath(cache_dir)
    path = joinpath(cache_dir, basename(url))
    isfile(path) && return path

    downloader = Downloads.Downloader()
    downloader.easy_hook = (easy, info) ->
        Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_LOW_SPEED_TIME, 0)

    tmp = path * ".part"
    isfile(tmp) && rm(tmp; force = true)

    last_err = nothing
    for attempt in 1:retries
        try
            Downloads.download(url, tmp; timeout, downloader)
            mv(tmp, path; force = true)
            return path
        catch e
            last_err = e
            isfile(tmp) && rm(tmp; force = true)
            if attempt < retries
                delay = 2.0 ^ (attempt - 1)
                @warn "Download failed (attempt $attempt/$retries) — retrying in $(delay)s" url=url error=sprint(showerror, e)
                sleep(delay)
            end
        end
    end
    throw(last_err)
end

# ══════════════════════════════════════════════════════════════
# Grids and masks
# ══════════════════════════════════════════════════════════════

function build_land_mask(grid)
    if grid isa ImmersedBoundaryGrid
        bh = Array(interior(grid.immersed_boundary.bottom_height, :, :, 1))
        return bh .>= 0
    else
        return falses(size(grid, 1), size(grid, 2))
    end
end

function build_ocean_mask_3d(grid)
    Nx, Ny, Nz = size(grid)
    mask = ones(Nx, Ny, Nz)
    if grid isa ImmersedBoundaryGrid
        bh = Array(interior(grid.immersed_boundary.bottom_height, :, :, 1))
        zc = znodes(grid, Center())
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            zc[k] < bh[i, j] && (mask[i, j, k] = 0.0)
        end
    end
    return mask
end

mask_land!(f, land) = (f[land] .= NaN; f)

# ══════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════

function panel!(fig, pos, data;
                title="", colormap=:thermal,
                colorrange=nothing, label="",
                nan_color=:lightgray)
    ax = Axis(fig[pos...]; title)
    kw = isnothing(colorrange) ? (;) : (; colorrange)
    hm = heatmap!(ax, data; colormap, nan_color, kw...)
    Colorbar(fig[pos[1], pos[2]+1], hm; label)
    return ax
end

# ══════════════════════════════════════════════════════════════
# ECCO4 free-surface climatology
# ══════════════════════════════════════════════════════════════

function ecco_ssh_climatology_native(; start_date = DateTime(1992, 1, 1),
                                       end_date   = DateTime(2012, 12, 1),
                                       cache_dir  = obs_cache_dir)
    cache_file = joinpath(cache_dir, "ssh_ecco4_$(year(start_date))_$(year(end_date))_native.jld2")
    dates = start_date:Month(1):end_date
    if isfile(cache_file)
        return JLD2.load(cache_file, "ssh_mean")
    end
    @info "  Computing ECCO4 SSH climatology over $(length(dates)) months (one-time)..."
    first_field = Field(Metadatum(:free_surface; dataset = ECCO4Monthly(), date = first(dates)), CPU())
    ssh_mean    = copy(Array(interior(first_field)))
    for date in dates[2:end]
        f = Field(Metadatum(:free_surface; dataset = ECCO4Monthly(), date), CPU())
        ssh_mean .+= Array(interior(f))
    end
    ssh_mean ./= length(dates)
    JLD2.jldsave(cache_file; ssh_mean)
    return ssh_mean
end

const ECCO_SSH_NATIVE_MEAN_REF = Ref{Any}(nothing)

function ecco_ssh_on_grid(grid; reference_date = DateTime(1992, 1, 1))
    if isnothing(ECCO_SSH_NATIVE_MEAN_REF[])
        ECCO_SSH_NATIVE_MEAN_REF[] = ecco_ssh_climatology_native()
    end
    template = Field(Metadatum(:free_surface; dataset = ECCO4Monthly(), date = reference_date), CPU())
    interior(template) .= ECCO_SSH_NATIVE_MEAN_REF[]
    dst = Field{Center, Center, Nothing}(grid)
    interpolate!(dst, template)
    return dropdims(Array(interior(dst)); dims = 3)
end

# ══════════════════════════════════════════════════════════════
# de Boyer Montégut MLD climatology
# ══════════════════════════════════════════════════════════════

const DBM_MLD_URL = get(ENV, "DBM_MLD_URL", "https://mld.ifremer.fr/data/mld_DR003_c1m_reg2.0.nc")

function centers_to_edges(centers; clamp_to = nothing)
    Δfirst = centers[2] - centers[1]
    Δlast  = centers[end] - centers[end-1]
    edges  = Vector{Float64}(undef, length(centers) + 1)
    edges[1]   = centers[1] - Δfirst / 2
    edges[end] = centers[end] + Δlast / 2
    for i in 2:length(centers)
        edges[i] = (centers[i-1] + centers[i]) / 2
    end
    if !isnothing(clamp_to)
        lo, hi = clamp_to
        edges[1]   = max(edges[1], lo)
        edges[end] = min(edges[end], hi)
    end
    return edges
end

function dbm_mld_climatology_on_grid(grid;
                                     file = get(ENV, "DBM_MLD_FILE", joinpath(obs_cache_dir, basename(DBM_MLD_URL))),
                                     var  = get(ENV, "DBM_MLD_VAR", "mld"))
    if !isfile(file)
        try
            @info "  Downloading dBM MLD climatology from $DBM_MLD_URL"
            file = cached_download(DBM_MLD_URL)
        catch e
            @warn "dBM MLD auto-download failed — skipping reference. Manually download from https://mld.ifremer.fr/Surface_Mixed_Layer_Depth.php and set DBM_MLD_FILE." error=sprint(showerror, e)
            return nothing
        end
    end
    ds = NCDatasets.NCDataset(file)
    mld_raw = Array(ds[var][:, :, :])
    lon_vec = Float64.(Array(ds["lon"][:]))
    lat_vec = Float64.(Array(ds["lat"][:]))
    close(ds)

    mld_raw = Float64.(coalesce.(mld_raw, NaN))
    mld_raw[mld_raw .> 1e8] .= NaN

    lon_edges = centers_to_edges(lon_vec)
    lat_edges = centers_to_edges(lat_vec; clamp_to = (-90, 90))
    Nlon, Nlat, Nm = size(mld_raw)

    src_grid = LatitudeLongitudeGrid(CPU();
        size = (Nlon, Nlat, 1),
        longitude = lon_edges,
        latitude  = lat_edges,
        z = (0, 1))

    src = Field{Center, Center, Nothing}(src_grid)
    dst = Field{Center, Center, Nothing}(grid)
    Nx, Ny = size(grid, 1), size(grid, 2)
    out = Array{Float64, 3}(undef, Nx, Ny, Nm)
    for m in 1:Nm
        clean = replace(mld_raw[:, :, m], NaN => 0.0)
        interior(src) .= reshape(clean, Nlon, Nlat, 1)
        interpolate!(dst, src)
        out[:, :, m] = Array(interior(dst))[:, :, 1]
    end
    return out
end

# ══════════════════════════════════════════════════════════════
# NCEP/NCAR Reanalysis 1 wind-stress climatology
# ══════════════════════════════════════════════════════════════

const NCEP_TAUU_URL = get(ENV, "NCEP_TAUU_URL", "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis.derived/surface_gauss/uflx.sfc.mon.ltm.nc")
const NCEP_TAUV_URL = get(ENV, "NCEP_TAUV_URL", "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis.derived/surface_gauss/vflx.sfc.mon.ltm.nc")

function ncep_wind_stress_on_grid(grid;
                                  tauu_file = get(ENV, "NCEP_TAUU_FILE", joinpath(obs_cache_dir, basename(NCEP_TAUU_URL))),
                                  tauv_file = get(ENV, "NCEP_TAUV_FILE", joinpath(obs_cache_dir, basename(NCEP_TAUV_URL))),
                                  tauu_var  = get(ENV, "NCEP_TAUU_VAR",  "uflx"),
                                  tauv_var  = get(ENV, "NCEP_TAUV_VAR",  "vflx"))
    function ensure(file, url)
        isfile(file) && return file
        @info "  Downloading NCEP wind-stress climatology from $url"
        return cached_download(url)
    end
    try
        tauu_file = ensure(tauu_file, NCEP_TAUU_URL)
        tauv_file = ensure(tauv_file, NCEP_TAUV_URL)
    catch e
        @warn "NCEP auto-download failed — skipping wind-stress reference. Provide netCDFs via NCEP_TAUU_FILE / NCEP_TAUV_FILE." error=sprint(showerror, e)
        return nothing, nothing
    end

    function read_stress(file, var)
        ds = NCDatasets.NCDataset(file)
        raw = Array(ds[var])
        lon_vec = Float64.(Array(ds["lon"][:]))
        lat_vec = Float64.(Array(ds["lat"][:]))
        close(ds)
        for d in reverse(findall(==(1), size(raw)))
            raw = dropdims(raw; dims=d)
        end
        return Float64.(coalesce.(raw, NaN)), lon_vec, lat_vec
    end

    τu_raw, lon_vec, lat_vec = read_stress(tauu_file, tauu_var)
    τv_raw, _, _             = read_stress(tauv_file, tauv_var)

    annual_mean(f) = ndims(f) == 3 ? dropdims(mean(f, dims=3), dims=3) : f
    τx_2d = -annual_mean(τu_raw)
    τy_2d = -annual_mean(τv_raw)

    if lat_vec[1] > lat_vec[end]
        lat_vec = reverse(lat_vec)
        τx_2d   = reverse(τx_2d; dims=2)
        τy_2d   = reverse(τy_2d; dims=2)
    end

    lon_edges = centers_to_edges(lon_vec)
    lat_edges = centers_to_edges(lat_vec; clamp_to = (-90, 90))
    Nlon, Nlat = length(lon_vec), length(lat_vec)

    src_grid = LatitudeLongitudeGrid(CPU();
        size = (Nlon, Nlat, 1),
        longitude = lon_edges,
        latitude  = lat_edges,
        z = (0, 1))

    src = Field{Center, Center, Nothing}(src_grid)
    dst = Field{Center, Center, Nothing}(grid)
    function regrid(data_2d)
        clean = replace(data_2d, NaN => 0.0)
        interior(src) .= reshape(clean, Nlon, Nlat, 1)
        interpolate!(dst, src)
        return dropdims(Array(interior(dst)); dims=3)
    end
    return regrid(τx_2d), regrid(τy_2d)
end

# ══════════════════════════════════════════════════════════════
# Postprocess MLD with a configurable reference depth (vertical interp only)
# ══════════════════════════════════════════════════════════════

"""
    mld_with_reference_depth(buoyancy, grid;
                             reference_depth = 10.0,
                             Δb★ = 2.87e-4)

Compute mixed-layer depth at every (i, j) by vertically interpolating
the buoyancy field at `z = -reference_depth` and walking down the
column until the buoyancy drop relative to that reference exceeds
`Δb★` (de Boyer Montégut DR003 default at ρ₀ = 1025 kg/m³). Returns a
2-D array of MLD in meters; NaN where the column is land.

Pure column-wise computation — no horizontal interpolation needed
because the z-coordinate is the same at every (i, j).
"""
function mld_with_reference_depth(buoyancy::AbstractArray{<:Any, 3}, grid;
                                  reference_depth = 10.0,
                                  Δb★ = 2.87e-4)
    Nx, Ny, Nz = size(buoyancy)
    z   = collect(znodes(grid, Center()))
    zʳ  = -reference_depth
    mld = fill(NaN, Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        bN = buoyancy[i, j, Nz]
        (isnan(bN) || bN == 0) && continue

        # Bracket zʳ between cell centers k⁻ (deeper) and k⁺ (shallower).
        k⁺ = Nz
        while k⁺ > 1 && z[k⁺ - 1] >= zʳ
            k⁺ -= 1
        end
        bʳ = if k⁺ == 1
            bN
        else
            k⁻ = k⁺ - 1
            w  = (zʳ - z[k⁻]) / (z[k⁺] - z[k⁻])
            buoyancy[i, j, k⁻] * (1 - w) + buoyancy[i, j, k⁺] * w
        end

        # Walk down from zʳ; linearly interpolate where Δb crosses Δb★.
        zₚ  = zʳ
        Δbₚ = zero(eltype(buoyancy))
        z★  = NaN
        for k in (Nz - 1):-1:1
            z[k] >= zʳ && continue
            bk = buoyancy[i, j, k]
            (isnan(bk) || bk == 0) && break
            Δb = bʳ - bk
            if Δb >= Δb★
                z★ = zₚ + (Δb★ - Δbₚ) / (Δb - Δbₚ) * (z[k] - zₚ)
                break
            end
            zₚ = z[k]; Δbₚ = Δb
        end

        mld[i, j] = isnan(z★) ? -zₚ : -z★
    end

    return mld
end

# ══════════════════════════════════════════════════════════════
# Sea-ice integral helpers (used by the :ice_diag loader)
# ══════════════════════════════════════════════════════════════

arctic_condition(i, j, k, grid, args...)    = φnode(i, j, k, grid, Center(), Center(), Center()) > 0
antarctic_condition(i, j, k, grid, args...) = φnode(i, j, k, grid, Center(), Center(), Center()) < 0

function compute_ice_diagnostics(run_dir, prefix, grid;
                                 start_time = 0, stop_time = Inf,
                                 reference_date = DateTime(1958, 1, 1),
                                 extent_threshold = 0.15)
    surface_file      = find_first_file(run_dir, prefix, "surface")
    thickness_fts     = FieldTimeSeries(surface_file, "sithick"; backend = deepcopy(FTS_BACKEND))
    concentration_fts = FieldTimeSeries(surface_file, "siconc";  backend = deepcopy(FTS_BACKEND))

    Nt = length(thickness_fts.times)
    arctic_volume      = zeros(Nt)
    antarctic_volume   = zeros(Nt)
    arctic_extent      = zeros(Nt)
    antarctic_extent   = zeros(Nt)
    arctic_area        = zeros(Nt)
    antarctic_area     = zeros(Nt)
    snapshot_dates     = [reference_date + Second(round(Int, t)) for t in thickness_fts.times]

    thickness     = Field{Center, Center, Nothing}(grid)
    concentration = Field{Center, Center, Nothing}(grid)
    ice_volume    = Field{Center, Center, Nothing}(grid)
    extent_mask   = Field{Center, Center, Nothing}(grid)

    arctic_vol_int     = Field(Integral(ice_volume;    condition = arctic_condition))
    antarctic_vol_int  = Field(Integral(ice_volume;    condition = antarctic_condition))
    arctic_area_int    = Field(Integral(concentration; condition = arctic_condition))
    antarctic_area_int = Field(Integral(concentration; condition = antarctic_condition))
    arctic_ext_int     = Field(Integral(extent_mask;   condition = arctic_condition))
    antarctic_ext_int  = Field(Integral(extent_mask;   condition = antarctic_condition))

    for n in 1:Nt
        set!(thickness,     thickness_fts[n])
        set!(concentration, concentration_fts[n])
        interior(ice_volume) .= interior(thickness) .* interior(concentration)

        compute!(arctic_vol_int);  compute!(antarctic_vol_int)
        arctic_volume[n]    = arctic_vol_int[1, 1, 1]
        antarctic_volume[n] = antarctic_vol_int[1, 1, 1]

        compute!(arctic_area_int); compute!(antarctic_area_int)
        arctic_area[n]    = arctic_area_int[1, 1, 1]
        antarctic_area[n] = antarctic_area_int[1, 1, 1]

        concentration_data = Array(interior(concentration, :, :, 1))
        set!(extent_mask, Float64.(concentration_data .> extent_threshold))
        compute!(arctic_ext_int); compute!(antarctic_ext_int)
        arctic_extent[n]    = arctic_ext_int[1, 1, 1]
        antarctic_extent[n] = antarctic_ext_int[1, 1, 1]
    end

    idx = findall(t -> start_time <= t <= stop_time, thickness_fts.times)
    months_used = month.(snapshot_dates[idx])
    monthly(field) = [mean(field[idx[months_used .== m]]) for m in 1:12]

    return (; arctic_volume, antarctic_volume,
              arctic_extent, antarctic_extent,
              arctic_area, antarctic_area, snapshot_dates,
              arctic_volume_monthly    = monthly(arctic_volume),
              antarctic_volume_monthly = monthly(antarctic_volume),
              arctic_extent_monthly    = monthly(arctic_extent),
              antarctic_extent_monthly = monthly(antarctic_extent),
              arctic_area_monthly      = monthly(arctic_area),
              antarctic_area_monthly   = monthly(antarctic_area))
end

# ══════════════════════════════════════════════════════════════
# Observational sea-ice climatologies (NSIDC, PIOMAS) — global cache.
# ══════════════════════════════════════════════════════════════

function load_piomas_monthly()
    url   = "https://psc.apl.uw.edu/wordpress/wp-content/uploads/schweiger/ice_volume/PIOMAS.monthly.Current.v2.1.csv"
    raw   = readdlm(cached_download(url), ','; skipstart=1)
    vol   = Float64.(raw[:, 2:13])
    vol[vol .== -1] .= NaN
    return vec(mapslices(x -> mean(filter(!isnan, x)), vol; dims=1))
end

function load_nsidc(hemisphere)
    prefix = hemisphere == "north" ? "N" : "S"
    extent_monthly = zeros(12)
    area_monthly   = zeros(12)
    for m in 1:12
        url = "https://noaadata.apps.nsidc.org/NOAA/G02135/$(hemisphere)/monthly/data/$(prefix)_$(lpad(m, 2, '0'))_extent_v4.0.csv"
        raw = readlines(cached_download(url))
        extents = Float64[]; areas = Float64[]
        for line in raw
            parts = split(line, ',')
            length(parts) >= 6 || continue
            ext = tryparse(Float64, strip(parts[5]))
            ar  = tryparse(Float64, strip(parts[6]))
            (isnothing(ext) || ext == -9999) && continue
            (isnothing(ar)  || ar  == -9999) && continue
            push!(extents, ext); push!(areas, ar)
        end
        extent_monthly[m] = mean(extents)
        area_monthly[m]   = mean(areas)
    end
    return (; extent_monthly, area_monthly)
end

# Global per-process cache for observational climatologies.
const _NSIDC_NORTH_REF = Ref{Any}(nothing)
const _NSIDC_SOUTH_REF = Ref{Any}(nothing)
const _PIOMAS_REF      = Ref{Any}(nothing)

nsidc_arctic()    = (isnothing(_NSIDC_NORTH_REF[]) && (_NSIDC_NORTH_REF[] = load_nsidc("north"));    _NSIDC_NORTH_REF[])
nsidc_antarctic() = (isnothing(_NSIDC_SOUTH_REF[]) && (_NSIDC_SOUTH_REF[] = load_nsidc("south"));    _NSIDC_SOUTH_REF[])
piomas_monthly()  = (isnothing(_PIOMAS_REF[])      && (_PIOMAS_REF[]      = load_piomas_monthly()); _PIOMAS_REF[])
