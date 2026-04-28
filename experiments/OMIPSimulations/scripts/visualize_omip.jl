#!/usr/bin/env julia
# visualize_omip.jl -- Generate all OMIP diagnostic figures as PNGs.
#
# Usage:
#     julia --project=.. visualize_omip.jl [output_dir]
#
# Edit the `cases` below before running. Each case carries its own
# averaging window via `start_time` and `stop_time`.

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

const years = 365 * 24 * 3600

# Ocean reference density and heat capacity used by `ocean_simulation`
# (TEOS-10 defaults). The diagnostics `hfds`, `tauuo`, `tauvo` are stored
# in kinematic units (m·K/s for heat, m²/s² for stress) because the
# net-flux kernel divides by ρ and ρ·cp before writing the tracer/momentum
# tendency. Multiply on read to recover CMIP-style W/m² and N/m².
const ρ_ocean  = 1026.0
const cp_ocean = 3991.86795711963

# Each case is identified by its `prefix` (used to build the run directory
# as `$(prefix)_run` and to match output filenames), a human-readable
# `label`, and a per-case averaging window in seconds (`start_time`,
# `stop_time`). Any number of cases is supported, with per-case heatmap
# figures laying out one column per case.
cases = [
    (prefix = "orca_corrected_snow_cb0.12_ksymm500",            label = "ORCA GM500",  start_time = 15 * years, stop_time = Inf),
    (prefix = "halfdegree_corrected_snow_cb0.01_kskew0_ksymm0", label = "Half Degree", start_time = 15 * years, stop_time = Inf),
    (prefix = "orca_corrected_snow_cb0.06_kskew0_ksymm0",       label = "ORCA NOGM",   start_time = 15 * years, stop_time = Inf),
    (prefix = "orca_corrected_snow_cb0.06_kskew1000_ksymm1000", label = "ORCA GM1000", start_time = 15 * years, stop_time = Inf),
]

run_dir_for(prefix) = "$(prefix)_run"

output_dir = length(ARGS) >= 1 ? ARGS[1] : "figures"

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
# Oceananigans 0.107.3 bug (field_time_series.jl:914-918): the
# inner FieldTimeSeries constructor only builds a SplitFilePath
# when `backend isa OnDisk`. With an InMemory backend on split
# output (..._part1.jld2, ..._part2.jld2, ...), fts.path collapses
# to a single part file. The construction-time load iterates over
# every part file, so the initial window is correct — but later,
# when update_field_time_series! slides the in-memory window
# (set!(fts) -> set!(fts, fts.path)), reads come from that one
# stored file only, silently leaving stale/zero data in every
# slot whose time is not in that part. That's the cause of the
# sawtooth KE, flat MLD extrema, and wrong SST bias in figures6.
#
# Upstream fix (for PR):
#   1. field_time_series.jl:914 — drop `&& backend isa OnDisk`.
#   2. set_field_time_series.jl — add a set! method for
#      (InMemoryFTS, SplitFilePath) that dispatches per part file.
#
# This block monkey-patches both until the PR lands.
import Oceananigans.Fields: set!
using Oceananigans.OutputReaders: SplitFilePath, InMemoryFTS,
                                  InMemory, time_indices,
                                  file_and_local_index

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

function _rebuild_fts_with_path(fts, new_path)
    LX, LY, LZ = _location_types(fts)
    return Oceananigans.OutputReaders.FieldTimeSeries{LX, LY, LZ}(
        fts.data,
        fts.grid,
        fts.backend,
        fts.boundary_conditions,
        fts.indices,
        fts.times,
        new_path,
        fts.name,
        fts.time_indexing,
        fts.reader_kw,
    )
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

mkpath(output_dir)
@info "Figures will be saved to: $output_dir"

# Cache for observational downloads + derived climatologies
obs_cache_dir = joinpath(output_dir, "obs_cache")
mkpath(obs_cache_dir)

# ══════════════════════════════════════════════════════════════
# Helpers
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

# Single-pass variant: return 12 monthly means (slot m = nothing if empty).
# Reads the FTS once and bins each snapshot into its calendar month.
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

# Fused pass: global time mean + 12 monthly means, one sweep through the FTS.
# Use this when the same field feeds both a climatological mean and a
# seasonal stratification (e.g. SIC for both time-mean map and March/September).
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

function cached_download(url; cache_dir = obs_cache_dir)
    mkpath(cache_dir)
    path = joinpath(cache_dir, basename(url))
    isfile(path) || Downloads.download(url, path)
    return path
end

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

# Per-case line colors for 1-D overlay plots. The palette cycles if there
# are more cases than base colors, so `case_colors[i]` is always valid.
const _BASE_CASE_COLORS = [:firebrick, :royalblue, :seagreen, :darkorange,
                           :purple, :teal, :goldenrod, :saddlebrown,
                           :magenta, :olive]
case_colors = [_BASE_CASE_COLORS[mod1(i, length(_BASE_CASE_COLORS))]
               for i in 1:length(cases)]

savefig(fig, name) = save(joinpath(output_dir, name), fig)

# ── ECCO4 free-surface climatology ────────────────────────────
# Cache the native-grid mean to avoid re-reading 252 monthly files on every run.
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

# Native ECCO mean is grid-invariant across cases — compute once, reuse.
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

# ── de Boyer Montégut MLD climatology ─────────────────────────
# Auto-downloads the IFREMER product (2° monthly climatology, DR003 density
# criterion). Override DBM_MLD_URL / DBM_MLD_FILE / DBM_MLD_VAR to point at a
# different product. Missing values appear either as Missing or as sentinel
# values (up to ~1e9); both are mapped out before interpolation.
const DBM_MLD_URL = get(ENV, "DBM_MLD_URL",
                        "https://mld.ifremer.fr/data/mld_DR003_c1m_reg2.0.nc")

# Cell-edge vector from center vector: midpoints plus half-spacing at ends.
# Clamped to ±90 for latitude to avoid constructor errors at the poles.
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
    mld_raw[mld_raw .> 1e8] .= NaN  # dBM DR003 uses ~1e9 as a sentinel on land

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
        # `interpolate!` doesn't propagate NaNs usefully across land, so replace
        # missing/sentinel cells with 0 before interpolation. Downstream we mask
        # land anyway.
        clean = replace(mld_raw[:, :, m], NaN => 0.0)
        interior(src) .= reshape(clean, Nlon, Nlat, 1)
        interpolate!(dst, src)
        out[:, :, m] = Array(interior(dst))[:, :, 1]
    end
    return out
end

# ── NCEP/NCAR Reanalysis 1 wind-stress climatology ───────────
# Long-term monthly mean (1991-2020) of surface momentum flux from the
# NCEP/NCAR Reanalysis 1, hosted at NOAA PSL on the T62 Gaussian grid.
# Variables `uflx`/`vflx` are upward momentum fluxes (positive away from
# the surface), so the atmosphere-to-ocean stress is the negative of the
# stored value. Override the URLs or local file paths via the env vars
# below; latitudes are reordered ascending if the file stores them N→S.
const NCEP_TAUU_URL = get(ENV, "NCEP_TAUU_URL", "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface_gauss/uflx.sfc.mon.ltm.1991-2020.nc")
const NCEP_TAUV_URL = get(ENV, "NCEP_TAUV_URL", "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface_gauss/vflx.sfc.mon.ltm.1991-2020.nc")

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
        # Drop singleton dims (some NCEP files include a length-1 level axis).
        for d in reverse(findall(==(1), size(raw)))
            raw = dropdims(raw; dims=d)
        end
        return Float64.(coalesce.(raw, NaN)), lon_vec, lat_vec
    end

    τu_raw, lon_vec, lat_vec = read_stress(tauu_file, tauu_var)
    τv_raw, _, _             = read_stress(tauv_file, tauv_var)

    # NCEP convention: positive upward (atmosphere gains momentum), so
    # τ_atm→ocean is the negative of the stored flux.
    annual_mean(f) = ndims(f) == 3 ? dropdims(mean(f, dims=3), dims=3) : f
    τx_2d = -annual_mean(τu_raw)
    τy_2d = -annual_mean(τv_raw)

    # Gaussian latitudes are usually stored descending; flip to ascending.
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
# Load surface diagnostics
# ══════════════════════════════════════════════════════════════

function load_surface_case(run_dir, prefix; start_time = 0, stop_time = Inf)
    surface_file = find_first_file(run_dir, prefix, "surface")
    @info "  surface: $surface_file"

    tos     = FieldTimeSeries(surface_file, "tos";    backend = InMemory(10))
    sos     = FieldTimeSeries(surface_file, "sos";    backend = InMemory(10))
    zos     = FieldTimeSeries(surface_file, "zos";    backend = InMemory(10))
    mld_fts = FieldTimeSeries(surface_file, "mlotst"; backend = InMemory(10))
    hfds    = FieldTimeSeries(surface_file, "hfds";   backend = InMemory(10))
    wfo     = FieldTimeSeries(surface_file, "wfo";    backend = InMemory(10))
    sic     = FieldTimeSeries(surface_file, "siconc"; backend = InMemory(10))
    zossq   = FieldTimeSeries(surface_file, "zossq";  backend = InMemory(10))
    tauuo   = FieldTimeSeries(surface_file, "tauuo";  backend = InMemory(10))
    tauvo   = FieldTimeSeries(surface_file, "tauvo";  backend = InMemory(10))

    grid = tos.grid
    Nx, Ny, Nz = size(grid)
    land = build_land_mask(grid)

    @info "  averaging window: [$(start_time / (365.25*86400)), $(stop_time / (365.25*86400))] years"

    SST = dropdims(compute_time_mean(tos;  start_time, stop_time);  dims=3)
    SSS = dropdims(compute_time_mean(sos;  start_time, stop_time);  dims=3)
    SSH = dropdims(compute_time_mean(zos;  start_time, stop_time);  dims=3)
    # `hfds` stores a kinematic temperature flux (m·K/s); scale by ρ·cp to W/m².
    # `tauuo`/`tauvo` store kinematic stresses (m²/s²); scale by ρ to N/m².
    HF  = dropdims(compute_time_mean(hfds;  start_time, stop_time); dims=3) .* (ρ_ocean * cp_ocean)
    FW  = dropdims(compute_time_mean(wfo;   start_time, stop_time); dims=3)
    τx  = dropdims(compute_time_mean(tauuo; start_time, stop_time); dims=3) .* ρ_ocean
    τy  = dropdims(compute_time_mean(tauvo; start_time, stop_time); dims=3) .* ρ_ocean

    SSH_sq  = dropdims(compute_time_mean(zossq; start_time, stop_time); dims=3)
    SSH_var = SSH_sq .- SSH .^ 2

    # SIC mean + monthly bins in one FTS pass (halves reads for `sic`).
    SIC_mean_raw, sic_monthly = compute_mean_and_monthly(sic; start_time, stop_time)
    SIC_mean = dropdims(SIC_mean_raw; dims=3)

    # MLD only needs monthly bins (min/max across months).
    mld_monthly = compute_monthly_means(mld_fts; start_time, stop_time)

    mld_available = findall(!isnothing, mld_monthly)
    MLD_stack = cat([dropdims(mld_monthly[m]; dims=3) for m in mld_available]...; dims=3)
    MLD_min   = dropdims(minimum(MLD_stack; dims=3); dims=3)
    MLD_max   = dropdims(maximum(MLD_stack; dims=3); dims=3)

    SIC_mar = isnothing(sic_monthly[3]) ? nothing : dropdims(sic_monthly[3]; dims=3)
    SIC_sep = isnothing(sic_monthly[9]) ? nothing : dropdims(sic_monthly[9]; dims=3)

    T_woa = Field(Metadatum(:temperature; dataset = WOAAnnual()), CPU())
    S_woa = Field(Metadatum(:salinity;    dataset = WOAAnnual()), CPU())
    T_interp = CenterField(grid); interpolate!(T_interp, T_woa)
    S_interp = CenterField(grid); interpolate!(S_interp, S_woa)
    T_woa_on_grid = Array(interior(T_interp))
    S_woa_on_grid = Array(interior(S_interp))
    δSST = SST .- T_woa_on_grid[:, :, Nz]
    δSSS = SSS .- S_woa_on_grid[:, :, Nz]

    # ECCO4 free-surface climatology (1992–2012) regridded onto case grid.
    # Simulated and ECCO SSH have different mean offsets (each model defines
    # its own reference); subtract global means before differencing.
    SSH_ecco = ecco_ssh_on_grid(grid)
    δSSH_ecco = (SSH .- mean(filter(isfinite, SSH))) .-
                (SSH_ecco .- mean(filter(isfinite, SSH_ecco)))

    # de Boyer Montégut monthly MLD climatology (optional — set DBM_MLD_FILE).
    dbm_mld = dbm_mld_climatology_on_grid(grid)
    MLD_min_dbm = isnothing(dbm_mld) ? nothing : dropdims(minimum(dbm_mld; dims=3); dims=3)
    MLD_max_dbm = isnothing(dbm_mld) ? nothing : dropdims(maximum(dbm_mld; dims=3); dims=3)

    # NCEP/NCAR Reanalysis wind-stress climatology (optional — override URLs/files via NCEP_* env vars).
    τx_ncep, τy_ncep = ncep_wind_stress_on_grid(grid)
    δτx_ncep = isnothing(τx_ncep) ? nothing : τx .- τx_ncep
    δτy_ncep = isnothing(τy_ncep) ? nothing : τy .- τy_ncep

    for f in (SST, SSS, SSH, HF, FW, SIC_mean, SSH_var, MLD_min, MLD_max,
              δSST, δSSS, SSH_ecco, δSSH_ecco, τx, τy)
        mask_land!(f, land)
    end
    !isnothing(SIC_mar)     && mask_land!(SIC_mar, land)
    !isnothing(SIC_sep)     && mask_land!(SIC_sep, land)
    !isnothing(MLD_min_dbm) && mask_land!(MLD_min_dbm, land)
    !isnothing(MLD_max_dbm) && mask_land!(MLD_max_dbm, land)
    !isnothing(τx_ncep)     && mask_land!(τx_ncep,  land)
    !isnothing(τy_ncep)     && mask_land!(τy_ncep,  land)
    !isnothing(δτx_ncep)    && mask_land!(δτx_ncep, land)
    !isnothing(δτy_ncep)    && mask_land!(δτy_ncep, land)

    return (; grid, Nx, Ny, Nz, land, surface_file,
              SST, SSS, SSH, HF, FW, SIC_mean, SSH_var,
              MLD_min, MLD_max, SIC_mar, SIC_sep,
              δSST, δSSS, SSH_ecco, δSSH_ecco,
              MLD_min_dbm, MLD_max_dbm,
              τx, τy, τx_ncep, τy_ncep, δτx_ncep, δτy_ncep,
              T_woa_on_grid, S_woa_on_grid)
end

D = Dict{String, Any}()
labels = [c.label for c in cases]
for c in cases
    @info "Loading surface: $(c.label)..."
    D[c.label] = load_surface_case(run_dir_for(c.prefix), c.prefix;
                                    start_time = c.start_time, stop_time = c.stop_time)
end

# ══════════════════════════════════════════════════════════════
# Figures 1-8: Surface diagnostics
# ══════════════════════════════════════════════════════════════

# Grid of per-case maps of a scalar field accessed by `getfield(D[lab], key)`.
function plot_field_grid(key::Symbol;
                         title_suffix::String, colormap, colorrange, label::String,
                         figsize = (800 * length(labels), 500))
    fig = Figure(size = figsize, fontsize = 14)
    for (i, lab) in enumerate(labels)
        panel!(fig, [1, 2i-1], getfield(D[lab], key);
               title = "$lab: $title_suffix", colormap, colorrange, label)
    end
    return fig
end

# Figure 1: SST bias
@info "Figure 1: SST bias"
savefig(plot_field_grid(:δSST;  title_suffix = "SST - WOA", colormap = :balance,
                        colorrange = (-5, 5), label = "deg C"),
        "fig01_sst_bias.png")

# Figure 2: SSS bias
@info "Figure 2: SSS bias"
savefig(plot_field_grid(:δSSS;  title_suffix = "SSS - WOA", colormap = :balance,
                        colorrange = (-3, 3), label = "PSU"),
        "fig02_sss_bias.png")

# Figure 3: SSH + SSH - ECCO(1992–2012) bias
@info "Figure 3: SSH and SSH - ECCO bias"
fig = Figure(size = (800 * length(labels), 900), fontsize = 14)
for (i, lab) in enumerate(labels)
    panel!(fig, [1, 2i-1], D[lab].SSH;
           title = "$lab: Time-mean SSH", colormap = :balance,
           colorrange = (-2, 2), label = "m")
    panel!(fig, [2, 2i-1], D[lab].δSSH_ecco;
           title = "$lab: SSH - ECCO (1992–2012), demeaned", colormap = :balance,
           colorrange = (-0.5, 0.5), label = "m")
end
savefig(fig, "fig03_ssh.png")

# Figure 4: MLD min/max with optional dBM reference row
@info "Figure 4: MLD"
lab_with_dbm = findfirst(lab -> !isnothing(D[lab].MLD_min_dbm), labels)
ncases = length(labels)
# With ≥ 2 cases, fit dBM Min and Max side-by-side in one row; with a single
# case, stack them as two separate rows.
nrows = if isnothing(lab_with_dbm)
    2
elseif ncases >= 2
    3
else
    4
end
fig = Figure(size = (800 * ncases, 450 * nrows), fontsize = 14)
for (i, lab) in enumerate(labels)
    panel!(fig, [1, 2i-1], D[lab].MLD_min;
           title = "$lab: Min MLD (summer)",
           colormap = Reverse(:deep), colorrange = (0, 70), label = "m")
    panel!(fig, [2, 2i-1], D[lab].MLD_max;
           title = "$lab: Max MLD (winter)",
           colormap = Reverse(:deep), colorrange = (0, 500), label = "m")
end
if !isnothing(lab_with_dbm)
    ref_lab = labels[lab_with_dbm]
    min_pos = [3, 1]
    max_pos = ncases >= 2 ? [3, 3] : [4, 1]
    panel!(fig, min_pos, D[ref_lab].MLD_min_dbm;
           title = "dBM climatology: Min MLD",
           colormap = Reverse(:deep), colorrange = (0, 70), label = "m")
    panel!(fig, max_pos, D[ref_lab].MLD_max_dbm;
           title = "dBM climatology: Max MLD",
           colormap = Reverse(:deep), colorrange = (0, 500), label = "m")
end
savefig(fig, "fig04_mld.png")

# Figure 5: Sea-ice concentration
@info "Figure 5: Sea-ice concentration"
fig = Figure(size = (800 * length(labels), 900), fontsize = 14)
for (i, lab) in enumerate(labels)
    d = D[lab]
    !isnothing(d.SIC_mar) && panel!(fig, [1, 2i-1], d.SIC_mar;
        title = "$lab: Sea-ice conc. March",
        colormap = :ice, colorrange = (0, 1), label = "fraction")
    !isnothing(d.SIC_sep) && panel!(fig, [2, 2i-1], d.SIC_sep;
        title = "$lab: Sea-ice conc. September",
        colormap = :ice, colorrange = (0, 1), label = "fraction")
end
savefig(fig, "fig05_seaice_conc.png")

# Figure 6: Surface fluxes
@info "Figure 6: Surface fluxes"
fig = Figure(size = (800 * length(labels), 900), fontsize = 14)
for (i, lab) in enumerate(labels)
    panel!(fig, [1, 2i-1], D[lab].HF;
           title = "$lab: Net heat flux", colormap = :balance,
           colorrange = (-200, 200), label = "W/m^2")
    panel!(fig, [2, 2i-1], D[lab].FW;
           title = "$lab: Net freshwater flux", colormap = :balance,
           colorrange = (-1e-4, 1e-4), label = "kg/m^2/s")
end
savefig(fig, "fig06_surface_fluxes.png")

# Figure 7: Wind stress and NCEP bias
@info "Figure 7: Wind stress and NCEP bias"
has_ncep = any(lab -> !isnothing(D[lab].δτx_ncep), labels)
nrows = has_ncep ? 4 : 2
fig = Figure(size = (800 * length(labels), 450 * nrows), fontsize = 14)
for (i, lab) in enumerate(labels)
    d = D[lab]
    panel!(fig, [1, 2i-1], d.τx;
           title = "$lab: Zonal wind stress", colormap = :balance,
           colorrange = (-0.3, 0.3), label = "N/m²")
    panel!(fig, [2, 2i-1], d.τy;
           title = "$lab: Meridional wind stress", colormap = :balance,
           colorrange = (-0.3, 0.3), label = "N/m²")
    if has_ncep
        !isnothing(d.δτx_ncep) && panel!(fig, [3, 2i-1], d.δτx_ncep;
            title = "$lab: τx - NCEP", colormap = :balance,
            colorrange = (-0.15, 0.15), label = "N/m²")
        !isnothing(d.δτy_ncep) && panel!(fig, [4, 2i-1], d.δτy_ncep;
            title = "$lab: τy - NCEP", colormap = :balance,
            colorrange = (-0.15, 0.15), label = "N/m²")
    end
end
savefig(fig, "fig07_wind_stress.png")

# Figure 8: SSH variance
@info "Figure 8: SSH variance"
savefig(plot_field_grid(:SSH_var; title_suffix = "SSH variance", colormap = :magma,
                        colorrange = (0, 0.05), label = "m²"),
        "fig08_ssh_variance.png")

# ══════════════════════════════════════════════════════════════
# Sea-ice diagnostics
# ══════════════════════════════════════════════════════════════

arctic_condition(i, j, k, grid, args...)    = φnode(i, j, k, grid, Center(), Center(), Center()) > 0
antarctic_condition(i, j, k, grid, args...) = φnode(i, j, k, grid, Center(), Center(), Center()) < 0

function compute_ice_diagnostics(run_dir, prefix, grid;
                                 start_time = 0, stop_time = Inf,
                                 reference_date = DateTime(1958, 1, 1),
                                 extent_threshold = 0.15)
    surface_file      = find_first_file(run_dir, prefix, "surface")
    thickness_fts     = FieldTimeSeries(surface_file, "sithick"; backend = InMemory(10))
    concentration_fts = FieldTimeSeries(surface_file, "siconc";  backend = InMemory(10))

    Nt = length(thickness_fts.times)
    arctic_volume      = zeros(Nt)
    antarctic_volume   = zeros(Nt)
    arctic_extent      = zeros(Nt)
    antarctic_extent   = zeros(Nt)
    arctic_area        = zeros(Nt)
    antarctic_area     = zeros(Nt)
    snapshot_dates     = [reference_date + Second(round(Int, t)) for t in thickness_fts.times]

    # Reusable operand buffers. Integrals below reference these by identity, so
    # mutating their data in place and calling `compute!` re-evaluates against
    # the new values — no per-iteration Field/Integral allocation.
    thickness     = Field{Center, Center, Nothing}(grid)
    concentration = Field{Center, Center, Nothing}(grid)
    ice_volume    = Field{Center, Center, Nothing}(grid)
    extent_mask   = Field{Center, Center, Nothing}(grid)

    arctic_vol_int    = Field(Integral(ice_volume;    condition = arctic_condition))
    antarctic_vol_int = Field(Integral(ice_volume;    condition = antarctic_condition))
    arctic_area_int   = Field(Integral(concentration; condition = arctic_condition))
    antarctic_area_int = Field(Integral(concentration; condition = antarctic_condition))
    arctic_ext_int    = Field(Integral(extent_mask;   condition = arctic_condition))
    antarctic_ext_int = Field(Integral(extent_mask;   condition = antarctic_condition))

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

ICE = Dict{String, Any}()
let ice_futures = [(c.label,
                    Threads.@spawn compute_ice_diagnostics(run_dir_for(c.prefix), c.prefix, D[c.label].grid;
                                                            start_time = c.start_time,
                                                            stop_time  = c.stop_time))
                   for c in cases]
    for (lab, fut) in ice_futures
        @info "Computing sea-ice diagnostics for $lab..."
        ICE[lab] = fetch(fut)
    end
end

# ── Download observational climatologies ─────────────────────

piomas_url  = "https://psc.apl.uw.edu/wordpress/wp-content/uploads/schweiger/ice_volume/PIOMAS.monthly.Current.v2.1.csv"
piomas_raw  = readdlm(cached_download(piomas_url), ','; skipstart=1)
piomas_volume = Float64.(piomas_raw[:, 2:13])
piomas_volume[piomas_volume .== -1] .= NaN
piomas_monthly = vec(mapslices(x -> mean(filter(!isnan, x)), piomas_volume; dims=1))

function download_nsidc(hemisphere)
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

@info "Downloading NSIDC..."
nsidc_arctic    = download_nsidc("north")
nsidc_antarctic = download_nsidc("south")

# ── Figures 8-12: Sea-ice climatologies and time series ──────

month_names  = ["J","F","M","A","M","J","J","A","S","O","N","D"]
m2_to_Mkm2   = 1e-12
m3_to_1e3km3 = 1e-12

# Figure 9: SIE
@info "Figure 9: SIE"
fig = Figure(size = (1200, 500), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Month", ylabel="SIE (Million km²)", title="Arctic SIE Climatology", xticks=(1:12, month_names))
lines!(ax, 1:12, nsidc_arctic.extent_monthly; color=:black, linewidth=2, label="NSIDC")
for (i, lab) in enumerate(labels)
    lines!(ax, 1:12, ICE[lab].arctic_extent_monthly .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:lb)
ax = Axis(fig[1, 2]; xlabel="Month", ylabel="SIE (Million km²)", title="Antarctic SIE Climatology", xticks=(1:12, month_names))
lines!(ax, 1:12, nsidc_antarctic.extent_monthly; color=:black, linewidth=2, label="NSIDC")
for (i, lab) in enumerate(labels)
    lines!(ax, 1:12, ICE[lab].antarctic_extent_monthly .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
savefig(fig, "fig09_sie.png")

# Figure 10: SIA
@info "Figure 10: SIA"
fig = Figure(size = (1200, 500), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Month", ylabel="SIA (Million km²)", title="Arctic SIA Climatology", xticks=(1:12, month_names))
lines!(ax, 1:12, nsidc_arctic.area_monthly; color=:black, linewidth=2, label="NSIDC")
for (i, lab) in enumerate(labels)
    lines!(ax, 1:12, ICE[lab].arctic_area_monthly .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:lb)
ax = Axis(fig[1, 2]; xlabel="Month", ylabel="SIA (Million km²)", title="Antarctic SIA Climatology", xticks=(1:12, month_names))
lines!(ax, 1:12, nsidc_antarctic.area_monthly; color=:black, linewidth=2, label="NSIDC")
for (i, lab) in enumerate(labels)
    lines!(ax, 1:12, ICE[lab].antarctic_area_monthly .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
savefig(fig, "fig10_sia.png")

# Figure 11: Arctic volume
@info "Figure 11: Arctic volume"
fig = Figure(size = (600, 500), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Month", ylabel="Ice volume (10³ km³)", title="Arctic sea-ice volume", xticks=(1:12, month_names))
lines!(ax, 1:12, piomas_monthly; color=:black, linewidth=2, label="PIOMAS")
for (i, lab) in enumerate(labels)
    lines!(ax, 1:12, ICE[lab].arctic_volume_monthly .* m3_to_1e3km3; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
savefig(fig, "fig11_arctic_volume.png")

# Figure 12: SIA time series
@info "Figure 12: SIA time series"
fig = Figure(size = (1200, 500), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Time (years)", ylabel="SIA (Million km²)", title="Arctic sea-ice area")
for (i, lab) in enumerate(labels)
    time_years = [Dates.value(d - ICE[lab].snapshot_dates[1]) / (365.25 * 86400 * 1000) for d in ICE[lab].snapshot_dates]
    lines!(ax, time_years, ICE[lab].arctic_area .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
ax = Axis(fig[1, 2]; xlabel="Time (years)", ylabel="SIA (Million km²)", title="Antarctic sea-ice area")
for (i, lab) in enumerate(labels)
    time_years = [Dates.value(d - ICE[lab].snapshot_dates[1]) / (365.25 * 86400 * 1000) for d in ICE[lab].snapshot_dates]
    lines!(ax, time_years, ICE[lab].antarctic_area .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
savefig(fig, "fig12_sia_timeseries.png")

# Figure 13: Arctic volume time series
@info "Figure 13: Arctic volume time series"
fig = Figure(size = (600, 500), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Time (years)", ylabel="Ice volume (10³ km³)", title="Arctic sea-ice volume")
for (i, lab) in enumerate(labels)
    time_years = [Dates.value(d - ICE[lab].snapshot_dates[1]) / (365.25 * 86400 * 1000) for d in ICE[lab].snapshot_dates]
    lines!(ax, time_years, ICE[lab].arctic_volume .* m3_to_1e3km3; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
savefig(fig, "fig13_arctic_volume_timeseries.png")

# ══════════════════════════════════════════════════════════════
# Load time series and 3-D fields
# ══════════════════════════════════════════════════════════════

function load_timeseries_case(run_dir, prefix, grid; start_time = 0, stop_time = Inf)
    averages_file = find_first_file(run_dir, prefix, "averages")
    temperature_mean_fts = FieldTimeSeries(averages_file, "tosga"; backend = InMemory(10))
    salinity_mean_fts    = FieldTimeSeries(averages_file, "soga";  backend = InMemory(10))
    temperature_mean = [Array(interior(temperature_mean_fts[n]))[1] for n in 1:length(temperature_mean_fts.times)]
    salinity_mean    = [Array(interior(salinity_mean_fts[n]))[1]  for n in 1:length(salinity_mean_fts.times)]
    time_in_years    = temperature_mean_fts.times ./ (365.25 * 24 * 3600)

    temperature_profile_fts = FieldTimeSeries(averages_file, "to_h"; backend = InMemory(10))
    salinity_profile_fts    = FieldTimeSeries(averages_file, "so_h"; backend = InMemory(10))
    temperature_profile = vec(compute_time_mean(temperature_profile_fts; start_time, stop_time))
    salinity_profile    = vec(compute_time_mean(salinity_profile_fts; start_time, stop_time))
    depth = collect(znodes(grid, Center()))

    Nt_profile = length(temperature_profile_fts.times)
    Nz_profile = size(temperature_profile_fts[1], 3)
    temperature_drift = zeros(Nt_profile, Nz_profile)
    salinity_drift    = zeros(Nt_profile, Nz_profile)
    for n in 1:Nt_profile
        temperature_drift[n, :] .= vec(interior(temperature_profile_fts[n]))
        salinity_drift[n, :]    .= vec(interior(salinity_profile_fts[n]))
    end
    temperature_drift .-= reshape(temperature_drift[1, :], 1, :)
    salinity_drift    .-= reshape(salinity_drift[1, :],    1, :)
    drift_time_in_years = temperature_profile_fts.times ./ (365.25 * 24 * 3600)

    fields_file = find_first_file(run_dir, prefix, "fields")
    tke_fts     = FieldTimeSeries(fields_file, "tke"; backend = InMemory(10))
    u_fts       = FieldTimeSeries(fields_file, "uo"; backend = InMemory(10))
    v_fts       = FieldTimeSeries(fields_file, "vo"; backend = InMemory(10))

    ocean_mask  = build_ocean_mask_3d(grid)
    ocean_cells = sum(ocean_mask)

    Nt_tke = length(tke_fts.times)
    tke_mean = zeros(Nt_tke)
    for n in 1:Nt_tke
        tke_mean[n] = sum(interior(tke_fts[n]) .* ocean_mask) / ocean_cells
    end

    # Build the KE operation once; reuse the scratch output Field every step.
    # Earlier version allocated a fresh `Field(@at(...))` per snapshot, which
    # compiled a new lazy op and new output buffer each iteration.
    u_scratch = Field{Face, Center, Center}(grid)
    v_scratch = Field{Center, Face, Center}(grid)
    ke_op     = @at((Center, Center, Center),
                    u_scratch * u_scratch + v_scratch * v_scratch)
    ke_field  = Field(ke_op)

    Nt_ke = length(u_fts.times)
    ke_mean = zeros(Nt_ke)
    for n in 1:Nt_ke
        set!(u_scratch, u_fts[n])
        set!(v_scratch, v_fts[n])
        compute!(ke_field)
        ke_mean[n] = sum(interior(ke_field) .* ocean_mask) / (2 * ocean_cells)
    end
    tke_time_in_years = tke_fts.times ./ (365.25 * 24 * 3600)

    return (; temperature_mean, salinity_mean, time_in_years,
              temperature_profile, salinity_profile, depth,
              temperature_drift, salinity_drift, drift_time_in_years,
              tke_mean, ke_mean, tke_time_in_years, ocean_mask, fields_file)
end

TS = Dict{String, Any}()
let ts_futures = [(c.label,
                   Threads.@spawn load_timeseries_case(run_dir_for(c.prefix), c.prefix, D[c.label].grid;
                                                       start_time = c.start_time,
                                                       stop_time  = c.stop_time))
                  for c in cases]
    for (lab, fut) in ts_futures
        @info "Loading time series: $lab..."
        TS[lab] = fetch(fut)
    end
end

# ══════════════════════════════════════════════════════════════
# Figures 14-16: Time series and profiles
# ══════════════════════════════════════════════════════════════

# Figure 14: TKE
@info "Figure 14: TKE and KE"
fig = Figure(size = (900, 600), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Time (years)", ylabel="TKE (m²/s²)", title="Global-mean turbulent kinetic energy")
for (i, lab) in enumerate(labels)
    lines!(ax, TS[lab].tke_time_in_years, TS[lab].tke_mean; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rb)
ax = Axis(fig[2, 1]; xlabel="Time (years)", ylabel="TKE (m²/s²)", title="Global-mean kinetic energy")
for (i, lab) in enumerate(labels)
    lines!(ax, TS[lab].tke_time_in_years, TS[lab].ke_mean; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rb)
savefig(fig, "fig14_tke.png")

# Figure 15: T and S drift
@info "Figure 15: T and S drift"
fig = Figure(size = (1200, 450), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Time (years)", ylabel="ΔT (deg C)", title="Global-mean temperature drift")
for (i, lab) in enumerate(labels)
    d = TS[lab]
    lines!(ax, d.time_in_years, d.temperature_mean .- d.temperature_mean[1]; color=case_colors[i], label=lab)
end
axislegend(ax; position=:lb)
ax = Axis(fig[1, 2]; xlabel="Time (years)", ylabel="ΔS (PSU)", title="Global-mean salinity drift")
for (i, lab) in enumerate(labels)
    d = TS[lab]
    lines!(ax, d.time_in_years, d.salinity_mean .- d.salinity_mean[1]; color=case_colors[i], label=lab)
end
axislegend(ax; position=:lb)
savefig(fig, "fig15_drift.png")

# Figure 16: Profiles
@info "Figure 16: Profiles"
fig = Figure(size = (1000, 600), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Temperature (deg C)", ylabel="Depth (m)", title="Horizontal-mean temperature")
for (i, lab) in enumerate(labels)
    lines!(ax, TS[lab].temperature_profile, TS[lab].depth; color=case_colors[i], label=lab)
end
ylims!(ax, (-5500, 0)); axislegend(ax; position=:rb)
ax = Axis(fig[1, 2]; xlabel="Salinity (PSU)", ylabel="Depth (m)", title="Horizontal-mean salinity")
for (i, lab) in enumerate(labels)
    lines!(ax, TS[lab].salinity_profile, TS[lab].depth; color=case_colors[i], label=lab)
end
ylims!(ax, (-5500, 0)); axislegend(ax; position=:rb)
savefig(fig, "fig16_profiles.png")

# ══════════════════════════════════════════════════════════════
# Zonal-mean sections
# ══════════════════════════════════════════════════════════════

Nlon, Nlat = 360, 180
latlon_grid = LatitudeLongitudeGrid(CPU();
    size = (Nlon, Nlat, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1))
dst_f = Field{Center, Center, Nothing}(latlon_grid)

function compute_zonal_mean(data_3d, ocean_mask_3d, regridder, Nlon, Nlat)
    Nz = size(data_3d, 3)
    zonal    = fill(NaN, Nlat, Nz)
    dst_data = zeros(Nlon * Nlat)
    dst_mask = zeros(Nlon * Nlat)
    areas    = regridder.dst_areas
    for k in 1:Nz
        ConservativeRegridding.regrid!(dst_data, regridder,
            vec(data_3d[:, :, k] .* ocean_mask_3d[:, :, k]))
        ConservativeRegridding.regrid!(dst_mask, regridder,
            vec(ocean_mask_3d[:, :, k]))
        data_sum = reshape(dst_data .* areas, Nlon, Nlat)
        mask_sum = reshape(dst_mask .* areas, Nlon, Nlat)
        for j in 1:Nlat
            m = sum(@view mask_sum[:, j])
            m > 0 && (zonal[j, k] = sum(@view data_sum[:, j]) / m)
        end
    end
    return zonal
end

ZM = Dict{String, Any}()
for c in cases
    lab  = c.label
    grid = D[lab].grid
    ocean_mask = TS[lab].ocean_mask

    # Build per-case regridder
    @info "Building regridder for $lab (may take a few minutes)..."
    src_f = Field{Center, Center, Nothing}(grid)
    regridder = ConservativeRegridding.Regridder(dst_f, src_f; progress = true)

    @info "Loading 3-D fields for $lab..."
    fields_file = TS[lab].fields_file
    to_fts = FieldTimeSeries(fields_file, "to";  backend = InMemory(10))
    so_fts = FieldTimeSeries(fields_file, "so";  backend = InMemory(10))
    bo_fts = FieldTimeSeries(fields_file, "bo";  backend = InMemory(10))
    eo_fts = FieldTimeSeries(fields_file, "tke"; backend = InMemory(10))

    start_time = c.start_time
    stop_time  = c.stop_time
    temperature_mean     = compute_time_mean(to_fts; start_time, stop_time)
    salinity_mean        = compute_time_mean(so_fts; start_time, stop_time)
    buoyancy_mean        = compute_time_mean(bo_fts; start_time, stop_time)
    kinetic_energy_mean  = compute_time_mean(eo_fts; start_time, stop_time)
    buoyancy_initial = Array(interior(bo_fts[1]))

    @info "Computing zonal means for $lab..."
    temperature_zonal     = compute_zonal_mean(temperature_mean,     ocean_mask, regridder, Nlon, Nlat)
    salinity_zonal        = compute_zonal_mean(salinity_mean,        ocean_mask, regridder, Nlon, Nlat)
    buoyancy_zonal        = compute_zonal_mean(buoyancy_mean,        ocean_mask, regridder, Nlon, Nlat)
    kinetic_energy_zonal  = compute_zonal_mean(kinetic_energy_mean,  ocean_mask, regridder, Nlon, Nlat)
    temperature_woa_zonal = compute_zonal_mean(D[lab].T_woa_on_grid, ocean_mask, regridder, Nlon, Nlat)
    salinity_woa_zonal    = compute_zonal_mean(D[lab].S_woa_on_grid, ocean_mask, regridder, Nlon, Nlat)
    buoyancy_init_zonal   = compute_zonal_mean(buoyancy_initial,     ocean_mask, regridder, Nlon, Nlat)

    depth = collect(znodes(grid, Center()))

    @info "Computing zonal-mean MLD for $lab..."
    surface_ocean_mask = reshape(ocean_mask[:, :, end], size(ocean_mask, 1), size(ocean_mask, 2), 1)
    nan_to_zero(a) = ifelse.(isnan.(a), zero(eltype(a)), a)
    mld_min_3d = reshape(nan_to_zero(D[lab].MLD_min), size(D[lab].MLD_min)..., 1)
    mld_max_3d = reshape(nan_to_zero(D[lab].MLD_max), size(D[lab].MLD_max)..., 1)
    mld_min_zonal = vec(compute_zonal_mean(mld_min_3d, surface_ocean_mask, regridder, Nlon, Nlat))
    mld_max_zonal = vec(compute_zonal_mean(mld_max_3d, surface_ocean_mask, regridder, Nlon, Nlat))

    mld_min_dbm_zonal = nothing
    mld_max_dbm_zonal = nothing
    if !isnothing(D[lab].MLD_min_dbm)
        mld_min_dbm_3d = reshape(nan_to_zero(D[lab].MLD_min_dbm), size(D[lab].MLD_min_dbm)..., 1)
        mld_max_dbm_3d = reshape(nan_to_zero(D[lab].MLD_max_dbm), size(D[lab].MLD_max_dbm)..., 1)
        mld_min_dbm_zonal = vec(compute_zonal_mean(mld_min_dbm_3d, surface_ocean_mask, regridder, Nlon, Nlat))
        mld_max_dbm_zonal = vec(compute_zonal_mean(mld_max_dbm_3d, surface_ocean_mask, regridder, Nlon, Nlat))
    end

    ZM[lab] = (; temperature_zonal, salinity_zonal, buoyancy_zonal, kinetic_energy_zonal,
                temperature_woa_zonal, salinity_woa_zonal, buoyancy_init_zonal,
                δtemperature_zonal = temperature_zonal .- temperature_woa_zonal,
                δsalinity_zonal    = salinity_zonal    .- salinity_woa_zonal,
                δbuoyancy_zonal    = buoyancy_zonal    .- buoyancy_init_zonal,
                mld_min_zonal, mld_max_zonal,
                mld_min_dbm_zonal, mld_max_dbm_zonal,
                depth)
end

latitude = collect(φnodes(latlon_grid, Center()))

# ══════════════════════════════════════════════════════════════
# Figures 17-18: Zonal means
# ══════════════════════════════════════════════════════════════

temperature_levels = -2:2:30
salinity_levels    = 33:0.25:37
buoyancy_levels    = range(-0.04, 0.02, length=13)

# Figure 17: Zonal-mean T, S, b
@info "Figure 17: Zonal means"
fig = Figure(size = (600 * length(labels), 1200), fontsize = 14)
for (i, lab) in enumerate(labels)
    zm = ZM[lab]
    ax = Axis(fig[1, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal T")
    hm = heatmap!(ax, latitude, zm.depth, zm.temperature_zonal; colormap=:thermal, colorrange=(-2,30), nan_color=:lightgray)
    contour!(ax, latitude, zm.depth, zm.temperature_woa_zonal; levels=temperature_levels, color=:grey, linestyle=:dash, linewidth=0.8)
    contour!(ax, latitude, zm.depth, zm.temperature_zonal; levels=temperature_levels, color=:black, linewidth=0.8)
    Colorbar(fig[1, 2i], hm; label="deg C"); ylims!(ax, (-5500, 0))

    ax = Axis(fig[2, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal S")
    hm = heatmap!(ax, latitude, zm.depth, zm.salinity_zonal; colormap=:haline, colorrange=(33,37), nan_color=:lightgray)
    contour!(ax, latitude, zm.depth, zm.salinity_woa_zonal; levels=salinity_levels, color=:grey, linestyle=:dash, linewidth=0.8)
    contour!(ax, latitude, zm.depth, zm.salinity_zonal; levels=salinity_levels, color=:black, linewidth=0.8)
    Colorbar(fig[2, 2i], hm; label="PSU"); ylims!(ax, (-5500, 0))

    ax = Axis(fig[3, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal b")
    hm = heatmap!(ax, latitude, zm.depth, zm.buoyancy_zonal; colormap=:balance, nan_color=:lightgray)
    contour!(ax, latitude, zm.depth, zm.buoyancy_init_zonal; levels=buoyancy_levels, color=:grey, linestyle=:dash, linewidth=0.8)
    contour!(ax, latitude, zm.depth, zm.buoyancy_zonal; levels=buoyancy_levels, color=:black, linewidth=0.8)
    Colorbar(fig[3, 2i], hm; label="m/s²"); ylims!(ax, (-5500, 0))

    ax = Axis(fig[4, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal e")
    hm = heatmap!(ax, latitude, zm.depth, zm.kinetic_energy_zonal; colormap=:solar, nan_color=:lightgray)
    Colorbar(fig[4, 2i], hm; label="m/s²"); ylims!(ax, (-5500, 0))
end
savefig(fig, "fig17_zonal_mean.png")

# Figure 18: Zonal-mean drift
@info "Figure 18: Zonal-mean drift"
fig = Figure(size = (600 * length(labels), 900), fontsize = 14)
for (i, lab) in enumerate(labels)
    zm = ZM[lab]
    ax = Axis(fig[1, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal T - WOA")
    hm = heatmap!(ax, latitude, zm.depth, zm.δtemperature_zonal; colormap=:balance, colorrange=(-5,5), nan_color=:lightgray)
    Colorbar(fig[1, 2i], hm; label="deg C"); ylims!(ax, (-5500, 0))

    ax = Axis(fig[2, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal S - WOA")
    hm = heatmap!(ax, latitude, zm.depth, zm.δsalinity_zonal; colormap=:balance, colorrange=(-1,1), nan_color=:lightgray)
    Colorbar(fig[2, 2i], hm; label="PSU"); ylims!(ax, (-5500, 0))

    ax = Axis(fig[3, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal b - b(t=0)")
    hm = heatmap!(ax, latitude, zm.depth, zm.δbuoyancy_zonal; colormap=:balance, nan_color=:lightgray)
    Colorbar(fig[3, 2i], hm; label="m/s²"); ylims!(ax, (-5500, 0))
end
savefig(fig, "fig18_zonal_drift.png")

# Figure 19: Zonal-mean MLD min/max
@info "Figure 19: Zonal-mean MLD min/max"
fig = Figure(size = (1300, 550), fontsize = 14)
ax_min = Axis(fig[1, 1]; xlabel = "Latitude", ylabel = "MLD (m)",
              title = "Zonal-mean MLD (summer minimum)", yreversed = true)
ax_max = Axis(fig[1, 2]; xlabel = "Latitude", ylabel = "MLD (m)",
              title = "Zonal-mean MLD (winter maximum)", yreversed = true)
for (i, lab) in enumerate(labels)
    zm = ZM[lab]
    lines!(ax_min, latitude, zm.mld_min_zonal; color = case_colors[i], label = lab, linewidth = 2)
    lines!(ax_max, latitude, zm.mld_max_zonal; color = case_colors[i], label = lab, linewidth = 2)
end
ref_idx = findfirst(lab -> !isnothing(ZM[lab].mld_min_dbm_zonal), labels)
if !isnothing(ref_idx)
    ref_zm = ZM[labels[ref_idx]]
    lines!(ax_min, latitude, ref_zm.mld_min_dbm_zonal;
           color = :black, linewidth = 2, linestyle = :dash, label = "dBM")
    lines!(ax_max, latitude, ref_zm.mld_max_dbm_zonal;
           color = :black, linewidth = 2, linestyle = :dash, label = "dBM")
end
axislegend(ax_min; position = :rt)
axislegend(ax_max; position = :rt)
savefig(fig, "fig19_mld_zonal_mean.png")

# Figure 20: Horizontal-mean T and S drift, time × depth, with z split into
# 0-1000 m (top half) and 1000-5500 m (bottom half) panels — continuous look,
# shared time axis. The split halves the figure vertically so the upper
# 1000 m gets the same display height as the rest of the column.
@info "Figure 20: T and S drift (time × depth, split z)"
ncases = length(labels)
fig = Figure(size = (700 * ncases, 1000), fontsize = 14)

for (i, lab) in enumerate(labels)
    ts  = TS[lab]
    z   = ts.depth
    t   = ts.drift_time_in_years
    δT  = ts.temperature_drift
    δS  = ts.salinity_drift
    col_T = 4i - 3
    col_S = 4i - 1

    ax_T_top = Axis(fig[1, col_T]; ylabel = "Depth (m)", title = "$lab: ΔT (deg C)",
                    xticklabelsvisible = false, xticksvisible = false,
                    bottomspinevisible = false)
    ax_T_bot = Axis(fig[2, col_T]; xlabel = "Time (years)", ylabel = "Depth (m)",
                    topspinevisible = false)
    linkxaxes!(ax_T_top, ax_T_bot)

    hm_T = heatmap!(ax_T_top, t, z, δT; colormap = :balance, colorrange = (-2, 2), nan_color = :lightgray)
    heatmap!(ax_T_bot, t, z, δT;        colormap = :balance, colorrange = (-2, 2), nan_color = :lightgray)
    ylims!(ax_T_top, (-1000, 0))
    ylims!(ax_T_bot, (-5500, -1000))
    Colorbar(fig[1:2, col_T + 1], hm_T; label = "deg C")

    ax_S_top = Axis(fig[1, col_S]; ylabel = "Depth (m)", title = "$lab: ΔS (PSU)",
                    xticklabelsvisible = false, xticksvisible = false,
                    bottomspinevisible = false)
    ax_S_bot = Axis(fig[2, col_S]; xlabel = "Time (years)", ylabel = "Depth (m)",
                    topspinevisible = false)
    linkxaxes!(ax_S_top, ax_S_bot)

    hm_S = heatmap!(ax_S_top, t, z, δS; colormap = :balance, colorrange = (-0.5, 0.5), nan_color = :lightgray)
    heatmap!(ax_S_bot, t, z, δS;        colormap = :balance, colorrange = (-0.5, 0.5), nan_color = :lightgray)
    ylims!(ax_S_top, (-1000, 0))
    ylims!(ax_S_bot, (-5500, -1000))
    Colorbar(fig[1:2, col_S + 1], hm_S; label = "PSU")
end

rowsize!(fig.layout, 1, Relative(0.5))
rowsize!(fig.layout, 2, Relative(0.5))
rowgap!(fig.layout, 1, 0)
savefig(fig, "fig20_TS_drift_heatmap.png")

# Figure 21: Strait transports (offline, dispatched on per-case grid configuration).
# Each case maps to a `:halfdegree` or `:orca` config either via an explicit
# `config` field on the case namedtuple, or by substring match on the prefix.
@info "Figure 21: Strait transports"

function strait_config_for(c)
    if haskey(c, :config)
        return c.config
    end
    p = lowercase(c.prefix)
    occursin("orca", p)       && return :orca
    occursin("halfdegree", p) && return :halfdegree
    occursin("tenthdegree", p) && return :tenthdegree
    return nothing
end

strait_data = Dict{String, Any}()
for c in cases
    cfg = strait_config_for(c)
    if isnothing(cfg)
        @warn "Cannot infer config for case '$(c.label)' — skipping strait transports."
        continue
    end
    @info "  $(c.label): computing strait transports ($cfg)..."
    strait_data[c.label] = strait_transports(cfg, TS[c.label].fields_file;
                                             start_time = c.start_time,
                                             stop_time  = c.stop_time)
end

if !isempty(strait_data)
    fig = Figure(size = (1500, 500), fontsize = 14)
    ax_b = Axis(fig[1, 1]; xlabel = "Time (years)", ylabel = "Transport (Sv)", title = "Bering Strait")
    ax_d = Axis(fig[1, 2]; xlabel = "Time (years)", ylabel = "Transport (Sv)", title = "Drake Passage")
    ax_i = Axis(fig[1, 3]; xlabel = "Time (years)", ylabel = "Transport (Sv)", title = "Indonesian Throughflow")
    for (i, lab) in enumerate(labels)
        haskey(strait_data, lab) || continue
        st = strait_data[lab]
        t  = st.time ./ (365.25 * 24 * 3600)
        lines!(ax_b, t, st.bering; color = case_colors[i], label = lab, linewidth = 2)
        lines!(ax_d, t, st.drake;  color = case_colors[i], label = lab, linewidth = 2)
        lines!(ax_i, t, st.itf;    color = case_colors[i], label = lab, linewidth = 2)
    end
    axislegend(ax_b; position = :rt)
    savefig(fig, "fig21_strait_transports.png")
end

@info "All figures saved to $output_dir"
