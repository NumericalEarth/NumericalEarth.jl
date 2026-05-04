# visualize/cache.jl
#
# Per-case lazy field cache with optional disk persistence.
#
# Each derived diagnostic is keyed by a `Symbol` and computed on first
# access via the `LOADERS` registry below; subsequent accesses hit the
# in-memory cache. Loaders may pull other fields recursively
# (e.g. `:sst_bias` ← `:sst` + `:woa_temperature`), so each figure file
# just calls `get_field(case_cache, :sst_bias)` and the loader DAG
# handles the rest. Across figures, every field is loaded at most once
# per case.
#
# Loaders flagged with `disk_cached(...)` additionally persist their
# result under `<output_dir>/diag_cache/<prefix>/<sym>.jld2`. The cache
# is keyed by the snapshot count of every source `FieldTimeSeries` it
# depends on, plus the case averaging window — so appending new
# snapshots invalidates the entry automatically while no-op file
# touches do not.
#
# Requires `common.jl` to have been included first.

#####
##### CaseCache
#####

mutable struct CaseCache
    label      :: String
    prefix     :: String
    run_dir    :: String
    start_time :: Float64
    stop_time  :: Float64
    case       :: NamedTuple
    fields     :: Dict{Symbol, Any}
end

function CaseCache(case::NamedTuple)
    return CaseCache(case.label, case.prefix, run_dir_for(case.prefix),
                     Float64(case.start_time), Float64(case.stop_time),
                     case, Dict{Symbol, Any}())
end

# Sentinel used to memoize loaders that legitimately return `nothing`
# (e.g. dBM climatology when the file is missing). Without this a
# successful "no result" would re-trigger the loader on every access.
const _CACHE_MISS = :__cache_miss__

"""
    get_field(case_cache, sym)

Return the value cached under `sym` for `case_cache`, computing it via
`LOADERS[sym]` on first access.
"""
function get_field(case_cache::CaseCache, sym::Symbol)
    val = get(case_cache.fields, sym, _CACHE_MISS)
    val === _CACHE_MISS || return val
    haskey(LOADERS, sym) || error("No loader registered for :$sym")
    val = LOADERS[sym](case_cache)
    case_cache.fields[sym] = val
    return val
end

get_fields(case_cache::CaseCache, syms::Symbol...) = ntuple(i -> get_field(case_cache, syms[i]), length(syms))

#####
##### Disk-side caching
#####

"""
    DiskCacheKey(snapshot_counts, start_time, stop_time)

Validation key for a disk-cached derived field. A cached value is
reusable only if all three components match the current state of the
source `FieldTimeSeries` and the case averaging window.

`snapshot_counts` is an `NTuple{N, Int}`, one entry per source FTS the
loader depends on. Use `N = 0` for fields that depend only on the grid
or external climatologies (always-valid cache once written).
"""
struct DiskCacheKey{N}
    snapshot_counts :: NTuple{N, Int}
    start_time      :: Float64
    stop_time       :: Float64
end

Base.:(==)(a::DiskCacheKey{N}, b::DiskCacheKey{N}) where N =
    a.snapshot_counts == b.snapshot_counts &&
    a.start_time      == b.start_time      &&
    a.stop_time       == b.stop_time

Base.:(==)(::DiskCacheKey, ::DiskCacheKey) = false

"""
    diag_cache_dir(case_cache)

Per-case directory holding the JLD2 disk cache for postprocessed
diagnostics. Lives under `output_dir` so the run directory itself
stays read-only.
"""
diag_cache_dir(case_cache::CaseCache) = joinpath(output_dir, "diag_cache", case_cache.prefix)

diag_cache_path(case_cache::CaseCache, sym::Symbol) = joinpath(diag_cache_dir(case_cache), string(sym) * ".jld2")

# Filled alongside each raw-FTS `LOADERS[...]` registration below.
const _FTS_DISK_PATH_SYM = Dict{Symbol, Symbol}()

"""
    fts_snapshot_count_for_disk_key(case_cache, fts_sym)

Number of time indices for the JLD2 output backing `fts_sym`, from JLD2
metadata (`timeseries/t` key count per part, summed across split parts).
Does not open a `FieldTimeSeries`. Per-part counts are memoized for the session
(see `total_jld2_timeseries_snapshot_count`).
"""
function fts_snapshot_count_for_disk_key(case_cache::CaseCache, fts_sym::Symbol)
    file_sym = get(_FTS_DISK_PATH_SYM, fts_sym, nothing)
    file_sym === nothing &&
        error("No JLD2 stem registered for FTS :$fts_sym — extend `_FTS_DISK_PATH_SYM` with the loader loops.")
    path = get_field(case_cache, file_sym)
    return total_jld2_timeseries_snapshot_count(path)
end

"""
    current_disk_cache_key(case_cache, source_fts_syms)

Build the `DiskCacheKey` reflecting the current state of every source
FTS named in `source_fts_syms`. Uses JLD2 metadata only (no `FieldTimeSeries`
construction). Pass `()` for fields whose validity does not depend on any
model output (pure climatologies).
"""
function current_disk_cache_key(case_cache::CaseCache, source_fts_syms::Tuple{Vararg{Symbol}})
    N = length(source_fts_syms)
    snapshot_counts = ntuple(i -> fts_snapshot_count_for_disk_key(case_cache, source_fts_syms[i]), N)
    return DiskCacheKey(snapshot_counts, case_cache.start_time, case_cache.stop_time)
end

current_disk_cache_key(case_cache::CaseCache, source_fts_sym::Symbol) = current_disk_cache_key(case_cache, (source_fts_sym,))

"""
    read_disk_cache(path)

Return `(value, key)` if `path` exists and parses cleanly; return
`nothing` otherwise. A failed read logs a warning and falls through
so the loader recomputes.
"""
function read_disk_cache(path)
    isfile(path) || return nothing
    try
        return JLD2.load(path, "value", "key")
    catch err
        @warn "  Disk cache read failed; will recompute" path=path error=sprint(showerror, err)
        return nothing
    end
end

"""
    write_disk_cache(path, value, key)

Persist `value` and its validation `key` to `path` (creating
directories as needed). Returns `value` so the caller can chain.
"""
function write_disk_cache(path, value, key)
    mkpath(dirname(path))
    try
        JLD2.jldsave(path; value, key)
    catch err
        @warn "  Disk cache write failed; continuing without persistence" path=path error=sprint(showerror, err)
    end
    return value
end

"""
    explain_key_mismatch(stored, current)

Return a short string describing what changed between `stored` and
`current` validation keys. Used to log *why* a disk cache entry is
being invalidated.
"""
function explain_key_mismatch(stored::DiskCacheKey{N}, current::DiskCacheKey{N}) where N
    diffs = String[]
    stored.snapshot_counts == current.snapshot_counts ||
        push!(diffs, "snapshots $(stored.snapshot_counts) → $(current.snapshot_counts)")
    stored.start_time == current.start_time ||
        push!(diffs, "start_time $(stored.start_time) → $(current.start_time)")
    stored.stop_time == current.stop_time ||
        push!(diffs, "stop_time $(stored.stop_time) → $(current.stop_time)")
    return join(diffs, ", ")
end

# Different N → keys aren't even comparable; report it explicitly.
explain_key_mismatch(::DiskCacheKey, ::DiskCacheKey) = "source-FTS arity changed"

"""
    disk_cached(loader, sym; source_fts_syms = ())

Wrap `loader(case_cache)` so its result is persisted under the
per-case JLD2 cache file for `sym`. Subsequent loads in any session
reuse the cached value when the validation key matches; otherwise
the loader is rerun and the cache is overwritten.

`source_fts_syms` is either a single FTS symbol or a tuple of them.
Use `()` for fields that depend only on the grid or climatologies.
"""
function disk_cached(loader::Function, sym::Symbol; source_fts_syms::Union{Symbol, Tuple{Vararg{Symbol}}} = ())
    sources = source_fts_syms isa Symbol ? (source_fts_syms,) : source_fts_syms
    function cached_loader(c::CaseCache)
        path        = diag_cache_path(c, sym)
        cached      = read_disk_cache(path)
        current_key = current_disk_cache_key(c, sources)
        if cached !== nothing
            value, stored_key = cached
            if stored_key isa DiskCacheKey && stored_key == current_key
                @info "  $(c.label): :$sym ← disk cache"
                return value
            elseif stored_key isa DiskCacheKey
                @info "  $(c.label): :$sym ← invalidated ($(explain_key_mismatch(stored_key, current_key)))"
            else
                @info "  $(c.label): :$sym ← invalidated (corrupt or unrecognized key)"
            end
        end
        @info "  $(c.label): :$sym ← computing"
        value = loader(c)
        return write_disk_cache(path, value, current_key)
    end
    return cached_loader
end

#####
##### Loader registry
#####
#
# Each entry is a function `(case_cache::CaseCache) -> value`. The
# returned value is cached verbatim, so loaders that return composite
# results (e.g. monthly bins tuple) can be unpacked by downstream
# loaders.

const LOADERS = Dict{Symbol, Function}()

#####
##### File paths
#####

LOADERS[:surface_file]  = c -> find_first_file(c.run_dir, c.prefix, "surface")
LOADERS[:fields_file]   = c -> find_first_file(c.run_dir, c.prefix, "fields")
LOADERS[:averages_file] = c -> find_first_file(c.run_dir, c.prefix, "averages")

#####
##### Raw FieldTimeSeries
#####

# Surface file
for (sym, name) in ((:tos_fts, "tos"), (:sos_fts, "sos"), (:zos_fts, "zos"),
                    (:mld_fts, "mlotst"), (:hfds_fts, "hfds"), (:wfo_fts, "wfo"),
                    (:sic_fts, "siconc"), (:zossq_fts, "zossq"),
                    (:tauuo_fts, "tauuo"), (:tauvo_fts, "tauvo"),
                    (:sithick_fts, "sithick"))
    _FTS_DISK_PATH_SYM[sym] = :surface_file
    LOADERS[sym] = let n = name
        c -> FieldTimeSeries(get_field(c, :surface_file), n; backend = deepcopy(FTS_BACKEND))
    end
end

# Fields file (3-D)
for (sym, name) in ((:to_fts, "to"), (:so_fts, "so"), (:bo_fts, "bo"),
                    (:uo_fts, "uo"), (:vo_fts, "vo"))
    _FTS_DISK_PATH_SYM[sym] = :fields_file
    LOADERS[sym] = let n = name
        c -> FieldTimeSeries(get_field(c, :fields_file), n; backend = deepcopy(FTS_BACKEND))
    end
end

# Averages file
for (sym, name) in ((:tosga_fts, "tosga"), (:soga_fts, "soga"),
                    (:to_h_fts, "to_h"), (:so_h_fts, "so_h"))
    _FTS_DISK_PATH_SYM[sym] = :averages_file
    LOADERS[sym] = let n = name
        c -> FieldTimeSeries(get_field(c, :averages_file), n; backend = deepcopy(FTS_BACKEND))
    end
end

#####
##### Grid + masks
#####

LOADERS[:grid]          = c -> get_field(c, :tos_fts).grid
LOADERS[:Nx]            = c -> size(get_field(c, :grid), 1)
LOADERS[:Ny]            = c -> size(get_field(c, :grid), 2)
LOADERS[:Nz]            = c -> size(get_field(c, :grid), 3)
LOADERS[:land]          = c -> build_land_mask(get_field(c, :grid))
LOADERS[:ocean_mask_3d] = c -> build_ocean_mask_3d(get_field(c, :grid))
LOADERS[:depth]         = c -> collect(znodes(get_field(c, :grid), Center()))

#####
##### Surface time means (with land masking)
#####

function time_mean_drop(c, fts_sym)
    fts = get_field(c, fts_sym)
    return dropdims(compute_time_mean(fts; start_time = c.start_time, stop_time = c.stop_time); dims = 3)
end

# Apply land mask in place after computing `value`. A `nothing` value
# is passed through unchanged (used by optional climatology fields).
function masked!(c, value)
    isnothing(value) && return value
    mask_land!(value, get_field(c, :land))
    return value
end

LOADERS[:sst] = disk_cached(:sst; source_fts_syms = :tos_fts) do c
    masked!(c, time_mean_drop(c, :tos_fts))
end
LOADERS[:sss] = disk_cached(:sss; source_fts_syms = :sos_fts) do c
    masked!(c, time_mean_drop(c, :sos_fts))
end
LOADERS[:ssh] = disk_cached(:ssh; source_fts_syms = :zos_fts) do c
    masked!(c, time_mean_drop(c, :zos_fts))
end

LOADERS[:heat_flux] = disk_cached(:heat_flux; source_fts_syms = :hfds_fts) do c
    masked!(c, time_mean_drop(c, :hfds_fts) .* (ρ_ocean * cp_ocean))
end
LOADERS[:freshwater_flux] = disk_cached(:freshwater_flux; source_fts_syms = :wfo_fts) do c
    masked!(c, time_mean_drop(c, :wfo_fts))
end

LOADERS[:ssh_squared_mean] = disk_cached(:ssh_squared_mean; source_fts_syms = :zossq_fts) do c
    time_mean_drop(c, :zossq_fts)
end
# `:ssh_variance` is just a subtraction of two cached arrays — keep in memory only.
LOADERS[:ssh_variance] = c -> masked!(c, get_field(c, :ssh_squared_mean) .- get_field(c, :ssh) .^ 2)

# Wind stress is stored as kinematic flux; flip sign (atm→ocean
# downward = positive CMIP convention) and multiply by ρ to get N/m².
# Then bring τx/τy from cell faces down to centers if the grid stores
# them staggered. Cache both components together in a single pair so
# the time-mean of `tauuo`/`tauvo` runs once even if both stresses
# are requested.
centered_stress(c, fts_sym) = -time_mean_drop(c, fts_sym) .* ρ_ocean

LOADERS[:wind_stress_pair] = disk_cached(:wind_stress_pair; source_fts_syms = :tauuo_fts) do c
    τx = centered_stress(c, :tauuo_fts)
    τy = centered_stress(c, :tauvo_fts)
    size(τx, 1) == size(τy, 1) + 1 && (τx = (τx[1:end-1, :] .+ τx[2:end, :]) ./ 2)
    size(τy, 2) == size(τx, 2) + 1 && (τy = (τy[:, 1:end-1] .+ τy[:, 2:end]) ./ 2)
    return (masked!(c, τx), masked!(c, τy))
end

LOADERS[:zonal_wind_stress]      = c -> get_field(c, :wind_stress_pair)[1]
LOADERS[:meridional_wind_stress] = c -> get_field(c, :wind_stress_pair)[2]

#####
##### Sea-ice concentration (mean + monthly bins)
#####

LOADERS[:sic_mean_and_monthly] = disk_cached(:sic_mean_and_monthly; source_fts_syms = :sic_fts) do c
    compute_mean_and_monthly(get_field(c, :sic_fts); start_time = c.start_time, stop_time = c.stop_time)
end

LOADERS[:sic_mean] = c -> masked!(c, begin
    mean_raw, _ = get_field(c, :sic_mean_and_monthly)
    dropdims(mean_raw; dims = 3)
end)

LOADERS[:sic_march] = c -> begin
    _, monthly = get_field(c, :sic_mean_and_monthly)
    isnothing(monthly[3]) && return nothing
    masked!(c, dropdims(monthly[3]; dims = 3))
end

LOADERS[:sic_september] = c -> begin
    _, monthly = get_field(c, :sic_mean_and_monthly)
    isnothing(monthly[9]) && return nothing
    masked!(c, dropdims(monthly[9]; dims = 3))
end

#####
##### Mixed-layer depth (kernel diagnostic, monthly min/max)
#####

LOADERS[:mld_monthly] = disk_cached(:mld_monthly; source_fts_syms = :mld_fts) do c
    compute_monthly_means(get_field(c, :mld_fts); start_time = c.start_time, stop_time = c.stop_time)
end

function stack_monthly(c, sym)
    monthly = get_field(c, sym)
    avail   = findall(!isnothing, monthly)
    return cat([dropdims(monthly[m]; dims = 3) for m in avail]...; dims = 3)
end

LOADERS[:mld_min] = c -> masked!(c, dropdims(minimum(stack_monthly(c, :mld_monthly); dims = 3); dims = 3))
LOADERS[:mld_max] = c -> masked!(c, dropdims(maximum(stack_monthly(c, :mld_monthly); dims = 3); dims = 3))

#####
##### WOA temperature & salinity on case grid
#####

function woa_on_grid(c, var)
    woa = Field(Metadatum(var; dataset = WOAAnnual()), CPU())
    out = CenterField(get_field(c, :grid))
    interpolate!(out, woa)
    return Array(interior(out))
end

LOADERS[:woa_temperature] = c -> woa_on_grid(c, :temperature)
LOADERS[:woa_salinity]    = c -> woa_on_grid(c, :salinity)

LOADERS[:sst_bias] = c -> masked!(c, get_field(c, :sst) .- get_field(c, :woa_temperature)[:, :, end])
LOADERS[:sss_bias] = c -> masked!(c, get_field(c, :sss) .- get_field(c, :woa_salinity)[:, :, end])

#####
##### ECCO SSH
#####

LOADERS[:ecco_ssh] = c -> masked!(c, ecco_ssh_on_grid(get_field(c, :grid)))

LOADERS[:ssh_bias_ecco] = c -> masked!(c, begin
    η     = get_field(c, :ssh)
    η_ref = get_field(c, :ecco_ssh)
    (η .- mean(filter(isfinite, η))) .- (η_ref .- mean(filter(isfinite, η_ref)))
end)

#####
##### dBM mixed-layer climatology
#####

LOADERS[:dbm_mld_monthly] = c -> dbm_mld_climatology_on_grid(get_field(c, :grid))

function dbm_mld_extreme(c, reduction)
    monthly = get_field(c, :dbm_mld_monthly)
    isnothing(monthly) && return nothing
    return masked!(c, dropdims(reduction(monthly; dims = 3); dims = 3))
end

LOADERS[:mld_min_dbm] = c -> dbm_mld_extreme(c, minimum)
LOADERS[:mld_max_dbm] = c -> dbm_mld_extreme(c, maximum)

#####
##### NCEP wind-stress climatology
#####

LOADERS[:ncep_wind_stress_pair] = c -> ncep_wind_stress_on_grid(get_field(c, :grid))

LOADERS[:ncep_zonal_wind_stress] = c -> begin
    τ, _ = get_field(c, :ncep_wind_stress_pair)
    isnothing(τ) ? nothing : masked!(c, τ)
end

LOADERS[:ncep_meridional_wind_stress] = c -> begin
    _, τ = get_field(c, :ncep_wind_stress_pair)
    isnothing(τ) ? nothing : masked!(c, τ)
end

LOADERS[:zonal_wind_stress_bias_ncep] = c -> begin
    τ_ref = get_field(c, :ncep_zonal_wind_stress)
    isnothing(τ_ref) ? nothing : masked!(c, get_field(c, :zonal_wind_stress) .- τ_ref)
end

LOADERS[:meridional_wind_stress_bias_ncep] = c -> begin
    τ_ref = get_field(c, :ncep_meridional_wind_stress)
    isnothing(τ_ref) ? nothing : masked!(c, get_field(c, :meridional_wind_stress) .- τ_ref)
end

#####
##### Sea-ice integrals (heavy: per-snapshot loop over sithick + sic)
#####

LOADERS[:sea_ice_diagnostics] = disk_cached(:sea_ice_diagnostics; source_fts_syms = :sithick_fts) do c
    compute_ice_diagnostics(c.run_dir, c.prefix, get_field(c, :grid);
                            start_time = c.start_time,
                            stop_time  = c.stop_time)
end

#####
##### Time-series scalars + horizontal-mean profiles
#####

scalar_timeseries(c, fts_sym) = let fts = get_field(c, fts_sym)
    [Array(interior(fts[n]))[1] for n in 1:length(fts.times)]
end

LOADERS[:global_mean_temperature_timeseries] =
    disk_cached(:global_mean_temperature_timeseries; source_fts_syms = :tosga_fts) do c
    scalar_timeseries(c, :tosga_fts)
end
LOADERS[:global_mean_salinity_timeseries] =
    disk_cached(:global_mean_salinity_timeseries; source_fts_syms = :soga_fts) do c
    scalar_timeseries(c, :soga_fts)
end

LOADERS[:time_in_years] = c -> get_field(c, :tosga_fts).times ./ (365.25 * 24 * 3600)

LOADERS[:horizontal_mean_temperature_profile] = c ->
    vec(compute_time_mean(get_field(c, :to_h_fts);
                           start_time = c.start_time,
                           stop_time  = c.stop_time))

LOADERS[:horizontal_mean_salinity_profile] = c ->
    vec(compute_time_mean(get_field(c, :so_h_fts);
                           start_time = c.start_time,
                           stop_time  = c.stop_time))

# Per-snapshot horizontal-mean drift relative to the first snapshot.
function profile_drift(c, fts_sym)
    fts = get_field(c, fts_sym)
    Nt  = length(fts.times)
    Nz  = size(fts[1], 3)
    Δ   = zeros(Nt, Nz)
    for n in 1:Nt
        Δ[n, :] .= vec(interior(fts[n]))
    end
    Δ .-= reshape(Δ[1, :], 1, :)
    return Δ
end

LOADERS[:temperature_drift] = disk_cached(:temperature_drift; source_fts_syms = :to_h_fts) do c
    profile_drift(c, :to_h_fts)
end
LOADERS[:salinity_drift] = disk_cached(:salinity_drift; source_fts_syms = :so_h_fts) do c
    profile_drift(c, :so_h_fts)
end

LOADERS[:drift_time_in_years] = c -> get_field(c, :to_h_fts).times ./ (365.25 * 24 * 3600)

#####
##### Global-mean kinetic energy from u, v snapshots
#####

LOADERS[:kinetic_energy_pair] = disk_cached(:kinetic_energy_pair; source_fts_syms = :uo_fts) do c
    grid   = get_field(c, :grid)
    mask   = get_field(c, :ocean_mask_3d)
    Ncells = sum(mask)
    u_fts  = get_field(c, :uo_fts)
    v_fts  = get_field(c, :vo_fts)

    u  = Field{Face,   Center, Center}(grid)
    v  = Field{Center, Face,   Center}(grid)
    e  = Field(@at((Center, Center, Center), u*u + v*v))

    Nt = length(u_fts.times)
    ke = zeros(Nt)
    for n in 1:Nt
        set!(u, u_fts[n])
        set!(v, v_fts[n])
        compute!(e)
        ke[n] = sum(interior(e) .* mask) / (2 * Ncells)
    end
    return (ke, u_fts.times ./ (365.25 * 24 * 3600))
end

LOADERS[:kinetic_energy]               = c -> get_field(c, :kinetic_energy_pair)[1]
LOADERS[:kinetic_energy_time_in_years] = c -> get_field(c, :kinetic_energy_pair)[2]

#####
##### Zonal means
#####
#
# Target lat-lon grid is shared across all cases.

const ZONAL_NLON = 360
const ZONAL_NLAT = 180

const _LATLON_GRID_REF = Ref{Any}(nothing)
const _LATLON_DST_REF  = Ref{Any}(nothing)

function zonal_latlon_grid()
    isnothing(_LATLON_GRID_REF[]) || return _LATLON_GRID_REF[]
    grid = LatitudeLongitudeGrid(CPU();
                                  size      = (ZONAL_NLON, ZONAL_NLAT, 1),
                                  longitude = (0, 360),
                                  latitude  = (-90, 90),
                                  z         = (0, 1))
    _LATLON_GRID_REF[] = grid
    return grid
end

function zonal_latlon_destination_field()
    isnothing(_LATLON_DST_REF[]) || return _LATLON_DST_REF[]
    field = Field{Center, Center, Nothing}(zonal_latlon_grid())
    _LATLON_DST_REF[] = field
    return field
end

zonal_latitude_centers() = collect(φnodes(zonal_latlon_grid(), Center()))

"""
    compute_zonal_mean(data_3d, ocean_mask_3d, regridder, Nlon, Nlat)

Conservatively regrid each level of `data_3d` (weighted by
`ocean_mask_3d`) onto the lat-lon target grid, then average each
latitude row.
"""
function compute_zonal_mean(data_3d, ocean_mask_3d, regridder, Nlon, Nlat)
    Nz    = size(data_3d, 3)
    zonal = fill(NaN, Nlat, Nz)
    fdata = zeros(Nlon * Nlat)
    fmask = zeros(Nlon * Nlat)
    A     = regridder.dst_areas
    for k in 1:Nz
        ConservativeRegridding.regrid!(fdata, regridder, vec(data_3d[:, :, k] .* ocean_mask_3d[:, :, k]))
        ConservativeRegridding.regrid!(fmask, regridder, vec(ocean_mask_3d[:, :, k]))
        Wd = reshape(fdata .* A, Nlon, Nlat)
        Wm = reshape(fmask .* A, Nlon, Nlat)
        for j in 1:Nlat
            m = sum(@view Wm[:, j])
            m > 0 && (zonal[j, k] = sum(@view Wd[:, j]) / m)
        end
    end
    return zonal
end

LOADERS[:zonal_regridder] = c -> begin
    src = Field{Center, Center, Nothing}(get_field(c, :grid))
    @info "  Building zonal regridder for $(c.label) (may take a few minutes)..."
    ConservativeRegridding.Regridder(zonal_latlon_destination_field(), src; progress = true)
end

# 3-D time-means (loaded on demand for zonal-section figures). Not
# disk-cached because each is several hundred megabytes per ORCA case;
# the smaller `:zonal_*` derivatives below are.
LOADERS[:time_mean_temperature_3d] = c -> compute_time_mean(get_field(c, :to_fts);
                                                              start_time = c.start_time,
                                                              stop_time  = c.stop_time)
LOADERS[:time_mean_salinity_3d] = c -> compute_time_mean(get_field(c, :so_fts);
                                                          start_time = c.start_time,
                                                          stop_time  = c.stop_time)
LOADERS[:time_mean_buoyancy_3d] = c -> compute_time_mean(get_field(c, :bo_fts);
                                                          start_time = c.start_time,
                                                          stop_time  = c.stop_time)
LOADERS[:initial_buoyancy_3d]   = c -> Array(interior(get_field(c, :bo_fts)[1]))

# Convenience: zonal mean of any 3-D field already in the cache.
zonal_of(c, sym) = compute_zonal_mean(get_field(c, sym),
                                        get_field(c, :ocean_mask_3d),
                                        get_field(c, :zonal_regridder),
                                        ZONAL_NLON, ZONAL_NLAT)

LOADERS[:zonal_temperature] = disk_cached(:zonal_temperature; source_fts_syms = :to_fts) do c
    zonal_of(c, :time_mean_temperature_3d)
end
LOADERS[:zonal_salinity] = disk_cached(:zonal_salinity; source_fts_syms = :so_fts) do c
    zonal_of(c, :time_mean_salinity_3d)
end
LOADERS[:zonal_buoyancy] = disk_cached(:zonal_buoyancy; source_fts_syms = :bo_fts) do c
    zonal_of(c, :time_mean_buoyancy_3d)
end
LOADERS[:zonal_initial_buoyancy] = disk_cached(:zonal_initial_buoyancy; source_fts_syms = :bo_fts) do c
    zonal_of(c, :initial_buoyancy_3d)
end

# WOA / dBM zonal sections do not depend on model output, so the cache
# key carries no source-FTS counts: once written, the entry is always
# reused for the same case grid.
LOADERS[:zonal_woa_temperature] = disk_cached(:zonal_woa_temperature) do c
    zonal_of(c, :woa_temperature)
end
LOADERS[:zonal_woa_salinity] = disk_cached(:zonal_woa_salinity) do c
    zonal_of(c, :woa_salinity)
end

LOADERS[:zonal_temperature_bias] = c -> get_field(c, :zonal_temperature) .-
                                          get_field(c, :zonal_woa_temperature)
LOADERS[:zonal_salinity_bias]    = c -> get_field(c, :zonal_salinity) .-
                                          get_field(c, :zonal_woa_salinity)
LOADERS[:zonal_buoyancy_drift]   = c -> get_field(c, :zonal_buoyancy) .-
                                          get_field(c, :zonal_initial_buoyancy)

# Zonal MLD: regrid the 2-D surface MLD diagnostic, weighted by the
# surface ocean mask (same convention as before).
function zonal_mld(c, sym)
    raw = get_field(c, sym)
    isnothing(raw) && return nothing
    cleaned = ifelse.(isnan.(raw), zero(eltype(raw)), raw)
    raw_3d  = reshape(cleaned, size(raw)..., 1)
    mask    = get_field(c, :ocean_mask_3d)
    surf    = reshape(mask[:, :, end], size(mask, 1), size(mask, 2), 1)
    return vec(compute_zonal_mean(raw_3d, surf, get_field(c, :zonal_regridder),
                                   ZONAL_NLON, ZONAL_NLAT))
end

LOADERS[:zonal_mld_min]     = c -> zonal_mld(c, :mld_min)
LOADERS[:zonal_mld_max]     = c -> zonal_mld(c, :mld_max)
LOADERS[:zonal_mld_min_dbm] = c -> zonal_mld(c, :mld_min_dbm)
LOADERS[:zonal_mld_max_dbm] = c -> zonal_mld(c, :mld_max_dbm)

#####
##### Strait transports (offline; depends on per-case grid configuration)
#####

function strait_config_for(case)
    if haskey(case, :config)
        return case.config
    end
    p = lowercase(case.prefix)
    occursin("orca", p)        && return :orca
    occursin("halfdegree", p)  && return :halfdegree
    occursin("tenthdegree", p) && return :tenthdegree
    return nothing
end

LOADERS[:strait_transports] = c -> begin
    config = strait_config_for(c.case)
    if isnothing(config)
        @warn "Cannot infer strait config for case '$(c.label)' — skipping."
        return nothing
    end
    @info "  $(c.label): computing strait transports ($config)..."
    return strait_transports(config, get_field(c, :fields_file);
                              start_time = 0,
                              stop_time  = Inf)
end
