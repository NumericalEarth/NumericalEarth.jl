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

# Maps every raw-FTS symbol (e.g. `:to_h_fts`) to the file symbol it
# lives in (`:averages_file`). Filled alongside each FTS loader
# registration below; consulted to validate disk caches via JLD2
# metadata only, never via a full `FieldTimeSeries` build.
const _FTS_DISK_PATH_SYM = Dict{Symbol, Symbol}()

"""
    current_disk_cache_key(case_cache, fts_syms)

Validation key for a disk-cached derived field whose value depends on
the FTS named by each symbol in `fts_syms`. Reads `Nt` per source via
`total_jld2_timeseries_snapshot_count`, so no `FieldTimeSeries` is
constructed. Pass `()` for fields independent of model output.
"""
function current_disk_cache_key(c::CaseCache, fts_syms::Tuple{Vararg{Symbol}})
    Nts = ntuple(length(fts_syms)) do i
        fts_sym  = fts_syms[i]
        file_sym = get(_FTS_DISK_PATH_SYM, fts_sym, nothing)
        isnothing(file_sym) && error("No JLD2 stem registered for FTS :$fts_sym")
        return total_jld2_timeseries_snapshot_count(get_field(c, file_sym))
    end
    return DiskCacheKey(Nts, c.start_time, c.stop_time)
end

current_disk_cache_key(c::CaseCache, fts_sym::Symbol) = current_disk_cache_key(c, (fts_sym,))

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
function disk_cached(loader::Function, sym::Symbol;
                     source_fts_syms::Union{Symbol, Tuple{Vararg{Symbol}}} = ())
    sources = source_fts_syms isa Symbol ? (source_fts_syms,) : source_fts_syms
    return function (c::CaseCache)
        path    = diag_cache_path(c, sym)
        new_key = current_disk_cache_key(c, sources)
        cached  = read_disk_cache(path)
        if !isnothing(cached)
            value, stored_key = cached
            if stored_key == new_key
                @info "  $(c.label): :$sym ← disk cache"
                return value
            end
            reason = stored_key isa DiskCacheKey ?
                     explain_key_mismatch(stored_key, new_key) :
                     "corrupt or unrecognized key"
            @info "  $(c.label): :$sym ← invalidated ($reason)"
        end
        @info "  $(c.label): :$sym ← computing"
        return write_disk_cache(path, loader(c), new_key)
    end
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

# Every raw `:..._fts` symbol → (file symbol, JLD2 variable name).
# One source of truth; consulted to register both the loader and the
# disk-cache file lookup.
const _FTS_VARS = (
    surface_file  = ((:tos_fts,    "tos"),    (:sos_fts,   "sos"),
                     (:zos_fts,    "zos"),    (:mld_fts,   "mlotst"),
                     (:hfds_fts,   "hfds"),   (:wfo_fts,   "wfo"),
                     (:sic_fts,    "siconc"), (:zossq_fts, "zossq"),
                     (:tauuo_fts,  "tauuo"),  (:tauvo_fts, "tauvo"),
                     (:sithick_fts, "sithick")),
    fields_file   = ((:to_fts, "to"), (:so_fts, "so"), (:bo_fts, "bo"),
                     (:uo_fts, "uo"), (:vo_fts, "vo")),
    averages_file = ((:tosga_fts, "tosga"), (:soga_fts, "soga"),
                     (:to_h_fts,  "to_h"),  (:so_h_fts,  "so_h")),
)

for (file_sym, mappings) in pairs(_FTS_VARS), (sym, var) in mappings
    _FTS_DISK_PATH_SYM[sym] = file_sym
    LOADERS[sym] = let v = var, f = file_sym
        c -> FieldTimeSeries(get_field(c, f), v; backend = deepcopy(FTS_BACKEND))
    end
end

#####
##### Grid + masks
#####

LOADERS[:grid]          = c -> total_jld2_serialized_grid(get_field(c, :surface_file), "tos")
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

# Apply land mask in place; `nothing` is passed through (for optional
# climatology fields). The `maybe_*` helpers compose this with a few
# common wrap-or-skip patterns to remove `isnothing(...) ? nothing : …`
# boilerplate from the loader registry below.
masked!(c, ::Nothing) = nothing
masked!(c, value)     = (mask_land!(value, get_field(c, :land)); value)

maybe_diff!(c, a_sym, b_sym) = let b = get_field(c, b_sym)
    isnothing(b) ? nothing : masked!(c, get_field(c, a_sym) .- b)
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

function sic_month(c, m)
    _, monthly = get_field(c, :sic_mean_and_monthly)
    isnothing(monthly[m]) ? nothing : masked!(c, dropdims(monthly[m]; dims = 3))
end

LOADERS[:sic_mean]      = c -> masked!(c, dropdims(get_field(c, :sic_mean_and_monthly)[1]; dims = 3))
LOADERS[:sic_march]     = c -> sic_month(c, 3)
LOADERS[:sic_september] = c -> sic_month(c, 9)

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

dbm_extreme(c, reduce) = masked!(c, let m = get_field(c, :dbm_mld_monthly)
    isnothing(m) ? nothing : dropdims(reduce(m; dims = 3); dims = 3)
end)

LOADERS[:mld_min_dbm] = c -> dbm_extreme(c, minimum)
LOADERS[:mld_max_dbm] = c -> dbm_extreme(c, maximum)

#####
##### NCEP wind-stress climatology
#####

LOADERS[:ncep_wind_stress_pair]            = c -> ncep_wind_stress_on_grid(get_field(c, :grid))
LOADERS[:ncep_zonal_wind_stress]           = c -> masked!(c, get_field(c, :ncep_wind_stress_pair)[1])
LOADERS[:ncep_meridional_wind_stress]      = c -> masked!(c, get_field(c, :ncep_wind_stress_pair)[2])
LOADERS[:zonal_wind_stress_bias_ncep]      = c -> maybe_diff!(c, :zonal_wind_stress,      :ncep_zonal_wind_stress)
LOADERS[:meridional_wind_stress_bias_ncep] = c -> maybe_diff!(c, :meridional_wind_stress, :ncep_meridional_wind_stress)

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

LOADERS[:time_in_years] = c ->
    total_jld2_timeseries_times(get_field(c, :averages_file)) ./ (365.25 * 24 * 3600)

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

LOADERS[:drift_time_in_years] = c ->
    total_jld2_timeseries_times(get_field(c, :averages_file)) ./ (365.25 * 24 * 3600)

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

# Lazy memo: build once on first call, then return the stored value.
lazy(builder) = let cell = Ref{Any}(nothing)
    () -> isnothing(cell[]) ? (cell[] = builder()) : cell[]
end

const zonal_latlon_grid = lazy() do
    LatitudeLongitudeGrid(CPU(); size = (ZONAL_NLON, ZONAL_NLAT, 1),
                                  longitude = (0, 360), latitude = (-90, 90), z = (0, 1))
end

const zonal_latlon_destination_field = lazy(() -> Field{Center, Center, Nothing}(zonal_latlon_grid()))

zonal_latitude_centers() = collect(φnodes(zonal_latlon_grid(), Center()))

# Conservatively regrid each level of `data_3d` (weighted by
# `ocean_mask_3d`) onto the lat-lon target grid, then average per latitude row.
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
for (sym, fts) in ((:time_mean_temperature_3d, :to_fts),
                   (:time_mean_salinity_3d,    :so_fts),
                   (:time_mean_buoyancy_3d,    :bo_fts))
    LOADERS[sym] = let f = fts
        c -> compute_time_mean(get_field(c, f); start_time = c.start_time, stop_time = c.stop_time)
    end
end
LOADERS[:initial_buoyancy_3d] = c -> Array(interior(get_field(c, :bo_fts)[1]))

# Zonal mean of any 3-D field already in the cache.
zonal_of(c, sym) = compute_zonal_mean(get_field(c, sym), get_field(c, :ocean_mask_3d),
                                       get_field(c, :zonal_regridder), ZONAL_NLON, ZONAL_NLAT)

# Each zonal section: (sym, source-3D, source-FTS for invalidation).
# WOA entries pass `nothing` since they don't depend on model output.
for (sym, src, fts) in ((:zonal_temperature,       :time_mean_temperature_3d, :to_fts),
                        (:zonal_salinity,          :time_mean_salinity_3d,    :so_fts),
                        (:zonal_buoyancy,          :time_mean_buoyancy_3d,    :bo_fts),
                        (:zonal_initial_buoyancy,  :initial_buoyancy_3d,      :bo_fts),
                        (:zonal_woa_temperature,   :woa_temperature,          nothing),
                        (:zonal_woa_salinity,      :woa_salinity,             nothing))
    LOADERS[sym] = let s = src
        wrapped = c -> zonal_of(c, s)
        isnothing(fts) ? disk_cached(wrapped, sym) :
                         disk_cached(wrapped, sym; source_fts_syms = fts)
    end
end

LOADERS[:zonal_temperature_bias] = c -> get_field(c, :zonal_temperature) .-
                                          get_field(c, :zonal_woa_temperature)
LOADERS[:zonal_salinity_bias]    = c -> get_field(c, :zonal_salinity) .-
                                          get_field(c, :zonal_woa_salinity)
LOADERS[:zonal_buoyancy_drift]   = c -> get_field(c, :zonal_buoyancy) .-
                                          get_field(c, :zonal_initial_buoyancy)

# Zonal MLD: regrid the 2-D surface MLD field, weighted by the surface
# ocean mask. NaNs (land) become 0 so they don't poison the regrid.
function zonal_mld(c, sym)
    raw = get_field(c, sym)
    isnothing(raw) && return nothing
    raw_3d  = reshape(replace(raw, NaN => zero(eltype(raw))), size(raw)..., 1)
    surface = get_field(c, :ocean_mask_3d)[:, :, end:end]
    return vec(compute_zonal_mean(raw_3d, surface, get_field(c, :zonal_regridder),
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
    haskey(case, :config) && return case.config
    p = lowercase(case.prefix)
    for cfg in (:tenthdegree, :halfdegree, :orca)
        occursin(string(cfg), p) && return cfg
    end
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
