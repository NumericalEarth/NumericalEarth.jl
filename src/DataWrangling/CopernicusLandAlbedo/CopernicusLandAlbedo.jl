module CopernicusLandAlbedo

export CopernicusAlbedo, CopernicusAlbedoClimatology, build_monthly_climatology!

using Dates: Dates, DateTime, Month, year, month, daysinmonth
using Downloads: Downloads
using NCDatasets: NCDataset, defVar, nomissing
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, Metadata, Metadatum, BoundingBox,
                       metadata_path, default_download_directory, native_convention_longitude

import Oceananigans

download_CopernicusLandAlbedo_cache::String = ""
function __init__()
    global download_CopernicusLandAlbedo_cache = DataWrangling.download_cache("CopernicusLandAlbedo")
    return nothing
end

#####
##### Decode / blend helpers
#####

"""
    bluesky_blend(α_bs, α_ws, f)

Blend the black-sky (directional-hemispherical) albedo `α_bs` and the white-sky
(bi-hemispherical) albedo `α_ws` into the blue-sky (actual) albedo, with `f` the
diffuse fraction of the downwelling shortwave radiation:

```math
α = (1 - f) α_bs + f α_ws
```
"""
@inline bluesky_blend(α_bs, α_ws, f) = (1 - f) * α_bs + f * α_ws

# Missing (the decoded fill value) and out-of-range values map to NaN; albedo ∈ [0, 1].
@inline copernicus_albedo_decode(α::Number) = ifelse((α < 0) | (α > 1), NaN32, Float32(α))
@inline copernicus_albedo_decode(::Missing) = NaN32

#####
##### Dataset types
#####

abstract type AbstractCopernicusAlbedo end

"""
    CopernicusAlbedo(; diffuse_fraction = 0.2)

The Copernicus Global Land Service (CGLS) 1 km surface-albedo dataset, a dekadal
(10-daily) time series derived from SPOT/VGT and PROBA-V observations. Provides
`:albedo`, the broadband blue-sky albedo blended from the black-sky (`AL_DH_BB`)
and white-sky (`AL_BH_BB`) broadband albedos with diffuse fraction `diffuse_fraction`.

Files are global on a regular 1/112° latitude-longitude grid; build the `Metadata`
with a lon/lat [`BoundingBox`](@ref) to window a region at read time. Downloads come
from the C3S `satellite-albedo` catalogue entry and require the CDSAPI backend:
`using CDSAPI` with `~/.cdsapirc` credentials, the same setup as ERA5 (see this
module's README).

```jldoctest
julia> using NumericalEarth

julia> CopernicusAlbedo()
CopernicusAlbedo{Float64}(0.2)
```
"""
struct CopernicusAlbedo{FT} <: AbstractCopernicusAlbedo
    diffuse_fraction :: FT
end

CopernicusAlbedo(; diffuse_fraction = 0.2) = CopernicusAlbedo(diffuse_fraction)

"""
    CopernicusAlbedoClimatology(; diffuse_fraction = 0.2, years = 2019:2019)

A 12-month climatology of the [`CopernicusAlbedo`](@ref) blue-sky broadband albedo,
built by averaging all dekadal files of `years` month by month (NaN-aware, per pixel).
The monthly-mean files are generated on demand by [`build_monthly_climatology!`](@ref)
(triggered automatically on download) and yield a 12-slot `FieldTimeSeries` with
`Cyclical()` time indexing.

```jldoctest
julia> using NumericalEarth

julia> CopernicusAlbedoClimatology()
CopernicusAlbedoClimatology{Float64, UnitRange{Int64}}(0.2, 2019:2019)
```
"""
struct CopernicusAlbedoClimatology{FT, Y} <: AbstractCopernicusAlbedo
    diffuse_fraction :: FT
    years :: Y
end

CopernicusAlbedoClimatology(; diffuse_fraction = 0.2, years = 2019:2019) =
    CopernicusAlbedoClimatology(diffuse_fraction, years)

const CopernicusAlbedoMetadata{D} = Metadata{<:AbstractCopernicusAlbedo, D}
const CopernicusAlbedoMetadatum   = Metadatum{<:AbstractCopernicusAlbedo}

#####
##### Variables
#####

# Black-sky / white-sky broadband pair: the variable names in the local files, the
# names requested from the CDS, and the (variant) names the CDS delivery may use.
const copernicus_albedo_variables = Dict(:albedo => ("AL_DH_BB", "AL_BH_BB"))
const albedo_cds_request_variables = Dict(:albedo => ("albb_dh", "albb_bh"))
const albedo_source_variable_candidates = (blacksky = ("AL_DH_BB", "ALBB_DH", "albb_dh", "AL_DH"),
                                           whitesky = ("AL_BH_BB", "ALBB_BH", "albb_bh", "AL_BH"))

DataWrangling.available_variables(::AbstractCopernicusAlbedo) = copernicus_albedo_variables

# The black-sky member stands in for the pair; the blue-sky blend reads both.
DataWrangling.dataset_variable_name(metadata::CopernicusAlbedoMetadata) =
    first(copernicus_albedo_variables[metadata.name])

#####
##### Grid traits
#####

# Global 1/112° grid: 360° of longitude, latitude 80°N to 60°S, stored north→south.
Base.size(::AbstractCopernicusAlbedo, variable) = (40320, 15680, 1)

DataWrangling.is_three_dimensional(::CopernicusAlbedoMetadata) = false
DataWrangling.reversed_latitude_axis(::AbstractCopernicusAlbedo) = true
DataWrangling.longitude_name(::CopernicusAlbedoMetadata) = "lon"
DataWrangling.latitude_name(::CopernicusAlbedoMetadata)  = "lat"
DataWrangling.default_inpainting(::CopernicusAlbedoMetadata) = nothing
DataWrangling.default_download_directory(::AbstractCopernicusAlbedo) = download_CopernicusLandAlbedo_cache

# Albedo is a surface property: a reduced (`Nothing` z-location) field can be indexed
# at any k, as the interface flux kernels do via `stateindex` at k = Nz (matches ASTERGED).
Oceananigans.Fields.location(::CopernicusAlbedoMetadatum) = (Center, Center, Nothing)

# Analytic interfaces of the fixed 1/112° global product (40320×15680): longitude
# −180..180, latitude −60..80 (Δφ·15680 = 140 ⇒ 80 − 140 = −60).
DataWrangling.longitude_interfaces(::CopernicusAlbedoMetadata) = (-180, 180)
DataWrangling.latitude_interfaces(::CopernicusAlbedoMetadata)  = (-60, 80)

#####
##### Dates
#####

# CGLS dekads are stamped on day 10, day 20, and the last day of each month.
function copernicus_albedo_dekadal_dates(start_date, end_date)
    dates = DateTime[]
    d = DateTime(year(start_date), month(start_date), 1)
    while d ≤ end_date
        for day in (10, 20, daysinmonth(d))
            t = DateTime(year(d), month(d), day)
            start_date ≤ t ≤ end_date && push!(dates, t)
        end
        d += Month(1)
    end
    return dates
end

# C3S coverage of the 1 km v2 collection: SPOT/VGT from April 1998, PROBA-V until
# June 2020 (verified against the `satellite-albedo` request constraints).
const first_albedo_date = DateTime(1998, 4, 10)
const last_albedo_date  = DateTime(2020, 6, 30)

# SPOT ends May 2014; PROBA-V takes over from June 2014.
albedo_satellite(date) = date < DateTime(2014, 6, 1) ? "spot" : "proba"

DataWrangling.all_dates(::CopernicusAlbedo, variable) = copernicus_albedo_dekadal_dates(first_albedo_date, last_albedo_date)

# 12 climatological months; the year is arbitrary, only the month matters.
DataWrangling.all_dates(::CopernicusAlbedoClimatology, variable) = [DateTime(2018, m, 1) for m in 1:12]

#####
##### Filenames (date + variable keyed, region-independent — reused across regions)
#####

date_tag(date) = Dates.format(DateTime(date), "yyyymmdd")
years_tag(years) = string(first(years), "-", last(years))

DataWrangling.metadata_filename(::CopernicusAlbedo, name, date, region) =
    string("CGLS_1km_", name, "_", date_tag(date), ".nc")

DataWrangling.metadata_filename(dataset::CopernicusAlbedoClimatology, name, date, region) =
    string("CGLS_1km_", name, "_climatology_", years_tag(dataset.years),
           "_m", lpad(month(date), 2, '0'), ".nc")

#####
##### Download
#####
##### The dekadal-file download lives in `ext/NumericalEarthCDSAPIExt.jl` (needs
##### `using CDSAPI`); it fetches the source pair and calls `repack_albedo_pair` below.
##### Everything here is CDS-free: repacking, climatology download, and reading.
#####

function Downloads.download(metadata::Metadata{<:CopernicusAlbedoClimatology})
    @root for metadatum in metadata
        m = month(metadatum.dates)
        build_monthly_climatology!(metadatum.dataset; name = metadatum.name,
                                   months = m:m, dir = metadatum.dir)
    end
    return metadata_path(metadata)
end

# Locate the albedo variable in a C3S-delivered file, tolerating naming variants.
function find_albedo_variable(path, candidates)
    NCDataset(path) do source
        for candidate in candidates
            haskey(source, candidate) && return candidate
        end
        return nothing
    end
end

# Copy the packed integers and their CF attributes verbatim (the `.var` accessor
# skips scale/offset decoding), keeping the compact on-disk representation.
function copy_packed_variable!(destination, source, source_name, destination_name)
    variable = source[source_name].var
    raw = ndims(variable) == 3 ? variable[:, :, 1] : variable[:, :]

    attributes = Dict{String, Any}()
    for key in ("scale_factor", "add_offset", "long_name", "units")
        haskey(source[source_name].attrib, key) && (attributes[key] = source[source_name].attrib[key])
    end
    fill_value = get(source[source_name].attrib, "_FillValue", nothing)

    packed = if isnothing(fill_value)
        defVar(destination, destination_name, eltype(raw), ("lon", "lat");
               attrib = attributes, deflatelevel = 2, shuffle = true)
    else
        defVar(destination, destination_name, eltype(raw), ("lon", "lat");
               attrib = attributes, fillvalue = fill_value,
               deflatelevel = 2, shuffle = true)
    end
    packed.var[:, :] = raw
    return nothing
end

"""
    repack_albedo_pair(blacksky, whitesky, destination_names, filepath, expected_size)

Repack the broadband albedo bands from a downloaded black-sky/white-sky product pair
into one compact local NetCDF at `filepath`. `blacksky` and `whitesky` are
`(path, name)` tuples locating each source file and its albedo variable;
`destination_names` are the canonical local names (`AL_DH_BB`, `AL_BH_BB`). The packed
integer representation, CF decoding attributes, and north→south latitude order of the
source are preserved.
"""
function repack_albedo_pair(blacksky, whitesky, destination_names, filepath, expected_size)
    blacksky_path, blacksky_name = blacksky
    whitesky_path, whitesky_name = whitesky
    staging_path = filepath * ".tmp"

    NCDataset(staging_path, "c") do destination
        NCDataset(blacksky_path) do source
            λ = nomissing(source[coordinate_name(source, ("lon", "longitude"))][:])
            φ = nomissing(source[coordinate_name(source, ("lat", "latitude"))][:])
            (length(λ), length(φ)) == expected_size ||
                error("CGLS albedo file grid $((length(λ), length(φ))) does not match ",
                      "the expected $expected_size; the dataset traits need updating.")
            defVar(destination, "lon", λ, ("lon",))
            defVar(destination, "lat", φ, ("lat",))
            copy_packed_variable!(destination, source, blacksky_name, destination_names[1])
        end
        NCDataset(whitesky_path) do source
            copy_packed_variable!(destination, source, whitesky_name, destination_names[2])
        end
    end

    mv(staging_path, filepath; force = true)
    return nothing
end

function coordinate_name(source, candidates)
    for candidate in candidates
        haskey(source, candidate) && return candidate
    end
    error("None of the coordinate names $candidates found in the source file; ",
          "available variables: $(keys(source)).")
end

#####
##### Reading — blend the black-sky/white-sky pair into the blue-sky array. Regional
##### reads hyperslab exactly the native-grid cell window off disk (no global
##### materialization), so `retrieve_data` and `read_file_coords` must window identically.
#####

# 1-based native CELL range covered by `bbox` on the axis `(left, right)` split into `N`
# cells — must match `restrict()` in metadata_field.jl (returned count = `i⁺ - i⁻`, so the
# cell range `(i⁻+1):i⁺` has that same length, pinning the region offset to di = dj = 0).
function albedo_cell_range(bbox, interfaces, N)
    left, right = interfaces
    Δ = (right - left) / N
    i⁻ = clamp(floor(Int, (bbox[1] - left) / Δ - 1/2), 0, N)
    i⁺ = clamp(ceil( Int, (bbox[2] - left) / Δ + 1/2), 0, N)
    if i⁺ ≤ i⁻
        i⁺ = min(i⁻ + 1, N)
        i⁻ = max(i⁺ - 1, 0)
    end
    return (i⁻ + 1):i⁺
end

"""
    albedo_read_window(metadatum)

Global native CELL window `(icols, jrows)` (columns and ascending-frame rows) covering
the metadatum's `BoundingBox`, mirroring `construct_native_grid`'s `restrict`. Returns
`nothing` for the global path (no region, or a longitude window that spans/wraps the
±180 seam), in which case the caller reads the whole global grid. Pure integer math — no
file I/O — so the offline native grid can pin `length(icols)`/`length(jrows)` against it.
"""
function albedo_read_window(metadatum)
    region = metadatum.region
    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) || return nothing

    Nx, Ny, _ = size(metadatum.dataset, metadatum.name)

    # Longitude — mirror restrict_longitude(): map the bbox into the native convention,
    # then bail to the global path if it spans 360° or wraps past the +180 seam.
    native_longitude = DataWrangling.longitude_interfaces(metadatum)
    bbox_longitude = native_convention_longitude(region.longitude, native_longitude)
    left, right = native_longitude
    span = bbox_longitude[2] - bbox_longitude[1]
    (span == 360 || (bbox_longitude[1] ≥ left && bbox_longitude[2] > right)) && return nothing
    icols = albedo_cell_range(bbox_longitude, native_longitude, Nx)

    # Latitude — restrict() on (−60, 80) gives the ascending-frame cell range.
    jrows = albedo_cell_range(region.latitude, DataWrangling.latitude_interfaces(metadatum), Ny)

    return icols, jrows
end

function DataWrangling.retrieve_data(metadatum::CopernicusAlbedoMetadatum)
    path = metadata_path(metadatum)
    blacksky_name, whitesky_name = copernicus_albedo_variables[metadatum.name]
    f = Float32(metadatum.dataset.diffuse_fraction)
    win = albedo_read_window(metadatum)

    α = if isnothing(win)
        NCDataset(path) do ds
            Nx = ds.dim["lon"]
            Ny = ds.dim["lat"]
            blended = Array{Float32}(undef, Nx, Ny)
            # Read in latitude bands to cap the transient decoded (Union{Missing, Float64})
            # array; the full 632M-pixel grid decoded at once is ~5 GB.
            chunk = 1120
            for j in 1:chunk:Ny
                rows = j:min(j + chunk - 1, Ny)
                α_bs = copernicus_albedo_decode.(ds[blacksky_name][:, rows])
                α_ws = copernicus_albedo_decode.(ds[whitesky_name][:, rows])
                @. blended[:, rows] = bluesky_blend(α_bs, α_ws, f)
            end
            blended
        end
    else
        icols, jrows = win
        _, Ny, _ = size(metadatum.dataset, metadatum.name)
        # The file stores latitude north→south, so the ascending cells `jrows` sit at file
        # rows (Ny − last + 1):(Ny − first + 1). Only this hyperslab leaves disk.
        file_rows = (Ny - last(jrows) + 1):(Ny - first(jrows) + 1)
        NCDataset(path) do ds
            α_bs = copernicus_albedo_decode.(ds[blacksky_name][icols, file_rows])
            α_ws = copernicus_albedo_decode.(ds[whitesky_name][icols, file_rows])
            bluesky_blend.(α_bs, α_ws, f)
        end
    end

    # Files store latitude north→south; flip to ascending to match the native grid.
    # `reversed_latitude_axis` flips the coordinate in `read_file_coords`; this override
    # owns the matching data flip, so the two must stay in sync.
    return reverse(α, dims = 2)
end

# Window the coordinates through the SAME `albedo_read_window` as the data, so the
# regional read is bit-exact with the global-then-slice path (region offset di = dj = 0).
function DataWrangling.read_file_coords(metadatum::CopernicusAlbedoMetadatum)
    λc, φc = NCDataset(metadata_path(metadatum)) do ds
        nomissing(ds[DataWrangling.longitude_name(metadatum)][:]),
        nomissing(ds[DataWrangling.latitude_name(metadatum)][:])
    end
    reverse!(φc)  # file latitude is north→south; make ascending to match the data flip
    win = albedo_read_window(metadatum)
    isnothing(win) && return λc, φc
    icols, jrows = win
    return λc[icols], φc[jrows]
end

#####
##### Monthly climatology builder
#####

"""
    build_monthly_climatology!(dataset::CopernicusAlbedoClimatology;
                               name = :albedo,
                               months = 1:12,
                               dir = default_download_directory(dataset),
                               latitude_chunk = 1120)

For each calendar month in `months`, download every dekadal albedo file of
`dataset.years` falling in that month, average the black-sky and white-sky
broadband albedos pixel by pixel (NaN-aware: pixels missing in a dekad are
excluded from its mean), and write one monthly-mean NetCDF to `dir` under the
name computed by `metadata_filename`. Months whose file already exists are
skipped. Returns the paths of the 12 (or `length(months)`) monthly files.
"""
function build_monthly_climatology!(dataset::CopernicusAlbedoClimatology;
                                    name = :albedo,
                                    months = 1:12,
                                    dir = default_download_directory(dataset),
                                    latitude_chunk = 1120)

    raw_dataset = CopernicusAlbedo(diffuse_fraction = dataset.diffuse_fraction)
    dekads = DataWrangling.all_dates(raw_dataset, name)
    variable_names = copernicus_albedo_variables[name]
    paths = String[]

    for m in months
        filepath = joinpath(dir, DataWrangling.metadata_filename(dataset, name, DateTime(2018, m, 1), nothing))
        push!(paths, filepath)
        isfile(filepath) && continue

        dates = [d for d in dekads if month(d) == m && year(d) in dataset.years]
        isempty(dates) && error("No CGLS albedo dekads fall in month $m of years $(dataset.years).")

        metadata = Metadata(name; dataset = raw_dataset, dates, dir)
        Downloads.download(metadata)
        source_paths = metadata_path(metadata)

        @info "Averaging $(length(source_paths)) dekads into month $m of the CGLS albedo climatology..."
        write_monthly_mean(filepath, source_paths, variable_names, latitude_chunk)
    end

    return paths
end

function write_monthly_mean(filepath, source_paths, variable_names, latitude_chunk)
    λ, φ = NCDataset(first(source_paths)) do ds
        nomissing(ds["lon"][:]), nomissing(ds["lat"][:])
    end
    Nx, Ny = length(λ), length(φ)
    staging_path = filepath * ".tmp"

    NCDataset(staging_path, "c") do destination
        defVar(destination, "lon", λ, ("lon",))
        defVar(destination, "lat", φ, ("lat",))

        for variable_name in variable_names
            monthly_mean = defVar(destination, variable_name, Float32, ("lon", "lat");
                                  deflatelevel = 2, shuffle = true)

            for j in 1:latitude_chunk:Ny
                rows = j:min(j + latitude_chunk - 1, Ny)
                Σα = zeros(Float32, Nx, length(rows))
                n = zeros(Int32, Nx, length(rows))

                for path in source_paths
                    NCDataset(path) do ds
                        α = copernicus_albedo_decode.(ds[variable_name][:, rows])
                        @. Σα += ifelse(isnan(α), 0f0, α)
                        @. n += !isnan(α)
                    end
                end

                monthly_mean[:, rows] = @. ifelse(n == 0, NaN32, Σα / n)
            end
        end
    end

    mv(staging_path, filepath; force = true)
    return nothing
end

end # module CopernicusLandAlbedo
