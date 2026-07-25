module CopernicusLandVegetation

export CopernicusVegetation, retrieval_flag_mask, unusable_retrieval_flags

using Dates: Dates, DateTime
using NCDatasets: NCDataset, nomissing
using Oceananigans: Center

using ..DataWrangling: DataWrangling, Metadata, Metadatum, BoundingBox,
                       metadata_path, dekadal_dates, nan_convert_missing,
                       native_convention_longitude, native_cell_range

import Oceananigans

download_CopernicusLandVegetation_cache::String = ""
function __init__()
    global download_CopernicusLandVegetation_cache = DataWrangling.download_cache("CopernicusLandVegetation")
    return nothing
end

#####
##### Retrieval flags
#####

# Single-bit diagnostics of the `retrieval_flag` bitfield, read from the product's own CF
# `flag_masks`/`flag_meanings` pair. The remaining masks in the file are multi-bit fields
# recording which TIP variant produced the retrieval, not quality rejections.
const retrieval_flag_masks = (obs_is_fillvalue = 0x00000001,
                              tip_untrusted    = 0x00000040,
                              obs_unusable     = 0x00000080,
                              obs_inconsistent = 0x00000100,
                              obs_nosnow_hiunc = 0x00000200,
                              obs_snow_hiunc   = 0x00000400,
                              tip_nounc        = 0x00000800,
                              obs_nosnow_only  = 0x00001000,
                              obs_snow_only    = 0x00002000)

"""
    retrieval_flag_mask(names::Symbol...)

Combine the named `retrieval_flag` bits into a single mask, for use as the
`screened_flags` of a [`CopernicusVegetation`](@ref) dataset. Pixels whose flag
shares any bit with the mask are read as `NaN`.

Available names are `$(keys(retrieval_flag_masks))`.

```jldoctest
julia> using NumericalEarth

julia> retrieval_flag_mask(:obs_unusable, :tip_untrusted)
0x000000c0
```
"""
function retrieval_flag_mask(names::Symbol...)
    mask = 0x00000000
    for name in names
        haskey(retrieval_flag_masks, name) ||
            throw(ArgumentError("$name is not a retrieval flag; valid names are $(keys(retrieval_flag_masks))."))
        mask |= retrieval_flag_masks[name]
    end
    return mask
end

"""
    unusable_retrieval_flags()

The recommended quality screen: the `retrieval_flag` bits that mark a pixel as a fill
value, an untrusted inversion, or built from unusable observations. Pass it as
`CopernicusVegetation(screened_flags = unusable_retrieval_flags())` to drop those
pixels — over a vegetated summer scene it removes a few hundredths of a percent.

Most of the other bits are descriptive rather than disqualifying, and cannot be
screened on by reading their names: `:obs_inconsistent` and `:obs_nosnow_only` are set
on ~95% and ~100% of a mid-latitude July scene respectively, so including either
discards nearly the whole field. The high-uncertainty bits (`:obs_nosnow_hiunc`,
`:obs_snow_hiunc`) are a defensible stricter screen — the first covers ~14% of that
same scene — and can be added with [`retrieval_flag_mask`](@ref).

```jldoctest
julia> using NumericalEarth

julia> unusable_retrieval_flags()
0x000000c1
```
"""
unusable_retrieval_flags() =
    retrieval_flag_mask(:obs_is_fillvalue, :tip_untrusted, :obs_unusable)

#####
##### Dataset type
#####

"""
    CopernicusVegetation(; screened_flags = 0x00000000)

The C3S 300 m leaf-area-index climate data record: a dekadal (10-daily) global time
series retrieved from Sentinel-3 OLCI and SLSTR broadband albedo with the Two-stream
Inversion Package (TIP). Provides `:leaf_area_index`, the *effective* leaf area index
``Λ`` (m² m⁻²).

Files are on a regular 1/336° latitude-longitude grid covering 80°N–60°S; build the
`Metadata` with a lon/lat [`BoundingBox`](@ref) to select a region. Downloads come from
the C3S `satellite-lai-fapar` catalogue entry and require the CDSAPI backend: `using
CDSAPI` with `~/.cdsapirc` credentials, the same setup as ERA5 (see this module's
README). Because the global grid holds 5.7 billion cells per dekad, a region is
requested from the Climate Data Store and subset server-side, so one file is stored per
date and region.

`screened_flags` is a mask of `retrieval_flag` bits whose pixels are read as `NaN`; the
default keeps every pixel, including the product's fill values. See
[`unusable_retrieval_flags`](@ref) and [`retrieval_flag_mask`](@ref).

```jldoctest
julia> using NumericalEarth

julia> CopernicusVegetation()
CopernicusVegetation(0x00000000)
```
"""
struct CopernicusVegetation
    screened_flags :: UInt32
end

CopernicusVegetation(; screened_flags = 0x00000000) = CopernicusVegetation(screened_flags)

const CopernicusVegetationMetadata{D} = Metadata{<:CopernicusVegetation, D}
const CopernicusVegetationMetadatum   = Metadatum{<:CopernicusVegetation}

#####
##### Variables
#####

const copernicus_vegetation_variables = Dict(:leaf_area_index => "LAI")

# The name of each variable in the Climate Data Store request.
const vegetation_cds_request_variables = Dict(:leaf_area_index => "lai")

const retrieval_flag_variable = "retrieval_flag"

DataWrangling.available_variables(::CopernicusVegetation) = copernicus_vegetation_variables

DataWrangling.dataset_variable_name(metadata::CopernicusVegetationMetadata) =
    copernicus_vegetation_variables[metadata.name]

#####
##### Grid traits
#####

const Δ = 1/336

# Global 1/336° grid: 360° of longitude, latitude 80°N to 60°S, stored north→south.
const Nλ = round(Int, 360 / Δ)
const Nφ = round(Int, 140 / Δ)

Base.size(::CopernicusVegetation, variable) = (Nλ, Nφ, 1)

DataWrangling.is_three_dimensional(::CopernicusVegetationMetadata) = false
DataWrangling.reversed_latitude_axis(::CopernicusVegetation) = true
DataWrangling.longitude_name(::CopernicusVegetationMetadata) = "lon"
DataWrangling.latitude_name(::CopernicusVegetationMetadata)  = "lat"
DataWrangling.default_inpainting(::CopernicusVegetationMetadata) = nothing
DataWrangling.default_download_directory(::CopernicusVegetation) = download_CopernicusLandVegetation_cache

Oceananigans.Fields.location(::CopernicusVegetationMetadatum) = (Center, Center, Nothing)

# The product's `GeoTransform` places the grid origin on the exact corner (-180, 80), so
# the cell interfaces are whole degrees.
DataWrangling.longitude_interfaces(::CopernicusVegetationMetadata) = (-180, 180)
DataWrangling.latitude_interfaces(::CopernicusVegetationMetadata)  = (-60, 80)

# Four native cells of margin for interpolation stencils at a region's boundary.
DataWrangling.default_horizontal_padding(::CopernicusVegetation) = 4Δ

#####
##### Dates
#####

# Sentinel-3 coverage of the 300 m collection, verified against the `satellite-lai-fapar`
# request constraints: the record opens mid-2018 and the consolidated CDR ends in 2024.
const first_vegetation_date = DateTime(2018, 7, 10)
const last_vegetation_date  = DateTime(2024, 12, 31)

DataWrangling.all_dates(::CopernicusVegetation, variable) =
    dekadal_dates(first_vegetation_date, last_vegetation_date)

#####
##### Filenames — keyed by variable, date, and region, because each region is a
##### separate server-side subset of the native grid.
#####

date_tag(date) = Dates.format(DateTime(date), "yyyy-mm-dd")

bound_tag(bound) = replace(string(round(bound, digits = 3)), "-" => "m")

region_tag(::Nothing) = "global"

function region_tag(region::BoundingBox)
    (isnothing(region.longitude) || isnothing(region.latitude)) && return "global"
    west, east = region.longitude
    south, north = region.latitude
    return string("lon", bound_tag(west), "-", bound_tag(east),
                  "_lat", bound_tag(south), "-", bound_tag(north))
end

DataWrangling.metadata_filename(::CopernicusVegetation, name, date, region) =
    string("C3S_300m_", name, "_", date_tag(date), "_", region_tag(region), ".nc")

#####
##### Reading
#####
##### The delivered files are netCDF-CF, so `NCDatasets` applies `scale_factor`,
##### `add_offset`, and `_FillValue` on read and fill values arrive as `missing`. The
##### `retrieval_flag` bitfield carries a `scale_factor` too, which would decode a
##### 30-bit integer into a lossy `Float32`, so it is read through `.var` unscaled.
#####
##### A file covers the requested region plus a download margin, so both `retrieve_data`
##### and `read_file_coords` hyperslab the same window: exactly the cells of the native
##### grid the data is set on. Handing `set_region_data!` that exact count pins the region
##### offset to zero, which keeps the read bit-exact and lifts only the cells that are
##### actually used off disk.
#####

# The coordinate variables label each cell's west and south edge rather than its center,
# the half-cell shift the product's `GeoTransform` implies.
function file_cell_corners(ds, metadatum)
    return nomissing(ds[DataWrangling.longitude_name(metadatum)][:]),
           nomissing(ds[DataWrangling.latitude_name(metadatum)][:])
end

# Global 1-based native cell ranges (longitude, ascending latitude) covering the region.
function native_window(metadatum)
    Nx, Ny, _ = size(metadatum.dataset, metadatum.name)
    region = metadatum.region

    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) ||
        return 1:Nx, 1:Ny

    native_longitude = DataWrangling.longitude_interfaces(metadatum)
    bbox_longitude = native_convention_longitude(region.longitude, native_longitude)

    last(bbox_longitude) > last(native_longitude) &&
        throw(ArgumentError("The requested longitude window $(region.longitude) wraps the ±180° " *
                            "seam of the C3S 300 m grid. Split it into two requests, one on each " *
                            "side of the seam."))

    icols = native_cell_range(bbox_longitude, native_longitude, Nx)
    jrows = native_cell_range(region.latitude, DataWrangling.latitude_interfaces(metadatum), Ny)

    return icols, jrows
end

# Map a global native cell range onto the columns/rows a file holds, from the global cell
# index its first coordinate labels.
function file_cell_range(global_range, first_corner, origin, cells_in_file, axis)
    offset = round(Int, (first_corner - origin) / Δ)
    local_range = (first(global_range) - offset):(last(global_range) - offset)

    (first(local_range) ≥ 1 && last(local_range) ≤ cells_in_file) ||
        error("The stored C3S 300 m file covers $cells_in_file $axis cells starting at " *
              "$first_corner, which does not span the $(length(global_range)) cells the " *
              "requested region needs. Delete the file and download it again.")

    return local_range
end

# The one place the native window is resolved against a file, so the data and the
# coordinates cannot come from different windows. `rows` indexes the stored north→south
# order; `ascending_rows` indexes the south→north order the native grid expects.
function file_read_window(ds, metadatum)
    icols, jrows = native_window(metadatum)
    λ, φ = file_cell_corners(ds, metadatum)

    columns = file_cell_range(icols, first(λ), -180, length(λ), "longitude")
    ascending_rows = file_cell_range(jrows, last(φ), -60, length(φ), "latitude")
    rows = (length(φ) - last(ascending_rows) + 1):(length(φ) - first(ascending_rows) + 1)

    return (; λ, φ, columns, rows, ascending_rows)
end

function DataWrangling.retrieve_data(metadatum::CopernicusVegetationMetadatum)
    variable = DataWrangling.dataset_variable_name(metadatum)
    screened_flags = metadatum.dataset.screened_flags

    Λ = NCDataset(metadata_path(metadatum)) do ds
        window = file_read_window(ds, metadatum)
        Λ = nan_convert_missing.(Float32, ds[variable][window.columns, window.rows, 1])

        if screened_flags != 0
            flags = ds[retrieval_flag_variable].var[window.columns, window.rows, 1]
            Λ = ifelse.(iszero.(flags .& screened_flags), Λ, NaN32)
        end

        Λ
    end

    # Files store latitude north→south; flip to ascending to match the native grid.
    return reverse(Λ, dims = 2)
end

function DataWrangling.read_file_coords(metadatum::CopernicusVegetationMetadatum)
    return NCDataset(metadata_path(metadatum)) do ds
        window = file_read_window(ds, metadatum)

        # Shift the west/south edges onto cell centers, and return latitude ascending to
        # match the flip in `retrieve_data`.
        window.λ[window.columns] .+ Δ/2, reverse(window.φ)[window.ascending_rows] .+ Δ/2
    end
end

#####
##### Download request — the area the Climate Data Store subsets server-side. Kept here
##### (rather than in the CDSAPI extension) so it is testable without the backend.
#####

# Native cells of margin requested around a region. The Climate Data Store snaps the
# requested area onto the product's pixel lattice, and `restrict` can put the native grid
# a cell outside the region, so ask for enough margin that the delivered file covers the
# grid whichever way the snap falls.
const request_margin = 6

"""
    vegetation_request_area(region)

The `[north, west, south, east]` area of a Climate Data Store request covering `region`,
widened by `$(request_margin)` native cells so the delivered file spans the whole
center-bracketed native grid, and clipped to the product's 80°N–60°S coverage. Returns
`nothing` for a global request.
"""
vegetation_request_area(::Nothing) = nothing

function vegetation_request_area(region::BoundingBox)
    (isnothing(region.longitude) || isnothing(region.latitude)) && return nothing

    native_longitude = (-180, 180)
    west, east = native_convention_longitude(region.longitude, native_longitude)

    east > last(native_longitude) &&
        throw(ArgumentError("The requested longitude window $(region.longitude) wraps the ±180° " *
                            "seam of the C3S 300 m grid. Split it into two requests, one on each " *
                            "side of the seam, to avoid downloading the whole globe."))

    south, north = region.latitude
    margin = request_margin * Δ

    return [min(north + margin, 80), west - margin, max(south - margin, -60), east + margin]
end

end # module CopernicusLandVegetation
