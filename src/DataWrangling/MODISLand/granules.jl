#####
##### Granule names
#####

"""
    parse_granule_name(name)

Split a MODIS land-product granule name — `MCD15A2H.A2020185.h10v05.061.2020340132006` —
into `(; date, tile, production)`: the composite's first day, the sinusoidal tile it covers,
and the processing timestamp that distinguishes reprocessings of the same tile and date.
"""
function parse_granule_name(name::AbstractString)
    m = match(r"\.A(\d{4})(\d{3})\.(h\d{2}v\d{2})\.\d{3}\.(\d+)", name)
    isnothing(m) &&
        throw(ArgumentError("could not parse a MODIS granule name from \"$name\""))
    composite_year, composite_day, tile, production = m.captures
    return (date = DateTime(parse(Int, composite_year)) + Day(parse(Int, composite_day) - 1),
            tile = String(tile),
            production = parse(Int, production))
end

"""
    select_granules(urls, date)

Keep the granule `urls` whose composite begins on `date`, one per sinusoidal tile: the
most recently processed, so a reprocessed tile supersedes its predecessor. A bounding-box
granule search returns the neighboring composites too (their date ranges overlap the
requested day), which is why the date is matched rather than trusted.
"""
function select_granules(urls, date)
    latest = Dict{String, Tuple{Int, String}}()
    for url in urls
        granule = parse_granule_name(basename(url))
        granule.date == DateTime(date) || continue
        current = get(latest, granule.tile, nothing)
        if isnothing(current) || granule.production > first(current)
            latest[granule.tile] = (granule.production, url)
        end
    end
    return [last(latest[tile]) for tile in sort!(collect(keys(latest)))]
end

#####
##### The regional lattice the granules are warped onto
#####

"""
    regional_lattice(metadata)

The regional latitude-longitude window the sinusoidal granules are reprojected onto:
`(; west, south, east, north, Nx, Ny)`, in degrees and cells of the product's 1/240°
lattice.

The window is exactly the set of native cells `native_grid` keeps for the metadata's
region, so the stored file and the native grid share their cells one for one. That pins the
region offset of the shared regrid to zero instead of leaving it to a floating-point
comparison between the grid's nodes and the file's coordinates — the difference between a
correct read and one shifted by a cell on a fine grid.
"""
function regional_lattice(metadata::MODISLandMetadata)
    region = metadata.region
    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) ||
        throw(ArgumentError("regional_lattice requires a bounded (longitude, latitude) BoundingBox."))

    Nx, Ny, _ = size(metadata.dataset, metadata.name)
    native_longitude = DataWrangling.longitude_interfaces(metadata)
    native_latitude  = DataWrangling.latitude_interfaces(metadata)
    bbox_longitude = native_convention_longitude(region.longitude, native_longitude)

    last(bbox_longitude) > last(native_longitude) &&
        throw(ArgumentError("The requested longitude window $(region.longitude) wraps the ±180° " *
                            "seam of the MODIS sinusoidal grid's reprojection. Split it into two " *
                            "requests, one on each side of the seam."))

    icols = native_cell_range(bbox_longitude, native_longitude, Nx)
    jrows = native_cell_range(region.latitude, native_latitude, Ny)

    return (west  = first(native_longitude) + (first(icols) - 1) * MODIS_LATTICE_SPACING,
            east  = first(native_longitude) + last(icols) * MODIS_LATTICE_SPACING,
            south = first(native_latitude) + (first(jrows) - 1) * MODIS_LATTICE_SPACING,
            north = first(native_latitude) + last(jrows) * MODIS_LATTICE_SPACING,
            Nx = length(icols), Ny = length(jrows))
end

#####
##### Granule discovery
#####

"""
    MissingGranulesError

Raised when the Common Metadata Repository holds no granule for a requested region and
date. The record has occasional holes where an instrument outage prevented a composite —
2016-02-18 is one — so [`build_lai_climatology!`](@ref) catches this and composites the
rest, while an explicit read of that date still fails.
"""
struct MissingGranulesError <: Exception
    message :: String
end

Base.showerror(io::IO, err::MissingGranulesError) = print(io, err.message)

"""
    granule_urls(metadatum)

Query the Common Metadata Repository and return the download URLs of the granules covering
the metadatum's region and date, one per sinusoidal tile (see [`select_granules`](@ref)).
Requires network access, but no credentials.
"""
function granule_urls(metadatum::MODISLandMetadatum)
    dataset = metadatum.dataset

    # One day of one product is a handful of sinusoidal tiles, so a single page always covers it.
    url = cmr_granules_url(modis_short_name(dataset), modis_version(dataset), metadatum.region;
                           date = metadatum.dates)

    candidates = mktempdir() do tmp
        json = joinpath(tmp, "cmr_granules.json")
        download_with_retries(url, json; description = "CMR granule query")
        text = read(json, String)
        unique(m.match for m in eachmatch(r"https://[^\"]+\.hdf", text))
    end

    granules = select_granules(candidates, metadatum.dates)
    isempty(granules) &&
        throw(MissingGranulesError(
            "The Common Metadata Repository holds no $(modis_short_name(dataset)) granules " *
            "for the region $(metadatum.region) on $(metadatum.dates). The record has " *
            "occasional holes where an instrument outage prevented a composite; a " *
            "climatology skips them, but a read of that date alone cannot."))

    return granules
end

#####
##### Download
#####

function Downloads.download(metadata::MODISLandMetadata)
    @root for metadatum in metadata
        path = metadata_path(metadatum)
        isfile(path) || modis_granules_to_netcdf(metadatum, path)
    end
    return metadata_path(metadata)
end

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded.
modis_granules_to_netcdf(metadatum, nc_path) =
    error("Reading MODIS HDF-EOS granules requires ArchGDAL.jl built with GDAL's HDF4 " *
          "driver, and NASA Earthdata credentials (EARTHDATA_USERNAME / " *
          "EARTHDATA_PASSWORD). Load ArchGDAL with `using ArchGDAL`.")

#####
##### Reading
#####

function DataWrangling.retrieve_data(metadatum::MODISLAIMetadatum)
    variable = DataWrangling.dataset_variable_name(metadatum)
    mask = screened_flags(metadatum.dataset)

    # The land-cover code describes the surface, not a retrieval, so the retrieval screen does
    # not apply to it: a cloudy urban pixel is still urban.
    metadatum.name === :landcover_code &&
        return NCDataset(metadata_path(metadatum)) do ds
            mask_lai_landcover.(ds[variable][:, :])
        end

    return NCDataset(metadata_path(metadatum)) do ds
        𝒜 = mask_lai_fill.(ds[variable][:, :])

        if !iszero(mask)
            qc = ds[lai_quality_variable][:, :]
            extra_qc = ds[lai_extra_quality_variable][:, :]
            𝒜 = ifelse.(lai_screened.(qc, extra_qc, mask), NaN32, 𝒜)
        end

        𝒜
    end
end

function DataWrangling.retrieve_data(metadatum::MODISLAIClimatologyMetadatum)
    variable = DataWrangling.dataset_variable_name(metadatum)
    return NCDataset(metadata_path(metadatum)) do ds
        Float32.(ds[variable][:, :])
    end
end

# `QC` is an enumerated classification outcome (0 good classified land, 10 no data), not a
# packed bitfield like `FparLai_QC` — nothing to screen, and bit-masking it would be nonsense.
function DataWrangling.retrieve_data(metadatum::MODISLandCoverMetadatum)
    variable = DataWrangling.dataset_variable_name(metadatum)
    valid = metadatum.name === :landcover_class ? landcover_valid_range(metadatum.dataset) :
            metadatum.name === :quality_flag    ? (0:10) : (1:2)

    return NCDataset(metadata_path(metadatum)) do ds
        mask_landcover_fill.(ds[variable][:, :], Ref(valid))
    end
end
