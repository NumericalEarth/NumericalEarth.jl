"""
    GloFASStaticMaps(; system_version="version_4_0")

The static maps of the GloFAS LISFLOOD setup, on the reanalysis' 0.05° grid, from the
[CEMS auxiliary data](https://confluence.ecmwf.int/display/CEMS/Auxiliary+Data). Provides the
local drainage direction `:drainage_direction` in the PCRaster convention: the keypad code (1–9)
of the neighbor each cell drains to, `5` for a pit where a catchment ends, and `NaN` over the ocean.
"""
struct GloFASStaticMaps <: GloFASDataset
    system_version :: String
end

GloFASStaticMaps(; system_version="version_4_0") = GloFASStaticMaps(system_version)

DataWrangling.all_dates(::GloFASStaticMaps, variable)  = nothing
DataWrangling.first_date(::GloFASStaticMaps, variable) = nothing

const GloFASStaticMapsMetadatum = Metadatum{<:GloFASStaticMaps}

DataWrangling.longitude_name(::GloFASStaticMapsMetadatum) = "lon"
DataWrangling.latitude_name(::GloFASStaticMapsMetadatum)  = "lat"

# e.g. ldd_glofas_v4_0.nc
DataWrangling.metadata_filename(dataset::GloFASStaticMaps, name, date, region) = string(GloFAS_netcdf_variable_names[name], "_glofas_", replace(dataset.system_version, "version_" => "v"), ".nc")
DataWrangling.metadata_url(metadatum::GloFASStaticMapsMetadatum) = string("https://confluence.ecmwf.int/download/attachments/242067380/", metadatum.filename)

function Downloads.download(metadatum::GloFASStaticMapsMetadatum)
    filepath = metadata_path(metadatum)

    @root if !isfile(filepath)
        @info "Downloading GloFAS static map: $(metadatum.name) in $(metadatum.dir)..."
        mkpath(dirname(filepath))
        download_with_retries(metadata_url(metadatum), filepath; progress=DownloadProgress())
    end

    return filepath
end
