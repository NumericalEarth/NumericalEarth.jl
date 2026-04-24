module GEBCO

export GEBCO2024

using Downloads
using ZipFile
using Oceananigans
using Oceananigans.DistributedComputations: @root
using Scratch

using ..DataWrangling: download_progress, Metadatum, metadata_path

import NumericalEarth.DataWrangling:
    metadata_filename,
    default_download_directory,
    all_dates,
    first_date,
    last_date,
    dataset_variable_name,
    download_dataset,
    longitude_interfaces,
    latitude_interfaces,
    z_interfaces,
    reversed_vertical_axis

download_GEBCO_cache::String = ""
function __init__()
    global download_GEBCO_cache = @get_scratch!("GEBCO")
end

GEBCO_bathymetry_variable_names = Dict(
    :bottom_height => "elevation",  # Variable name in NetCDF
)

"""
    GEBCO2024

General Bathymetric Chart of the Oceans 2024 release.
Global bathymetry and topography at 15 arc-second resolution.

The GEBCO_2024 Grid is a global terrain model for ocean and land,
providing elevation data on a 15 arc-second interval grid.

Reference: GEBCO Compilation Group (2024) GEBCO 2024 Grid
Data source: https://www.gebco.net/data_and_products/gridded_bathymetry_data/
"""
struct GEBCO2024 end

default_download_directory(::GEBCO2024) = download_GEBCO_cache
reversed_vertical_axis(::GEBCO2024) = false

# GEBCO covers the entire globe
longitude_interfaces(::GEBCO2024) = (-180, 180)
latitude_interfaces(::GEBCO2024) = (-90, 90)

# Grid size for 15 arc-second resolution
# 360° / (15/3600)° = 86400 points in longitude
# 180° / (15/3600)° = 43200 points in latitude
Base.size(::GEBCO2024) = (86400, 43200, 1)
Base.size(dataset::GEBCO2024, variable) = size(dataset)

all_dates(::GEBCO2024, args...) = nothing
first_date(::GEBCO2024, args...) = nothing
last_date(::GEBCO2024, args...) = nothing

# Bathymetry is 2D, so z_interfaces is just a placeholder
z_interfaces(::GEBCO2024) = (0, 1)

const GEBCOMetadatum = Metadatum{<:GEBCO2024}

dataset_variable_name(data::GEBCOMetadatum) = GEBCO_bathymetry_variable_names[data.name]

# GEBCO 2024 download URL from BODC
# Note: This is a large file (~8 GB zipped, ~22 GB unzipped)
const GEBCO_zip_url = "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2024/zip/"

# The expected NetCDF filename inside the ZIP
const GEBCO_nc_filename = "GEBCO_2024.nc"
metadata_filename(::GEBCO2024, name, date, bounding_box) = GEBCO_nc_filename

function download_dataset(metadatum::GEBCOMetadatum)
    filepath = metadata_path(metadatum)
    download_dir = metadatum.dir

    @root if !isfile(filepath)
        @info "Downloading GEBCO data: $(metadatum.name) to $download_dir..."
        @info "Note: GEBCO is a large dataset (~8 GB download, ~22 GB uncompressed). This may take a while."

        # Download the ZIP file
        zip_path = joinpath(download_dir, "GEBCO_2024.zip")

        try
            @info "Downloading from BODC..."
            Downloads.download(GEBCO_zip_url, zip_path; progress=download_progress)

            # Extract the NetCDF file from the ZIP using ZipFile.jl
            @info "Extracting NetCDF from ZIP archive..."
            zf = ZipFile.Reader(zip_path)
            extracted = false
            for f in zf.files
                if endswith(f.name, GEBCO_nc_filename)
                    open(filepath, "w") do io
                        write(io, read(f))
                    end
                    extracted = true
                    break
                end
            end
            close(zf)

            if !extracted
                error("Could not find $GEBCO_nc_filename in ZIP archive")
            end

            if isfile(filepath)
                @info "GEBCO data extracted successfully"
            else
                error("Failed to extract GEBCO NetCDF file")
            end

            # Clean up ZIP file to save space
            rm(zip_path; force=true)

        catch e
            @warn "Failed to download GEBCO: $e"

            # Clean up any partial download
            rm(zip_path; force=true)

            rethrow(e)
        end
    end

    return filepath
end

end # module
