"""
Incorporate various datasets to be used for bathymetry, initialization, forcing,
restoring, or validation.
"""
module DataWrangling

export Metadata, Metadatum, DatewiseFilename, ECCOMetadatum, EN4Metadatum, all_dates, first_date, last_date
export WOAClimatology, WOAAnnual, WOAMonthly
export metadata_time_step, metadata_epoch
export LinearlyTaperedPolarMask
export DatasetRestoring, SurfaceFluxRestoring
export ERA5Hourly, ERA5Monthly
export AbstractDataset, SpatialLayout, GriddedLatLon, StationColumn, spatial_layout
export dataset_url, authenticate, download_file!, download_dataset
export preprocess_data
export conversion_units, convert_units, mangle
export Celsius, Kelvin, Millibar, InverseSign, MillimetersPerHour, CentimetersPerSecond
export GramPerKilogramMinus35, MilliliterPerLiter
export MolePerKilogram, MolePerLiter, MillimolePerKilogram, MillimolePerLiter
export MicromolePerKilogram, MicromolePerLiter, NanomolePerKilogram, NanomolePerLiter
export ShiftSouth, AverageNorthSouth
export test_dataset_contract, ContractReport, ContractCheck, is_conforming

using Oceananigans
using Downloads
using Printf
using Downloads

include("Datasets.jl")
using .Datasets
# Explicit `import` so methods defined later in DataWrangling (on `convert_units`,
# `mangle`, etc.) extend the generic functions declared in Datasets rather than
# shadowing them.
import .Datasets: convert_units, mangle, conversion_units, preprocess_data,
                  dataset_url, authenticate, download_file!, download_dataset,
                  spatial_layout

using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Grids: node
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interpolate
using Oceananigans: pretty_filesize, location
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

using Oceananigans.DistributedComputations
using Adapt

import Oceananigans.Fields: set!

#####
##### Downloading utilities
#####

next_fraction = Ref(0.0)
download_start_time = Ref(time_ns())

"""
    download_progress(total, now; filename="")
"""
function download_progress(total, now; filename="")
    messages = 10

    if total > 0
        fraction = now / total

        if fraction < 1 / messages && next_fraction[] == 0
            @info @sprintf("Downloading %s (size: %s)...", filename, pretty_filesize(total))
            next_fraction[] = 1 / messages
            download_start_time[] = time_ns()
        end

        if fraction > next_fraction[]
            elapsed = 1e-9 * (time_ns() - download_start_time[])
            msg = @sprintf(" ... downloaded %s (%d%% complete, %s)", pretty_filesize(now),
                           100fraction, prettytime(elapsed))
            @info msg
            next_fraction[] = next_fraction[] + 1 / messages
        end
    else
        if now > 0 && next_fraction[] == 0
            @info "Downloading $filename..."
            next_fraction[] = 1 / messages
            download_start_time[] = time_ns()
        end
    end

    return nothing
end

"""
    netrc_downloader(username, password, machine, dir)

Create a downloader that uses a netrc file to authenticate with the given machine.
This downloader writes the username and password in a file named `auth.netrc` (for Unix) and
`auth_netrc` (for Windows), located in the directory `dir`.
To avoid leaving the password on disk after the downloader has been used,
it is recommended to initialize the downloader in a temporary directory, which will be removed
after the download is complete.

For example:

```
mktempdir(dir) do tmp
    dowloader = netrc_downloader(username, password, machine, tmp)
    Downloads.download(fileurl, filepath; downloader)
end
```
"""
function netrc_downloader(username, password, machine, dir)
    netrc_file = netrc_permission_file(username, password, machine, dir)
    downloader = Downloads.Downloader()
    easy_hook  = (easy, _) -> begin
        Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_NETRC_FILE, netrc_file)
        # Bypass certificate verification because ecco.jpl.nasa.gov is using an untrusted CA certificate
        Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_SSL_VERIFYPEER, false)
    end
    downloader.easy_hook = easy_hook
    return downloader
end

# Code snippet adapted from https://github.com/evetion/SpaceLiDAR.jl/blob/master/src/utils.jl#L150
function netrc_permission_file(username, password, machine, dir)
    if Sys.iswindows()
        filepath = joinpath(dir, "auth_netrc")
    else
        filepath = joinpath(dir, "auth.netrc")
    end

    open(filepath, "a") do f
        write(f, "machine $machine login $username password $password\n")
    end

    return filepath
end

#####
##### FieldTimeSeries utilities
#####

function save_field_time_series!(fts; path, name, overwrite_existing=false)
    overwrite_existing && rm(path; force=true)

    times = on_architecture(CPU(), fts.times)
    grid  = on_architecture(CPU(), fts.grid)

    LX, LY, LZ = location(fts)
    ondisk_fts = FieldTimeSeries{LX, LY, LZ}(grid, times;
                                             backend = OnDisk(), path, name)

    Nt = length(times)
    for n = 1:Nt
        fill_halo_regions!(fts[n])
        set!(ondisk_fts, fts[n], n)
    end

    return nothing
end

function inpainted_metadata_path end

"""
    z_interfaces(dataset)

Return an array with the vertical interfaces (``z``-faces) of the dataset
that `metadata` corresponds to.
"""
function z_interfaces end
function longitude_interfaces end
function latitude_interfaces end
function reversed_vertical_axis end
function native_grid end
function binary_data_grid end
function binary_data_size end

default_mask_value(dataset) = NaN

# Fundamentals
include("metadata.jl")
include("metadata_field.jl")
include("metadata_field_time_series.jl")
include("inpainting.jl")
include("restoring.jl")
include("contract.jl")

#####
##### Default download implementation (requires Metadata, metadata_path)
#####

# Default transport: plain HTTP download via the standard library.
download_file!(path, url, ::AbstractDataset) =
    Downloads.download(url, path; progress=download_progress)

# Default orchestrator: for each date in the metadata, authenticate, resolve the URL,
# and call download_file!. Works for single-date (Metadatum) and multi-date cases.
# Dataset-specific methods (ECCO asyncmap, EN4 bulk zip, GLORYS SDK) override this.
function download_dataset(metadata::Metadata{<:AbstractDataset})
    @root for metadatum in metadata
        path = metadata_path(metadatum)
        isfile(path) && continue

        url = dataset_url(metadatum)
        if isnothing(url)
            error("No URL for dataset $(metadatum.dataset). Define ",
                  "`dataset_url(::Metadatum{<:$(typeof(metadatum.dataset))})`, ",
                  "override `download_dataset`, or place files manually at $path.")
        end

        authenticate(metadatum.dataset)
        mkpath(dirname(path))
        @info "Downloading $(metadatum.name) for $(typeof(metadatum.dataset)) to $(metadatum.dir)..."
        download_file!(path, url, metadatum.dataset)
    end
    return nothing
end

function metadata_time_step end
function metadata_epoch end

# Only temperature and salinity need a thorough inpainting because of stability,
# other variables can do with only a couple of passes. Sea ice variables
# cannot be inpainted because zeros in the data are physical, not missing values.
function default_inpainting(metadata)
    if metadata.name in (:temperature, :salinity)
        return NearestNeighborInpainting(Inf)
    elseif metadata.name in (:sea_ice_thickness, :sea_ice_concentration)
        return nothing
    else
        return NearestNeighborInpainting(5)
    end
end

# Datasets
include("ETOPO/ETOPO.jl")
include("ECCO/ECCO.jl")
include("GLORYS/GLORYS.jl")
include("ERA5/ERA5.jl")
include("EN4/EN4.jl")
include("ORCA/ORCA.jl")
include("WOA/WOA.jl")
include("JRA55/JRA55.jl")

using .ETOPO
using .ECCO
using .GLORYS
using .ERA5
using .EN4
using .ORCA
using .WOA
using .JRA55

end # module
