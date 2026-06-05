module GLODAP

export GLODAPClimatology

using Adapt: Adapt
using Dates: month
using Downloads: Downloads
using NCDatasets: Dataset
using Scratch: Scratch, @get_scratch!
using Tar: Tar
using CodecZlib: GzipDecompressorStream

using ..DataWrangling: DataWrangling, Metadata, Metadatum, metadata_path,
                       reversed_vertical_axis, DownloadProgress

download_GLODAP_cache::String = ""
function __init__()
    global download_GLODAP_cache = @get_scratch!("GLODAP")
end

const GLODAP_url = "https://www.nodc.noaa.gov/archive/arc0107/0162565/1.1/data/0-data/mapped/GLODAPv2_Mapped_Climatology.tar.gz"

GLODAP_variable_names = Dict(
    :temperature       => "temperature",
    :salinity          => "salinity",
    :phosphate         => "PO4",
    :nitrate           => "NO3",
    :silicate          => "silicate",
    :dissolved_oxygen  => "oxygen",
    :dic               => "TCO2",
    :preindustrial_dic => "PI_TCO2",
    :alkalinity        => "TALK"
    )

# this is necessary because the variable name in the dataset
# does not necessarily correspond to what is in the filename
# Note: We are excluding the following files for now:
#       GLODAPv2.OmegaCinsitu.nc
#       GLODAPv2.pHtsinsitu.nc
#       GLODAPv2.OmegaAinsitu.nc
#       GLODAPv2.pHts25p0.nc
GLODAP_file_variable_names = Dict(
    :temperature       => "theta",
    :salinity          => "salinity",
    :phosphate         => "phosphate",
    :nitrate           => "nitrate",
    :silicate          => "silicate",
    :dissolved_oxygen  => "oxygen",
    :dic               => "tco2",
    :preindustrial_dic => "tco2", 
    :alkalinity        => "talk"
    )

# Dataset types
abstract type GLODAPDataset end

struct GLODAPClimatology <: GLODAPDataset
    product_year :: Int
end

GLODAPClimatology(; product_year=2016) = GLODAPClimatology(product_year)

function DataWrangling.default_download_directory(::GLODAPClimatology)
    return mkpath(download_GLODAP_cache)
end

# GLODAP stores depth as positive values, surface first 
# GLODAP actually stores pressure in dbar, not depth in meters
DataWrangling.reversed_vertical_axis(::GLODAPClimatology) = true

DataWrangling.longitude_interfaces(::GLODAPClimatology) = (-180, 180)
DataWrangling.latitude_interfaces(::GLODAPClimatology) = (-90, 90)
DataWrangling.longitude_name(::Metadata{<:GLODAPDataset}) = "lon"
DataWrangling.latitude_name(::Metadata{<:GLODAPDataset})  = "lat"
DataWrangling.available_variables(::GLODAPClimatology) = GLODAP_variable_names

"""
    glodap_z_interfaces_from_centers(depth_centers)

Compute cell interfaces (negative z, bottom-first) from GLODAP standard
depth centers (positive, surface-first).

** We use the assumption that pressure is dbar is close enough to depth in meters **
"""
function glodap_z_interfaces_from_centers(depth_centers)
    N = length(depth_centers)

    # Compute cell faces (positive depth, surface-first)
    faces = Vector{Float64}(undef, N + 1)
    faces[1] = 0.0 # surface
    for k in 1:N-1
        faces[k+1] = (depth_centers[k] + depth_centers[k+1]) / 2
    end
    # Bottom face: extrapolate below deepest center
    faces[N+1] = depth_centers[N] + (depth_centers[N] - depth_centers[N-1]) / 2

    # Convert to negative z, bottom-first
    return [-faces[N + 2 - k] for k in 1:N+1]
end

# Read size and z_interfaces from the actual GLODAP NetCDF file.
function Base.size(metadata::Metadata{<:GLODAPDataset})
    path = metadata_path(first(metadata))
    ds = Dataset(path)
    Nlon = length(ds["lon"])
    Nlat = length(ds["lat"])
    Nz = length(ds["Pressure"])
    close(ds)

    Nt = metadata.dates isa AbstractArray ? length(metadata.dates) : 1
    return (Nlon, Nlat, Nz, Nt)
end

function DataWrangling.z_interfaces(metadata::Metadata{<:GLODAPDataset})
    path = metadata_path(first(metadata))
    ds = Dataset(path)
    depth_centers = Float64.(ds["Pressure"][:])
    close(ds)
    return glodap_z_interfaces_from_centers(depth_centers)
end

const GLODAPMetadatum   = Metadatum{<:GLODAPDataset}

DataWrangling.metaprefix(::GLODAPMetadatum) = "GLODAPMetadatum"

function DataWrangling.metadata_filename(::GLODAPClimatology, name, date, region)
    varname = GLODAP_file_variable_names[name]
    return "GLODAPv2.$(varname).nc"
end

# augment the metadata file path
DataWrangling.metadata_path(metadata::GLODAPMetadatum) =
    joinpath(metadata.dir, "GLODAPv2_Mapped_Climatologies", metadata.filename)

DataWrangling.is_three_dimensional(::GLODAPMetadatum) = true

function inpainted_metadata_filename(metadata::GLODAPMetadatum)
    without_extension = metadata.filename[1:end-3]
    var = string(metadata.name)
    return without_extension * "_" * var * "_inpainted.jld2"
end

DataWrangling.inpainted_metadata_path(metadata::GLODAPMetadatum) = joinpath(metadata.dir, inpainted_metadata_filename(metadata))

function Downloads.download(metadata::Metadatum{<:GLODAPDataset})
    path = metadata_path(metadata)
    needs_download = !isfile(path)

    if needs_download
        mktempdir() do tmpdir
            archive = joinpath(tmpdir, "GLODAPv2_Mapped_Climatology.tar.gz")
            @info "path to GLODAP file in tmp directory" archive
            if !isfile(archive)
                Downloads.download(GLODAP_url, archive; progress=DownloadProgress())
            end

            open(GzipDecompressorStream, archive) do io
                Tar.extract(io, metadata.dir)
            end
        end
    end
end

# Custom retrieve_data: GLODAP NetCDF files contain Missing values (from _FillValue)
# which must be converted to NaN before the GPU kernel in set_metadata_field!.
function DataWrangling.retrieve_data(metadata::Metadatum{<:GLODAPDataset})

    path = metadata_path(metadata)
    name = GLODAP_file_variable_names[metadata.name]

    ds = Dataset(path)
    raw = ds[name][:, :, :, 1]
    close(ds)

    # Convert Union{Missing, Float32} → Float32 with NaN for missing
    data = Array{Float32}(undef, size(raw))
    for i in eachindex(raw)
        data[i] = ismissing(raw[i]) ? NaN32 : Float32(raw[i])
    end

    if reversed_vertical_axis(metadata.dataset)
        data = reverse(data, dims=3)
    end

    return data
end

end # module
