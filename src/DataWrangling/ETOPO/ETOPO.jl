module ETOPO

export ETOPO2022

using Downloads: Downloads
using Oceananigans.DistributedComputations: @root
using Scratch: Scratch, @get_scratch!

using ..DataWrangling: DataWrangling, DownloadProgress, AbstractStaticBathymetry, Metadatum,
                       metadata_path, metadata_url

download_ETOPO_cache::String = ""
function __init__()
    global download_ETOPO_cache = @get_scratch!("ETOPO")
end

ETOPO_bathymetry_variable_names = Dict(:bottom_height => "z",)

struct ETOPO2022 <: AbstractStaticBathymetry end

DataWrangling.default_download_directory(::ETOPO2022) = download_ETOPO_cache
DataWrangling.reversed_vertical_axis(::ETOPO2022) = true
DataWrangling.longitude_interfaces(::ETOPO2022) = (-180, 180)
DataWrangling.latitude_interfaces(::ETOPO2022) = (-90, 90)
Base.size(::ETOPO2022) = (21600, 10800, 1)

const ETOPOMetadatum = Metadatum{<:ETOPO2022}

DataWrangling.dataset_variable_name(data::ETOPOMetadatum) = ETOPO_bathymetry_variable_names[data.name]

const ETOPO_url = "https://www.dropbox.com/scl/fi/6pwalcuuzgtpanysn4h6f/" *
    "ETOPO_2022_v1_60s_N90W180_surface.nc?rlkey=2t7890ruyk4nd5t5eov5768lt&st=yfxsy1lu&dl=0"

DataWrangling.metadata_url(::ETOPOMetadatum) = ETOPO_url
DataWrangling.metadata_filename(::ETOPO2022, name, date, region) = "ETOPO_2022_v1_60s_N90W180_surface.nc"

function Downloads.download(metadatum::ETOPOMetadatum)
    fileurl  = metadata_url(metadatum)
    filepath = metadata_path(metadatum)

    @root if !isfile(filepath)
        @info "Downloading ETOPO data: $(metadatum.name) in $(metadatum.dir)..."
        Downloads.download(fileurl, filepath; progress=DownloadProgress())
    end
    return filepath
end

end # module
