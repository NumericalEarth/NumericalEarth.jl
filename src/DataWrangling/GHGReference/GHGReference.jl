module GHGReference

export NOAAMarineBoundaryLayer

using Dates: Dates, DateTime, Millisecond
using DocStringExtensions: TYPEDSIGNATURES
using Downloads: Downloads
using Oceananigans.DistributedComputations: @root
using Oceananigans.Grids: Bounded, Center, Face, Flat, LatitudeLongitudeGrid

using ..DataWrangling: DataWrangling, Metadata, Metadatum, metadata_path, download_with_retries

download_GHGReference_cache::String = ""
function __init__()
    global download_GHGReference_cache = DataWrangling.download_cache("GHGReference")
    return nothing
end

"""
    NOAAMarineBoundaryLayer()

The NOAA Greenhouse Gas Marine Boundary Layer Reference
[Lan et al. (2024)](@cite lan2024marine): zonal-mean dry-air mole fractions of
carbon dioxide (ppm), methane (ppb), nitrous oxide (ppb), and sulfur hexafluoride (ppt)
over the marine boundary layer, and their uncertainties, 48 times per year on 41 latitudes
equally spaced in sine of latitude from the South Pole to the North Pole.
The values sit on the latitude faces of the native grid, so both poles are included.
"""
struct NOAAMarineBoundaryLayer end

const GHGReferenceMetadata = Metadata{<:NOAAMarineBoundaryLayer}
const GHGReferenceMetadatum = Metadatum{<:NOAAMarineBoundaryLayer}

# Each variable is a gas and the column offset within each (value, uncertainty) pair.
const GHGReference_variables = Dict(
    :carbon_dioxide                  => ("CO2", 0),
    :carbon_dioxide_uncertainty      => ("CO2", 1),
    :methane                         => ("CH4", 0),
    :methane_uncertainty             => ("CH4", 1),
    :nitrous_oxide                   => ("N2O", 0),
    :nitrous_oxide_uncertainty       => ("N2O", 1),
    :sulfur_hexafluoride             => ("SF6", 0),
    :sulfur_hexafluoride_uncertainty => ("SF6", 1),
)

# Decimal years of the first and last time steps.
const GHGReference_first_year = Dict("CO2" => 1979, "CH4" => 1983.5, "N2O" => 2001, "SF6" => 1997.5)
const GHGReference_last_year = 2026

const GHGReference_url = "https://gml.noaa.gov/ccgg/mbl/"

DataWrangling.default_download_directory(::NOAAMarineBoundaryLayer) = download_GHGReference_cache
DataWrangling.available_variables(::NOAAMarineBoundaryLayer) = GHGReference_variables
DataWrangling.dataset_location(::NOAAMarineBoundaryLayer, name) = (Center, Face, Nothing)
DataWrangling.is_three_dimensional(::GHGReferenceMetadata) = false
DataWrangling.default_inpainting(::GHGReferenceMetadata) = nothing
DataWrangling.latitude_interfaces(::NOAAMarineBoundaryLayer) = asind.(-1:0.05:1)
Base.size(::NOAAMarineBoundaryLayer, variable) = (1, 41, 1)

DataWrangling.metadata_filename(::NOAAMarineBoundaryLayer, name, date, region) =
    string("mbl_", lowercase(first(GHGReference_variables[name])), "_surface.txt")

function DataWrangling.all_dates(::NOAAMarineBoundaryLayer, name)
    gas, _ = GHGReference_variables[name]
    first_year = GHGReference_first_year[gas]
    Nt = round(Int, 48 * (GHGReference_last_year - first_year)) + 1
    return [decimal_year_to_date(first_year + (n - 1) / 48) for n in 1:Nt]
end

function decimal_year_to_date(decimal_year)
    year = floor(Int, decimal_year)
    year_length = Dates.value(DateTime(year + 1) - DateTime(year))
    return DateTime(year) + Millisecond(round(Int, (decimal_year - year) * year_length))
end

# The values vary only with latitude, so longitude is `Flat`.
function DataWrangling.construct_native_grid(metadata::GHGReferenceMetadata, ::Nothing, arch; halo)
    latitude = DataWrangling.latitude_interfaces(metadata)
    return LatitudeLongitudeGrid(arch, eltype(metadata); size = length(latitude) - 1, latitude,
                                 halo = halo[2], topology = (Flat, Bounded, Flat))
end

DataWrangling.read_file_coords(metadatum::GHGReferenceMetadatum) =
    nothing, DataWrangling.latitude_interfaces(metadatum)

"""
$(TYPEDSIGNATURES)

Read the values of `metadatum.name` at `metadatum.dates` from south to north.
"""
function DataWrangling.retrieve_data(metadatum::GHGReferenceMetadatum)
    _, column = GHGReference_variables[metadatum.name]
    n = findfirst(==(metadatum.dates), DataWrangling.all_dates(metadatum.dataset, metadatum.name))
    rows = [line for line in eachline(metadata_path(metadatum)) if !startswith(line, '#')]
    row = parse.(Float64, split(rows[n]))
    return reshape(row[2+column:2:end], 1, :, 1)
end

function Downloads.download(metadata::GHGReferenceMetadata)
    path = metadata_path(first(metadata))

    @root if !isfile(path)
        gas, _ = GHGReference_variables[metadata.name]
        query = string("ghg.php?param=", gas, "&reference_type=surface&reference=global",
                       "&startyear=1979&startmonth=1&endyear=", GHGReference_last_year - 1, "&endmonth=12")
        response = String(take!(Downloads.download(GHGReference_url * query, IOBuffer())))
        link = match(r"tmp/[^']+_surface\.txt", response)
        isnothing(link) && error("NOAA returned no marine boundary layer reference file for $gas.")
        @info "Downloading NOAA marine boundary layer reference for $gas in $(metadata.dir)..."
        download_with_retries(GHGReference_url * link.match, path)
    end

    return path
end

end # module
