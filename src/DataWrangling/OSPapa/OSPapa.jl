module OSPapa

export OSPapaPrescribedAtmosphere
export OSPapaPrescribedRadiation
export os_papa_prescribed_fluxes
export os_papa_prescribed_flux_boundary_conditions
export OSPapaHourly
export OSPapaFluxHourly

using Dates: Dates, DateTime, Hour
using Downloads: Downloads
using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: Bounded, Center, Flat, RectilinearGrid
using Oceananigans.Fields: Field, interior
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, FluxBoundaryCondition
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Units: Units
using NCDatasets: NCDatasets, NCDataset, defDim, defVar
using Scratch: Scratch, @get_scratch!
using Thermodynamics: q_vap_from_RH, Liquid

using NumericalEarth: NumericalEarth
using NumericalEarth.Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux, AtmosphereThermodynamicsParameters
using NumericalEarth.DataWrangling: download_progress,
                                    download_dataset,
                                    Metadata,
                                    Metadatum,
                                    metadata_path,
                                    first_date,
                                    last_date,
                                    centers_to_interfaces,
                                    fill_gaps!,
                                    CentimetersPerSecond,
                                    Celsius,
                                    Millibar,
                                    MillimetersPerHour

const OSPAPA_S3_URL  = "https://noaa-oar-keo-papa-pds.s3.amazonaws.com/PAPA/"
const OSPAPA_FILENAME = "OS_PAPA_200706_M_TSVMBP_50N145W_hr.nc"
const OSPAPA_LONGITUDE = -144.9
const OSPAPA_LATITUDE  = 50.1

download_OSPapa_cache::String = ""

function __init__()
    global download_OSPapa_cache = @get_scratch!("OSPapa")
end

function download_ospapa_file(dir=download_OSPapa_cache)
    filepath = joinpath(dir, OSPAPA_FILENAME)
    if !isfile(filepath)
        url = OSPAPA_S3_URL * OSPAPA_FILENAME
        @info "Downloading Ocean Station Papa data from AWS S3..."
        Downloads.download(url, filepath; progress=download_progress)
    end
    return filepath
end

include("OSPapa_ocean_observations.jl")
include("OSPapa_flux_observations.jl")
include("OSPapa_prescribed_atmosphere.jl")
include("OSPapa_prescribed_radiation.jl")
include("OSPapa_prescribed_fluxes.jl")

end # module
