include("runtests_setup.jl")
include("dataset_status.jl")

using NumericalEarth.OSPapa
using NumericalEarth.DataWrangling: metadata_path

# Ocean Station Papa is a single mooring: the ocean-observation file is one modest NetCDF
# from PMEL's S3 bucket, and the flux record is an ERDDAP query over a date window. Both
# are cheap, which is why the whole date range here is three days.
const OSPAPA_DOWNLOAD_START = DateTime(2012, 10, 1)
const OSPAPA_DOWNLOAD_END   = DateTime(2012, 10, 3)

@testset "Downloading OSPapa ocean observations" begin
    for name in (:temperature, :salinity)
        @dataset_check "OSPapaHourly" string(name) begin
            metadatum = Metadatum(name; dataset=OSPapaHourly(), date=OSPAPA_DOWNLOAD_START)
            filepath = metadata_path(metadatum)
            isfile(filepath) && rm(filepath; force=true)

            download(metadatum)
            isfile(filepath) || error("OSPapaHourly $(name) download produced no file at $(filepath)")
            filepath
        end
    end
end

@testset "Downloading OSPapa flux observations" begin
    # `OSPapaFluxHourly` has no single-metadatum download path: the uniform hourly cache is
    # built from the raw ERDDAP response, so the public constructor is the download.
    @dataset_check "OSPapaFluxHourly" "surface_fluxes" begin
        fluxes = os_papa_prescribed_fluxes(; start_date = OSPAPA_DOWNLOAD_START,
                                             end_date   = OSPAPA_DOWNLOAD_END)

        all(isfinite, fluxes.Qnet) || error("OSPapaFluxHourly returned non-finite net heat flux")
        fluxes
    end
end
