include("runtests_setup.jl")
include("dataset_status.jl")

using CDSAPI  # activates NumericalEarthCDSAPIExt (the EWDS request path)

using NumericalEarth.DataWrangling: BoundingBox, metadata_path
using NumericalEarth.DataWrangling.GloFAS: GloFASReanalysis

# GloFAS v4 is global at 0.05°; a bounding box is sent as the CDS `area` key so the server
# subsets before delivery. This window covers the lower Amazon, where discharge is large
# enough that a all-missing result means a broken read rather than a dry basin.
const glofas_region = BoundingBox(longitude = (-56, -54), latitude = (-3, -1))
const glofas_date = DateTime(2020, 6, 1)

@testset "Downloading GloFAS river discharge" begin
    @dataset_check "GloFASReanalysis" "river_discharge" begin
        metadatum = Metadatum(:river_discharge; dataset=GloFASReanalysis(),
                              region=glofas_region, date=glofas_date)
        filepath = metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)

        download(metadatum)
        isfile(filepath) || error("GloFASReanalysis download produced no file at $(filepath)")
        filepath
    end
end
