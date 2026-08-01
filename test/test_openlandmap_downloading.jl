include("runtests_setup.jl")
include("dataset_status.jl")

using ArchGDAL  # activates NumericalEarthArchGDALExt (the windowed /vsicurl COG read)

using NumericalEarth.DataWrangling: BoundingBox, metadata_path
using NumericalEarth.DataWrangling.OpenLandMap: OpenLandMapSoilDB

# `test_openlandmap.jl` covers the dataset interface and the windowing logic against a
# synthetic on-disk tile; it never touches the network. This is the counterpart that proves
# the real /vsicurl reads still resolve.
#
# The global grid is ~1.44M × 528k cells, so a region is mandatory. A ~0.05° box keeps each
# depth window to a few hundred kilobytes.
const openlandmap_region = BoundingBox(longitude = (-112.05, -112.00), latitude = (36.00, 36.05))

@testset "Downloading OpenLandMap soil properties" begin
    for name in (:clay_fraction, :bulk_density)
        @dataset_check "OpenLandMapSoilDB" string(name) begin
            metadatum = Metadatum(name; dataset=OpenLandMapSoilDB(), region=openlandmap_region)
            filepath = metadata_path(metadatum)
            isfile(filepath) && rm(filepath; force=true)

            download(metadatum)
            isfile(filepath) || error("OpenLandMapSoilDB $(name) download produced no file at $(filepath)")
            filepath
        end
    end
end
