module NumericalEarthCDSAPIExt

using CDSAPI: CDSAPI
using DocStringExtensions: TYPEDSIGNATURES
using Downloads: Downloads

using Dates: Dates
using Oceananigans.DistributedComputations: @root

using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: CDSAPIClient, BoundingBox, metadata_path, retrieve_with_retries
using NumericalEarth.DataWrangling.ERA5: ERA5_TIME_DIMNAMES, group_by_calendar_month, foreach_nc,
                                         split_era5_nc_by_datetime
using NumericalEarth.DataWrangling.GloFAS: GloFASDataset, GloFASMetadata, GloFASMetadatum,
                                           GloFAS_dataset_variable_names, GloFAS_netcdf_variable_names

NumericalEarth.DataWrangling.cds_retrieve(::CDSAPIClient) = CDSAPI.retrieve

const BBOX = BoundingBox

include("glofas.jl")

end # module NumericalEarthCDSAPIExt
