module NumericalEarthCDSAPIExt

using CDSAPI: CDSAPI
using Downloads: Downloads

using Dates: Dates
using Oceananigans.DistributedComputations: @root

using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: Metadatum, MetadataSet, default_download_directory, metadata_path
using NumericalEarth.DataWrangling.ERA5: ERA5Dataset, ERA5Metadata, ERA5Metadatum,
                                         ERA5_dataset_variable_names,
                                         ERA5PressureLevelsDataset,
                                         ERA5PressureMetadatum,
                                         ERA5PL_dataset_variable_names,
                                         ERA5_TIME_DIMNAMES,
                                         coord_vars, nc_varnames,
                                         group_by_calendar_month,
                                         batch_datetimes_for_cds, foreach_nc,
                                         split_era5_nc, split_era5_nc_by_datetime
using NumericalEarth.DataWrangling.GloFAS: GloFASDataset, GloFASMetadata, GloFASMetadatum,
                                           GloFAS_netcdf_variable_names
using NumericalEarth.DataWrangling.CopernicusLandAlbedo: ALBEDO_CDS_PRODUCT,
                                                         CopernicusAlbedoDatasetMetadata,
                                                         download_ten_day_albedo!

include("cds_utils.jl")
include("era5.jl")
include("glofas.jl")
include("copernicus_land_albedo.jl")

end # module NumericalEarthCDSAPIExt
