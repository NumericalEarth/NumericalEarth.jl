module NumericalEarthCDSAPIExt

using CDSAPI: CDSAPI
using Downloads: Downloads

using Dates: Dates
using Oceananigans: Oceananigans
using Oceananigans.DistributedComputations: @root

using NCDatasets: NCDatasets, name, path

using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: Metadatum, MetadataSet, default_download_directory, metadata_path
using NumericalEarth.DataWrangling.ERA5: ERA5Dataset, ERA5Metadata, ERA5Metadatum,
                                         ERA5_dataset_variable_names, ERA5_netcdf_variable_names,
                                         ERA5PressureLevelsDataset,
                                         ERA5PressureMetadata, ERA5PressureMetadatum,
                                         ERA5PL_dataset_variable_names, ERA5PL_netcdf_variable_names
using NumericalEarth.DataWrangling.GloFAS: GloFASDataset, GloFASMetadata, GloFASMetadatum,
                                           GloFAS_dataset_variable_names, GloFAS_netcdf_variable_names
using NumericalEarth.DataWrangling.CopernicusLandAlbedo: CopernicusAlbedo,
                                                         albedo_cds_request_variables,
                                                         albedo_source_variable_candidates,
                                                         copernicus_albedo_variables,
                                                         albedo_satellite, find_albedo_variable,
                                                         repack_albedo_pair

include("cds_utils.jl")
include("era5.jl")
include("glofas.jl")
include("copernicus_land_albedo.jl")

end # module NumericalEarthCDSAPIExt
