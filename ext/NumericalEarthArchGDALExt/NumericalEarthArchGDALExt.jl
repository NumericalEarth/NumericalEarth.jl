module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using GDAL: OGREnvelope, ogr_l_getextent, vsireaddirrecursive,
            cplsetconfigoption, cplgetconfigoption
using NCDatasets: NCDataset, defDim, defVar
using Downloads: Downloads
using NetworkOptions: NetworkOptions
using NumericalEarth: NumericalEarth

using Oceananigans: Center, CPU
using Oceananigans.Fields: Field, interior
using Oceananigans.Grids: λnodes, φnodes

using NumericalEarth.DataWrangling: BoundingBox, native_grid, native_region_grid,
                                    dataset_variable_name, bounding_box_intersects,
                                    cmr_granules, earthdata_download, earthdata_download_cached,
                                    figshare_article_url, write_atomically
using NumericalEarth.DataWrangling.ASTERGED: asterged_short_name, asterged_version,
                                             asterged_decode_emissivity, asterged_decode_uncertainty,
                                             broadband_map, place_tile!,
                                             OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS
using NumericalEarth.DataWrangling.ETHSentinel2Canopy: ETHSentinel2CanopyHeight,
                                                       ETHSentinel2CanopyHeightMetadatum,
                                                       ETH_LIBDRIVE_TOKEN, eth_tile_urls,
                                                       canopy_regional_raster, mask_eth
using NumericalEarth.DataWrangling.GloBFP3D: GlobalBuildingFootprints3DMetadatum,
                                             GLOBFP3D_FIGSHARE_ARTICLE_IDS,
                                             globfp3d_parse_tile_bounds,
                                             globfp3d_native_cell_size
using NumericalEarth.DataWrangling.GHSL: GHSBuiltS, GHSLMetadatum, native_resolution,
                                         ghsl_tile_url, ghsl_tile_tif_name, ghsl_tiles_in_bbox,
                                         ghsl_regional_raster, built_surface_to_fraction,
                                         mask_building_height
using NumericalEarth.DataWrangling.MODISLand: MODISLand, granule_urls, regional_lattice,
                                              stored_granule_layers
using NumericalEarth.DataWrangling.WorldCover: ESAWorldCoverMetadatum, version_year, version_string,
                                               worldcover_window, aggregate_landcover,
                                               class_fraction_variable_name,
                                               ESA_WORLDCOVER_NATIVE_STEP

include("gdal_utils.jl")
include("ibcao.jl")
include("asterged.jl")
include("globfp3d.jl")
include("ghsl.jl")
include("ethcanopy.jl")
include("modis_land.jl")
include("openlandmap.jl")
include("worldcover.jl")

end # module NumericalEarthArchGDALExt
