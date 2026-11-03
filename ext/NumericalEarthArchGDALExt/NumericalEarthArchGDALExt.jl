module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using GDAL: OGREnvelope, ogr_l_getextent, vsireaddirrecursive
using NCDatasets: NCDataset, defDim, defVar
using Downloads: Downloads
using NetworkOptions: NetworkOptions
using NumericalEarth: NumericalEarth, stateindex

using Oceananigans: Bounded, Center, CPU, Field, Flat, LatitudeLongitudeGrid, interior
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: dimension_faces
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.Grids: λnodes, φnodes, longitude_in_same_window
using Oceananigans.Utils: launch!

using NumericalEarth.DataWrangling: BoundingBox, native_grid, native_region_grid,
                                    available_variables, bounding_box_intersects,
                                    cmr_granules, earthdata_download_cached,
                                    figshare_article_url, tile_indices, default_tile_bytes
using NumericalEarth.DataWrangling.ASTERGED: asterged_short_name, asterged_version,
                                             asterged_decode_emissivity, asterged_decode_uncertainty,
                                             broadband_map, place_tile!,
                                             OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS
using NumericalEarth.DataWrangling.GloBFP3D: GlobalBuildingFootprints3DMetadatum,
                                             GLOBFP3D_FIGSHARE_ARTICLE_IDS,
                                             globfp3d_parse_tile_bounds,
                                             globfp3d_native_cell_size
using NumericalEarth.DataWrangling.GHSL: GHSBuilt, GHSBuiltH, GHSBuiltS, GHSLMetadatum, native_resolution,
                                         ghsl_tile_url, ghsl_tile_tif_name, ghsl_tiles_in_bbox,
                                         ghsl_regional_raster, built_surface_to_fraction,
                                         mask_building_height, ghsl_pixel_window, download_GHSL_cache,
                                         longitude_latitude_to_mollweide, MOLLWEIDE_RADIUS
using NumericalEarth.DataWrangling.MODISLand: MODISLand, granule_urls, regional_lattice,
                                              stored_granule_layers, write_lattice_file
using NumericalEarth.DataWrangling.OpenLandMap: assemble_raster_window, raster_window_indices,
                                                validate_wgs84_longitude_latitude, validate_geographic_northup
using NumericalEarth.DataWrangling.WorldCover: ESAWorldCoverMetadatum, version_year, version_string,
                                               worldcover_window, aggregate_landcover,
                                               class_fraction_variable_name,
                                               ESA_WORLDCOVER_CLASS_NAMES,
                                               ESA_WORLDCOVER_NATIVE_STEP,
                                               ESA_WORLDCOVER_PIXELS_PER_DEGREE

using KernelAbstractions: @kernel, @index

include("gdal_utils.jl")
include("ibcao.jl")
include("asterged.jl")
include("globfp3d.jl")
include("ghsl.jl")
include("modis_land.jl")
include("openlandmap.jl")
include("worldcover.jl")

end # module NumericalEarthArchGDALExt
