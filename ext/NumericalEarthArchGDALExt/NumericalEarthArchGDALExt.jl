module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using GDAL: OGREnvelope, ogr_l_getextent, vsireaddirrecursive
using NCDatasets: NCDataset, defDim, defVar
using Downloads: Downloads
using NetworkOptions: NetworkOptions
using NumericalEarth: NumericalEarth

using Oceananigans: Center, CPU
using Oceananigans.Grids: λnodes, φnodes

using NumericalEarth.DataWrangling: BoundingBox, native_grid,
                                    cmr_granules, earthdata_download_cached
using NumericalEarth.DataWrangling.ASTERGED: asterged_short_name, asterged_version,
                                             asterged_decode_emissivity, asterged_decode_uncertainty,
                                             broadband_map, place_tile!,
                                             OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS
using NumericalEarth.DataWrangling.GloBFP3D: GlobalBuildingFootprints3DMetadatum,
                                             FIGSHARE_ARTICLE_IDS, figshare_article_url,
                                             parse_tile_bounds, tile_intersects,
                                             native_cell_size, native_region_grid

include("gdal_utils.jl")
include("ibcao.jl")
include("asterged.jl")
include("globfp3d.jl")
include("openlandmap.jl")

end # module NumericalEarthArchGDALExt
