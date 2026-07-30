module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using GDAL: OGREnvelope, ogr_l_getextent, vsireaddirrecursive
using NCDatasets: NCDataset, defDim, defVar
using Downloads: Downloads
using NetworkOptions: NetworkOptions
using NumericalEarth: NumericalEarth

using Oceananigans: Center, CPU
using Oceananigans.Grids: λnodes, φnodes

using NumericalEarth.DataWrangling: BoundingBox, native_grid, native_region_grid,
                                    cmr_granules, earthdata_download_cached,
                                    figshare_article_url
using NumericalEarth.DataWrangling.ASTERGED: asterged_short_name, asterged_version,
                                             asterged_decode_emissivity, asterged_decode_uncertainty,
                                             broadband_map, place_tile!,
                                             OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS
using NumericalEarth.DataWrangling.GloBFP3D: GlobalBuildingFootprints3DMetadatum,
                                             GLOBFP3D_FIGSHARE_ARTICLE_IDS,
                                             globfp3d_parse_tile_bounds, globfp3d_tile_intersects,
                                             globfp3d_native_cell_size

include("gdal_utils.jl")
include("ibcao.jl")
include("asterged.jl")
include("globfp3d.jl")
include("openlandmap.jl")

end # module NumericalEarthArchGDALExt
