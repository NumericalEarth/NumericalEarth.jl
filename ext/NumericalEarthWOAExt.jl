module NumericalEarthWOAExt

using Downloads: Downloads
using Oceananigans.DistributedComputations: @root
using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: Metadata, metadata_path
using NumericalEarth.DataWrangling.WOA: WOAClimatology, WOA_variable_names, woa_period
using WorldOceanAtlasTools: WorldOceanAtlasTools

woa_filepath(woa_tracer, product_year, period) =
    WorldOceanAtlasTools.WOAfile(woa_tracer; product_year, period, resolution=1)

function Downloads.download(metadata::Metadata{<:WOAClimatology}; skip_existing=true)
    @root for metadatum in metadata
        linkpath = metadata_path(metadatum)

        if isfile(linkpath) && skip_existing
            continue
        end

        woa_tracer = WOA_variable_names[metadatum.name]
        period = woa_period(metadatum.dataset, metadatum.dates)
        product_year = metadatum.dataset.product_year

        # Trigger DataDeps download and get the path to the original WOA file
        source = woa_filepath(woa_tracer, product_year, period)

        rm(linkpath; force=true)
        cp(source, linkpath)
    end

    return metadata_path(metadata)
end

end # module
