module NumericalEarthWOAExt

using WorldOceanAtlasTools

using Oceananigans.DistributedComputations: @root
using NumericalEarth.DataWrangling: Metadata, Metadatum
using NumericalEarth.WOA: WOAClimatology, WOA_variable_names, woa_period

import NumericalEarth.DataWrangling: download_dataset, metadata_path

# NOAA servers have inconsistent availability across product years.
# We try the user-specified product_year first, then fall back to others.
const fallback_product_years = (2023, 2018, 2013)

function woa_filepath(woa_tracer, product_year, period)
    # Try user-specified product year first
    try
        return WorldOceanAtlasTools.WOAfile(woa_tracer; product_year, period, resolution=1)
    catch e
        @warn "Errored with exception $(e)"
    end

    # Fall back to other product years
    for py in fallback_product_years
        py == product_year && continue
        try
            @info "WOA product year $product_year unavailable for tracer \"$woa_tracer\", trying $py..."
            return WorldOceanAtlasTools.WOAfile(woa_tracer; product_year=py, period, resolution=1)
        catch e
            @warn "Errored with exception $(e)"
        end
    end

    error("Could not download WOA data for tracer \"$woa_tracer\" " *
          "(tried product years: $product_year, $(join(fallback_product_years, ", ")))")
end

function download_dataset(metadata::Metadata{<:WOAClimatology}; skip_existing=true)
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

    return nothing
end

end # module
