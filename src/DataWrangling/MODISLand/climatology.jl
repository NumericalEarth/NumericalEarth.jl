#####
##### Seasonal climatology
#####

"""
    reduce_retained(reducer, samples)

Apply `reducer` to the finite entries of `samples`, returning `(value, count)`, or
`(NaN32, 0)` when none is finite.
"""
function reduce_retained(reducer, samples)
    retained = filter(isfinite, samples)
    isempty(retained) && return NaN32, 0
    return Float32(reducer(retained)), length(retained)
end

"""
    build_lai_climatology!(dataset::MODISLAIClimatology;
                           name = :leaf_area_index,
                           region,
                           periods = 1:periods_per_year(dataset),
                           dir = default_download_directory(dataset))

Build the cached per-period files behind a [`MODISLAIClimatology`](@ref) over `region`. For
each composite period in `periods`, every contributing date of `dataset.years` is
downloaded and screened, the retained retrievals are combined pixel by pixel with
`dataset.reducer`, and one file is written holding the reduction and the retained-retrieval
count. Periods already on disk are skipped, so an interrupted build resumes. Returns the
paths of the files.

A date the archive holds no composite for is skipped with a warning.
"""
function build_lai_climatology!(dataset::MODISLAIClimatology;
                                name = :leaf_area_index,
                                region,
                                periods = 1:periods_per_year(dataset),
                                dir = default_download_directory(dataset))

    haskey(MODISLAI_variable_names, name) ||
        throw(ArgumentError("$name cannot be composited; the variables that can are $(keys(MODISLAI_variable_names))."))

    source = source_dataset(dataset)
    period_days = composite_period_days(dataset)
    source_dates = DataWrangling.all_dates(source, name)
    paths = String[]

    for period in periods
        stamp = climatology_year_start() + Day((period - 1) * period_days)
        filepath = joinpath(dir, DataWrangling.metadata_filename(dataset, name, stamp, region))
        push!(paths, filepath)
        isfile(filepath) && continue

        dates = [date for date in source_dates
                 if Dates.year(date) in dataset.years && period_index(date, period_days) == period]
        isempty(dates) &&
            error("No $(modis_short_name(source)) composites fall in period $period of " *
                  "the years $(dataset.years).")

        available = materialize_composites(name, source, dates, region, dir)
        isempty(available) &&
            error("The archive holds no $(modis_short_name(source)) composite of period " *
                  "$period in any of the years $(dataset.years).")

        metadata = Metadata(name; dataset = source, dates = available, region, dir)

        @info string("Compositing ", length(available), " retrievals into period ", period,
                     " of the ", modis_short_name(source), " ", name, " climatology...")
        write_lai_composite(filepath, metadata, dataset.reducer)
    end

    return paths
end

# The dates contributing to one period, minus the ones the archive has no composite for.
function materialize_composites(name, source, dates, region, dir)
    available = eltype(dates)[]

    for date in dates
        metadatum = Metadatum(name; dataset = source, region, date, dir)
        try
            Downloads.download(metadatum)
            push!(available, date)
        catch err
            err isa MissingGranulesError || rethrow(err)
            @warn string("The ", modis_short_name(source), " record has no composite on ",
                         date, "; compositing this period without it.")
        end
    end

    return available
end

function write_lai_composite(filepath, metadata, reducer)
    variable = DataWrangling.dataset_variable_name(first(metadata))
    λ, φ = DataWrangling.read_file_coords(first(metadata))
    samples = stack(DataWrangling.retrieve_data(metadatum) for metadatum in metadata)

    Nx, Ny = size(samples, 1), size(samples, 2)
    composite = Array{Float32}(undef, Nx, Ny)
    retained_count = Array{Int32}(undef, Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        composite[i, j], retained_count[i, j] = reduce_retained(reducer, view(samples, i, j, :))
    end

    retained_fraction = sum(retained_count) / (Nx * Ny * size(samples, 3))
    @info string(" ... retained ", round(100 * retained_fraction, digits = 1),
                 "% of the retrievals; ", round(100 * mean(isnan.(composite)), digits = 1),
                 "% of the cells have none")

    write_atomically(filepath) do staging_path
        NCDataset(staging_path, "c") do ds
            defDim(ds, "lon", Nx)
            defDim(ds, "lat", Ny)
            defVar(ds, "lon", collect(λ), ("lon",);
                   attrib = ["units" => "degrees_east", "long_name" => "longitude"])
            defVar(ds, "lat", collect(φ), ("lat",);
                   attrib = ["units" => "degrees_north", "long_name" => "latitude"])
            defVar(ds, variable, composite, ("lon", "lat"); deflatelevel = 2, shuffle = true)
            defVar(ds, retained_count_variable, retained_count, ("lon", "lat");
                   attrib = ["long_name" => "number of retained retrievals"],
                   deflatelevel = 2, shuffle = true)
        end
    end
    return nothing
end

function Downloads.download(metadata::MODISLAIClimatologyMetadata)
    @root for metadatum in metadata
        isfile(metadata_path(metadatum)) && continue
        period = period_index(metadatum.dates, composite_period_days(metadatum.dataset))
        build_lai_climatology!(metadatum.dataset; name = climatology_build_name(metadatum),
                               periods = period:period, region = metadatum.region,
                               dir = metadatum.dir)
    end
    return metadata_path(metadata)
end

# The variable a count's file composites, recovered from the filename the two share.
function climatology_build_name(metadatum::MODISLAIClimatologyMetadatum)
    metadatum.name === :retained_retrieval_count || return metadatum.name

    for name in keys(MODISLAI_variable_names)
        DataWrangling.metadata_filename(metadatum.dataset, name, metadatum.dates,
                                        metadatum.region) == metadatum.filename && return name
    end

    throw(ArgumentError("A retained-retrieval count is read with `retained_retrieval_metadatum` of the composited variable's metadatum."))
end
