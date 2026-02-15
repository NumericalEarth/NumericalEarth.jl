# Common test setup file to make stand-alone tests easy
include("runtests_setup.jl")
include("download_utils.jl")

using CUDA
using Scratch

test_group = get(ENV, "TEST_GROUP", :all)
test_group = Symbol(test_group)

using NumericalEarth.DataWrangling: download_dataset

function delete_inpainted_files(dir)
    @info "Cleaning inpainted files..."
    for (root, _, files) in walkdir(dir)
        for file in files
            if endswith(file, "_inpainted.jld2")
                filepath = joinpath(root, file)
                rm(filepath; force=true)
                @info "    Deleted: $filepath"
            end
        end
    end
end

if test_group == :init || test_group == :all
    #####
    ##### Delete inpainted files
    #####

    delete_inpainted_files(@get_scratch!("."))

    #####
    ##### Download bathymetry data
    #####

    ETOPOmetadata = Metadatum(:bottom_height, dataset=NumericalEarth.ETOPO.ETOPO2022())
    download_dataset_with_fallback(metadata_path(ETOPOmetadata); dataset_name="ETOPO2022") do
        NumericalEarth.DataWrangling.download_dataset(ETOPOmetadata)
    end

    #####
    ##### Download JRA55 data
    #####

    try
        atmosphere = JRA55PrescribedAtmosphere(backend=JRA55NetCDFBackend(2))
    catch e
        @warn "Original JRA55 download failed, trying NumericalEarthArtifacts fallback..." exception=(e, catch_backtrace())
        emit_ci_warning("Broken JRA55 download", "Original source failed during init")
        for name in NumericalEarth.DataWrangling.JRA55.JRA55_variable_names
            datum = Metadatum(name; dataset=JRA55.RepeatYearJRA55())
            download_from_artifacts(metadata_path(datum))
        end
        atmosphere = JRA55PrescribedAtmosphere(backend=JRA55NetCDFBackend(2))
    end

    #####
    ##### Download Dataset data
    #####

    # Download few datasets for tests
    for dataset in test_datasets
        time_resolution = dataset isa ECCO2Daily ? Day(1) : Month(1)
        end_date = start_date + 2 * time_resolution
        dates = start_date:time_resolution:end_date

        temperature_metadata = Metadata(:temperature; dataset, dates)
        salinity_metadata    = Metadata(:salinity; dataset, dates)

        for md in (temperature_metadata, salinity_metadata)
            download_dataset_with_fallback(metadata_path(md); dataset_name="$(typeof(dataset)) $(md.name)") do
                download_dataset(md)
            end
        end

        if dataset isa Union{ECCO2DarwinMonthly, ECCO4DarwinMonthly}
            PO₄_metadata = Metadata(:phosphate; dataset, dates)
            download_dataset_with_fallback(metadata_path(PO₄_metadata); dataset_name="$(typeof(dataset)) phosphate") do
                download_dataset(PO₄_metadata)
            end
        end
    end
end

# Tests JRA55 utilities, plus some DataWrangling utilities
if test_group == :JRA55 || test_group == :all
    include("test_jra55.jl")
end

if test_group == :ecco2_monthly || test_group == :all
    include("test_ecco2_monthly.jl")
end

if test_group == :ecco2_daily || test_group == :all
    include("test_ecco2_daily.jl")
end

if test_group == :ecco4_en4 || test_group == :all
    include("test_ecco4_en4.jl")
end

if test_group == :ecco_atmosphere || test_group == :all
    include("test_ecco_atmosphere.jl")
end

# Tests that we can download JRA55 utilities
if test_group == :downloading || test_group == :all
    include("test_downloading.jl")
end

# Tests that we can download from Copernicus Climate Data Store (ERA5, etc.)
if test_group == :cds_downloading || test_group == :all
    include("test_cds_downloading.jl")
end

if test_group == :fluxes || test_group == :all
    include("test_surface_fluxes.jl")
    include("test_sea_ice_ocean_heat_fluxes.jl")
end

if test_group == :bathymetry || test_group == :all
    include("test_bathymetry.jl")
end

if test_group == :earth_system_model || test_group == :all
    include("test_earth_system_model.jl")
    include("test_diagnostics.jl")
end

if test_group == :distributed || test_group == :all
    include("test_distributed_utils.jl")
end

if test_group == :reactant || test_group == :all
    include("test_reactant.jl")
end

if test_group == :speedy_weather || test_group == :all
    include("test_speedy_coupling.jl")
end

if test_group == :veros || test_group == :all
    include("test_veros.jl")
end
