"""
Incorporate various datasets to be used for bathymetry, initialization, forcing,
restoring, or validation.
"""
module DataWrangling

export Metadata, Metadatum, MetadataSet, DatewiseFilename, ECCOMetadatum, EN4Metadatum, all_dates, first_date, last_date
export validate_dataset_coverage, metadata_filename
export BoundingBox, Column, Linear, Nearest
export WOAClimatology, WOAAnnual, WOAMonthly
export metadata_time_step, metadata_epoch
export supported_datasets
export LinearlyTaperedPolarMask
export DatasetRestoring, SurfaceFluxRestoring
export ERA5HourlySingleLevel, ERA5MonthlySingleLevel, ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels
export native_grid

using Adapt: Adapt
using Downloads: Downloads
using LibCURL: LibCURL
using JLD2: JLD2, jldopen
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans, pretty_filesize, location
using Oceananigans.Architectures: AbstractArchitecture, CPU, architecture,
                                  on_architecture, child_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!, FieldBoundaryConditions
using Oceananigans.DistributedComputations: DistributedComputations, @root
using Oceananigans.Grids: AbstractGrid, Center, Flat, Bounded,
                          LatitudeLongitudeGrid, RectilinearGrid
using Oceananigans.Fields: Fields, Field, interpolate, interpolate!, interior, set!
using Oceananigans.Grids: node
using Oceananigans.OutputReaders: OnDisk, AbstractInMemoryBackend, Cyclical,
                                  FieldTimeSeries, FlavorOfFTS, time_indices
using Oceananigans.Utils: launch!, prettytime, prettysummary
using NCDatasets: NCDatasets, Dataset
using Printf: Printf, @sprintf
using Scratch: @get_scratch!

using ..NumericalEarth: NumericalEarth, stateindex

#####
##### Downloading utilities
#####

"""
    download_cache(key)

Return the directory used to cache `key` data downloaded by NumericalEarth.

By default this is a [Scratch.jl](https://github.com/JuliaPackaging/Scratch.jl) space
managed by Julia under the active depot. If the environment variable
`NUMERICALEARTH_DATA_DIRECTORY` is set, data is instead cached under
`joinpath(ENV["NUMERICALEARTH_DATA_DIRECTORY"], key)`. This is useful on systems where the
Julia depot lives on a small or quota-limited filesystem (e.g. `\$HOME` on HPC clusters),
or to share a single cache of large datasets across depots and users.

The variable is read when NumericalEarth is loaded, so it must be set *before*
`using NumericalEarth`.
"""
function download_cache(key)
    if haskey(ENV, "NUMERICALEARTH_DATA_DIRECTORY")
        return mkpath(joinpath(ENV["NUMERICALEARTH_DATA_DIRECTORY"], key))
    else
        return @get_scratch!(key)
    end
end

mutable struct DownloadProgress <: Function
    next_fraction :: Float64
    download_start_time :: UInt64
end

DownloadProgress() = DownloadProgress(0.0, time_ns())

"""
    DownloadProgress(total, now; filename="")
"""
function (d::DownloadProgress)(total, now; filename="")
    messages = 10

    if total > 0
        fraction = now / total

        if fraction < 1 / messages && d.next_fraction == 0
            @info @sprintf("Downloading %s (size: %s)...", filename, pretty_filesize(total))
            d.next_fraction = 1 / messages
            d.download_start_time = time_ns()
        end

        if fraction > d.next_fraction
            elapsed = 1e-9 * (time_ns() - d.download_start_time)
            msg = @sprintf(" ... downloaded %s (%d%% complete, %s)", pretty_filesize(now),
                           100fraction, prettytime(elapsed))
            @info msg
            d.next_fraction = d.next_fraction + 1 / messages
        end
    else
        if now > 0 && d.next_fraction == 0
            @info "Downloading $filename..."
            d.next_fraction = 1 / messages
            d.download_start_time = time_ns()
        end
    end

    return nothing
end

"""
    netrc_downloader(username, password, machine, dir)

Create a downloader that uses a netrc file to authenticate with the given machine.
This downloader writes the username and password in a file named `auth.netrc` (for Unix) and
`auth_netrc` (for Windows), located in the directory `dir`.
To avoid leaving the password on disk after the downloader has been used,
it is recommended to initialize the downloader in a temporary directory, which will be removed
after the download is complete.

For example:

```
mktempdir(dir) do tmp
    dowloader = netrc_downloader(username, password, machine, tmp)
    Downloads.download(fileurl, filepath; downloader)
end
```
"""
function netrc_downloader(username, password, machine, dir)
    netrc_file = netrc_permission_file(username, password, machine, dir)
    downloader = Downloads.Downloader()
    easy_hook  = (easy, _) -> begin
        Downloads.Curl.setopt(easy, LibCURL.CURLOPT_NETRC_FILE, netrc_file)
        # Bypass certificate verification because ecco.jpl.nasa.gov is using an untrusted CA certificate
        Downloads.Curl.setopt(easy, LibCURL.CURLOPT_SSL_VERIFYPEER, false)
    end
    downloader.easy_hook = easy_hook
    return downloader
end

# Code snippet adapted from https://github.com/evetion/SpaceLiDAR.jl/blob/master/src/utils.jl#L150
function netrc_permission_file(username, password, machine, dir)
    if Sys.iswindows()
        filepath = joinpath(dir, "auth_netrc")
    else
        filepath = joinpath(dir, "auth.netrc")
    end

    open(filepath, "a") do f
        write(f, "machine $machine login $username password $password\n")
    end

    return filepath
end

#####
##### FieldTimeSeries utilities
#####

function save_field_time_series!(fts; path, name, overwrite_existing=false)
    overwrite_existing && rm(path; force=true)

    times = on_architecture(CPU(), fts.times)
    grid  = on_architecture(CPU(), fts.grid)

    LX, LY, LZ = location(fts)
    ondisk_fts = FieldTimeSeries{LX, LY, LZ}(grid, times;
                                             backend = OnDisk(), path, name)

    Nt = length(times)
    for n = 1:Nt
        fill_halo_regions!(fts[n])
        set!(ondisk_fts, fts[n], n)
    end

    return nothing
end

"""
    download(metadata; url = urls(metadata))

Download the dataset specified by the `metadata::ECCOMetadata`. If `metadata.dates` is a single date,
the dataset is downloaded directly. If `metadata.dates` is a vector of dates, each date
is downloaded individually.

Note: if called by multiple processes via MPI, `download` should only run on the root process.

Arguments
=========
- `metadata`: The metadata specifying the dataset to be downloaded. Available options are metadata for
              ETOPO, ECCO4, ECCO2, EN4, and JRA55 datasets.

!!! info "Credential setup requirements for ECCO datasets"

    For ECCO datasets, the data download requires "WebDAV/Programmatic API" credentials from
    NASA's Earthdrive. The WebDAV/Programmatic API username and password need to be provided in
    the `ECCO_USERNAME` and `ECCO_WEBDAV_PASSWORD` environment variables respectively. This can be
    done by exporting the environment variables in the shell before running the script, or by
    launching julia with

    ```
    ECCO_USERNAME=myusername ECCO_WEBDAV_PASSWORD=mypassword julia
    ```

    or by invoking

    ```julia
    julia> ENV["ECCO_USERNAME"] = "myusername"

    julia> ENV["ECCO_WEBDAV_PASSWORD"] = "mypassword"
    ```

    within julia. More detailed instructions for obtaining WebDAV credentials are at:

        https://github.com/CliMA/NumericalEarth.jl/blob/main/src/DataWrangling/ECCO/README.md
"""
# `download(::Metadata)` extends `Downloads.download` (the modern stdlib function,
# not `Base.download` which is a 1.0-era shim). Per-dataset methods are added
# within each dataset module via `Downloads.download(metadata::FooMetadata) = ...`.

function inpainted_metadata_path end

"""
    z_interfaces(dataset)

Return an array with the vertical interfaces (``z``-faces) of the dataset
that `metadata` corresponds to.
"""
function z_interfaces end
function longitude_interfaces end
function latitude_interfaces end
function reversed_vertical_axis end
reversed_latitude_axis(dataset) = false
function native_grid end
function binary_data_grid end
function binary_data_size end

default_mask_value(dataset) = NaN

"""
    AbstractStaticDataset

Supertype for datasets without a time dimension. Provides default no-op implementations for the date-related interface
(`all_dates`, `first_date`, `last_date`).
"""
abstract type AbstractStaticDataset end

all_dates(::AbstractStaticDataset,  args...) = nothing
first_date(::AbstractStaticDataset, args...) = nothing
last_date(::AbstractStaticDataset,  args...) = nothing

"""
    AbstractStaticBathymetry <: AbstractStaticDataset

Supertype for static, two-dimensional bathymetry datasets (e.g. ETOPO, GEBCO, IBCSO, IBCAO).
Adds defaults for the degenerate vertical axis and a variable-agnostic `Base.size`.
"""
abstract type AbstractStaticBathymetry <: AbstractStaticDataset end

z_interfaces(::AbstractStaticBathymetry) = (0, 1)
Base.size(dataset::AbstractStaticBathymetry, variable) = size(dataset)

# Fundamentals
include("metadata.jl")
include("set_region_data.jl")
include("metadata_field.jl")
include("dataset_backend.jl")
include("metadata_field_time_series.jl")
include("inpainting.jl")
include("restoring.jl")

function metadata_time_step end
function metadata_epoch end

"""
    variable_glossary :: Dict{Symbol, Symbol}

Global map from *verbose* dataset variable names (the symbols a user passes to
`Metadata` and `MetadataSet`) to the *short* model field-name symbols
established in `docs/src/appendix/notation.md`. Used by `set!(model, mset)` to
auto-route `mset.eastward_velocity` → `u = ...` etc.

Verbose names absent from this map are silently ignored by `set!(model, mset)`
(they remain fetchable via `download(mset)` and accessible via `mset.<name>`).
Synonyms across dataset modules (e.g. `:u_velocity`, `:eastward_velocity`,
`:eastward_wind` all → `:u`) are intentional: they serve as domain
disambiguators when a single dataset (e.g. `ECCO4Monthly`) carries both ocean
and atmosphere fields.

Every value here is documented in `docs/src/appendix/notation.md` (or in
[Breeze.jl's notation](https://numericalearth.github.io/BreezeDocumentation/stable/appendix/notation/)
for the microphysics symbols).
"""
const variable_glossary = Dict{Symbol, Symbol}(
    # Ocean & atmosphere state (notation.md existing rows)
    :temperature                          => :T,
    :air_temperature                      => :T,
    :salinity                             => :S,
    :u_velocity                           => :u,
    :v_velocity                           => :v,
    :eastward_velocity                    => :u,
    :northward_velocity                   => :v,
    :eastward_wind                        => :u,
    :northward_wind                       => :v,
    :sea_level_pressure                   => :p,
    # Atmosphere moisture / microphysics (Breeze notation.md rows)
    :specific_humidity                    => :qᵛ,
    :air_specific_humidity                => :qᵛ,
    :specific_cloud_liquid_water_content  => :qᶜˡ,
    :specific_cloud_ice_water_content     => :qᶜⁱ,
    :specific_rain_water_content          => :qʳ,
    # Sea ice (notation.md `ℵ` row; `:h` matches ClimaSeaIce field name)
    :sea_ice_thickness                    => :h,
    :sea_ice_concentration                => :ℵ,
    # Freshwater fluxes (notation.md "Net surface freshwater fluxes" subsection)
    :rain_freshwater_flux                 => :Jʳⁿ,
    :snow_freshwater_flux                 => :Jˢⁿ,
    # Biogeochemistry (matches the short symbols dispatched in restoring.jl:49-61)
    :dissolved_inorganic_carbon           => :DIC,
    :alkalinity                           => :ALK,
    :nitrate                              => :NO₃,
    :phosphate                            => :PO₄,
    :dissolved_organic_phosphorus         => :DOP,
    :particulate_organic_phosphorus       => :POP,
    :dissolved_iron                       => :Fe,
    :dissolved_silicate                   => :SiO₂,
    :dissolved_oxygen                     => :O₂,
    # Land surface variables
    :sand_fraction                        => :sand,
    :silt_fraction                        => :silt,
    :clay_fraction                        => :clay,
    :bulk_density                         => :ρ_soil,
    :organic_carbon_density               => :ρ_soc,
    :soil_organic_carbon                  => :SOC
)

# Only temperature and salinity need a thorough inpainting because of stability,
# other variables can do with only a couple of passes. Sea ice variables
# cannot be inpainted because zeros in the data are physical, not missing values.
function default_inpainting(metadata)
    if metadata.name in (:temperature, :salinity)
        return NearestNeighborInpainting(Inf)
    elseif metadata.name in (:sea_ice_thickness, :sea_ice_concentration)
        return nothing
    else
        return NearestNeighborInpainting(5)
    end
end

# Datasets
include("ETOPO/ETOPO.jl")
include("ECCO/ECCO.jl")
include("GLORYS/GLORYS.jl")
include("ERA5/ERA5.jl")
include("EN4/EN4.jl")
include("ORCA/ORCA.jl")
include("WOA/WOA.jl")
include("JRA55/JRA55.jl")
include("GloFAS/GloFAS.jl")
include("OSPapa/OSPapa.jl")
include("FLUXNET/FLUXNET.jl")
include("SoilGrids/SoilGrids.jl")
include("IBCSO/IBCSO.jl")
include("GEBCO/GEBCO.jl")
include("IBCAO/IBCAO.jl")
include("CopernicusDEM/CopernicusDEM.jl")

using .ETOPO
using .ECCO
using .GLORYS
using .ERA5
using .EN4
using .ORCA
using .WOA
using .JRA55
using .GloFAS
using .OSPapa
using .FLUXNET
using .IBCSO
using .GEBCO
using .IBCAO
using .CopernicusDEM

function dataset_modules()
    modules = Module[]

    for name in names(DataWrangling; all = true, imported = false)
        isdefined(DataWrangling, name) || continue
        child = getfield(DataWrangling, name)
        child isa Module || continue
        child === DataWrangling && continue
        parentmodule(child) === DataWrangling || continue
        push!(modules, child)
    end

    sort!(modules; by = string)
    return modules
end

function is_supported_dataset_constructor(dataset_constructor)
    dataset_constructor isa DataType || return false
    isconcretetype(dataset_constructor) || return false
    applicable(dataset_constructor) || return false

    dataset = try
        dataset_constructor()
    catch
        return false
    end

    return applicable(default_download_directory, dataset)
end

function dataset_constructor_list()
    constructors = DataType[]
    seen = Set{DataType}()

    for mod in dataset_modules()
        for name in names(mod)
            isdefined(mod, name) || continue
            constructor = getfield(mod, name)
            is_supported_dataset_constructor(constructor) || continue
            constructor in seen && continue
            push!(seen, constructor)
            push!(constructors, constructor)
        end
    end

    sort!(constructors; by = string)
    return constructors
end

function dataset_constructor_docstring()
    constructors = dataset_constructor_list()
    names = ["`$(nameof(constructor))()`" for constructor in constructors]
    N = length(names)

    if N == 0
        return "No dataset constructors are currently available."
    elseif N == 1
        return only(names)
    else
        return string(join(names[1:N-1], ", "), ", and ", names[N])
    end
end

"""
    supported_datasets()

Return the dataset constructors currently supported by [`Metadata`](@ref).

Currently, these are: $(dataset_constructor_docstring()).

!!! info "Importing datasets"
    Some of the above datasets are not exported and need to be explicitly imported
    before, e.g., passed to [`Metadata`](@ref).
"""
supported_datasets() = dataset_constructor_list()

# Fallback: if no download extension is loaded, check that all files already exist
function Downloads.download(metadata::Metadata)
    error("No download method for $metadata is available (is the backend package loaded?)")
end

end # module
