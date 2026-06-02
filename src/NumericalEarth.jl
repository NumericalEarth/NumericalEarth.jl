module NumericalEarth

# Use the README as the module docs
@doc let
    """Helper function to read the README.md file and convert it to plain Markdown for use as module documentation."""
    function readme_for_module_docs(path::AbstractString)
        readme = read(path, String)

        # Julia docstrings use Base.Markdown, which escapes raw HTML.
        # Translate the HTML-heavy README sections to plain Markdown on the fly.
        readme = replace(readme, r"<!--.*?-->"s => "")
        readme = replace(readme,
                        r"<a\s+href=\"([^\"]+)\"[^>]*>\s*<img\s+src=\"([^\"]+)\"[^>]*?(?:alt=\"([^\"]*)\")?[^>]*>\s*</a>"s =>
                        s"[![\3](\2)](\1)")
        readme = replace(readme,
                        r"<a\s+href=\"([^\"]+)\"[^>]*>([^<]+)</a>"s =>
                        s"[\2](\1)")
        readme = replace(readme, r"<h1[^>]*>\s*([^<]+?)\s*</h1>"s => s"# \1\n")
        readme = replace(readme, r"<pre>\s*<code>"s => "```bibtex\n")
        readme = replace(readme, r"</code>\s*</pre>"s => "\n```")
        readme = replace(readme, r"<code>\s*(.*?)\s*</code>"s => s"```\n\1\n```")
        readme = replace(readme, r"</?p\b[^>]*>" => "")
        readme = replace(readme, r"</?strong\b[^>]*>" => "")
        readme = replace(readme,
                        r"<details>\s*<summary>\s*([^<]+?)\s*</summary>"s =>
                        s"### \1\n")
        readme = replace(readme, "</details>" => "")
        readme = replace(readme, r"\n{3,}" => "\n\n")

        return strip(readme)
    end

    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    readme_for_module_docs(path)
end NumericalEarth

export
    EarthSystemModel,
    OceanOnlyModel,
    OceanSeaIceModel,
    AtmosphereOceanModel,
    AtmosphereLandModel,
    SkinHumidity,
    FractionalHumidity,
    CriticalSaturation,
    ElevationCorrection,
    atmosphere_land_interface,
    SlabOcean,
    PrescribedOcean,
    AbstractPrescribedComponent,
    PrescribedRadiation,
    PrescribedAtmosphere,
    PrescribedLand,
    ECCOPrescribedRadiation,
    JRA55PrescribedRadiation,
    JRA55PrescribedAtmosphere,
    JRA55PrescribedLand,
    ERA5PrescribedAtmosphere,
    ERA5PrescribedRadiation,
    OSPapaPrescribedRadiation,
    OSPapaPrescribedAtmosphere,
    os_papa_prescribed_fluxes,
    os_papa_prescribed_flux_boundary_conditions,
    FreezingLimitedOceanTemperature,
    SurfaceRadiationProperties,
    InterfaceRadiationFlux,
    LatitudeDependentAlbedo,
    TabulatedAlbedo,
    SeaIceAlbedo,
    SimilarityTheoryFluxes,
    CoefficientBasedFluxes,
    SimilarityScales,
    MomentumRoughnessLength,
    ScalarRoughnessLength,
    ComponentInterfaces,
    SkinTemperature,
    BulkTemperature,
    # Land (prognostic SlabLand + closures)
    SlabLand,
    SlabEnergy,
    BucketHydrology,
    surface_temperature,
    regrid_bathymetry,
    regrid_topography,
    Metadata, Metadatum, MetadataSet,
    BoundingBox,
    Column, Linear, Nearest,
    ECCOMetadatum,
    EN4Metadatum,
    ETOPO2022,
    ECCO2Daily, ECCO2Monthly, ECCO4Monthly,
    ECCO2DarwinMonthly, ECCO4DarwinMonthly,
    EN4Monthly,
    WOAClimatology, WOAAnnual, WOAMonthly,
    GLORYSDaily, GLORYSMonthly, GLORYSStatic,
    RepeatYearJRA55, MultiYearJRA55,
    ERA5HourlySingleLevel, ERA5MonthlySingleLevel,
    ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels,
    OSPapaHourly,
    JRA55FieldTimeSeries,
    ORCA1, ORCA12,
    ORCAGrid,
    first_date, last_date, all_dates,
    LinearlyTaperedPolarMask,
    DatasetRestoring,
    atmosphere_simulation,
    ocean_simulation,
    sea_ice_simulation,
    default_sea_ice,
    sea_ice_dynamics,
    initialize!,
    net_ocean_heat_flux, sea_ice_ocean_heat_flux, atmosphere_ocean_heat_flux,
    net_ocean_freshwater_flux, sea_ice_ocean_freshwater_flux, atmosphere_ocean_freshwater_flux,
    meridional_heat_transport,
    location,
    native_grid

using DataDeps: DataDeps
using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: _node
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.OutputReaders: GPUAdaptedFieldTimeSeries, FieldTimeSeries

import Oceananigans: location

const SomeKindOfFieldTimeSeries = Union{FieldTimeSeries,
                                        GPUAdaptedFieldTimeSeries}

const SKOFTS = SomeKindOfFieldTimeSeries

@inline stateindex(a::Number, i, j, k, args...) = a
@inline stateindex(a::AbstractArray, i, j, k, args...) = @inbounds a[i, j, k]
@inline stateindex(a::SKOFTS, i, j, k, grid, time, args...) = @inbounds a[i, j, k, time]

@inline function stateindex(a::Function, i, j, k, grid, time, (LX, LY, LZ), args...)
    # `_node` always returns the full (λ, φ, z) triple — with placeholder
    # values for Flat dimensions — whereas `node` drops Flat-dim entries
    # and produces a shorter tuple that breaks the destructuring below.
    λ, φ, z = _node(i, j, k, grid, LX(), LY(), LZ())
    return a(λ, φ, z, time)
end

@inline function stateindex(a::Tuple, i, j, k, grid, time, args...)
    N = length(a)
    ntuple(Val(N)) do n
        stateindex(a[n], i, j, k, grid, time, args...)
    end
end

@inline function stateindex(a::NamedTuple, i, j, k, grid, time, args...)
    vals = stateindex(values(a), i, j, k, grid, time, args...)
    names = keys(a)
    return NamedTuple{names}(vals)
end

#####
##### Source code
#####

include("Grids/Grids.jl")
include("EarthSystemModels/EarthSystemModels.jl")
include("Oceans/Oceans.jl")
include("Atmospheres/Atmospheres.jl")
include("Lands/Lands.jl")
include("Radiations/Radiations.jl")
include("SeaIces/SeaIces.jl")
include("InitialConditions/InitialConditions.jl")
include("DataWrangling/DataWrangling.jl")
include("Bathymetry/Bathymetry.jl")
include("Diagnostics/Diagnostics.jl")

using .Grids
using .DataWrangling
using .DataWrangling: ETOPO, ECCO, GLORYS, EN4, WOA, JRA55, OSPapa
using .Bathymetry
using .InitialConditions
using .EarthSystemModels
using .Atmospheres
using .Lands
using .Radiations
using .Oceans
using .SeaIces
using .Diagnostics
using .EarthSystemModels: ComponentInterfaces, MomentumRoughnessLength, ScalarRoughnessLength, default_sea_ice
using .DataWrangling.ETOPO
using .DataWrangling.ECCO
using .DataWrangling.GLORYS
using .DataWrangling.EN4
using .DataWrangling.ORCA
using .DataWrangling.WOA
using .DataWrangling.JRA55
using .DataWrangling.OSPapa
using .DataWrangling.ERA5

using PrecompileTools: @setup_workload, @compile_workload

"""
Process-level entry point that fires once after every submodule's `__init__` has run.

- In `:auto` mode (the default), auto-downloads datasets listed in `NumericalEarthDataManifest.toml`
  whenever a manifest sits next to the active project's `Project.toml`. Cached files are skipped by
  each dataset's per-dataset `Downloads.download` method, so subsequent runs are cheap.
- In `:pregenerate` mode (`NUMERICALEARTH_DATA=pregenerate` or `pregenerate:<dir>`), traces
  `Base.PROGRAM_FILE` via `pregenerate_dataset_manifest` and `exit(0)` — the script's real
  execution is skipped. The trace runs silently (`quiet = true`) so only the final
  `wrote manifest` log appears.

Both paths are no-ops during precompilation, in `:strict` mode, and when no real `PROGRAM_FILE`
is set (REPL / `julia -e ...`).
"""
function __init__()
    ccall(:jl_generating_output, Cint, ()) == 1 && return nothing
    (!isempty(Base.PROGRAM_FILE) && isfile(Base.PROGRAM_FILE)) || return nothing

    mode = DataWrangling.DataModes.DATA_MODE[]
    if mode === :pregenerate
        script = abspath(Base.PROGRAM_FILE)
        # `MANIFEST_DIR[]` is populated by `DataModes.__init__` from the env var; if that init
        # somehow hasn't run (precompile workload edge case), fall back to the current directory.
        dir = isempty(DataWrangling.DataModes.MANIFEST_DIR[]) ? pwd() : DataWrangling.DataModes.MANIFEST_DIR[]
        try
            manifest = DataWrangling.DataModes.pregenerate_dataset_manifest(script; dir)
            @info "NUMERICALEARTH_DATA=pregenerate: wrote manifest via AST trace" manifest script
        catch err
            @error "NUMERICALEARTH_DATA=pregenerate: trace failed" dir script exception=(err, catch_backtrace())
        end
        exit(0)
    end

    mode === :auto || return nothing

    project = Base.active_project()
    project === nothing && return nothing
    project_dir = dirname(project)
    manifest = joinpath(project_dir, DataWrangling.DataModes.MANIFEST_FILENAME)
    isfile(manifest) || return nothing

    @info "NumericalEarth: auto-downloading datasets from manifest" manifest
    try
        DataWrangling.DataModes.download_datasets(; dir = project_dir)
    catch err
        @error "NumericalEarth: auto-download failed; continuing without it" manifest exception=(err, catch_backtrace())
    end
    return nothing
end

@setup_workload begin
    Nx, Ny, Nz = 32, 32, 10
    @compile_workload begin
        depth = 6000
        z = Oceananigans.Grids.ExponentialDiscretization(Nz, -depth, 0)
        grid = Oceananigans.OrthogonalSphericalShellGrids.TripolarGrid(CPU(); size=(Nx, Ny, Nz), halo=(7, 7, 7), z)
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -5000))
        # ocean = ocean_simulation(grid)
        # model = OceanOnlyModel(ocean)
    end
end

end # module
