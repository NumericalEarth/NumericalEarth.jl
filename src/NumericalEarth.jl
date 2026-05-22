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
    regrid_bathymetry,
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
    OSPapaHourly,
    JRA55FieldTimeSeries,
    ORCA1, ORCA12,
    ORCAGrid,
    first_date, last_date, all_dates,
    LinearlyTaperedPolarMask,
    DatasetRestoring,
    atmosphere_simulation,
    ocean_simulation,
    nonhydrostatic_ocean_simulation,
    sea_ice_simulation,
    default_sea_ice,
    sea_ice_dynamics,
    initialize!,
    frazil_temperature_flux, net_ocean_temperature_flux, sea_ice_ocean_temperature_flux, atmosphere_ocean_temperature_flux,
    frazil_heat_flux, net_ocean_heat_flux, sea_ice_ocean_heat_flux, atmosphere_ocean_heat_flux,
    net_ocean_salinity_flux, sea_ice_ocean_salinity_flux, atmosphere_ocean_salinity_flux,
    net_ocean_freshwater_flux, sea_ice_ocean_freshwater_flux, atmosphere_ocean_freshwater_flux,
    meridional_heat_transport,
    location,
    native_grid

using DataDeps: DataDeps
using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: node
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
    λ, φ, z = node(i, j, k, grid, LX(), LY(), LZ())
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

using PrecompileTools: @setup_workload, @compile_workload

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
