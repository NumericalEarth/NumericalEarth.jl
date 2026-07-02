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
    NestedModel,
    NestedSimulation,
    nested_atmosphere_model,
    parent_boundary_conditions,
    parent_forcings,
    # Atmosphere-land interface closures
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
    GloFASPrescribedLand,
    GloFASReanalysis,
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
    DiffusiveFlux,
    InteriorDiffusivity,
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
    atmosphere_model,
    atmosphere_simulation,
    breeze_prognostic_state,
    bulk_drag,
    hydrostatic_pressure_from_surface,
    density_from_pressure,
    ocean_simulation,
    sea_ice_simulation,
    default_sea_ice,
    sea_ice_dynamics,
    initialize!,
    net_ocean_heat_flux, sea_ice_ocean_heat_flux, atmosphere_ocean_heat_flux,
    net_ocean_freshwater_flux, sea_ice_ocean_freshwater_flux, atmosphere_ocean_freshwater_flux,
    meridional_heat_transport,
    location,
    native_grid,
    natural_earth_lines,
    surface_elevation,
    exchange_state!,
    total_density,
    visualize_nested_domain

using DataDeps: DataDeps
using Oceananigans: Oceananigans, initialize!
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: _node
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.OutputReaders: GPUAdaptedFieldTimeSeries, FieldTimeSeries

import Oceananigans: location

"""
    natural_earth_lines(name; scale = 50)

Return `(longitudes, latitudes)` for the Natural Earth feature layer `name` (e.g.
`"admin_1_states_provinces_lines"` or `"admin_0_boundary_lines_land"`) as NaN-separated
vectors, ready to draw as disjoint segments with a plotting backend, as in
`lines!(axis, longitudes, latitudes)`. `scale` selects the Natural Earth resolution
(`10`, `50`, or `110`).

Requires `NaturalEarth` and `GeoInterface` to be loaded — the method lives in
`NumericalEarthNaturalEarthExt`.
"""
function natural_earth_lines end

"""
    visualize_nested_domain(grid; parent = nothing, kw...)

Return a `Makie.Figure` mapping the domain of `grid` — a topography/bathymetry basemap
with the domain drawn as a box — for checking a configuration's geography before running.
When `parent` (a `BoundingBox` or another grid) is given, its region is drawn as a second
box, visualizing the nesting; omit it to map a single domain.

Keyword arguments
=================

- `parent`: a `BoundingBox` or grid whose region is drawn as the outer (parent) box. Default: `nothing`.
- `padding`: margin (degrees) between the outermost region and the map edge. Default: `2.5`.
- `resolution`: basemap sampling resolution (degrees). Default: `1/30`.
- `dataset`: relief dataset for the basemap. Default: `ETOPO2022()`.
- `boundaries`: draw Natural Earth state/country boundary lines (requires `NaturalEarth`
  and `GeoInterface` to be loaded). Default: `true`.
- `landmarks`: `label => (λ, φ)` pairs marked with stars, e.g. `tuple("ARM SGP" => (-97.485, 36.605))`.
  Default: `tuple()`.
- `label`, `parent_label`: legend labels for the two domain boxes. Defaults: `"grid"`, `"parent"`.
- `title`: axis title. Default: `""`.

Requires a Makie backend (e.g. `CairoMakie`) to be loaded — the method lives in
`NumericalEarthMakieExt`.
"""
function visualize_nested_domain end

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
include("NestedModels/NestedModels.jl")   # last: wraps a parent + a child (any component) model

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
using .NestedModels
using .NestedModels: NestedModel, NestedSimulation, nested_atmosphere_model, parent_boundary_conditions, parent_forcings
using .DataWrangling.ETOPO
using .DataWrangling.ECCO
using .DataWrangling.GLORYS
using .DataWrangling.EN4
using .DataWrangling.ORCA
using .DataWrangling.WOA
using .DataWrangling.JRA55
using .DataWrangling.GloFAS
using .DataWrangling.OSPapa
using .DataWrangling.ERA5
using .DataWrangling.SoilGrids

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
