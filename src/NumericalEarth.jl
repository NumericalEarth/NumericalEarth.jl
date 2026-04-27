module NumericalEarth

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end NumericalEarth

export
    EarthSystemModel,
    OceanOnlyModel,
    OceanSeaIceModel,
    AtmosphereOceanModel,
    SlabOcean,
    default_sea_ice,
    FreezingLimitedOceanTemperature,
    Radiation,
    LatitudeDependentAlbedo,
    SimilarityTheoryFluxes,
    CoefficientBasedFluxes,
    MomentumRoughnessLength,
    ScalarRoughnessLength,
    ComponentInterfaces,
    SkinTemperature,
    BulkTemperature,
    PrescribedAtmosphere,
    PrescribedLand,
    JRA55PrescribedAtmosphere,
    JRA55PrescribedLand,
    OSPapaPrescribedAtmosphere,
    os_papa_prescribed_fluxes,
    os_papa_prescribed_flux_boundary_conditions,
    OSPapaHourly,
    JRA55NetCDFBackend,
    regrid_bathymetry,
    Metadata,
    Metadatum,
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
    ORCA1, ORCA12,
    RepeatYearJRA55, MultiYearJRA55,
    first_date,
    last_date,
    all_dates,
    JRA55FieldTimeSeries,
    LinearlyTaperedPolarMask,
    DatasetRestoring,
    ocean_simulation,
    ORCAGrid,
    sea_ice_simulation,
    atmosphere_simulation,
    sea_ice_dynamics,
    initialize!,
    frazil_temperature_flux, net_ocean_temperature_flux, sea_ice_ocean_temperature_flux, atmosphere_ocean_temperature_flux,
    frazil_heat_flux, net_ocean_heat_flux, sea_ice_ocean_heat_flux, atmosphere_ocean_heat_flux,
    net_ocean_salinity_flux, sea_ice_ocean_salinity_flux, atmosphere_ocean_salinity_flux,
    net_ocean_freshwater_flux, sea_ice_ocean_freshwater_flux, atmosphere_ocean_freshwater_flux,
    meridional_heat_transport,
    location,
    native_grid

using Oceananigans
import Oceananigans: location
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ
using DataDeps

using Oceananigans.OutputReaders: GPUAdaptedFieldTimeSeries, FieldTimeSeries
using Oceananigans.Grids: node

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

include("EarthSystemModels/EarthSystemModels.jl")
include("Oceans/Oceans.jl")
include("Atmospheres/Atmospheres.jl")
include("Lands/Lands.jl")
include("SeaIces/SeaIces.jl")
include("InitialConditions/InitialConditions.jl")
include("DataWrangling/DataWrangling.jl")
include("Bathymetry/Bathymetry.jl")
include("Diagnostics/Diagnostics.jl")

using .DataWrangling
using .DataWrangling: ETOPO, ECCO, GLORYS, EN4, WOA, JRA55, OSPapa
using .Bathymetry
using .InitialConditions
using .EarthSystemModels
using .Atmospheres
using .Lands
using .Oceans
using .SeaIces
using .Diagnostics

using NumericalEarth.EarthSystemModels: ComponentInterfaces, MomentumRoughnessLength, ScalarRoughnessLength, default_sea_ice

using NumericalEarth.DataWrangling.ETOPO
using NumericalEarth.DataWrangling.ECCO
using NumericalEarth.DataWrangling.GLORYS
using NumericalEarth.DataWrangling.EN4
using NumericalEarth.DataWrangling.ORCA
using NumericalEarth.DataWrangling.WOA
using NumericalEarth.DataWrangling.JRA55
using NumericalEarth.DataWrangling.JRA55: JRA55NetCDFBackend
using NumericalEarth.DataWrangling.OSPapa

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
