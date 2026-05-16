using Test
using NumericalEarth
# using ArchGDAL
# using Breeze
# using CDSAPI
# using PythonCall, CondaPkg
# using SpeedyWeather, ConservativeRegridding
# using WorldOceanAtlasTools
using ExplicitImports: ExplicitImports

@testset "ExplicitImports" begin
    modules = Module[
        # NumericalEarth,
        # NumericalEarth.Atmospheres,
        NumericalEarth.Bathymetry,
        # NumericalEarth.DataWrangling,
        # NumericalEarth.DataWrangling.ECCO,
        # NumericalEarth.DataWrangling.EN4,
        # NumericalEarth.DataWrangling.ERA5,
        # NumericalEarth.DataWrangling.ETOPO,
        # NumericalEarth.DataWrangling.GEBCO,
        # NumericalEarth.DataWrangling.GLORYS,
        # NumericalEarth.DataWrangling.IBCAO,
        # NumericalEarth.DataWrangling.IBCSO,
        # NumericalEarth.DataWrangling.JRA55,
        # NumericalEarth.DataWrangling.ORCA,
        # NumericalEarth.DataWrangling.OSPapa,
        # NumericalEarth.DataWrangling.WOA,
        NumericalEarth.Diagnostics,
        # NumericalEarth.EarthSystemModels,
        # NumericalEarth.EarthSystemModels.InterfaceComputations,
        NumericalEarth.InitialConditions,
        # NumericalEarth.Lands,
        # NumericalEarth.Oceans,
        # NumericalEarth.Radiations,
        NumericalEarth.SeaIces,
    ]

    function maybe_extension(parent, name::Symbol)
        if isdefined(Base, :get_extension)
            return Base.get_extension(parent, name)
        else
            return isdefined(parent, name) ? getproperty(parent, name) : nothing
        end
    end

    for ext in (
        # maybe_extension(NumericalEarth, :NumericalEarthSpeedyWeatherExt),
        # maybe_extension(NumericalEarth, :NumericalEarthVerosExt),
        # maybe_extension(NumericalEarth, :NumericalEarthBreezeExt),
        # maybe_extension(NumericalEarth, :NumericalEarthWOAExt),
        # maybe_extension(NumericalEarth, :NumericalEarthCDSAPIExt),
        # maybe_extension(NumericalEarth, :NumericalEarthArchGDALExt),
    )
        isnothing(ext) || push!(modules, ext)
    end

    @testset "Explicit Imports [$(mod)]" for mod in modules
        @test ExplicitImports.check_no_implicit_imports(mod) === nothing
        @test ExplicitImports.check_no_stale_explicit_imports(mod) === nothing
        @test ExplicitImports.check_no_self_qualified_accesses(mod) === nothing
        @test ExplicitImports.check_all_explicit_imports_via_owners(mod) === nothing
    end
end
