using Test
using NumericalEarth
using ExplicitImports: ExplicitImports

@testset "ExplicitImports" begin
    modules = (
        NumericalEarth,
        NumericalEarth.Atmospheres,
        NumericalEarth.Bathymetry,
        NumericalEarth.DataWrangling,
        NumericalEarth.DataWrangling.ECCO,
        NumericalEarth.DataWrangling.EN4,
        NumericalEarth.DataWrangling.ERA5,
        NumericalEarth.DataWrangling.ETOPO,
        NumericalEarth.DataWrangling.GEBCO,
        NumericalEarth.DataWrangling.GLORYS,
        NumericalEarth.DataWrangling.IBCAO,
        NumericalEarth.DataWrangling.IBCSO,
        NumericalEarth.DataWrangling.JRA55,
        NumericalEarth.DataWrangling.ORCA,
        NumericalEarth.DataWrangling.OSPapa,
        NumericalEarth.DataWrangling.WOA,
        NumericalEarth.Diagnostics,
        NumericalEarth.EarthSystemModels,
        NumericalEarth.EarthSystemModels.InterfaceComputations,
        NumericalEarth.InitialConditions,
        NumericalEarth.Lands,
        NumericalEarth.Oceans,
        NumericalEarth.Radiations,
        NumericalEarth.SeaIces,
    )

    @testset "Explicit Imports [$(mod)]" for mod in modules
        @test ExplicitImports.check_no_implicit_imports(mod) === nothing
        @test ExplicitImports.check_no_stale_explicit_imports(mod) === nothing
        @test ExplicitImports.check_no_self_qualified_accesses(mod) === nothing
    end
end
