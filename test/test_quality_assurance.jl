using Test
using NumericalEarth
using ArchGDAL
using Breeze
using CDSAPI
using PythonCall, CondaPkg
using SpeedyWeather, ConservativeRegridding
using WorldOceanAtlasTools
using ExplicitImports: ExplicitImports

@testset "Explicit Imports" begin
    @test ExplicitImports.check_no_implicit_imports(NumericalEarth) === nothing
end

@testset "Import via Owner" begin
    @test ExplicitImports.check_all_explicit_imports_via_owners(NumericalEarth) === nothing
end

@testset "Stale Explicit Imports" begin
    @test ExplicitImports.check_no_stale_explicit_imports(NumericalEarth) === nothing
end

@testset "Qualified Accesses" begin
    @test ExplicitImports.check_all_qualified_accesses_via_owners(NumericalEarth) === nothing
end

@testset "Self Qualified Accesses" begin
    @test ExplicitImports.check_no_self_qualified_accesses(NumericalEarth) === nothing
end

# DataWrangling sits at the bottom of NumericalEarth's module stack: it may import Oceananigans
# and other packages, but nothing owned by another NumericalEarth module.
function foreign_numericalearth_imports(mod, found = Tuple{Module, Symbol, Module}[])
    for name in names(mod; all = true, imported = true)
        isdefined(mod, name) || continue
        owner = Base.which(mod, name)
        value = getfield(mod, name)
        if value isa Module && parentmodule(value) === mod && value !== mod
            foreign_numericalearth_imports(value, found)
        elseif Base.moduleroot(owner) === NumericalEarth && !is_within(owner, NumericalEarth.DataWrangling)
            push!(found, (mod, name, owner))
        end
    end
    return found
end

is_within(mod, ancestor) = mod === ancestor || (parentmodule(mod) !== mod && is_within(parentmodule(mod), ancestor))

@testset "DataWrangling imports nothing else from NumericalEarth" begin
    @test isempty(foreign_numericalearth_imports(NumericalEarth.DataWrangling))
end
