include("runtests_setup.jl")

using NumericalEarth.DataWrangling: test_dataset_contract, ContractReport, is_conforming

# Fixtures must be at top level so their methods extend the right generics.
struct ContractFixtureBare <: NumericalEarth.AbstractDataset end
struct ContractFixtureFull <: NumericalEarth.AbstractDataset end

import NumericalEarth.DataWrangling: dataset_variable_name, all_dates,
    retrieve_data, longitude_interfaces, latitude_interfaces, z_interfaces,
    available_variables, reversed_vertical_axis, default_download_directory,
    metadata_filename

const _FIXTURE_VARIABLES = Dict(:temperature => "T")
available_variables(::ContractFixtureFull) = _FIXTURE_VARIABLES
dataset_variable_name(md::NumericalEarth.DataWrangling.Metadata{<:ContractFixtureFull}) = _FIXTURE_VARIABLES[md.name]
all_dates(::ContractFixtureFull, ::Symbol) = DateTime(2020,1,1) : Day(1) : DateTime(2020,1,5)
retrieve_data(::NumericalEarth.DataWrangling.Metadatum{<:ContractFixtureFull}) = zeros(2, 2, 2)
longitude_interfaces(::ContractFixtureFull) = (-10, 10)
latitude_interfaces(::ContractFixtureFull) = (-10, 10)
z_interfaces(::ContractFixtureFull) = [-100.0, -50.0, 0.0]
reversed_vertical_axis(::ContractFixtureFull) = false
default_download_directory(::ContractFixtureFull) = mktempdir()
metadata_filename(::ContractFixtureFull, name, date, bbox) = "fixture_$(name)_$(date).nc"
Base.size(::ContractFixtureFull, ::Symbol) = (2, 2, 2)

@testset "test_dataset_contract harness" begin

    @testset "bare AbstractDataset subtype flags missing requirements" begin
        r = test_dataset_contract(ContractFixtureBare(); verbose=false)
        @test r isa ContractReport

        # Every required method should be :missing
        required_names = (:dataset_variable_name, :all_dates, :retrieve_data, :longitude_interfaces, :latitude_interfaces, :z_interfaces)
        for name in required_names
            c = only(filter(c -> c.name === name, r.checks))
            @test c.status === :missing
            @test c.required
        end

        # authenticate and download_file! ship with defaults from DataWrangling
        for name in (:authenticate, :download_file!)
            c = only(filter(c -> c.name === name, r.checks))
            @test c.status === :default
        end

        @test !is_conforming(r)
    end

    @testset "fully-overridden dataset reports as CONFORMING" begin
        r = test_dataset_contract(ContractFixtureFull(); verbose=false)

        required = filter(c -> c.required, r.checks)
        @test all(c -> c.status !== :missing, required)
        @test count(c -> c.status === :error, r.checks) == 0
        @test is_conforming(r)
    end

    @testset "ECCO4Monthly reports as CONFORMING" begin
        r = test_dataset_contract(ECCO4Monthly(); verbose=false)

        required = filter(c -> c.required, r.checks)
        @test all(c -> c.status !== :missing, required)
        @test is_conforming(r)

        # dataset_url should be overridden
        u = only(filter(c -> c.name === :dataset_url, r.checks))
        @test u.status === :ok
    end
end
