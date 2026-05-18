include("runtests_setup.jl")

using NumericalEarth.DataWrangling: MetadataSet, MetadatumSet, Metadata, Metadatum,
                                    BoundingBox, variable_aliases, metadata_path

# `MetadataSet` is a pure DataWrangling concept: no downloads, no field
# construction. These tests exercise construction, accessors, iteration,
# and the global verbose→short alias map. Downstream `set!(model, mset)`,
# `Field(::MetadataSet)`, `FieldTimeSeries(::MetadataSet)`, and the
# `download` rename live in their own tests once those land.

snapshot_date = DateTime(1993, 1, 1)
date_range    = DateTime(1993, 1, 1):Month(1):DateTime(1993, 4, 1)

@testset "MetadataSet construction" begin
    # Snapshot set → MetadatumSet
    mset = MetadataSet(:temperature, :salinity;
                       dataset = ECCO4Monthly(),
                       date    = snapshot_date)

    @test mset isa MetadataSet
    @test mset isa MetadatumSet
    @test mset.names === (:temperature, :salinity)
    @test mset.dataset isa ECCO4Monthly
    @test mset.dates  == snapshot_date
    @test mset.region === nothing
    @test length(mset) == 2

    # Time-series set → NOT a MetadatumSet
    mts = MetadataSet(:temperature, :salinity;
                      dataset = ECCO4Monthly(),
                      dates   = date_range)

    @test mts isa MetadataSet
    @test !(mts isa MetadatumSet)
    @test mts.dates == date_range
    @test length(mts) == 2

    # `date` and `dates` are mutually exclusive
    @test_throws ArgumentError MetadataSet(:temperature;
                                           dataset = ECCO4Monthly(),
                                           date    = snapshot_date,
                                           dates   = date_range)

    # At least one variable required
    @test_throws ArgumentError MetadataSet(; dataset = ECCO4Monthly(),
                                             date    = snapshot_date)

    # Custom region threads through
    region = BoundingBox(longitude = (-20.0, 20.0), latitude = (-10.0, 10.0))
    mset_r = MetadataSet(:temperature, :salinity;
                         dataset = ECCO4Monthly(),
                         date    = snapshot_date,
                         region  = region)
    @test mset_r.region === region
end

@testset "MetadataSet accessors" begin
    mset = MetadataSet(:temperature, :salinity;
                       dataset = ECCO4Monthly(),
                       date    = snapshot_date)

    # Property access for variables yields a Metadatum
    @test mset.temperature isa Metadatum
    @test mset.salinity    isa Metadatum
    @test mset.temperature.name == :temperature
    @test mset.salinity.name    == :salinity
    @test mset.temperature.dataset === mset.dataset
    @test mset.temperature.dates    == snapshot_date

    # Property access still reaches struct fields
    @test mset.dataset isa ECCO4Monthly
    @test mset.dir     isa String

    # Indexed access is symmetric with property access
    @test mset[:temperature].name == :temperature
    @test mset[:salinity].name    == :salinity
    @test mset[1].name == :temperature
    @test mset[2].name == :salinity
    @test mset[end].name == :salinity

    # Unknown property throws
    @test_throws KeyError mset.nonexistent_variable

    # `propertynames` includes both variables and struct fields
    pn = propertynames(mset)
    @test :temperature in pn
    @test :salinity    in pn
    @test :dataset     in pn
    @test :names       in pn
end

@testset "MetadataSet iteration & length" begin
    mset = MetadataSet(:temperature, :salinity;
                       dataset = ECCO4Monthly(),
                       date    = snapshot_date)

    @test length(mset) == 2
    @test keys(mset)   === (:temperature, :salinity)
    @test eltype(mset) === Metadata
    @test firstindex(mset) == 1
    @test lastindex(mset)  == 2

    # Iteration walks the variable axis
    collected = collect(mset)
    @test length(collected) == 2
    @test collected[1].name == :temperature
    @test collected[2].name == :salinity

    # Multi-date set also iterates over *variables*, not dates
    mts = MetadataSet(:temperature, :salinity;
                      dataset = ECCO4Monthly(),
                      dates   = date_range)
    @test length(mts) == 2                # 2 variables, not length(date_range)
    collected_ts = collect(mts)
    @test length(collected_ts) == 2
    @test collected_ts[1] isa Metadata    # Multi-date → Metadata, not Metadatum
    @test collected_ts[1].dates == date_range
end

@testset "MetadataSet → NamedTuple & metadata_path" begin
    mset = MetadataSet(:temperature, :salinity;
                       dataset = ECCO4Monthly(),
                       date    = snapshot_date)

    nt = NamedTuple(mset)
    @test nt isa NamedTuple
    @test keys(nt) === (:temperature, :salinity)
    @test nt.temperature.name == :temperature
    @test nt.salinity.name    == :salinity

    paths = metadata_path(mset)
    @test paths isa NamedTuple
    @test keys(paths) === (:temperature, :salinity)
    @test paths.temperature isa AbstractString
    @test paths.salinity    isa AbstractString
    @test paths.temperature == metadata_path(mset.temperature)
    @test paths.salinity    == metadata_path(mset.salinity)
end

@testset "variable_aliases registry" begin
    # The verbose→short map is the single source of truth used by
    # set!(model, mset). Check that every value lines up with the
    # notation conventions documented in docs/src/appendix/notation.md.

    # Ocean & atmosphere state
    @test variable_aliases[:temperature]              === :T
    @test variable_aliases[:salinity]                 === :S
    @test variable_aliases[:eastward_velocity]        === :u
    @test variable_aliases[:northward_velocity]       === :v
    @test variable_aliases[:eastward_wind]            === :u    # synonym
    @test variable_aliases[:u_velocity]               === :u    # synonym
    @test variable_aliases[:sea_level_pressure]       === :p

    # Atmosphere moisture (Breeze convention)
    @test variable_aliases[:specific_humidity]                    === :qᵛ
    @test variable_aliases[:air_specific_humidity]                === :qᵛ
    @test variable_aliases[:specific_cloud_liquid_water_content]  === :qᶜˡ
    @test variable_aliases[:specific_cloud_ice_water_content]     === :qᶜⁱ

    # Sea ice
    @test variable_aliases[:sea_ice_thickness]     === :h
    @test variable_aliases[:sea_ice_concentration] === :ℵ

    # Freshwater fluxes (notation.md "Net surface freshwater fluxes")
    @test variable_aliases[:rain_freshwater_flux] === :Jʳⁿ
    @test variable_aliases[:snow_freshwater_flux] === :Jˢⁿ

    # Biogeochemistry (matches restoring.jl:49-61 short symbols)
    @test variable_aliases[:dissolved_inorganic_carbon] === :DIC
    @test variable_aliases[:phosphate]                  === :PO₄
    @test variable_aliases[:dissolved_oxygen]           === :O₂

    # Out-of-map variables are absent (silently fall through set!(model, mset))
    @test !haskey(variable_aliases, :vorticity)
    @test !haskey(variable_aliases, :mesh_mask)
    @test !haskey(variable_aliases, :significant_wave_height)
end
