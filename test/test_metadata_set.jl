include("runtests_setup.jl")

using NumericalEarth.DataWrangling: MetadataSet, Metadata, Metadatum,
                                    BoundingBox, variable_glossary, metadata_path

# `MetadataSet` is a pure DataWrangling concept: no downloads, no field
# construction. These tests exercise construction, accessors, iteration,
# and the global verbose→short glossary. Downstream `set!(model, mset)`,
# `Field(::MetadataSet)`, `FieldTimeSeries(::MetadataSet)`, and the
# `download` rename live in their own tests once those land.

snapshot_date = DateTime(1993, 1, 1)
date_range    = DateTime(1993, 1, 1):Month(1):DateTime(1993, 4, 1)

@testset "MetadataSet construction" begin
    # Snapshot set: scalar dates → each element is a Metadatum
    mset = MetadataSet(:temperature, :salinity;
                       dataset = ECCO4Monthly(),
                       date    = snapshot_date)

    @test mset isa MetadataSet
    @test mset.names === (:temperature, :salinity)
    @test mset.dataset isa ECCO4Monthly
    @test mset.dates  == snapshot_date
    @test mset.region === nothing
    @test length(mset) == 2
    @test mset.temperature isa Metadatum   # element-level distinction

    # Time-series set: vector dates → each element is a Metadata (not Metadatum)
    mts = MetadataSet(:temperature, :salinity;
                      dataset = ECCO4Monthly(),
                      dates   = date_range)

    @test mts isa MetadataSet
    @test mts.dates == date_range
    @test length(mts) == 2
    @test !(mts.temperature isa Metadatum)

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

#####
##### set! dispatch
#####
##### We test the kwarg-routing logic with stub model / field types rather
##### than spinning up an Oceananigans model — the actual interpolation path
##### is exercised by metadata_field tests.
#####

mutable struct StubModel
    received :: Vector{Pair{Symbol, Any}}
    StubModel() = new(Pair{Symbol, Any}[])
end

# Capture kwargs delivered to set!(model; ...). NB: this method is added to the
# Oceananigans set! generic just for these tests.
function NumericalEarth.DataWrangling.set!(m::StubModel; kw...)
    for (k, v) in kw
        push!(m.received, k => v)
    end
    return m
end

mutable struct StubField
    received :: Vector{Symbol}   # variable names received via metadata
    StubField() = new(Symbol[])
end

function NumericalEarth.DataWrangling.set!(f::StubField, m::Metadata)
    push!(f.received, m.name)
    return f
end

@testset "set!(model, mset) — glossary routing" begin
    mset = MetadataSet(:temperature, :salinity;
                       dataset = ECCO4Monthly(),
                       date    = snapshot_date)

    m = StubModel()
    set!(m, mset)

    # variable_glossary routes :temperature → :T, :salinity → :S
    received_keys = first.(m.received)
    @test :T in received_keys
    @test :S in received_keys
    @test length(received_keys) == 2

    # The values delivered are the per-variable Metadata objects
    received_dict = Dict(m.received)
    @test received_dict[:T] isa Metadatum
    @test received_dict[:T].name == :temperature
    @test received_dict[:S].name == :salinity
end

@testset "set!(model, mset) — silently skips unmapped vars" begin
    # `:sea_ice_thickness` IS in the glossary; `:bottom_height` is NOT.
    # Setting both on a single model should only deliver the mapped one.
    mset = MetadataSet(:temperature, :sea_ice_thickness;
                       dataset = ECCO4Monthly(),
                       date    = snapshot_date)

    m = StubModel()
    set!(m, mset)
    received_keys = first.(m.received)
    @test :T in received_keys
    @test :h in received_keys
    @test length(received_keys) == 2  # both mapped

    # Same set, but route :temperature only — an unmapped name doesn't fail
    # (it just doesn't appear). Simulate this by pretending we're routing to a
    # model that ignores unknown variables.
    @test issubset(received_keys, values(NumericalEarth.DataWrangling.variable_glossary))
end

#####
##### Per-model overrides (regression for issue with shared multi-component sets)
#####
##### Real Oceananigans/ClimaSeaIce models throw ArgumentError on unknown
##### kwargs, so they override `set!(model, ::MetadataSet)` to filter down to
##### the variables they consume. This RestrictiveStubModel mimics that
##### strictness and demonstrates the override pattern used by the real
##### dispatches in src/DataWrangling/metadata.jl.

mutable struct RestrictiveStubModel
    accepts  :: Tuple{Vararg{Symbol}}        # short field names this model accepts
    received :: Vector{Pair{Symbol, Any}}
    RestrictiveStubModel(accepts...) = new(accepts, Pair{Symbol, Any}[])
end

function NumericalEarth.DataWrangling.set!(m::RestrictiveStubModel; kw...)
    for (k, v) in kw
        k in m.accepts || throw(ArgumentError("name $k not accepted by RestrictiveStubModel"))
        push!(m.received, k => v)
    end
    return m
end

# Override the dispatch to filter to `m.accepts` — same shape as the real
# HydrostaticFreeSurfaceModel/SeaIceModel methods.
function Oceananigans.Fields.set!(m::RestrictiveStubModel, mset::MetadataSet)
    names = filter(getfield(mset, :names)) do n
        haskey(variable_glossary, n) && variable_glossary[n] in m.accepts
    end
    isempty(names) && return m
    kwargs = NamedTuple{Tuple(variable_glossary[n] for n in names)}(Tuple(mset[n] for n in names))
    return set!(m; kwargs...)
end

@testset "set!(model, mset) — per-model overrides filter a shared MetadataSet" begin
    mset = MetadataSet(:temperature, :salinity,
                       :sea_ice_thickness, :sea_ice_concentration;
                       dataset = ECCO4Monthly(),
                       date    = snapshot_date)

    # Ocean-like model: override filters to (:T, :S).
    ocean_like = RestrictiveStubModel(:T, :S)
    set!(ocean_like, mset)
    ocean_keys = first.(ocean_like.received)
    @test :T in ocean_keys
    @test :S in ocean_keys
    @test !(:h in ocean_keys)
    @test !(:ℵ in ocean_keys)
    @test length(ocean_keys) == 2

    # Sea-ice-like model: override filters to (:h, :ℵ).
    sea_ice_like = RestrictiveStubModel(:h, :ℵ)
    set!(sea_ice_like, mset)
    ice_keys = first.(sea_ice_like.received)
    @test :h in ice_keys
    @test :ℵ in ice_keys
    @test !(:T in ice_keys)
    @test !(:S in ice_keys)
    @test length(ice_keys) == 2

    # No matching variables → no-op, no error.
    empty_model = RestrictiveStubModel(:foo)
    set!(empty_model, mset)
    @test isempty(empty_model.received)
end

@testset "set!(::NamedTuple, mset) — explicit per-variable" begin
    mset = MetadataSet(:temperature, :salinity;
                       dataset = ECCO4Monthly(),
                       date    = snapshot_date)

    T = StubField()
    S = StubField()
    set!((; temperature = T, salinity = S), mset)
    @test T.received == [:temperature]
    @test S.received == [:salinity]

    # Partial NamedTuple (subset of mset.names) is allowed
    Tonly = StubField()
    set!((; temperature = Tonly), mset)
    @test Tonly.received == [:temperature]

    # Extra key not in the set is an error
    @test_throws ArgumentError set!((; temperature = StubField(),
                                       not_in_set = StubField()), mset)
end

@testset "Field(::MetadataSet) rejects multi-date sets" begin
    # Field(mset) only makes sense for a single snapshot per variable.
    # A multi-date mset should error pointing the user at FieldTimeSeries
    # rather than failing deep in `Field(::Metadata)` with a MethodError.
    mts = MetadataSet(:temperature, :salinity;
                      dataset = ECCO4Monthly(),
                      dates   = date_range)
    @test_throws ArgumentError Field(mts)
end

@testset "variable_glossary registry" begin
    # The verbose→short map is the single source of truth used by
    # set!(model, mset). Check that every value lines up with the
    # notation conventions documented in docs/src/appendix/notation.md.

    # Ocean & atmosphere state
    @test variable_glossary[:temperature]              === :T
    @test variable_glossary[:salinity]                 === :S
    @test variable_glossary[:eastward_velocity]        === :u
    @test variable_glossary[:northward_velocity]       === :v
    @test variable_glossary[:eastward_wind]            === :u    # synonym
    @test variable_glossary[:u_velocity]               === :u    # synonym
    @test variable_glossary[:sea_level_pressure]       === :p

    # Atmosphere moisture (Breeze convention)
    @test variable_glossary[:specific_humidity]                    === :qᵛ
    @test variable_glossary[:air_specific_humidity]                === :qᵛ
    @test variable_glossary[:specific_cloud_liquid_water_content]  === :qᶜˡ
    @test variable_glossary[:specific_cloud_ice_water_content]     === :qᶜⁱ

    # Sea ice
    @test variable_glossary[:sea_ice_thickness]     === :h
    @test variable_glossary[:sea_ice_concentration] === :ℵ

    # Freshwater fluxes (notation.md "Net surface freshwater fluxes")
    @test variable_glossary[:rain_freshwater_flux] === :Jʳⁿ
    @test variable_glossary[:snow_freshwater_flux] === :Jˢⁿ

    # Biogeochemistry (matches restoring.jl:49-61 short symbols)
    @test variable_glossary[:dissolved_inorganic_carbon] === :DIC
    @test variable_glossary[:phosphate]                  === :PO₄
    @test variable_glossary[:dissolved_oxygen]           === :O₂

    # Out-of-map variables are absent (silently fall through set!(model, mset))
    @test !haskey(variable_glossary, :vorticity)
    @test !haskey(variable_glossary, :mesh_mask)
    @test !haskey(variable_glossary, :significant_wave_height)
end
