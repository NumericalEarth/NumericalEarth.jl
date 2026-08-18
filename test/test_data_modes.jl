include("runtests_setup.jl")

using NumericalEarth.DataWrangling: AbstractMetadata, Metadatum, Metadata, MetadataSet,
                                    BoundingBox, Column, Nearest, Linear, DatewiseFilename,
                                    download_dataset
using NumericalEarth.DataWrangling.DataModes: DataModes, parse_data_mode, register_dataset!,
                                              write_manifest, read_manifest, download_datasets,
                                              pregenerate_dataset_manifest, DryRunValue,
                                              MANIFEST_FILENAME, manifest_path_in

using Downloads: Downloads
using Dates: DateTime
using TOML: TOML

struct FakeDataset end

NumericalEarth.DataWrangling.all_dates(::FakeDataset, ::Symbol) = [DateTime(2020, m, 1) for m in 1:12]
NumericalEarth.DataWrangling.build_filename(::FakeDataset, name, dates::AbstractArray, region) = "$(name).nc"
NumericalEarth.DataWrangling.build_filename(::FakeDataset, name, date, region) = "$(name).nc"
NumericalEarth.DataWrangling.metadata_filename(::FakeDataset, name, date, region) = "$(name)_$(date).nc"
NumericalEarth.DataWrangling.default_download_directory(::FakeDataset) = "/tmp/fake_dataset_test"
NumericalEarth.DataWrangling.first_date(::FakeDataset, ::Symbol) = DateTime(2020, 1, 1)

struct MockMetadatum <: AbstractMetadata
    name :: Symbol
end

const MOCK_DOWNLOAD_CALLS = Ref(0)
Downloads.download(::MockMetadatum) = (MOCK_DOWNLOAD_CALLS[] += 1; nothing)

@testset "AbstractMetadata supertype" begin
    @test Metadata <: AbstractMetadata
    @test Metadatum <: AbstractMetadata
    @test MetadataSet <: AbstractMetadata
end

@testset "parse_data_mode" begin
    @test parse_data_mode("auto")            == (:auto, "")
    @test parse_data_mode("")                == (:auto, "")
    @test parse_data_mode("strict")          == (:strict, "")
    @test parse_data_mode("pregenerate")     == (:pregenerate, "")
    @test parse_data_mode("pregenerate:/tmp/m")    == (:pregenerate, "/tmp/m")
    @test parse_data_mode("pregenerate:relative/dir") == (:pregenerate, "relative/dir")

    @test_throws ArgumentError parse_data_mode("pregenerate:")
    @test_throws ArgumentError parse_data_mode("garbage")
end

@testset "check_files_exist" begin
    mktempdir() do dir
        m_missing = Metadata(:t, nothing, nothing, nothing, dir, "missing_file.nc")
        @test_throws ErrorException DataModes.check_files_exist(m_missing)

        present_path = joinpath(dir, "present.nc")
        write(present_path, "x")
        m_present = Metadata(:t, nothing, nothing, nothing, dir, "present.nc")
        @test DataModes.check_files_exist(m_present) === nothing

        dates_vec = [DateTime(2020, 1, 1), DateTime(2020, 1, 2)]
        m_multi_missing = Metadata(:t, nothing, dates_vec, nothing, dir,
                                   DatewiseFilename(["a.nc", "b.nc"]))
        err = try
            DataModes.check_files_exist(m_multi_missing)
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("a.nc", sprint(showerror, err))
        @test occursin("b.nc", sprint(showerror, err))
    end
end

@testset "write_manifest groups by dataset" begin
    register_dataset!(FakeDataset, "FakeDataset")

    md_um  = Metadata(:bathymetry, FakeDataset(), nothing, nothing, "/tmp", "b.nc")
    md_one = Metadata(:temperature, FakeDataset(), DateTime(2020, 1, 1), nothing, "/tmp", "t.nc")
    dates  = [DateTime(2020, 1, 1), DateTime(2020, 12, 31)]
    md_range = Metadata(:salinity, FakeDataset(), dates, nothing, "/tmp",
                        DatewiseFilename(["s1.nc", "s2.nc"]))
    region = BoundingBox(longitude=(200.0, 220.0), latitude=(35.0, 55.0))
    md_region = Metadata(:eastward_velocity, FakeDataset(), DateTime(2020, 1, 1), region, "/tmp", "u.nc")
    mset = MetadataSet((:T, :S), FakeDataset(), dates, nothing, "/tmp",
                       (T = DatewiseFilename(["T1.nc", "T2.nc"]), S = DatewiseFilename(["S1.nc", "S2.nc"])))

    records = AbstractMetadata[md_um, md_one, md_range, md_region, mset]
    io = IOBuffer()
    write_manifest(io, records)
    parsed = TOML.parse(String(take!(io)))

    @test collect(keys(parsed)) == ["FakeDataset"]
    entries = parsed["FakeDataset"]
    @test length(entries) == 5
    @test all(!haskey(e, "dataset") for e in entries)

    @test any(e -> get(e, "variable_name", nothing) == "bathymetry" && !haskey(e, "date") && !haskey(e, "region"), entries)
    @test any(e -> get(e, "variable_name", nothing) == "temperature" && get(e, "date", nothing) == DateTime(2020, 1, 1), entries)
    @test any(e -> get(e, "variable_name", nothing) == "eastward_velocity"
                   && haskey(e, "region") && e["region"]["kind"] == "BoundingBox"
                   && e["region"]["longitude"] == [200.0, 220.0],
              entries)
    @test any(e -> get(e, "variable_name", nothing) == "salinity"
                   && get(e, "start_date", nothing) == DateTime(2020, 1, 1)
                   && get(e, "end_date",   nothing) == DateTime(2020, 12, 31),
              entries)
    @test any(e -> get(e, "variable_names", nothing) == ["T", "S"]
                   && get(e, "start_date", nothing) == DateTime(2020, 1, 1)
                   && get(e, "end_date",   nothing) == DateTime(2020, 12, 31),
              entries)

    col = Column(45.0, 30.0; z=(-400.0, 0.0), interpolation=Nearest())
    md_col = Metadata(:temperature, FakeDataset(), DateTime(2020, 1, 1), col, "/tmp", "t.nc")
    io2 = IOBuffer()
    write_manifest(io2, AbstractMetadata[md_col])
    parsed2 = TOML.parse(String(take!(io2)))
    col_entry = parsed2["FakeDataset"][1]
    @test col_entry["region"]["kind"]          == "Column"
    @test col_entry["region"]["longitude"]     == 45.0
    @test col_entry["region"]["latitude"]      == 30.0
    @test col_entry["region"]["interpolation"] == "Nearest"
end

@testset "read_manifest round-trip" begin
    register_dataset!(FakeDataset, "FakeDataset")

    md_one = Metadatum(:temperature; dataset=FakeDataset(), date=DateTime(2020, 6, 1))
    md_range = Metadata(:salinity; dataset=FakeDataset(),
                        start_date=DateTime(2020, 3, 1), end_date=DateTime(2020, 8, 1))
    region = BoundingBox(longitude=(200.0, 220.0), latitude=(35.0, 55.0))
    md_region = Metadatum(:temperature; dataset=FakeDataset(), date=DateTime(2020, 6, 1), region=region)
    mset = MetadataSet(:T, :S; dataset=FakeDataset(),
                       start_date=DateTime(2020, 3, 1), end_date=DateTime(2020, 8, 1))

    mktempdir() do dir
        path = manifest_path_in(dir)
        write_manifest(path, AbstractMetadata[md_one, md_range, md_region, mset])
        records = read_manifest(; dir)
        @test length(records) == 4

        rt = first(r for r in records if r isa Metadatum && r.name == :temperature && r.region === nothing)
        @test rt.dataset isa FakeDataset
        @test rt.dates == DateTime(2020, 6, 1)

        rrange = first(r for r in records if r isa Metadata && !(r isa Metadatum) && r.name == :salinity)
        @test rrange.dates == [DateTime(2020, m, 1) for m in 3:8]

        rregion = first(r for r in records if r isa Metadatum && r.region !== nothing)
        @test rregion.region isa BoundingBox
        @test rregion.region.longitude == (200.0, 220.0)
        @test rregion.region.latitude  == (35.0, 55.0)

        rset = first(r for r in records if r isa MetadataSet)
        @test rset.names == (:T, :S)
        @test rset.dates == [DateTime(2020, m, 1) for m in 3:8]
    end
end

@testset "download_datasets varargs and manifest dir" begin
    register_dataset!(FakeDataset, "FakeDataset")
    saved = DataModes.DATA_MODE[]
    REAL_DOWNLOAD_CALLS = Ref(0)
    Downloads.download(::Metadata{<:FakeDataset}) = (REAL_DOWNLOAD_CALLS[] += 1; nothing)
    try
        DataModes.DATA_MODE[] = :auto
        m1 = Metadatum(:temperature; dataset=FakeDataset(), date=DateTime(2020, 6, 1))
        m2 = Metadatum(:salinity; dataset=FakeDataset(), date=DateTime(2020, 7, 1))
        REAL_DOWNLOAD_CALLS[] = 0
        download_datasets(m1, m2)
        @test REAL_DOWNLOAD_CALLS[] == 2

        mktempdir() do dir
            write_manifest(manifest_path_in(dir), AbstractMetadata[m1, m2])
            REAL_DOWNLOAD_CALLS[] = 0
            download_datasets(; dir)
            @test REAL_DOWNLOAD_CALLS[] == 2
        end
    finally
        DataModes.DATA_MODE[] = saved
    end
end

@testset "observe_metadata hook fires inside library-style functions" begin
    register_dataset!(FakeDataset, "FakeDataset")
    saved = DataModes.DATA_MODE[]
    try
        DataModes.DATA_MODE[] = :pregenerate
        empty!(DataModes.RECORDED)

        library_constructor() = (Metadatum(:temperature; dataset=FakeDataset(), date=DateTime(2020, 6, 1)),
                                 Metadatum(:salinity;    dataset=FakeDataset(), date=DateTime(2020, 6, 1)))
        library_constructor()

        @test length(DataModes.RECORDED) == 2
        names = sort([String(r.name) for r in DataModes.RECORDED])
        @test names == ["salinity", "temperature"]
    finally
        DataModes.DATA_MODE[] = saved
        empty!(DataModes.RECORDED)
    end
end

@testset "DryRunValue minimal stub" begin
    v = DryRunValue()
    @test v.anything isa DryRunValue
    @test v.something_else isa DryRunValue
    @test sprint(show, v) == "DryRunValue()"
end

@testset "pregenerate_dataset_manifest append (overwrite_existing=false)" begin
    mktempdir() do dir
        manifest = manifest_path_in(dir)

        script_one = joinpath(dir, "one.jl")
        write(script_one, """
        using NumericalEarth
        using NumericalEarth.DataWrangling: Metadatum, download_dataset
        using NumericalEarth.DataWrangling.DataModes: register_dataset!
        using Dates: DateTime

        struct AppendDataset end
        NumericalEarth.DataWrangling.metadata_filename(::AppendDataset, name, date, region) = string(name, ".nc")
        NumericalEarth.DataWrangling.default_download_directory(::AppendDataset) = "/tmp"
        register_dataset!(AppendDataset, "AppendDataset")

        download_dataset(Metadatum(:temperature; dataset=AppendDataset(), date=DateTime(2020, 1, 1)))
        """)
        pregenerate_dataset_manifest(script_one; dir)
        parsed_after_one = TOML.parsefile(manifest)
        @test length(parsed_after_one["AppendDataset"]) == 1

        script_two = joinpath(dir, "two.jl")
        write(script_two, """
        using NumericalEarth
        using NumericalEarth.DataWrangling: Metadatum, download_dataset
        using NumericalEarth.DataWrangling.DataModes: register_dataset!
        using Dates: DateTime

        struct AppendDataset end
        NumericalEarth.DataWrangling.metadata_filename(::AppendDataset, name, date, region) = string(name, ".nc")
        NumericalEarth.DataWrangling.default_download_directory(::AppendDataset) = "/tmp"
        register_dataset!(AppendDataset, "AppendDataset")

        download_dataset(Metadatum(:salinity; dataset=AppendDataset(), date=DateTime(2020, 1, 1)))
        """)

        pregenerate_dataset_manifest(script_two; dir, overwrite_existing = false)
        parsed_after_two = TOML.parsefile(manifest)
        @test length(parsed_after_two["AppendDataset"]) == 2
        @test sort([e["variable_name"] for e in parsed_after_two["AppendDataset"]]) == ["salinity", "temperature"]

        pregenerate_dataset_manifest(script_two; dir, overwrite_existing = false)
        parsed_after_two_repeat = TOML.parsefile(manifest)
        @test length(parsed_after_two_repeat["AppendDataset"]) == 2

        pregenerate_dataset_manifest(script_two; dir, overwrite_existing = true)
        parsed_after_overwrite = TOML.parsefile(manifest)
        @test length(parsed_after_overwrite["AppendDataset"]) == 1
        @test parsed_after_overwrite["AppendDataset"][1]["variable_name"] == "salinity"
    end
end

@testset "pregenerate_dataset_manifest end-to-end" begin
    mktempdir() do dir
        script = joinpath(dir, "demo.jl")
        write(script, """
        using NumericalEarth
        using NumericalEarth.DataWrangling: Metadatum, download_dataset
        using NumericalEarth.DataWrangling.DataModes: register_dataset!
        using Dates: DateTime

        struct DemoDataset end
        NumericalEarth.DataWrangling.metadata_filename(::DemoDataset, name, date, region) = string(name, ".nc")
        NumericalEarth.DataWrangling.default_download_directory(::DemoDataset) = "/tmp"
        register_dataset!(DemoDataset, "DemoDataset")

        bad = something_undefined()

        function helper()
            x = bad.field
            y = download_dataset(Metadatum(:T; dataset=DemoDataset(), date=DateTime(2020, 1, 1)))
            z = download_dataset(Metadatum(:S; dataset=DemoDataset(), date=DateTime(2020, 1, 1)))
            return z
        end

        helper()
        """)
        pregenerate_dataset_manifest(script; dir)
        parsed = TOML.parsefile(manifest_path_in(dir))
        @test length(get(parsed, "DemoDataset", [])) == 2
        @test sort([e["variable_name"] for e in parsed["DemoDataset"]]) == ["S", "T"]
    end
end

@testset "download_dataset chokepoint" begin
    md = MockMetadatum(:t)
    saved_mode = DataModes.DATA_MODE[]
    try
        DataModes.DATA_MODE[] = :auto
        MOCK_DOWNLOAD_CALLS[] = 0
        download_dataset(md)
        @test MOCK_DOWNLOAD_CALLS[] == 1

        DataModes.DATA_MODE[] = :pregenerate
        MOCK_DOWNLOAD_CALLS[] = 0
        download_dataset(md)
        @test MOCK_DOWNLOAD_CALLS[] == 0

        DataModes.DATA_MODE[] = :strict
        @test_throws Exception download_dataset(md)
    finally
        DataModes.DATA_MODE[] = saved_mode
        empty!(DataModes.RECORDED)
    end
end
