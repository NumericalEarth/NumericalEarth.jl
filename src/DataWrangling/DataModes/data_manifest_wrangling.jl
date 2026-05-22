const MANIFEST_PATH    = Ref{String}("")
const RECORDED         = AbstractMetadata[]
const DATASET_REGISTRY = Dict{String, Any}()

"""
    $(TYPEDSIGNATURES)

Parse a `NUMERICALEARTH_DATA` value into a `(mode, manifest_path)` tuple.

Recognized values:
- `""` or `"auto"`     → `(:auto, "")`
- `"existing"`         → `(:existing, "")`
- `"build:<path>"`     → `(:build, "<path>")`

Throws `ArgumentError` on any other value or on `"build:"` without a path.
"""
function parse_data_mode(s::AbstractString)
    (isempty(s) || s == "auto") && return (:auto, "")
    s == "existing" && return (:existing, "")
    if startswith(s, "build:")
        path = s[length("build:")+1:end]
        isempty(path) && throw(ArgumentError("`NUMERICALEARTH_DATA=build:<path>` requires a non-empty manifest path"))
        return (:build, path)
    end
    throw(ArgumentError("Unrecognized NUMERICALEARTH_DATA value: $(repr(s)). Expected \"auto\", \"existing\", or \"build:<path>\"."))
end

"""
    $(TYPEDSIGNATURES)

Record `metadata` into [`RECORDED`](@ref) for later serialization to a `DataManifest.toml`. Deduplication
is by `metadata` equality on the recorded vector. Returns `nothing`.
"""
function record_for_manifest(metadata::AbstractMetadata)
    any(==(metadata), RECORDED) || push!(RECORDED, metadata)
    return nothing
end

"""
    $(TYPEDSIGNATURES)

Register a dataset constructor under a string name so that the manifest writer can serialize it
(`dataset = "Name"`) and the loader can reconstruct it via `DATASET_REGISTRY[name]()`. Idempotent.
"""
function register_dataset!(constructor, name::AbstractString)
    DATASET_REGISTRY[String(name)] = constructor
    return nothing
end

function dataset_name(d)
    T = typeof(d)
    for (name, ctor) in DATASET_REGISTRY
        ctor === T && return name
    end
    return string(nameof(T))
end

region_to_dict(::Nothing) = nothing

function region_to_dict(bb::BoundingBox)
    d = Dict{String, Any}("kind" => "BoundingBox")
    bb.longitude === nothing || (d["longitude"] = collect(bb.longitude))
    bb.latitude  === nothing || (d["latitude"]  = collect(bb.latitude))
    bb.z         === nothing || (d["z"]         = collect(bb.z))
    return d
end

function region_to_dict(col::Column)
    interp = col.interpolation isa Nearest ? "Nearest" : "Linear"
    d = Dict{String, Any}("kind" => "Column", "longitude" => col.longitude, "latitude" => col.latitude, "interpolation" => interp)
    col.z === nothing || (d["z"] = collect(col.z))
    return d
end

filename_to_toml(::Nothing) = nothing
filename_to_toml(s::AbstractString) = String(s)
filename_to_toml(f::DatewiseFilename) = collect(String, f.filenames)

function metadata_to_dict(m::Metadatum)
    d = Dict{String, Any}("variable_name" => String(m.name), "dataset" => dataset_name(m.dataset))
    m.dates    === nothing || (d["date"]     = m.dates)
    m.region   === nothing || (d["region"]   = region_to_dict(m.region))
    m.filename === nothing || (d["filename"] = filename_to_toml(m.filename))
    return d
end

function metadata_to_dict(m::Metadata)
    d = Dict{String, Any}("variable_name" => String(m.name), "dataset" => dataset_name(m.dataset),
                          "start_date" => first(m.dates), "end_date" => last(m.dates))
    m.region   === nothing || (d["region"]   = region_to_dict(m.region))
    m.filename === nothing || (d["filename"] = filename_to_toml(m.filename))
    return d
end

function metadata_to_dict(mset::MetadataSet)
    d = Dict{String, Any}("variable_names" => [String(n) for n in mset.names], "dataset" => dataset_name(mset.dataset))
    if mset.dates isa AbstractVector
        d["start_date"] = first(mset.dates)
        d["end_date"]   = last(mset.dates)
    elseif mset.dates !== nothing
        d["date"] = mset.dates
    end
    mset.region === nothing || (d["region"] = region_to_dict(mset.region))
    return d
end

manifest_table_key(::Metadatum)    = "metadatum"
manifest_table_key(::Metadata)     = "metadata"
manifest_table_key(::MetadataSet)  = "metadataset"

"""
    $(TYPEDSIGNATURES)

Serialize `records` (a vector of `AbstractMetadata`) to `io` as a `DataManifest.toml` with three
table arrays: `[[metadatum]]`, `[[metadata]]`, `[[metadataset]]`.

The download directory (`dir`) is not stored. The loader uses each dataset's default directory
unless overridden by `download_datasets(...; dir=...)`.
"""
function write_manifest(io::IO, records::AbstractVector)
    grouped = Dict{String, Vector{Dict{String, Any}}}("metadatum"   => [],
                                                      "metadata"    => [],
                                                      "metadataset" => [])
    for r in records
        push!(grouped[manifest_table_key(r)], metadata_to_dict(r))
    end
    for k in ("metadatum", "metadata", "metadataset")
        isempty(grouped[k]) && delete!(grouped, k)
    end
    TOML.print(io, grouped)
    return nothing
end

function write_manifest(path::AbstractString, records::AbstractVector)
    open(io -> write_manifest(io, records), path, "w")
    return nothing
end

#####
##### filename and `region` reconstruction
#####

region_from_toml(::Nothing) = nothing

function region_from_toml(d::AbstractDict)
    kind = d["kind"]
    if kind == "BoundingBox"
        longitude = haskey(d, "longitude") ? Tuple(d["longitude"]) : nothing
        latitude  = haskey(d, "latitude")  ? Tuple(d["latitude"])  : nothing
        z         = haskey(d, "z")         ? Tuple(d["z"])         : nothing
        return BoundingBox(; longitude, latitude, z)
    elseif kind == "Column"
        z = haskey(d, "z") ? Tuple(d["z"]) : nothing
        interpolation = get(d, "interpolation", "Linear") == "Nearest" ? Nearest() : Linear()
        return Column(d["longitude"], d["latitude"]; z, interpolation)
    else
        throw(ArgumentError("Unknown region kind: $(repr(kind))"))
    end
end

filename_from_toml(::Nothing) = nothing
filename_from_toml(s::AbstractString) = String(s)
filename_from_toml(v::AbstractVector) = DatewiseFilename(collect(String, v))

function lookup_dataset(name::AbstractString)
    haskey(DATASET_REGISTRY, name) ||
        throw(ArgumentError("Unknown dataset $(repr(name)). Did you `using` the dataset module so its __init__ runs and registers it?"))
    return Base.invokelatest(DATASET_REGISTRY[name])
end

#####
##### AbstractMetadata reconstruction
#####

function from_toml(kind::Symbol, entry::AbstractDict; dir = nothing)
    dataset  = lookup_dataset(entry["dataset"])
    region   = region_from_toml(get(entry, "region", nothing))
    filename = filename_from_toml(get(entry, "filename", nothing))
    download_dir = dir === nothing ? default_download_directory(dataset) : String(dir)
    if kind === :metadatum
        name = Symbol(entry["variable_name"])
        return Metadatum(name; dataset, region, filename, dir = download_dir, date = get(entry, "date", nothing))
    elseif kind === :metadata
        name = Symbol(entry["variable_name"])
        return Metadata(name; dataset, region, filename, dir = download_dir,
                        start_date = entry["start_date"], end_date = entry["end_date"])
    elseif kind === :metadataset
        names = Tuple(Symbol(n) for n in entry["variable_names"])
        haskey(entry, "date") &&
            return MetadataSet(names...; dataset, region, dir = download_dir, date = entry["date"])
        return MetadataSet(names...; dataset, region, dir = download_dir,
                           start_date = entry["start_date"], end_date = entry["end_date"])
    else
        throw(ArgumentError("Unknown manifest record kind: $(repr(kind))"))
    end
end

"""
    $(TYPEDSIGNATURES)

Read a `DataManifest.toml` and reconstruct every record as the matching `Metadatum`/`Metadata`/`MetadataSet`.
Datasets are looked up by name in [`DATASET_REGISTRY`](@ref).

Pass `dir` to override every reconstructed record's download directory (useful when login-node and
compute-node filesystems differ); otherwise `default_download_directory(dataset)` is used.
"""
function read_manifest(path::AbstractString; dir = nothing)
    raw = TOML.parsefile(path)
    return manifest_from_dict(raw; dir)
end

function read_manifest(io::IO; dir = nothing)
    raw = TOML.parse(read(io, String))
    return manifest_from_dict(raw; dir)
end

function manifest_from_dict(raw::AbstractDict; dir = nothing)
    records = AbstractMetadata[]
    for k in (:metadatum, :metadata, :metadataset)
        haskey(raw, String(k)) || continue
        for entry in raw[String(k)]
            push!(records, Base.invokelatest(from_toml, k, entry; dir))
        end
    end
    return records
end

"""
    $(TYPEDSIGNATURES)

Trace `script` in build-mode and write the resulting `DataManifest.toml` to `manifest`.

The script's source is parsed with `Meta.parseall`, every statement is wrapped in a per-statement
`try`/`catch` that rebinds failed assignments to [`DryRunValue`](@ref), and the rewritten code is
evaluated in a fresh sandbox module with `DATA_MODE[] = :build`. Each [`download_dataset`](@ref) call
records its metadata into [`RECORDED`](@ref) instead of downloading. The accumulated records are
then serialized via [`write_manifest`](@ref).

When `overwrite_existing = false` and `manifest` already exists, the existing records are read first
and merged (deduplicated) with the newly recorded ones, so this call appends rather than replaces.
Defaults to `true` (replace).
"""
function build_dataset_manifest(script::AbstractString;
                                manifest::AbstractString = "DataManifest.toml",
                                overwrite_existing::Bool = true)
    script_abs = abspath(script)
    source = read(script_abs, String)
    parsed = Meta.parseall(source; filename = script_abs)
    basedir = dirname(script_abs)
    rewritten = Expr(:toplevel, [rewrite_statement(a, basedir) for a in parsed.args]...)

    saved_mode    = DATA_MODE[]
    saved_records = copy(RECORDED)
    empty!(RECORDED)
    DATA_MODE[] = :build

    new_records = AbstractMetadata[]
    try
        sandbox = Module(:DataModesSandbox)
        Core.eval(sandbox, :(eval(x)    = Core.eval($sandbox, x)))
        Core.eval(sandbox, :(include(p) = Base.include($sandbox, p)))
        Core.eval(sandbox, rewritten)
        new_records = copy(RECORDED)
    finally
        DATA_MODE[] = saved_mode
        empty!(RECORDED)
        append!(RECORDED, saved_records)
    end

    if !overwrite_existing && isfile(manifest)
        for r in read_manifest(manifest)
            any(==(r), new_records) || pushfirst!(new_records, r)
        end
    end
    write_manifest(manifest, new_records)
    return manifest
end