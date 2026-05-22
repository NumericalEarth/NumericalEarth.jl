const MANIFEST_FILENAME = "NumericalEarthDataManifest.toml"
const MANIFEST_DIR      = Ref{String}("")
const RECORDED          = AbstractMetadata[]
const DATASET_REGISTRY  = Dict{String, Any}()

"""
    $(TYPEDSIGNATURES)

Return the absolute path to the data manifest inside `dir`. The basename is fixed
(`NumericalEarthDataManifest.toml`) to avoid name collisions with Pkg's `Project.toml` /
`Manifest.toml` and similar Julia-ecosystem files, and to give one canonical manifest per
directory — analogous to how each project directory has one `Project.toml`.
"""
manifest_path_in(dir::AbstractString) = joinpath(abspath(dir), MANIFEST_FILENAME)

"""
    $(TYPEDSIGNATURES)

Parse a `NUMERICALEARTH_DATA` value into a `(mode, dir)` tuple. `dir` is the directory the manifest
will be written to / read from; the filename is always `NumericalEarthDataManifest.toml`.

Recognized values:
- `""` or `"auto"`        → `(:auto, "")`
- `"strict"`              → `(:strict, "")`
- `"pregenerate"`         → `(:pregenerate, "")` — writes to the cwd at trace time
- `"pregenerate:<dir>"`   → `(:pregenerate, "<dir>")` — writes to `<dir>/NumericalEarthDataManifest.toml`

Throws `ArgumentError` on any other value.
"""
function parse_data_mode(s::AbstractString)
    (isempty(s) || s == "auto") && return (:auto, "")
    s == "strict" && return (:strict, "")
    s == "pregenerate" && return (:pregenerate, "")
    if startswith(s, "pregenerate:")
        dir = s[length("pregenerate:")+1:end]
        isempty(dir) && throw(ArgumentError("`NUMERICALEARTH_DATA=pregenerate:<dir>` requires a non-empty directory"))
        return (:pregenerate, dir)
    end
    throw(ArgumentError("Unrecognized NUMERICALEARTH_DATA value: $(repr(s)). Expected \"auto\", \"strict\", \"pregenerate\", or \"pregenerate:<dir>\"."))
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
    d = Dict{String, Any}("variable_name" => String(m.name))
    m.dates    === nothing || (d["date"]     = m.dates)
    m.region   === nothing || (d["region"]   = region_to_dict(m.region))
    m.filename === nothing || (d["filename"] = filename_to_toml(m.filename))
    return d
end

function metadata_to_dict(m::Metadata)
    d = Dict{String, Any}("variable_name" => String(m.name),
                          "start_date" => first(m.dates), "end_date" => last(m.dates))
    m.region   === nothing || (d["region"]   = region_to_dict(m.region))
    m.filename === nothing || (d["filename"] = filename_to_toml(m.filename))
    return d
end

function metadata_to_dict(mset::MetadataSet)
    d = Dict{String, Any}("variable_names" => [String(n) for n in mset.names])
    if mset.dates isa AbstractVector
        d["start_date"] = first(mset.dates)
        d["end_date"]   = last(mset.dates)
    elseif mset.dates !== nothing
        d["date"] = mset.dates
    end
    mset.region === nothing || (d["region"] = region_to_dict(mset.region))
    return d
end

"""
    $(TYPEDSIGNATURES)

Serialize `records` (a vector of `AbstractMetadata`) to `io` as a `NumericalEarthDataManifest.toml`,
with one table array per dataset:

```toml
[[ETOPO2022]]
variable_name = "bathymetry"

[[JRA55RepeatYear]]
variable_names = ["eastward_wind", "northward_wind", ...]
start_date     = "1990-01-01T00:00:00"
end_date       = "1990-12-31T18:00:00"

[[GLORYSDaily]]
variable_name = "temperature"
date          = "2020-06-15T00:00:00"
```

The download directory (`dir`) is not stored. The loader uses each dataset's default directory
unless overridden by `download_datasets(; dir=...)`.
"""
function write_manifest(io::IO, records::AbstractVector)
    grouped = Dict{String, Vector{Dict{String, Any}}}()
    for r in records
        entries = get!(() -> Dict{String, Any}[], grouped, dataset_name(r.dataset))
        push!(entries, metadata_to_dict(r))
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

function from_toml(dataset_name::AbstractString, entry::AbstractDict; download_dir = nothing)
    dataset  = lookup_dataset(dataset_name)
    region   = region_from_toml(get(entry, "region", nothing))
    filename = filename_from_toml(get(entry, "filename", nothing))
    dir      = download_dir === nothing ? default_download_directory(dataset) : String(download_dir)
    if haskey(entry, "variable_names")
        names = Tuple(Symbol(n) for n in entry["variable_names"])
        haskey(entry, "date") &&
            return MetadataSet(names...; dataset, region, dir, date = entry["date"])
        return MetadataSet(names...; dataset, region, dir,
                           start_date = entry["start_date"], end_date = entry["end_date"])
    end
    name = Symbol(entry["variable_name"])
    haskey(entry, "start_date") &&
        return Metadata(name; dataset, region, filename, dir,
                        start_date = entry["start_date"], end_date = entry["end_date"])
    return Metadatum(name; dataset, region, filename, dir, date = get(entry, "date", nothing))
end

"""
    $(TYPEDSIGNATURES)

Read the manifest at `joinpath(dir, "NumericalEarthDataManifest.toml")` and reconstruct every
record as the matching `Metadatum`/`Metadata`/`MetadataSet`. Datasets are looked up by name in
[`DATASET_REGISTRY`](@ref).

Pass `download_dir` to override every reconstructed record's download directory (useful when
login-node and compute-node filesystems differ); otherwise `default_download_directory(dataset)`
is used.
"""
function read_manifest(; dir::AbstractString = pwd(), download_dir = nothing)
    raw = TOML.parsefile(manifest_path_in(dir))
    return manifest_from_dict(raw; download_dir)
end

read_manifest(io::IO; download_dir = nothing) = manifest_from_dict(TOML.parse(read(io, String)); download_dir)

function manifest_from_dict(raw::AbstractDict; download_dir = nothing)
    records = AbstractMetadata[]
    for (name, entries) in raw
        for entry in entries
            push!(records, Base.invokelatest(from_toml, name, entry; download_dir))
        end
    end
    return records
end

"""
    $(TYPEDSIGNATURES)

Trace `script` in build-mode and write the resulting manifest to
`joinpath(dir, "NumericalEarthDataManifest.toml")`.

The script's source is parsed with `Meta.parseall`, every statement is wrapped in a per-statement
`try`/`catch` that rebinds failed assignments to [`DryRunValue`](@ref), and the rewritten code is
evaluated in a fresh sandbox module with `DATA_MODE[] = :pregenerate`. Each [`download_dataset`](@ref) call
records its metadata into [`RECORDED`](@ref) instead of downloading. The accumulated records are
then serialized via [`write_manifest`](@ref).

When `overwrite_existing = false` and a manifest already exists at `dir`, the existing records are
read first and merged (deduplicated) with the newly recorded ones, so this call appends rather
than replaces. Defaults to `true` (replace).
"""
function pregenerate_dataset_manifest(script::AbstractString;
                                dir::AbstractString = pwd(),
                                overwrite_existing::Bool = true)
    script_abs = abspath(script)
    source = read(script_abs, String)
    parsed = Meta.parseall(source; filename = script_abs)
    basedir = dirname(script_abs)
    rewritten = Expr(:toplevel, [rewrite_statement(a, basedir) for a in parsed.args]...)

    saved_mode    = DATA_MODE[]
    saved_records = copy(RECORDED)
    empty!(RECORDED)
    DATA_MODE[] = :pregenerate

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

    manifest = manifest_path_in(dir)
    if !overwrite_existing && isfile(manifest)
        for r in read_manifest(; dir)
            any(==(r), new_records) || pushfirst!(new_records, r)
        end
    end
    write_manifest(manifest, new_records)
    return manifest
end