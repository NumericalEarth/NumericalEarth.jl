"""
    DataModes

Three-mode download dispatch and a declarative `DataManifest.toml` for NumericalEarth.
Modes are selected by the `NUMERICALEARTH_DATA` environment variable:

| Value             | Behavior                                                   |
|-------------------|------------------------------------------------------------|
| `"auto"` (default)| Download on demand (current behavior).                     |
| `"existing"`      | Error if any required file is missing. Never download.     |
| `"build:<path>"`  | Trace the running script; write a manifest to `<path>`.    |

See [`NumericalEarth.DataWrangling.download_dataset`](@ref) for the dispatch and [`build_dataset_manifest`](@ref) for the trace entry point.
"""
module DataModes

using DocStringExtensions: TYPEDSIGNATURES
using TOML: TOML

using ..DataWrangling: DataWrangling, AbstractMetadata, Metadata, Metadatum, MetadataSet, BoundingBox, Column, Linear, Nearest
using ..DataWrangling: DatewiseFilename, metadata_path, default_download_directory, download_dataset

export DryRunValue
export build_dataset_manifest, download_datasets
export register_dataset!

const DATA_MODE = Ref{Symbol}(:auto)

include("dry_run_value.jl")
include("data_manifest_wrangling.jl")
include("parse_and_rewrite_script.jl")

DataWrangling.observe_metadata(m::Metadata)    = (DATA_MODE[] === :build && record_for_manifest(m); nothing)
DataWrangling.observe_metadata(m::MetadataSet) = (DATA_MODE[] === :build && record_for_manifest(m); nothing)

"""
    $(TYPEDSIGNATURES)

Acquire every dataset listed in `metadata...` (varargs form) or in the manifest at `path`
(file-path form). Each entry is routed through [`download_dataset`](@ref), so the current
`NUMERICALEARTH_DATA` mode applies.

For the file-path form, pass `dir` to override the default download directory for every reconstructed
entry (e.g. when login-node and compute-node filesystems differ).
"""
function download_datasets(metadata::AbstractMetadata...)
    foreach(download_dataset, metadata)
    return nothing
end

function download_datasets(path::AbstractString; dir = nothing)
    foreach(download_dataset, read_manifest(path; dir))
    return nothing
end

expected_paths(metadata::Metadatum) = String[metadata_path(metadata)]

function expected_paths(metadata::Metadata)
    p = metadata_path(metadata)
    return p isa Vector ? collect(String, p) : String[p]
end

function expected_paths(mset::MetadataSet)
    paths = String[]
    for name in mset.names
        append!(paths, expected_paths(mset[name]))
    end
    return paths
end

"""
    $(TYPEDSIGNATURES)

Verify that every file required by `metadata` is already on disk. Raises a single error listing
every missing file. Returns `nothing` on success.
"""
function check_files_exist(metadata::AbstractMetadata)
    paths = expected_paths(metadata)
    missing_paths = filter(p -> !isfile(p), paths)
    isempty(missing_paths) && return nothing
    list = join(("  " * p for p in missing_paths), "\n")
    error("NUMERICALEARTH_DATA=existing: $(length(missing_paths)) required file(s) missing:\n$list")
end

function __init__()
    env = get(ENV, "NUMERICALEARTH_DATA", "auto")
    mode, path = parse_data_mode(env)
    DATA_MODE[] = mode
    MANIFEST_PATH[] = path
    mode === :build || return nothing

    if !isempty(Base.PROGRAM_FILE)
        script = abspath(Base.PROGRAM_FILE)
        atexit() do
            try
                build_dataset_manifest(script; manifest = MANIFEST_PATH[])
                @info "NUMERICALEARTH_DATA=build: wrote manifest via AST trace" path=MANIFEST_PATH[] script
            catch err
                @error "NUMERICALEARTH_DATA=build: trace failed" path=MANIFEST_PATH[] script exception=(err, catch_backtrace())
            end
        end
    else
        atexit() do
            try
                write_manifest(MANIFEST_PATH[], copy(RECORDED))
                @info "NUMERICALEARTH_DATA=build: wrote manifest" path=MANIFEST_PATH[] entries=length(RECORDED)
            catch err
                @error "NUMERICALEARTH_DATA=build: failed to write manifest" path=MANIFEST_PATH[] exception=(err, catch_backtrace())
            end
        end
    end
    return nothing
end

end # module
