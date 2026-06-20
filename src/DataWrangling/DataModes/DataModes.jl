"""
    DataModes

Three-mode download dispatch and a declarative `NumericalEarthDataManifest.toml` for NumericalEarth.
Modes are selected by the `NUMERICALEARTH_DATA` environment variable:

| Value             | Behavior                                                                   |
|-------------------|----------------------------------------------------------------------------|
| `"auto"` (default)| Download on demand (current behavior).                                     |
| `"strict"`        | Error if any required file is missing. Never download.                     |
| `"pregenerate"`   | Trace the running script; write the manifest to `pwd()`.                   |
| `"pregenerate:<dir>"` | Same as `"pregenerate"` but write to `<dir>/NumericalEarthDataManifest.toml`. |

The filename is fixed (`NumericalEarthDataManifest.toml`) so manifests don't collide with Pkg's
`Project.toml` / `Manifest.toml` and there is one canonical manifest per directory.

See [`NumericalEarth.DataWrangling.download_dataset`](@ref) for the dispatch and
[`pregenerate_dataset_manifest`](@ref) for the trace entry point.
"""
module DataModes

using DocStringExtensions: TYPEDSIGNATURES
using TOML: TOML

using ..DataWrangling: DataWrangling, AbstractMetadata, Metadata, Metadatum, MetadataSet, BoundingBox, Column, Linear, Nearest
using ..DataWrangling: DatewiseFilename, metadata_path, default_download_directory, download_dataset

export DryRunValue
export pregenerate_dataset_manifest, download_datasets
export register_dataset!

const DATA_MODE = Ref{Symbol}(:auto)

include("dry_run_value.jl")
include("data_manifest_wrangling.jl")
include("parse_and_rewrite_script.jl")

DataWrangling.observe_metadata(m::Metadata)    = (DATA_MODE[] === :pregenerate && record_for_manifest(m); nothing)
DataWrangling.observe_metadata(m::MetadataSet) = (DATA_MODE[] === :pregenerate && record_for_manifest(m); nothing)

"""
    $(TYPEDSIGNATURES)

Acquire every dataset listed in `metadata...` (varargs form) or in the manifest at
`joinpath(dir, "NumericalEarthDataManifest.toml")` (zero-arg form). Each entry is routed through
[`download_dataset`](@ref), so the current `NUMERICALEARTH_DATA` mode applies.

For the manifest form, `dir` is the directory containing the manifest (defaults to `pwd()`). Pass
`download_dir` to override the default download directory for every reconstructed entry (e.g. when
login-node and compute-node filesystems differ).
"""
function download_datasets(metadata::AbstractMetadata...)
    foreach(download_dataset, metadata)
    return nothing
end

function download_datasets(; dir::AbstractString = pwd(), download_dir = nothing)
    foreach(download_dataset, read_manifest(; dir, download_dir))
    return nothing
end

function expected_paths(m::AbstractMetadata)
    m isa MetadataSet &&
        return reduce(vcat, expected_paths(m[n]) for n in m.names; init = String[])
    p = metadata_path(m)
    return p isa AbstractVector ? collect(String, p) : String[p]
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
    error("NUMERICALEARTH_DATA=strict: $(length(missing_paths)) required file(s) missing:\n$list")
end

function __init__()
    env = get(ENV, "NUMERICALEARTH_DATA", "auto")
    mode, dir_from_env = parse_data_mode(env)
    DATA_MODE[] = mode
    MANIFEST_DIR[] = isempty(dir_from_env) ? pwd() : abspath(dir_from_env)
    return nothing
end

end # module
