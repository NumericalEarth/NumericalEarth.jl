#####
##### Cache for `Field(metadatum, grid)` reads — the dataset-agnostic analog of the
##### regridded-bathymetry cache. The key covers everything that determines the
##### result: the dataset (with its parameters), variable, date, region, the target
##### grid geometry, the forwarded read keywords, and a size/mtime stamp of the
##### local dataset file so a re-download invalidates the cache.
#####

struct FieldRegridding
    grid_type     :: String
    grid_size     :: NTuple{3, Int}
    longitude     :: Tuple{Float64, Float64}
    latitude      :: Tuple{Float64, Float64}
    z_extent      :: Union{Nothing, Tuple{Float64, Float64}}
    topology      :: NTuple{3, Symbol}
    float_type    :: Symbol
    location      :: NTuple{3, Symbol}
    dataset       :: String
    variable      :: Symbol
    date          :: String
    region        :: String
    read_keywords :: String
    file_stamp    :: Union{Nothing, Tuple{Int, Int}}
end

# The source stamp is best-effort: datasets that stream (Zarr, /vsicurl/ COGs) or
# span several files have no single local path, and their caches are keyed by the
# metadata alone.
function field_cache_file_stamp(metadatum)
    path = try
        metadata_path(metadatum)
    catch
        return nothing
    end
    path isa AbstractString && isfile(path) || return nothing
    return (Int(filesize(path)), round(Int, mtime(path)))
end

function FieldRegridding(grid, metadatum, read_keywords)
    Nx, Ny, Nz = size(grid)
    TX, TY, TZ = topology(grid)
    lon = x_domain(grid)
    lat = y_domain(grid)
    z_extent = TZ === Flat ? nothing : Float64.(z_domain(grid))
    LX, LY, LZ = location(metadatum)

    return FieldRegridding(string(typeof(grid).name.wrapper),
                           (Nx, Ny, Nz),
                           (Float64(lon[1]), Float64(lon[2])),
                           (Float64(lat[1]), Float64(lat[2])),
                           z_extent,
                           (Symbol(TX), Symbol(TY), Symbol(TZ)),
                           Symbol(eltype(grid)),
                           (Symbol(LX), Symbol(LY), Symbol(LZ)),
                           sprint(show, metadatum.dataset),
                           metadatum.name,
                           string(metadatum.dates),
                           bounding_box_suffix(metadatum.region),
                           sprint(show, read_keywords),
                           field_cache_file_stamp(metadatum))
end

function Base.:(==)(a::FieldRegridding, b::FieldRegridding)
    return all(getfield(a, name) == getfield(b, name) for name in fieldnames(FieldRegridding))
end

function Base.hash(c::FieldRegridding, h::UInt)
    for name in fieldnames(FieldRegridding)
        h = hash(getfield(c, name), h)
    end
    return h
end

function field_cache_filename(config::FieldRegridding)
    Nx, Ny, Nz = config.grid_size
    h = string(hash(config) % UInt32, base = 16, pad = 8)
    return "field_$(config.variable)_$(Nx)x$(Ny)x$(Nz)_$(h).jld2"
end

field_cache_path(config::FieldRegridding) =
    joinpath(download_cache("field_cache"), field_cache_filename(config))

function load_field_cache(config::FieldRegridding)
    filepath = field_cache_path(config)
    isfile(filepath) || return nothing
    try
        jldopen(filepath, "r") do file
            if file["config"] == config
                @info "Loading cached field from $filepath"
                return file["data"]
            else
                return nothing
            end
        end
    catch err
        @warn "Failed to load field cache from $filepath: $err"
        return nothing
    end
end

function save_field_cache(config::FieldRegridding, data::AbstractArray)
    filepath = field_cache_path(config)
    try
        jldopen(filepath, "w") do file
            file["config"] = config
            file["data"] = data
        end
        @info "Saved field cache to $filepath"
    catch err
        @warn "Failed to save field cache to $filepath: $err"
    end
    return nothing
end
