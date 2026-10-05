#####
##### GC-safe Zstd compression for JLD2
#####
#
# JLD2's `ZstdFilter` (JLD2 0.6.7, src/Filters.jl:288-318) compresses through ChunkCodecLibZstd's `try_encode!`
# (ChunkCodecLibZstd 1.0.0, src/encode.jl:76-118), a plain `ccall(:ZSTD_compress2, ...)`. While a plain ccall
# runs, its thread is not at a safepoint, so a stop-the-world GC requested by another thread waits for the whole
# compression (≈ 0.3 s for 256 MB at level 9). When `AsyncJLD2Writer`'s background task compresses, the
# time-stepping thread stalls in GC for that long.
#
# `GCSafeZstdFilter` does the same compression with `@ccall gc_safe=true` (Julia ≥ 1.12, base/c.jl:278-296),
# so the GC can run while libzstd works. It replicates `try_encode!` call for call (ZSTD_createCCtx,
# ZSTD_CCtx_setParameter(ZSTD_c_compressionLevel), ZSTD_compress2, ZSTD_freeCCtx, with the same level clamping
# as `ZstdEncodeOptions`, encode.jl:39-47), against the same libzstd (ChunkCodecLibZstd's `libzstd`, from
# Zstd_jll), so the compressed bytes are those `ZstdFilter` writes. It is written with `ZstdFilter`'s filter id
# (32015), name ("ZSTD") and client values (the level), so files are read back by plain JLD2 (which maps id
# 32015 to `ZstdFilter`, Filters.jl:303, 368-386) or h5py/HDF5 with the Zstd plugin, with no special code.
#
# Only the compression call is GC-safe: libzstd allocates with its own malloc and never calls back into Julia,
# and the source and destination buffers are rooted with `GC.@preserve` and passed as raw pointers.

using JLD2: JLD2, ZstdFilter

const ZstdCodecs = JLD2.Filters.ChunkCodecLibZstd
const libzstd = ZstdCodecs.libzstd # the library ZstdFilter uses (`using Zstd_jll: libzstd`, ChunkCodecLibZstd.jl:3)

"""
    GCSafeZstdFilter(; level = 3)
    GCSafeZstdFilter(filter::ZstdFilter)

A JLD2 compression filter that writes exactly what `JLD2.ZstdFilter(; level)` writes (same HDF5 filter id
32015, name and client values, same compressed bytes), but compresses in a `gc_safe` ccall so that a garbage
collection requested by another thread need not wait for the compression. Use it as `jld2_kw[:compress]`.
Decompression is `ZstdFilter`'s (on reading, JLD2 constructs a `ZstdFilter` from the filter id).
"""
struct GCSafeZstdFilter <: JLD2.Filters.Filter
    level :: Int32
    GCSafeZstdFilter(level) = new(ZstdFilter(level).level) # ZstdFilter's conversion: level % Int32, capped at ZSTD_maxCLevel()
end

GCSafeZstdFilter(; level::Integer = Int32(3)) = GCSafeZstdFilter(clamp(level, Int32)) # as ZstdFilter(; level), Filters.jl:294-298
GCSafeZstdFilter(filter::ZstdFilter) = GCSafeZstdFilter(filter.level)

# Written to the file exactly as ZstdFilter (Filters.jl:300-302)
JLD2.Filters.filterid(::Type{GCSafeZstdFilter}) = UInt16(32015)
JLD2.Filters.filtername(::Type{GCSafeZstdFilter}) = "ZSTD"
JLD2.Filters.client_values(filter::GCSafeZstdFilter) = (filter.level % UInt32,)

zstd_is_error(ret::Csize_t) = @ccall(libzstd.ZSTD_isError(ret::Csize_t)::Cuint) != 0
zstd_error_name(ret::Csize_t) = unsafe_string(@ccall libzstd.ZSTD_getErrorName(ret::Csize_t)::Ptr{Cchar})

function check_zstd(ret::Csize_t, what)
    zstd_is_error(ret) && error("libzstd error in $what: $(zstd_error_name(ret))")
    return ret
end

const ZSTD_c_compressionLevel = Cint(100) # zstd.h; as ChunkCodecLibZstd encode.jl:89

function gc_safe_zstd_compress(src::Vector{UInt8}, level::Int32)
    # ZstdEncodeOptions clamps the level to the library's range (encode.jl:45)
    level = clamp(level, ZstdCodecs.ZSTD_minCLevel(), ZstdCodecs.ZSTD_maxCLevel())
    src_size = length(src)
    bound = @ccall libzstd.ZSTD_compressBound(src_size::Csize_t)::Csize_t
    check_zstd(bound, "ZSTD_compressBound")
    dst = Vector{UInt8}(undef, bound)

    cctx = @ccall libzstd.ZSTD_createCCtx()::Ptr{Cvoid}
    cctx == C_NULL && throw(OutOfMemoryError())
    n = try
        check_zstd(@ccall(libzstd.ZSTD_CCtx_setParameter(cctx::Ptr{Cvoid}, ZSTD_c_compressionLevel::Cint, level::Cint)::Csize_t),
                   "ZSTD_CCtx_setParameter(compressionLevel = $level)")
        GC.@preserve src dst begin
            ret = @ccall gc_safe=true libzstd.ZSTD_compress2(cctx::Ptr{Cvoid},
                                                              pointer(dst)::Ptr{UInt8}, bound::Csize_t,
                                                              pointer(src)::Ptr{UInt8}, src_size::Csize_t)::Csize_t
        end
        check_zstd(ret, "ZSTD_compress2")
    finally
        @ccall libzstd.ZSTD_freeCCtx(cctx::Ptr{Cvoid})::Csize_t
    end

    return resize!(dst, n)
end

function JLD2.Filters.apply_filter!(filter::GCSafeZstdFilter, ref, forward::Bool = true,
                                    output_size::Union{Nothing, Integer} = nothing)
    forward || return JLD2.Filters.apply_filter!(ZstdFilter(filter.level), ref, false, output_size)
    src = ref[]
    src isa Vector{UInt8} || (src = Vector{UInt8}(src))
    ref[] = gc_safe_zstd_compress(src, filter.level)
    return 0
end
