mutable struct SharedTransportStatus{T}
    last_time :: Union{Nothing, T}
    untimed_components :: UInt8
    computations :: Int
end

struct RegriddedTransportCache{R, U, V, SU, SV, B, S}
    regridder :: R
    source_u :: U
    source_v :: V
    vertically_integrated_u :: SU
    vertically_integrated_v :: SV
    source_u_buffer :: B
    source_v_buffer :: B
    destination_u_buffer :: B
    destination_v_buffer :: B
    status :: S
end

const RegriddedUTransportOperation =
    RegriddedOperation{Face, Center, Nothing, G, T, S, D, R} where
        {G, T, S, D, R <: RegriddedTransportCache}

const RegriddedVTransportOperation =
    RegriddedOperation{Center, Face, Nothing, G, T, S, D, R} where
        {G, T, S, D, R <: RegriddedTransportCache}

function should_compute_transport!(status, time, component)
    if isnothing(time)
        component_already_served = !iszero(status.untimed_components & component)

        if component_already_served || iszero(status.untimed_components)
            status.untimed_components = component
            return true
        end

        status.untimed_components |= component
        return false
    end

    status.untimed_components = 0x00
    already_computed = status.last_time == time
    status.last_time = time
    return !already_computed
end

function compute_regridded_transport!(cache, time, component)
    should_compute_transport!(cache.status, time, component) || return nothing

    compute_at!(cache.vertically_integrated_u, time)
    compute_at!(cache.vertically_integrated_v, time)

    source_u = vec(Array(interior(cache.vertically_integrated_u)))
    source_v = vec(Array(interior(cache.vertically_integrated_v)))
    copyto!(cache.source_u_buffer, source_u)
    copyto!(cache.source_v_buffer, source_v)

    regridder = cache.regridder
    mul!(cache.destination_u_buffer, regridder.Wuu, cache.source_u_buffer)
    mul!(cache.destination_u_buffer, regridder.Wuv, cache.source_v_buffer, 1.0, 1.0)
    mul!(cache.destination_v_buffer, regridder.Wvu, cache.source_u_buffer)
    mul!(cache.destination_v_buffer, regridder.Wvv, cache.source_v_buffer, 1.0, 1.0)

    cache.status.computations += 1
    return nothing
end

function copy_regridded_transport!(field, buffer)
    Nx, Ny, _ = size(field)
    copyto!(interior(field), reshape(buffer, Nx, Ny, 1))
    fill_halo_regions!(field)
    return field
end

function Fields.compute_at!(operation::RegriddedUTransportOperation, time)
    cache = operation.regridder
    compute_regridded_transport!(cache, time, 0x01)
    copy_regridded_transport!(operation.destination, cache.destination_u_buffer)
    return nothing
end

function Fields.compute_at!(operation::RegriddedVTransportOperation, time)
    cache = operation.regridder
    compute_regridded_transport!(cache, time, 0x02)
    copy_regridded_transport!(operation.destination, cache.destination_v_buffer)
    return nothing
end

function transport_component_operation(destination::AbstractField{LX, LY, LZ, G, T},
                                       cache,
                                       source) where {LX, LY, LZ, G, T}
    S = typeof(source)
    D = typeof(destination)
    R = typeof(cache)
    return RegriddedOperation{LX, LY, LZ, G, T, S, D, R}(destination.grid,
                                                          source,
                                                          destination,
                                                          cache)
end

function Oceananigans.AbstractOperations.RegriddedOperation(
    source::NamedTuple{
        (:u, :v),
        <:Tuple{
            <:AbstractField{Face, Center, Center},
            <:AbstractField{Center, Face, Center},
        },
    },
    destination_grid;
    time = zero(eltype(source.u.grid)),
)
    source.u.grid === source.v.grid ||
        throw(ArgumentError("source.u and source.v must be defined on the same grid"))

    # Integrating first keeps the expensive three-dimensional fields on their
    # original architecture. Only two horizontal transport arrays move to CPU.
    # TODO: Move the reusable C-grid operation into Oceananigans if it gains an
    # optional geometry interface. NumericalEarth should then keep only this
    # diagnostic constructor.
    vertically_integrated_u = Field(Integral(source.u * Δy; dims=3); compute=false)
    vertically_integrated_v = Field(Integral(source.v * Δx; dims=3); compute=false)

    cpu_destination_grid = on_architecture(CPU(), destination_grid)
    regridder = Diagnostics.velocity_transport_regridder(cpu_destination_grid, source.u.grid)

    source_u_buffer = zeros(Float64, prod(regridder.source_u_size))
    source_v_buffer = zeros(Float64, prod(regridder.source_v_size))
    destination_u_buffer = zeros(Float64, prod(regridder.destination_u_size))
    destination_v_buffer = zeros(Float64, prod(regridder.destination_v_size))
    status = SharedTransportStatus{typeof(time)}(nothing, 0x00, 0)

    cache = RegriddedTransportCache(regridder,
                                    source.u,
                                    source.v,
                                    vertically_integrated_u,
                                    vertically_integrated_v,
                                    source_u_buffer,
                                    source_v_buffer,
                                    destination_u_buffer,
                                    destination_v_buffer,
                                    status)

    destination_u = Field{Face, Center, Nothing}(cpu_destination_grid)
    destination_v = Field{Center, Face, Nothing}(cpu_destination_grid)
    u = transport_component_operation(destination_u, cache, source)
    v = transport_component_operation(destination_v, cache, source)

    return (; u, v)
end

function Fields.Field(
    operations::NamedTuple{
        (:u, :v),
        <:Tuple{<:RegriddedUTransportOperation, <:RegriddedVTransportOperation},
    };
    kwargs...,
)
    return (; u=Field(operations.u; kwargs...),
              v=Field(operations.v; kwargs...))
end

function Diagnostics.regridded_transport_operation(source_u::AbstractField{Face, Center, Center},
                                                    source_v::AbstractField{Center, Face, Center},
                                                    destination_grid;
                                                    kwargs...)
    operations = RegriddedOperation((; u=source_u, v=source_v), destination_grid; kwargs...)
    return Field(operations)
end
