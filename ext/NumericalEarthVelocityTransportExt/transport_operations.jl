mutable struct SharedTransportStatus{T}
    last_time :: Union{Nothing, T}
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

struct RegriddedUTransportOperand{C}
    cache :: C
end

struct RegriddedVTransportOperand{C}
    cache :: C
end

const RegriddedUTransportField = Field{Face, Center, Nothing, <:RegriddedUTransportOperand}
const RegriddedVTransportField = Field{Center, Face, Nothing, <:RegriddedVTransportOperand}

function compute_regridded_transport!(cache, time)
    already_computed = !isnothing(time) && cache.status.last_time == time
    already_computed && return nothing

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

    cache.status.last_time = time
    cache.status.computations += 1
    return nothing
end

function copy_regridded_transport!(field, buffer)
    Nx, Ny, _ = size(field)
    copyto!(interior(field), reshape(buffer, Nx, Ny, 1))
    fill_halo_regions!(field)
    return field
end

function Oceananigans.Fields.compute!(field::RegriddedUTransportField, time=nothing)
    cache = field.operand.cache
    compute_regridded_transport!(cache, time)
    return copy_regridded_transport!(field, cache.destination_u_buffer)
end

function Oceananigans.Fields.compute!(field::RegriddedVTransportField, time=nothing)
    cache = field.operand.cache
    compute_regridded_transport!(cache, time)
    return copy_regridded_transport!(field, cache.destination_v_buffer)
end

function regridded_transport_field(grid, operand, location)
    indices = (:, :, :)
    boundary_conditions = FieldBoundaryConditions(grid, location)
    data = new_data(grid, location, indices)
    return Field(location, grid, data, boundary_conditions, indices, operand, nothing)
end

function Diagnostics.regridded_transport_operation(source_u::AbstractField{Face, Center, Center},
                                                    source_v::AbstractField{Center, Face, Center},
                                                    destination_grid;
                                                    time=zero(eltype(source_u.grid)))
    source_u.grid === source_v.grid ||
        throw(ArgumentError("source_u and source_v must be defined on the same grid"))

    # Integrating first keeps the expensive three-dimensional fields on their
    # original architecture. Only two horizontal transport arrays move to CPU.
    # TODO: Move the reusable C-grid operation into Oceananigans if it gains an
    # optional geometry interface. NumericalEarth should then keep only this
    # diagnostic constructor.
    vertically_integrated_u = Field(Integral(source_u * Δy; dims=3); compute=false)
    vertically_integrated_v = Field(Integral(source_v * Δx; dims=3); compute=false)

    cpu_destination_grid = on_architecture(CPU(), destination_grid)
    regridder = Diagnostics.velocity_transport_regridder(cpu_destination_grid, source_u.grid)

    source_u_buffer = zeros(Float64, prod(regridder.source_u_size))
    source_v_buffer = zeros(Float64, prod(regridder.source_v_size))
    destination_u_buffer = zeros(Float64, prod(regridder.destination_u_size))
    destination_v_buffer = zeros(Float64, prod(regridder.destination_v_size))
    status = SharedTransportStatus{typeof(time)}(nothing, 0)

    cache = RegriddedTransportCache(regridder,
                                    source_u,
                                    source_v,
                                    vertically_integrated_u,
                                    vertically_integrated_v,
                                    source_u_buffer,
                                    source_v_buffer,
                                    destination_u_buffer,
                                    destination_v_buffer,
                                    status)

    u_operand = RegriddedUTransportOperand(cache)
    v_operand = RegriddedVTransportOperand(cache)
    u = regridded_transport_field(cpu_destination_grid, u_operand, (Face(), Center(), nothing))
    v = regridded_transport_field(cpu_destination_grid, v_operand, (Center(), Face(), nothing))

    return (; u, v)
end
