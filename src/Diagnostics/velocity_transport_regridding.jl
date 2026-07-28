using DocStringExtensions: TYPEDSIGNATURES

"""
$(TYPEDSIGNATURES)

Construct the horizontal line-intersection operator used to conservatively
map a pair of C-grid face transports to another grid.

This method is supplied by the optional velocity-transport extension. Load
`LibGEOS` before calling it.
"""
function velocity_transport_regridder(args...; kwargs...)
    throw(ArgumentError("velocity transport regridding requires `using LibGEOS`"))
end

"""
$(TYPEDSIGNATURES)

Return lazy destination `u` and `v` transport fields for a pair of native
C-grid face-normal flux densities. The vertical integral is evaluated before
the two-dimensional conservative remap.

This method is supplied by the optional velocity-transport extension. Load
`LibGEOS` before calling it.
"""
function regridded_transport_operation(args...; kwargs...)
    throw(ArgumentError("lazy velocity transport regridding requires `using LibGEOS`"))
end
