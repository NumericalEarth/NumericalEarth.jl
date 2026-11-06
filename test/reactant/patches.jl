# Temporary monkey patches for the Reactant tests, each standing in for an upstream fix that has not
# landed. Every patch names what retires it: when that lands, delete the patch and rerun the tests;
# delete this file when no patch is left.

using Reactant: Reactant
using Oceananigans: Oceananigans
using Oceananigans.Grids: Face, Center, λnodes, φnodes, hack_cosd
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.NestedModels: NestedModel, exchange_state!

# Guarded so the file can be included by more than one test of a session
if !(@isdefined ReactantNestedModel)
    const ReactantNestedModel = NestedModel{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ReactantState}
end

# PATCH 1. The nested step computes `Δt_parent = child.clock.time - parent.clock.time` and branches on it to
# tick the parent's clock: a host branch on a traced value once the child's clock is a device clock, or
# inside a traced loop over steps. The parent's clock only drives its geopotential interpolation and its
# series windows, both pinned when the exchanger's window holds every parent snapshot, so on
# ReactantState the step advances the child alone.
# Retire when: the parent follows the child's clock (NumericalEarth/NumericalEarth.jl#817).
function Oceananigans.TimeSteppers.time_step!(nm::ReactantNestedModel, Δt; callbacks = ())
    exchange_state!(nm.exchanger, nm.clock.time + Δt)
    time_step!(nm.child, Δt; callbacks = (callbacks..., nm.child_callbacks...))
    return nothing
end

# PATCH 2. The Reactant extension builds a lat-lon grid on the CPU and moves every coordinate array to the
# device. The plural node accessors are host conveniences built on views of those arrays: `xnodes`
# broadcasts `R deg2rad(λ') cos(φ)` and `ynodes` `R deg2rad(φ)`, element by element on the host, which the
# GPU refuses ("Scalar indexing is disallowed", e.g. Oceananigans's `Relaxation` checking its target's
# extent while a nested model is built). Here they work from host copies of the node arrays, one transfer
# each; traced arrays are left alone, so inside a program the broadcast traces.
# Retire when: OceananigansReactantExt defines host-returning `xnodes`/`ynodes` for its lat-lon grid
# (CliMA/Oceananigans.jl#6325).
if !(@isdefined ReactantLatLonGrid)
    const ReactantLatLonGrid = Base.get_extension(Oceananigans, :OceananigansReactantExt).Grids.ReactantStateLatitudeLongitudeGrid
end

host_nodes(a) = a
host_nodes(a::Reactant.AbstractConcreteArray) = Array(a)
host_nodes(a::Oceananigans.Grids.OffsetArray) = Oceananigans.Grids.OffsetArray(host_nodes(parent(a)), a.offsets)
host_nodes(v::SubArray) = view(host_nodes(parent(v)), v.indices...)

function Oceananigans.Grids.xnodes(grid::ReactantLatLonGrid, ℓx::Union{Face, Center}, ℓy::Union{Face, Center}; with_halos = false)
    λ = host_nodes(λnodes(grid, ℓx; with_halos))'
    φ = host_nodes(φnodes(grid, ℓy; with_halos))
    R = grid.radius
    return @. R * deg2rad(λ) * hack_cosd(φ)
end

function Oceananigans.Grids.ynodes(grid::ReactantLatLonGrid, ℓy::Union{Face, Center}; with_halos = false)
    φ = host_nodes(φnodes(grid, ℓy; with_halos))
    R = grid.radius
    return @. R * deg2rad(φ)
end

#####
##### Still to patch properly
#####

# TODO. `validate_bracket` (src/NestedModels/interpolated_fts_boundary.jl) reduces node views of the
# parent grid and a computed boundary line with `extrema`; on a GPU ReactantState grid those are views of
# device arrays, which reduce only element by element on the host. It runs under `@allowscalar` for now.
# The proper fix copies the nodes to the host in one transfer, as `host_nodes` (patch 2) does for
# `xnodes`/`ynodes` (NumericalEarth/NumericalEarth.jl#856).
