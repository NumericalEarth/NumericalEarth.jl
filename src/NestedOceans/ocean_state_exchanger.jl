#####
##### OceanStateExchanger: parent-derived state a nested ocean child needs each step
#####
#
# The parent hands over the child's own variables (u, v, T, S, η), so — unlike the atmosphere nest —
# there is no thermodynamic combine. What the parent does NOT carry is the barotropic transport the
# Flather (`GravityWaveRadiation`) condition needs. The exchanger builds it, per side, by integrating
# the interpolated parent velocity over the CHILD's column: integrating over the parent's column
# instead would prescribe a transport that does not fit the child's bathymetry, and the domain would
# leak volume.
#
# `GravityWaveRadiation`'s exterior value is a 2-tuple whose elements may be arrays, so each side's
# condition is literally `(Uᵉˣᵗ, ηᵉˣᵗ)` — two slabs this exchanger refreshes.

struct OceanStateExchanger{P, V, B, G}
    parent_ocean :: P   # the parent whose series drive the child (e.g. a volumetric `PrescribedOcean`)
    variables    :: V   # (u, v, T, S, η) drawn from the parent by `parent_ocean_variables`
    boundaries   :: B   # NamedTuple keyed by side: (U = array, η = array) on the child boundary
    grid         :: G   # the child grid
end

@inline boundary_index(::Val{:west},  N) = 1
@inline boundary_index(::Val{:south}, N) = 1
@inline boundary_index(::Val{:east},  N) = N + 1
@inline boundary_index(::Val{:north}, N) = N + 1

@inline normal_dimension(::Union{Val{:west}, Val{:east}}) = 1
@inline normal_dimension(::Union{Val{:south}, Val{:north}}) = 2

@kernel function _compute_zonal_boundary_transport!(U, η, iᵇ, grid, uᵖ, uᵍ, ηᵖ, ηᵍ, parent_location, surface_location, t)
    j = @index(Global, Linear)
    Nz = size(grid, 3)

    Σ = zero(grid)
    for k in 1:Nz
        X = node(iᵇ, j, k, grid, Face(), Center(), Center())
        uₚ = interpolate(X, Time(t), uᵖ, parent_location, uᵍ)
        active = !inactive_node(iᵇ, j, k, grid, Face(), Center(), Center())
        Σ += ifelse(active, Δzᶠᶜᶜ(iᵇ, j, k, grid) * uₚ, zero(grid))
    end

    Xη = node(iᵇ, j, 1, grid, Face(), Center(), Center())

    @inbounds U[j, 1, 1] = Σ
    @inbounds η[j, 1, 1] = interpolate(Xη, Time(t), ηᵖ, surface_location, ηᵍ)
end

@kernel function _compute_meridional_boundary_transport!(V, η, jᵇ, grid, vᵖ, vᵍ, ηᵖ, ηᵍ, parent_location, surface_location, t)
    i = @index(Global, Linear)
    Nz = size(grid, 3)

    Σ = zero(grid)
    for k in 1:Nz
        X = node(i, jᵇ, k, grid, Center(), Face(), Center())
        vₚ = interpolate(X, Time(t), vᵖ, parent_location, vᵍ)
        active = !inactive_node(i, jᵇ, k, grid, Center(), Face(), Center())
        Σ += ifelse(active, Δzᶜᶠᶜ(i, jᵇ, k, grid) * vₚ, zero(grid))
    end

    Xη = node(i, jᵇ, 1, grid, Center(), Face(), Center())

    @inbounds V[i, 1, 1] = Σ
    @inbounds η[i, 1, 1] = interpolate(Xη, Time(t), ηᵖ, surface_location, ηᵍ)
end

function boundary_transport_arrays(grid, side)
    FT = eltype(grid)
    Nx, Ny, _ = size(grid)
    N = normal_dimension(Val(side)) == 1 ? Ny : Nx
    arch = Architectures.architecture(grid)
    U = on_architecture(arch, zeros(FT, N, 1, 1))
    η = on_architecture(arch, zeros(FT, N, 1, 1))
    return (; U, η)
end

"""
$(TYPEDSIGNATURES)

Build the state exchanger of a nested ocean: the `parent_ocean` series plus, for every side in
`sides`, the `(Uᵉˣᵗ, ηᵉˣᵗ)` slabs that drive the child's Flather boundary condition there.
"""
function ocean_state_exchanger(parent_ocean, child_grid; sides = (:west, :east, :south, :north))
    boundaries = NamedTuple(side => boundary_transport_arrays(child_grid, side) for side in sides)
    return OceanStateExchanger(parent_ocean, parent_ocean_variables(parent_ocean), boundaries, child_grid)
end

function refresh_boundary_transport!(exchanger, side, time)
    grid = exchanger.grid
    Nx, Ny, _ = size(grid)
    slabs = getproperty(exchanger.boundaries, side)
    η = exchanger.variables.η
    surface_location = instantiated_location(η)
    arch = Architectures.architecture(grid)

    if normal_dimension(Val(side)) == 1
        u = exchanger.variables.u
        iᵇ = boundary_index(Val(side), Nx)
        launch!(arch, grid, tuple(Ny), _compute_zonal_boundary_transport!,
                slabs.U, slabs.η, iᵇ, grid, u, u.grid, η, η.grid,
                instantiated_location(u), surface_location, time)
    else
        v = exchanger.variables.v
        jᵇ = boundary_index(Val(side), Ny)
        launch!(arch, grid, tuple(Nx), _compute_meridional_boundary_transport!,
                slabs.U, slabs.η, jᵇ, grid, v, v.grid, η, η.grid,
                instantiated_location(v), surface_location, time)
    end

    return nothing
end

# Advance the parent windows, then rebuild every side's barotropic exterior from them.
function NestedModels.exchange_state!(exchanger::OceanStateExchanger, time)
    refresh_parent_state!(exchanger.parent_ocean, Time(time))

    for side in keys(exchanger.boundaries)
        refresh_boundary_transport!(exchanger, side, time)
    end

    return nothing
end

Base.summary(exchanger::OceanStateExchanger) =
    string("OceanStateExchanger(sides=", keys(exchanger.boundaries), ")")

function Base.show(io::IO, exchanger::OceanStateExchanger)
    print(io, summary(exchanger), '\n',
              "├── parent: ", summary(exchanger.parent_ocean), '\n',
              "└── grid: ", summary(exchanger.grid))
end
