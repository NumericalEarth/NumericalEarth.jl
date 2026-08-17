#####
##### Advective bottom boundary layer (Campin & Goosse 1999)
#####
#
# Ported from NEMO `src/OCE/TRA/trabbl.F90`, `tra_bbl_adv` with `nn_bbl_adv = 2`.
#
# The diffusive scheme in `bottom_boundary_layer.jl` exchanges tracer between the two bottom cells
# either side of a step. A symmetric exchange drives both cells to their volume-weighted mean, so the
# density delivered downslope is capped by the volume ratio of the two cells no matter how large the
# coefficient — on the eORCA1 Denmark Strait the cap is σθ = 27.878 against the 27.902 needed to fill
# the abyss. This scheme instead closes an overturning circuit, which has no such cap:
#
#   1. dense shelf water enters the *bottom* cell of the deep column,
#   2. the water it displaces is advected up the deep column, level by level,
#   3. and returns to the shelf at the shelf's own level.
#
# The deep bottom cell therefore relaxes towards the shelf value rather than towards a mean: it is
# flushed, not mixed. The transport follows Campin & Goosse, `u = γ g (ρˢ - ρᵈ) / ρ₀`, so it is a
# reduced-gravity plume speed and switches itself off as the contrast is consumed.

using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!
using KernelAbstractions: @index, @kernel
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: Center, znode
using Oceananigans.Operators: Azᶜᶜᶜ, Δxᶜᶠᶜ, Δyᶠᶜᶜ, Δzᶜᶜᶜ
using SeawaterPolynomials: haline_contraction, thermal_expansion
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using Adapt: Adapt, adapt

"""
    AdvectiveBottomBoundaryLayer(grid, equation_of_state; transport_coefficient, gravitational_acceleration)

Overturning bottom boundary layer following [Campin and Goosse (1999)](@cite CampinGoosse1999), NEMO's
`nn_bbl_adv = 2`. `transport_coefficient` is NEMO's `rn_gambbl` in seconds (default 10), setting the
downslope speed `u = γ g (ρˢ - ρᵈ) / ρ₀` from the density contrast across the step.

Shares `bottom_index` semantics with [`BottomBoundaryLayer`](@ref): the deepest wet level of every
column, zero over land.
"""
struct AdvectiveBottomBoundaryLayer{K, T, FT, E}
    bottom_index               :: K
    transport_x                :: T
    transport_y                :: T
    transport_coefficient      :: FT
    gravitational_acceleration :: FT
    equation_of_state          :: E
end

Adapt.adapt_structure(to, bbl::AdvectiveBottomBoundaryLayer) =
    AdvectiveBottomBoundaryLayer(adapt(to, bbl.bottom_index),
                                 adapt(to, bbl.transport_x),
                                 adapt(to, bbl.transport_y),
                                 adapt(to, bbl.transport_coefficient),
                                 adapt(to, bbl.gravitational_acceleration),
                                 adapt(to, bbl.equation_of_state))

function AdvectiveBottomBoundaryLayer(grid, equation_of_state;
                                      transport_coefficient = 10,
                                      gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration)

    FT = eltype(grid)

    bottom_index = Field{Center, Center, Nothing}(grid)
    set!(bottom_index, deepest_wet_level(grid))
    fill_halo_regions!(bottom_index)

    transport_x = Field{Center, Center, Nothing}(grid)
    transport_y = Field{Center, Center, Nothing}(grid)

    return AdvectiveBottomBoundaryLayer(bottom_index, transport_x, transport_y,
                                        convert(FT, transport_coefficient),
                                        convert(FT, gravitational_acceleration),
                                        equation_of_state)
end

#####
##### Face transport
#####

# Which side of a step carries the shelf. `k` counts up from the seafloor, so the shallower column —
# the shelf — is the one with the *larger* deepest-wet level.
@inline function step_columns(kᴬ, kᴮ)
    shelf_is_first = kᴬ > kᴮ
    return shelf_is_first, max(kᴬ, kᴮ), min(kᴬ, kᴮ)
end

# `u = γ g δρ` with `δρ = (ρˢ - ρᵈ)/ρ₀ ≈ ᾱ (Tᵈ - Tˢ) - β̄ (Sᵈ - Sˢ)`, clipped at zero so water only ever
# runs downslope. NEMO's `zgdrho` with `MAX(0, ...)`.
@inline function advective_face_transport(bbl, fields, grid, iˢ, jˢ, kˢ, iᵈ, jᵈ, kᵈ, face_width)

    wet = (kˢ > 0) & (kᵈ > 0) & (kˢ > kᵈ)   # a step is required; a flat bottom has no downslope direction

    kˢ⁺ = max(kˢ, 1)
    kᵈ⁺ = max(kᵈ, 1)

    @inbounds begin
        Tˢ = fields.T[iˢ, jˢ, kˢ⁺];  Sˢ = fields.S[iˢ, jˢ, kˢ⁺]
        Tᵈ = fields.T[iᵈ, jᵈ, kᵈ⁺];  Sᵈ = fields.S[iᵈ, jᵈ, kᵈ⁺]
    end

    zˢ = znode(iˢ, jˢ, kˢ⁺, grid, Center(), Center(), Center())
    zᵈ = znode(iᵈ, jᵈ, kᵈ⁺, grid, Center(), Center(), Center())

    ℰ = bbl.equation_of_state
    αˢ = thermal_expansion(Tˢ, Sˢ, zˢ, ℰ);  βˢ = haline_contraction(Tˢ, Sˢ, zˢ, ℰ)
    αᵈ = thermal_expansion(Tᵈ, Sᵈ, zᵈ, ℰ);  βᵈ = haline_contraction(Tᵈ, Sᵈ, zᵈ, ℰ)

    δρ = 0.5 * ((αˢ + αᵈ) * (Tᵈ - Tˢ) - (βˢ + βᵈ) * (Sᵈ - Sˢ))

    thickness = min(Δzᶜᶜᶜ(iˢ, jˢ, kˢ⁺, grid), Δzᶜᶜᶜ(iᵈ, jᵈ, kᵈ⁺, grid))
    u = bbl.transport_coefficient * bbl.gravitational_acceleration * max(0, δρ)

    return ifelse(wet, u * face_width * thickness, zero(grid))
end

@inline function x_face_transport(bbl::AdvectiveBottomBoundaryLayer, fields, grid, i, j)
    kᴸ = bottom_level(bbl, i,   j)
    kᴿ = bottom_level(bbl, i+1, j)
    left_is_shelf, kˢ, kᵈ = step_columns(kᴸ, kᴿ)

    iˢ = ifelse(left_is_shelf, i,   i+1)
    iᵈ = ifelse(left_is_shelf, i+1, i)

    face_width = Δyᶠᶜᶜ(i+1, j, max(kᵈ, 1), grid)

    return advective_face_transport(bbl, fields, grid, iˢ, j, kˢ, iᵈ, j, kᵈ, face_width)
end

@inline function y_face_transport(bbl::AdvectiveBottomBoundaryLayer, fields, grid, i, j)
    kᴰ = bottom_level(bbl, i, j)
    kᵁ = bottom_level(bbl, i, j+1)
    down_is_shelf, kˢ, kᵈ = step_columns(kᴰ, kᵁ)

    jˢ = ifelse(down_is_shelf, j,   j+1)
    jᵈ = ifelse(down_is_shelf, j+1, j)

    face_width = Δxᶜᶠᶜ(i, j+1, max(kᵈ, 1), grid)

    return advective_face_transport(bbl, fields, grid, i, jˢ, kˢ, i, jᵈ, kᵈ, face_width)
end

"""
    update_advective_bottom_boundary_layer!(sim, bbl)

Refresh the two face-transport fields from the current ocean state. Called once per step so the four
equation-of-state evaluations per face are paid on the 2D bottom surface rather than per tracer per cell.
"""
function update_advective_bottom_boundary_layer!(sim, bbl::AdvectiveBottomBoundaryLayer)
    ocean = sim.model.ocean
    fields = (T = ocean.model.tracers.T, S = ocean.model.tracers.S)
    return update_advective_bottom_boundary_layer!(bbl, ocean.model.grid, fields)
end

function update_advective_bottom_boundary_layer!(bbl::AdvectiveBottomBoundaryLayer, grid, fields)
    launch!(architecture(grid), grid, :xy, _compute_advective_bottom_boundary_layer_transport!,
            bbl.transport_x, bbl.transport_y, grid, bbl, fields)

    fill_halo_regions!(bbl.transport_x)
    fill_halo_regions!(bbl.transport_y)

    return nothing
end

@kernel function _compute_advective_bottom_boundary_layer_transport!(Qx, Qy, grid, bbl, fields)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Qx[i, j, 1] = x_face_transport(bbl, fields, grid, i, j)
        Qy[i, j, 1] = y_face_transport(bbl, fields, grid, i, j)
    end
end

#####
##### Tendency
#####

# The three limbs of the overturning circuit, for one step and one cell. Written so the three cases are
# mutually exclusive and sum to zero over the cells the circuit touches, which is what makes the scheme
# conservative in the volume integral.
@inline function overturning_limb(c, Q, iˢ, jˢ, kˢ, iᵈ, jᵈ, kᵈ, i, j, k, grid)

    kˢ⁺ = max(kˢ, 1)
    kᵈ⁺ = max(kᵈ, 1)

    @inbounds begin
        cˢ  = c[iˢ, jˢ, kˢ⁺]            # shelf bottom cell, the dense source
        cᵈˢ = c[iᵈ, jᵈ, kˢ⁺]            # deep column at the shelf's own level, the return water
        cᵈ  = c[iᵈ, jᵈ, kᵈ⁺]            # deep bottom cell, the destination
        cᵏ  = c[iᵈ, jᵈ, max(k, 1)]
        cᵏ⁻ = c[iᵈ, jᵈ, max(k - 1, 1)]  # one level deeper
    end

    on_shelf  = (i == iˢ) & (j == jˢ) & (k == kˢ)
    in_column = (i == iᵈ) & (j == jᵈ) & (k > kᵈ) & (k <= kˢ)
    on_deep   = (i == iᵈ) & (j == jᵈ) & (k == kᵈ)

    return ifelse(on_shelf,  Q * (cᵈˢ - cˢ),
           ifelse(in_column, Q * (cᵏ⁻ - cᵏ),
           ifelse(on_deep,   Q * (cˢ  - cᵈ), zero(grid))))
end

@inline function x_overturning_limb(c, bbl, grid, iface, i, j, k)
    Q = @inbounds bbl.transport_x[iface, j, 1]

    kᴸ = bottom_level(bbl, iface,   j)
    kᴿ = bottom_level(bbl, iface+1, j)
    left_is_shelf, kˢ, kᵈ = step_columns(kᴸ, kᴿ)

    iˢ = ifelse(left_is_shelf, iface,   iface+1)
    iᵈ = ifelse(left_is_shelf, iface+1, iface)

    return overturning_limb(c, Q, iˢ, j, kˢ, iᵈ, j, kᵈ, i, j, k, grid)
end

@inline function y_overturning_limb(c, bbl, grid, jface, i, j, k)
    Q = @inbounds bbl.transport_y[i, jface, 1]

    kᴰ = bottom_level(bbl, i, jface)
    kᵁ = bottom_level(bbl, i, jface+1)
    down_is_shelf, kˢ, kᵈ = step_columns(kᴰ, kᵁ)

    jˢ = ifelse(down_is_shelf, jface,   jface+1)
    jᵈ = ifelse(down_is_shelf, jface+1, jface)

    return overturning_limb(c, Q, i, jˢ, kˢ, i, jᵈ, kᵈ, i, j, k, grid)
end

"""
    advective_bottom_boundary_layer_tendency(i, j, k, grid, clock, fields, parameters)

Discrete-form forcing. Each of the four steps around a column contributes through whichever limb of its
overturning circuit the cell `(i, j, k)` sits on; cells on no limb return zero.

Divides by `Az * Δz` where NEMO divides by area alone, because NEMO's tracer tendency is thickness
weighted and Oceananigans' is not. `Δz` is evaluated live so the metric follows z-star.
"""
@inline function advective_bottom_boundary_layer_tendency(i, j, k, grid, clock, fields, parameters)

    bbl = parameters.bottom_boundary_layer
    c   = tracer_field(fields, parameters.tracer_name)

    transport = x_overturning_limb(c, bbl, grid, i,   i, j, k) +
                x_overturning_limb(c, bbl, grid, i-1, i, j, k) +
                y_overturning_limb(c, bbl, grid, j,   i, j, k) +
                y_overturning_limb(c, bbl, grid, j-1, i, j, k)

    volume = Azᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)

    return transport / volume
end

"""
    advective_bottom_boundary_layer_forcing(bbl::AdvectiveBottomBoundaryLayer)

Return `(T, S)` forcings to merge into the ocean model's `forcing` keyword.
"""
function advective_bottom_boundary_layer_forcing(bbl::AdvectiveBottomBoundaryLayer)
    T = Forcing(advective_bottom_boundary_layer_tendency; discrete_form = true,
                parameters = (bottom_boundary_layer = bbl, tracer_name = Val(:T)))

    S = Forcing(advective_bottom_boundary_layer_tendency; discrete_form = true,
                parameters = (bottom_boundary_layer = bbl, tracer_name = Val(:S)))

    return (; T, S)
end

"""
    AdvectiveBottomBoundaryLayerUpdate(bbl)

Callable that refreshes the face transports; register once per step with `add_callback!`.
"""
struct AdvectiveBottomBoundaryLayerUpdate{B}
    bottom_boundary_layer :: B
end

(u::AdvectiveBottomBoundaryLayerUpdate)(sim) = update_advective_bottom_boundary_layer!(sim, u.bottom_boundary_layer)

"""
    advective_bottom_boundary_layer_forcing(grid, transport_coefficient)

Convenience form for `omip_simulation`. Returns `(forcing, bbl)`; the forcing is an empty `NamedTuple`
and `bbl` is `nothing` when `transport_coefficient` is not set, so the scheme costs nothing when it is
off. The caller must register `AdvectiveBottomBoundaryLayerUpdate(bbl)` as a callback.
"""
function advective_bottom_boundary_layer_forcing(grid, transport_coefficient)
    isnothing(transport_coefficient) && return NamedTuple(), nothing
    bbl = AdvectiveBottomBoundaryLayer(grid, TEOS10EquationOfState(); transport_coefficient)
    return advective_bottom_boundary_layer_forcing(bbl), bbl
end

"""
    merge_tracer_forcings(forcings...)

Combine per-tracer forcing `NamedTuple`s, collecting any tracer named by more than one of them into a
tuple. Oceananigans materializes a tuple of forcings into `MultipleForcings`, so the two bottom boundary
layer schemes can run together: the diffusive one suppresses the instability that erodes the plume, the
advective one supplies the downslope transport.
"""
function merge_tracer_forcings(forcings...)
    names = union(keys.(forcings)...)
    isempty(names) && return NamedTuple()

    combined = map(names) do name
        contributions = Tuple(f[name] for f in forcings if haskey(f, name))
        length(contributions) == 1 ? only(contributions) : contributions
    end

    return NamedTuple{Tuple(names)}(combined)
end
