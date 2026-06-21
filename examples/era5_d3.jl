# Telescoping inner D3 nest (Fan Domain-3, 3 km / 1/36°), SGP-centered, driven LIVE by the
# D2 model `model` in memory. Included by era5_breeze.jl when TELESCOPE=on, AFTER the D2 model
# is built (so it sees `model`, `arch`, `constants`, `z_discretization`, the ERA5 metadata,
# `p̄₀`, the advection/microphysics/coriolis/dynamics choices, `damping_depth`, `cut_plane`, …).
# Adapts the baroclinic_lam `build_inner` / `set_inner_ic!` / `build_inner_davies` templates.
#
# Defines: `D3` (the inner AtmosphereModel), `d3_davies!` (per-step hand-rolled Davies callback),
# and `capture_d3_slice!` (2-km-AGL w/ρqʳ capture into `d3_slice_frames`). The caller wraps
# `NestedModel(model, D3)` in the NestedSimulation and registers the callbacks.

using Oceananigans.Fields: interpolate!

thermo_density(m) = Breeze.AtmosphereModels.thermodynamic_density(m.formulation)

# --- D3 grid: Fan-D3 footprint (300×270 @ 1/36°), SGP-centered, inside D2 with fringe margin ---
const D3_Δ  = 1/36
const D3_Nx = parse(Int, get(ENV, "D3_NX", "300"))
const D3_Ny = parse(Int, get(ENV, "D3_NY", "270"))
const D3_λ0 = λ₀ - D3_Nx * D3_Δ / 2
const D3_λ1 = λ₀ + D3_Nx * D3_Δ / 2
const D3_φ0 = φ₀ - D3_Ny * D3_Δ / 2
const D3_φ1 = φ₀ + D3_Ny * D3_Δ / 2
@assert D3_λ0 > λ_west + 5/12 && D3_λ1 < λ_east - 5/12 "D3 must sit inside D2 with fringe margin"
@assert D3_φ0 > φ_south + 5/12 && D3_φ1 < φ_north - 5/12 "D3 must sit inside D2 with fringe margin"

D3_grid = LatitudeLongitudeGrid(arch;
                                longitude = (D3_λ0, D3_λ1), latitude = (D3_φ0, D3_φ1),
                                z = TerrainFollowingVerticalDiscretization(z_discretization),
                                size = (D3_Nx, D3_Ny, Nz), halo = (5, 5, 5),
                                topology = (Bounded, Bounded, Bounded))
materialize_terrain!(D3_grid, regrid_topography(D3_grid; dataset = ETOPO2022()))

# --- D3 lateral BCs via the ROLLING-FTS workaround (dodges G1) ---
# `Interpolated` accepts an FTS or a live AbstractField. A live AbstractField source hard-fails
# GPU codegen (G1: Adapt strips it to a bare OffsetArray → `_query_source(::AbstractField)` misses
# → dynamic dispatch in the halo-fill kernel → InvalidIRError). An FTS source survives Adapt as a
# `FlavorOfFTS` and compiles clean. So we wrap each D2 boundary field in a 2-slot FieldTimeSeries on
# D2's grid, refreshed from the live D2 state each step (`refresh_d3_bc_fts!`). Both slots hold the
# same (current) D2 state over a wide time bracket, so the time-interpolation returns it for any
# clock time; the per-step refresh carries the (one-step-lagged) live D2 boundary into D3.
const D3_FTS_TIMES = [0.0, 1.0e9]
function build_d2_fts(LX, LY, LZ, field)
    fts = FieldTimeSeries{LX, LY, LZ}(model.grid, D3_FTS_TIMES)
    for n in 1:2
        interior(fts[n]) .= interior(field)
    end
    return fts
end
d2_ρu_fts  = build_d2_fts(Face,   Center, Center, model.momentum.ρu)
d2_ρv_fts  = build_d2_fts(Center, Face,   Center, model.momentum.ρv)
d2_ρ_fts   = build_d2_fts(Center, Center, Center, model.dynamics.density)
d2_ρθ_fts  = build_d2_fts(Center, Center, Center, thermo_density(model))
d2_ρqᵉ_fts = build_d2_fts(Center, Center, Center, model.moisture_density)
const D3_BC_FTS = ((d2_ρu_fts, model.momentum.ρu), (d2_ρv_fts, model.momentum.ρv),
                   (d2_ρ_fts, model.dynamics.density), (d2_ρθ_fts, thermo_density(model)),
                   (d2_ρqᵉ_fts, model.moisture_density))
function refresh_d3_bc_fts!(sim)
    for (fts, fld) in D3_BC_FTS, n in 1:2
        interior(fts[n]) .= interior(fld)
    end
    return nothing
end

D3_ρu_bcs = parent_boundary_conditions(D3_grid; variables = (ρu = d2_ρu_fts,), sides = (:west, :east))
D3_ρv_bcs = parent_boundary_conditions(D3_grid; variables = (ρv = d2_ρv_fts,), sides = (:south, :north))
D3_sc_bcs = parent_boundary_conditions(D3_grid;
                                       variables = (ρ = d2_ρ_fts, ρe = d2_ρθ_fts, ρqᵉ = d2_ρqᵉ_fts),
                                       sides    = (:west, :east, :south, :north),
                                       bc_types = (ρ = ValueBoundaryCondition, ρe = ValueBoundaryCondition, ρqᵉ = ValueBoundaryCondition))
D3_bcs = merge(D3_ρu_bcs, D3_ρv_bcs, D3_sc_bcs)

# --- D3 forcing: explicit Rayleigh ρw sponge near the lid, same profile as D2 (no Davies here;
#     the hand-rolled Davies below is a per-step callback) ---
D3_forcing = let z_top = z_discretization.faces[end], depth = float(damping_depth)
    mask = (λ, φ, z) -> (s = clamp((z - (z_top - depth)) / depth, zero(z), one(z)); s * s * (3 - 2s))
    (ρw = Relaxation(rate = 1/damping_timescale, mask = mask, target = 0.0),)
end

D3 = atmosphere_simulation(D3_grid;
                           thermodynamic_constants = constants,
                           momentum_advection  = momentum_advection_scheme,
                           microphysics        = microphysics_scheme,
                           coriolis            = coriolis_scheme,
                           dynamics            = CompressibleDynamics(time_discretization; surface_pressure = p̄₀),
                           boundary_conditions = D3_bcs,
                           forcing             = D3_forcing).model

# --- D3 IC: regrid the developed D2 state onto the finer D3 grid ---
# NB: do NOT `interpolate!` into ρw — that fills ρw's halo via the terrain-kinematic bottom BC,
# which reads `model_fields.ρu/ρv`, but a bare `interpolate!`/`set!`-time halo fill passes empty
# args (no model context) → the BC's `getbc` hits a method-error and the GPU kernel won't compile.
# Mirror D2's IC: interpolate the scalars + horizontal momentum, then set ρw via consistent-w after
# `update_state!` (which fills the terrain halo WITH the model context).
interpolate!(D3.dynamics.density, model.dynamics.density)
interpolate!(thermo_density(D3),  thermo_density(model))
interpolate!(D3.moisture_density, model.moisture_density)
interpolate!(D3.momentum.ρu, model.momentum.ρu)
interpolate!(D3.momentum.ρv, model.momentum.ρv)
update_state!(D3)
interior(D3.momentum.ρw) .-= interior(D3.dynamics.contravariant_vertical_momentum)
update_state!(D3)

# --- Hand-rolled Davies fringe: relax D3 ρu,ρv,ρw,ρθ toward the regridded LIVE D2 over a
#     FRINGE_N-cell cosine ramp each step (parent_forcings/Relaxation can't take a live target) ---
const D3_FRINGE_N = 5
const D3_TAU_RELAX = 10 * Δt

function online_fringe_mask(arch, FT, nx, ny, nz)
    m = zeros(FT, nx, ny, nz)
    ramp(d) = d >= D3_FRINGE_N ? 0.0 : 0.5 * (1 + cos(π * d / D3_FRINGE_N))
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        m[i, j, k] = max(ramp(i - 1), ramp(nx - i), ramp(j - 1), ramp(ny - j))
    end
    return on_architecture(arch, m)
end
function xface_density!(ρf, ρc); n = size(ρc, 1)
    @views @. ρf[2:n, :, :] = 0.5 * (ρc[1:n-1, :, :] + ρc[2:n, :, :])
    @views ρf[1, :, :] .= ρc[1, :, :]; @views ρf[n+1, :, :] .= ρc[n, :, :]; end
function yface_density!(ρf, ρc); n = size(ρc, 2)
    @views @. ρf[:, 2:n, :] = 0.5 * (ρc[:, 1:n-1, :] + ρc[:, 2:n, :])
    @views ρf[:, 1, :] .= ρc[:, 1, :]; @views ρf[:, n+1, :] .= ρc[:, n, :]; end
function zface_density!(ρf, ρc); n = size(ρc, 3)
    @views @. ρf[:, :, 2:n] = 0.5 * (ρc[:, :, 1:n-1] + ρc[:, :, 2:n])
    @views ρf[:, :, 1] .= ρc[:, :, 1]; @views ρf[:, :, n+1] .= ρc[:, :, n]; end

let inner = D3, outer = model
    Nxi, Nyi, Nzi = size(inner.grid); FT = eltype(inner.grid); ai = Oceananigans.Architectures.architecture(inner.grid)
    mask_u = online_fringe_mask(ai, FT, Nxi + 1, Nyi, Nzi)
    mask_v = online_fringe_mask(ai, FT, Nxi, Nyi + 1, Nzi)
    mask_w = online_fringe_mask(ai, FT, Nxi, Nyi, Nzi + 1)
    mask_c = online_fringe_mask(ai, FT, Nxi, Nyi, Nzi)
    dev(s...) = on_architecture(ai, zeros(FT, s...))
    ρxf = dev(Nxi + 1, Nyi, Nzi); ρyf = dev(Nxi, Nyi + 1, Nzi); ρzf = dev(Nxi, Nyi, Nzi + 1)
    u_t = XFaceField(inner.grid); v_t = YFaceField(inner.grid); w_t = ZFaceField(inner.grid)
    ρ_t = CenterField(inner.grid); ρθ_t = CenterField(inner.grid)
    global function d3_davies!(sim)
        α = 1 - exp(-sim.Δt / D3_TAU_RELAX)
        interpolate!(u_t, outer.velocities.u); interpolate!(v_t, outer.velocities.v); interpolate!(w_t, outer.velocities.w)
        interpolate!(ρ_t, outer.dynamics.density); interpolate!(ρθ_t, thermo_density(outer))
        θt = interior(ρθ_t) ./ interior(ρ_t)
        ut = interior(u_t); vt = interior(v_t); wt = interior(w_t)
        ρc = interior(inner.dynamics.density)
        ρu = interior(inner.momentum.ρu); ρv = interior(inner.momentum.ρv); ρw = interior(inner.momentum.ρw)
        ρθ = interior(thermo_density(inner))
        xface_density!(ρxf, ρc); yface_density!(ρyf, ρc); zface_density!(ρzf, ρc)
        @. ρu += α * mask_u * (ρxf * ut - ρu)
        @. ρv += α * mask_v * (ρyf * vt - ρv)
        @. ρw += α * mask_w * (ρzf * wt - ρw)
        @. ρθ += α * mask_c * (ρc * θt - ρθ)
        return nothing
    end
end

# --- D3 2-km-AGL slice capture (w, ρqʳ) into d3_slice_frames ---
d3_slice_frames = NamedTuple[]
function capture_d3_slice!()
    λs, φs, w_slice = cut_plane(D3.velocities.w, slice_height)
    pf = Oceananigans.prognostic_fields(D3)
    ρqʳ_slice = haskey(pf, :ρqʳ) ? cut_plane(pf[:ρqʳ], slice_height)[3] : zero(w_slice)
    push!(d3_slice_frames, (t = D3.clock.time, iter = D3.clock.iteration, λ = λs, φ = φs, w = w_slice, ρqʳ = ρqʳ_slice))
end

@info @sprintf("D3 telescoping nest: %d×%d @ 1/36° (λ %.2f–%.2f, φ %.2f–%.2f), SGP-centered inside D2",
               D3_Nx, D3_Ny, D3_λ0, D3_λ1, D3_φ0, D3_φ1)
