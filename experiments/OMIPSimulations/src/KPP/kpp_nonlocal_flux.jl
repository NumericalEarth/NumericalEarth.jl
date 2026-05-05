# Nonlocal-tracer-flux override for KPP. The total tracer flux at interface
# (i, j, k) is
#
#     F = - κ · ∂c/∂z  -  κ · γ · Q₀
#         ╰──────────╯    ╰─────────╯
#         local            nonlocal (γ ≡ 0 outside the BL and for momentum)
#
# `Q₀` is the per-tracer surface flux fetched from `K.top_tracer_bcs`. The
# nonlocal piece is independent of `c`, so it must be treated explicitly
# even when the local piece runs through the vertically-implicit solver.

@inline function surface_tracer_flux(i, j, grid, K, ::Val{id}, clock, fields) where id
    bc = getproperty(K.top_tracer_bcs, id)
    return getbc(bc, i, j, grid, clock, fields)
end

@inline function nonlocal_z_flux(i, j, k, grid, K, val_id, clock, fields)
    κ  = @inbounds K.κc[i, j, k]
    γ  = @inbounds K.γ[i, j, k]
    Q₀ = surface_tracer_flux(i, j, grid, K, val_id, clock, fields)
    return - κ * γ * Q₀
end

# Explicit time discretization: full local + nonlocal flux.
@inline function diffusive_flux_z(i, j, k, grid, ::KPPVD, K, val_id::Val, c, clock, fields, buoyancy)
    κ = @inbounds K.κc[i, j, k]
    return - κ * ∂zᶜᶜᶠ(i, j, k, grid, c) + nonlocal_z_flux(i, j, k, grid, K, val_id, clock, fields)
end

# VITD on a periodic-z grid: only the nonlocal piece is explicit; the local
# diffusion runs through the implicit tridiagonal solve.
@inline diffusive_flux_z(i, j, k, grid, ::VITD, ::KPPVD, K, val_id::Val, c, clock, fields, buoyancy) =
    nonlocal_z_flux(i, j, k, grid, K, val_id, clock, fields)

# VITD on a vertically-bounded grid: at z-boundary cells fall back to the
# explicit (local + nonlocal) flux; in the bulk only the nonlocal piece.
@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::KPPVD,
                                  K, val_id::Val, c, clock, fields, buoyancy)
    return ifelse((k == 1) | (k == grid.Nz + 1),
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, K, val_id, c, clock, fields, buoyancy),
                  nonlocal_z_flux(i, j, k, grid, K, val_id, clock, fields))
end
