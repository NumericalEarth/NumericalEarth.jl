# Driver: per-step computation of `κu`, `κc`, `γ`, `hbl`, `u★`. Two kernels:
# Kernel 1 (`:xy`) writes column-level scalars; Kernel 2 reads them and writes
# κu, κc, γ at every interface.

function compute_closure_fields!(diffusivities, closure::FlavorOfKPP, model; parameters = :xyz)
    arch  = model.architecture
    grid  = model.grid
    clock = model.clock

    radiation        = get_radiative_forcing(model)
    coriolis         = model.coriolis
    top_velocity_bcs = (u = model.velocities.u.boundary_conditions.top,
                        v = model.velocities.v.boundary_conditions.top)
    top_bcs = KPPTopBoundaryConditions(top_velocity_bcs, diffusivities.top_tracer_bcs.bcs)

    launch!(arch, grid, :xy, compute_kpp_column_fields!,
            diffusivities, grid, closure,
            model.velocities, model.tracers, model.buoyancy,
            top_bcs, radiation, coriolis, clock)

    launch!(arch, grid, parameters, compute_kpp_diffusivities!,
            diffusivities, grid, closure,
            model.velocities, model.tracers, model.buoyancy, radiation)

    return nothing
end

#####
##### Kernel 1: column-level scalars (u★, Bo, hbl, K and dK/dz at BL base)
#####

@kernel function compute_kpp_column_fields!(K, grid, closure, velocities, tracers, buoyancy,
                                            top_bcs, radiation, coriolis, clock)
    i, j = @index(Global, NTuple)

    FT = eltype(grid)
    Nz = grid.Nz
    p  = getclosure(i, j, closure).parameters
    fields = merge(velocities, tracers)

    u★  = friction_velocity(i, j, grid, clock, fields, top_bcs.velocities, p)
    Bo  = non_solar_buoyancy(i, j, grid, clock, fields, buoyancy, top_bcs.tracers)
    α   = αᶜᶜᶜ(i, j, grid, buoyancy, tracers)
    g   = buoyancy.formulation.gravitational_acceleration

    hbl = compute_boundary_layer_depth(i, j, grid, closure,
                                       velocities, tracers, buoyancy,
                                       u★, Bo, α, g, radiation, coriolis)

    # Column sweep: track interior K at the deepest face below hbl (subscript ₋)
    # and the first face above (subscript ₊) for the FD derivative dK/dz at hbl.
    z₀ = znode(i, j, Nz, grid, Center(), Center(), Center())
    ν₋ = zero(FT); ν₊ = zero(FT)
    κ₋ = zero(FT); κ₊ = zero(FT)
    z₋ = zero(FT); z₊ = zero(FT)
    crossed    = false
    have_below = false
    for k in 1:(Nz + 1)
        zf = znode(i, j, k, grid, Center(), Center(), Face())
        d  = z₀ - zf
        νₖ = interior_viscosityᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
        κₖ = interior_diffusivityᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)

        below   = d > hbl
        capture = !below & !crossed

        ν₋ = ifelse(below,   νₖ, ν₋)
        κ₋ = ifelse(below,   κₖ, κ₋)
        z₋ = ifelse(below,   zf, z₋)
        ν₊ = ifelse(capture, νₖ, ν₊)
        κ₊ = ifelse(capture, κₖ, κ₊)
        z₊ = ifelse(capture, zf, z₊)

        crossed    = crossed    | !below
        have_below = have_below | below
    end

    # When hbl extends to the bottom, the FD pair is degenerate → dKdz = 0
    # so the matching reduces to a smooth G(σ) without spurious gradient terms.
    Δz  = max(z₊ - z₋, FT(1e-10))
    νₕ  = ν₋
    κₕ  = κ₋
    dνₕ = ifelse(have_below, (ν₊ - ν₋) / Δz, zero(FT))
    dκₕ = ifelse(have_below, (κ₊ - κ₋) / Δz, zero(FT))

    # Branchless land-column mask: zero everywhere on fully-land columns.
    wet = static_column_depthᶜᶜᵃ(i, j, grid) > zero(FT)
    @inbounds K.hbl[i, j, 1] = ifelse(wet, hbl, zero(FT))
    @inbounds K.u★[i, j, 1]  = ifelse(wet, u★,  zero(FT))
    @inbounds K.Bo[i, j, 1]  = ifelse(wet, Bo,  zero(FT))
    @inbounds K.νₕ[i, j, 1]  = ifelse(wet, νₕ,  zero(FT))
    @inbounds K.κₕ[i, j, 1]  = ifelse(wet, κₕ,  zero(FT))
    @inbounds K.dνₕ[i, j, 1] = ifelse(wet, dνₕ, zero(FT))
    @inbounds K.dκₕ[i, j, 1] = ifelse(wet, dκₕ, zero(FT))
end

#####
##### Kernel 2: per-interface κu, κc, γ
#####

@kernel function compute_kpp_diffusivities!(K, grid, closure, velocities, tracers, buoyancy, radiation)
    i, j, k = @index(Global, NTuple)
    _kpp_interface!(i, j, k, K, grid, closure, velocities, tracers, buoyancy, radiation)
end

@inline function _kpp_interface!(i, j, k, K, grid, closure, velocities, tracers, buoyancy, radiation)
    FT  = eltype(grid)
    Nz  = grid.Nz
    p   = getclosure(i, j, closure).parameters
    clo = getclosure(i, j, closure)

    @inbounds hbl = K.hbl[i, j, 1]
    @inbounds u★  = K.u★[i, j, 1]
    @inbounds Bo  = K.Bo[i, j, 1]
    @inbounds νₕ  = K.νₕ[i, j, 1]
    @inbounds κₕ  = K.κₕ[i, j, 1]
    @inbounds dνₕ = K.dνₕ[i, j, 1]
    @inbounds dκₕ = K.dκₕ[i, j, 1]

    α = αᶜᶜᶜ(i, j, grid, buoyancy, tracers)
    g = buoyancy.formulation.gravitational_acceleration

    z₀    = znode(i, j, Nz, grid, Center(), Center(), Center())
    d     = z₀ - znode(i, j, k, grid, Center(), Center(), Face())
    σ     = d / max(hbl, FT(1e-10))
    in_BL = (σ < one(FT)) & (σ ≥ zero(FT))

    νᵢ = interior_viscosityᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
    κᵢ = interior_diffusivityᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)

    # Matching coefficients at σ = 1 (column-level; recomputed per interface).
    σ₁        = ifelse(Bo ≥ zero(FT), one(FT), p.ε)
    wm₁, ws₁  = velocity_scales(σ₁, hbl, u★, Bo, p)
    G1u, dG1u = matching_coefficients(hbl, νₕ, dνₕ, wm₁, Bo, u★, p)
    G1s, dG1s = matching_coefficients(hbl, κₕ, dκₕ, ws₁, Bo, u★, p)

    # Local turbulent scales at this interface (SW-aware Bf).
    Bf     = buoyancy_forcing_above(i, j, d, Bo, radiation, α, g)
    σw     = ifelse(Bf ≥ zero(FT), one(FT), p.ε)
    wm, ws = velocity_scales(σw, hbl, u★, Bf, p)

    νᵇ = boundary_layer_diffusivity(σ, hbl, wm, G1u, dG1u)
    κᵇ = boundary_layer_diffusivity(σ, hbl, ws, G1s, dG1s)

    ν = min(ifelse(in_BL, max(νᵇ, νᵢ), νᵢ), clo.maximum_viscosity)
    κ = min(ifelse(in_BL, max(κᵇ, κᵢ), κᵢ), clo.maximum_diffusivity)
    γ = ifelse(in_BL, nonlocal_transport(hbl, ws, Bo, p), zero(FT))

    @inbounds K.κu[i, j, k] = ν
    @inbounds K.κc[i, j, k] = κ
    @inbounds K.γ[i, j, k]  = γ
end
