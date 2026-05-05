# Driver: per-step computation of `κu`, `κc`, `γ`, `hbl`, `u★` for KPP.
#
# Split into two kernels matching Oceananigans' convention:
#
#   Kernel 1 (`compute_kpp_column_fields!`, launched with `:xy`)
#     For each column (i, j) computes the column-level scalars u★, Bo, hbl
#     and the interior diffusivities at the BL base (used as matching values
#     for the cubic shape function). Writes them to the 2D closure-fields.
#
#   Kernel 2 (`compute_kpp_diffusivities!`, launched with `parameters`)
#     For each interface (i, j, k) reads the column-level scalars and writes
#     κu, κc, γ by combining the BL shape function with the interior K.

function compute_closure_fields!(diffusivities, closure::FlavorOfKPP, model; parameters = :xyz)
    arch  = model.architecture
    grid  = model.grid
    clock = model.clock

    radiation = get_radiative_forcing(model)
    coriolis  = model.coriolis
    top_velocity_bcs = (u = model.velocities.u.boundary_conditions.top,
                        v = model.velocities.v.boundary_conditions.top)
    top_bcs = KPPTopBoundaryConditions(top_velocity_bcs, diffusivities.top_tracer_bcs)

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
##### Kernel 1: column-level scalars (u★, Bo, hbl, Kint at BL base)
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
    g   = buoyancy.gravitational_acceleration

    hbl = compute_boundary_layer_depth(i, j, grid, closure,
                                       velocities, tracers, buoyancy,
                                       u★, Bo, α, g, radiation, coriolis)

    # Kint at the BL base — value at the deepest face still below hbl.
    z₀     = znode(i, j, Nz, grid, Center(), Center(), Center())
    Kint_u = zero(FT)
    Kint_c = zero(FT)
    for k in 1:(Nz + 1)
        d  = z₀ - znode(i, j, k, grid, Center(), Center(), Face())
        νₖ = interior_viscosityᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
        κₖ = interior_diffusivityᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
        below = d > hbl
        Kint_u = ifelse(below, νₖ, Kint_u)
        Kint_c = ifelse(below, κₖ, Kint_c)
    end

    @inbounds K.hbl[i, j, 1]    = hbl
    @inbounds K.u★[i, j, 1]     = u★
    @inbounds K.Bo[i, j, 1]     = Bo
    @inbounds K.Kint_u[i, j, 1] = Kint_u
    @inbounds K.Kint_c[i, j, 1] = Kint_c
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

    @inbounds hbl    = K.hbl[i, j, 1]
    @inbounds u★     = K.u★[i, j, 1]
    @inbounds Bo     = K.Bo[i, j, 1]
    @inbounds Kint_u = K.Kint_u[i, j, 1]
    @inbounds Kint_c = K.Kint_c[i, j, 1]

    α = αᶜᶜᶜ(i, j, grid, buoyancy, tracers)
    g = buoyancy.gravitational_acceleration

    z₀    = znode(i, j, Nz, grid, Center(), Center(), Center())
    d     = z₀ - znode(i, j, k, grid, Center(), Center(), Face())
    σ     = d / max(hbl, FT(1e-10))
    in_BL = (σ < one(FT)) & (σ >= zero(FT))

    νᵢ = interior_viscosityᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
    κᵢ = interior_diffusivityᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)

    # Matching coefficients at σ = 1 (column-level; cheap to recompute per interface).
    σ₁        = ifelse(Bo >= zero(FT), one(FT), p.ε)
    wm₁, ws₁  = velocity_scales(σ₁, hbl, u★, Bo, p)
    G1u, dG1u = matching_coefficients(hbl, Kint_u, zero(FT), wm₁, Bo, u★, p)
    G1s, dG1s = matching_coefficients(hbl, Kint_c, zero(FT), ws₁, Bo, u★, p)

    # Local turbulent scales at this interface (SW-aware Bf).
    Bf     = buoyancy_forcing_above(i, j, d, Bo, radiation, α, g)
    σw     = ifelse(Bf >= zero(FT), one(FT), p.ε)
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
