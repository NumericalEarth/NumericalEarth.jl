using Oceananigans: fields
using Oceananigans.Advection: div_Uc
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: CenterField
using Oceananigans.Grids: Center
using Oceananigans.ImmersedBoundaries: inactive_node
using Oceananigans.Operators: Az_qᶜᶜᶠ, Ayᶜᶠᶜ, Vᶜᶜᶜ, V⁻¹ᶜᶜᶜ, Δzᶜᶜᶜ, δzᵃᵃᶜ, ∂zᶜᶜᶠ, ℑyᵃᶠᵃ
using Oceananigans.TimeSteppers: time_discretization
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, IsopycnalSkewSymmetricDiffusivity,
                                       VerticallyImplicitTimeDiscretization, κzᶜᶜᶠ, ∇_dot_qᶜ
using Oceananigans.Utils: launch!
using JLD2: ZstdFilter, jldopen
using KernelAbstractions: @index, @kernel
using SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

const WMT_EOS = TEOS10EquationOfState()

"""
σ₂ minus 1000: potential density referenced to 2000 dbar, from the same equation of state the model steps with.
"""
@inline σ₂(Θ, Sᴬ) = SeawaterPolynomials.ρ(Θ, Sᴬ, -2000, WMT_EOS) - 1000

"""
`(∂σ₂/∂Θ, ∂σ₂/∂S)` by centred differences, which is how a tracer tendency is mapped onto a density tendency.
"""
@inline function σ₂_derivatives(Θ, Sᴬ)
    δΘ = 1e-4
    δS = 1e-4
    σΘ = (σ₂(Θ + δΘ, Sᴬ) - σ₂(Θ - δΘ, Sᴬ)) / 2δΘ
    σS = (σ₂(Θ, Sᴬ + δS) - σ₂(Θ, Sᴬ - δS)) / 2δS
    return σΘ, σS
end

#####
##### The non-advective and advective density tendencies
#####

# ⚠ `∇_dot_qᶜ` hands its own time discretization down to the vertical flux, and a
# `VerticallyImplicitTimeDiscretization` returns zero there: that flux is applied by the tridiagonal solver, not by
# the tendency. CATKE is vertically implicit, so `∇_dot_qᶜ` alone reports exactly zero transformation for it — the
# largest physical term in the ocean — and the residual `AMOC_full - AMOC_phys` would book it as numerical mixing.
# The implicit part is therefore rebuilt here from the diffusivity the solver itself uses.
@inline vertical_diffusive_flux(i, j, k, grid, closure, K, id, c, clock, model_fields) =
    - κzᶜᶜᶠ(i, j, k, grid, closure, K, id, clock, model_fields) * ∂zᶜᶜᶠ(i, j, k, grid, c)

@inline implicit_flux_divergence(i, j, k, grid, closure, K, id, c, clock, model_fields) =
    implicit_flux_divergence(time_discretization(closure), i, j, k, grid, closure, K, id, c, clock, model_fields)

@inline implicit_flux_divergence(::ExplicitTimeDiscretization, i, j, k, grid, args...) = zero(grid)

@inline implicit_flux_divergence(::VerticallyImplicitTimeDiscretization, i, j, k, grid,
                                 closure, K, id, c, clock, model_fields) =
    V⁻¹ᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, vertical_diffusive_flux,
                                   closure, K, id, c, clock, model_fields)

# The tendency is minus the flux divergence. `id` is the tracer's `Val(index)` in `model.tracers`; T is 1 and S is 2
# in every OMIP configuration.
@inline function closure_tracer_tendency(i, j, k, grid, closure, K, id, c, clock, model_fields, buoyancy)
    explicit = ∇_dot_qᶜ(i, j, k, grid, closure, K, id, c, clock, model_fields, buoyancy)
    implicit = implicit_flux_divergence(i, j, k, grid, closure, K, id, c, clock, model_fields)
    return - (explicit + implicit)
end

@inline advective_tracer_tendency(i, j, k, grid, advection, U, c) = - div_Uc(i, j, k, grid, advection, U, c)

"""
Density tendency σ̇ = ∂σ₂/∂Θ · Θ̇ + ∂σ₂/∂S · Ṡ in kg m⁻³ s⁻¹ from one closure's diffusive flux divergence.
"""
@kernel function _compute_closure_density_tendency!(σ̇, grid, closure, K, T, S, clock, model_fields, buoyancy)
    i, j, k = @index(Global, NTuple)
    dry = inactive_node(i, j, k, grid, Center(), Center(), Center())
    @inbounds begin
        σΘ, σS = σ₂_derivatives(T[i, j, k], S[i, j, k])
        Θ̇ = closure_tracer_tendency(i, j, k, grid, closure, K, Val(1), T, clock, model_fields, buoyancy)
        Ṡ = closure_tracer_tendency(i, j, k, grid, closure, K, Val(2), S, clock, model_fields, buoyancy)
        σ̇[i, j, k] = ifelse(dry, zero(grid), σΘ * Θ̇ + σS * Ṡ)
    end
end

"""
Density tendency from the *discrete* tracer advection. Advection cannot change a water mass's density in the
continuum, so everything this accumulates is numerical mixing — the term the residual `AMOC_full − AMOC_phys` is
supposed to recover.
"""
@kernel function _compute_advective_density_tendency!(σ̇, grid, advection, U, T, S)
    i, j, k = @index(Global, NTuple)
    dry = inactive_node(i, j, k, grid, Center(), Center(), Center())
    @inbounds begin
        σΘ, σS = σ₂_derivatives(T[i, j, k], S[i, j, k])
        Θ̇ = advective_tracer_tendency(i, j, k, grid, advection, U, T)
        Ṡ = advective_tracer_tendency(i, j, k, grid, advection, U, S)
        σ̇[i, j, k] = ifelse(dry, zero(grid), σΘ * Θ̇ + σS * Ṡ)
    end
end

"""
Density tendency from the surface fluxes, which reach the ocean only through the top cell, so σ̇ = (∂σ₂/∂Θ · Jᵀ +
∂σ₂/∂S · Jˢ) / Δz there and zero below. `Jᵀ` and `Jˢ` are the assembled net ocean fluxes, in tracer units × m s⁻¹.
"""
@kernel function _compute_surface_density_tendency!(σ̇, grid, T, S, Jᵀ, Jˢ, Nz)
    i, j, k = @index(Global, NTuple)
    dry = inactive_node(i, j, k, grid, Center(), Center(), Center())
    @inbounds begin
        σΘ, σS = σ₂_derivatives(T[i, j, k], S[i, j, k])
        # A flux INTO the ocean is negative in Oceananigans' top boundary condition convention.
        rate = - (σΘ * Jᵀ[i, j, 1] + σS * Jˢ[i, j, 1]) / Δzᶜᶜᶜ(i, j, k, grid)
        σ̇[i, j, k] = ifelse(dry | (k != Nz), zero(grid), rate)
    end
end

"""
σ₂ of every cell, so the binning kernel never re-evaluates the equation of state.
"""
@kernel function _compute_sigma!(σ, grid, T, S)
    i, j, k = @index(Global, NTuple)
    @inbounds σ[i, j, k] = σ₂(T[i, j, k], S[i, j, k])
end

#####
##### Binning onto fixed σ₂ classes
#####

# One work item owns one (latitude row, density class) pair and sweeps the row, so nothing is scattered and no
# atomics are needed. The sweep costs Nx·Nz reads per pair and runs once per output interval, not once per step.
@kernel function _accumulate_transformation!(Ω, grid, σ, σ̇, σ₀, Δσ, Nx, Nz)
    j, n = @index(Global, NTuple)
    σ★ = σ₀ + (n - 1) * Δσ
    total = zero(grid)
    for k in 1:Nz, i in 1:Nx
        @inbounds begin
            inside = abs(σ[i, j, k] - σ★) < Δσ / 2
            total += ifelse(inside, Vᶜᶜᶜ(i, j, k, grid) * σ̇[i, j, k] / Δσ, zero(grid))
        end
    end
    @inbounds Ω[j, n] = total
end

# Meridional volume transport per density class: Σᵢₖ v Aʸ over the cells of the class, on the v face, where the
# class is set by the σ₂ average of the two cells the face separates. A cumulative sum over n gives AMOC_full(y, σ₂).
@kernel function _accumulate_class_transport!(Ψ, grid, σ, v, σ₀, Δσ, Nx, Nz)
    j, n = @index(Global, NTuple)
    σ★ = σ₀ + (n - 1) * Δσ
    total = zero(grid)
    for k in 1:Nz, i in 1:Nx
        @inbounds begin
            wet = !inactive_node(i, j, k, grid, Center(), Center(), Center()) &
                  !inactive_node(i, j - 1, k, grid, Center(), Center(), Center())
            σᶠ = ℑyᵃᶠᵃ(i, j, k, grid, σ)
            inside = wet & (abs(σᶠ - σ★) < Δσ / 2)
            total += ifelse(inside, v[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid), zero(grid))
        end
    end
    @inbounds Ψ[j, n] = total
end

#####
##### The diagnostic
#####

"""
Online water-mass-transformation accumulator: the physical and numerical parts of the overturning in σ₂ space,
following Sidorenko et al., *Un-physical mixing dominates projected weakening of the Atlantic overturning
circulation*.

Per model-latitude row `j` and fixed σ₂ class `n` it holds the transformation rate (m³ s⁻¹) of each process that
moves water across σ₂, plus the meridional volume transport within the class:

| field | what it is |
|---|---|
| `Ω_surface` | transformation by the assembled net surface fluxes |
| `Ω_vertical` | transformation by the vertical closures (CATKE and the background diffusivity) |
| `Ω_isopycnal` | transformation by the isopycnal closure (GM skew flux and Redi) |
| `Ω_advective` | transformation by the *discrete* tracer advection — zero in the continuum, so this is numerical mixing |
| `Ψ_class` | Σ v Aʸ over the cells of the class; a cumulative sum over classes is `AMOC_full(y, σ₂)` |

`AMOC_phys` is the cumulative sum of `Ω_surface + Ω_vertical + Ω_isopycnal`, `AMOC_full` that of `Ψ_class`, and
`AMOC_num = AMOC_full − AMOC_phys` their residual. `Ω_advective` measures the same numerical term *directly*, so the
two disagree only by what the residual also absorbs — the class-boundary advective flux and the σ₂ binning error.
Reporting both is the point: they cross-check each other.

`σ_range` sets the fixed class centres; its default spans the Atlantic overturning from the thermocline to the
densest bottom water the ORCA configurations reach.
"""
mutable struct WaterMassTransformation{G, F, A, C, N}
    grid        :: G
    σ           :: F
    σ̇           :: F
    Ω_surface   :: A
    Ω_vertical  :: A
    Ω_isopycnal :: A
    Ω_advective :: A
    Ψ_class     :: A
    σ₀          :: C
    Δσ          :: C
    filename    :: N
end

function WaterMassTransformation(ocean_model; σ_range = 30.0:0.05:38.0, filename = "omip_transformation.jld2")
    grid = ocean_model.grid
    Ny   = size(grid, 2)
    Nσ   = length(σ_range)
    FT   = eltype(grid)
    arch = architecture(grid)
    accumulator() = on_architecture(arch, zeros(FT, Ny, Nσ))
    return WaterMassTransformation(grid, CenterField(grid), CenterField(grid),
                                   accumulator(), accumulator(), accumulator(), accumulator(), accumulator(),
                                   convert(FT, first(σ_range)), convert(FT, step(σ_range)), filename)
end

@inline is_isopycnal_closure(closure) = closure isa IsopycnalSkewSymmetricDiffusivity

"Bin `wmt.σ̇` onto the σ₂ classes and write the row totals into `Ω`."
function bin_transformation!(Ω, wmt, arch)
    grid = wmt.grid
    Nx, Ny, Nz = size(grid)
    launch!(arch, grid, (Ny, size(Ω, 2)), _accumulate_transformation!,
            Ω, grid, wmt.σ, wmt.σ̇, wmt.σ₀, wmt.Δσ, Nx, Nz)
    return nothing
end

"""
Recompute every accumulator from the ocean model's current state. Each process is evaluated into the single `σ̇`
scratch field and binned before the next one overwrites it, so the diagnostic costs two 3-D fields, not six.
"""
function compute_transformation!(wmt::WaterMassTransformation, ocean_model, net_ocean_fluxes)
    grid = wmt.grid
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    T, S = ocean_model.tracers.T, ocean_model.tracers.S
    model_fields = fields(ocean_model)
    clock = ocean_model.clock
    buoyancy = ocean_model.buoyancy

    launch!(arch, grid, :xyz, _compute_sigma!, wmt.σ, grid, T, S)
    fill_halo_regions!(wmt.σ)

    launch!(arch, grid, :xyz, _compute_surface_density_tendency!,
            wmt.σ̇, grid, T, S, net_ocean_fluxes.T, net_ocean_fluxes.S, Nz)
    bin_transformation!(wmt.Ω_surface, wmt, arch)

    launch!(arch, grid, :xyz, _compute_advective_density_tendency!,
            wmt.σ̇, grid, ocean_model.advection.T, ocean_model.velocities, T, S)
    bin_transformation!(wmt.Ω_advective, wmt, arch)

    closures = ocean_model.closure isa Tuple ? ocean_model.closure : (ocean_model.closure,)
    K        = ocean_model.closure isa Tuple ? ocean_model.closure_fields : (ocean_model.closure_fields,)
    fill!(wmt.Ω_vertical, 0)
    fill!(wmt.Ω_isopycnal, 0)
    scratch = similar(wmt.Ω_vertical)
    for (closure, Kn) in zip(closures, K)
        launch!(arch, grid, :xyz, _compute_closure_density_tendency!,
                wmt.σ̇, grid, closure, Kn, T, S, clock, model_fields, buoyancy)
        bin_transformation!(scratch, wmt, arch)
        Ω = is_isopycnal_closure(closure) ? wmt.Ω_isopycnal : wmt.Ω_vertical
        Ω .+= scratch
    end

    launch!(arch, grid, (Ny, size(wmt.Ψ_class, 2)), _accumulate_class_transport!,
            wmt.Ψ_class, grid, wmt.σ, ocean_model.velocities.v, wmt.σ₀, wmt.Δσ, Nx, Nz)

    return nothing
end

"""
Append the current accumulators under `timeseries/<name>/<iteration>`, the layout every other OMIP stream uses, so
the offline readers need no special case.
"""
function write_transformation!(wmt::WaterMassTransformation, clock)
    iter = clock.iteration
    jldopen(wmt.filename, "a+"; compress = ZstdFilter()) do file
        file["timeseries/t/$iter"] = clock.time
        for (name, Ω) in (("wmtsurf", wmt.Ω_surface), ("wmtvert", wmt.Ω_vertical),
                          ("wmtiso", wmt.Ω_isopycnal), ("wmtadv", wmt.Ω_advective), ("psisig", wmt.Ψ_class))
            file["timeseries/$name/$iter"] = Array(Ω)
        end
    end
    return nothing
end

"Callback: recompute and append, on the schedule the caller attaches it with."
function (wmt::WaterMassTransformation)(simulation)
    ocean_model = simulation.model.ocean.model
    compute_transformation!(wmt, ocean_model, simulation.model.interfaces.net_fluxes.ocean)
    write_transformation!(wmt, ocean_model.clock)
    return nothing
end
