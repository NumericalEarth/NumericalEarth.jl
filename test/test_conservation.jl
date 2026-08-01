include("runtests_setup.jl")

using Oceananigans.Units
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Fields: interior
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind
using Oceananigans.Operators: Azᶜᶜᶜ, volume
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.TimeSteppers: time_step!

using ClimaSeaIce
using ClimaSeaIce.SeaIceThermodynamics: latent_heat

using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.Radiations: PrescribedRadiation
using NumericalEarth.Diagnostics: atmosphere_ocean_heat_flux, frazil_heat_flux
using NumericalEarth.EarthSystemModels: OceanSeaIceModel, update_state!
using NumericalEarth.Oceans: ocean_simulation
using NumericalEarth.SeaIces: sea_ice_simulation

# Diagnostic-side override: ClimaSeaIce's slab mass balance uses a
# T-dependent latent heat. A single state-based
# `Eᵢₛ = −ℵ * ρᵢ * ℒ * h * Az` cannot match both freeze at the ice-base
# temperature and top-melt at 0 ᵒC under `ℒ(T)`. Overriding `latent_heat`
# to the constant `pt.reference_latent_heat` isolates coupler-side
# bookkeeping errors from this intrinsic mismatch. The override is local to
# this test's process; it does not touch any package source.

@inline function ClimaSeaIce.SeaIceThermodynamics.latent_heat(
        pt::ClimaSeaIce.SeaIceThermodynamics.PhaseTransitions, T)
    return pt.reference_latent_heat
end

function test_coupled_energy_conservation(grid, atmosphere_grid; ocean_kwargs...)

    arch = architecture(grid)

    ocean = ocean_simulation(grid;
                             momentum_advection      = nothing,
                             closure                 = CATKEVerticalDiffusivity(),
                             coriolis                = nothing,
                             radiative_forcing       = nothing,
                             bottom_drag_coefficient = 0,
                             ocean_kwargs...)

    Tᵢ = -1.5
    Sᵢ = 34.0
    set!(ocean.model, T = Tᵢ, S = Sᵢ)

    sea_ice = sea_ice_simulation(grid, ocean;
                                 dynamics  = nothing,
                                 advection = nothing)

    set!(sea_ice.model, h = 1.0, ℵ = 1.0, hs = 0.10)

    atmosphere = PrescribedAtmosphere(atmosphere_grid, [0.0, 1e9])
    radiation  = PrescribedRadiation(atmosphere_grid, [0.0, 1e9])

    coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

    function set_forcing!(atmosphere, radiation;
                          T_air, q_air, u_air, v_air, p_air,
                          SW_down, LW_down, rain_flux, snow_flux)
        for (fts, value) in ((atmosphere.temperature,       T_air),
                             (atmosphere.specific_humidity, q_air),
                             (atmosphere.velocities.u, u_air),
                             (atmosphere.velocities.v, v_air),
                             (atmosphere.pressure,     p_air),
                             (atmosphere.precipitation_flux.rain, rain_flux),
                             (atmosphere.precipitation_flux.snow, snow_flux),
                             (radiation.downwelling_shortwave, SW_down),
                             (radiation.downwelling_longwave,  LW_down))
            fill!(parent(fts), value)
        end
        return nothing
    end

    ρᵢ  = coupled_model.sea_ice.model.sea_ice_density[1, 1, 1]
    ρₛ  = coupled_model.sea_ice.model.snow_density[1, 1, 1]
    ℒ₀  = ClimaSeaIce.SeaIceThermodynamics.latent_heat(coupled_model.sea_ice.model.phase_transitions, 0.0)
    ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density
    cᵒᶜ = coupled_model.interfaces.ocean_properties.heat_capacity

    Nz   = size(grid, 3)
    Az   = Azᶜᶜᶜ(1, 1, 1, grid)
    Aᵗᵒᵗ = Az * size(grid, 1) * size(grid, 2)
    Vᶜ   = sum(KernelFunctionOperation{Center, Center, Center}(volume, grid, Center(), Center(), Center()))

    # Built once and re-used every step via `compute!`; the operations keep a reference to the live
    # `T`/`S` fields, so the reductions track them and pick up the z-star volume as the grid moves.
    ∫T = Field(Integral(ocean.model.tracers.T))
    ∫S = Field(Integral(ocean.model.tracers.S))
    mean_S = Field(Average(ocean.model.tracers.S))

    # Area-integrate a surface field, or the product of two of them. The fluxes are horizontally
    # uniform here, but summing keeps the diagnostics identical for a single column and for a
    # horizontally resolved grid.
    surface(field) = vec(Array(interior(field)))
    ∫dA(field) = sum(surface(field)) * Az
    ∫dA(a, b) = sum(a .* b) * Az

    surface_salinity(ocean) = vec(Array(interior(ocean.model.tracers.S))[:, :, Nz])

    function column_state(coupled_model)
        h  = Array(interior(coupled_model.sea_ice.model.ice_thickness))
        ℵ  = Array(interior(coupled_model.sea_ice.model.ice_concentration))
        hs = Array(interior(coupled_model.sea_ice.model.snow_thickness))

        Mᵢₛ = sum(@. (ρᵢ * h + ρₛ * hs) * ℵ) * Az
        Eᵢₛ = -sum(@. ℵ * (ρᵢ * ℒ₀ * h + ρₛ * ℒ₀ * hs)) * Az
        Hₒ  = ρᵒᶜ * cᵒᶜ * first(Array(interior(compute!(∫T))))
        Cₒ  = first(Array(interior(compute!(∫S))))
        Sₒ  = first(Array(interior(compute!(mean_S))))

        return (; h = first(h), ℵ = first(ℵ), hs = first(hs), Mᵢₛ, Eᵢₛ, Hₒ, Cₒ, Sₒ)
    end

    function net_top_heat_flux(coupled_model)
        ΣQt  = ∫dA(coupled_model.interfaces.net_fluxes.sea_ice.top.heat)
        ΣQao = ∫dA(Field(atmosphere_ocean_heat_flux(coupled_model)))
        return -(ΣQt + ΣQao)
    end

    # Enthalpy the atmospheric freshwater carries into the ocean, `ρᵒᶜ cᵒᶜ ∮ Jᴴ dA` with `Jᴴ = Σᵢ Tᵢ Jʷᵢ`.
    # Only a mutable grid admits the freshwater volume, so only there does its enthalpy cross the boundary.
    freshwater_enthalpy_rate(coupled_model) = ρᵒᶜ * cᵒᶜ * ∫dA(coupled_model.interfaces.net_fluxes.ocean.freshwater_heat_content)

    # Short phases so the test stays under a few seconds of wall time.
    Δt = 10minutes
    Δτ = 2days      # phase duration
    Nsteps = Int(Δτ ÷ Δt)

    freeze_phase = (T_air     = 253.15,
                    q_air     = 1.0e-4,
                    u_air     = 2.0, v_air = 0.0,
                    p_air     = 101325.0,
                    SW_down   = 50.0,
                    LW_down   = 180.0,
                    rain_flux = 0.0,
                    snow_flux = 1.0e-5)

    melt_phase   = (T_air     = 278.15,
                    q_air     = 5.0e-3,
                    u_air     = 2.0, v_air = 0.0,
                    p_air     = 101325.0,
                    SW_down   = 250.0,
                    LW_down   = 320.0,
                    rain_flux = 5.0e-6,
                    snow_flux = 0.0)

    history = (t     = Float64[],
               phase = Int[],
               h     = Float64[],
               ℵ     = Float64[],
               hs    = Float64[],
               Mᵢₛ   = Float64[],
               Eᵢₛ   = Float64[],
               Hₒ    = Float64[],
               Cₒ    = Float64[],
               Sₒ    = Float64[],
               Jˢ    = Float64[],
               Sᴺ    = Vector{Float64}[],
               Jʷᴬ   = Vector{Float64}[],
               Jʷ    = Float64[],
               ∂tM   = Float64[],
               Ṁ     = Float64[],
               Q     = Float64[],
               Qᴴ    = Float64[],
               𝒬ᶠʳᶻ  = Float64[])

    # The intercepted snowfall the ocean loses (−Pₛᵃᵇˢ) and the ice gains (+Pₛᵃᵇˢ) cancel in the
    # combined ice+ocean budget, so Ṁ is the total atmospheric freshwater input to the coupled system.
    function freshwater_flux_state(coupled_model, rain_flux, snow_flux)
        Jʷfield = coupled_model.interfaces.net_fluxes.ocean.η
        Jˢ  = ∫dA(coupled_model.interfaces.net_fluxes.ocean.S)
        Jʷ  = ∫dA(Jʷfield)
        Jʷᴬ = surface(Jʷfield)
        Sᴺ  = surface_salinity(coupled_model.ocean)
        mass_fluxes = coupled_model.sea_ice.model.mass_fluxes
        ∂tM = ∫dA(mass_fluxes.thermodynamics.ice) + ∫dA(mass_fluxes.thermodynamics.snow) +
              ∫dA(mass_fluxes.intercepted_snowfall)
        Jᵛ  = surface(coupled_model.interfaces.atmosphere_ocean_interface.fluxes.water_vapor)
        ℵ   = surface(coupled_model.sea_ice.model.ice_concentration)
        Ṁ   = (rain_flux + snow_flux) * Aᵗᵒᵗ - ∫dA(1 .- ℵ, Jᵛ)   # rain and snow reach the ocean in full
        return (; Jˢ, Sᴺ, Jʷᴬ, Jʷ, ∂tM, Ṁ)
    end

    function record!(history, coupled_model, phase_id, rain_flux, snow_flux, Q)
        st = column_state(coupled_model)
        fw = freshwater_flux_state(coupled_model, rain_flux, snow_flux)
        𝒬f = ∫dA(Field(frazil_heat_flux(coupled_model)))
        push!(history.t,     coupled_model.clock.time)
        push!(history.phase, phase_id)
        push!(history.h,     st.h)
        push!(history.ℵ,     st.ℵ)
        push!(history.hs,    st.hs)
        push!(history.Mᵢₛ,  st.Mᵢₛ)
        push!(history.Eᵢₛ,  st.Eᵢₛ)
        push!(history.Hₒ,   st.Hₒ)
        push!(history.Cₒ,   st.Cₒ)
        push!(history.Sₒ,   st.Sₒ)
        push!(history.Jˢ,   fw.Jˢ)
        push!(history.Sᴺ,   fw.Sᴺ)
        push!(history.Jʷᴬ,  fw.Jʷᴬ)
        push!(history.Jʷ,   fw.Jʷ)
        push!(history.∂tM,  fw.∂tM)
        push!(history.Ṁ,     fw.Ṁ)
        push!(history.Q,     Q)
        push!(history.Qᴴ,    freshwater_enthalpy_rate(coupled_model))
        push!(history.𝒬ᶠʳᶻ,  𝒬f)
        return nothing
    end

    record!(history, coupled_model, 0, 0.0, 0.0, 0.0)

    # The phase-boundary `update_state!` would zero the pending frazil flux
    # written at the end of the previous phase's final step, stranding the
    # latent energy that was already deposited into the ocean by that same
    # frazil mutation. We preserve `𝒬ᶠʳᶻ` across the refresh and add it
    # back into the sea-ice bottom heat flux that the slab will read.
    function run_phase!(coupled_model, spec, phase_id; Nsteps, Δt, history, atmosphere, radiation)
        set_forcing!(atmosphere, radiation;
                     T_air = spec.T_air, q_air = spec.q_air,
                     u_air = spec.u_air, v_air = spec.v_air,
                     p_air = spec.p_air,
                     SW_down = spec.SW_down, LW_down = spec.LW_down,
                     rain_flux = spec.rain_flux, snow_flux = spec.snow_flux)

        Qᵖ = - spec.snow_flux * ℒ₀ * Aᵗᵒᵗ                                    # snowfall enthalpy

        𝒬ᶠʳᶻ = coupled_model.interfaces.sea_ice_ocean_interface.fluxes.frazil_heat
        ΣQb  = coupled_model.interfaces.net_fluxes.sea_ice.bottom.heat
        𝒬⁻   = Array(interior(𝒬ᶠʳᶻ))                                         # pending frazil
        update_state!(coupled_model)
        interior(𝒬ᶠʳᶻ) .= on_architecture(arch, 𝒬⁻)
        interior(ΣQb)  .+= on_architecture(arch, 𝒬⁻)

        # The last record's fluxes apply to the upcoming interval, so recompute them under this
        # phase's forcing (the phase-boundary `update_state!` reassembled them).
        fw = freshwater_flux_state(coupled_model, spec.rain_flux, spec.snow_flux)
        history.Q[end]   = net_top_heat_flux(coupled_model) + Qᵖ
        history.Qᴴ[end]  = freshwater_enthalpy_rate(coupled_model)
        history.Jˢ[end]  = fw.Jˢ
        history.Sᴺ[end]  = fw.Sᴺ
        history.Jʷᴬ[end] = fw.Jʷᴬ
        history.Jʷ[end]  = fw.Jʷ
        history.∂tM[end] = fw.∂tM
        history.Ṁ[end]   = fw.Ṁ

        for _ in 1:Nsteps
            time_step!(coupled_model, Δt)
            Q = net_top_heat_flux(coupled_model) + Qᵖ
            record!(history, coupled_model, phase_id, spec.rain_flux, spec.snow_flux, Q)
        end
    end

    run_phase!(coupled_model, freeze_phase, 1; Nsteps, Δt, history, atmosphere, radiation)
    run_phase!(coupled_model, melt_phase,   2; Nsteps, Δt, history, atmosphere, radiation)

    t = history.t

    # A rate recorded at step n drives the ocean over the interval (n, n+1], which is how the
    # coupler applies the fluxes it assembled at the end of step n.
    accumulate_rate(rate) = [n == 1 ? 0.0 : sum(rate[m] * (t[m+1] - t[m]) for m in 1:(n-1)) for n in 1:length(t)]

    Δt⁺ = similar(t)
    for n in 1:(length(t) - 1)
        Δt⁺[n] = t[n+1] - t[n]
    end
    Δt⁺[end] = Δt⁺[end-1]

    # --- Energy budget ---

    # On a mutable grid the freshwater enters the ocean and carries its enthalpy `ρᵒᶜ cᵒᶜ Jᴴ` with it, so
    # that enthalpy is an energy input alongside the surface heat flux. A static grid admits no freshwater
    # volume, so nothing is carried and the surface heat flux is the only input.
    ∫Q = accumulate_rate(history.Q)
    if grid isa MutableGridOfSomeKind
        ∫Q .+= accumulate_rate(history.Qᴴ)
    end

    δE = history.𝒬ᶠʳᶻ .* Δt⁺

    Ẽᵢₛ = history.Eᵢₛ .+ δE
    ΔE = (Ẽᵢₛ .+ history.Hₒ) .- (Ẽᵢₛ[1] + history.Hₒ[1])
    Rₑ = ΔE .- ∫Q

    εₑ = abs(Rₑ[end]) / max(maximum(abs.(ΔE)), 1)
    @test εₑ < 1e-10

    # --- Freshwater budget ---
    #
    # The ocean freshwater mass is ρᵒᶜ ∫Jʷ, where Jʷ = η is the assembled freshwater volume flux
    # (rain + snow + runoff − evaporation + sea-ice melt) — one explicit stream, no salinity weighting.
    # The ice mass is a state that already moved during the final step, while the freshwater it
    # released is assembled at the end of that step and reaches the ocean only on the next one, so
    # the last step's exchange is still pending and is subtracted via ∂tM.

    ∫Ṁ = accumulate_rate(history.Ṁ)
    Mᶠʷ = ρᵒᶜ .* accumulate_rate(history.Jʷ)

    pending = history.∂tM[end] * Δt⁺[end]
    ΔM = (history.Mᵢₛ .+ Mᶠʷ) .- history.Mᵢₛ[1]
    Rₘ = ΔM .- ∫Ṁ
    εₘ = abs(Rₘ[end] - pending) / max(maximum(abs.(ΔM)), 1)
    @test εₘ < 1e-10

    # --- Salt budget ---
    #
    # On a mutable grid the volume the freshwater adds carries Sᴺ Jʷ back in, cancelling the virtual
    # salt flux at every Runge-Kutta stage, so the salt content follows the sea-ice salt alone. A
    # static grid admits no volume and the virtual flux stands: the surface salinity it sees sweeps
    # from Sᴺ[n] to Sᴺ[n+1] across the step while the assembled Jʷ stays frozen, so the trapezoid in
    # Sᴺ is the quadrature that recovers the flux the Runge-Kutta stages actually applied.
    ∫Jˢ = accumulate_rate(history.Jˢ)[end]
    if !(grid isa MutableGridOfSomeKind)
        ∫Jˢ += sum(∫dA(0.5 .* (history.Sᴺ[n] .+ history.Sᴺ[n+1]), history.Jʷᴬ[n]) * Δt⁺[n]
                   for n in 1:(length(t) - 1))
    end

    ΔC = history.Cₒ[end] - history.Cₒ[1]
    Sᵣ = history.Sₒ[1]
    @test abs(ΔC + ∫Jˢ) / (Sᵣ * Vᶜ) < 1e-12

    # --- Sanity checks on the physics ---

    nᶠ = findlast(p -> p == 1, history.phase)

    @test history.h[nᶠ] > history.h[1]   # ice grew during freeze
    @test history.h[end] < history.h[nᶠ] # ice shrank during melt
    @test history.ℵ[nᶠ] ≈ 1.0            # stays fully ice-covered through freeze

    return nothing
end

@testset "Coupled energy conservation" begin
    # Four levels is the floor: the default tracer advection needs a halo of 4, which cannot exceed
    # the number of cells it spans.
    Nz = 4

    for arch in test_architectures
        A = typeof(arch)

        # A static ocean cannot change volume, so the freshwater flux never forces `η` and the
        # temperature exchange in the tracer top BC stays inert.
        @testset "Static grid [$A]" begin
            @info "Testing coupled energy conservation on a static grid [$A]..."
            grid = RectilinearGrid(arch; size=Nz, z=(-100, 0), topology=(Flat, Flat, Bounded))
            atmosphere_grid = RectilinearGrid(arch; size=(), topology=(Flat, Flat, Flat))
            test_coupled_energy_conservation(grid, atmosphere_grid; tracer_advection=nothing)
        end

        # A mutable ocean grows with the freshwater it takes in. This needs a free surface for `η` to
        # respond to `Jʷ`, and tracer advection, which carries the z-star grid-motion term that the
        # live `Tᴺ Jʷ` exchange in the tracer top BCs cancels.
        @testset "Mutable grid [$A]" begin
            @info "Testing coupled energy conservation on a mutable ZStar grid [$A]..."
            Lx = Ly = 1e5
            grid = RectilinearGrid(arch;
                                   size     = (4, 4, Nz),
                                   halo     = (4, 4, 4),
                                   x        = (0, Lx), y = (0, Ly),
                                   z        = MutableVerticalDiscretization((-100, 0)),
                                   topology = (Periodic, Periodic, Bounded))

            atmosphere_grid = RectilinearGrid(arch;
                                              size     = (4, 4, 1),
                                              x        = (0, Lx), y = (0, Ly), z = (-1, 0),
                                              topology = (Periodic, Periodic, Bounded))

            test_coupled_energy_conservation(grid, atmosphere_grid;
                                             free_surface = SplitExplicitFreeSurface(substeps=30))
        end
    end
end

@testset "Rain increases the ocean volume and dilutes salinity" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing the freshwater volume flux on a ZStar grid [$A]..."

        Lx = Ly = 1e5
        grid = RectilinearGrid(arch;
                               size = (4, 4, 4),
                               halo = (4, 4, 4),
                               x = (0, Lx), y = (0, Ly),
                               z = MutableVerticalDiscretization((-10, 0)),
                               topology = (Periodic, Periodic, Bounded))

        ocean = ocean_simulation(grid;
                                 momentum_advection = nothing,
                                 closure = nothing,
                                 free_surface = SplitExplicitFreeSurface(substeps=30),
                                 radiative_forcing = nothing,
                                 bottom_drag_coefficient = 0)

        S₀ = 35.0
        set!(ocean.model, T=20, S=S₀)

        atmosphere_grid = RectilinearGrid(arch;
                                          size = (4, 4, 1),
                                          x = (0, Lx), y = (0, Ly), z = (-1, 0),
                                          topology = (Periodic, Periodic, Bounded))

        # A quiescent atmosphere (default state, zero winds) with heavy rain, so that
        # the rainfall dominates the small evaporative flux into the dry atmosphere
        atmosphere = PrescribedAtmosphere(atmosphere_grid, [0.0])
        rainfall = 1e-2 # kg m⁻² s⁻¹
        fill!(parent(atmosphere.precipitation_flux.rain), rainfall)

        coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation=nothing)
        update_state!(coupled_model)

        # The net freshwater volume flux assembled at the surface is (rain - evaporation) / ρᵒᶜ
        ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density
        Jʷ = coupled_model.interfaces.net_fluxes.ocean.η
        Jᵛ = coupled_model.interfaces.atmosphere_ocean_interface.fluxes.water_vapor
        evaporation = first(Array(interior(Jᵛ)))
        @test all(Array(interior(Jʷ)) .≈ (rainfall - evaporation) / ρᵒᶜ)

        S = ocean.model.tracers.S
        cell_volume = KernelFunctionOperation{Center, Center, Center}(volume, grid, Center(), Center(), Center())
        ocean_volume() = sum(cell_volume)
        total_salt() = sum(S * cell_volume)

        V⁻ = ocean_volume()
        ∫S⁻ = total_salt()

        # Accumulate the surface-integrated freshwater volume flux ∫∫ Jʷ dA dt over the run (the
        # flux is spatially uniform here); comparing against a single-sample flux × elapsed time
        # would only close to O(Δt) as the evaporative flux drifts.
        Δt = 2minutes
        Nsteps = 30
        expected_volume_change = 0.0
        for _ in 1:Nsteps
            freshwater_volume_flux = first(Array(interior(Jʷ)))
            time_step!(coupled_model, Δt)
            expected_volume_change += Lx * Ly * freshwater_volume_flux * ocean.model.clock.last_Δt
        end

        V = ocean_volume()
        ∫S = total_salt()

        # Rain increases the ocean volume by exactly the integrated freshwater volume flux.
        @test V > V⁻
        @test isapprox(V - V⁻, expected_volume_change, rtol=1e-10)

        # It carries no salt: the live virtual salt flux cancels the salt the volume change advects
        # in, so the total salt content is conserved to machine precision...
        @test abs(∫S - ∫S⁻) < 1e-10 * S₀ * (V - V⁻)

        # ... and therefore dilutes the mean salinity as pure volume growth.
        @test ∫S / V < S₀
        @test isapprox(∫S / V, S₀ * V⁻ / V, rtol=1e-10)
    end
end
