include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels: OceanSeaIceModel
using NumericalEarth.SeaIces: sea_ice_simulation, sea_ice_dynamics
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.Fields: interior
using Statistics: mean

# The sea ice and the ocean exchange momentum through a drag that each side integrates implicitly in
# its own velocity. If both sides read the *other* velocity from the previous step, the exchange is
# simultaneous and its amplification factor is `1/(1+a) + 1/(1+b) - 1`, which approaches -1 as the
# drag stiffens: the velocity difference survives, flipping sign every step. Stepping the ocean first
# lets the ice drag read the ocean velocity of the step it shares, and the factor becomes
# `1/((1+a)(1+b))` — positive and small, so the difference decays without ringing. Here
# `a = λ Δt / mᵢ` and `b = λ Δt / (ρᵒ Δz)`.
#
# This test drives the exchange deliberately stiff (a ≈ 200, b ≈ 120) and pins the absence of ringing,
# which is what distinguishes the two sequencings.

@testset "Ice-ocean drag decays without ringing" begin
    for arch in test_architectures
        @info "Testing ice-ocean momentum sequencing on $(typeof(arch))..."

        # A thin top cell gives the ocean little inertia against the ice, which is the stiff limit.
        grid = LatitudeLongitudeGrid(arch,
                                     size = (8, 8, 4),
                                     z = [-200, -60, -15, -1.5, 0],
                                     latitude = (50, 58),
                                     longitude = (0, 8),
                                     halo = (7, 7, 7))

        ocean = ocean_simulation(grid;
                                 coriolis = nothing,
                                 momentum_advection = nothing,
                                 bottom_drag_coefficient = 0)

        set!(ocean.model, T = -1.8, S = 34.0, u = 0.5)

        # No basal stress or Coriolis, and uniform ice, so the drag is the only momentum term acting.
        dynamics = sea_ice_dynamics(grid, ocean;
                                    sea_ice_ocean_drag_coefficient = 0.05,
                                    basal_stress = nothing,
                                    coriolis = nothing)

        sea_ice = sea_ice_simulation(grid, ocean; dynamics)
        set!(sea_ice.model, h = 1.0, ℵ = 1.0)
        set!(sea_ice.model.velocities.u, -0.5)

        atmosphere = PrescribedAtmosphere(grid, [0.0, 1e9])
        radiation  = PrescribedRadiation(grid, [0.0, 1e9])
        for component in (atmosphere.velocities.u, atmosphere.velocities.v)
            parent(component) .= 0
        end

        coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

        kᴺ = size(grid, 3)
        surface_difference() =
            mean(Array(interior(sea_ice.model.velocities.u))) -
            mean(Array(interior(ocean.model.velocities.u))[:, :, kᴺ])

        Δu = [surface_difference()]
        for _ in 1:6
            time_step!(coupled_model, 3600)
            push!(Δu, surface_difference())
        end

        @test all(isfinite, Δu)

        # The drag can only remove the velocity difference, never reverse it. A simultaneous exchange
        # flips the sign on the first step, so this is the assertion that separates the two.
        @test all(≥(0), Δu .* Δu[1])

        # A stiff exchange read sequentially annihilates most of the difference in a single step;
        # read simultaneously it survives nearly intact.
        @test abs(Δu[2]) < 0.2 * abs(Δu[1])

        # ... and stays collapsed. The residual is a small physical equilibrium between the drag and
        # the remaining momentum terms, not a decaying oscillation about zero.
        @test all(<(0.2 * abs(Δu[1])), abs.(Δu[2:end]))
    end
end
