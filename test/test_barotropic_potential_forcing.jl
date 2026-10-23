include("runtests_setup.jl")

using NumericalEarth.Oceans: BarotropicPotentialForcing, XDirection, forcing_barotropic_potential
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.TimeSteppers: update_state!

# The potential a `TimeInterpolatedPotential` gives the ocean at time `τ`
function potential_at(Φ, τ)
    t₁, t₂ = Array(Φ.times)
    χ = t₂ == t₁ ? 0.0 : (τ - t₁) / (t₂ - t₁)
    return (1 - χ) .* Array(interior(Φ.previous)) .+ χ .* Array(interior(Φ.next))
end

x_forcing(potential, grid, clock) =
    Array(interior(compute!(Field(KernelFunctionOperation{Face, Center, Center}(BarotropicPotentialForcing(XDirection(), potential), grid, clock, nothing)))))

@testset "Barotropic potential forcing" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (16, 12, 1), longitude = (0, 40), latitude = (-30, 30), z = (-1, 0))
        clock = Clock(time = 5.0)

        Φ(λ, φ, t) = 1e-2 * sind(3λ) * cosd(2φ) + 1e-6 * t
        Φᶠ = Field{Center, Center, Nothing}(grid)
        set!(Φᶠ, (λ, φ) -> Φ(λ, φ, clock.time))

        # Interior faces, whose stencil does not reach the extrapolated halo of the field
        @test x_forcing(Φ, grid, clock)[2:end-1, :, :] ≈ x_forcing(Φᶠ, grid, clock)[2:end-1, :, :]
        @test x_forcing((Φᶠ, Φᶠ), grid, clock) ≈ 2 * x_forcing(Φᶠ, grid, clock)

        interpolated = TimeInterpolatedPotential(grid)
        set!(interpolated.previous, Φᶠ)
        set!(interpolated.next, 3 * Φᶠ)
        copyto!(interpolated.times, [0.0, 10.0])
        @test x_forcing(interpolated, grid, clock) ≈ 2 * x_forcing(Φᶠ, grid, clock)
    end
end

@testset "Cached ocean tendencies see the atmospheric pressure at the start of the step" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (16, 12, 4), halo = (7, 7, 7), longitude = (0, 40), latitude = (-30, 30), z = (-1000, 0))
        ocean = ocean_simulation(grid; warn = false)
        atmosphere = PrescribedAtmosphere(grid, [0.0, 3600.0])
        set!(atmosphere.pressure[1], (λ, φ, z) -> 101325 + 300 * sind(4λ))
        set!(atmosphere.pressure[2], (λ, φ, z) -> 101325 + 300 * cosd(6φ))
        coupled_model = OceanOnlyModel(ocean; atmosphere)

        Δt = 600
        time_step!(coupled_model, Δt)

        Gⁿ = ocean.model.timestepper.Gⁿ
        cached_Gu = Array(interior(Gⁿ.u))
        cached_Gv = Array(interior(Gⁿ.v))
        update_state!(ocean.model)

        @test cached_Gu == Array(interior(Gⁿ.u))
        @test cached_Gv == Array(interior(Gⁿ.v))
    end
end

@testset "Barotropic potential from a windowed pressure time series" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (16, 12, 1), halo = (7, 7, 7), longitude = (0, 40), latitude = (-30, 30), z = (-1000, 0))
        times = 60 * (0:10)
        path = tempname() * ".jld2"

        p_in_memory = FieldTimeSeries{Center, Center, Center}(grid, times)
        set!(p_in_memory, (λ, φ, z, t) -> 101325 + 300 * sind(4λ + t / 10) * cosd(3φ))
        p_on_disk = FieldTimeSeries{Center, Center, Center}(grid, times; backend = OnDisk(), path, name = "p")
        for n in eachindex(times)
            set!(p_on_disk, p_in_memory[n], n)
        end

        for Δt in (45, 25)
            pressure = FieldTimeSeries(path, "p"; architecture = arch, backend = InMemory(3))
            atmosphere = PrescribedAtmosphere(grid, times; pressure)
            ocean = ocean_simulation(grid; warn = false)
            coupled_model = OceanOnlyModel(ocean; atmosphere)
            Φ = forcing_barotropic_potential(ocean)
            ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density

            for n in 1:floor(Int, (last(times) - Δt) / Δt)
                time_step!(coupled_model, Δt)
                t = coupled_model.clock.time
                t₁, t₂ = Array(Φ.times)
                # Exact anywhere between the bracketing snapshots, including the current time
                for τ in (t, (t₁ + t₂) / 2)
                    @test potential_at(Φ, τ) ≈ Array(interior(p_in_memory[Oceananigans.Units.Time(τ)])) ./ ρᵒᶜ
                end
            end

            @test all(isfinite, interior(ocean.model.velocities.u))
        end

        rm(path)
    end
end

@testset "Barotropic potential across the wrap of a cyclical pressure series" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (16, 12, 4), halo = (7, 7, 7), longitude = (0, 40), latitude = (-30, 30), z = (-1000, 0))
        ocean = ocean_simulation(grid; warn = false)
        atmosphere = PrescribedAtmosphere(grid, [0.0, 3600.0])
        set!(atmosphere.pressure[1], (λ, φ, z) -> 101325 + 300 * sind(4λ))
        set!(atmosphere.pressure[2], (λ, φ, z) -> 101325 + 300 * cosd(6φ))
        coupled_model = OceanOnlyModel(ocean; atmosphere)
        Φ = forcing_barotropic_potential(ocean)
        ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density

        # The period is 7200 s, so snapshot 2 → 1 brackets times in (3600, 7200)
        for n in 1:10
            time_step!(coupled_model, 900)
            t = coupled_model.clock.time
            @test potential_at(Φ, t) ≈ Array(interior(atmosphere.pressure[Oceananigans.Units.Time(t)], :, :, 1:1)) ./ ρᵒᶜ
        end
    end
end
