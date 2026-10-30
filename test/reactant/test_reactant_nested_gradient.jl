using Test
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using Oceananigans: Oceananigans, CenterField, interior, set!, prognostic_fields
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Bounded, LatitudeLongitudeGrid
using Oceananigans.TimeSteppers: Clock, time_step!, update_state!
using NumericalEarth
using Breeze
using CUDA

include("patches.jl")

#####
##### Reactant — gradient through one hour of a nested Breeze model
#####
#
# A semi-realistic regional forecast: a 6° × 6° Breeze child at ≈ 17 km with 24 levels to 16 km and
# saturation-adjusted moisture, nested in an analytic parent with a meridional temperature gradient and
# a sheared westerly. The test takes the gradient of the child's domain-mean squared temperature after
# one hour (360 steps of 10 s) with respect to a perturbation of its initial ρθ: Enzyme's reverse mode
# through a checkpointed traced loop, compiled with Reactant. GPU only (GPU_TEST=true): on the CPU the
# compile and the hour are too slow for CI.

gpu_test = get(ENV, "GPU_TEST", "false") == "true"

if !gpu_test
    @info "Skipping the nested gradient test, which runs on the GPU only (GPU_TEST=true)"
else
    Reactant.set_default_backend("gpu")

    @testset "Nested atmosphere: gradient through one hour" begin
        arch  = ReactantState()
        ext   = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
        times = [0.0, 1800.0, 3600.0]

        parent_grid = LatitudeLongitudeGrid(arch; size = (24, 24, 16),
                                            longitude = (-6, 6), latitude = (30, 42),
                                            z = (0, 16000), halo = (5, 5, 5),
                                            topology = (Bounded, Bounded, Bounded))
        parent_atmosphere = PrescribedAtmosphere(parent_grid, times)
        set!(parent_atmosphere.temperature,       (λ, φ, z, t) -> 288 - 6.5e-3 * z - 0.5 * (φ - 36))
        set!(parent_atmosphere.specific_humidity, (λ, φ, z, t) -> 0.012 * exp(-z / 2500))
        set!(parent_atmosphere.velocities.u,      (λ, φ, z, t) -> 5 + 1.5e-3 * z)
        set!(parent_atmosphere.velocities.v,      (λ, φ, z, t) -> 0)
        set!(parent_atmosphere.pressure,          (λ, φ, z, t) -> 1e5 * exp(-z / 8000))

        child_grid = LatitudeLongitudeGrid(arch; size = (32, 32, 24),
                                           longitude = (-3, 3), latitude = (33, 39),
                                           z = (0, 16000), halo = (5, 5, 5),
                                           topology = (Bounded, Bounded, Bounded))

        # A fixed number of acoustic substeps: the adaptive count does not trace. The device clock makes the
        # model time advance through the compiled loop; the exchanger holds every parent snapshot.
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps = 6); base_pressure = 1e5)
        model = nested_atmosphere_model(parent_atmosphere, child_grid; dynamics, clock = Clock(child_grid),
                                        time_indices_in_memory = length(times),
                                        relaxation_rate = 1/300, relaxation_width = 4,
                                        coriolis = nothing, terrain = nothing, parent_condensates = nothing)

        @info "Compiling and running initialize! and initialize_nested_child!..."
        compiled_initialize! = Reactant.@compile sync=true Oceananigans.initialize!(model)
        compiled_initialize!(model)
        initialize_child!(model) = ext.initialize_nested_child!(model; balancer = false)
        compiled_initialize_child! = Reactant.@compile sync=true initialize_child!(model)
        compiled_initialize_child!(model)

        function loss(model, δρθ, Δt, Nsteps)
            ρθ = prognostic_fields(model.child).ρθ
            parent(ρθ) .+= parent(δρθ)
            update_state!(model)
            @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
                time_step!(model, Δt)
            end
            return mean(interior(model.child.temperature) .^ 2)
        end

        function grad_loss(model, dmodel, δρθ, dδρθ, Δt, Nsteps)
            parent(dδρθ) .= 0
            _, J = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                                   loss, Enzyme.Active,
                                   Enzyme.Duplicated(model, dmodel),
                                   Enzyme.Duplicated(δρθ, dδρθ),
                                   Enzyme.Const(Δt),
                                   Enzyme.Const(Nsteps))
            return dδρθ, J
        end

        δρθ    = CenterField(child_grid)
        dδρθ   = CenterField(child_grid)
        dmodel = Enzyme.make_zero(model)
        Δt     = 10.0
        Nsteps = 360

        # As Breeze's AD benchmark: these two patterns make the gradient compile run out of host memory or
        # fail (NumericalEarth/Breeze.jl#1151, EnzymeAD/Reactant.jl#3573)
        compile_options = Reactant.CompileOptions(; raise = true, raise_first = true, sync = true,
                                                  excluded_passes = ["greedy_while_loop_batch_fission",
                                                                     "while_elementwise_reduction_to_reduce"])

        @info "Compiling the gradient of one hour ($Nsteps steps of $Δt s)..."
        compile_time = @elapsed compiled_grad = Reactant.@compile compile_options = compile_options grad_loss(
            model, dmodel, δρθ, dδρθ, Δt, Nsteps)
        @info "Compiled the gradient in $(round(compile_time, digits = 1)) s; running it..."
        ∇J, J = compiled_grad(model, dmodel, δρθ, dδρθ, Δt, Nsteps)
        ∇ = Array(interior(∇J))

        @test isfinite(J)
        @test J > 0
        @test all(isfinite, ∇)
        @test maximum(abs, ∇) > 0
        @test Reactant.to_number(model.clock.time) ≈ Nsteps * Δt
    end
end
