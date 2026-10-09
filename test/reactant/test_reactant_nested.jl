using Test
using Reactant
using Oceananigans: Oceananigans, interior, set!
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Bounded, LatitudeLongitudeGrid
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth
using Breeze
using CUDA

gpu_test = get(ENV, "GPU_TEST", "false") == "true"
Reactant.set_default_backend(gpu_test ? "gpu" : "cpu")

# Construction only allocates on `ReactantState`; `initialize!` is compiled, the child setup is eager.
@testset "Nested atmosphere: compiled initialize!, initialize_nested_child! and time_step!" begin
    arch  = ReactantState()
    ext   = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
    times = [0.0, 4.0, 8.0]

    parent_grid = LatitudeLongitudeGrid(arch; size = (12, 12, 8),
                                        longitude = (-2, 2), latitude = (34.6, 38.6),
                                        z = (0, 16000), halo = (5, 5, 5),
                                        topology = (Bounded, Bounded, Bounded))
    parent = PrescribedAtmosphere(parent_grid, times)
    set!(parent.temperature,       (λ, φ, z, t) -> 288 - 6.5e-3 * z + 1e-3 * t)
    set!(parent.specific_humidity, (λ, φ, z, t) -> 0.006)
    set!(parent.velocities.u,      (λ, φ, z, t) -> 8)
    set!(parent.velocities.v,      (λ, φ, z, t) -> 0)
    set!(parent.pressure,          (λ, φ, z, t) -> 1e5 * exp(-z / 8000))

    child_grid = LatitudeLongitudeGrid(arch; size = (8, 8, 8),
                                       longitude = (-1, 1), latitude = (35.6, 37.6),
                                       z = (0, 16000), halo = (5, 5, 5),
                                       topology = (Bounded, Bounded, Bounded))

    @info "Constructing the nested atmosphere model..."
    # A fixed number of acoustic substeps: the adaptive count does not trace.
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps = 3); base_pressure = 1e5)
    model = nested_atmosphere_model(parent, child_grid; dynamics,
                relaxation_rate = 1/300, relaxation_width = 3,
                coriolis = nothing, terrain = nothing, parent_condensates = nothing)

    @test all(iszero, Array(interior(model.exchanger.prognostic.ρᵈ[1])))
    @test all(iszero, Array(interior(model.child.dynamics.reference_state.pressure)))

    @info "Compiling initialize!..."
    compile_time = @elapsed compiled_initialize! = Reactant.@compile sync=true initialize!(model)
    @info "Compiled initialize! in $(round(compile_time, digits=1)) s; running it..."
    compiled_initialize!(model)
    @test all(>(0), Array(interior(model.exchanger.prognostic.ρᵈ[1])))
    @test all(>(0), Array(interior(model.child.dynamics.reference_state.pressure)))

    initialize_child!(model) = ext.initialize_nested_child!(model; balancer = false)
    @info "Compiling initialize_nested_child!..."
    compile_time = @elapsed compiled_initialize_child! = Reactant.@compile sync=true initialize_child!(model)
    @info "Compiled initialize_nested_child! in $(round(compile_time, digits=1)) s; running it..."
    compiled_initialize_child!(model)
    @test all(isfinite, Array(interior(model.child.momentum.ρu)))
    @test all(>(0), Array(interior(model.child.dynamics.dry_density)))

    Δt = 0.5
    @info "Compiling (raised) time_step!..."
    compile_time = @elapsed compiled_step! = Reactant.@compile raise=true raise_first=true sync=true time_step!(model, Δt)
    @info "Compiled time_step! in $(round(compile_time, digits=1)) s; running one step..."
    compiled_step!(model, Δt)
    @test Reactant.to_number(model.clock.time) ≈ Δt
    @test all(isfinite, Array(interior(model.child.momentum.ρu)))
    @info "Nested atmosphere Reactant test done"
end
