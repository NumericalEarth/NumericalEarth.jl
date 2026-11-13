using Test
using Reactant
using Oceananigans: Oceananigans, interior, set!
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Bounded, LatitudeLongitudeGrid
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth
using Breeze

include("setup.jl")
include("patches.jl")

# Defined here, not inside the testset: a closure over the testset's locals makes Reactant compile it as a closure,
# whose generated `Thunk` call alone costs ~20 GB to compile on the first call (out of memory on CI's runners)
initialize_from_parent!(model) =
    Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt).initialize_nested_child!(model; balancer = false)

# On `ReactantState` construction leaves the exchanger's window empty; one compiled
# `initialize_nested_child!` fills it and initializes the child from it.
@testset "Nested atmosphere: compiled initialization and time_step!" begin
    arch  = ReactantState()
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

    @info "Compiling the initialization (exchanger window and child)..."
    compile_time = @elapsed compiled_initialize! = @with_stack_size Reactant.@compile sync=true initialize_from_parent!(model)
    @info "Compiled the initialization in $(round(compile_time, digits=1)) s (peak memory $(peak_memory())); running it..."
    @with_stack_size compiled_initialize!(model)
    @info "Ran the initialization (peak memory $(peak_memory()))"
    @test all(>(0), Array(interior(model.exchanger.prognostic.ρᵈ[1])))
    @test all(>(0), Array(interior(model.child.dynamics.reference_state.pressure)))
    @test all(isfinite, Array(interior(model.child.momentum.ρu)))
    @test all(>(0), Array(interior(model.child.dynamics.dry_density)))

    Δt = 0.5
    @info "Compiling (raised) time_step!..."
    compile_time = @elapsed compiled_step! = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true time_step!(model, Δt)
    @info "Compiled time_step! in $(round(compile_time, digits=1)) s (peak memory $(peak_memory())); running one step..."
    @with_stack_size compiled_step!(model, Δt)
    @info "Ran one step (peak memory $(peak_memory()))"
    @test Reactant.to_number(model.clock.time) ≈ Δt
    @test all(isfinite, Array(interior(model.child.momentum.ρu)))
    @info "Nested atmosphere Reactant test done"
end
