# ERA5 → Breeze NestedModel on ReactantState: build the nest and compile one time step.
#
#   julia --project=experiments/reactant_nested_era5 experiments/reactant_nested_era5/debug_compile.jl
#
# ERA5 is fetched through the Copernicus Climate Data Store on the first run (credentials in
# ~/.cdsapirc or CDSAPI_URL / CDSAPI_KEY) and cached in ./era5.

using NumericalEarth
using Oceananigans
using Oceananigans.Architectures: ReactantState, on_architecture
using Breeze
using CopernicusClimateDataStore   # ERA5 downloader extension
using Reactant, CUDA               # CUDA loads the Reactant KernelAbstractions extension
using Dates: DateTime, Hour

Reactant.set_default_backend("cpu")
Oceananigans.defaults.FloatType = Float32

arch = ReactantState()

# Monkey patch: the ReactantState lat-lon grid constructor builds on CPU and moves each grid field
# through `_to_reactant`, which only knows StaticVerticalDiscretization. The ERA5 parent's native
# grid carries a PressureLevelVerticalDiscretization; route it through NumericalEarth's own
# `on_architecture` (geopotential FieldTimeSeries + surface geopotential Field → XLA buffers).
const OceananigansReactantExt = Base.get_extension(Oceananigans, :OceananigansReactantExt)
OceananigansReactantExt.Architectures._to_reactant(z::NumericalEarth.Grids.PressureLevelVerticalDiscretization) =
    on_architecture(ReactantState(), z)

# Monkey patch: Oceananigans decides whether a lat-lon boundary is polar by reading a latitude node
# (`φsouth ≈ -90`), a host branch on a traced value inside a compiled program. That makes allocating any
# Field on the grid (ours and Breeze's) impossible under `@compile`. This regional grid is never polar,
# so the topology-only default is the exact answer.
using Oceananigans.BoundaryConditions: default_auxiliary_bc, default_prognostic_bc, _default_auxiliary_bc
using Oceananigans.Grids: topology
const ReactantLatLonGrid = OceananigansReactantExt.Grids.ReactantStateLatitudeLongitudeGrid
for side in (:north, :south)
    @eval begin
        Oceananigans.BoundaryConditions.default_auxiliary_bc(grid::ReactantLatLonGrid, ::Val{$(QuoteNode(side))}, (ℓx, ℓy, ℓz)) =
            _default_auxiliary_bc(topology(grid, 2)(), ℓy)
        Oceananigans.BoundaryConditions.default_prognostic_bc(grid::ReactantLatLonGrid, ::Val{$(QuoteNode(side))}, (ℓx, ℓy, ℓz), default) =
            default_prognostic_bc(topology(grid, 2)(), ℓy, default)
    end
end

# Monkey patch: inside a program a Field's element type is `TracedRNumber`, so Oceananigans' reduced-Field
# `getindex`/`setindex!` (which drop the reduced index) are ambiguous with Reactant's `getindex` for
# arrays of traced numbers. Breeze's `set!` scalar-reads column means under `@allowscalar`. Same bodies
# as Oceananigans' `src/Fields/field.jl`, on traced reduced fields with concrete index types.
# Specificity has to beat both neighbors, and Reactant resolves the call with the compiler's `findsup`,
# which rejects any tie: reduced locations are the literal `Nothing` (a `<:Nothing` bound ties with
# Oceananigans' `Field{Nothing, Nothing}`), and the element type is `TracedRNumber{T}` with a free `T`
# (a `<:TracedRNumber` bound ties with Reactant's `AbstractArray{TracedRNumber{T}, N}`).
const TracedIndex = Union{Int, Reactant.TracedRNumber{Int}}
for (LX, LY, LZ, ii, jj, kk) in ((:Nothing, :Any,     :Any,     1, :j, :k),
                                 (:Any,     :Nothing, :Any,     :i, 1, :k),
                                 (:Any,     :Any,     :Nothing, :i, :j, 1),
                                 (:Any,     :Nothing, :Nothing, :i, 1, 1),
                                 (:Nothing, :Any,     :Nothing, 1, :j, 1),
                                 (:Nothing, :Nothing, :Any,     1, 1, :k),
                                 (:Nothing, :Nothing, :Nothing, 1, 1, 1))
    location(L) = L == :Nothing ? :Nothing : :(<:Any)
    TracedField = :(Field{$(location(LX)), $(location(LY)), $(location(LZ)), <:Any, <:Any, <:Any, <:Any, Reactant.TracedRNumber{T}})
    @eval begin
        Base.@propagate_inbounds Base.getindex(r::$TracedField, i::TracedIndex, j::TracedIndex, k::TracedIndex) where T =
            getindex(r.data, $ii, $jj, $kk)
        Base.@propagate_inbounds Base.setindex!(r::$TracedField, v, i::TracedIndex, j::TracedIndex, k::TracedIndex) where T =
            setindex!(r.data, v, $ii, $jj, $kk)
    end
end

# Monkey patch: `set_to_mean!` reduces the z = 0 datum to the bottom face with the horizontal-mean θ and qᵛ
# as scalars, which Breeze routes through its step-doubling integrator: a `z == 0` early return and a
# convergence test, both host branches on traced values. A constant moist profile has the same closed form
# the dry `θ₀::Number` method already dispatches to, so this is exact, not an approximation.
using Breeze.Thermodynamics: constant_moist_hydrostatic_pressure, dry_air_gas_constant, vapor_gas_constant
Breeze.Thermodynamics.moist_hydrostatic_pressure(z, p₀, θ₀::Number, qᵛ₀::Number, pˢᵗ, constants) =
    constant_moist_hydrostatic_pressure(z, p₀, θ₀, qᵛ₀, pˢᵗ,
                                        dry_air_gas_constant(constants), vapor_gas_constant(constants),
                                        constants.dry_air.heat_capacity, constants.vapor.heat_capacity,
                                        constants.gravitational_acceleration)

# Monkey patch: the same two functions `convert` their traced scalar results to the reference's float type,
# and a traced number has no conversion back to a host `Float32`. Every input already carries that type, so
# the converts are no-ops in eager mode. Same bodies as Breeze's `src/AtmosphereModels/set_to_mean.jl`.
using Breeze.Thermodynamics: bottom_face_height, moist_hydrostatic_pressure, set_surface_state!, surface_reference_density
function Breeze.AtmosphereModels.update_exner_surface_state!(ref::ExnerReferenceState, θ, qᵛ, grid, constants)
    Rᵈ  = dry_air_gas_constant(constants)
    Rᵛ  = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    θˢ, qᵛˢ = @allowscalar (θ[1, 1, 1], qᵛ[1, 1, 1])
    zˢ = bottom_face_height(grid)
    pˢ = moist_hydrostatic_pressure(zˢ, ref.base_pressure, θˢ, qᵛˢ, ref.standard_pressure, constants)
    set_surface_state!(ref.surface_pressure, pˢ)
    Breeze.AtmosphereModels.update_exner_surface_density!(ref, pˢ, θˢ, qᵛˢ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    return nothing
end
function Breeze.AtmosphereModels.update_exner_surface_density!(ref::ExnerReferenceState, pˢ, θˢ, qᵛˢ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    isnothing(ref.surface_density) && return nothing
    ρˢ = surface_reference_density(pˢ, θˢ, qᵛˢ, ref.standard_pressure, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    set_surface_state!(ref.surface_density, ρˢ)
    return nothing
end

start_date = DateTime(2011, 5, 20, 0)
dates = (start_date, start_date + Hour(2))   # 3 hourly snapshots
λ₀, φ₀ = -97.485, 36.605                     # ARM SGP
Nx = Ny = Nz = 8
Δt = 1.0

dataset = ERA5HourlyPressureLevels(pressure_levels = [1000, 925, 850, 700, 500, 300, 200, 100] .* 100)   # Pa

grid = LatitudeLongitudeGrid(arch;
                             longitude = (λ₀ - 1, λ₀ + 1),
                             latitude  = (φ₀ - 1, φ₀ + 1),
                             z = (0, 12_000),
                             size = (Nx, Ny, Nz),
                             halo = (5, 5, 5),
                             topology = (Bounded, Bounded, Bounded))

# Manual construction, timed stage by stage. On ReactantState every eager kernel launch is a
# separate `@jit` compile, so the timings show where construction spends its time. This mirrors
# `nested_atmosphere_model(grid, dataset; dates, …)` step for step.
const BreezeExt = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
era5_dir = joinpath(@__DIR__, "era5")
region = BoundingBox(grid; padding = 0.5)

@info "1. Parent: ERA5 PrescribedAtmosphere on ReactantState (download + load + halo fills)"
@time era5_parent = PrescribedAtmosphere(region, dates, dataset; architecture = arch, dir = era5_dir)

@info "2. Surface-pressure anchor: domain-mean sea-level pressure regridded onto the child"
@time surface_pressure = BreezeExt.mean_sea_level_pressure(dataset, grid, start_date, era5_dir)

# `acoustic_substeps`: a compiled step needs a fixed acoustic substep count (Breeze sizes it from the
# acoustic CFL every stage otherwise). 3 is ample for Δt = 1 s on ~25 km cells.
@info "3. Exchanger (derived prognostics on the parent grid) + child AtmosphereModel"
@time nest = nested_atmosphere_model(era5_parent, grid; surface_pressure, acoustic_substeps = 3)

# The initialization (parent→child interpolation, Breeze set! with the reference state, the consistent-w
# state refresh) compiled as ONE program, instead of an eager compile per kernel launch.
initialize_child!(nest) = BreezeExt.initialize_nested_child!(nest, dataset, start_date, era5_dir; balancer = false)

@info "4a. Compiling the child initialization"
@time compiled_initialize! = Reactant.@compile raise=true raise_first=true sync=true initialize_child!(nest)

@info "4b. Running it"
@time compiled_initialize!(nest)

@info "5. Compiling one time step"
@time compiled_step! = Reactant.@compile raise=true raise_first=true sync=true time_step!(nest, Δt)

@info "6. Running it"
@time compiled_step!(nest, Δt)
@info "Done" time = Reactant.to_number(nest.clock.time)
