using NumericalEarth
using NumericalEarth.EN4
using NumericalEarth.EN4: download_dataset
using NumericalEarth.WOA: WOAAnnual

using Oceananigans
using CUDA
using Oceananigans.Units
using Dates
using Printf
using Statistics

function run_one_degree_diagnostic(output_name, arch = GPU();
                                   grid = default_one_degree_grid(arch),
                                   ocean_kwargs =  default_ocean_configuration(grid),
                                   initial_conditions = nothing,
                                   atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(100), include_rivers_and_icebergs=true),
                                   Δt = 30minutes,
                                   stop_time = 500days)

    arch = Oceananigans.Architectures.architecture(grid)
    Nz = size(grid, 3)

    ocean = ocean_simulation(grid; Δt, ocean_kwargs...)
    @info "Built ocean simulation" grid=summary(grid) ocean_kwargs

    # --- Initial conditions ---
    if isnothing(initial_conditions)
        date = DateTime(1965, 1, 1)
        dataset = EN4Monthly()
        set!(ocean.model, T = Metadatum(:temperature; dir="./", date, dataset),
                          S = Metadatum(:salinity;    dir="./", date, dataset))
        @info "Set default EN4 initial conditions for 1965-01-01"
    else
        set!(ocean.model; initial_conditions...)
        @info "Set custom initial conditions" keys(initial_conditions)
    end

    if !isnothing(atmosphere)
        # --- Sea ice ---
        sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=5, minimum_buffer_upwind_order=1))
        set!(sea_ice.model, h = Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly()),
                            ℵ = Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly()))
        @info "Built sea ice simulation with ECCO4 initial conditions"

        # --- Atmosphere ---
        radiation = Radiation(arch)
        @info "Built atmosphere and radiation"

        # --- Coupled model ---
        coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
        simulation = Simulation(coupled_model; Δt, stop_time)
    else
        simulation = ocean
    end
        
    # --- Progress callback ---
    wall_time = Ref(time_ns())

    function progress(sim)
        ocean = try
            sim.model.ocean
        catch 
            sim
        end
        u, v, w = ocean.model.velocities
        T, S = ocean.model.tracers
        η = ocean.model.free_surface.displacement

        Trange = (minimum(T), maximum(T))
        Srange = (minimum(S), maximum(S))
        ηrange = (minimum(η), maximum(η))
        umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))
        step_time = 1e-9 * (time_ns() - wall_time[])

        msg = @sprintf("time: %s, iter: %d, Δt: %s, max|u|: (%.2e, %.2e, %.2e) m/s, T: (%.2f, %.2f) °C, S: (%.2f, %.2f) g/kg, η: (%.2f, %.2f) m, wall: %s",
                       prettytime(sim), iteration(sim), prettytime(sim.Δt),
                       umax..., Trange..., Srange..., ηrange..., prettytime(step_time))
        @info msg

        wall_time[] = time_ns()
        return nothing
    end

    add_callback!(simulation, progress, IterationInterval(10))

    ocean.output_writers[:checkpointer2] = Checkpointer(ocean.model;
                                                        schedule = TimeInterval(150days),
                                                        prefix = output_name * "_checkpoint",
                                                        overwrite_existing = true)
    # --- Run ---
    @info "Starting simulation" output_name stop_time
    run!(simulation)
    @info "Simulation complete" output_name

    return simulation
end

"""
    default_ocean_configuration(grid)

Return a `NamedTuple` of default keyword arguments for `ocean_simulation`,
matching the test_omip.jl configuration.
"""
function default_ocean_configuration(grid)
    Nz = size(grid, 3)
    z_faces = grid.z
    z_surf = CUDA.@allowscalar znodes(grid, Center())[Nz]

    # Salinity restoring to EN4
    dates = collect(DateTime(1961, 1, 1):Month(1):DateTime(1963, 5, 1))
    
    dataset = EN4Monthly()
    salinity = Metadata(:salinity; dates, dataset)

    restoring_rate = 1 / 30days
    @inline mask(x, y, z, t) = z ≥ z_surf - 1
    FS = DatasetRestoring(salinity, grid; mask, rate=restoring_rate, time_indices_in_memory=10)

    tracer_advection   = WENO(order=5)
    momentum_advection = WENOVectorInvariant(order=5)
    free_surface       = SplitExplicitFreeSurface(grid; substeps=150)

    @inline νhb(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, λ) = Oceananigans.Operators.Az(i, j, k, grid, ℓx, ℓy, ℓz)^2 / λ
    horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true, parameters=20days)
    catke = NumericalEarth.Oceans.default_ocean_closure()
    closure = (catke, VerticalScalarDiffusivity(κ=1e-5, ν=1e-4), horizontal_viscosity)


    return (;
        closure,
        free_surface,
        momentum_advection,
        tracer_advection,
        timestepper = :SplitRungeKutta3,
        forcing = (; S=FS),
        radiative_forcing = nothing,
    )
end

"""
    default_one_degree_grid(arch=GPU(); Nz=75, depth=6000meters)

Build the default 1° tripolar grid with bathymetry matching test_omip.jl.
"""
function default_one_degree_grid(arch=GPU(); Nz=75, depth=6000meters, mutable=true, active_cells_map=true)
    Nx, Ny = 360, 180
    z = ExponentialDiscretization(Nz, -depth, 0; mutable, scale=1800)
    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), z, halo=(7, 7, 7))

    bottom_height = regrid_bathymetry(underlying_grid;
                                      minimum_depth=15,
                                      interpolation_passes=24,
                                      major_basins=4)

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map)
    return grid
end

