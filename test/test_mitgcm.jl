include("runtests_setup.jl")

using Test
using NumericalEarth
using Oceananigans
using MITgcm

# Path to pre-built library from MITgcm.jl examples
const MITGCM_JL_DIR = pkgdir(MITgcm)
const EXAMPLE_DIR = joinpath(MITGCM_JL_DIR, "examples", "global_oce_latlon")
const LIB_PATH = joinpath(EXAMPLE_DIR, Sys.isapple() ? "libmitgcm.dylib" : "libmitgcm.so")
const RUN_DIR = joinpath(EXAMPLE_DIR, "run")

if !isfile(LIB_PATH) || !isdir(RUN_DIR)
    @warn "Pre-built MITgcm library not found at $EXAMPLE_DIR. Skipping MITgcm tests."
else

@testset "MITgcm ocean model interface" begin
    # Test that the extension is available
    MITgcmExt = Base.get_extension(NumericalEarth, :NumericalEarthMITgcmExt)
    @test !isnothing(MITgcmExt)

    # Test creating a MITgcmOceanSimulation from pre-built library
    ocean = MITgcmOceanSimulation(LIB_PATH, RUN_DIR)
    @test ocean isa MITgcmOceanSimulation

    # Test dimensions
    @test size(ocean.theta) == (90, 40, 15)
    @test size(ocean.salt) == (90, 40, 15)
    @test size(ocean.uvel) == (90, 40, 15)
    @test size(ocean.vvel) == (90, 40, 15)
    @test size(ocean.etan) == (90, 40)
    @test size(ocean.xc) == (90, 40)
    @test size(ocean.yc) == (90, 40)
    @test length(ocean.rc) == 15
    @test length(ocean.drf) == 15

    # Test basic interface functions
    ρₒ = NumericalEarth.EarthSystemModels.reference_density(ocean)
    cₚ = NumericalEarth.EarthSystemModels.heat_capacity(ocean)
    @test ρₒ ≈ 1029.0
    @test cₚ ≈ 3994.0

    # Test ocean state accessors
    T = NumericalEarth.EarthSystemModels.ocean_temperature(ocean)
    S = NumericalEarth.EarthSystemModels.ocean_salinity(ocean)
    @test T === ocean.theta
    @test S === ocean.salt
    @test ndims(T) == 3
    @test ndims(S) == 3

    T_surf = NumericalEarth.EarthSystemModels.ocean_surface_temperature(ocean)
    S_surf = NumericalEarth.EarthSystemModels.ocean_surface_salinity(ocean)
    @test size(T_surf) == (90, 40)
    @test size(S_surf) == (90, 40)

    u_surf, v_surf = NumericalEarth.EarthSystemModels.ocean_surface_velocities(ocean)
    @test size(u_surf) == (90, 40)
    @test size(v_surf) == (90, 40)

    # Test surface grid construction
    grid = MITgcmExt.surface_grid(ocean)
    @test grid isa Oceananigans.Grids.LatitudeLongitudeGrid

    # Test time stepping
    theta_before = copy(ocean.theta)
    MITgcmExt.time_step!(ocean, 43200.0)  # 12 hours
    @test ocean.theta != theta_before  # State should change

    # Test a few more steps
    for _ in 1:3
        MITgcmExt.time_step!(ocean, 43200.0)
    end
    @test MITgcm.get_niter(ocean.library) == 4

    # Test with PrescribedAtmosphere and EarthSystemModel
    @testset "Coupled model with PrescribedAtmosphere" begin
        grid = MITgcmExt.surface_grid(ocean)
        Nx, Ny = size(ocean.xc)

        atmosphere = NumericalEarth.PrescribedAtmosphere(grid)

        coupled_model = NumericalEarth.EarthSystemModels.OceanOnlyModel(ocean; atmosphere)
        @test coupled_model isa NumericalEarth.EarthSystemModels.EarthSystemModel

        # Initialize and step the coupled model
        NumericalEarth.EarthSystemModels.initialize!(coupled_model)
        NumericalEarth.EarthSystemModels.time_step!(coupled_model, 43200.0)

        @test MITgcm.get_niter(ocean.library) == 5
    end

    # Finalize
    MITgcm.finalize!(ocean.library)
end

end # if pre-built library exists
