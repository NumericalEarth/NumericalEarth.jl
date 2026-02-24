include("runtests_setup.jl")

using Test
using NumericalEarth
using Oceananigans
using PythonCall, CondaPkg

@testset "Veros ocean model interface" begin
    # Test that the Veros extension is available
    VerosModule = Base.get_extension(NumericalEarth, :NumericalEarthVerosExt)
    @test !isnothing(VerosModule)
    
    # Test Veros installation
    @test begin
        VerosModule.install_veros()
        true
    end
    
    # Test creating a VerosOceanSimulation
    # Using a small setup for testing
    @test begin
        ocean = VerosModule.VerosOceanSimulation("global_4deg", :GlobalFourDegreeSetup)
        @test ocean isa VerosModule.VerosOceanSimulation
        @test !isnothing(ocean.setup)
        true
    end
    
    # Test basic interface functions
    ocean = VerosModule.VerosOceanSimulation("global_4deg", :GlobalFourDegreeSetup)
    
    # Test interface
    ρᵒᶜ = NumericalEarth.EarthSystemModels.reference_density(ocean)
    cᵒᶜ = NumericalEarth.EarthSystemModels.heat_capacity(ocean)
    @test ρᵒᶜ isa Real
    @test cᵒᶜ isa Real
    
    T = NumericalEarth.EarthSystemModels.ocean_temperature(ocean)
    S = NumericalEarth.EarthSystemModels.ocean_salinity(ocean)
    
    @test !isnothing(T)
    @test !isnothing(S)
    
    @test length(size(T)) == 3
    @test length(size(S)) == 3
    
    S_surf = NumericalEarth.EarthSystemModels.ocean_surface_salinity(ocean)
    u_surf, v_surf = NumericalEarth.EarthSystemModels.ocean_surface_velocities(ocean)
    
    @test !isnothing(S_surf)
    @test !isnothing(u_surf)
    @test !isnothing(v_surf)
    
    # Test surface grid
    grid = VerosModule.surface_grid(ocean)
    @test grid isa Oceananigans.Grids.LatitudeLongitudeGrid
    
    # Test time stepping
    @test begin
        Δt = 60.0
        VerosModule.time_step!(ocean, Δt)
        true
    end
    
    # Clean up output files
    VerosModule.remove_outputs(:global_4deg)
end
