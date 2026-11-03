include("runtests_setup.jl")

using Test
using NumericalEarth
using Oceananigans
using PythonCall, CondaPkg

@testset "Veros ocean model interface" begin
    VerosModule = Base.get_extension(NumericalEarth, :NumericalEarthVerosExt)
    @test !isnothing(VerosModule)

    VerosModule.install_veros()

    ocean = VerosModule.VerosOceanSimulation("global_4deg", :GlobalFourDegreeSetup)

    ρᵒᶜ = NumericalEarth.EarthSystemModels.reference_density(ocean)
    cᵒᶜ = NumericalEarth.EarthSystemModels.heat_capacity(ocean)
    @test ρᵒᶜ isa Real
    @test cᵒᶜ isa Real

    T = NumericalEarth.EarthSystemModels.ocean_temperature(ocean)
    S = NumericalEarth.EarthSystemModels.ocean_salinity(ocean)
    @test ndims(T) == 3
    @test ndims(S) == 3

    S_surf = NumericalEarth.EarthSystemModels.ocean_surface_salinity(ocean)
    u_surf, v_surf = NumericalEarth.EarthSystemModels.ocean_surface_velocities(ocean)
    @test ndims(S_surf) == 2
    @test ndims(u_surf) == 2
    @test ndims(v_surf) == 2

    grid = VerosModule.surface_grid(ocean)
    @test grid isa Oceananigans.Grids.LatitudeLongitudeGrid

    time_step!(ocean, 60.0)

    VerosModule.remove_outputs(:global_4deg)
end
