using Test
using Reactant
using Oceananigans: Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Bounded, Flat, LatitudeLongitudeGrid, Periodic
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: SplitExplicitFreeSurface
using NumericalEarth
using CUDA

gpu_test = get(ENV, "GPU_TEST", "false") == "true"

if gpu_test
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

@testset "Reactant extension tests" begin
    arch = ReactantState()
    grid = LatitudeLongitudeGrid(arch;
                                 size = (256, 128, 10),
                                 longitude = (0, 360),
                                 latitude = (-80, 80),
                                 halo = (7, 7, 7),
                                 z = (-6000, 0))

    free_surface = SplitExplicitFreeSurface(substeps=10)
    ocean = ocean_simulation(grid; Δt=300, free_surface)

    # We use an idealized atmosphere to avoid downloading the whole JRA55 data
    atmos_grid  = LatitudeLongitudeGrid(arch, Float32; size=(320, 200), 
                                                       latitude=(-90, 90), 
                                                       longitude=(0, 360), 
                                                       topology=(Periodic, Bounded, Flat))

    atmos_times = range(0, 360Oceananigans.Units.days, length=10)
    atmosphere  = PrescribedAtmosphere(atmos_grid, atmos_times)

    coupled_model = OceanOnlyModel(ocean; atmosphere)

    # reconcile_state! initializes the exchanger regridder at construction (via @jit on Reactant).
    exchanger = coupled_model.interfaces.exchanger.atmosphere
    state     = exchanger.state
    regridder = exchanger.regridder
    @test any(regridder.i .!= 0)
    @test any(regridder.j .!= 0)

    # update_state! populates the exchange state with the interpolated air temperature
    @test any(state.T .!= 0)
end
