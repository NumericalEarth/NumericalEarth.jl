include("runtests_setup.jl")

using Oceananigans.OrthogonalSphericalShellGrids
using NumericalEarth.EarthSystemModels: above_freezing_ocean_temperature!
using NumericalEarth.EarthSystemModels.InterfaceComputations: kernel_radiation_properties,
                                                             air_sea_interface_radiation_state,
                                                             air_sea_ice_interface_radiation_state
using ClimaSeaIce.SeaIceThermodynamics: ConductiveFlux
using Oceananigans.Units: Time

# Float64 parameters in a Float32 coupled model made the sea-ice interface temperature iteration type-unstable,
# which the GPU compiler rejects (InvalidIRError in `_compute_atmosphere_sea_ice_interface_state!`).
# The default sea-ice conductive flux and the radiative surface properties must follow the grid's float type.
function all_conductive_fluxes(x, depth = 0)
    x isa ConductiveFlux && return Any[x]
    skip = depth > 6 || x isa Union{AbstractArray, Oceananigans.Fields.AbstractField, Oceananigans.Grids.AbstractGrid,
                                    Number, Function, DataType, Module}
    (skip || !isstructtype(typeof(x))) && return Any[]
    return reduce(vcat, (all_conductive_fluxes(getfield(x, f), depth + 1) for f in fieldnames(typeof(x)) if isdefined(x, f));
                  init = Any[])
end

@testset "Float32 ocean-sea ice coupled model" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing a Float32 OceanSeaIceModel on $A..."

        grid = TripolarGrid(arch, Float32; size = (40, 40, 8), halo = (7, 7, 7), z = (-5000, 0))
        bottom_height = synthetic_bottom_height(grid; minimum_depth = 10, interpolation_passes = 5, major_basins = 1)
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map = true)

        free_surface = SplitExplicitFreeSurface(grid; substeps = 20)
        ocean = ocean_simulation(grid; free_surface)
        sea_ice = sea_ice_simulation(grid, ocean)

        set!(ocean.model, MetadataSet(:temperature, :salinity; dataset = SyntheticOcean(), date = DateTime(2000, 1, 1)))
        above_freezing_ocean_temperature!(ocean, grid, sea_ice)
        set!(sea_ice.model, h = 1, ℵ = 0.5)

        fluxes = all_conductive_fluxes(sea_ice.model.ice_thermodynamics)
        @test !isempty(fluxes)
        @test all(f -> f.conductivity isa Float32, fluxes)

        atmosphere = synthetic_prescribed_atmosphere(arch)
        radiation = synthetic_prescribed_radiation(arch)
        coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

        # The radiation state seen by the ocean and sea-ice interface kernels is in the grid's float type
        rk = kernel_radiation_properties(coupled_model.radiation, coupled_model.interfaces.exchanger.radiation)
        exchanger_state = coupled_model.interfaces.exchanger.radiation.state
        Nz = size(grid, 3)
        for radiation_state in (air_sea_interface_radiation_state, air_sea_ice_interface_radiation_state)
            state = @allowscalar radiation_state(rk, exchanger_state, 1, 1, Nz, grid, Time(0))
            @test all(x -> x isa Float32, values(state))
        end

        time_step!(coupled_model, 600)

        @test eltype(coupled_model.interfaces.atmosphere_sea_ice_interface.temperature) == Float32
        @test eltype(ocean.model.tracers.T) == Float32
        @test eltype(sea_ice.model.ice_thickness) == Float32
    end
end
