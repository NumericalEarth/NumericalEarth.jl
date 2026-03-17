include("runtests_setup.jl")

using CUDA
using Oceananigans.OrthogonalSphericalShellGrids
using NumericalEarth.EarthSystemModels: above_freezing_ocean_temperature!
using ClimaSeaIce.SeaIceDynamics
using ClimaSeaIce.SeaIceThermodynamics: melting_temperature
using ClimaSeaIce.Rheologies

@inline kernel_melting_temperature(i, j, k, grid, liquidus, S) = @inbounds melting_temperature(liquidus, S[i, j, k])

@testset "Time stepping test" begin
    for dataset in [ECCO4Monthly(), EN4Monthly()]

        start_date = DateTimeProlepticGregorian(1993, 1, 1)
        time_resolution = dataset isa ECCO2Daily ? Day(1) : Month(1)
        end_date = DateTimeProlepticGregorian(1993, 2, 1)
        dates = start_date : time_resolution : end_date

        temperature_metadata = Metadata(:temperature; dataset, dates)
        salinity_metadata    = Metadata(:salinity; dataset, dates)

        for arch in test_architectures

            A = typeof(arch)

            @info "Testing timestepping with $(typeof(dataset)) on $A"

            #####
            ##### Coupled ocean-sea ice and prescribed atmosphere
            #####

            sea_ice  = sea_ice_simulation(grid, ocean; advection=WENO(order=7))
            liquidus = sea_ice.model.ice_thermodynamics.phase_transitions.liquidus

            # Set the ocean temperature and salinity
            set!(ocean.model, T=temperature_metadata[1], S=salinity_metadata[1])
            above_freezing_ocean_temperature!(ocean, grid, sea_ice)

            # Test that ocean temperatures are above freezing
            T = on_architecture(CPU(), ocean.model.tracers.T)
            S = on_architecture(CPU(), ocean.model.tracers.S)

            Tm = KernelFunctionOperation{Center, Center, Center}(kernel_melting_temperature, grid, liquidus, S)
            @test all(T .≥ Tm)

            # Fluxes are computed when the model is constructed, so we just test that this works.
            # And that we can time step with sea ice
            @test begin
                coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
                time_step!(coupled_model, 1)
                true
            end
        end
    end
end
