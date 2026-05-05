include("runtests_setup.jl")

using NumericalEarth.Radiations: PrescribedRadiation,
                                 SurfaceRadiationProperties,
                                 InterfaceRadiationFlux

@testset "PrescribedRadiation construction" begin
    for arch in test_architectures
        A = typeof(arch)

        # Form B: grid-only constructor (zero downwelling, surface properties only)
        @info "Testing PrescribedRadiation(grid) on $A..."
        grid = RectilinearGrid(arch, size = 10, z = (-100, 0), topology = (Flat, Flat, Bounded))
        rad = PrescribedRadiation(grid)
        @test rad isa PrescribedRadiation
        @test rad.surface_properties isa NamedTuple
        @test haskey(rad.surface_properties, :ocean)
        @test haskey(rad.surface_properties, :sea_ice)
        @test rad.surface_properties.ocean isa SurfaceRadiationProperties
        @test rad.surface_properties.ocean.albedo == 0.05
        @test rad.surface_properties.ocean.emissivity == 0.97
        @test rad.surface_properties.sea_ice.albedo == 0.7
        @test rad.surface_properties.sea_ice.emissivity == 1.0
        @test rad.stefan_boltzmann_constant ≈ 5.67e-8 atol=1e-10
        @test isnothing(rad.interface_fluxes)

        # Surfaces can be omitted
        @info "Testing PrescribedRadiation(grid; sea_ice_surface=nothing) on $A..."
        rad_ocean_only = PrescribedRadiation(grid; sea_ice_surface = nothing)
        @test haskey(rad_ocean_only.surface_properties, :ocean)
        @test !haskey(rad_ocean_only.surface_properties, :sea_ice)

        # Custom surface properties
        custom_ocean = SurfaceRadiationProperties(0.1, 0.95)
        rad_custom = PrescribedRadiation(grid; ocean_surface = custom_ocean)
        @test rad_custom.surface_properties.ocean.albedo == 0.1
        @test rad_custom.surface_properties.ocean.emissivity == 0.95

        # time_step! works
        @info "Testing time_step!(::PrescribedRadiation) on $A..."
        rad2 = PrescribedRadiation(grid)
        time_step!(rad2, 60.0)
        @test rad2.clock.time == 60.0
    end
end

@testset "PrescribedRadiation paired with model" begin
    for arch in test_architectures
        A = typeof(arch)

        @info "Testing OceanOnlyModel with PrescribedRadiation on $A..."

        grid = RectilinearGrid(arch, size = 10, z = (-100, 0), topology = (Flat, Flat, Bounded))
        ocean = ocean_simulation(grid)
        radiation = PrescribedRadiation(grid)
        model = OceanOnlyModel(ocean; radiation)

        # interface_fluxes are allocated for present surfaces (ocean + sea_ice
        # via FreezingLimitedOceanTemperature).
        @test !isnothing(model.radiation.interface_fluxes)
        @test model.radiation.interface_fluxes.ocean isa InterfaceRadiationFlux
        @test model.radiation.interface_fluxes.sea_ice isa InterfaceRadiationFlux

        time_step!(model, 60)
        @test iteration(model) == 1
    end
end

@testset "JRA55PrescribedRadiation" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing JRA55PrescribedRadiation on $A..."

        backend = JRA55NetCDFBackend(2)
        radiation = JRA55PrescribedRadiation(arch; backend)

        @test radiation isa PrescribedRadiation
        @test radiation.downwelling_shortwave isa FieldTimeSeries
        @test radiation.downwelling_longwave isa FieldTimeSeries
        @test radiation.surface_properties.ocean isa SurfaceRadiationProperties
    end
end
