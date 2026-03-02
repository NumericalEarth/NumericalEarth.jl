include("runtests_setup.jl")

using SeawaterPolynomials: TEOS10EquationOfState
using Oceananigans: location
using Oceananigans.Models: buoyancy_operation
using NumericalEarth.Diagnostics: MixedLayerDepthField, MixedLayerDepthOperand

#=
for arch in test_architectures, dataset in (ECCO4Monthly(),)
    A = typeof(arch)
    @info "Testing MixedLayerDepthField with $(typeof(dataset)) on $A"

    @testset "MixedLayerDepthField" begin
        grid = LatitudeLongitudeGrid(arch;
                                     size = (3, 3, 100),
                                     latitude  = (0, 30),
                                     longitude = (150, 180),
                                     z = (-1000, 0))

        bottom_height = regrid_bathymetry(grid;
                                          minimum_depth = 10,
                                          interpolation_passes = 5,
                                          major_basins = 1)

        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

        start = DateTimeProlepticGregorian(1993, 1, 1)
        stop  = DateTimeProlepticGregorian(1993, 2, 1)
        dates = range(start; stop, step=Month(1))

        Tmeta = Metadata(:temperature; dataset, dates)
        Smeta = Metadata(:salinity; dataset, dates)

        Tt = FieldTimeSeries(Tmeta, grid; time_indices_in_memory=2)
        St = FieldTimeSeries(Smeta, grid; time_indices_in_memory=2)

        equation_of_state = TEOS10EquationOfState()
        sb = SeawaterBuoyancy(; equation_of_state)
        tracers = (T=Tt[1], S=St[1])
        h = MixedLayerDepthField(sb, grid, tracers)

        @test h isa Field
        @test location(h) == (Center, Center, Nothing)
        @test h.operand isa MixedLayerDepthOperand
        @test h.operand.buoyancy_perturbation isa KernelFunctionOperation

        compute!(h)
        if dataset isa ECCO4Monthly
            @test @allowscalar h[1, 1, 1] ≈ 16.2558363 # m
        end

        tracers = (T=Tt[2], S=St[2])
        h.operand.buoyancy_perturbation = buoyancy_operation(sb, grid, tracers)
        compute!(h)
        if dataset isa ECCO4Monthly
            @test @allowscalar h[1, 1, 1] ≈ 9.2957298 # m
        end
    end
end
=#

for arch in test_architectures
    A = typeof(arch)
    @info "Testing InterfaceFluxOutputs on $A"

    @testset "InterfaceFluxOutputs on $A" begin
        grid = RectilinearGrid(arch;
                               size = (4, 4, 2),
                               extent = (1, 1, 1),
                               topology = (Periodic, Periodic, Bounded))

        ocean = ocean_simulation(grid;
                                 momentum_advection = nothing,
                                 tracer_advection = nothing,
                                 closure = nothing,
                                 coriolis = nothing,
                                 bottom_drag_coefficient = 0.0)

        sea_ice = sea_ice_simulation(grid, ocean)
        atmosphere = PrescribedAtmosphere(grid, [0.0])
        coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation = Radiation())

        T_flux = ocean.model.tracers.T.boundary_conditions.top.condition
        S_flux = ocean.model.tracers.S.boundary_conditions.top.condition
        ice_ocean_fluxes = coupled_model.interfaces.sea_ice_ocean_interface.fluxes

        T_flux_value = 2.0
        S_flux_value = 5.0
        frazil_heat_flux_value = 0.2
        interface_heat_flux_value = 0.3
        ice_ocean_salt_flux_value = 0.9

        fill!(T_flux, T_flux_value)
        fill!(S_flux, S_flux_value)
        fill!(ice_ocean_fluxes.frazil_heat, frazil_heat_flux_value)
        fill!(ice_ocean_fluxes.interface_heat, interface_heat_flux_value)
        fill!(ice_ocean_fluxes.salt, ice_ocean_salt_flux_value)

        ρ₀ = coupled_model.interfaces.ocean_properties.reference_density
        cₚ = coupled_model.interfaces.ocean_properties.heat_capacity
        S₀ = 35.0

        flux_outputs = interface_flux_outputs(coupled_model)
        @test keys(flux_outputs) == (:heat_flux, :freshwater_flux)

        compute!(flux_outputs.heat_flux)
        compute!(flux_outputs.freshwater_flux)

        @test location(flux_outputs.heat_flux) == (Center, Center, Nothing)
        @test location(flux_outputs.freshwater_flux) == (Center, Center, Nothing)

        @allowscalar begin
            @test flux_outputs.heat_flux[1, 1, 1] ≈ ρ₀ * cₚ * (T_flux_value + frazil_heat_flux_value)
            @test flux_outputs.freshwater_flux[1, 1, 1] ≈ -ρ₀ * S_flux_value / S₀
        end

        split_outputs = interface_flux_outputs(coupled_model; separate_sea_ice = true)
        @test keys(split_outputs) == (:heat_flux,
                                      :ocean_heat_flux,
                                      :sea_ice_heat_flux,
                                      :freshwater_flux,
                                      :ocean_freshwater_flux,
                                      :sea_ice_freshwater_flux)

        for fld in values(split_outputs)
            compute!(fld)
        end

        @allowscalar begin
            @test split_outputs.ocean_heat_flux[1, 1, 1] ≈ ρ₀ * cₚ * (T_flux_value - interface_heat_flux_value)
            @test split_outputs.sea_ice_heat_flux[1, 1, 1] ≈ ρ₀ * cₚ * (frazil_heat_flux_value + interface_heat_flux_value)
            @test split_outputs.ocean_freshwater_flux[1, 1, 1] ≈ -ρ₀ * (S_flux_value - ice_ocean_salt_flux_value) / S₀
            @test split_outputs.sea_ice_freshwater_flux[1, 1, 1] ≈ -ρ₀ * ice_ocean_salt_flux_value / S₀
            @test split_outputs.heat_flux[1, 1, 1] ≈ split_outputs.ocean_heat_flux[1, 1, 1] + split_outputs.sea_ice_heat_flux[1, 1, 1]
            @test split_outputs.freshwater_flux[1, 1, 1] ≈ split_outputs.ocean_freshwater_flux[1, 1, 1] + split_outputs.sea_ice_freshwater_flux[1, 1, 1]
        end

        tracer_outputs = interface_flux_outputs(coupled_model; separate_sea_ice = true, units = TracerFluxUnits())
        @test keys(tracer_outputs) == keys(split_outputs)
        for fld in values(tracer_outputs)
            compute!(fld)
        end

        @allowscalar begin
            @test tracer_outputs.heat_flux[1, 1, 1] ≈ T_flux_value + frazil_heat_flux_value
            @test tracer_outputs.freshwater_flux[1, 1, 1] ≈ S_flux_value
            @test tracer_outputs.ocean_heat_flux[1, 1, 1] ≈ T_flux_value - interface_heat_flux_value
            @test tracer_outputs.sea_ice_heat_flux[1, 1, 1] ≈ frazil_heat_flux_value + interface_heat_flux_value
            @test tracer_outputs.ocean_freshwater_flux[1, 1, 1] ≈ S_flux_value - ice_ocean_salt_flux_value
            @test tracer_outputs.sea_ice_freshwater_flux[1, 1, 1] ≈ ice_ocean_salt_flux_value
        end
    end
end
