include("runtests_setup.jl")

using Oceananigans: location
using Oceananigans.Models: buoyancy_operation
using NumericalEarth.Diagnostics: MixedLayerDepthField, MixedLayerDepthOperand
using SeawaterPolynomials: TEOS10EquationOfState

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

for arch in test_architectures
    A = typeof(arch)
    @info "Testing interface fluxes diagnostics on $A"

    @testset "Interface fluxes diagnostics on $A" begin
        grid = RectilinearGrid(arch;
                               size = (4, 4, 2),
                               extent = (1, 1, 1),
                               topology = (Periodic, Periodic, Bounded))

        ocean = ocean_simulation(grid;
                                 momentum_advection = nothing,
                                 tracer_advection = nothing,
                                 closure = nothing,
                                 coriolis = nothing)

        sea_ice = sea_ice_simulation(grid, ocean)
        atmosphere = PrescribedAtmosphere(grid, [0.0])
        esm = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation = Radiation())

        T_flux = ocean.model.tracers.T.boundary_conditions.top.condition
        S_flux = ocean.model.tracers.S.boundary_conditions.top.condition
        sea_ice_ocean_fluxes = esm.interfaces.sea_ice_ocean_interface.fluxes
        atmosphere_ocean_fluxes = esm.interfaces.atmosphere_ocean_interface.fluxes

        T_flux_value = 2.0
        S_flux_value = 5.0
        frazil_heat_flux_value = 0.2
        interface_heat_flux_value = 0.3
        sea_ice_ocean_salt_flux_value = 0.9
        sea_ice_ocean_freshwater_flux_value = 0.8
        atmosphere_ocean_freshwater_flux_value = 1.7

        fill!(T_flux, T_flux_value)
        fill!(S_flux, S_flux_value)
        fill!(sea_ice_ocean_fluxes.frazil_heat, frazil_heat_flux_value)
        fill!(sea_ice_ocean_fluxes.interface_heat, interface_heat_flux_value)
        fill!(sea_ice_ocean_fluxes.salt, sea_ice_ocean_salt_flux_value)
        fill!(sea_ice_ocean_fluxes.freshwater_flux, sea_ice_ocean_freshwater_flux_value)
        fill!(atmosphere_ocean_fluxes.freshwater_flux, atmosphere_ocean_freshwater_flux_value)

        ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
        cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
        S₀ = 34.0 # different from the default value

        frazil_heat = frazil_heat_flux(esm)
        net_ocean_heat = net_ocean_heat_flux(esm)
        sea_ice_ocean_heat = sea_ice_ocean_heat_flux(esm)
        atmosphere_ocean_heat = atmosphere_ocean_heat_flux(esm)
        net_ocean_freshwater = net_ocean_freshwater_flux(esm)
        sea_ice_ocean_freshwater = sea_ice_ocean_freshwater_flux(esm)
        atmosphere_ocean_freshwater = atmosphere_ocean_freshwater_flux(esm)

        diags = (frazil_heat, net_ocean_heat, sea_ice_ocean_heat, atmosphere_ocean_heat,
                 net_ocean_freshwater, sea_ice_ocean_freshwater, atmosphere_ocean_freshwater)

        for d in diags
            @test d isa Oceananigans.Fields.AbstractField
            @test location(d) == (Center, Center, Nothing)
            d |> Field
            compute!(d)
        end

        @allowscalar begin
            @test net_ocean_heat[1, 1, 1] ≈ ρᵒᶜ * cᵒᶜ * T_flux_value + frazil_heat_flux_value
            @test atmosphere_ocean_heat[1, 1, 1] ≈ ρᵒᶜ * cᵒᶜ * T_flux_value - interface_heat_flux_value
            @test sea_ice_ocean_heat[1, 1, 1] ≈ frazil_heat_flux_value + interface_heat_flux_value
            @test net_ocean_heat[1, 1, 1] ≈ atmosphere_ocean_heat[1, 1, 1] + sea_ice_ocean_heat[1, 1, 1]

            @test net_ocean_freshwater[1, 1, 1] ≈ sea_ice_ocean_freshwater_flux_value + ρᵒᶜ * atmosphere_ocean_freshwater_flux_value
            @test sea_ice_ocean_freshwater[1, 1, 1] ≈ sea_ice_ocean_freshwater_flux_value
            @test atmosphere_ocean_freshwater[1, 1, 1] ≈ ρᵒᶜ * atmosphere_ocean_freshwater_flux_value
        end
    end
end
