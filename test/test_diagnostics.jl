include("runtests_setup.jl")

using Oceananigans: location
using Oceananigans.Models: buoyancy_operation
using NumericalEarth.Diagnostics: MixedLayerDepthField, MixedLayerDepthOperand, Streamfunction
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
    @info "Testing Streamfunction diagnostic on $A"

    @testset "Streamfunction on $A" begin
        grid = RectilinearGrid(arch;
                               size = (8, 6, 4),
                               extent = (1, 1, 1),
                               topology = (Periodic, Bounded, Bounded))

        ρ = CenterField(grid)
        v = CenterField(grid)

        set!(ρ, (x, y, z) -> 1018 + 4y + 0.5z)
        set!(v, 1.0)

        bins = 1018:0.25:1023

        ψ_hist = Streamfunction(grid;
                                x_field = ρ,
                                vel_field = v,
                                x_bins = bins,
                                dims = 2,
                                cumulative = false,
                                in_sverdrups = false)

        ψ_cumulative = Streamfunction(grid;
                                      x_field = ρ,
                                      vel_field = v,
                                      x_bins = bins,
                                      dims = 2,
                                      cumulative = true,
                                      in_sverdrups = false)

        ψ_tuple_dims = Streamfunction(grid;
                                      x_field = ρ,
                                      vel_field = v,
                                      x_bins = bins,
                                      dims = (1, 3),
                                      cumulative = false,
                                      in_sverdrups = false)

        ψ_inferred_grid = Streamfunction(; x_field = ρ,
                                          vel_field = v,
                                          x_bins = bins,
                                          dims = 2,
                                          cumulative = false,
                                          in_sverdrups = false)

        regridder_calls = Ref(0)
        identity_regridder = (x_field, vel_field) -> begin
            regridder_calls[] += 1
            return (x_field, vel_field)
        end

        ψ_regridded = Streamfunction(grid;
                                     regridder = identity_regridder,
                                     x_field = ρ,
                                     vel_field = v,
                                     x_bins = bins,
                                     dims = 2,
                                     cumulative = false,
                                     in_sverdrups = false)

        compute!(ψ_hist)
        compute!(ψ_cumulative)
        compute!(ψ_tuple_dims)
        compute!(ψ_inferred_grid)
        compute!(ψ_regridded)

        Ψhist = Array(interior(on_architecture(CPU(), ψ_hist)))
        Ψcum = Array(interior(on_architecture(CPU(), ψ_cumulative)))
        Ψtuple = Array(interior(on_architecture(CPU(), ψ_tuple_dims)))
        Ψinferred = Array(interior(on_architecture(CPU(), ψ_inferred_grid)))

        @test size(Ψhist, 1) == length(bins) - 1
        @test size(Ψhist, 2) == size(grid, 2)
        @test size(Ψhist, 3) == 1
        @test Ψtuple ≈ Ψhist
        @test Ψinferred ≈ Ψhist

        expected_cumulative = reverse(cumsum(reverse(Ψhist; dims = 1); dims = 1); dims = 1)
        @test Ψcum ≈ expected_cumulative
        @test regridder_calls[] == 1
    end
end

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
                                 coriolis = nothing)

        sea_ice = sea_ice_simulation(grid, ocean)
        atmosphere = PrescribedAtmosphere(grid, [0.0])
        esm = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation = Radiation())

        T_flux = ocean.model.tracers.T.boundary_conditions.top.condition
        S_flux = ocean.model.tracers.S.boundary_conditions.top.condition
        sea_ice_ocean_fluxes = esm.interfaces.sea_ice_ocean_interface.fluxes

        T_flux_value = 2.0
        S_flux_value = 5.0
        frazil_heat_flux_value = 0.2
        interface_heat_flux_value = 0.3
        sea_ice_ocean_salt_flux_value = 0.9

        fill!(T_flux, T_flux_value)
        fill!(S_flux, S_flux_value)
        fill!(sea_ice_ocean_fluxes.frazil_heat, frazil_heat_flux_value)
        fill!(sea_ice_ocean_fluxes.interface_heat, interface_heat_flux_value)
        fill!(sea_ice_ocean_fluxes.salt, sea_ice_ocean_salt_flux_value)

        ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
        cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
        S₀ = 35.0

        frazil_temperature = frazil_temperature_flux(esm)
        net_ocean_temperature = net_ocean_temperature_flux(esm)
        sea_ice_ocean_temperature = sea_ice_ocean_temperature_flux(esm)
        atmosphere_ocean_temperature = atmosphere_ocean_temperature_flux(esm)
        frazil_heat = frazil_heat_flux(esm)
        net_ocean_heat = net_ocean_heat_flux(esm)
        sea_ice_ocean_heat = sea_ice_ocean_heat_flux(esm)
        atmosphere_ocean_heat = atmosphere_ocean_heat_flux(esm)
        net_ocean_salinity = net_ocean_salinity_flux(esm)
        sea_ice_ocean_salinity = sea_ice_ocean_salinity_flux(esm)
        atmosphere_ocean_salinity = atmosphere_ocean_salinity_flux(esm)
        net_ocean_freshwater = net_ocean_freshwater_flux(esm; reference_salinity = 35)
        sea_ice_ocean_freshwater = sea_ice_ocean_freshwater_flux(esm; reference_salinity = 35)
        atmosphere_ocean_freshwater = atmosphere_ocean_freshwater_flux(esm; reference_salinity = 35)

        for f in (frazil_temperature, net_ocean_temperature, sea_ice_ocean_temperature,
                  atmosphere_ocean_temperature, frazil_heat, net_ocean_heat, sea_ice_ocean_heat,
                  atmosphere_ocean_heat, net_ocean_salinity, sea_ice_ocean_salinity,
                  atmosphere_ocean_salinity, net_ocean_freshwater, sea_ice_ocean_freshwater,
                  atmosphere_ocean_freshwater)

            @test f isa Field
            @test location(f) == (Center, Center, Nothing)
            compute!(f)
        end

        @allowscalar begin
            @test net_ocean_heat[1, 1, 1] ≈ ρᵒᶜ * cᵒᶜ * T_flux_value + frazil_heat_flux_value
            @test atmosphere_ocean_heat[1, 1, 1] ≈ ρᵒᶜ * cᵒᶜ * T_flux_value - interface_heat_flux_value
            @test sea_ice_ocean_heat[1, 1, 1] ≈ frazil_heat_flux_value + interface_heat_flux_value
            @test net_ocean_heat[1, 1, 1] ≈ atmosphere_ocean_heat[1, 1, 1] + sea_ice_ocean_heat[1, 1, 1]

            @test net_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * S_flux_value
            @test sea_ice_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * sea_ice_ocean_salt_flux_value
            @test atmosphere_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * (S_flux_value - sea_ice_ocean_salt_flux_value)
            @test net_ocean_freshwater[1, 1, 1] ≈ atmosphere_ocean_freshwater[1, 1, 1] + sea_ice_ocean_freshwater[1, 1, 1]

            @test net_ocean_temperature[1, 1, 1] ≈ T_flux_value + 1 / (ρᵒᶜ * cᵒᶜ) * frazil_heat_flux_value
            @test atmosphere_ocean_temperature[1, 1, 1] ≈ T_flux_value - 1 / (ρᵒᶜ * cᵒᶜ) * interface_heat_flux_value
            @test sea_ice_ocean_temperature[1, 1, 1] ≈ 1 / (ρᵒᶜ * cᵒᶜ) * (frazil_heat_flux_value + interface_heat_flux_value)
            @test net_ocean_temperature[1, 1, 1] ≈ atmosphere_ocean_temperature[1, 1, 1] + sea_ice_ocean_temperature[1, 1, 1]

            @test net_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * S_flux_value
            @test sea_ice_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * sea_ice_ocean_salt_flux_value
            @test atmosphere_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * (S_flux_value - sea_ice_ocean_salt_flux_value)
            @test net_ocean_freshwater[1, 1, 1] ≈ atmosphere_ocean_freshwater[1, 1, 1] + sea_ice_ocean_freshwater[1, 1, 1]
        end
    end
end
