include("runtests_setup.jl")

using CUDA
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction
using NumericalEarth.Oceans: MultipleFluxes, FreshwaterExchange, net_flux
using NumericalEarth.EarthSystemModels.InterfaceComputations: net_fluxes

# A constant top-cell tendency `G`, mimicking the part of a `DatasetRestoring`
# that the `MultipleFluxes` BC actually evaluates: `r * μ * (ψ_dataset - ψ)`.
struct ConstantTendency{T}
    G :: T
end

@inline (c::ConstantTendency)(i, j, grid, clock, fields) = c.G

@testset "MultipleFluxes boundary condition" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch;
                                     size = (8, 8, 4),
                                     halo = (7, 7, 7),
                                     latitude = (-10, 10),
                                     longitude = (0, 360),
                                     z = (-100, 0))

        @testset "default path: no additional_fluxes" begin
            ocean = ocean_simulation(grid; warn=false)
            S_top = ocean.model.tracers.S.boundary_conditions.top
            T_top = ocean.model.tracers.T.boundary_conditions.top
            u_top = ocean.model.velocities.u.boundary_conditions.top

            # Tracer top boundary conditions always pair the solver's flux field with the
            # freshwater exchange, so they are wrapped even with no user-supplied flux.
            @test S_top.condition isa DiscreteBoundaryFunction
            @test T_top.condition isa DiscreteBoundaryFunction
            @test S_top.condition.func isa MultipleFluxes
            @test T_top.condition.func isa MultipleFluxes
            @test S_top.condition.func.additional_fluxes isa FreshwaterExchange
            @test T_top.condition.func.additional_fluxes isa FreshwaterExchange
            @test S_top.condition.func.additional_fluxes.additional === nothing
            @test T_top.condition.func.additional_fluxes.additional === nothing

            # Momentum carries no freshwater exchange: a plain FluxBoundaryCondition over a writable Field
            @test u_top.condition isa Field

            # net_fluxes unwraps to the writable flux fields the coupled solver writes into
            nf = net_fluxes(ocean)
            @test nf.S isa Field
            @test nf.T isa Field
            @test nf.S === S_top.condition.func.flux_field
            @test nf.T === T_top.condition.func.flux_field
            @test nf.u === u_top.condition
        end

        @testset "wrapped path: salinity restoring" begin
            G = 1.0e-6  # constant top-cell tendency
            ocean = ocean_simulation(grid;
                                     warn=false,
                                     additional_surface_fluxes=(; S=ConstantTendency(G)))

            S_top = ocean.model.tracers.S.boundary_conditions.top
            T_top = ocean.model.tracers.T.boundary_conditions.top

            # The user flux rides along inside S's freshwater exchange; T keeps an empty slot
            @test S_top.condition.func.additional_fluxes.additional isa ConstantTendency
            @test T_top.condition.func.additional_fluxes.additional === nothing

            fr = S_top.condition.func
            @test fr.flux_field isa Field
            @test size(fr.flux_field) == (8, 8, 1)

            # net_fluxes peeks through DiscreteBoundaryFunction → MultipleFluxes
            nf = net_fluxes(ocean)
            @test nf.S isa Field
            @test nf.S === fr.flux_field
            @test nf.T === T_top.condition.func.flux_field

            # J_solver = 0 by default and the freshwater volume flux is still zero, so only G survives
            fields = ocean.model.tracers
            CUDA.@allowscalar val = S_top.condition.func(2, 2, grid, ocean.model.clock, fields)
            @test val ≈ G

            # Now pretend the OMIP coupled flux solver wrote into the underlying field
            J = 7.0e-5
            fill!(net_flux(S_top.condition), J)
            CUDA.@allowscalar val = S_top.condition.func(2, 2, grid, ocean.model.clock, fields)
            @test val ≈ J + G
        end
    end
end
