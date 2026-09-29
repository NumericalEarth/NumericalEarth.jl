include("runtests_setup.jl")

using NumericalEarth.Oceans: RiverTracerContent, tracer_freshwater_content
using Oceananigans.Grids: inactive_node

freshwater_content(ocean, name) = tracer_freshwater_content(ocean.model.tracers[name].boundary_conditions.top.condition)

@testset "RiverConcentration freshwater tracer content" begin
    for arch in test_architectures
        grid = TripolarGrid(arch; size = (40, 40, 5), halo = (7, 7, 7), z = (-5000, 0))
        bottom_height = synthetic_bottom_height(grid; minimum_depth = 10, interpolation_passes = 5, major_basins = 1)
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map = true)
        free_surface = SplitExplicitFreeSurface(grid; substeps = 20)
        Nx, Ny, Nz = size(grid)

        @info "Testing RiverConcentration on $arch"

        # No RiverConcentration anywhere: no runoff volume flux field is allocated
        ocean = ocean_simulation(grid; free_surface, tracers = (:T, :S, :X))
        @test isnothing(NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(ocean).river_freshwater_volume_flux)
        @test !(freshwater_content(ocean, :X) isa RiverTracerContent)

        # A tracer carried only by rivers at a prescribed concentration (tracer units, e.g. mmol m⁻³)
        c = 2.5
        ocean = ocean_simulation(grid; free_surface, tracers = (:T, :S, :X, :Y),
                                 freshwater_tracer_content = (; X = RiverConcentration(c)))

        X_content = freshwater_content(ocean, :X)
        @test X_content isa RiverTracerContent
        @test !(freshwater_content(ocean, :Y) isa RiverTracerContent)
        @test !(freshwater_content(ocean, :T) isa RiverTracerContent)
        @test !(freshwater_content(ocean, :S) isa RiverTracerContent)

        Jʳ = X_content.river_volume_flux
        @test Jʳ isa Field
        set!(Jʳ, (λ, φ) -> 1e-6 * (1 + cosd(φ))) # m s⁻¹
        @allowscalar for (i, j) in ((1, 1), (Nx ÷ 2, Ny ÷ 3), (Nx, Ny))
            @test X_content[i, j, 1] == c * Jʳ[i, j, 1]
        end

        # Coupled model with a prescribed land: the assembler writes the land runoff volume flux.
        # Start from zero, as a fresh simulation does: the assembler only visits active columns.
        set!(Jʳ, 0)

        sea_ice    = sea_ice_simulation(grid, ocean; advection = nothing)
        atmosphere = synthetic_prescribed_atmosphere(arch)
        radiation  = synthetic_prescribed_radiation(arch)
        land       = synthetic_prescribed_land(arch)

        coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation, land)
        Oceananigans.TimeSteppers.update_state!(coupled_model)

        river_flux = coupled_model.interfaces.net_fluxes.ocean.river_freshwater_volume_flux
        @test river_flux isa Field
        @test river_flux === Jʳ # the same field the tracer content reads

        ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density
        land_runoff = Array(interior(coupled_model.interfaces.exchanger.land.state.freshwater_flux, :, :, 1)) # kg m⁻² s⁻¹
        Jʳ_cpu = Array(interior(river_flux, :, :, 1)) # m s⁻¹, positive into the ocean

        inactive = [inactive_node(i, j, Nz, on_architecture(CPU(), grid), Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny]

        @test all(Jʳ_cpu .>= 0)
        @test maximum(Jʳ_cpu) > 0
        @test all(Jʳ_cpu[inactive] .== 0)
        @test Jʳ_cpu[.!inactive] ≈ land_runoff[.!inactive] ./ ρᵒᶜ

        @allowscalar @test X_content[Nx ÷ 2, Ny ÷ 3, 1] == c * river_flux[Nx ÷ 2, Ny ÷ 3, 1]
    end
end
