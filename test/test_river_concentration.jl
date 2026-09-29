include("runtests_setup.jl")

using NumericalEarth.Oceans: RiverTracerContent, tracer_freshwater_content, freshwater_content
using Oceananigans.Grids: inactive_node
using Oceananigans.OutputReaders: Cyclical

tracer_content(ocean, name) = tracer_freshwater_content(ocean.model.tracers[name].boundary_conditions.top.condition)

# the content flux as the top boundary condition evaluates it
content_at(content, i, j, grid, clock) = freshwater_content(content, i, j, grid, clock, nothing)

# a discrete-form concentration, as for a boundary condition
@inline latitude_dependent_concentration(i, j, grid, clock, fields) = 1 + j / grid.Ny

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
        @test !(tracer_content(ocean, :X) isa RiverTracerContent)

        # A tracer carried only by rivers at a prescribed concentration (tracer units, e.g. mmol m⁻³)
        c = 2.5
        ocean = ocean_simulation(grid; free_surface, tracers = (:T, :S, :X, :Y),
                                 freshwater_tracer_content = (; X = RiverConcentration(c)))

        X_content = tracer_content(ocean, :X)
        @test X_content isa RiverTracerContent
        @test !(tracer_content(ocean, :Y) isa RiverTracerContent)
        @test !(tracer_content(ocean, :T) isa RiverTracerContent)
        @test !(tracer_content(ocean, :S) isa RiverTracerContent)

        clock = ocean.model.clock
        Jʳ = X_content.river_volume_flux
        @test Jʳ isa Field
        set!(Jʳ, (λ, φ) -> 1e-6 * (1 + cosd(φ))) # m s⁻¹
        @allowscalar for (i, j) in ((1, 1), (Nx ÷ 2, Ny ÷ 3), (Nx, Ny))
            @test content_at(X_content, i, j, grid, clock) == c * Jʳ[i, j, 1]
        end

        # The concentration can be anything a boundary condition can: a Field, a FieldTimeSeries, or a
        # discrete function, all sharing the one runoff volume flux field
        c_field = Field{Center, Center, Nothing}(grid)
        set!(c_field, (λ, φ) -> 2 + sind(φ))

        c_series = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 10.0]; time_indexing = Cyclical(20.0))
        set!(c_series[1], 1)
        set!(c_series[2], 3)

        ocean₂ = ocean_simulation(grid; free_surface, tracers = (:T, :S, :F, :G, :H),
                                  freshwater_tracer_content = (; F = RiverConcentration(c_field),
                                                                 G = RiverConcentration(c_series),
                                                                 H = RiverConcentration(latitude_dependent_concentration)))

        F_content, G_content, H_content = tracer_content(ocean₂, :F), tracer_content(ocean₂, :G), tracer_content(ocean₂, :H)
        @test F_content.river_volume_flux === G_content.river_volume_flux === H_content.river_volume_flux

        Jʳ₂ = F_content.river_volume_flux
        set!(Jʳ₂, (λ, φ) -> 1e-6 * (1 + cosd(φ)))
        clock = ocean₂.model.clock
        clock.time = 5 # halfway between the two snapshots

        @allowscalar for (i, j) in ((1, 1), (Nx ÷ 2, Ny ÷ 3), (Nx, Ny))
            @test content_at(F_content, i, j, grid, clock) ≈ c_field[i, j, 1] * Jʳ₂[i, j, 1]
            @test content_at(G_content, i, j, grid, clock) ≈ 2 * Jʳ₂[i, j, 1]
            @test content_at(H_content, i, j, grid, clock) ≈ (1 + j / Ny) * Jʳ₂[i, j, 1]
        end

        # a plain content is evaluated like a boundary condition too, so numbers work as fluxes
        ocean_number = ocean_simulation(grid; free_surface, tracers = (:T, :S, :Z), freshwater_tracer_content = (; Z = 1e-9))
        @test content_at(tracer_content(ocean_number, :Z), 1, 1, grid, clock) == 1e-9

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
        land_runoff = Array(interior(coupled_model.interfaces.exchanger.land.state.runoff_freshwater_flux, :, :, 1)) # kg m⁻² s⁻¹
        Jʳ_cpu = Array(interior(river_flux, :, :, 1)) # m s⁻¹, positive into the ocean

        inactive = [inactive_node(i, j, Nz, on_architecture(CPU(), grid), Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny]

        @test all(Jʳ_cpu .>= 0)
        @test maximum(Jʳ_cpu) > 0
        @test all(Jʳ_cpu[inactive] .== 0)
        @test Jʳ_cpu[.!inactive] ≈ land_runoff[.!inactive] ./ ρᵒᶜ

        @allowscalar @test content_at(X_content, Nx ÷ 2, Ny ÷ 3, grid, coupled_model.clock) == c * river_flux[Nx ÷ 2, Ny ÷ 3, 1]
    end
end
