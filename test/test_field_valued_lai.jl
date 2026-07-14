include("runtests_setup.jl")

using CUDA
using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyConductanceHumidity, JarvisConductance,
    surface_field_value, kernel_leaf_area_index, PrescribedLAIData,
    interface_vegetation_state
using NumericalEarth.Lands: SlabLand, SlabEnergy, SaturatedSurface
using NumericalEarth.Atmospheres: PrescribedAtmosphere

#####
##### Unit — the per-cell accessor for a constant, a static `Field`, and a
##### time-interpolated `FieldTimeSeries` LAI. Runs on the CPU: the accessor is
##### built for the flux kernel, so calling it on a GPU field outside a kernel
##### would scalar-index. Its logic is architecture-independent.
#####

@testset "surface_field_value: constant, Field, FieldTimeSeries" begin
    arch = CPU()
    for FT in (Float32, Float64)
        # Constant: returns the scalar, ignoring index and time.
        @test surface_field_value(FT(2.5), 1, 1, nothing) === FT(2.5)
        @inferred surface_field_value(FT(2.5), 1, 1, nothing)

        grid = LatitudeLongitudeGrid(arch, FT;
                                     size = (2, 1, 1),
                                     longitude = (0, 20), latitude = (0, 10),
                                     z = (-1, 0),
                                     topology = (Bounded, Bounded, Bounded))

        # Static Field: reads the per-cell value.
        lai_field = Field{Center, Center, Nothing}(grid)
        set!(lai_field, (λ, φ) -> ifelse(λ < 10, FT(4), FT(1)))
        @test surface_field_value(lai_field, 1, 1, nothing) == lai_field[1, 1, 1]
        @test surface_field_value(lai_field, 2, 1, nothing) == lai_field[2, 1, 1]
        @test surface_field_value(lai_field, 1, 1, nothing) != surface_field_value(lai_field, 2, 1, nothing)

        # FieldTimeSeries: linear in time between snapshots, exact at endpoints.
        times   = [0.0, 100.0]
        lai_fts = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        set!(lai_fts[1], 1)   # LAI = 1 at t = 0
        set!(lai_fts[2], 5)   # LAI = 5 at t = 100

        bundle, titp0   = kernel_leaf_area_index(lai_fts, arch, 0.0)
        _,      titpmid = kernel_leaf_area_index(lai_fts, arch, 50.0)
        _,      titp1   = kernel_leaf_area_index(lai_fts, arch, 100.0)

        @test bundle isa PrescribedLAIData
        @test surface_field_value(bundle, 1, 1, titp0)   ≈ FT(1)
        @test surface_field_value(bundle, 1, 1, titpmid) ≈ FT(3)   # midpoint
        @test surface_field_value(bundle, 1, 1, titp1)   ≈ FT(5)

        # Materialization converts to the grid float type (times may be Float64).
        canopy = CanopyConductanceHumidity(FT; leaf_area_index = lai_fts)
        veg    = interface_vegetation_state(1, 1, grid, canopy, bundle, titpmid)
        @test veg.leaf_area_index isa FT
        @test veg.leaf_area_index ≈ FT(3)
    end
end

#####
##### Integration — a static spatial LAI map drives spatially varying transpiration.
#####

@testset "Field-valued LAI: spatial coupled fluxes" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT;
                                     size = (2, 1, 1),
                                     longitude = (0, 20), latitude = (0, 10),
                                     z = (-1, 0),
                                     topology = (Bounded, Bounded, Bounded))

        # Cell 1 (λ = 5): dense canopy; cell 2 (λ = 15): bare (LAI = 0).
        lai_field = Field{Center, Center, Nothing}(grid)
        set!(lai_field, (λ, φ) -> ifelse(λ < 10, 4.0, 0.0))

        atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
        fill!(parent(atmosphere.temperature),       290.0)
        fill!(parent(atmosphere.specific_humidity), 0.006)
        fill!(parent(atmosphere.velocities.u),      5.0)
        fill!(parent(atmosphere.pressure),          101325.0)

        land = SlabLand(grid; hydrology = SaturatedSurface(), energy = SlabEnergy(FT))
        set!(land; T = 300.0)

        canopy = CanopyConductanceHumidity(FT; leaf_area_index = lai_field, conductance = JarvisConductance(FT))
        model = AtmosphereLandModel(atmosphere, land; radiation = nothing,
                                    atmosphere_land_interface_specific_humidity = canopy)
        update_state!(model)
        LE = Array(interior(model.interfaces.atmosphere_land_interface.fluxes.latent_heat))[:, 1, 1]

        # Dense-canopy cell transpires; the bare cell (g_c = LAI · gₛ = 0) has ~no latent flux.
        @test LE[1] > LE[2]
        @test isapprox(LE[2], 0; atol = 1e-6)
        @test all(isfinite, LE)
    end
end

#####
##### Integration — a monthly-style LAI time series drives time-varying transpiration.
#####

@testset "Field-valued LAI: temporal coupled fluxes" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT;
                                     size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))

        times   = [0.0, 100.0]
        lai_fts = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        set!(lai_fts[1], 0.5)   # sparse at t = 0
        set!(lai_fts[2], 4.0)   # dense at t = 100

        function latent_heat_at(t)
            atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
            fill!(parent(atmosphere.temperature),       290.0)
            fill!(parent(atmosphere.specific_humidity), 0.006)
            fill!(parent(atmosphere.velocities.u),      5.0)
            fill!(parent(atmosphere.pressure),          101325.0)
            land = SlabLand(grid; hydrology = SaturatedSurface(), energy = SlabEnergy(FT))
            set!(land; T = 300.0)
            canopy = CanopyConductanceHumidity(FT; leaf_area_index = lai_fts, conductance = JarvisConductance(FT))
            model = AtmosphereLandModel(atmosphere, land; radiation = nothing,
                                        atmosphere_land_interface_specific_humidity = canopy)
            model.clock.time = t
            update_state!(model)
            return Array(interior(model.interfaces.atmosphere_land_interface.fluxes.latent_heat))[1, 1, 1]
        end

        LE_t0  = latent_heat_at(0.0)     # LAI = 0.5
        LE_mid = latent_heat_at(50.0)    # LAI = 2.25 (interpolated)
        LE_t1  = latent_heat_at(100.0)   # LAI = 4.0

        # Rising LAI ⇒ rising canopy conductance ⇒ rising latent flux.
        @test LE_t0 < LE_mid < LE_t1
        @test all(isfinite, (LE_t0, LE_mid, LE_t1))
    end
end
