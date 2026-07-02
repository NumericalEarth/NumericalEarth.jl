include("runtests_setup.jl")

using CUDA
using Oceananigans
using Oceananigans: set!
using Oceananigans.TimeSteppers: update_state!
using Thermodynamics
using NumericalEarth.Lands: SlabLand, SlabEnergy, SaturatedSurface
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    FarquharPhotosynthesis, MedlynConductance, CanopyConductanceHumidity, BulkHumidity,
    net_assimilation, medlyn_conductance, stomatal_conductance
using NumericalEarth.Atmospheres: PrescribedAtmosphere

#####
##### Photosynthesis + conductance physics (pure functions — no grid needed).
#####

@testset "Farquhar–Medlyn canopy physics" begin
    for FT in (Float32, Float64)
        photo = FarquharPhotosynthesis(FT)
        cond  = MedlynConductance(FT)

        Tₗ   = FT(298)      # leaf temperature (K)
        P    = FT(101325)   # air pressure (Pa)
        ca   = FT(40)       # CO₂ partial pressure (Pa)
        ci   = FT(28)       # intercellular CO₂ (Pa)
        β    = FT(1)

        # Net assimilation rises with absorbed PAR and saturates (light limitation).
        A_dim   = net_assimilation(photo, ci, FT(1e-4), Tₗ, P, β)
        A_light = net_assimilation(photo, ci, FT(1e-3), Tₗ, P, β)
        A_sat   = net_assimilation(photo, ci, FT(1e-2), Tₗ, P, β)
        @test A_light > A_dim
        @test A_sat ≥ A_light
        @test A_sat - A_light < A_light - A_dim   # saturating, not linear

        # ... and rises with intercellular CO₂.
        @test net_assimilation(photo, FT(35), FT(1e-3), Tₗ, P, β) >
              net_assimilation(photo, FT(15), FT(1e-3), Tₗ, P, β)

        # Medlyn conductance decreases with VPD and never falls below g₀.
        An = FT(1e-5)
        @test medlyn_conductance(cond, An, FT(500),  ca / P) >
              medlyn_conductance(cond, An, FT(2000), ca / P)
        @test medlyn_conductance(cond, FT(-1e-6), FT(1000), ca / P) ≈ cond.g0

        # Coupled solve: light drives conductance up, VPD and moisture stress down.
        gs_ref, An_ref, ci_ref = stomatal_conductance(photo, cond, FT(1e-3), FT(1000), Tₗ, ca, P, FT(1))
        gs_dry, _, _           = stomatal_conductance(photo, cond, FT(1e-3), FT(3000), Tₗ, ca, P, FT(1))
        gs_str, _, _           = stomatal_conductance(photo, cond, FT(1e-3), FT(1000), Tₗ, ca, P, FT(0.3))
        gs_drk, _, _           = stomatal_conductance(photo, cond, FT(0),    FT(1000), Tₗ, ca, P, FT(1))

        @test gs_ref > cond.g0
        @test An_ref > 0
        @test photo.Γstar25 < ci_ref < ca          # intercellular CO₂ in physical band
        @test gs_dry < gs_ref                    # higher VPD closes stomata
        @test gs_str < gs_ref                    # moisture stress closes stomata
        @test gs_drk ≈ cond.g0 atol=1e-3         # no light → minimum conductance
        @test eltype(gs_ref) == FT
    end
end

#####
##### Coupled single-column: the canopy resistance raises the Bowen ratio.
#####

@testset "CanopyConductanceHumidity coupled fluxes" begin
    for arch in test_architectures
        FT = Float64
        grid = LatitudeLongitudeGrid(arch, FT;
                                     size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))

        function latent_heat(q_formulation; leaf_area_index = 2.0)
            atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10,
                                                    boundary_layer_height = 512)
            @allowscalar begin
                fill!(parent(atmosphere.temperature),       290.0)
                fill!(parent(atmosphere.specific_humidity), 0.006)
                fill!(parent(atmosphere.velocities.u),      5.0)
                fill!(parent(atmosphere.pressure),          101325.0)
            end
            land = SlabLand(grid; hydrology = SaturatedSurface(), energy = SlabEnergy(FT))
            set!(land; T = 300.0)   # warm, wet surface → upward evaporation
            model = AtmosphereLandModel(atmosphere, land; radiation = nothing,
                                        atmosphere_land_interface_specific_humidity = q_formulation)
            update_state!(model)
            f = model.interfaces.atmosphere_land_interface.fluxes
            return @allowscalar f.latent_heat[1, 1, 1]
        end

        # Saturated bare surface (no resistance) evaporates most; the canopy
        # stomatal resistance in series reduces the latent-heat flux magnitude.
        LE_bulk   = latent_heat(BulkHumidity(Thermodynamics.Liquid()))
        LE_canopy = latent_heat(CanopyConductanceHumidity(FT))
        @test abs(LE_canopy) < abs(LE_bulk)
        @test LE_canopy > 0   # upward evaporation → positive (evaporative cooling) latent flux

        # More leaf area → larger canopy conductance → stronger latent flux.
        LE_lo = latent_heat(CanopyConductanceHumidity(FT; leaf_area_index = 0.5), leaf_area_index = 0.5)
        LE_hi = latent_heat(CanopyConductanceHumidity(FT; leaf_area_index = 4.0), leaf_area_index = 4.0)
        @test abs(LE_hi) > abs(LE_lo)

        # Moisture stress throttles transpiration (constant β here; the
        # saturation-dependent CriticalSaturation path is covered above).
        LE_stressed = latent_heat(CanopyConductanceHumidity(FT; moisture_stress = 0.3))
        @test abs(LE_stressed) < abs(LE_canopy)
    end
end
