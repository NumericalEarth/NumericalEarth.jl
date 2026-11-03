include("runtests_setup.jl")

using Test
using NumericalEarth
using Oceananigans
using Oceananigans.Fields: interior
using PythonCall, CondaPkg

@testset "Veros ocean model interface" begin
    VerosModule = Base.get_extension(NumericalEarth, :NumericalEarthVerosExt)
    @test !isnothing(VerosModule)

    ocean = VerosModule.VerosOceanSimulation("global_4deg", :GlobalFourDegreeSetup)
    variables = ocean.setup.state.variables
    grid = Oceananigans.Grids.grid(ocean)
    ρᵒᶜ = NumericalEarth.EarthSystemModels.reference_density(ocean)
    Δt = 1800.0

    veros_array(name) = pyconvert(Array, pyimport("numpy").asarray(getproperty(variables, name)))
    interior_cells(array) = view(array, 3:size(array, 1)-2, 3:size(array, 2)-2)
    wet = interior_cells(veros_array(:maskT)[:, :, end]) .> 0
    wet_mean(array) = sum(interior_cells(array)[wet]) / count(wet)

    atmosphere_grid = LatitudeLongitudeGrid(size=(36, 18), longitude=(0, 360), latitude=(-90, 90),
                                            topology=(Periodic, Bounded, Flat), halo=(3, 3))

    # An eastward wind, and raining air warmer and moister than the ocean surface everywhere
    atmosphere = PrescribedAtmosphere(atmosphere_grid, [0.0])
    set!(atmosphere.velocities.u, 10)
    set!(atmosphere.temperature, 318.15)
    set!(atmosphere.specific_humidity, 0.06)
    set!(atmosphere.pressure, 101325)
    set!(atmosphere.precipitation_flux.rain, 1e-4)

    coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere)
    state = coupled_model.interfaces.exchanger.ocean.state

    @testset "Surface forcing pushes the ocean the way the atmosphere does" begin
        time_step!(coupled_model, Δt)

        @test wet_mean(veros_array(:surface_taux)) > 0
        @test wet_mean(veros_array(:forc_temp_surface)) > 0
        @test wet_mean(veros_array(:forc_salt_surface)) < 0
        @test all(iszero, interior_cells(veros_array(:forc_temp_surface))[.!wet])
    end

    @testset "The exchanger reads the current time level and fills its halos" begin
        for _ in 1:3
            time_step!(coupled_model, Δt)
            τ = pyconvert(Int, variables.tau.item()) + 1
            current_temperature = interior_cells(veros_array(:temp)[:, :, end, τ])

            @test interior(state.T)[:, :, 1] == current_temperature
            @test view(state.T, 0, 1:size(grid, 2), 1) == view(state.T, size(grid, 1), 1:size(grid, 2), 1)
        end
    end

    @testset "Staggered fluxes land on the faces they were computed on" begin
        net_fluxes = coupled_model.interfaces.net_fluxes.ocean
        set!(net_fluxes.u, (λ, φ) -> λ)
        set!(net_fluxes.v, (λ, φ) -> φ)

        NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, ocean)

        xu = veros_array(:xu)[3:end-2]
        yu = veros_array(:yu)[3:end-2]
        @test mod.(.- interior_cells(veros_array(:surface_taux))[:, 1] ./ ρᵒᶜ, 360) ≈ mod.(xu, 360)
        @test .- interior_cells(veros_array(:surface_tauy))[1, :] ./ ρᵒᶜ ≈ yu
    end

    @testset "Staggered velocities are read from the faces they live on" begin
        τ = pyconvert(Int, variables.tau.item()) + 1
        u = Array(veros_array(:u))
        v = Array(veros_array(:v))
        u[:, :, end, τ] .= veros_array(:xu)
        v[:, :, end, τ] .= reshape(veros_array(:yu), 1, :)
        set!(ocean, "u", u)
        set!(ocean, "v", v)

        NumericalEarth.EarthSystemModels.interpolate_state!(coupled_model.interfaces.exchanger.ocean, grid, ocean, coupled_model)

        west_faces = Oceananigans.Grids.λnodes(grid, Face())[1:size(grid, 1)]
        south_faces = Oceananigans.Grids.φnodes(grid, Face())[1:size(grid, 2)+1]
        @test mod.(interior(state.u)[:, 1, 1], 360) ≈ mod.(west_faces, 360)
        @test interior(state.v)[1, :, 1] ≈ south_faces
    end

    @testset "Sea ice" begin
        VerosModule.remove_outputs(:global_4deg)
        ocean = VerosModule.VerosOceanSimulation("global_4deg", :GlobalFourDegreeSetup)
        variables = ocean.setup.state.variables
        set!(ocean, "temp", fill(-1.8, size(veros_array(:temp))))
        sea_ice = sea_ice_simulation(Oceananigans.Grids.grid(ocean), ocean)

        polar(λ, φ) = abs(φ) > 60
        set!(sea_ice.model, h=polar, ℵ=polar)

        # A cold eastward wind over water at the freezing point grows the ice and pushes it along
        atmosphere = PrescribedAtmosphere(atmosphere_grid, [0.0])
        set!(atmosphere.velocities.u, 10)
        set!(atmosphere.temperature, 250)
        set!(atmosphere.specific_humidity, 5e-4)
        set!(atmosphere.pressure, 101325)

        coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere)
        ice_covered = Array(interior(sea_ice.model.ice_concentration))[:, :, 1] .> 0 .&& wet
        initial_ice_thickness = Array(interior(sea_ice.model.ice_thickness))[:, :, 1]

        for _ in 1:4
            time_step!(coupled_model, Δt)
        end

        ice_thickness = Array(interior(sea_ice.model.ice_thickness))[:, :, 1]
        @test sum(ice_thickness[ice_covered]) > sum(initial_ice_thickness[ice_covered])

        # Brine rejected by the growing ice, and the eastward ice drag on the ocean below
        @test sum(interior_cells(veros_array(:forc_salt_surface))[ice_covered]) > 0
        @test sum(interior_cells(veros_array(:surface_taux))[ice_covered]) > 0

        # Water below freezing warms back to the freezing point and sends its heat to the ice
        temperature = Array(veros_array(:temp))
        τ = pyconvert(Int, variables.tau.item()) + 1
        temperature[:, :, end, τ] .= -5
        set!(ocean, "temp", temperature)
        Oceananigans.TimeSteppers.update_state!(coupled_model)

        frazil_heat = coupled_model.interfaces.sea_ice_ocean_interface.fluxes.frazil_heat
        @test all(<(0), interior(frazil_heat)[:, :, 1][wet])

        time_step!(ocean, Δt)
        adjusted_temperature = interior_cells(veros_array(:temp)[:, :, end, τ])
        @test minimum(adjusted_temperature[wet]) > -3
    end

    VerosModule.remove_outputs(:global_4deg)
end
