include("runtests_setup.jl")

using CUDA: @allowscalar
using Oceananigans: UpdateStateCallsite
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind
using Oceananigans.Simulations: Callback
using Oceananigans.Operators: volume
using Oceananigans.Units
using NumericalEarth.Diagnostics: net_ocean_heat_flux, ocean_top_advective_heat_flux
using NumericalEarth.Oceans: get_radiative_forcing

function cache_penultimate_stage_temperature!(model, temperature)
    if model.clock.stage == model.timestepper.Nstages - 1
        set!(temperature, model.tracers.T)
    end
    return nothing
end

# Heat and freshwater have different noise floors, so they get separate tolerances.
function test_tracer_budget(coupled_model, Sᵒᶜ, Δt, nsteps; heat_rtol, freshwater_rtol)
    ocean = coupled_model.ocean
    grid  = ocean.model.grid

    ocean_properties = coupled_model.interfaces.ocean_properties
    ρᵒᶜ = ocean_properties.reference_density
    cᵒᶜ = ocean_properties.heat_capacity

    T = ocean.model.tracers.T
    S = ocean.model.tracers.S

    # This diagnostic includes all heat crossing the ocean surface.
    Jᵀ  = net_ocean_heat_flux(coupled_model)
    Jʷ  = coupled_model.interfaces.net_fluxes.ocean.η   # freshwater volume flux
    heat_rate     = Integral(Jᵀ, dims=(1, 2))
    volume_rate   = Integral(Jʷ, dims=(1, 2))

    penetrating_radiation = get_radiative_forcing(ocean)
    radiative_rate = if isnothing(penetrating_radiation)
        nothing
    else
        radiative_forcing = KernelFunctionOperation{Center, Center, Center}(penetrating_radiation, grid,
                                                                            ocean.model.clock,
                                                                            Oceananigans.fields(ocean.model))
        Integral(ρᵒᶜ * cᵒᶜ * radiative_forcing, dims=(1, 2, 3))
    end

    cell_volume = KernelFunctionOperation{Center, Center, Center}(volume, grid, Center(), Center(), Center())

    VT⁻ = CenterField(grid); ΔVT = Field(T * volume  - VT⁻)
    VV⁻ = CenterField(grid); ΔVV = Field(cell_volume - VV⁻)
    VS⁻ = CenterField(grid); ΔVS = Field(S * volume  - VS⁻)

    set!(VS⁻, S * volume)
    ∫S⁻ = sum(VS⁻)

    stage_temperature = grid isa MutableGridOfSomeKind ? nothing : CenterField(grid)
    stage_heat_rate = if isnothing(stage_temperature)
        nothing
    else
        stage_flux = ocean_top_advective_heat_flux(coupled_model, stage_temperature)
        Integral(stage_flux, dims=(1, 2))
    end

    callback_name = gensym(:tracer_budget_stage_temperature)
    if !isnothing(stage_temperature)
        ocean.callbacks[callback_name] = Callback(cache_penultimate_stage_temperature!;
                                                   parameters=stage_temperature,
                                                   callsite=UpdateStateCallsite())
    end

    for _ = 1:nsteps
        set!(VT⁻, T * volume)
        set!(VV⁻, cell_volume)

        previous_heat_flux      = @allowscalar first(Field(heat_rate))
        previous_volume_flux    = @allowscalar first(Field(volume_rate))
        previous_radiative_rate = isnothing(radiative_rate) ? zero(previous_heat_flux) : @allowscalar first(Field(radiative_rate))

        time_step!(coupled_model, Δt)
        last_Δt = ocean.model.clock.last_Δt

        compute!(ΔVT)
        compute!(ΔVV)

        # Fixed grids exchange heat through their stationary upper boundary.
        # Use the tracer state and transport from the final split-RK stage, which
        # is the tendency that advances the tracer over the complete time step.
        stage_heat_flux = isnothing(stage_heat_rate) ? zero(previous_heat_flux) :
                          @allowscalar first(Field(stage_heat_rate))
        heat_content_tendency = sum(ρᵒᶜ * cᵒᶜ * ΔVT)
        expected_heat_tendency = (previous_radiative_rate - previous_heat_flux - stage_heat_flux) * last_Δt
        @test isapprox(heat_content_tendency, expected_heat_tendency; rtol=heat_rtol)

        # Volume grows by exactly the surface-integrated freshwater volume flux.
        volume_tendency = sum(ΔVV)
        expected_volume_tendency = previous_volume_flux * last_Δt
        @test isapprox(volume_tendency, expected_volume_tendency; rtol=freshwater_rtol)
    end

    !isnothing(stage_temperature) && pop!(ocean.callbacks, callback_name)

    # Freshwater carries no salt, so the total salt content is conserved over the run.
    compute!(ΔVS)
    @test abs(sum(ΔVS)) / ∫S⁻ < freshwater_rtol

    return nothing
end

@testset "Tracer budget closure under surface fluxes" begin
    for arch in test_architectures
        for z in (MutableVerticalDiscretization((-100, 0)), (-100,0))
            for fold_topology in (RightFaceFolded,
                                  RightCenterFolded)
                              
            @info ".. on $(typeof(arch)) with $(typeof(z)) and $fold_topology topology"
            underlying_grid = TripolarGrid(arch;
                                           size = (20, 20, 20),
                                           z,
                                           halo = (7, 7, 4),
                                           fold_topology)

            bottom_height = regrid_bathymetry(underlying_grid,
                                              Metadatum(:bottom_height, dataset=ETOPO2022());
                                              minimum_depth=15,
                                              interpolation_passes=1,
                                              major_basins=1)

            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map = true)

            time_indices_in_memory = 4
            radiation  = JRA55PrescribedRadiation(arch; time_indices_in_memory)
            atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory)

            # An idealized, stably stratified initial state
            Tᵢ(λ, φ, z) = 2 + 26 * cosd(φ)^2 * exp(z / 30)
            Sᵢ(λ, φ, z) = 35 - 1//2 * exp(z / 30)

            # Sea ice is largest at the poles and tapers to zero over 20° of latitude.
            polar_ice_fraction(φ) = clamp((abs(φ) - 70) / 20, 0, 1)
            hᵢ(λ, φ) = 2 * polar_ice_fraction(φ)
            ℵᵢ(λ, φ) = polar_ice_fraction(φ)

            Δt = 605seconds
            Sᵒᶜ = 35 # reference salinity [psu]
            free_surface = SplitExplicitFreeSurface(substeps=20)

            # Without shortwave penetration
            @testset "Surface-only fluxes" begin
                ocean = ocean_simulation(deepcopy(grid); free_surface, radiative_forcing=nothing)
                set!(ocean.model, T=Tᵢ, S=Sᵢ)
                coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation)
                test_tracer_budget(coupled_model, Sᵒᶜ, Δt, 4; heat_rtol=1e-11, freshwater_rtol=√eps(eltype(grid)))
            end

            # With penetrative shortwave radiation
            @testset "Surface fluxes + Penetrating shortwave radiation" begin
                ocean = ocean_simulation(deepcopy(grid); free_surface)
                set!(ocean.model, T=Tᵢ, S=Sᵢ)
                coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation)
                test_tracer_budget(coupled_model, Sᵒᶜ, Δt, 4; heat_rtol=1e-11, freshwater_rtol=√eps(eltype(grid)))
            end

            @testset "Surface fluxes + penetrating shortwave radiation + Sea ice" begin
                @info "    .. Surface fluxes + penetrating shortwave radiation + Sea ice"
                new_grid = deepcopy(grid) # because the grid is mutable
                ocean = ocean_simulation(new_grid; free_surface)
                sea_ice = sea_ice_simulation(new_grid, ocean) # test works with dynamics = nothing
                set!(ocean.model, T=Tᵢ, S=Sᵢ)
                set!(sea_ice.model, h=hᵢ, ℵ=ℵᵢ)
                coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
                tolerance = 2√eps(eltype(new_grid))
                test_tracer_budget(coupled_model, Sᵒᶜ, Δt, 4;
                                   heat_rtol=tolerance,
                                   freshwater_rtol=tolerance)
            end
        end
        end
    end
end
