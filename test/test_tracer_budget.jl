include("runtests_setup.jl")
include("download_utils.jl")

using CUDA: @allowscalar
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Operators: volume
using Oceananigans.Units
using NumericalEarth.Diagnostics: frazil_heat_flux
using NumericalEarth.Oceans: get_radiative_forcing

<<<<<<< HEAD
function test_tracer_budget(coupled_model, reference_salinity, Δt, nsteps; rtol)
=======
# Heat and freshwater have different noise floors, so they get separate tolerances.
function test_tracer_budget(coupled_model, Sᵒᶜ, Δt, nsteps; heat_rtol, freshwater_rtol)
>>>>>>> main
    ocean = coupled_model.ocean
    grid  = ocean.model.grid

    ocean_properties = coupled_model.interfaces.ocean_properties
    ρᵒᶜ = ocean_properties.reference_density
    cᵒᶜ = ocean_properties.heat_capacity
    Sᵒᶜ = reference_salinity

    T = ocean.model.tracers.T
    S = ocean.model.tracers.S

<<<<<<< HEAD
    heat_rate       = Integral(net_ocean_heat_flux(coupled_model), dims=(1, 2))
    freshwater_rate = Integral(net_ocean_freshwater_flux(coupled_model; reference_salinity = Sᵒᶜ), dims=(1, 2))

    # frazil_heat_flux is updated in update_state! at the END of each time_step!, so it
    # reflects the clamping applied in that step. We measure it both before and after each
    # step so that the expected budget uses the post-step (current) value rather than the
    # pre-step (previous) value, eliminating the one-step lag.
    # ZeroField has no grid and cannot be integrated, so we short-circuit to zero.
    frazil = frazil_heat_flux(coupled_model)
    frazil_rate = frazil isa ZeroField ? nothing : Integral(frazil, dims=(1, 2))
    measure_frazil() = isnothing(frazil_rate) ? zero(eltype(grid)) :
                       @allowscalar first(Field(frazil_rate))
=======
    # The surface tracer fluxes are discrete boundary conditions (the virtual salt flux and heat
    # exchange are evaluated live), so the applied fluxes are read from the assembled net-flux
    # fields rather than integrated from the boundary condition.
    Jᵀ  = coupled_model.interfaces.net_fluxes.ocean.T   # surface heat flux (turbulent + radiative)
    Jʷ  = coupled_model.interfaces.net_fluxes.ocean.η   # freshwater volume flux
    Jᴴ = coupled_model.interfaces.net_fluxes.ocean.freshwater_heat_content # Σᵢ Tᵢ Jʷᵢ (rain − evap at SST)
    heat_rate     = Integral(ρᵒᶜ * cᵒᶜ * Jᵀ,  dims=(1, 2))
    enthalpy_rate = Integral(ρᵒᶜ * cᵒᶜ * Jᴴ, dims=(1, 2))
    volume_rate   = Integral(Jʷ, dims=(1, 2))
>>>>>>> main

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

    for _ = 1:nsteps
        set!(VT⁻, T * volume)
        set!(VV⁻, cell_volume)

<<<<<<< HEAD
        previous_heat_flux       = @allowscalar first(Field(heat_rate))
        previous_freshwater_flux = @allowscalar first(Field(freshwater_rate))
        previous_frazil_rate     = measure_frazil()
        previous_radiative_rate  = isnothing(radiative_rate) ? zero(previous_heat_flux) :
                                   @allowscalar first(Field(radiative_rate))
=======
        previous_heat_flux      = @allowscalar first(Field(heat_rate))
        previous_enthalpy       = @allowscalar first(Field(enthalpy_rate))
        previous_volume_flux    = @allowscalar first(Field(volume_rate))
        previous_radiative_rate = isnothing(radiative_rate) ? zero(previous_heat_flux) : @allowscalar first(Field(radiative_rate))
>>>>>>> main

        time_step!(coupled_model, Δt)
        last_Δt = ocean.model.clock.last_Δt

        compute!(ΔVT)
        compute!(ΔVV)

        # Heat content changes by the surface heat flux plus the enthalpy carried by the freshwater
        # (rain − evaporation at SST). The live Tᴺ Jʷ exchange cancels the z-star ambient carry, so
        # the freshwater's own enthalpy Σᵢ Tᵢ Jʷᵢ is what remains.
        heat_content_tendency = sum(ρᵒᶜ * cᵒᶜ * ΔVT)
        expected_heat_content_tendency = (previous_radiative_rate - previous_heat_flux + previous_enthalpy) * last_Δt
        @test isapprox(heat_content_tendency, expected_heat_content_tendency; rtol=heat_rtol)

<<<<<<< HEAD
        current_frazil_rate = measure_frazil()

        # net_ocean_heat_flux = ρᵒᶜ * Jᵀ + frazil_heat. Jᵀ is set before the ocean step so
        # previous_heat_flux correctly captures the surface flux applied this step.
        # frazil_heat is set after the step, so we replace previous_frazil_rate with the
        # post-step current_frazil_rate to get exact budget closure.
        expected_heat_content_tendency       = (previous_radiative_rate
                                                - previous_heat_flux
                                                + previous_frazil_rate
                                                - current_frazil_rate) * last_Δt
        expected_freshwater_content_tendency = -previous_freshwater_flux * last_Δt

        heat_rtol       = abs(heat_content_tendency - expected_heat_content_tendency) /
                          max(abs(expected_heat_content_tendency), eps(typeof(expected_heat_content_tendency)))
        freshwater_rtol = abs(freshwater_content_tendency - expected_freshwater_content_tendency) /
                          max(abs(expected_freshwater_content_tendency), eps(typeof(expected_freshwater_content_tendency)))

        @debug "Heat budget rtol: $heat_rtol (tolerance: $rtol)"
        @test isapprox(heat_content_tendency, expected_heat_content_tendency; rtol)

        @debug "Freshwater budget rtol: $freshwater_rtol (tolerance: $rtol)"
        @test isapprox(freshwater_content_tendency, expected_freshwater_content_tendency; rtol)
=======
        # Volume grows by exactly the surface-integrated freshwater volume flux.
        volume_tendency = sum(ΔVV)
        expected_volume_tendency = previous_volume_flux * last_Δt
        @test isapprox(volume_tendency, expected_volume_tendency; rtol=freshwater_rtol)
>>>>>>> main
    end

    # Freshwater carries no salt, so the total salt content is conserved over the run.
    compute!(ΔVS)
    @test abs(sum(ΔVS)) / ∫S⁻ < freshwater_rtol

    return nothing
end

@testset "Tracer budget closure under surface fluxes" begin
    for arch in test_architectures
        for z in (MutableVerticalDiscretization((-100, 0)), (-100, 0))
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

<<<<<<< HEAD
            ecco_set = MetadataSet(:temperature, :salinity,
                                   :sea_ice_thickness, :sea_ice_concentration,
                                   dataset = ECCO4Monthly(),
                                   date = DateTime(1993, 1, 1))
=======
            # An idealized, stably stratified initial state
            Tᵢ(λ, φ, z) = 2 + 26 * cosd(φ)^2 * exp(z / 30)
            Sᵢ(λ, φ, z) = 35 - 1//2 * exp(z / 30)
>>>>>>> main

            Δt = 605seconds
            Sᵒᶜ = 35 # reference salinity [psu]
            free_surface = SplitExplicitFreeSurface(substeps=20)

<<<<<<< HEAD
            # OceanSeaIceModel without SeaIce
            @testset "Surface-only fluxes without shortwave penetration" begin
                @info "    .. Surface-only fluxes without shortwave penetration"
                new_grid = deepcopy(grid) # because the grid is mutable
                ocean = ocean_simulation(new_grid; free_surface, radiative_forcing=nothing)
                set!(ocean.model, ecco_set)
                coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation)
                test_tracer_budget(coupled_model, Sᵒᶜ, Δt, 4; rtol=2√eps(eltype(new_grid)))
            end

            @testset "Surface fluxes + OceanOnlyModel" begin
                @info "    .. Surface fluxes + OceanOnlyModel"
                new_grid = deepcopy(grid) # because the grid is mutable
                ocean = ocean_simulation(new_grid; free_surface)
                set!(ocean.model, ecco_set)
                coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)
                # rtol scales with nsteps: computing ΔVT = T_new⋅V - T_old⋅V subtracts two large
                # numbers; the O(eps⋅|T⋅V|) absolute error per step accumulates additively.
                test_tracer_budget(coupled_model, Sᵒᶜ, Δt, 4; rtol=2√eps(eltype(new_grid)))
            end

            # With penetrative shortwave radiation
            @testset "Surface fluxes + penetrating shortwave radiation" begin
                @info "    .. Surface fluxes + penetrating shortwave radiation"
                new_grid = deepcopy(grid) # because the grid is mutable
                ocean = ocean_simulation(new_grid; free_surface)
                set!(ocean.model, ecco_set)
                coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation)
                test_tracer_budget(coupled_model, Sᵒᶜ, Δt, 4; rtol=2√eps(eltype(new_grid)))
=======
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
>>>>>>> main
            end

            @testset "Surface fluxes + penetrating shortwave radiation + Sea ice" begin
                @info "    .. Surface fluxes + penetrating shortwave radiation + Sea ice"
                for name in (:sea_ice_thickness, :sea_ice_concentration)
                    metadata = ecco_set[name]
                    download_dataset_with_fallback(metadata_path(metadata);
                                                   dataset_name = "ECCO4Monthly $name") do
                        download(metadata)
                    end
                end

                new_grid = deepcopy(grid) # because the grid is mutable
                ocean = ocean_simulation(new_grid; free_surface)
                sea_ice = sea_ice_simulation(new_grid, ocean)
                set!(ocean.model, ecco_set)
                set!(sea_ice.model, ecco_set)
                coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
                test_tracer_budget(coupled_model, Sᵒᶜ, Δt, 4; rtol=2√eps(eltype(new_grid)))
            end
        end
        end
    end
end
