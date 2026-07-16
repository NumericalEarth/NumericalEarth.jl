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

function test_tracer_budget(coupled_model, reference_salinity, Δt, nsteps; rtol)
    ocean = coupled_model.ocean
    grid  = ocean.model.grid

    ocean_properties = coupled_model.interfaces.ocean_properties
    ρᵒᶜ = ocean_properties.reference_density
    cᵒᶜ = ocean_properties.heat_capacity
    Sᵒᶜ = reference_salinity

    T = ocean.model.tracers.T
    S = ocean.model.tracers.S

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

    penetrating_radiation = get_radiative_forcing(ocean)
    if isnothing(penetrating_radiation)
        radiative_rate = nothing
    else
        radiative_forcing = KernelFunctionOperation{Center, Center, Center}(penetrating_radiation, grid,
                                                                            ocean.model.clock,
                                                                            Oceananigans.fields(ocean.model))
        radiative_rate = Integral(ρᵒᶜ * cᵒᶜ * radiative_forcing, dims=(1, 2, 3))
    end

    VT⁻ = CenterField(grid)
    VS⁻ = CenterField(grid)
    ΔVT = Field(T * volume - VT⁻)
    ΔVS = Field(S * volume - VS⁻)

    for _ = 1:nsteps
        set!(VT⁻, T * volume)
        set!(VS⁻, S * volume)

        previous_heat_flux       = @allowscalar first(Field(heat_rate))
        previous_freshwater_flux = @allowscalar first(Field(freshwater_rate))
        previous_frazil_rate     = measure_frazil()
        previous_radiative_rate  = isnothing(radiative_rate) ? zero(previous_heat_flux) :
                                   @allowscalar first(Field(radiative_rate))

        time_step!(coupled_model, Δt)
        last_Δt = ocean.model.clock.last_Δt

        compute!(ΔVT)
        compute!(ΔVS)

        heat_content_tendency       = sum( ρᵒᶜ * cᵒᶜ * ΔVT)
        freshwater_content_tendency = sum(-ρᵒᶜ / Sᵒᶜ * ΔVS)

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
    end

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

            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                                        active_cells_map = true)

            time_indices_in_memory = 4
            radiation  = JRA55PrescribedRadiation(arch; time_indices_in_memory)
            atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory)

            ecco_set = MetadataSet(:temperature, :salinity,
                                   :sea_ice_thickness, :sea_ice_concentration,
                                   dataset = ECCO4Monthly(),
                                   date = DateTime(1993, 1, 1))

            Δt = 605seconds
            Sᵒᶜ = 35 # reference salinity [psu]
            free_surface = SplitExplicitFreeSurface(substeps=20)

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
