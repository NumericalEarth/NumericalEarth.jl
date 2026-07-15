include("runtests_setup.jl")

using CUDA: @allowscalar
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Operators: volume
using Oceananigans.Units
using NumericalEarth.Oceans: get_radiative_forcing

# Heat and freshwater have very different noise floors, so they get separate tolerances. Heat
# closes to ~1e-13 once a column absorbs exactly its surface flux. Freshwater only reaches ~1e-9:
# S ≈ 35 is nearly uniform, so `S * volume - VS⁻` differences ~1e18 to recover a signal ~1e-6 of
# it, and `eps` over that ratio is ~4e-10. A single `√eps` tolerance was loose enough to hide a
# 10⁴ error in the radiative budget.
function test_tracer_budget(coupled_model, Sᵒᶜ, Δt, nsteps; heat_rtol, freshwater_rtol)
    ocean = coupled_model.ocean
    grid  = ocean.model.grid

    ocean_properties = coupled_model.interfaces.ocean_properties
    ρᵒᶜ = ocean_properties.reference_density
    cᵒᶜ = ocean_properties.heat_capacity

    T = ocean.model.tracers.T
    S = ocean.model.tracers.S

    heat_rate       = Integral(ρᵒᶜ * cᵒᶜ * T.boundary_conditions.top.condition, dims=(1, 2))
    freshwater_rate = Integral(net_ocean_freshwater_flux(coupled_model; reference_salinity=Sᵒᶜ), dims=(1, 2))

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
        previous_radiative_rate  = isnothing(radiative_rate) ? zero(previous_heat_flux) :
                                   @allowscalar first(Field(radiative_rate))

        time_step!(coupled_model, Δt)
        last_Δt = ocean.model.clock.last_Δt

        compute!(ΔVT)
        compute!(ΔVS)

        heat_content_tendency       = sum( ρᵒᶜ * cᵒᶜ * ΔVT)
        freshwater_content_tendency = sum(-ρᵒᶜ / Sᵒᶜ * ΔVS)

        expected_heat_content_tendency       = (previous_radiative_rate - previous_heat_flux) * last_Δt
        expected_freshwater_content_tendency = -previous_freshwater_flux * last_Δt

        @test isapprox(heat_content_tendency, expected_heat_content_tendency; rtol=heat_rtol)
        @test isapprox(freshwater_content_tendency, expected_freshwater_content_tendency; rtol=freshwater_rtol)
    end

    return nothing
end

@testset "Tracer budget closure under surface fluxes" begin
    for arch in test_architectures
        for fold_topology in (RightFaceFolded,
                              # RightCenterFolded # requires https://github.com/CliMA/Oceananigans.jl/pull/5099
                              )

            @info ".. on $(typeof(arch)) with $fold_topology topology"
            underlying_grid = TripolarGrid(arch;
                                           size = (20, 20, 20),
                                           z = MutableVerticalDiscretization((-100, 0)),
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

            # An idealized, stably stratified initial state: a warm, slightly fresh surface over
            # a cold, saltier interior, with surface temperature falling from equator to pole. The
            # budget closes on any initial state, so this needs no dataset. It does need to stay in
            # a realistic regime: a uniform column convects, and a uniform surface leaves the fluxes
            # no meridional gradient. The halocline keeps the column stable at high latitude, where
            # T alone is nearly uniform.
            Tᵢ(λ, φ, z) = 2 + 26 * cosd(φ)^2 * exp(z / 30)
            Sᵢ(λ, φ, z) = 35 - 1//2 * exp(z / 30)

            Δt = 605seconds
            Sᵒᶜ = 35 # reference salinity [psu]
            free_surface = SplitExplicitFreeSurface(substeps=20)

            # `deepcopy` the grid per testset: the z-star vertical coordinate (σ, ∂t_σ) lives in
            # the grid, so timestepping mutates it. A second model built on the same grid starts
            # from an evolved σ but a zero η, and its budget misses by that inconsistency.

            # Without shortwave penetration
            @testset "Surface-only fluxes" begin
                ocean = ocean_simulation(deepcopy(grid); free_surface, radiative_forcing=nothing)
                set!(ocean.model, T=Tᵢ, S=Sᵢ)
                coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation)
                test_tracer_budget(coupled_model, Sᵒᶜ, Δt, 4; heat_rtol=1e-11,
                                   freshwater_rtol=√eps(eltype(grid)))
            end

            # With penetrative shortwave radiation
            @testset "Surface fluxes + Penetrating shortwave radiation" begin
                ocean = ocean_simulation(deepcopy(grid); free_surface)
                set!(ocean.model, T=Tᵢ, S=Sᵢ)
                coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation)
                test_tracer_budget(coupled_model, Sᵒᶜ, Δt, 4; heat_rtol=1e-11,
                                   freshwater_rtol=√eps(eltype(grid)))
            end
        end
    end
end
