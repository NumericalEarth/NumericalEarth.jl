include("runtests_setup.jl")

using CUDA: @allowscalar
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Operators: volume
using Oceananigans.Units
using NumericalEarth.Oceans: get_radiative_forcing

function test_tracer_budget(coupled_model, Δt, nsteps; rtol=1e-8)
    ocean = coupled_model.ocean
    grid  = ocean.model.grid

    ocean_properties = coupled_model.interfaces.ocean_properties
    ρᵒᶜ = ocean_properties.reference_density
    cᵒᶜ = ocean_properties.heat_capacity

    T = ocean.model.tracers.T
    S = ocean.model.tracers.S

    # The surface tracer fluxes are discrete boundary conditions (the virtual salt flux and heat
    # exchange are evaluated live), so the applied fluxes are read from the assembled net-flux
    # fields rather than integrated from the boundary condition.
    Jᵀ  = coupled_model.interfaces.net_fluxes.ocean.T   # surface heat flux (turbulent + radiative)
    Jʷ  = coupled_model.interfaces.net_fluxes.ocean.η   # freshwater volume flux
    Jᴴ = coupled_model.interfaces.net_fluxes.ocean.freshwater_heat_content # Σᵢ Tᵢ Jʷᵢ (rain − evap at SST)
    heat_rate     = Integral(ρᵒᶜ * cᵒᶜ * Jᵀ,  dims=(1, 2))
    enthalpy_rate = Integral(ρᵒᶜ * cᵒᶜ * Jᴴ, dims=(1, 2))
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

    for _ = 1:nsteps
        set!(VT⁻, T * volume)
        set!(VV⁻, cell_volume)

        previous_heat_flux      = @allowscalar first(Field(heat_rate))
        previous_enthalpy       = @allowscalar first(Field(enthalpy_rate))
        previous_volume_flux    = @allowscalar first(Field(volume_rate))
        previous_radiative_rate = isnothing(radiative_rate) ? zero(previous_heat_flux) : @allowscalar first(Field(radiative_rate))

        time_step!(coupled_model, Δt)
        last_Δt = ocean.model.clock.last_Δt

        compute!(ΔVT)
        compute!(ΔVV)

        # Heat content changes by the surface heat flux plus the enthalpy carried by the freshwater
        # (rain − evaporation at SST). The live Tᴺ Jʷ exchange cancels the z-star ambient carry, so
        # the freshwater's own enthalpy Σᵢ Tᵢ Jʷᵢ is what remains.
        heat_content_tendency = sum(ρᵒᶜ * cᵒᶜ * ΔVT)
        expected_heat_content_tendency = (previous_radiative_rate - previous_heat_flux + previous_enthalpy) * last_Δt
        @test isapprox(heat_content_tendency, expected_heat_content_tendency; rtol)

        # Volume grows by exactly the surface-integrated freshwater volume flux.
        volume_tendency = sum(ΔVV)
        expected_volume_tendency = previous_volume_flux * last_Δt
        @test isapprox(volume_tendency, expected_volume_tendency; rtol)
    end

    # Freshwater carries no salt, so the total salt content is conserved over the run.
    compute!(ΔVS)
    @test abs(sum(ΔVS)) / ∫S⁻ < rtol

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

            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map = true)

            time_indices_in_memory = 4
            radiation  = JRA55PrescribedRadiation(arch; time_indices_in_memory)
            atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory)

            en4_set = MetadataSet(:temperature, :salinity, dataset = EN4Monthly(), date = DateTime(1993, 1, 1))
            Δt  = 605seconds
            Sᵒᶜ = 35 # reference salinity [psu]
            free_surface = SplitExplicitFreeSurface(substeps=20)

            # Without shortwave penetration
            @testset "Surface-only fluxes" begin
                ocean = ocean_simulation(deepcopy(grid); free_surface, radiative_forcing=nothing)
                set!(ocean.model, en4_set)
                coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation)
                test_tracer_budget(coupled_model, Δt, 4)
            end

            # With penetrative shortwave radiation
            @testset "Surface fluxes + Penetrating shortwave radiation" begin
                ocean = ocean_simulation(deepcopy(grid); free_surface)
                set!(ocean.model, en4_set)
                coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation)
                test_tracer_budget(coupled_model, Δt, 4)
            end
        end
    end
end
