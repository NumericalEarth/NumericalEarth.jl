include("runtests_setup.jl")

using NumericalEarth.OSPapa
using NumericalEarth.Atmospheres: PrescribedAtmosphere
using Oceananigans.BoundaryConditions: BoundaryCondition, Flux, getbc
using Oceananigans.Units: minutes
using CUDA: @allowscalar

const OSPAPA_TEST_START = DateTime(2012, 10, 1)
const OSPAPA_TEST_END   = DateTime(2012, 10, 3)

@testset "OS Papa Prescribed Atmosphere" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing OSPapaPrescribedAtmosphere on $A..."

        atmosphere = OSPapaPrescribedAtmosphere(arch;
                                                start_date = OSPAPA_TEST_START,
                                                end_date   = OSPAPA_TEST_END)

        @test atmosphere isa PrescribedAtmosphere

        # All expected fields are present
        @test haskey(atmosphere.velocities, :u)
        @test haskey(atmosphere.velocities, :v)
        @test haskey(atmosphere.tracers, :T)
        @test haskey(atmosphere.tracers, :q)
        @test !isnothing(atmosphere.pressure)
        @test !isnothing(atmosphere.downwelling_radiation)
        @test haskey(atmosphere.freshwater_flux, :rain)

        # Radiation sanity checks
        ℐꜜˢʷ = atmosphere.downwelling_radiation.shortwave
        ℐꜜˡʷ = atmosphere.downwelling_radiation.longwave

        @allowscalar begin
            sw_data = interior(ℐꜜˢʷ)
            lw_data = interior(ℐꜜˡʷ)

            @test all(sw_data .>= 0)                  # downwelling SW is non-negative
            @test all(lw_data .>= 0)                  # downwelling LW is non-negative
            @test maximum(sw_data) < 1500             # below solar constant
            @test maximum(lw_data) < 600              # reasonable atmospheric LW
            @test maximum(lw_data) > 50               # not all zero

            # Air temperature in physical range (K)
            T_data = interior(atmosphere.tracers.T)
            @test all(T_data .>= 240)
            @test all(T_data .<= 320)
        end
    end
end

@testset "OS Papa Prescribed Fluxes" begin
    @info "Testing os_papa_prescribed_fluxes on CPU..."

    fluxes = os_papa_prescribed_fluxes(; start_date = OSPAPA_TEST_START,
                                         end_date   = OSPAPA_TEST_END)

    # NamedTuple structure
    @test haskey(fluxes, :Qnet)
    @test haskey(fluxes, :τx)
    @test haskey(fluxes, :τy)
    @test haskey(fluxes, :EMP)

    times = fluxes.Qnet.times

    # Uniform hourly grid: length must match start_date:Hour(1):end_date
    expected_Nt = length(OSPAPA_TEST_START:Hour(1):OSPAPA_TEST_END)
    @test length(times) == expected_Nt

    # Times start at 0 and are uniformly spaced at 3600s intervals
    @test times[1] == 0.0
    @test times[end] ≈ (expected_Nt - 1) * 3600.0
    @test all(diff(collect(times)) .≈ 3600.0)

    # Times are monotonically increasing
    @test issorted(times)

    # No NaN after gap filling (short window with no large gaps)
    @test all(isfinite, fluxes.Qnet)
    @test all(isfinite, fluxes.τx)
    @test all(isfinite, fluxes.τy)
    @test all(isfinite, fluxes.EMP)
end

@testset "OS Papa Prescribed Flux BCs" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing os_papa_prescribed_flux_boundary_conditions on $A..."

        fluxes = os_papa_prescribed_fluxes(arch; start_date = OSPAPA_TEST_START,
                                                 end_date   = OSPAPA_TEST_END)

        bcs = os_papa_prescribed_flux_boundary_conditions(fluxes)

        # Returns BCs for all four fields
        @test haskey(bcs, :u)
        @test haskey(bcs, :v)
        @test haskey(bcs, :T)
        @test haskey(bcs, :S)

        # Each has a FluxBoundaryCondition at the top
        @test bcs.u.top isa BoundaryCondition{<:Flux}
        @test bcs.v.top isa BoundaryCondition{<:Flux}
        @test bcs.T.top isa BoundaryCondition{<:Flux}
        @test bcs.S.top isa BoundaryCondition{<:Flux}

        # Check the implemented sign conventions and salinity-coupled EMP flux.
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = -144.9, y = 50.1,
                               z = (-10, 0),
                               topology = (Flat, Flat, Bounded))

        S = CenterField(grid)
        set!(S, 35)

        clock = Clock(time = 0.0)
        fields = (; S)

        @allowscalar begin
            @test getbc(bcs.u.top, 1, 1, grid, clock, fields) ≈ -fluxes.τx[1, 1, 1, 1] / 1020.0
            @test getbc(bcs.v.top, 1, 1, grid, clock, fields) ≈ -fluxes.τy[1, 1, 1, 1] / 1020.0
            @test getbc(bcs.T.top, 1, 1, grid, clock, fields) ≈ -fluxes.Qnet[1, 1, 1, 1] / (1020.0 * 3991.0)
            @test getbc(bcs.S.top, 1, 1, grid, clock, fields) ≈ -35 * fluxes.EMP[1, 1, 1, 1] / (1020.0 * 3600.0)
        end
    end
end

@testset "OS Papa ocean profile set!" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing OSPapaMetadatum set! on $A..."

        grid = RectilinearGrid(arch;
                               size = 20,
                               x = -144.9, y = 50.1,
                               z = (-200, 0),
                               topology = (Flat, Flat, Bounded))

        T_field = CenterField(grid)
        S_field = CenterField(grid)

        @test begin
            set!(T_field, Metadatum(:temperature, dataset=OSPapaHourly(), date=OSPAPA_TEST_START))
            set!(S_field, Metadatum(:salinity,    dataset=OSPapaHourly(), date=OSPAPA_TEST_START))
            true
        end

        # Values should be finite and physically reasonable
        @allowscalar begin
            T_interior = Array(interior(T_field, 1, 1, :))
            S_interior = Array(interior(S_field, 1, 1, :))
            @test all(isfinite, T_interior)
            @test all(isfinite, S_interior)
            @test all(T_interior .> -2)    # above freezing
            @test all(T_interior .< 35)    # below boiling
            @test all(S_interior .> 20)    # saline ocean
            @test all(S_interior .< 45)    # not hypersaline
        end
    end
end

@testset "OS Papa flux BC simulation" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing short simulation with os_papa_prescribed_flux_boundary_conditions on $A..."

        fluxes = os_papa_prescribed_fluxes(arch; start_date = OSPAPA_TEST_START,
                                                 end_date   = OSPAPA_TEST_END)

        bcs = os_papa_prescribed_flux_boundary_conditions(fluxes)

        grid = RectilinearGrid(arch;
                               size = 10,
                               x = -144.9, y = 50.1,
                               z = (-200, 0),
                               topology = (Flat, Flat, Bounded))

        ocean = ocean_simulation(grid; Δt = 10minutes,
                                 coriolis = FPlane(latitude=50.1),
                                 boundary_conditions = bcs)

        set!(ocean.model,
             T = Metadatum(:temperature, dataset=OSPapaHourly(), date=OSPAPA_TEST_START),
             S = Metadatum(:salinity,    dataset=OSPapaHourly(), date=OSPAPA_TEST_START))

        ocean.stop_iteration = 2

        @test begin
            run!(ocean)
            true
        end
    end
end

@testset "OS Papa prescribed atmosphere simulation" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing OceanOnlyModel with OSPapaPrescribedAtmosphere on $A..."

        grid = RectilinearGrid(arch;
                               size = 10,
                               x = -144.9, y = 50.1,
                               z = (-200, 0),
                               topology = (Flat, Flat, Bounded))

        ocean = ocean_simulation(grid; Δt = 10minutes,
                                 coriolis = FPlane(latitude=50.1))

        set!(ocean.model,
             T = Metadatum(:temperature, dataset=OSPapaHourly(), date=OSPAPA_TEST_START),
             S = Metadatum(:salinity,    dataset=OSPapaHourly(), date=OSPAPA_TEST_START))

        atmosphere = OSPapaPrescribedAtmosphere(arch;
                                                start_date = OSPAPA_TEST_START,
                                                end_date   = OSPAPA_TEST_END)

        coupled_model = OceanOnlyModel(ocean; atmosphere, radiation=Radiation(arch))
        simulation = Simulation(coupled_model; Δt=ocean.Δt, stop_iteration=2)

        @test begin
            run!(simulation)
            true
        end
    end
end
