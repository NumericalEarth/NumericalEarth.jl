include("runtests_setup.jl")

using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth.TPXO: TPXO10Atlas
using Oceananigans.BoundaryConditions: NormalFlowBoundaryCondition, NormalRadiation
using Oceananigans.Units: hours
using Statistics: mean

# NOAA CO-OPS published angular speeds [° per hour], an independent source for the frequencies that
# `TidalHarmonics` builds out of the constituents' astronomical arguments.
const noaa_speeds = (M2 = 28.9841042, S2 = 30.0,       N2 = 28.4397295, K2 = 30.0821373,
                     K1 = 15.0410686, O1 = 13.9430356, P1 = 14.9589314, Q1 = 13.3986609,
                     Mf =  1.0980331, Mm =  0.5443747)

# A TPXO-like atlas holding one constituent, written in the atlas's own layout: [latitude, longitude]
# integer arrays, elevations in millimeters and transports in centimeters squared per second. The
# elevation is uniform, and the transports are those of a basin centered on `center` that
# co-oscillates with it, U = -x ∂η/∂t / 2 and V = -y ∂η/∂t / 2.
function write_synthetic_atlas(dir, harmonics; amplitude = 0.5, center = (287, 35),
                               longitude = (282, 292), latitude = (30, 40), resolution = 1//4)

    λᶠ = collect(longitude[1]:resolution:(longitude[2] - resolution))
    φᶠ = collect(latitude[1]:resolution:(latitude[2] - resolution))
    Δ = resolution / 2

    NCDataset(joinpath(dir, "grid_tpxo10atlas_v2.nc"), "c") do atlas
        defDim(atlas, "nx", length(λᶠ))
        defDim(atlas, "ny", length(φᶠ))
        defVar(atlas, "lon_u", Float64, ("nx",))[:] = λᶠ
        defVar(atlas, "lat_v", Float64, ("ny",))[:] = φᶠ
    end

    ω = first(harmonics.frequencies)
    λ₀, φ₀ = center
    eastward(λ, φ) = -Oceananigans.defaults.planet_radius * cosd(φ) * deg2rad(λ - λ₀) * ω * amplitude / 2
    northward(λ, φ) = -Oceananigans.defaults.planet_radius * deg2rad(φ - φ₀) * ω * amplitude / 2

    constituent = lowercase(string(first(harmonics.constituents)))

    NCDataset(joinpath(dir, "h_$(constituent)_tpxo10_atlas_30_v2.nc"), "c") do atlas
        defDim(atlas, "nx", length(λᶠ))
        defDim(atlas, "ny", length(φᶠ))
        defVar(atlas, "hRe", Int32, ("ny", "nx"))[:, :] .= round(Int32, 1e3 * amplitude)
        defVar(atlas, "hIm", Int32, ("ny", "nx"))[:, :] .= Int32(0)
    end

    NCDataset(joinpath(dir, "u_$(constituent)_tpxo10_atlas_30_v2.nc"), "c") do atlas
        defDim(atlas, "nx", length(λᶠ))
        defDim(atlas, "ny", length(φᶠ))
        defVar(atlas, "uRe", Int32, ("ny", "nx"))[:, :] .= Int32(0)
        defVar(atlas, "vRe", Int32, ("ny", "nx"))[:, :] .= Int32(0)
        defVar(atlas, "uIm", Int32, ("ny", "nx"))[:, :] =
            [round(Int32, 1e4 * eastward(λ, φ + Δ)) for φ in φᶠ, λ in λᶠ]
        defVar(atlas, "vIm", Int32, ("ny", "nx"))[:, :] =
            [round(Int32, 1e4 * northward(λ + Δ, φ)) for φ in φᶠ, λ in λᶠ]
    end

    return dir
end

# Least-squares amplitude and phase lag of `constituent` in a time series, the analysis that a tide
# gauge record gets, so that a modeled tide can be compared with the constants that forced it.
function tidal_constants(times, series, harmonics)
    ω, f, Θ = first(harmonics.frequencies), first(harmonics.nodal_factors), first(harmonics.phases)
    design = [cos.(ω .* times .+ Θ) sin.(ω .* times .+ Θ) ones(length(times))]
    cosine, sine, _ = design \ series
    return (amplitude = sqrt(cosine^2 + sine^2) / f, phase_lag = mod(atan(-sine, cosine), 2π))
end

@testset "Tidal astronomy" begin
    harmonics = TidalHarmonics(DateTime(2019, 4, 1))

    speeds = rad2deg.(harmonics.frequencies) .* 3600
    @test all(isapprox(speed, noaa_speeds[name]; rtol = 1e-7)
              for (name, speed) in zip(harmonics.constituents, speeds))

    # The phases are the equilibrium arguments at the reference date, so they must advance with the
    # frequencies: this is what keeps the body force and the boundary tide in phase with each other.
    later = TidalHarmonics(DateTime(2019, 4, 1, 6))
    drift = @. mod(later.phases - harmonics.phases - harmonics.frequencies * 6hours, 2π)
    @test all(@. min(drift, 2π - drift) < 1e-4)
end

@testset "Equilibrium tidal body force" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch;
                                     size = (20, 12, 1),
                                     longitude = (-78, -68),
                                     latitude = (35, 41),
                                     z = (-4000, 0),
                                     topology = (Bounded, Bounded, Bounded))

        period = 2π / first(TidalHarmonics(DateTime(2019, 4, 1); constituents = (:M2,)).frequencies)
        harmonics = TidalHarmonics(DateTime(2019, 4, 1); constituents = (:M2,), ramp_time = period)

        model = HydrostaticFreeSurfaceModel(grid;
                                            forcing = tidal_forcing(harmonics),
                                            free_surface = SplitExplicitFreeSurface(grid; substeps = 30),
                                            coriolis = HydrostaticSphericalCoriolis(),
                                            momentum_advection = nothing,
                                            tracer_advection = nothing,
                                            buoyancy = nothing,
                                            tracers = (),
                                            closure = nothing)

        simulation = Simulation(model; Δt = 120, stop_time = 5period)

        times = Float64[]
        elevation = Matrix{Float64}[]
        function sample!(sim)
            push!(times, sim.model.clock.time)
            push!(elevation, Array(interior(sim.model.free_surface.displacement, :, :, 1)))
        end
        add_callback!(simulation, sample!, TimeInterval(period / 24))

        run!(simulation)

        # The equilibrium tide of one semidiurnal constituent, written out from its own harmonic
        # constants: η = f A cos²φ cos(ω t + Θ + 2λ).
        ω, f, A, Θ = (first(harmonics.frequencies), first(harmonics.nodal_factors),
                      first(harmonics.equilibrium_amplitudes), first(harmonics.phases))
        λ, φ = λnodes(grid, Center()), φnodes(grid, Center())
        equilibrium(t) = [f * A * cosd(φⱼ)^2 * cos(ω * t + Θ + 2 * deg2rad(λᵢ)) for λᵢ in λ, φⱼ in φ]

        anomaly(field) = field .- mean(field)
        analyzed = findall(t -> t > 3period, times)
        modeled = [anomaly(elevation[n]) for n in analyzed]
        expected = [anomaly(equilibrium(times[n])) for n in analyzed]

        # A basin this small responds statically, so it fills to the equilibrium tide itself.
        gain = sum(sum(m .* e) for (m, e) in zip(modeled, expected)) / sum(sum(abs2, e) for e in expected)
        @test 0.9 < gain < 1.15
    end
end

@testset "Tidal boundary conditions" begin
    for arch in test_architectures
        harmonics = TidalHarmonics(DateTime(2019, 4, 1); constituents = (:M2,))
        period = 2π / first(harmonics.frequencies)
        amplitude = 0.5
        dir = write_synthetic_atlas(mktempdir(), harmonics; amplitude)

        grid = LatitudeLongitudeGrid(arch;
                                     size = (8, 8, 1),
                                     longitude = (-74, -72),
                                     latitude = (34, 36),
                                     z = (-100, 0),
                                     topology = (Bounded, Bounded, Bounded))

        tide = tidal_boundary_conditions(grid, harmonics, TPXO10Atlas(); dir)

        open_scheme = NormalRadiation(inflow_timescale = Inf, outflow_timescale = Inf)
        open_flow = NormalFlowBoundaryCondition(0; scheme = open_scheme)
        u = FieldBoundaryConditions(west = open_flow, east = open_flow)
        v = FieldBoundaryConditions(south = open_flow, north = open_flow)

        model = HydrostaticFreeSurfaceModel(grid;
                                            boundary_conditions = merge(tide, (; u, v)),
                                            free_surface = SplitExplicitFreeSurface(grid; substeps = 30),
                                            coriolis = HydrostaticSphericalCoriolis(),
                                            momentum_advection = nothing,
                                            tracer_advection = nothing,
                                            buoyancy = nothing,
                                            tracers = (),
                                            closure = nothing)

        simulation = Simulation(model; Δt = 120, stop_time = 4period)

        times = Float64[]
        center = Float64[]
        i, j = size(grid, 1) ÷ 2, size(grid, 2) ÷ 2
        function sample!(sim)
            push!(times, sim.model.clock.time)
            push!(center, Array(interior(sim.model.free_surface.displacement, i:i, j:j, 1))[1])
        end
        add_callback!(simulation, sample!, TimeInterval(period / 24))

        run!(simulation)

        # A basin far shorter than the tidal wavelength co-oscillates with the tide imposed on its
        # boundaries, so the interior carries the atlas's own amplitude and phase.
        analyzed = findall(t -> t > 2period, times)
        constants = tidal_constants(times[analyzed], center[analyzed], harmonics)
        @test isapprox(constants.amplitude, amplitude; rtol = 0.1)
        @test min(constants.phase_lag, 2π - constants.phase_lag) < 0.2
    end
end
