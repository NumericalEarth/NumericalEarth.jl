include("runtests_setup.jl")

using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth.TPXO: TPXO10Atlas
using Oceananigans.BoundaryConditions: NormalFlowBoundaryCondition, NormalRadiation
using Oceananigans.Units: hours

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
