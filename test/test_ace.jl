include("runtests_setup.jl")

using ConservativeRegridding
using CondaPkg
using PythonCall
using NumericalEarth
using Oceananigans
using Oceananigans.Fields: interior
using Oceananigans.Units
using Dates: DateTime
using SeawaterPolynomials.TEOS10: θ_from_Θ
using Test

Python = Base.get_extension(NumericalEarth, :NumericalEarthPythonCallExt)
ace = Python.load_ace_module()

stub = pydict()
pyexec("""
import types
import numpy, torch, xarray

class StubStepper:
    prognostic_names = frozenset({"surface_temperature", "TMP2m", "UGRD10m", "VGRD10m", "Q2m", "PRESsfc"})
    out_names = prognostic_names | {"PRATEsfc", "DSWRFsfc", "DLWRFsfc"}
    config = types.SimpleNamespace(input_names=prognostic_names | {"land_fraction", "ocean_fraction", "sea_ice_fraction"},
                                   next_step_forcing_names=[])
    calls = 0

    def step(self, args):
        self.calls += 1
        T = args.input["surface_temperature"]
        output = {name: args.input[name] for name in self.prognostic_names}
        output["TMP2m"] = T - 2
        output["PRATEsfc"] = args.input["sea_ice_fraction"]
        output["DSWRFsfc"] = 100 * args.input["ocean_fraction"]
        output["DLWRFsfc"] = torch.full_like(T, 300.0)
        return types.SimpleNamespace(output=output, stepper_state=None)

def stub_emulator(Emulator, Nφ, Nλ, land_temperature):
    state = {name: torch.zeros(1, Nφ, Nλ) for name in StubStepper.prognostic_names}
    state["surface_temperature"][:] = land_temperature
    state["TMP2m"][:] = 290.0
    state["Q2m"][:] = 0.01
    state["PRESsfc"][:] = 101325.0
    time = numpy.datetime64("2020-01-01T00:00") + numpy.arange(8) * numpy.timedelta64(6, "h")
    land = numpy.zeros((len(time), Nφ, Nλ), dtype=numpy.float32)
    land[:, 0, :] = 1
    land[:, 1, :] = 0.5
    ice = numpy.float32(0.8) * (1 - land)
    dimensions = ("time", "latitude", "longitude")
    forcing = xarray.Dataset({"land_fraction": (dimensions, land),
                              "surface_temperature": (dimensions, numpy.full_like(land, 260)),
                              "sea_ice_fraction": (dimensions, ice),
                              "ocean_fraction": (dimensions, 1 - land - ice)},
                             coords={"time": time})
    return Emulator(StubStepper(), state, {2020: forcing})
""", stub)

# Water between 60°S and 60°N, with a land block that shares ACE cells with the water
function ocean_grid()
    underlying_grid = LatitudeLongitudeGrid(size = (24, 12, 1), longitude = (0, 360), latitude = (-60, 60), z = (-50, 0))
    land(λ, φ) = ifelse((90 < λ < 150) & (-15 < φ < 15), 0, -50)
    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(land))
end

ace_grid() = LatitudeLongitudeGrid(size = (8, 6), longitude = (0, 360), latitude = (-90, 90),
                                   topology = (Periodic, Bounded, Flat), halo = (3, 3))

function stub_atmosphere(land_temperature)
    grid = ace_grid()
    emulator = stub["stub_emulator"](ace.Emulator, size(grid, 2), size(grid, 1), land_temperature)
    return Python.ACEAtmosphere(grid, ace, emulator, DateTime(2020, 1, 1), 6hours)
end

air_temperature(atmosphere) = Array(interior(atmosphere.surface_state.T, :, :, 1))

@testset "ACE responds to the ocean and the sea ice" begin
    @testset "Ocean" begin
        atmosphere = stub_atmosphere(250)
        ocean = ocean_simulation(ocean_grid(); momentum_advection = nothing, tracer_advection = nothing, closure = nothing)
        set!(ocean.model, T = 20, S = 35)

        model = OceanOnlyModel(ocean; atmosphere)
        time_step!(model, 60minutes)

        T = air_temperature(atmosphere)
        θ = θ_from_Θ(35, 20)
        # Rows 3 to 5 read the SST, half-land row 2 averages it with ACE's land temperature, the northern row reads the forcing file
        @test all(isapprox.(T[:, 3:end-1], 273.15 + θ - 2; rtol = eps(Float32)))
        @test all(isapprox.(T[:, 2], 0.5 * (273.15 + θ) + 0.5 * 250 - 2; rtol = eps(Float32)))
        @test all(isapprox.(T[:, 1], 250 - 2; rtol = eps(Float32)))
        @test all(isapprox.(T[:, end], 260 - 2; rtol = eps(Float32)))

        ℐꜜˢʷ = Array(interior(atmosphere.downwelling_radiation.ℐꜜˢʷ, :, :, 1))
        @test all(isapprox.(ℐꜜˢʷ[:, 3:end-1], -100; rtol = eps(Float32)))
        @test all(isapprox.(ℐꜜˢʷ[:, 2], -50; rtol = eps(Float32)))
        @test all(isapprox.(ℐꜜˢʷ[:, end], -100 * 0.2; rtol = eps(Float32)))
        @test all(iszero, ℐꜜˢʷ[:, 1])
    end

    @testset "Sea ice" begin
        atmosphere = stub_atmosphere(250)
        grid = ocean_grid()
        ocean = ocean_simulation(grid; momentum_advection = nothing, tracer_advection = nothing, closure = nothing)
        set!(ocean.model, T = -1, S = 35)
        sea_ice = sea_ice_simulation(grid, ocean)
        set!(sea_ice.model, h = 1, ℵ = 0.5)

        model = OceanSeaIceModel(ocean, sea_ice; atmosphere)
        time_step!(model, 60minutes)

        Jʳⁿ = Array(interior(atmosphere.surface_state.Jʳⁿ, :, :, 1))
        @test all(isapprox.(Jʳⁿ[:, 3:end-1], 0.5; rtol = eps(Float32)))
        @test all(isapprox.(Jʳⁿ[:, 2], 0.5 * 0.5; rtol = eps(Float32)))
        @test all(isapprox.(Jʳⁿ[:, 1], 0; rtol = eps(Float32)))
        @test all(isapprox.(Jʳⁿ[:, end], 0.8; rtol = eps(Float32)))

        ℐꜜˢʷ = Array(interior(atmosphere.downwelling_radiation.ℐꜜˢʷ, :, :, 1))
        @test all(isapprox.(ℐꜜˢʷ[:, 3:end-1], -50; rtol = eps(Float32)))
        @test all(isapprox.(ℐꜜˢʷ[:, 2], -25; rtol = eps(Float32)))
        @test all(isapprox.(ℐꜜˢʷ[:, end], -100 * 0.2; rtol = eps(Float32)))
        @test all(iszero, ℐꜜˢʷ[:, 1])
    end
end

@testset "ACE steps every 6 hours" begin
    atmosphere = stub_atmosphere(250)
    ocean = ocean_simulation(ocean_grid(); momentum_advection = nothing, tracer_advection = nothing, closure = nothing)
    set!(ocean.model, T = 20, S = 35)
    model = OceanOnlyModel(ocean; atmosphere)

    for _ in 1:7
        time_step!(model, 60minutes)
    end

    @test pyconvert(Int, atmosphere.emulator.stepper.calls) == 2
end
