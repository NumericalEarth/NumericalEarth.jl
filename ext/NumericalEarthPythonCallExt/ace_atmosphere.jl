const ace_version = "2026.10.0"

function install_ace()
    # netCDF4's macOS wheels target a newer macOS than pixi does, so pip would build it against the machine's libnetcdf
    CondaPkg.add("netcdf4"; channel = "conda-forge", resolve = false)
    # Python 3.14 resolves to its free-threaded build next to conda-forge's netCDF4, and fme's dependencies have no wheels for it
    CondaPkg.add("python"; version = "<3.14", resolve = false)
    CondaPkg.add_pip("fme"; version = "==" * ace_version)
    return nothing
end

function load_ace_module()
    util = pyimport("importlib.util")
    spec = util.spec_from_file_location("numericalearth_ace", joinpath(@__DIR__, "ace.py"))
    ace = util.module_from_spec(spec)
    spec.loader.exec_module(ace)
    return ace
end

struct ACEAtmosphere{G, C, FT, S, R, F}
    grid :: G
    clock :: C
    start_date :: DateTime
    time_step :: FT
    emulator :: Py
    inputs :: Py
    outputs :: Py
    surface_state :: S
    downwelling_radiation :: R
    surface_forcing :: F
end

# Julia's column-major (λ, φ) is torch's row-major [φ, λ]
function torch_tensor(ace, field)
    data = interior(field, :, :, 1)
    bytes = sizeof(eltype(data))
    return ace.julia_tensor(UInt(pointer(data)), reverse(size(data)), reverse(strides(data)) .* bytes, "<f$bytes",
                            architecture(field) isa GPU)
end

function ACEAtmosphere(grid, ace, emulator, start_date, time_step)
    surface_field() = Field{Center, Center, Nothing}(grid)
    surface_state = (; u = surface_field(), v = surface_field(), T = surface_field(), p = surface_field(),
                       q = surface_field(), Jʳⁿ = surface_field(), Jˢⁿ = surface_field())
    downwelling_radiation = (; ℐꜜˢʷ = surface_field(), ℐꜜˡʷ = surface_field())
    surface_forcing = (; surface_temperature = surface_field(), sea_ice_concentration = surface_field(),
                         water_fraction = surface_field())

    inputs = pydict(; (name => torch_tensor(ace, field) for (name, field) in pairs(surface_forcing))...)
    state_targets = pydict(UGRD10m = (torch_tensor(ace, surface_state.u), 1),
                           VGRD10m = (torch_tensor(ace, surface_state.v), 1),
                           TMP2m   = (torch_tensor(ace, surface_state.T), 1),
                           PRESsfc = (torch_tensor(ace, surface_state.p), 1),
                           Q2m     = (torch_tensor(ace, surface_state.q), 1))

    outputs = pybuiltins.dict(state_targets; PRATEsfc = (torch_tensor(ace, surface_state.Jʳⁿ), 1),
                                             DSWRFsfc = (torch_tensor(ace, downwelling_radiation.ℐꜜˢʷ), -1),
                                             DLWRFsfc = (torch_tensor(ace, downwelling_radiation.ℐꜜˡʷ), -1))

    KernelAbstractions.synchronize(device(architecture(grid)))
    emulator.copy_outputs(emulator.state, state_targets)

    return ACEAtmosphere(grid, Clock{Float64}(time = 0), start_date, convert(Float64, time_step), emulator,
                         inputs, outputs, surface_state, downwelling_radiation, surface_forcing)
end

"""
$(TYPEDSIGNATURES)

Return an `ACEAtmosphere` on `arch` that steps the ACE2-ERA5 emulator from its initial condition at the first of
`dates` with its forcing through the last, downloading the checkpoint and data on first use.
`fme` must be installed once with `install_ace()` from the `NumericalEarthPythonCallExt` extension, followed by a
Julia restart.
"""
function NumericalEarth.Atmospheres.atmosphere_simulation(dataset::ACE2ERA5, arch = CPU(); dates)
    start_date, end_date = dates
    year(start_date) in ace2_era5_initial_condition_years && start_date == DateTime(firstdayofmonth(start_date)) ||
        throw(ArgumentError("ACE2-ERA5 starts at 00:00 on the first of a month in $ace2_era5_initial_condition_years, not at $start_date"))

    ace = load_ace_module()

    checkpoint = ace_file(dataset, "ace2_era5_ckpt.tar")
    initial_condition = ace_file(dataset, "initial_conditions/ic_$(year(start_date)).nc")
    forcing = [ace_file(dataset, "forcing_data/forcing_$forcing_year.nc") for forcing_year in year(start_date):year(end_date)]

    emulator, latitude, longitude = ace.open_emulator(checkpoint, initial_condition, forcing, start_date, arch isa CPU)
    φ = pyconvert(Vector{Float64}, latitude)
    Nλ = pyconvert(Int, longitude.size)

    grid = LatitudeLongitudeGrid(arch; size = (Nλ, length(φ)), longitude = (0, 360),
                                 latitude = vcat(-90, (φ[1:end-1] .+ φ[2:end]) ./ 2, 90),
                                 topology = (Periodic, Bounded, Flat), halo = (3, 3))

    time_step = pyconvert(Float64, emulator.stepper.training_dataset_info.timestep.total_seconds())

    return ACEAtmosphere(grid, ace, emulator, start_date, time_step)
end

Base.summary(atmosphere::ACEAtmosphere) =
    string("ACEAtmosphere(", size(atmosphere.grid, 1), "×", size(atmosphere.grid, 2), ", Δt = ", prettytime(atmosphere.time_step), ")")

function Oceananigans.TimeSteppers.time_step!(atmosphere::ACEAtmosphere, Δt)
    rem(atmosphere.time_step, Δt) == 0 ||
        throw(ArgumentError("the coupled time step $(prettytime(Δt)) does not divide the ACE time step $(prettytime(atmosphere.time_step))"))

    if rem(atmosphere.clock.time, atmosphere.time_step) == 0
        time = atmosphere.start_date + Second(round(Int, atmosphere.clock.time))
        KernelAbstractions.synchronize(device(architecture(atmosphere.grid)))
        atmosphere.emulator.step(time, time + Second(round(Int, atmosphere.time_step)), atmosphere.inputs, atmosphere.outputs)
    end

    tick!(atmosphere.clock, Δt)
    return nothing
end

Oceananigans.Grids.grid(atmosphere::ACEAtmosphere) = atmosphere.grid
EarthSystemModels.surface_state(atmosphere::ACEAtmosphere) = atmosphere.surface_state
InterfaceComputations.net_fluxes(::ACEAtmosphere) = nothing
EarthSystemModels.thermodynamics_parameters(atmosphere::ACEAtmosphere) = AtmosphereThermodynamicsParameters(eltype(atmosphere.grid))
EarthSystemModels.surface_layer_height(atmosphere::ACEAtmosphere) = convert(eltype(atmosphere.grid), 10)
EarthSystemModels.boundary_layer_height(atmosphere::ACEAtmosphere) = convert(eltype(atmosphere.grid), 512)

EarthSystemModels.atmosphere_radiation(atmosphere::ACEAtmosphere; kw...) =
    AtmosphereRadiation(atmosphere, atmosphere.downwelling_radiation; kw...)

function InterfaceComputations.ComponentExchanger(atmosphere::ACEAtmosphere, exchange_grid; kw...)
    exchanger = invoke(ComponentExchanger, Tuple{Any, Any}, atmosphere, exchange_grid; kw...)
    return exchange_over_water(exchanger, exchange_grid, interior(atmosphere.surface_forcing.water_fraction))
end

function EarthSystemModels.update_net_fluxes!(coupled_model, atmosphere::ACEAtmosphere)
    regridder = coupled_model.interfaces.exchanger.atmosphere.regridder
    forcing = atmosphere.surface_forcing
    regrid_over_water!(interior(forcing.surface_temperature), regridder, coupled_model.interfaces.surface_partition.surface_temperature)
    ℵ = sea_ice_concentration(coupled_model.sea_ice)
    ℵ isa ZeroField || regrid_over_water!(interior(forcing.sea_ice_concentration), regridder, ℵ)

    return nothing
end
