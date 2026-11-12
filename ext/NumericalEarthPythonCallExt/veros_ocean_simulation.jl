using Oceananigans.Grids: Grids, topology
using Oceananigans.Fields: set!
using NumericalEarth: NumericalEarth
using NumericalEarth.EarthSystemModels: EarthSystemModel

const veros_version = "1.6.2"
const veros_requirements_url = "https://raw.githubusercontent.com/team-ocean/veros/v$(veros_version)/requirements.txt"

"""
    install_veros()

Install the Veros ocean model Marine CLI and its JAX backend using CondaPkg.
Returns a NamedTuple containing package information if successful.
Also patches Veros's signal handling to work with PythonCall.
Installing can replace the Python of the running session, so restart Julia before using Veros.
"""
function install_veros()
    # Veros' setup.py rewrites every `==` pin in its requirements.txt into a `<=` cap,
    # and pixi treats conda-installed packages as hard pins during the pip solve, so any
    # conda-forge release newer than a veros cap makes the environment unsatisfiable.
    # Mirroring the caps as conda pins keeps the conda solve within what veros accepts.
    requirements = Downloads.download(veros_requirements_url, IOBuffer())
    for line in eachline(seekstart(requirements))
        # skip requirements with environment markers (`pkg==1.0; python_version < ...`)
        pinned_requirement = match(r"^([A-Za-z0-9._-]+)==([^;\s]+)$", strip(line))
        isnothing(pinned_requirement) && continue
        package_name, pinned_version = pinned_requirement.captures
        CondaPkg.add(package_name; version="<=" * pinned_version, channel="conda-forge", resolve=false)
    end

    CondaPkg.add_pip("veros", version="==" * veros_version, extras=["jax"])
    cli = CondaPkg.which("veros")

    # Patch signal handling as early as possible
    patch_veros_signal_handling()

    @info "... the veros CLI has been installed at $(cli)."
    return cli
end

"""
    patch_veros_signal_handling()

Patch Veros's signal handling to work with PythonCall.
This prevents TypeError when Veros tries to set signal handlers from within Julia/PythonCall context.
The issue is that Veros tries to set signal handlers, but PythonCall's wrapped Python objects aren't recognized
as valid callable handlers by Python's signal.signal() function. We work around this by monkey-patching
signal.signal() to accept any handler and convert invalid ones to SIG_DFL.
"""
function patch_veros_signal_handling()
    pyexec("""
    import signal

    # Monkey-patch signal.signal() to handle invalid handlers gracefully
    # This is needed because PythonCall-wrapped objects aren't recognized
    # as valid callable handlers by Python's signal.signal()
    if not hasattr(signal, '_NumericalEarth_patched'):
        _original_signal = signal.signal

        def _patched_signal(signum, handler):
            # Check if handler is valid according to Python's rules
            if handler in (signal.SIG_IGN, signal.SIG_DFL):
                # Standard handlers are always valid
                return _original_signal(signum, handler)
            elif callable(handler):
                # Try to use the original handler if it's callable
                try:
                    return _original_signal(signum, handler)
                except TypeError:
                    # If Python rejects it (e.g., PythonCall wrapper), use SIG_DFL
                    return _original_signal(signum, signal.SIG_DFL)
            else:
                # Invalid handler type - use SIG_DFL instead
                return _original_signal(signum, signal.SIG_DFL)

        signal.signal = _patched_signal
        signal._NumericalEarth_patched = True

    # Also patch Veros's signal wrapper to skip signal handling entirely
    try:
        import veros.signals
        # Only patch if not already patched
        if not hasattr(veros.signals, '_NumericalEarth_patched'):
            def _patched_dnd_wrapper():
                # Skip signal handling entirely when running from PythonCall
                # The signal handlers aren't needed when Veros is embedded in Julia
                pass

            veros.signals.dnd_wrapper = _patched_dnd_wrapper
            veros.signals._NumericalEarth_patched = True
    except (ImportError, AttributeError):
        # If veros.signals doesn't exist or can't be patched, that's okay
        # This might happen if Veros version doesn't have signal handling
        pass
    """, Main)
end

struct VerosOceanSimulation{S, G, E, F}
    setup         :: S
    grid          :: G
    surface_state :: E
    net_fluxes    :: F
end

Oceananigans.initialize!(::VerosOceanSimulation) = nothing
Oceananigans.Diagnostics.default_nan_checker(::EarthSystemModel{<:Any, <:Any, <:Any, <:Any, <:VerosOceanSimulation}) = nothing

function Oceananigans.TimeSteppers.time_step!(ocean::VerosOceanSimulation, Δt)
    set!(ocean, "dt_tracer", Δt; path=:settings)
    set!(ocean, "dt_mom",    Δt; path=:settings)
    ocean.setup.step(ocean.setup.state)
    NumericalEarth.EarthSystemModels.update_surface_state!(ocean.surface_state, ocean)
    return nothing
end

Base.eltype(::VerosOceanSimulation) = Float64

NumericalEarth.EarthSystemModels.reference_density(ocean::VerosOceanSimulation) = pyconvert(eltype(ocean), ocean.setup.state.settings.rho_0)
NumericalEarth.EarthSystemModels.heat_capacity(ocean::VerosOceanSimulation) = convert(eltype(ocean), 3991.86795711963)

# Prognostic variables carry three time levels in their last dimension, and `tau` rotates through them every step.
function current_time_level(ocean::VerosOceanSimulation, name)
    variables = ocean.setup.state.variables
    τ = pyconvert(Int, variables.tau.item()) + 1
    return selectdim(PyArray(getproperty(variables, name)), 4, τ)
end

function remove_outputs(setup::Symbol)
    rm("$(setup).averages.nc", force=true)
    rm("$(setup).energy.nc", force=true)
    rm("$(setup).overturning.nc", force=true)
    rm("$(setup).snapshot.nc", force=true)
    return nothing
end

"""
    VerosOceanSimulation(setup, setup_name::Symbol; backend)

Creates and initializes a preconfigured Veros ocean simulation using the
specified setup module and setup name. The setup's `set_forcing` is replaced
so that the surface fluxes of the coupled model force the ocean.

Arguments
==========
- `setup::AbstractString`: The name of the Veros setup module to import (e.g., `"global_4deg"`).
- `setup_name::Symbol`: The name of the setup class or function within the module to instantiate (e.g., `:GlobalFourDegreeSetup`).

Keyword Arguments
=================
- `backend`: Veros's array backend, `"numpy"` or `"jax"`. Default: `VEROS_BACKEND` if set, otherwise `"numpy"`.
"""
function VerosOceanSimulation(setup::String, setup_name::Symbol; backend=get(ENV, "VEROS_BACKEND", "numpy"))
    # Patch signal handling BEFORE importing any Veros modules
    # Veros may set up signal handlers during module import
    patch_veros_signal_handling()

    # Veros refuses any assignment to its settings once its core modules are loaded, even of the same value
    runtime_settings = pyimport("veros").runtime_settings
    pyconvert(String, runtime_settings.backend) == backend || (runtime_settings.backend = backend)

    setups = pyimport("veros.setups." * setup)
    ocean  = @eval $setups.$setup_name()

    # Remove any existing output files to avoid "file exists" errors from Veros diagnostics
    remove_outputs(Symbol(setup))

    return VerosOceanSimulation(ocean)
end

function VerosOceanSimulation(setup::Py)
    patch_veros_signal_handling()
    setup.setup()

    # The only forcing left to Veros is the surface TKE input, which follows from the coupled wind stress.
    pyexec("""
    def set_tke_forcing(state):
        from veros.core.operators import numpy as npx, update, at

        vs = state.variables
        settings = state.settings

        if settings.enable_tke:
            vs.forc_tke_surface = update(
                vs.forc_tke_surface,
                at[1:-1, 1:-1],
                npx.sqrt(
                    (0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / settings.rho_0) ** 2
                    + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / settings.rho_0) ** 2
                ) ** 1.5,
            )

    setup.set_forcing = set_tke_forcing
    """, Main, (; setup))

    grid = surface_grid(setup)

    surface_state = (; u = Field{Face, Center, Nothing}(grid),
                       v = Field{Center, Face, Nothing}(grid),
                       T = Field{Center, Center, Nothing}(grid),
                       S = Field{Center, Center, Nothing}(grid))

    net_fluxes = (; u = Field{Face, Center, Nothing}(grid),
                    v = Field{Center, Face, Nothing}(grid),
                    T = Field{Center, Center, Nothing}(grid),
                    S = Field{Center, Center, Nothing}(grid),
                    η = Field{Center, Center, Nothing}(grid),
                    freshwater_heat_content = Field{Center, Center, Nothing}(grid),
                    u_coefficient = Field{Face, Center, Nothing}(grid),
                    v_coefficient = Field{Center, Face, Nothing}(grid))

    return VerosOceanSimulation(setup, grid, surface_state, net_fluxes)
end

# We assume that if we pass a python object, this is a veros simulation
NumericalEarth.Oceans.ocean_simulation(ocean::Py) = VerosOceanSimulation(ocean)

Grids.grid(ocean::VerosOceanSimulation) = ocean.grid

function surface_grid(setup)
    variables = setup.state.variables
    settings = setup.state.settings

    xf = Array(PyArray(variables.xu))[2:end-2]
    yf = Array(PyArray(variables.yu))[2:end-2]
    xc = Array(PyArray(variables.xt))[3:end-2]
    yc = Array(PyArray(variables.yt))[3:end-2]

    # The western face sits in the halo, which holds the wrapped longitude in a periodic domain.
    xf[1] = 2xc[1] - xf[2]
    yf[1] = 2yc[1] - yf[2]

    Δzᴺ = PyArray(variables.dzt)[end]
    TX = pyconvert(Bool, settings.enable_cyclic_x) ? Periodic : Bounded
    Nx = length(xc)
    Ny = length(yc)

    # TODO: fix in Oceananigans; its metrics on a stretched longitude end one point short of ClimaSeaIce's rheology in the halos
    Δλ = diff(xf)
    longitude = all(≈(first(Δλ)), Δλ) ? (first(xf), last(xf)) : xf

    grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                 longitude = longitude,
                                 latitude = yf,
                                 z = (-Δzᴺ, 0),
                                 topology = (TX, Bounded, Bounded),
                                 halo = (7, 7, 1),
                                 radius = pyconvert(Float64, settings.radius))

    maskT = PyArray(variables.maskT)
    surface_mask = view(maskT, 3:Nx+2, 3:Ny+2, size(maskT, 3))
    bottom_height = ifelse.(surface_mask .> 0, -2Δzᴺ, 0)

    return ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
end

"""
    set!(ocean, v, x; path = :variable)

Set the `v` variable in the `ocean` model to the value of `x`.
the path corresponds to the path inside the class where to locate the
variable `v` to set. It can be either `:variables` or `:settings`.
"""
function Oceananigans.Fields.set!(ocean::VerosOceanSimulation, v, x; path = :variables)
    setup = ocean.setup
    if path == :variables
        pyexec("""
        with setup.state.variables.unlock():
            setup.state.variables.__setattr__(y, t)
        """, Main, (y=v, t=x, setup=setup))
    elseif path == :settings
        pyexec("""
        with setup.state.settings.unlock():
            setup.state.settings.__setattr__(y, t)
        """, Main, (y=v, t=x, setup=setup))
    else
        error("path must be either :variable or :settings.")
    end
end

function Oceananigans.Simulations.reset_clock!(ocean::VerosOceanSimulation)
    set!(ocean, "time", 0)
    set!(ocean, "itt",  Int32(0))
    return ocean
end
