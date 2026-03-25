using Oceananigans.Units
using Oceananigans.OutputReaders: TimeInterpolator, Cyclical

const ERDDAP_BASE = "https://data.pmel.noaa.gov/pmel/erddap/tabledap"
const ERDDAP_FLUX_VARS = "time,QLAT,QSEN,QNET,LWNET,SWNET,TAU,TAUX,TAUY,RAIN,EVAP,EMP,TSK"

"""
    download_ospapa_flux(; start_date, end_date, dir=download_OSPapa_cache)

Download precomputed air-sea fluxes for Ocean Station Papa from the PMEL ERDDAP
server (dataset `ocs_papa_flux`). The data is computed using COARE 3.0b and
includes latent/sensible heat fluxes, net radiation, wind stress, and
precipitation/evaporation at hourly resolution.

Returns the path to the downloaded NetCDF file.
"""
function download_ospapa_flux(; start_date, end_date, dir=download_OSPapa_cache)
    filename = "ocs_papa_flux_$(Dates.format(start_date, "yyyymmddTHHMMSS"))_$(Dates.format(end_date, "yyyymmddTHHMMSS")).nc"
    filepath = joinpath(dir, filename)
    if !isfile(filepath)
        t0 = Dates.format(start_date, "yyyy-mm-ddTHH:MM:SSZ")
        t1 = Dates.format(end_date, "yyyy-mm-ddTHH:MM:SSZ")
        url = "$(ERDDAP_BASE)/ocs_papa_flux.nc?$(ERDDAP_FLUX_VARS)&time>=$(t0)&time<=$(t1)"
        @info "Downloading OS Papa flux data from ERDDAP..."
        Downloads.download(url, filepath; progress=download_progress)
    end
    return filepath
end

"""
    OSPapaPrescribedFluxes(FT = Float64;
                           start_date = DateTime(2007, 6, 8),
                           end_date = DateTime(2022, 2, 24),
                           dir = download_OSPapa_cache,
                           max_gap_hours = 72)

Download precomputed air-sea fluxes from the PMEL ERDDAP `ocs_papa_flux` dataset
and return them as a `NamedTuple` of 1D arrays with associated time information.

The returned tuple contains:
- `Qnet`: net heat flux (W/m², positive into ocean)
- `Qlat`: latent heat flux (W/m², positive upward = ocean cooling)
- `Qsen`: sensible heat flux (W/m², positive upward = ocean cooling)
- `SWnet`: net shortwave radiation (W/m², positive downward)
- `LWnet`: net longwave radiation (W/m², positive upward = ocean cooling)
- `τx`, `τy`: zonal and meridional wind stress (N/m²)
- `evap`, `rain`, `EMP`: evaporation, precipitation, E-P (mm/hr)
- `Tsk`: skin temperature (°C)
- `times`: time in seconds relative to `start_date`
- `start_date`: the reference DateTime
"""
function OSPapaPrescribedFluxes(FT = Float64;
                                start_date = DateTime(2007, 6, 8),
                                end_date = DateTime(2022, 2, 24),
                                dir = download_OSPapa_cache,
                                max_gap_hours = 72)

    filepath = download_ospapa_flux(; start_date, end_date, dir)
    ds = NCDataset(filepath)

    times_datetime = ds["time"][:]
    Qlat  = Float64.(replace(ds["QLAT"][:],  missing => NaN))
    Qsen  = Float64.(replace(ds["QSEN"][:],  missing => NaN))
    Qnet  = Float64.(replace(ds["QNET"][:],  missing => NaN))
    LWnet = Float64.(replace(ds["LWNET"][:], missing => NaN))
    SWnet = Float64.(replace(ds["SWNET"][:], missing => NaN))
    τx    = Float64.(replace(ds["TAUX"][:],  missing => NaN))
    τy    = Float64.(replace(ds["TAUY"][:],  missing => NaN))
    rain  = Float64.(replace(ds["RAIN"][:],  missing => NaN))
    evap  = Float64.(replace(ds["EVAP"][:],  missing => NaN))
    EMP   = Float64.(replace(ds["EMP"][:],   missing => NaN))
    Tsk   = Float64.(replace(ds["TSK"][:],   missing => NaN))

    close(ds)

    # Fill small gaps
    for data in (Qlat, Qsen, Qnet, LWnet, SWnet, τx, τy, rain, evap, EMP, Tsk)
        fill_gaps!(data; max_gap=max_gap_hours)
    end

    # Times in seconds relative to start_date
    times = FT[Dates.value(t - times_datetime[1]) / 1000 for t in times_datetime]

    return (; Qnet = FT.(Qnet),
              Qlat = FT.(Qlat),
              Qsen = FT.(Qsen),
              SWnet = FT.(SWnet),
              LWnet = FT.(LWnet),
              τx = FT.(τx),
              τy = FT.(τy),
              evap = FT.(evap),
              rain = FT.(rain),
              EMP = FT.(EMP),
              Tsk = FT.(Tsk),
              times,
              start_date = times_datetime[1])
end

"""
    PrescribedFluxCallback(ocean, fluxes; ρ₀=reference_density(ocean), cₚ=heat_capacity(ocean))

Create a `Callback` that applies prescribed air-sea fluxes to an ocean simulation
at every time step. The callback interpolates flux time series to the current
model time and writes the result into the ocean's boundary condition fields.

Arguments
=========
- `ocean`: an `Oceananigans.Simulation` or ocean model returned by `ocean_simulation`
- `fluxes`: a `NamedTuple` returned by `OSPapaPrescribedFluxes`

Keyword Arguments
=================
- `ρ₀`: reference ocean density (default: from ocean's equation of state, typically 1020 kg/m³)
- `cₚ`: ocean heat capacity (default: from ocean's equation of state, typically 3991.87 J/(kg·K))

The callback converts:
- Wind stress (N/m²) → kinematic stress (m²/s²) by dividing by ρ₀
- Net heat flux (W/m²) → temperature flux (K·m/s) by dividing by ρ₀·cₚ
- Freshwater flux is applied as a salinity flux using surface salinity

!!! note "Sign conventions"
    ERDDAP QNET is positive into the ocean (warming). The callback converts
    to the Oceananigans convention where a positive top flux is *out of* the
    domain, so the sign is flipped for heat.
"""
function PrescribedFluxCallback(ocean, fluxes; ρ₀=reference_density(ocean), cₚ=heat_capacity(ocean))
    # Pre-extract BC fields (ocean_simulation returns a Simulation)
    model = ocean.model
    τˣ = model.velocities.u.boundary_conditions.top.condition
    τʸ = model.velocities.v.boundary_conditions.top.condition
    Jᵀ = model.tracers.T.boundary_conditions.top.condition
    Jˢ = model.tracers.S.boundary_conditions.top.condition

    flux_times = fluxes.times
    Nt = length(flux_times)
    Δt = flux_times[end] - flux_times[end-1]
    period = flux_times[end] - flux_times[1] + Δt
    time_indexing = Cyclical(period)

    function update_fluxes!(sim)
        t = sim.model.clock.time

        # Use Oceananigans' TimeInterpolator for consistency with PrescribedAtmosphere
        time_interpolator = TimeInterpolator(time_indexing, flux_times, t)
        n₁ = time_interpolator.first_index
        n₂ = time_interpolator.second_index
        α  = time_interpolator.fractional_index

        interp(field) = field[n₁] + α * (field[n₂] - field[n₁])

        # Interpolate fluxes
        τx_now  = interp(fluxes.τx)
        τy_now  = interp(fluxes.τy)
        Qnet_now = interp(fluxes.Qnet)

        # Momentum: stress (N/m²) → kinematic stress (m²/s²)
        # ERDDAP: TAUX > 0 = eastward stress ON ocean (downward momentum flux INTO domain)
        # Oceananigans FluxBoundaryCondition: positive top flux = OUT of domain
        # Therefore negate: downward flux into ocean → negative top BC
        τˣ[1, 1, 1] = -τx_now / ρ₀
        τʸ[1, 1, 1] = -τy_now / ρ₀

        # Heat: Qnet (W/m²) → temperature flux (K·m/s)
        # ERDDAP Qnet positive = into ocean (warming)
        # Oceananigans FluxBoundaryCondition: positive = out of domain
        Jᵀ[1, 1, 1] = -Qnet_now / (ρ₀ * cₚ)

        # Salinity: EMP (mm/hr ≡ kg/m²/hr) → salinity flux
        # EMP > 0 means net evaporation (ocean loses freshwater, salinity increases)
        # Convert mass flux to Boussinesq volume flux: divide by (ρ₀ * 3600)
        EMP_now = interp(fluxes.EMP)
        EMP_ms = EMP_now / (ρ₀ * 3600)  # kg/m²/hr → m/s (Boussinesq volume flux)

        # Salinity flux: following assemble_net_ocean_fluxes.jl: Jˢ = -S * ΣFao
        # EMP > 0 = net evaporation = salinity should increase
        # Negative Jˢ at top = salt INTO domain = S increases
        S_surface = model.tracers.S[1, 1, size(model.tracers.S, 3)]
        Jˢ[1, 1, 1] = -S_surface * EMP_ms

        return nothing
    end

    return Callback(update_fluxes!, IterationInterval(1))
end

using Oceananigans.OutputReaders: interpolating_time_indices

"""
    interp_flux(data, times, Nt, time_indexing, t)

Linearly interpolate a 1D flux time series to time `t` using Oceananigans'
`interpolating_time_indices`. GPU-kernel safe.
"""
@inline function interp_flux(data, times, Nt, time_indexing, t)
    ñ, n₁, n₂ = interpolating_time_indices(time_indexing, times, t)
    return @inbounds data[n₁] + ñ * (data[n₂] - data[n₁])
end

"""
    no_correction(i, j, grid, clock, model_fields, p)

Default correction function that returns zero. Used as the default value for
the correction keyword arguments in [`OSPapaPrescribedFluxBoundaryConditions`](@ref).
"""
no_correction(i, j, grid, clock, model_fields, p) = zero(grid)

"""
    OSPapaPrescribedFluxBoundaryConditions(fluxes, architecture=CPU(); ρ₀=1020.0, cₚ=3991.0)

Create Oceananigans `FluxBoundaryCondition`s for u, v, T, S from prescribed
OS Papa flux data. Returns a `NamedTuple` of `FieldBoundaryConditions` that
can be passed directly to `ocean_simulation` or `HydrostaticFreeSurfaceModel`
via the `boundary_conditions` keyword argument.

Uses discrete-form boundary condition functions that interpolate flux time
series at each grid point during tendency computation — no callback needed.
GPU-safe: flux data is transferred to the appropriate device and interpolation
uses direct index arithmetic (no `searchsortedfirst`).

Arguments
=========
- `fluxes`: a `NamedTuple` returned by [`OSPapaPrescribedFluxes`](@ref)
- `architecture`: `CPU()` or `GPU()` (default: `CPU()`)

Keyword Arguments
=================
- `ρ₀`: reference ocean density (default: 1020 kg/m³)
- `cₚ`: ocean heat capacity (default: 3991 J/(kg·K))
- `u_correction`: discrete-form correction function added to the zonal stress BC (default: [`no_correction`](@ref))
- `v_correction`: discrete-form correction function added to the meridional stress BC (default: [`no_correction`](@ref))
- `T_correction`: discrete-form correction function added to the temperature flux BC (default: [`no_correction`](@ref))
- `S_correction`: discrete-form correction function added to the freshwater (EMP) flux before computing salinity flux (default: [`no_correction`](@ref))

Each correction function must have the signature `(i, j, grid, clock, model_fields, p)` and return a value
in the same units as the corresponding flux boundary condition.

Example
=======
```julia
fluxes = OSPapaPrescribedFluxes(; start_date, end_date)
bcs = OSPapaPrescribedFluxBoundaryConditions(fluxes, GPU())
ocean = ocean_simulation(grid; Δt=10minutes, boundary_conditions=bcs)
```
"""
function OSPapaPrescribedFluxBoundaryConditions(fluxes, architecture=CPU(); 
                                                ρ₀=1020.0, cₚ=3991.0, 
                                                u_correction=no_correction, 
                                                v_correction=no_correction, 
                                                T_correction=no_correction, 
                                                S_correction=no_correction)

    flux_times = fluxes.times
    Nt = length(flux_times)
    Δt_data = flux_times[2] - flux_times[1]
    period = flux_times[end] - flux_times[1] + Δt_data
    time_indexing = Cyclical(period)

    # Transfer flux data and times to the appropriate device
    on_arch = arr -> Oceananigans.on_architecture(architecture, arr)

    times_arch = on_arch(flux_times)
    τx_data    = on_arch(fluxes.τx)
    τy_data    = on_arch(fluxes.τy)
    Qnet_data  = on_arch(fluxes.Qnet)
    EMP_data   = on_arch(fluxes.EMP)

    # Momentum: ERDDAP TAUX > 0 = eastward stress ON ocean (INTO domain)
    # Oceananigans: positive top flux = OUT of domain → negate
    @inline function τx_bc(i, j, grid, clock, model_fields, p)
        return -interp_flux(p.τx, p.times, p.Nt, p.time_indexing, clock.time) / p.ρ₀ + u_correction(i, j, grid, clock, model_fields, p)
    end

    @inline function τy_bc(i, j, grid, clock, model_fields, p)
        return -interp_flux(p.τy, p.times, p.Nt, p.time_indexing, clock.time) / p.ρ₀ + v_correction(i, j, grid, clock, model_fields, p)
    end

    # Heat: ERDDAP Qnet > 0 = into ocean → negate for Oceananigans
    @inline function Jᵀ_bc(i, j, grid, clock, model_fields, p)
        return -interp_flux(p.Qnet, p.times, p.Nt, p.time_indexing, clock.time) / (p.ρ₀ * p.cₚ) + T_correction(i, j, grid, clock, model_fields, p)
    end

    # Salinity: EMP (mm/hr ≡ kg/m²/hr) > 0 = net evaporation → salinity should increase
    # Jˢ = -S * EMP_ms (negative top flux = INTO domain = S increases)
    @inline function Jˢ_bc(i, j, grid, clock, model_fields, p)
        EMP_ms = interp_flux(p.EMP, p.times, p.Nt, p.time_indexing, clock.time) / (p.ρ₀ * 3600) + S_correction(i, j, grid, clock, model_fields, p)
        S = model_fields.S[i, j, grid.Nz]
        return -S * EMP_ms
    end

    params_τx = (; τx=τx_data, times=times_arch, Nt, time_indexing, ρ₀)
    params_τy = (; τy=τy_data, times=times_arch, Nt, time_indexing, ρ₀)
    params_T  = (; Qnet=Qnet_data, times=times_arch, Nt, time_indexing, ρ₀, cₚ)
    params_S  = (; EMP=EMP_data, times=times_arch, Nt, time_indexing, ρ₀)

    u_top = FluxBoundaryCondition(τx_bc, discrete_form=true, parameters=params_τx)
    v_top = FluxBoundaryCondition(τy_bc, discrete_form=true, parameters=params_τy)
    T_top = FluxBoundaryCondition(Jᵀ_bc, discrete_form=true, parameters=params_T)
    S_top = FluxBoundaryCondition(Jˢ_bc, discrete_form=true, parameters=params_S)

    return (; u = FieldBoundaryConditions(top=u_top),
              v = FieldBoundaryConditions(top=v_top),
              T = FieldBoundaryConditions(top=T_top),
              S = FieldBoundaryConditions(top=S_top))
end
