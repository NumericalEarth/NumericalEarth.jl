using Oceananigans.Units
using Oceananigans.OutputReaders: Cyclical

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

Examples
========
```julia
# Basic usage on GPU:
fluxes = OSPapaPrescribedFluxes(; start_date, end_date)
bcs = OSPapaPrescribedFluxBoundaryConditions(fluxes, GPU())
ocean = ocean_simulation(grid; Δt=10minutes, boundary_conditions=bcs)

# With a uniform heat flux correction of +5 W/m² to close the heat budget:
heat_correction = (i, j, grid, clock, model_fields, p) -> 5.0 / (p.ρ₀ * p.cₚ)
bcs = OSPapaPrescribedFluxBoundaryConditions(fluxes, GPU(); T_correction=heat_correction)
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
