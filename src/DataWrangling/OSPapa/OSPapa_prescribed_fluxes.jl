using Oceananigans.Units
using Oceananigans.Units: Time
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

    # Build a uniform hourly grid and map raw (possibly gappy) ERDDAP data onto it.
    uniform_datetimes = start_date:Hour(1):end_date
    N_uniform = length(uniform_datetimes)

    dt_to_raw_idx = Dict(t => i for (i, t) in enumerate(times_datetime))

    function expand_to_uniform(raw_data)
        result = fill(NaN, N_uniform)
        for (j, t) in enumerate(uniform_datetimes)
            i = get(dt_to_raw_idx, t, nothing)
            isnothing(i) || (result[j] = raw_data[i])
        end
        return result
    end

    Qlat  = expand_to_uniform(Qlat)
    Qsen  = expand_to_uniform(Qsen)
    Qnet  = expand_to_uniform(Qnet)
    LWnet = expand_to_uniform(LWnet)
    SWnet = expand_to_uniform(SWnet)
    τx    = expand_to_uniform(τx)
    τy    = expand_to_uniform(τy)
    rain  = expand_to_uniform(rain)
    evap  = expand_to_uniform(evap)
    EMP   = expand_to_uniform(EMP)
    Tsk   = expand_to_uniform(Tsk)

    for data in (Qlat, Qsen, Qnet, LWnet, SWnet, τx, τy, rain, evap, EMP, Tsk)
        fill_gaps!(data; max_gap=max_gap_hours)
    end

    # Times in seconds relative to start_date on the uniform grid
    times = FT[Dates.value(t - start_date) / 1000 for t in uniform_datetimes]

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
              start_date)
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
    time_indexing = Cyclical()  # period auto-inferred from times

    # Build FieldTimeSeries on a zero-dimensional grid for each flux variable
    flux_grid = RectilinearGrid(architecture, eltype(flux_times); size=(), topology=(Flat, Flat, Flat))
    on_arch = arr -> Oceananigans.on_architecture(architecture, arr)

    τx_fts = FieldTimeSeries{Center, Center, Nothing}(flux_grid, flux_times; time_indexing)
    parent(τx_fts) .= on_arch(reshape(fluxes.τx, 1, 1, 1, Nt))

    τy_fts = FieldTimeSeries{Center, Center, Nothing}(flux_grid, flux_times; time_indexing)
    parent(τy_fts) .= on_arch(reshape(fluxes.τy, 1, 1, 1, Nt))

    Qnet_fts = FieldTimeSeries{Center, Center, Nothing}(flux_grid, flux_times; time_indexing)
    parent(Qnet_fts) .= on_arch(reshape(fluxes.Qnet, 1, 1, 1, Nt))

    EMP_fts = FieldTimeSeries{Center, Center, Nothing}(flux_grid, flux_times; time_indexing)
    parent(EMP_fts) .= on_arch(reshape(fluxes.EMP, 1, 1, 1, Nt))

    # Momentum: ERDDAP TAUX > 0 = eastward stress ON ocean (INTO domain)
    @inline function τx_bc(i, j, grid, clock, model_fields, p)
        return -p.τx[1, 1, 1, Time(clock.time)] / p.ρ₀ + u_correction(i, j, grid, clock, model_fields, p)
    end

    @inline function τy_bc(i, j, grid, clock, model_fields, p)
        return -p.τy[1, 1, 1, Time(clock.time)] / p.ρ₀ + v_correction(i, j, grid, clock, model_fields, p)
    end

    # Heat: ERDDAP Qnet > 0 = into ocean → negate for Oceananigans
    @inline function Jᵀ_bc(i, j, grid, clock, model_fields, p)
        return -p.Qnet[1, 1, 1, Time(clock.time)] / (p.ρ₀ * p.cₚ) + T_correction(i, j, grid, clock, model_fields, p)
    end

    # Salinity: EMP (mm/hr ≡ kg/m²/hr) > 0 = net evaporation → salinity should increase
    @inline function Jˢ_bc(i, j, grid, clock, model_fields, p)
        EMP_ms = p.EMP[1, 1, 1, Time(clock.time)] / (p.ρ₀ * 3600)
        S = model_fields.S[i, j, grid.Nz]
        return -S * EMP_ms + S_correction(i, j, grid, clock, model_fields, p)
    end

    params_τx = (; τx=τx_fts, ρ₀)
    params_τy = (; τy=τy_fts, ρ₀)
    params_T  = (; Qnet=Qnet_fts, ρ₀, cₚ)
    params_S  = (; EMP=EMP_fts, ρ₀)

    u_top = FluxBoundaryCondition(τx_bc, discrete_form=true, parameters=params_τx)
    v_top = FluxBoundaryCondition(τy_bc, discrete_form=true, parameters=params_τy)
    T_top = FluxBoundaryCondition(Jᵀ_bc, discrete_form=true, parameters=params_T)
    S_top = FluxBoundaryCondition(Jˢ_bc, discrete_form=true, parameters=params_S)

    return (; u = FieldBoundaryConditions(top=u_top),
              v = FieldBoundaryConditions(top=v_top),
              T = FieldBoundaryConditions(top=T_top),
              S = FieldBoundaryConditions(top=S_top))
end
