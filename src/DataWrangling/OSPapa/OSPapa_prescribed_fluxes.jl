using Oceananigans.Units
using Oceananigans.Units: Time
using Oceananigans.OutputReaders: Cyclical
using Oceananigans.Architectures: on_architecture

"""
    OSPapaPrescribedFluxes(architecture, FT = Float64;
                            start_date = first_date(OSPapaFluxHourly(), :net_heat_flux),
                            end_date   = last_date(OSPapaFluxHourly(), :net_heat_flux),
                            dir = download_OSPapa_cache,
                            max_gap_hours = 72)

Download precomputed air-sea fluxes for Ocean Station Papa from the PMEL ERDDAP
`ocs_papa_flux` dataset (computed with COARE 3.0b) and return them as a
`NamedTuple` of `FieldTimeSeries` on a `Flat, Flat, Flat` grid at `architecture`.

The returned tuple contains:
- `Qnet`: net heat flux (W/m², positive into ocean)
- `Qlat`: latent heat flux (W/m², positive upward = ocean cooling)
- `Qsen`: sensible heat flux (W/m², positive upward = ocean cooling)
- `SWnet`: net shortwave radiation (W/m², positive downward)
- `LWnet`: net longwave radiation (W/m², positive upward = ocean cooling)
- `τx`, `τy`: zonal and meridional wind stress (N/m²)
- `evap`, `rain`, `EMP`: evaporation, precipitation, E-P (mm/hr)
- `Tsk`: skin temperature (°C)

Keyword Arguments
=================
- `start_date`: start of the time range
- `end_date`: end of the time range
- `dir`: directory for cached data files
- `max_gap_hours`: maximum gap size (in hours) to fill by linear interpolation
  (default: 72)
"""
function OSPapaPrescribedFluxes(architecture = CPU(), FT = Float64;
                                start_date = first_date(OSPapaFluxHourly(), :net_heat_flux),
                                end_date   = last_date(OSPapaFluxHourly(), :net_heat_flux),
                                dir = download_OSPapa_cache,
                                max_gap_hours = 72)

    mdkw = (; dataset = OSPapaFluxHourly(), start_date, end_date, dir)

    surface_grid = RectilinearGrid(architecture, FT; size=(), topology=(Flat, Flat, Flat))

    function flux_fts(name)
        md = Metadata(name; mdkw...)
        download_dataset(md)
        fts = FieldTimeSeries(md, surface_grid;
                              time_indices_in_memory = length(md),
                              time_indexing = Cyclical())
        fill_gaps!(fts; max_gap = max_gap_hours)
        return fts
    end

    return (; Qnet  = flux_fts(:net_heat_flux),
              Qlat  = flux_fts(:latent_heat_flux),
              Qsen  = flux_fts(:sensible_heat_flux),
              SWnet = flux_fts(:net_shortwave_radiation),
              LWnet = flux_fts(:net_longwave_radiation),
              τx    = flux_fts(:zonal_stress),
              τy    = flux_fts(:meridional_stress),
              evap  = flux_fts(:evaporation),
              rain  = flux_fts(:rain),
              EMP   = flux_fts(:evaporation_minus_precipitation),
              Tsk   = flux_fts(:skin_temperature))
end

"""
    no_correction(i, j, grid, clock, model_fields, p)

Default correction function that returns zero. Used as the default value for
the correction keyword arguments in [`os_papa_prescribed_flux_boundary_conditions`](@ref).
"""
no_correction(i, j, grid, clock, model_fields, p) = zero(grid)

"""
    os_papa_prescribed_flux_boundary_conditions(fluxes; ρ₀=1020.0, cₚ=3991.0, ...)

Create Oceananigans `FluxBoundaryCondition`s for u, v, T, S from prescribed
OS Papa flux data. Returns a `NamedTuple` of `FieldBoundaryConditions` that
can be passed directly to `ocean_simulation` or `HydrostaticFreeSurfaceModel`
via the `boundary_conditions` keyword argument.

Uses discrete-form boundary condition functions that index flux time series at
each grid point during tendency computation — no callback needed. GPU-safe: the
flux `FieldTimeSeries` are used on the architecture they already carry.

Arguments
=========
- `fluxes`: a `NamedTuple` returned by [`OSPapaPrescribedFluxes`](@ref)

Keyword Arguments
=================
- `arch`: target architecture for the flux `FieldTimeSeries`. If provided, each
  flux FTS is moved to `arch` via `on_architecture`. Defaults to `nothing`,
  which keeps them on whatever architecture they were built on.
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
fluxes = OSPapaPrescribedFluxes(GPU(); start_date, end_date)
bcs = os_papa_prescribed_flux_boundary_conditions(fluxes)
ocean = ocean_simulation(grid; Δt=10minutes, boundary_conditions=bcs)

# With a uniform heat flux correction of +5 W/m² to close the heat budget:
heat_correction = (i, j, grid, clock, model_fields, p) -> 5.0 / (p.ρ₀ * p.cₚ)
bcs = os_papa_prescribed_flux_boundary_conditions(fluxes; T_correction=heat_correction)
```
"""
function os_papa_prescribed_flux_boundary_conditions(fluxes;
                                                     arch=nothing,
                                                     ρ₀=1020.0, cₚ=3991.0,
                                                     u_correction=no_correction,
                                                     v_correction=no_correction,
                                                     T_correction=no_correction,
                                                     S_correction=no_correction)

    if !isnothing(arch)
        fluxes = map(fts -> on_architecture(arch, fts), fluxes)
    end

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

    params_τx = (; τx=fluxes.τx, ρ₀)
    params_τy = (; τy=fluxes.τy, ρ₀)
    params_T  = (; Qnet=fluxes.Qnet, ρ₀, cₚ)
    params_S  = (; EMP=fluxes.EMP, ρ₀)

    u_top = FluxBoundaryCondition(τx_bc, discrete_form=true, parameters=params_τx)
    v_top = FluxBoundaryCondition(τy_bc, discrete_form=true, parameters=params_τy)
    T_top = FluxBoundaryCondition(Jᵀ_bc, discrete_form=true, parameters=params_T)
    S_top = FluxBoundaryCondition(Jˢ_bc, discrete_form=true, parameters=params_S)

    return (; u = FieldBoundaryConditions(top=u_top),
              v = FieldBoundaryConditions(top=v_top),
              T = FieldBoundaryConditions(top=T_top),
              S = FieldBoundaryConditions(top=S_top))
end
