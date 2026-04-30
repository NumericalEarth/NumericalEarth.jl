
using JLD2 

"""
    add_omip_diagnostics!(simulation; kwargs...)

Attach OMIP-protocol output writers to a coupled ocean--sea-ice
simulation built by [`omip_simulation`](@ref).

Creates four output writers:

1. **Surface diagnostics** (`<prefix>_surface.nc`): 2-D fields averaged
   over `surface_averaging_interval` -- SST, SSS, SSH, surface velocities,
   squared fields for variance, mixed-layer depth, wind stress,
   heat/freshwater fluxes, and sea-ice state variables.
2. **3-D field diagnostics** (`<prefix>_fields.nc`): full 3-D temperature,
   salinity, velocity, buoyancy, and (when present) TKE, averaged over
   `field_averaging_interval`.
3. **Averages** (`<prefix>_averages.nc`): global means of T, S, buoyancy
   and horizontal-mean (dims=(1,2)) depth profiles of the same, on the
   same `field_averaging_interval` schedule.
4. **Checkpointer** (`<prefix>_checkpoint`): JLD2 checkpoint of the
   coupled model at `checkpoint_interval`. Use `run!(sim; pickup=true)`
   to restart from the latest checkpoint.

# Keyword arguments

- `surface_averaging_interval`: averaging window for surface output. Default: `5days`.
- `field_averaging_interval`: averaging window for 3-D / averages output. Default: `15days`.
- `checkpoint_interval`: interval between checkpoints. Default: `90days`.
- `output_dir`: directory for all output files. Default: `"."`.
- `filename_prefix`: prefix for output filenames. Default: `"omip"`.
- `file_splitting_interval`: time interval for splitting output files. Default: `360days`.
"""
function add_omip_diagnostics!(simulation;
                               field_mean_interval = 5days,
                               surface_averaging_interval = 5days,
                               field_averaging_interval = 15days,
                               checkpoint_interval = 720days,
                               output_dir = ".",
                               filename_prefix = "omip",
                               file_splitting_interval = 360days)

    model    = simulation.model
    ocean    = model.ocean
    sea_ice  = model.sea_ice
    grid     = ocean.model.grid
    Nz       = size(grid, 3)

    T, S = ocean.model.tracers.T, ocean.model.tracers.S
    u, v, w = ocean.model.velocities
    η = ocean.model.free_surface.displacement

    τx = model.interfaces.net_fluxes.ocean.u
    τy = model.interfaces.net_fluxes.ocean.v
    JT = model.interfaces.net_fluxes.ocean.T
    Js = model.interfaces.net_fluxes.ocean.S
    Qc = model.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
    Qv = model.interfaces.atmosphere_ocean_interface.fluxes.latent_heat

    JTf  = NumericalEarth.Diagnostics.frazil_temperature_flux(model)
    JTn  = NumericalEarth.Diagnostics.net_ocean_temperature_flux(model)
    JTio = NumericalEarth.Diagnostics.sea_ice_ocean_temperature_flux(model)
    JTao = NumericalEarth.Diagnostics.atmosphere_ocean_temperature_flux(model)
    JSn  = NumericalEarth.Diagnostics.net_ocean_salinity_flux(model)
    JSio = NumericalEarth.Diagnostics.sea_ice_ocean_salinity_flux(model)

    hi = sea_ice.model.ice_thickness
    ℵi = sea_ice.model.ice_concentration
    ui, vi = sea_ice.model.velocities

    sitemptop = try
        sea_ice.model.ice_thermodynamics.top_surface_temperature
    catch
        nothing
    end

    mld = MixedLayerDepthField(ocean.model.buoyancy, grid, ocean.model.tracers)

    # Surface diagnostics
    surface_indices = (:, :, Nz)

    tos = view(T, :, :, Nz)
    sos = view(S, :, :, Nz)
    uo_surface = view(u, :, :, Nz)
    vo_surface = view(v, :, :, Nz)

    tossq = tos * tos
    sossq = sos * sos
    zossq = Field(η * η)

    surface_outputs = Dict{Symbol, Any}(
        :tos      => tos,
        :sos      => sos,
        :zos      => η,
        :uos      => uo_surface,
        :vos      => vo_surface,
        :tossq    => tossq,
        :sossq    => sossq,
        :zossq    => zossq,
        :mlotst   => mld,
        :tauuo    => τx,
        :tauvo    => τy,
        :hfds     => JT,
        :wfo      => Js,
        :hfss     => Qc,
        :hfls     => Qv,
        :siconc   => ℵi,
        :sithick  => hi,
        :siu      => ui,
        :siv      => vi,
        :JTf      => JTf,
        :JTn      => JTn,
        :JTio     => JTio,
        :JTao     => JTao,
        :JSn      => JSn,
        :JSio     => JSio
    )

    if !isnothing(sitemptop)
        surface_outputs[:sitemptop] = sitemptop
    end

    hs = sea_ice.model.snow_thickness
    if !isnothing(hs)
        surface_outputs[:sisnthick] = hs
    end

    simulation.output_writers[:surface] = JLD2Writer(ocean.model, surface_outputs;
                                                     schedule = AveragedTimeInterval(surface_averaging_interval),
                                                     dir = output_dir,
                                                     filename = filename_prefix * "_surface",
                                                     file_splitting = TimeInterval(file_splitting_interval),
                                                     overwrite_existing = true,
                                                     jld2_kw = Dict(:compress => ZstdFilter()))

    # 3-D fields (including buoyancy)
    bop = Oceananigans.Models.buoyancy_operation(ocean.model)

    field_outputs = Dict{Symbol, Any}(
        :to => T,
        :so => S,
        :uo => u,
        :vo => v,
        :wo => w,
        :bo => bop,
    )

    if haskey(ocean.model.tracers, :e)
        field_outputs[:tke] = ocean.model.tracers.e
    end

    simulation.output_writers[:fields] = JLD2Writer(ocean.model, field_outputs;
                                                    schedule = AveragedTimeInterval(field_averaging_interval),
                                                    dir = output_dir,
                                                    filename = filename_prefix * "_fields",
                                                    file_splitting = TimeInterval(file_splitting_interval),
                                                    overwrite_existing = true,
                                                    jld2_kw = Dict(:compress => ZstdFilter()))

    # Global means and horizontal-mean depth profiles for T, S, b.
    # `:zosga` (global-mean free-surface displacement) is a Boussinesq mass-conservation check.
    average_outputs = Dict{Symbol, Any}(
        :tosga => Average(T),
        :soga  => Average(S),
        :bga   => Average(bop),
        :zosga => Average(η),
        :to_h  => Average(T,   dims=(1, 2)),
        :so_h  => Average(S,   dims=(1, 2)),
        :bo_h  => Average(bop, dims=(1, 2)),
    )

    simulation.output_writers[:averages] = JLD2Writer(ocean.model, average_outputs;
                                                      schedule = AveragedTimeInterval(field_mean_interval),
                                                      dir = output_dir,
                                                      filename = filename_prefix * "_averages",
                                                      file_splitting = TimeInterval(file_splitting_interval),
                                                      overwrite_existing = true)

    # Checkpointer (drives `run!(sim; pickup=true)`)
    simulation.output_writers[:checkpointer] = Checkpointer(simulation.model;
                                                            schedule = TimeInterval(checkpoint_interval),
                                                            prefix   = joinpath(output_dir, filename_prefix * "_checkpoint"),
                                                            cleanup  = false,
                                                            verbose  = true)

    @info "OMIP diagnostics attached:" *
          " surface ($(length(surface_outputs)) fields, every $(prettytime(surface_averaging_interval)))," *
          " 3-D ($(length(field_outputs)) fields, every $(prettytime(field_averaging_interval)))," *
          " averages ($(length(average_outputs)) fields, every $(prettytime(field_averaging_interval)))," *
          " checkpointer (every $(prettytime(checkpoint_interval)))"

    return nothing
end
