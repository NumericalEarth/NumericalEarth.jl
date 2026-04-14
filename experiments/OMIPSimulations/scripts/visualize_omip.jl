#!/usr/bin/env julia
# visualize_omip.jl -- Generate all OMIP diagnostic figures as PNGs.
#
# Usage:
#     julia --project=.. visualize_omip.jl [output_dir]
#
# Edit the `cases`, `start_time`, `stop_time` below before running.

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

cases = [
    (run_dir = "halfdegree_run", prefix = "halfdegree", label = "Half-degree"),
    (run_dir = "orca_run",       prefix = "orca",       label = "ORCA"),
]

start_time = 0
stop_time  = Inf

output_dir = length(ARGS) >= 1 ? ARGS[1] : "figures"

# ══════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════

using CairoMakie
using Statistics
using Dates
using Downloads
using DelimitedFiles
using Oceananigans
using Oceananigans.Grids: znodes, φnodes, φnode
using Oceananigans.Fields: interpolate!
using ConservativeRegridding
using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum
using NumericalEarth.DataWrangling.WOA: WOAAnnual

mkpath(output_dir)
@info "Figures will be saved to: $output_dir"

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

function find_first_file(run_dir, prefix, group)
    tag = "$(prefix)_$(group)"
    candidates = filter(f -> startswith(f, tag) && endswith(f, ".jld2") &&
                             !contains(f, "checkpoint"), readdir(run_dir))
    isempty(candidates) && error("No $group files for prefix '$prefix' in $run_dir")
    filename = first(sort(candidates))
    basename_no_part = replace(filename, r"_part\d+" => "")
    return joinpath(run_dir, basename_no_part)
end

function in_window(fts; start_time = 0, stop_time = Inf)
    return findall(t -> start_time <= t <= stop_time, fts.times)
end

function compute_time_mean(fts; start_time = 0, stop_time = Inf)
    idx = in_window(fts; start_time, stop_time)
    isempty(idx) && error("No snapshots in [$start_time, $stop_time]")
    avg = zeros(size(Array(interior(fts[first(idx)]))))
    for n in idx
        avg .+= Array(interior(fts[n]))
    end
    return avg ./ length(idx)
end

function compute_monthly_mean(fts, target_months;
                              start_time = 0, stop_time = Inf,
                              reference_date = DateTime(1958, 1, 1))
    dates = [reference_date + Second(round(Int, t)) for t in fts.times]
    idx   = findall(i -> month(dates[i]) in target_months &&
                         start_time <= fts.times[i] <= stop_time,
                    eachindex(dates))
    isempty(idx) && return nothing
    avg = zeros(size(Array(interior(fts[first(idx)]))))
    for n in idx
        avg .+= Array(interior(fts[n]))
    end
    return avg ./ length(idx)
end

function build_land_mask(grid)
    if grid isa ImmersedBoundaryGrid
        bh = Array(interior(grid.immersed_boundary.bottom_height, :, :, 1))
        return bh .>= 0
    else
        return falses(size(grid, 1), size(grid, 2))
    end
end

function build_ocean_mask_3d(grid)
    Nx, Ny, Nz = size(grid)
    mask = ones(Nx, Ny, Nz)
    if grid isa ImmersedBoundaryGrid
        bh = Array(interior(grid.immersed_boundary.bottom_height, :, :, 1))
        zc = znodes(grid, Center())
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            zc[k] < bh[i, j] && (mask[i, j, k] = 0.0)
        end
    end
    return mask
end

mask_land!(f, land) = (f[land] .= NaN; f)

function panel!(fig, pos, data;
                title="", colormap=:thermal,
                colorrange=nothing, label="",
                nan_color=:lightgray)
    ax = Axis(fig[pos...]; title)
    kw = isnothing(colorrange) ? (;) : (; colorrange)
    hm = heatmap!(ax, data; colormap, nan_color, kw...)
    Colorbar(fig[pos[1], pos[2]+1], hm; label)
    return ax
end

case_colors = [:firebrick, :royalblue, :seagreen, :darkorange]

savefig(fig, name) = save(joinpath(output_dir, name), fig)

# ══════════════════════════════════════════════════════════════
# Load surface diagnostics
# ══════════════════════════════════════════════════════════════

function load_surface_case(run_dir, prefix; start_time = 0, stop_time = Inf)
    surface_file = find_first_file(run_dir, prefix, "surface")
    @info "  surface: $surface_file"

    tos     = FieldTimeSeries(surface_file, "tos";    backend = OnDisk())
    sos     = FieldTimeSeries(surface_file, "sos";    backend = OnDisk())
    zos     = FieldTimeSeries(surface_file, "zos";    backend = OnDisk())
    mld_fts = FieldTimeSeries(surface_file, "mlotst"; backend = OnDisk())
    hfds    = FieldTimeSeries(surface_file, "hfds";   backend = OnDisk())
    wfo     = FieldTimeSeries(surface_file, "wfo";    backend = OnDisk())
    sic     = FieldTimeSeries(surface_file, "siconc"; backend = OnDisk())
    zossq   = FieldTimeSeries(surface_file, "zossq";  backend = OnDisk())

    grid = tos.grid
    Nx, Ny, Nz = size(grid)
    land = build_land_mask(grid)

    @info "  averaging window: [$(start_time / (365.25*86400)), $(stop_time / (365.25*86400))] years"

    SST = dropdims(compute_time_mean(tos;  start_time, stop_time);  dims=3)
    SSS = dropdims(compute_time_mean(sos;  start_time, stop_time);  dims=3)
    SSH = dropdims(compute_time_mean(zos;  start_time, stop_time);  dims=3)
    HF  = dropdims(compute_time_mean(hfds; start_time, stop_time);  dims=3)
    FW  = dropdims(compute_time_mean(wfo;  start_time, stop_time);  dims=3)
    SIC_mean = dropdims(compute_time_mean(sic; start_time, stop_time); dims=3)

    SSH_sq  = dropdims(compute_time_mean(zossq; start_time, stop_time); dims=3)
    SSH_var = SSH_sq .- SSH .^ 2

    MLD_monthly = [compute_monthly_mean(mld_fts, [m]; start_time, stop_time) for m in 1:12]
    avail = findall(!isnothing, MLD_monthly)
    MLD_stack = cat([dropdims(MLD_monthly[m]; dims=3) for m in avail]...; dims=3)
    MLD_min = dropdims(minimum(MLD_stack; dims=3); dims=3)
    MLD_max = dropdims(maximum(MLD_stack; dims=3); dims=3)

    SIC_mar = compute_monthly_mean(sic, [3]; start_time, stop_time)
    SIC_sep = compute_monthly_mean(sic, [9]; start_time, stop_time)
    SIC_mar = isnothing(SIC_mar) ? nothing : dropdims(SIC_mar; dims=3)
    SIC_sep = isnothing(SIC_sep) ? nothing : dropdims(SIC_sep; dims=3)

    T_woa = Field(Metadatum(:temperature; dataset = WOAAnnual()), CPU())
    S_woa = Field(Metadatum(:salinity;    dataset = WOAAnnual()), CPU())
    T_interp = CenterField(grid); interpolate!(T_interp, T_woa)
    S_interp = CenterField(grid); interpolate!(S_interp, S_woa)
    T_woa_on_grid = Array(interior(T_interp))
    S_woa_on_grid = Array(interior(S_interp))
    δSST = SST .- T_woa_on_grid[:, :, Nz]
    δSSS = SSS .- S_woa_on_grid[:, :, Nz]

    for f in (SST, SSS, SSH, HF, FW, SIC_mean, SSH_var, MLD_min, MLD_max, δSST, δSSS)
        mask_land!(f, land)
    end
    !isnothing(SIC_mar) && mask_land!(SIC_mar, land)
    !isnothing(SIC_sep) && mask_land!(SIC_sep, land)

    return (; grid, Nx, Ny, Nz, land, surface_file,
              SST, SSS, SSH, HF, FW, SIC_mean, SSH_var,
              MLD_min, MLD_max, SIC_mar, SIC_sep,
              δSST, δSSS, T_woa_on_grid, S_woa_on_grid)
end

D = Dict{String, Any}()
labels = [c.label for c in cases]
for c in cases
    @info "Loading surface: $(c.label)..."
    D[c.label] = load_surface_case(c.run_dir, c.prefix; start_time, stop_time)
end

# ══════════════════════════════════════════════════════════════
# Figures 1-7: Surface diagnostics
# ══════════════════════════════════════════════════════════════

# Figure 1: SST bias
@info "Figure 1: SST bias"
fig = Figure(size = (800 * length(labels), 500), fontsize = 14)
for (i, lab) in enumerate(labels)
    panel!(fig, [1, 2i-1], D[lab].δSST;
           title = "$lab: SST - WOA", colormap = :balance,
           colorrange = (-5, 5), label = "deg C")
end
savefig(fig, "fig01_sst_bias.png")

# Figure 2: SSS bias
@info "Figure 2: SSS bias"
fig = Figure(size = (800 * length(labels), 500), fontsize = 14)
for (i, lab) in enumerate(labels)
    panel!(fig, [1, 2i-1], D[lab].δSSS;
           title = "$lab: SSS - WOA", colormap = :balance,
           colorrange = (-3, 3), label = "PSU")
end
savefig(fig, "fig02_sss_bias.png")

# Figure 3: SSH
@info "Figure 3: SSH"
fig = Figure(size = (800 * length(labels), 500), fontsize = 14)
for (i, lab) in enumerate(labels)
    panel!(fig, [1, 2i-1], D[lab].SSH;
           title = "$lab: Time-mean SSH", colormap = :balance,
           colorrange = (-2, 2), label = "m")
end
savefig(fig, "fig03_ssh.png")

# Figure 4: MLD min/max
@info "Figure 4: MLD"
fig = Figure(size = (800 * length(labels), 900), fontsize = 14)
for (i, lab) in enumerate(labels)
    panel!(fig, [1, 2i-1], D[lab].MLD_min;
           title = "$lab: Min MLD (summer)",
           colormap = Reverse(:deep), colorrange = (0, 150), label = "m")
    panel!(fig, [2, 2i-1], D[lab].MLD_max;
           title = "$lab: Max MLD (winter)",
           colormap = Reverse(:deep), colorrange = (10, 3000), label = "m")
end
savefig(fig, "fig04_mld.png")

# Figure 5: Sea-ice concentration
@info "Figure 5: Sea-ice concentration"
fig = Figure(size = (800 * length(labels), 900), fontsize = 14)
for (i, lab) in enumerate(labels)
    d = D[lab]
    !isnothing(d.SIC_mar) && panel!(fig, [1, 2i-1], d.SIC_mar;
        title = "$lab: Sea-ice conc. March",
        colormap = :ice, colorrange = (0, 1), label = "fraction")
    !isnothing(d.SIC_sep) && panel!(fig, [2, 2i-1], d.SIC_sep;
        title = "$lab: Sea-ice conc. September",
        colormap = :ice, colorrange = (0, 1), label = "fraction")
end
savefig(fig, "fig05_seaice_conc.png")

# Figure 6: Surface fluxes
@info "Figure 6: Surface fluxes"
fig = Figure(size = (800 * length(labels), 900), fontsize = 14)
for (i, lab) in enumerate(labels)
    panel!(fig, [1, 2i-1], D[lab].HF;
           title = "$lab: Net heat flux", colormap = :balance,
           colorrange = (-200, 200), label = "W/m^2")
    panel!(fig, [2, 2i-1], D[lab].FW;
           title = "$lab: Net freshwater flux", colormap = :balance,
           colorrange = (-1e-4, 1e-4), label = "kg/m^2/s")
end
savefig(fig, "fig06_surface_fluxes.png")

# Figure 7: SSH variance
@info "Figure 7: SSH variance"
fig = Figure(size = (800 * length(labels), 500), fontsize = 14)
for (i, lab) in enumerate(labels)
    panel!(fig, [1, 2i-1], D[lab].SSH_var;
           title = "$lab: SSH variance", colormap = :magma,
           colorrange = (0, 0.05), label = "m²")
end
savefig(fig, "fig07_ssh_variance.png")

# ══════════════════════════════════════════════════════════════
# Sea-ice diagnostics
# ══════════════════════════════════════════════════════════════

arctic_condition(i, j, k, grid, args...)    = φnode(i, j, k, grid, Center(), Center(), Center()) > 0
antarctic_condition(i, j, k, grid, args...) = φnode(i, j, k, grid, Center(), Center(), Center()) < 0

function compute_ice_diagnostics(run_dir, prefix, grid;
                                 start_time = 0, stop_time = Inf,
                                 reference_date = DateTime(1958, 1, 1),
                                 extent_threshold = 0.15)
    surface_file      = find_first_file(run_dir, prefix, "surface")
    thickness_fts     = FieldTimeSeries(surface_file, "sithick"; backend = OnDisk())
    concentration_fts = FieldTimeSeries(surface_file, "siconc";  backend = OnDisk())

    Nt = length(thickness_fts.times)
    arctic_volume      = zeros(Nt)
    antarctic_volume   = zeros(Nt)
    arctic_extent      = zeros(Nt)
    antarctic_extent   = zeros(Nt)
    arctic_area        = zeros(Nt)
    antarctic_area     = zeros(Nt)
    snapshot_dates     = [reference_date + Second(round(Int, t)) for t in thickness_fts.times]

    extent_mask = Field{Center, Center, Nothing}(grid)
    arctic_extent_integral    = Field(Integral(extent_mask; condition = arctic_condition))
    antarctic_extent_integral = Field(Integral(extent_mask; condition = antarctic_condition))

    for n in 1:Nt
        concentration_field = concentration_fts[n]

        ice_volume_field   = thickness_fts[n] * concentration_field
        arctic_vol_int     = Field(Integral(ice_volume_field; condition = arctic_condition))
        antarctic_vol_int  = Field(Integral(ice_volume_field; condition = antarctic_condition))
        compute!(arctic_vol_int);  compute!(antarctic_vol_int)
        arctic_volume[n]    = arctic_vol_int[1, 1, 1]
        antarctic_volume[n] = antarctic_vol_int[1, 1, 1]

        arctic_area_int    = Field(Integral(concentration_field; condition = arctic_condition))
        antarctic_area_int = Field(Integral(concentration_field; condition = antarctic_condition))
        compute!(arctic_area_int);  compute!(antarctic_area_int)
        arctic_area[n]    = arctic_area_int[1, 1, 1]
        antarctic_area[n] = antarctic_area_int[1, 1, 1]

        concentration_data = Array(interior(concentration_field, :, :, 1))
        set!(extent_mask, Float64.(concentration_data .> extent_threshold))
        compute!(arctic_extent_integral);  compute!(antarctic_extent_integral)
        arctic_extent[n]    = arctic_extent_integral[1, 1, 1]
        antarctic_extent[n] = antarctic_extent_integral[1, 1, 1]
    end

    idx = findall(t -> start_time <= t <= stop_time, thickness_fts.times)
    months_used = month.(snapshot_dates[idx])
    monthly(field) = [mean(field[idx[months_used .== m]]) for m in 1:12]

    return (; arctic_volume, antarctic_volume,
              arctic_extent, antarctic_extent,
              arctic_area, antarctic_area, snapshot_dates,
              arctic_volume_monthly    = monthly(arctic_volume),
              antarctic_volume_monthly = monthly(antarctic_volume),
              arctic_extent_monthly    = monthly(arctic_extent),
              antarctic_extent_monthly = monthly(antarctic_extent),
              arctic_area_monthly      = monthly(arctic_area),
              antarctic_area_monthly   = monthly(antarctic_area))
end

ICE = Dict{String, Any}()
for c in cases
    @info "Computing sea-ice diagnostics for $(c.label)..."
    ICE[c.label] = compute_ice_diagnostics(c.run_dir, c.prefix, D[c.label].grid; start_time, stop_time)
end

# ── Download observational climatologies ─────────────────────

piomas_url  = "https://psc.apl.uw.edu/wordpress/wp-content/uploads/schweiger/ice_volume/PIOMAS.monthly.Current.v2.1.csv"
piomas_raw  = readdlm(Downloads.download(piomas_url), ','; skipstart=1)
piomas_volume = Float64.(piomas_raw[:, 2:13])
piomas_volume[piomas_volume .== -1] .= NaN
piomas_monthly = vec(mapslices(x -> mean(filter(!isnan, x)), piomas_volume; dims=1))

function download_nsidc(hemisphere)
    prefix = hemisphere == "north" ? "N" : "S"
    extent_monthly = zeros(12)
    area_monthly   = zeros(12)
    for m in 1:12
        url = "https://noaadata.apps.nsidc.org/NOAA/G02135/$(hemisphere)/monthly/data/$(prefix)_$(lpad(m, 2, '0'))_extent_v4.0.csv"
        raw = readlines(Downloads.download(url))
        extents = Float64[]; areas = Float64[]
        for line in raw
            parts = split(line, ',')
            length(parts) >= 6 || continue
            ext = tryparse(Float64, strip(parts[5]))
            ar  = tryparse(Float64, strip(parts[6]))
            (isnothing(ext) || ext == -9999) && continue
            (isnothing(ar)  || ar  == -9999) && continue
            push!(extents, ext); push!(areas, ar)
        end
        extent_monthly[m] = mean(extents)
        area_monthly[m]   = mean(areas)
    end
    return (; extent_monthly, area_monthly)
end

@info "Downloading NSIDC..."
nsidc_arctic    = download_nsidc("north")
nsidc_antarctic = download_nsidc("south")

# ── Figures 8-12: Sea-ice climatologies and time series ──────

month_names  = ["J","F","M","A","M","J","J","A","S","O","N","D"]
m2_to_Mkm2   = 1e-12
m3_to_1e3km3 = 1e-12

# Figure 8: SIE
@info "Figure 8: SIE"
fig = Figure(size = (1200, 500), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Month", ylabel="SIE (Million km²)", title="Arctic SIE Climatology", xticks=(1:12, month_names))
lines!(ax, 1:12, nsidc_arctic.extent_monthly; color=:black, linewidth=2, label="NSIDC")
for (i, lab) in enumerate(labels)
    lines!(ax, 1:12, ICE[lab].arctic_extent_monthly .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:lb)
ax = Axis(fig[1, 2]; xlabel="Month", ylabel="SIE (Million km²)", title="Antarctic SIE Climatology", xticks=(1:12, month_names))
lines!(ax, 1:12, nsidc_antarctic.extent_monthly; color=:black, linewidth=2, label="NSIDC")
for (i, lab) in enumerate(labels)
    lines!(ax, 1:12, ICE[lab].antarctic_extent_monthly .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
savefig(fig, "fig08_sie.png")

# Figure 9: SIA
@info "Figure 9: SIA"
fig = Figure(size = (1200, 500), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Month", ylabel="SIA (Million km²)", title="Arctic SIA Climatology", xticks=(1:12, month_names))
lines!(ax, 1:12, nsidc_arctic.area_monthly; color=:black, linewidth=2, label="NSIDC")
for (i, lab) in enumerate(labels)
    lines!(ax, 1:12, ICE[lab].arctic_area_monthly .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:lb)
ax = Axis(fig[1, 2]; xlabel="Month", ylabel="SIA (Million km²)", title="Antarctic SIA Climatology", xticks=(1:12, month_names))
lines!(ax, 1:12, nsidc_antarctic.area_monthly; color=:black, linewidth=2, label="NSIDC")
for (i, lab) in enumerate(labels)
    lines!(ax, 1:12, ICE[lab].antarctic_area_monthly .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
savefig(fig, "fig09_sia.png")

# Figure 10: Arctic volume
@info "Figure 10: Arctic volume"
fig = Figure(size = (600, 500), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Month", ylabel="Ice volume (10³ km³)", title="Arctic sea-ice volume", xticks=(1:12, month_names))
lines!(ax, 1:12, piomas_monthly; color=:black, linewidth=2, label="PIOMAS")
for (i, lab) in enumerate(labels)
    lines!(ax, 1:12, ICE[lab].arctic_volume_monthly .* m3_to_1e3km3; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
savefig(fig, "fig10_arctic_volume.png")

# Figure 11: SIA time series
@info "Figure 11: SIA time series"
fig = Figure(size = (1200, 500), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Time (years)", ylabel="SIA (Million km²)", title="Arctic sea-ice area")
for (i, lab) in enumerate(labels)
    time_years = [Dates.value(d - ICE[lab].snapshot_dates[1]) / (365.25 * 86400 * 1000) for d in ICE[lab].snapshot_dates]
    lines!(ax, time_years, ICE[lab].arctic_area .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
ax = Axis(fig[1, 2]; xlabel="Time (years)", ylabel="SIA (Million km²)", title="Antarctic sea-ice area")
for (i, lab) in enumerate(labels)
    time_years = [Dates.value(d - ICE[lab].snapshot_dates[1]) / (365.25 * 86400 * 1000) for d in ICE[lab].snapshot_dates]
    lines!(ax, time_years, ICE[lab].antarctic_area .* m2_to_Mkm2; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
savefig(fig, "fig11_sia_timeseries.png")

# Figure 12: Arctic volume time series
@info "Figure 12: Arctic volume time series"
fig = Figure(size = (600, 500), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Time (years)", ylabel="Ice volume (10³ km³)", title="Arctic sea-ice volume")
for (i, lab) in enumerate(labels)
    time_years = [Dates.value(d - ICE[lab].snapshot_dates[1]) / (365.25 * 86400 * 1000) for d in ICE[lab].snapshot_dates]
    lines!(ax, time_years, ICE[lab].arctic_volume .* m3_to_1e3km3; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rt)
savefig(fig, "fig12_arctic_volume_timeseries.png")

# ══════════════════════════════════════════════════════════════
# Load time series and 3-D fields
# ══════════════════════════════════════════════════════════════

function load_timeseries_case(run_dir, prefix, grid; start_time = 0, stop_time = Inf)
    averages_file = find_first_file(run_dir, prefix, "averages")
    temperature_mean_fts = FieldTimeSeries(averages_file, "tosga"; backend = OnDisk())
    salinity_mean_fts    = FieldTimeSeries(averages_file, "soga";  backend = OnDisk())
    temperature_mean = [Array(interior(temperature_mean_fts[n]))[1] for n in 1:length(temperature_mean_fts.times)]
    salinity_mean    = [Array(interior(salinity_mean_fts[n]))[1]  for n in 1:length(salinity_mean_fts.times)]
    time_in_years    = temperature_mean_fts.times ./ (365.25 * 24 * 3600)

    temperature_profile_fts = FieldTimeSeries(averages_file, "to_h"; backend = OnDisk())
    salinity_profile_fts    = FieldTimeSeries(averages_file, "so_h"; backend = OnDisk())
    temperature_profile = vec(compute_time_mean(temperature_profile_fts; start_time, stop_time))
    salinity_profile    = vec(compute_time_mean(salinity_profile_fts; start_time, stop_time))
    depth = collect(znodes(grid, Center()))

    fields_file = find_first_file(run_dir, prefix, "fields")
    tke_fts     = FieldTimeSeries(fields_file, "tke"; backend = OnDisk())
    ocean_mask  = build_ocean_mask_3d(grid)
    ocean_cells = sum(ocean_mask)
    tke_mean = [sum(Array(interior(tke_fts[n])) .* ocean_mask) / ocean_cells
                for n in 1:length(tke_fts.times)]
    tke_time_in_years = tke_fts.times ./ (365.25 * 24 * 3600)

    return (; temperature_mean, salinity_mean, time_in_years,
              temperature_profile, salinity_profile, depth,
              tke_mean, tke_time_in_years, ocean_mask, fields_file)
end

TS = Dict{String, Any}()
for c in cases
    @info "Loading time series: $(c.label)..."
    TS[c.label] = load_timeseries_case(c.run_dir, c.prefix, D[c.label].grid; start_time, stop_time)
end

# ══════════════════════════════════════════════════════════════
# Figures 13-15: Time series and profiles
# ══════════════════════════════════════════════════════════════

# Figure 13: TKE
@info "Figure 13: TKE"
fig = Figure(size = (900, 450), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Time (years)", ylabel="TKE (m²/s²)", title="Global-mean turbulent kinetic energy")
for (i, lab) in enumerate(labels)
    lines!(ax, TS[lab].tke_time_in_years, TS[lab].tke_mean; color=case_colors[i], label=lab)
end
axislegend(ax; position=:rb)
savefig(fig, "fig13_tke.png")

# Figure 14: T and S drift
@info "Figure 14: T and S drift"
fig = Figure(size = (1200, 450), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Time (years)", ylabel="ΔT (deg C)", title="Global-mean temperature drift")
for (i, lab) in enumerate(labels)
    d = TS[lab]
    lines!(ax, d.time_in_years, d.temperature_mean .- d.temperature_mean[1]; color=case_colors[i], label=lab)
end
axislegend(ax; position=:lb)
ax = Axis(fig[1, 2]; xlabel="Time (years)", ylabel="ΔS (PSU)", title="Global-mean salinity drift")
for (i, lab) in enumerate(labels)
    d = TS[lab]
    lines!(ax, d.time_in_years, d.salinity_mean .- d.salinity_mean[1]; color=case_colors[i], label=lab)
end
axislegend(ax; position=:lb)
savefig(fig, "fig14_drift.png")

# Figure 15: Profiles
@info "Figure 15: Profiles"
fig = Figure(size = (1000, 600), fontsize = 14)
ax = Axis(fig[1, 1]; xlabel="Temperature (deg C)", ylabel="Depth (m)", title="Horizontal-mean temperature")
for (i, lab) in enumerate(labels)
    lines!(ax, TS[lab].temperature_profile, TS[lab].depth; color=case_colors[i], label=lab)
end
ylims!(ax, (-5500, 0)); axislegend(ax; position=:rb)
ax = Axis(fig[1, 2]; xlabel="Salinity (PSU)", ylabel="Depth (m)", title="Horizontal-mean salinity")
for (i, lab) in enumerate(labels)
    lines!(ax, TS[lab].salinity_profile, TS[lab].depth; color=case_colors[i], label=lab)
end
ylims!(ax, (-5500, 0)); axislegend(ax; position=:rb)
savefig(fig, "fig15_profiles.png")

# ══════════════════════════════════════════════════════════════
# Zonal-mean sections
# ══════════════════════════════════════════════════════════════

Nlon, Nlat = 360, 180
latlon_grid = LatitudeLongitudeGrid(CPU();
    size = (Nlon, Nlat, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1))
dst_f = Field{Center, Center, Nothing}(latlon_grid)

function compute_zonal_mean(data_3d, ocean_mask_3d, regridder, Nlon, Nlat)
    Nz = size(data_3d, 3)
    zonal    = fill(NaN, Nlat, Nz)
    dst_data = zeros(Nlon * Nlat)
    dst_mask = zeros(Nlon * Nlat)
    areas    = regridder.dst_areas
    for k in 1:Nz
        ConservativeRegridding.regrid!(dst_data, regridder,
            vec(data_3d[:, :, k] .* ocean_mask_3d[:, :, k]))
        ConservativeRegridding.regrid!(dst_mask, regridder,
            vec(ocean_mask_3d[:, :, k]))
        data_sum = reshape(dst_data .* areas, Nlon, Nlat)
        mask_sum = reshape(dst_mask .* areas, Nlon, Nlat)
        for j in 1:Nlat
            m = sum(@view mask_sum[:, j])
            m > 0 && (zonal[j, k] = sum(@view data_sum[:, j]) / m)
        end
    end
    return zonal
end

ZM = Dict{String, Any}()
for c in cases
    lab  = c.label
    grid = D[lab].grid
    ocean_mask = TS[lab].ocean_mask

    # Build per-case regridder
    @info "Building regridder for $lab (may take a few minutes)..."
    src_f = Field{Center, Center, Nothing}(grid)
    regridder = ConservativeRegridding.Regridder(dst_f, src_f; progress = true)

    @info "Loading 3-D fields for $lab..."
    fields_file = TS[lab].fields_file
    to_fts = FieldTimeSeries(fields_file, "to"; backend = OnDisk())
    so_fts = FieldTimeSeries(fields_file, "so"; backend = OnDisk())
    bo_fts = FieldTimeSeries(fields_file, "bo"; backend = OnDisk())

    temperature_mean = compute_time_mean(to_fts; start_time, stop_time)
    salinity_mean    = compute_time_mean(so_fts; start_time, stop_time)
    buoyancy_mean    = compute_time_mean(bo_fts; start_time, stop_time)
    buoyancy_initial = Array(interior(bo_fts[1]))

    @info "Computing zonal means for $lab..."
    temperature_zonal     = compute_zonal_mean(temperature_mean, ocean_mask, regridder, Nlon, Nlat)
    salinity_zonal        = compute_zonal_mean(salinity_mean,    ocean_mask, regridder, Nlon, Nlat)
    buoyancy_zonal        = compute_zonal_mean(buoyancy_mean,    ocean_mask, regridder, Nlon, Nlat)
    temperature_woa_zonal = compute_zonal_mean(D[lab].T_woa_on_grid, ocean_mask, regridder, Nlon, Nlat)
    salinity_woa_zonal    = compute_zonal_mean(D[lab].S_woa_on_grid, ocean_mask, regridder, Nlon, Nlat)
    buoyancy_init_zonal   = compute_zonal_mean(buoyancy_initial,     ocean_mask, regridder, Nlon, Nlat)

    depth = collect(znodes(grid, Center()))

    ZM[lab] = (; temperature_zonal, salinity_zonal, buoyancy_zonal,
                temperature_woa_zonal, salinity_woa_zonal, buoyancy_init_zonal,
                δtemperature_zonal = temperature_zonal .- temperature_woa_zonal,
                δsalinity_zonal    = salinity_zonal    .- salinity_woa_zonal,
                δbuoyancy_zonal    = buoyancy_zonal    .- buoyancy_init_zonal,
                depth)
end

latitude = collect(φnodes(latlon_grid, Center()))

# ══════════════════════════════════════════════════════════════
# Figures 16-17: Zonal means
# ══════════════════════════════════════════════════════════════

temperature_levels = -2:2:30
salinity_levels    = 33:0.25:37
buoyancy_levels    = range(-0.04, 0.02, length=13)

# Figure 16: Zonal-mean T, S, b
@info "Figure 16: Zonal means"
fig = Figure(size = (600 * length(labels), 900), fontsize = 14)
for (i, lab) in enumerate(labels)
    zm = ZM[lab]
    ax = Axis(fig[1, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal T")
    hm = heatmap!(ax, latitude, zm.depth, zm.temperature_zonal; colormap=:thermal, colorrange=(-2,30), nan_color=:lightgray)
    contour!(ax, latitude, zm.depth, zm.temperature_woa_zonal; levels=temperature_levels, color=:grey, linestyle=:dash, linewidth=0.8)
    contour!(ax, latitude, zm.depth, zm.temperature_zonal; levels=temperature_levels, color=:black, linewidth=0.8)
    Colorbar(fig[1, 2i], hm; label="deg C"); ylims!(ax, (-5500, 0))

    ax = Axis(fig[2, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal S")
    hm = heatmap!(ax, latitude, zm.depth, zm.salinity_zonal; colormap=:haline, colorrange=(33,37), nan_color=:lightgray)
    contour!(ax, latitude, zm.depth, zm.salinity_woa_zonal; levels=salinity_levels, color=:grey, linestyle=:dash, linewidth=0.8)
    contour!(ax, latitude, zm.depth, zm.salinity_zonal; levels=salinity_levels, color=:black, linewidth=0.8)
    Colorbar(fig[2, 2i], hm; label="PSU"); ylims!(ax, (-5500, 0))

    ax = Axis(fig[3, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal b")
    hm = heatmap!(ax, latitude, zm.depth, zm.buoyancy_zonal; colormap=:balance, nan_color=:lightgray)
    contour!(ax, latitude, zm.depth, zm.buoyancy_init_zonal; levels=buoyancy_levels, color=:grey, linestyle=:dash, linewidth=0.8)
    contour!(ax, latitude, zm.depth, zm.buoyancy_zonal; levels=buoyancy_levels, color=:black, linewidth=0.8)
    Colorbar(fig[3, 2i], hm; label="m/s²"); ylims!(ax, (-5500, 0))
end
savefig(fig, "fig16_zonal_mean.png")

# Figure 17: Zonal-mean drift
@info "Figure 17: Zonal-mean drift"
fig = Figure(size = (600 * length(labels), 900), fontsize = 14)
for (i, lab) in enumerate(labels)
    zm = ZM[lab]
    ax = Axis(fig[1, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal T - WOA")
    hm = heatmap!(ax, latitude, zm.depth, zm.δtemperature_zonal; colormap=:balance, colorrange=(-5,5), nan_color=:lightgray)
    Colorbar(fig[1, 2i], hm; label="deg C"); ylims!(ax, (-5500, 0))

    ax = Axis(fig[2, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal S - WOA")
    hm = heatmap!(ax, latitude, zm.depth, zm.δsalinity_zonal; colormap=:balance, colorrange=(-1,1), nan_color=:lightgray)
    Colorbar(fig[2, 2i], hm; label="PSU"); ylims!(ax, (-5500, 0))

    ax = Axis(fig[3, 2i-1]; xlabel="Latitude", ylabel="Depth (m)", title="$lab: Zonal b - b(t=0)")
    hm = heatmap!(ax, latitude, zm.depth, zm.δbuoyancy_zonal; colormap=:balance, nan_color=:lightgray)
    Colorbar(fig[3, 2i], hm; label="m/s²"); ylims!(ax, (-5500, 0))
end
savefig(fig, "fig17_zonal_drift.png")

@info "All 17 figures saved to $output_dir"
