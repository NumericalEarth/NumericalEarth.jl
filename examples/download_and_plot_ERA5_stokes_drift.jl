using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum, download_dataset, metadata_path
using NumericalEarth.DataWrangling.ERA5: ERA5Hourly, ERA5_netcdf_variable_names
using CDSAPI

using NCDatasets
using CairoMakie
using Dates

# Download Stokes drift and 10m wind for a specific date
dataset = ERA5Hourly()
date = DateTime(2020, 1, 15, 12) # January 15, 2020 at 12:00 UTC

u_stokes_meta = Metadatum(:eastward_stokes_drift;  dataset, date)
v_stokes_meta = Metadatum(:northward_stokes_drift; dataset, date)
u_wind_meta   = Metadatum(:eastward_velocity;      dataset, date)
v_wind_meta   = Metadatum(:northward_velocity;     dataset, date)

download_dataset(u_stokes_meta)
download_dataset(v_stokes_meta)
download_dataset(u_wind_meta)
download_dataset(v_wind_meta)

# Read downloaded data, replacing missing values with NaN
function read_era5_field(meta)
    path = metadata_path(meta)
    ncvar = ERA5_netcdf_variable_names[meta.name]
    ds = NCDataset(path)
    lon = ds["longitude"][:]
    lat = ds["latitude"][:]
    raw = ds[ncvar][:, :, 1] # (lon, lat, time) → first time step
    close(ds)
    data = Float32[ismissing(v) ? NaN32 : Float32(v) for v in raw]
    return lon, lat, data
end

lon_s, lat_s, u_stokes = read_era5_field(u_stokes_meta)
_,     _,     v_stokes = read_era5_field(v_stokes_meta)
lon_w, lat_w, u_wind   = read_era5_field(u_wind_meta)
_,     _,     v_wind   = read_era5_field(v_wind_meta)

stokes_speed = @. sqrt(u_stokes^2 + v_stokes^2)
wind_speed   = @. sqrt(u_wind^2   + v_wind^2)

# Plot: Stokes drift speed and 10m wind speed side by side
fig = Figure(size=(1200, 600))

ax1 = Axis(fig[1, 1]; title="Stokes drift speed (m/s)",
           xlabel="Longitude", ylabel="Latitude")
ax2 = Axis(fig[1, 2]; title="10m wind speed (m/s)",
           xlabel="Longitude", ylabel="Latitude")

hm1 = heatmap!(ax1, lon_s, lat_s, stokes_speed; colormap=:solar, colorrange=(0, 0.4))
hm2 = heatmap!(ax2, lon_w, lat_w, wind_speed;   colormap=:solar, colorrange=(0, 25))

Colorbar(fig[2, 1], hm1; vertical=false, width=Relative(0.8), label="m/s")
Colorbar(fig[2, 2], hm2; vertical=false, width=Relative(0.8), label="m/s")

Label(fig[0, :],
      "ERA5 Stokes Drift and Surface Wind — $(Dates.format(date, "yyyy-mm-dd HH:MM")) UTC";
      fontsize=20)

save("ERA5_stokes_drift_and_wind.png", fig)

display(fig)

#####
##### Globe visualization
#####

function lonlat2xyz(lons::AbstractVector, lats::AbstractVector)
    x = [cosd(lat) * cosd(lon) for lon in lons, lat in lats]
    y = [cosd(lat) * sind(lon) for lon in lons, lat in lats]
    z = [sind(lat)             for lon in lons, lat in lats]
    return (x, y, z)
end

xs, ys, zs = lonlat2xyz(lon_s, lat_s)
xw, yw, zw = lonlat2xyz(lon_w, lat_w)

globe_fig = Figure(size=(1000, 500))

ax_g1 = Axis3(globe_fig[1, 1]; aspect=(1, 1, 1), title="Stokes drift speed")
ax_g2 = Axis3(globe_fig[1, 2]; aspect=(1, 1, 1), title="10m wind speed")

sf1 = surface!(ax_g1, xs, ys, zs; color=stokes_speed, colormap=:solar, colorrange=(0, 0.4), nan_color=:gray80)
sf2 = surface!(ax_g2, xw, yw, zw; color=wind_speed,   colormap=:solar, colorrange=(0, 25),  nan_color=:gray80)

Colorbar(globe_fig[2, 1], sf1; vertical=false, width=Relative(0.5), label="m/s")
Colorbar(globe_fig[2, 2], sf2; vertical=false, width=Relative(0.5), label="m/s")

for ax in (ax_g1, ax_g2)
    hidedecorations!(ax)
    hidespines!(ax)
    ax.viewmode = :fit
end

save("ERA5_stokes_drift_and_wind_globe.png", globe_fig)

display(globe_fig)
