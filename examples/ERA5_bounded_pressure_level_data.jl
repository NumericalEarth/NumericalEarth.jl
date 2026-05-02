# Download 3-D snapshots of the atmospheric state from ERA5, e.g., to be used with FieldTimeSeries
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add NumericalEarth CDSAPI"
# ```
#
# You also need CDS API credentials in `~/.cdsapirc`.
# See <https://cds.climate.copernicus.eu/how-to-api> for setup instructions.
#
# See <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download> for
# more information about pressure-level data

using NumericalEarth
using NumericalEarth.DataWrangling: Metadata, BoundingBox, download_dataset
using NumericalEarth.DataWrangling.ERA5
using CDSAPI
using Dates
using Oceananigans

using Statistics
using CairoMakie

selected_levels = filter(≥(250hPa), ERA5_all_pressure_levels) # select all levels below 250 hPa
dataset = ERA5HourlyPressureLevels(selected_levels)

dates = DateTime(2004, 12, 16):Hour(1):DateTime(2005, 01, 09)  # van Zanten et al 2011 study period

# Rauber et al 2007, Fig 1: focus region
region = BoundingBox(latitude=(17, 18.5), longitude=(-62.5, -61))

# OPTIONAL: download all variables at once (fewer CDS API requests)
variables = [:geopotential, # for calculating zlevels as the mean geopotential height
             :eastward_velocity,
             :northward_velocity,
             :temperature,
             :specific_humidity]
download_dataset(variables, dataset, dates; region)

# ## Create time series

T_meta = Metadata(:temperature;        dataset, dates, region)
q_meta = Metadata(:specific_humidity;  dataset, dates, region)
u_meta = Metadata(:eastward_velocity;  dataset, dates, region)
v_meta = Metadata(:northward_velocity; dataset, dates, region)

T_series = FieldTimeSeries(T_meta; time_indices_in_memory=2)
q_series = FieldTimeSeries(q_meta; time_indices_in_memory=2)
u_series = FieldTimeSeries(u_meta; time_indices_in_memory=2)
v_series = FieldTimeSeries(v_meta; time_indices_in_memory=2)

# ## Vertical profiles (mean ± spread)

# Height coordinate (m) from the grid
z = znodes(T_series[1])
Nz = length(z)
Nt = length(T_series.times)

# Pressure at each level (hPa), ordered bottom-to-top (k=1 ⇒ highest pressure)
p_levels = sort(selected_levels, rev=true) ./ hPa  # Pa → hPa

# Horizontal-mean profile for each snapshot → (Nz, Nt) arrays
function horizontal_mean_profiles(series)
    profiles = zeros(Nz, Nt)
    for n in 1:Nt
        data = interior(series[n], :, :, :)
        profiles[:, n] = mean(data, dims=(1, 2))
    end
    return profiles
end

T_profiles = horizontal_mean_profiles(T_series)
q_profiles = horizontal_mean_profiles(q_series)
u_profiles = horizontal_mean_profiles(u_series)
v_profiles = horizontal_mean_profiles(v_series)

# Convert T → potential temperature: θ = T (p₀/p)^(R/cₚ)
Rₐ_over_cₚ = 0.286
θ_profiles = T_profiles .* (1000 ./ p_levels) .^ Rₐ_over_cₚ

# Convert specific humidity from kg/kg → g/kg
q_profiles .*= 1000

# ## Plot

fig = Figure(size=(900, 500), fontsize=12)

ax_θ = Axis(fig[1, 1], xlabel="θ [K]",       ylabel="Height [m]")
ax_q = Axis(fig[1, 2], xlabel="qᵥ [g kg⁻¹]", ylabel="Height [m]")
ax_u = Axis(fig[1, 3], xlabel="u [m s⁻¹]",   ylabel="Height [m]")
ax_v = Axis(fig[1, 4], xlabel="v [m s⁻¹]",   ylabel="Height [m]")

for (ax, profiles) in [(ax_θ, θ_profiles),
                       (ax_q, q_profiles),
                       (ax_u, u_profiles),
                       (ax_v, v_profiles)]
    μ = vec(mean(profiles, dims=2))
    #lo = vec(minimum(profiles, dims=2))
    #hi = vec(maximum(profiles, dims=2))
    lo = [quantile(r, 0.25) for r in eachrow(profiles)]
    hi = [quantile(r, 0.75) for r in eachrow(profiles)]
    band!(ax, z, lo, hi; direction=:y, color=(:gray, 0.4))
    lines!(ax, μ, z; color=:black, linewidth=2)
end

xlims!(ax_θ, 293, 317)
xlims!(ax_q, 0, 20.4)
xlims!(ax_u, -10, 0)
xlims!(ax_v, -10, 0)

linkyaxes!(ax_θ, ax_q, ax_u, ax_v)
ylims!(ax_θ, 0, 4000)

hideydecorations!(ax_q)
hideydecorations!(ax_u)
hideydecorations!(ax_v)

save("ERA5_pressure_level_profiles.png", fig; px_per_unit=4)
