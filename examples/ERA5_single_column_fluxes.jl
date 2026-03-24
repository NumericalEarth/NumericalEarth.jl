# # Single-column surface fluxes from ERA5 reanalysis
#
# In this example, we download ERA5 atmospheric state data at a single
# point, build a `PrescribedAtmosphere` from ERA5 winds, temperature,
# humidity, and radiation, and pair it with a `PrescribedOcean` whose
# SST comes from ERA5. NumericalEarth's bulk formulae then compute the
# turbulent surface fluxes — sensible heat, latent heat, and wind stress.
#
# This demonstrates the full pipeline:
# `Metadata` → `BoundingBox` / `Column` → `Field` →
# `PrescribedAtmosphere` + `PrescribedOcean` → `OceanOnlyModel` → fluxes.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, NumericalEarth, CDSAPI, CairoMakie"
# ```
#
# You also need CDS API credentials in `~/.cdsapirc`.
# See <https://cds.climate.copernicus.eu/how-to-api> for setup instructions.

using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum, BoundingBox
using NumericalEarth.DataWrangling.ERA5: ERA5Hourly

using Oceananigans
using Oceananigans.Units

using CDSAPI
using CairoMakie
using Dates
using Printf

# ## Choose a location and time window
#
# We pick a point in the North Atlantic where air--sea heat exchange
# is strong, and download a small ERA5 patch around it.

λ★, φ★ = -30.0, 45.0   # 30°W, 45°N — North Atlantic

region = BoundingBox(longitude = (λ★ - 1, λ★ + 1),
                     latitude  = (φ★ - 1, φ★ + 1))

date = DateTime(2020, 1, 15, 12)  # a winter day — strong heat loss expected
dataset = ERA5Hourly()

# ## Download ERA5 atmospheric state and SST
#
# We download everything needed for a `PrescribedAtmosphere` plus the
# ERA5 sea-surface temperature.

u_meta   = Metadatum(:eastward_velocity;               dataset, region, date)
v_meta   = Metadatum(:northward_velocity;              dataset, region, date)
T_meta   = Metadatum(:temperature;                     dataset, region, date)
q_meta   = Metadatum(:specific_humidity;               dataset, region, date)
p_meta   = Metadatum(:surface_pressure;                dataset, region, date)
Qsw_meta = Metadatum(:downwelling_shortwave_radiation; dataset, region, date)
Qlw_meta = Metadatum(:downwelling_longwave_radiation;  dataset, region, date)
sst_meta = Metadatum(:sea_surface_temperature;         dataset, region, date)

for meta in (u_meta, v_meta, T_meta, q_meta, p_meta, Qsw_meta, Qlw_meta, sst_meta)
    download_dataset(meta)
end

# ## Load fields and extract values at the target point

u_field   = Field(u_meta)
v_field   = Field(v_meta)
T_field   = Field(T_meta)
q_field   = Field(q_meta)
p_field   = Field(p_meta)
Qsw_field = Field(Qsw_meta)
Qlw_field = Field(Qlw_meta)
sst_field = Field(sst_meta)

# Find the nearest grid cell to our point.

grid = u_field.grid
λ_arr = λnodes(grid, Center(); with_halos=false)
φ_arr = φnodes(grid, Center(); with_halos=false)
i★ = argmin(abs.(λ_arr .- λ★))
j★ = argmin(abs.(φ_arr .- φ★))

u₁₀ = u_field[i★, j★, 1]
v₁₀ = v_field[i★, j★, 1]
Tₐ  = T_field[i★, j★, 1]       # 2-m temperature [K]
qₐ  = q_field[i★, j★, 1]       # specific humidity [kg/kg]
pₐ  = p_field[i★, j★, 1]       # surface pressure [Pa]
Qsw = Qsw_field[i★, j★, 1]     # shortwave ↓ [J/m²]
Qlw = Qlw_field[i★, j★, 1]     # longwave ↓ [J/m²]
SST = sst_field[i★, j★, 1]     # SST [K]

@info "ERA5 state at ($(λ★)°, $(φ★)°) on $(date):" u₁₀ v₁₀ Tₐ qₐ pₐ Qsw Qlw SST

# ## Build a PrescribedAtmosphere
#
# We construct a single-point `PrescribedAtmosphere` with constant-in-time
# ERA5 state.

atmos_grid  = RectilinearGrid(size=(), topology=(Flat, Flat, Flat))
atmos_times = [0.0, 1days]
atmosphere  = PrescribedAtmosphere(atmos_grid, atmos_times)

parent(atmosphere.velocities.u) .= u₁₀
parent(atmosphere.velocities.v) .= v₁₀
parent(atmosphere.tracers.T)    .= Tₐ
parent(atmosphere.tracers.q)    .= qₐ
parent(atmosphere.pressure)     .= pₐ
parent(atmosphere.downwelling_radiation.shortwave) .= Qsw
parent(atmosphere.downwelling_radiation.longwave)  .= Qlw

# ## Build a PrescribedOcean from ERA5 SST
#
# Rather than running a dynamical ocean model, we use `PrescribedOcean`
# to hold the ERA5 SST as the ocean surface state.

ocean_grid = RectilinearGrid(size = 1,
                              x = λ★, y = φ★,
                              z = (-10, 0),
                              topology = (Flat, Flat, Bounded))

ocean = PrescribedOcean(ocean_grid, NamedTuple())

# Set SST from ERA5 (converting from Kelvin to Celsius):
SST_celsius = SST - 273.15
set!(ocean.tracers.T, SST_celsius)
set!(ocean.tracers.S, 35.0)

# ## Compute fluxes
#
# Constructing an `AtmosphereOceanModel` triggers the bulk formula flux
# computation.  There is no sea ice in this setup — just a prescribed
# atmosphere and a prescribed ocean.

radiation = Radiation()
coupled_model = AtmosphereOceanModel(atmosphere, ocean; radiation)

# ## Extract computed fluxes

fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

Qsens = first(interior(fluxes.sensible_heat))
Qlat  = first(interior(fluxes.latent_heat))
τx    = first(interior(fluxes.x_momentum))
τy    = first(interior(fluxes.y_momentum))
Jv    = first(interior(fluxes.water_vapor))

wind_speed = sqrt(u₁₀^2 + v₁₀^2)
ΔT = Tₐ - SST

@info "Computed bulk surface fluxes:" Qsens Qlat τx τy Jv
@info "Context:" wind_speed ΔT SST_celsius

# ## Visualize

fig = Figure(size = (800, 500))

ax = Axis(fig[1, 1];
          title = "Bulk formula surface fluxes at ($(λ★)°, $(φ★)°)\n$(Dates.format(date, "yyyy-mm-dd HH:MM")) UTC",
          ylabel = "W m⁻²",
          xticks = (1:2, ["Sensible heat", "Latent heat"]))

barplot!(ax, [1, 2], [Qsens, Qlat];
         color = [Qsens > 0 ? :indianred : :steelblue,
                  Qlat  > 0 ? :indianred : :steelblue],
         strokewidth = 1, strokecolor = :black)

hlines!(ax, [0]; color = :black, linewidth = 0.5)

text!(ax, 1, Qsens; text = @sprintf("%.1f W/m²", Qsens),
      align = (:center, Qsens > 0 ? :bottom : :top), fontsize = 14)
text!(ax, 2, Qlat;  text = @sprintf("%.1f W/m²", Qlat),
      align = (:center, Qlat > 0 ? :bottom : :top), fontsize = 14)

ax2 = Axis(fig[1, 2];
           title = "Wind stress",
           ylabel = "N m⁻²",
           xticks = (1:2, ["Zonal (τˣ)", "Meridional (τʸ)"]))

barplot!(ax2, [1, 2], [τx, τy];
         color = [:steelblue, :steelblue],
         strokewidth = 1, strokecolor = :black)

hlines!(ax2, [0]; color = :black, linewidth = 0.5)

text!(ax2, 1, τx; text = @sprintf("%.4f", τx),
      align = (:center, τx > 0 ? :bottom : :top), fontsize = 14)
text!(ax2, 2, τy; text = @sprintf("%.4f", τy),
      align = (:center, τy > 0 ? :bottom : :top), fontsize = 14)

Label(fig[2, :],
      @sprintf("ERA5: T₂ₘ = %.1f K, SST = %.1f°C, |u₁₀| = %.1f m/s, q = %.4f kg/kg",
               Tₐ, SST_celsius, wind_speed, qₐ);
      fontsize = 12)

current_figure()
