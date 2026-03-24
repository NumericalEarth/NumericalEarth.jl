# # Single-column surface fluxes at Ocean Station Papa
#
# In this example, we build a single-column coupled atmosphere--ocean
# system at Ocean Station Papa (145°W, 50°N) using ERA5 reanalysis
# for the atmosphere and GLORYS reanalysis for the ocean.
# NumericalEarth's bulk formulae then compute the turbulent surface
# fluxes — sensible heat, latent heat, and momentum.
#
# The example demonstrates:
#
# - `BoundingBox` and `Column` regions in `Metadata`
# - Downloading ERA5 and GLORYS data at a single point
# - Building a `PrescribedAtmosphere` from ERA5 fields
# - Building a `PrescribedOcean` from GLORYS fields
# - Computing fluxes with `AtmosphereOceanModel`
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, NumericalEarth, CDSAPI, CopernicusMarine, CairoMakie"
# ```
#
# You need CDS API credentials for ERA5
# (see <https://cds.climate.copernicus.eu/how-to-api>)
# and Copernicus Marine credentials for GLORYS
# (see <https://data.marine.copernicus.eu/register>).

using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum, BoundingBox, Column
using NumericalEarth.DataWrangling.ERA5: ERA5Hourly
using NumericalEarth.DataWrangling.GLORYS: GLORYSMonthly

using Oceananigans
using Oceananigans.Units

using CDSAPI
using CopernicusMarine
using CairoMakie
using Dates
using Printf

# ## Location and date
#
# Ocean Station Papa sits in the northeast Pacific at about 145°W, 50°N —
# a site of strong wintertime heat loss to the atmosphere.

λ★, φ★ = -145.0, 50.0  # Ocean Station Papa

date = DateTime(2020, 1, 15, 12)  # mid-January — strong fluxes expected

# ## Download ERA5 atmospheric state
#
# We use a `BoundingBox` to download a small patch of ERA5 data around
# the station, then extract point values for the `PrescribedAtmosphere`.

era5_region = BoundingBox(longitude = (λ★ - 1, λ★ + 1),
                          latitude  = (φ★ - 1, φ★ + 1))

era5 = ERA5Hourly()

u_meta   = Metadatum(:eastward_velocity;               dataset = era5, region = era5_region, date)
v_meta   = Metadatum(:northward_velocity;              dataset = era5, region = era5_region, date)
T_meta   = Metadatum(:temperature;                     dataset = era5, region = era5_region, date)
q_meta   = Metadatum(:specific_humidity;               dataset = era5, region = era5_region, date)
p_meta   = Metadatum(:surface_pressure;                dataset = era5, region = era5_region, date)
Qsw_meta = Metadatum(:downwelling_shortwave_radiation; dataset = era5, region = era5_region, date)
Qlw_meta = Metadatum(:downwelling_longwave_radiation;  dataset = era5, region = era5_region, date)

for meta in (u_meta, v_meta, T_meta, q_meta, p_meta, Qsw_meta, Qlw_meta)
    download_dataset(meta)
end

# Load the fields and find the grid cell nearest to Ocean Station Papa.

u_field   = Field(u_meta)
v_field   = Field(v_meta)
T_field   = Field(T_meta)
q_field   = Field(q_meta)
p_field   = Field(p_meta)
Qsw_field = Field(Qsw_meta)
Qlw_field = Field(Qlw_meta)

grid_era5 = u_field.grid
λ_arr = λnodes(grid_era5, Center(); with_halos = false)
φ_arr = φnodes(grid_era5, Center(); with_halos = false)
i★ = argmin(abs.(λ_arr .- λ★))
j★ = argmin(abs.(φ_arr .- φ★))

u₁₀ = u_field[i★, j★, 1]
v₁₀ = v_field[i★, j★, 1]
Tₐ  = T_field[i★, j★, 1]
qₐ  = q_field[i★, j★, 1]
pₐ  = p_field[i★, j★, 1]
Qsw = Qsw_field[i★, j★, 1]
Qlw = Qlw_field[i★, j★, 1]

@info "ERA5 atmosphere at Ocean Station Papa:" u₁₀ v₁₀ Tₐ qₐ pₐ

# ## Build a PrescribedAtmosphere
#
# A single-point, constant-in-time atmosphere assembled from the ERA5 state.

atmos_grid  = RectilinearGrid(size = (), topology = (Flat, Flat, Flat))
atmos_times = [0.0, 1days]
atmosphere  = PrescribedAtmosphere(atmos_grid, atmos_times)

parent(atmosphere.velocities.u) .= u₁₀
parent(atmosphere.velocities.v) .= v₁₀
parent(atmosphere.tracers.T)    .= Tₐ
parent(atmosphere.tracers.q)    .= qₐ
parent(atmosphere.pressure)     .= pₐ
parent(atmosphere.downwelling_radiation.shortwave) .= Qsw
parent(atmosphere.downwelling_radiation.longwave)  .= Qlw

# ## Download GLORYS ocean state
#
# GLORYS supports spatial subsetting on download. We use a `Column`
# region to download only the water column at Ocean Station Papa,
# and build a `Field` from which we initialise the `PrescribedOcean`.

glorys = GLORYSMonthly()
glorys_col = Column(λ★, φ★)

sst_meta = Metadatum(:temperature; dataset = glorys, region = glorys_col, date)
sss_meta = Metadatum(:salinity;    dataset = glorys, region = glorys_col, date)

sst_field = Field(sst_meta)
sss_field = Field(sss_meta)

# Extract the surface values (top of the water column).
Nz = size(sst_field, 3)
SST = sst_field[1, 1, Nz]
SSS = sss_field[1, 1, Nz]

@info "GLORYS ocean at Ocean Station Papa:" SST SSS

# ## Build a PrescribedOcean
#
# The `PrescribedOcean` only needs the surface state — it is analogous
# to `PrescribedAtmosphere` for the ocean side. We use a 0D grid with
# `(Flat, Flat, Flat)` topology.

ocean_grid = RectilinearGrid(size = (), topology = (Flat, Flat, Flat))
ocean = PrescribedOcean(ocean_grid, NamedTuple())

set!(ocean.tracers.T, SST)
set!(ocean.tracers.S, SSS)

# ## Compute fluxes
#
# Constructing an `AtmosphereOceanModel` computes the bulk formula
# surface fluxes immediately.

radiation = Radiation()
coupled_model = AtmosphereOceanModel(atmosphere, ocean; radiation)

fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

Qsens = first(interior(fluxes.sensible_heat))
Qlat  = first(interior(fluxes.latent_heat))
τx    = first(interior(fluxes.x_momentum))
τy    = first(interior(fluxes.y_momentum))

wind_speed = sqrt(u₁₀^2 + v₁₀^2)
@info "Bulk formula surface fluxes:" Qsens Qlat τx τy wind_speed

# ## Visualize

fig = Figure(size = (900, 500))

ax1 = Axis(fig[1, 1];
           title = "Heat fluxes at Ocean Station Papa\n$(Dates.format(date, "yyyy-mm-dd HH:MM")) UTC",
           ylabel = "W m⁻²",
           xticks = (1:2, ["Sensible", "Latent"]))

barplot!(ax1, [1, 2], [Qsens, Qlat];
         color = [Qsens > 0 ? :indianred : :steelblue,
                  Qlat  > 0 ? :indianred : :steelblue],
         strokewidth = 1, strokecolor = :black)

hlines!(ax1, [0]; color = :black, linewidth = 0.5)

text!(ax1, 1, Qsens; text = @sprintf("%.1f W/m²", Qsens),
      align = (:center, Qsens > 0 ? :bottom : :top), fontsize = 14)
text!(ax1, 2, Qlat;  text = @sprintf("%.1f W/m²", Qlat),
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
      @sprintf("ERA5: T₂ₘ = %.1f K, q = %.4f kg/kg, |u₁₀| = %.1f m/s  |  GLORYS: SST = %.1f°C, SSS = %.1f g/kg",
               Tₐ, qₐ, wind_speed, SST, SSS);
      fontsize = 12)

current_figure()
