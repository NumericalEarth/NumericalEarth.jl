# Regions, locations, and FieldTimeSeries

The [`Metadata`](@ref) abstraction supports spatial restriction through _regions_,
automatic field location inference, and time-evolving data via `FieldTimeSeries`.
This page covers these features in detail.

## Spatial regions

By default, `Metadata` represents data on the full global domain.
The `region` keyword restricts the spatial extent.

### `BoundingBox`

A [`BoundingBox`](@ref NumericalEarth.DataWrangling.BoundingBox) selects a
longitude--latitude--depth sub-region:

```julia
using NumericalEarth

bbox = BoundingBox(longitude = (200, 220), latitude = (35, 55))
```

When passed to `Metadata`, the native grid shrinks to cover only the bounding box
and, for datasets that support spatial subsetting on download (GLORYS, ERA5),
only the relevant data is fetched:

```julia
T_meta = Metadatum(:temperature; dataset = GLORYSMonthly(), region = bbox)
```

For datasets that always download globally (ECCO, JRA55), the bounding box
restricts the grid that `native_grid` returns.

A `BoundingBox` can also restrict the vertical extent:

```julia
bbox = BoundingBox(longitude = (200, 220),
                   latitude = (35, 55),
                   z = (-500, 0))
```

### `Column`

A [`Column`](@ref NumericalEarth.DataWrangling.Column) represents a single
horizontal point that extends through the water column:

```julia
col = Column(35.1, 50.1)  # (longitude, latitude)
```

When a `Metadata` object has a `Column` region:

- `native_grid` returns a single-column `RectilinearGrid` with `(Flat, Flat, Bounded)` topology.
- `location` reduces horizontal dimensions to `Nothing`, preserving only the vertical location.
- `Field(metadata)` loads data onto an intermediate grid and interpolates to the column point.

```julia
T_meta = Metadatum(:temperature; dataset = ECCO4Monthly(), region = col)

native_grid(T_meta)   # RectilinearGrid at (35.1, 50.1) with Nz vertical levels
location(T_meta)      # (Nothing, Nothing, Center)
T_field = Field(T_meta)  # column Field{Nothing, Nothing, Center}
```

This is particularly useful for single-column ocean simulations. For example, to
initialize an ocean column at Ocean Station Papa:

```julia
using Oceananigans
using Oceananigans.Units

λ★, φ★ = 35.1, 50.1
col = Column(λ★, φ★)

grid = RectilinearGrid(size = 200, x = λ★, y = φ★,
                        z = (-400, 0),
                        topology = (Flat, Flat, Bounded))

ocean = ocean_simulation(grid; Δt = 10minutes, coriolis = FPlane(latitude = φ★))

set!(ocean.model, T = Metadatum(:temperature, dataset = ECCO4Monthly(), region = col),
                  S = Metadatum(:salinity,    dataset = ECCO4Monthly(), region = col))
```

#### Interpolation methods

`Column` supports two interpolation methods for extracting data from the surrounding grid:

- `Linear()` (default) — bilinearly interpolates from surrounding cells to the exact point.
- `Nearest()` — selects the nearest grid cell with no interpolation.

```julia
col_linear  = Column(35.1, 50.1, interpolation = Linear())
col_nearest = Column(35.1, 50.1, interpolation = Nearest())
```

## Field location

Every dataset variable has a native grid location (e.g., temperature lives at
cell centers). The function `location(metadata)` returns this location,
automatically restricted based on the region:

| Region | Input location | `location(metadata)` |
|--------|---------------|---------------------|
| `nothing` | `(Center, Center, Center)` | `(Center, Center, Center)` |
| `BoundingBox(...)` | `(Center, Center, Center)` | `(Center, Center, Center)` |
| `Column(...)` | `(Center, Center, Center)` | `(Nothing, Nothing, Center)` |
| `Column(...)` | `(Face, Center, Center)` | `(Nothing, Nothing, Center)` |
| `Column(...)` | `(Center, Center, Nothing)` | `(Nothing, Nothing, Nothing)` |

For `BoundingBox` and full-domain metadata, the location is unchanged.
For `Column` regions, horizontal locations become `Nothing` (representing `Flat` dimensions)
while the vertical location is preserved.

## `FieldTimeSeries` from `Metadata`

`FieldTimeSeries` can be constructed directly from multi-date `Metadata`,
creating a time-evolving field that loads data on demand:

```julia
using NumericalEarth
using Oceananigans
using Dates

dates = Date(2010, 1, 1) : Month(1) : Date(2010, 3, 1)
metadata = Metadata(:temperature; dataset = EN4Monthly(), dates)
fts = FieldTimeSeries(metadata)
```

The returned `FieldTimeSeries` holds `time_indices_in_memory` snapshots in memory
at a time (default: 2) and cycles through dates as needed.
This is powered by the `DatasetBackend`, which reads individual files for each
time index.

### Controlling memory usage

For long time series, keep only a small window in memory:

```julia
fts = FieldTimeSeries(metadata; time_indices_in_memory = 4)
```

### Interpolating onto a custom grid

Pass a grid instead of an architecture to interpolate the data:

```julia
grid = LatitudeLongitudeGrid(size = (360, 180, 42),
                              longitude = (0, 360),
                              latitude = (-90, 90),
                              z = (-5000, 0))

fts = FieldTimeSeries(metadata, grid)
```

### ECCO and JRA55 convenience constructors

For common workflows, NumericalEarth provides convenience constructors:

```julia
using Dates

# ECCO temperature over a date range
T_fts = FieldTimeSeries(:temperature;
                        dataset = ECCO4Monthly(),
                        dir = "path/to/ecco/data",
                        start_date = Date(1992, 1, 1),
                        end_date = Date(1992, 6, 1))

# JRA55 downwelling shortwave radiation
Qsw = JRA55FieldTimeSeries(:downwelling_shortwave_radiation;
                              start_date = Date(1990, 1, 1),
                              end_date = Date(1990, 2, 1),
                              backend = InMemory())
```

## ERA5 `FieldTimeSeries`

ERA5 reanalysis data can also be loaded as `FieldTimeSeries`.
ERA5 is a 2D surface dataset, so fields have a single vertical level:

```julia
using NumericalEarth
using NumericalEarth.DataWrangling.ERA5: ERA5Hourly
using Dates

# Download and load a month of ERA5 surface temperature
region = BoundingBox(longitude = (0, 30), latitude = (30, 60))
dates = DateTime(2020, 1, 1) : Hour(1) : DateTime(2020, 1, 31, 23)

T_meta = Metadata(:temperature; dataset = ERA5Hourly(), dates, region)
T_fts = FieldTimeSeries(T_meta)
```

### Available ERA5 variables

ERA5 provides atmospheric state variables and surface fluxes:

**State variables:**
`:temperature`, `:dewpoint_temperature`, `:eastward_velocity`,
`:northward_velocity`, `:surface_pressure`, `:specific_humidity`

**Radiation:**
`:downwelling_shortwave_radiation`, `:downwelling_longwave_radiation`

**Surface fluxes and other:**
`:total_precipitation`, `:evaporation`, `:total_cloud_cover`,
`:sea_surface_temperature`

**Wave variables (0.5° grid):**
`:eastward_stokes_drift`, `:northward_stokes_drift`,
`:significant_wave_height`, `:mean_wave_period`, `:mean_wave_direction`
