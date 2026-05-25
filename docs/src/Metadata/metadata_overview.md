# Metadata

[`Metadata`](@ref) is an abstraction that _represents_ data, but does not embody it.
Unlike [`Oceananigans.Field`](https://clima.github.io/OceananigansDocumentation/stable/appendix/library#Fields),
which points to an array occupying space in memory, `Metadata` only contains information about
where files are stored, their origin, the grid they live on, and the date(s) they correspond to (if any).

```@docs
Metadata
```

When `Metadata` represents just one date, we call it [`Metadatum`](@ref).
For example, consider global temperature from January 1st, 2010 from the
[EN4 dataset](https://www.metoffice.gov.uk/hadobs/en4/),

```@example metadata
using NumericalEarth, Dates

metadatum = Metadatum(:temperature;
                      dataset = EN4Monthly(),
                      date = Date(2010, 1, 1))
```

To materialize the data described by a `metadatum`, we wrap it in an Oceananigans' `Field`,

```@example metadata
using Oceananigans

T_native = Field(metadatum)
```

We can also interpolate the data on a user-defined grid by using the function `set!`,

```@example metadata
grid = LatitudeLongitudeGrid(size = (360, 90, 1),
                             latitude = (-90, 90),
                             longitude = (0, 360),
                             z = (0, 1))
T = CenterField(grid)
set!(T, metadatum)
```

and then we can plot it:

```@example metadata
using CairoMakie
heatmap(T)
```

This looks a bit odd, but less so if we download bathymetry (for which we also use `Metadata`
under the hood) to create a temperature field with a land mask,

```@example metadata
bottom_height = regrid_bathymetry(grid)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
T = CenterField(grid)
set!(T, metadatum)
heatmap(T)
```

The key ingredients stored in a [`Metadata`](@ref) or [`Metadatum`](@ref) object are:

- the variable name (for example `:temperature` or `:u_velocity`);
- the dataset (such as `EN4Monthly`, `ECCO2Daily`, or `GLORYSMonthly`);
- the temporal coverage: either a single timestamp (`Metadatum`) or a range/vector of dates (`Metadata`);
- an optional `region` describing the spatial extent — either a
  [`BoundingBox`](@ref NumericalEarth.DataWrangling.BoundingBox) for a rectangular sub-domain,
  a [`Column`](@ref NumericalEarth.DataWrangling.Column) for a single horizontal location, or
  `nothing` for the full global domain (see [Regions, locations, and FieldTimeSeries](@ref));
- the on-disk `dir`ectory where the dataset files are cached.

This bookkeeping lets downstream utilities (for example `set!` or `FieldTimeSeries`) request exactly the
slices of data they need, and it keeps track of where those slices live so we do not redownload
them unnecessarily.

## Bundling variables with `MetadataSet`

Workflows often need _many_ variables from the same dataset — for example, temperature and salinity
to initialize an ocean model, or wind, humidity, pressure, and precipitation to drive an atmosphere.
Writing one `Metadata` (or `Metadatum`) per variable repeats the same `dataset`, `dates`, `region`, and
`dir` over and over. A [`MetadataSet`](@ref) bundles those variables into one object whose elements
are still individual `Metadata`/`Metadatum`:

```@example metadata
mset = MetadataSet(:temperature, :salinity;
                   dataset = EN4Monthly(),
                   date    = Date(2010, 1, 1))
```

The variable axis is exposed via property and indexed access; struct fields stay reachable too.
With a scalar `date`, each element is a `Metadatum`:

```@example metadata
mset.temperature              # → a `Metadatum`
```

```@example metadata
mset[:salinity] === mset[2]   # property and indexed access are symmetric
```

```@example metadata
keys(mset), mset.dataset      # the variable axis, plus shared kwargs
```

Pass a `dates` range (or vector) instead of a scalar `date` to bundle a time axis;
each element is then a `Metadata` covering that range:

```@example metadata
mset_ts = MetadataSet(:temperature, :salinity;
                      dataset = EN4Monthly(),
                      dates   = DateTime(2010, 1, 1):Month(1):DateTime(2010, 3, 1))
mset_ts.temperature           # → a `Metadata` (multi-date)
```

### `set!(model, mset)` — auto-routing

`set!(model, mset)` translates each variable's verbose dataset name (`:temperature`, `:salinity`, ...)
to the short model field-name the model expects (`:T`, `:S`, `:u`, `:ℵ`, ...) and forwards the result
as keyword arguments to the model's underlying `set!`. The translation table lives in
`NumericalEarth.DataWrangling.variable_glossary`, populated from the conventions in
[Notation](@ref Notation) — so a coupled set can drive ocean and sea-ice components from one call:

```julia
mset = MetadataSet(:temperature, :salinity,
                   :sea_ice_thickness, :sea_ice_concentration;
                   dataset = ECCO4Monthly(), date = start_date)

set!(ocean.model,   mset)   # picks up :temperature, :salinity → T, S
set!(sea_ice.model, mset)   # picks up :sea_ice_thickness, :sea_ice_concentration → h, ℵ
```

Variables absent from `variable_glossary` are silently skipped (lets one set partially drive each
component without manual filtering).

### Building `Field`s and `FieldTimeSeries` in bulk

`Field(mset, arch=CPU(); kw...)` and `FieldTimeSeries(mset, arch_or_grid; kw...)` build a
`NamedTuple` keyed by the variable names, with each value materialized from the underlying
per-variable `Metadata`. `Field` requires a scalar `date` (one snapshot per variable); for
multi-date sets, use `FieldTimeSeries`:

```@example metadata
fields = Field(mset)              # (; temperature = Field, salinity = Field)
fields.temperature
```

```@example metadata
fts = FieldTimeSeries(mset_ts)    # NamedTuple of FieldTimeSeries, one per variable
fts.temperature[1]                # first temperature snapshot, as a Field
```

### Downloading

`download(mset)` fetches every variable in the set. The default is a per-variable loop; backends
that support batched multi-variable requests override this — for example, ERA5 pressure-level sets
route through one CDS API request per calendar day instead of one per (variable, day) pair.

## Supported datasets

NumericalEarth currently ships connectors for the following data products:

| Dataset            | Supported Variables                                      | Documentation Link                                                                                 |
|--------------------|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Bathymetry**     |                                                           |                                                                                                     |
| `ETOPO2022`        | [Supported variables](@ref dataset-etopo2022-vars)        | [NOAA ETOPO 2022 overview](https://www.ncei.noaa.gov/products/etopo-global-relief-model)           |
| `GEBCO2024`        | [Supported variables](@ref dataset-gebco2024-vars)        | [GEBCO 2024 overview](https://www.gebco.net/data_and_products/gridded_bathymetry_data/)             |
| `IBCSOv2`          | [Supported variables](@ref dataset-ibcsov2-vars)          | [IBCSO overview](https://ibcso.org/ibcso-2024-annual-release/)                                      |
| `IBCAOv5`          | [Supported variables](@ref dataset-ibcaov5-vars)          | [IBCAO overview](https://www.gebco.net/data_and_products/gridded_bathymetry_data/arctic_ocean/)     |
| **Ocean reanalysis** |                                                         |                                                                                                     |
| `ECCO2Monthly`     | [Supported variables](@ref dataset-ecco2monthly-vars)     | [ECCO2 documentation](https://ecco.jpl.nasa.gov/products/all/)                                     |
| `ECCO2Daily`       | [Supported variables](@ref dataset-ecco2daily-vars)       | [ECCO2 documentation](https://ecco.jpl.nasa.gov/products/all/)                                     |
| `ECCO4Monthly`     | [Supported variables](@ref dataset-ecco4monthly-vars)     | [ECCO V4r4 product guide](https://ecco-group.org/products-ECCO-V4r4.htm)                           |
| `EN4Monthly`       | [Supported variables](@ref dataset-en4monthly-vars)       | [Met Office EN4 overview](https://www.metoffice.gov.uk/hadobs/en4/)                                |
| `GLORYSDaily`      | [Supported variables](@ref dataset-glorysdaily-vars)      | [Copernicus GLORYS product page](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) |
| `GLORYSMonthly`    | [Supported variables](@ref dataset-glorysmonthly-vars)    | [Copernicus GLORYS product page](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) |
| **Atmospheric forcing** |                                                      |                                                                                                     |
| `RepeatYearJRA55`  | [Supported variables](@ref dataset-repeatyearjra55-vars)  | [JRA-55 Reanalysis](https://www.data.jma.go.jp/jra/html/JRA-55/index_en.html)                                 |
| `MultiYearJRA55`   | [Supported variables](@ref dataset-multiyearjra55-vars)   | [JRA-55 Reanalysis](https://www.data.jma.go.jp/jra/html/JRA-55/index_en.html)                                 |
