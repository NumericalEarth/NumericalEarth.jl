module BedMachine

export BedMachineAntarctica

using Downloads: Downloads
using NumericalEarth: TYPEDSIGNATURES
using Oceananigans.DistributedComputations: @root

using ..DataWrangling: DataWrangling, AbstractStaticDataset, Metadatum, Dataset, metadata_path,
                       dataset_variable_name, bounding_box_suffix, projected_window,
                       projected_window_centers, validate_projected_region, write_projected_window,
                       earthdata_download_cached, x_interfaces, y_interfaces

download_BedMachine_cache::String = ""
function __init__()
    global download_BedMachine_cache = DataWrangling.download_cache("BedMachine")
    return nothing
end

"""
    BedMachineAntarctica <: AbstractStaticDataset

BedMachine Antarctica version 4.1 (NSIDC-0756) on its native 500 m grid of the Antarctic polar
stereographic projection (EPSG:3031). Delivers, in metres, the bed elevation (`:bottom_height`), the
ice thickness (`:ice_thickness`), the surface elevation (`:surface_height`), the bed and thickness
error (`:bed_error`), the firn air content (`:firn_air_content`) and the EIGEN-6C4 geoid height above
the WGS84 ellipsoid (`:geoid_height`), and the surface type (`:ice_mask`: 0 ocean, 1 ice-free land,
2 grounded ice, 3 floating ice, 4 Lake Vostok). Cells without data are `NaN`.

Construct the [`Metadatum`](@ref) with a [`ProjectedBoundingBox`](@ref) in EPSG:3031 metres; the
native grid is a `RectilinearGrid` on those coordinates. The surface type is categorical, so it is
meaningful on the native grid only. The 1.2 GB file is downloaded once, with NASA Earthdata
credentials in the `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables (register
free at https://urs.earthdata.nasa.gov), and each region is cached as its own NetCDF file.

Reference: Morlighem et al. (2020), https://doi.org/10.1038/s41561-019-0510-8
Data source: https://nsidc.org/data/nsidc-0756

```jldoctest
using NumericalEarth

metadatum = Metadatum(:ice_thickness;
                      dataset = BedMachineAntarctica(),
                      region = ProjectedBoundingBox(x = (-1.7e6, -1.5e6), y = (-4e5, -2e5)))

metadatum.filename

# output
"BedMachine_Antarctica_ice_thickness_x_-1.7e6_-1.5e6_y_-400000.0_-200000.0.nc"
```
"""
struct BedMachineAntarctica <: AbstractStaticDataset end

const BedMachineAntarcticaMetadatum = Metadatum{<:BedMachineAntarctica}

BedMachine_dataset_variable_names = Dict(
    :bottom_height    => "bed",
    :ice_thickness    => "thickness",
    :surface_height   => "surface",
    :ice_mask         => "mask",
    :bed_error        => "errbed",
    :firn_air_content => "firn",
    :geoid_height     => "geoid")

BedMachine_units = Dict(:ice_mask => "1")

const BedMachineAntarctica_url = "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/" *
                                 "MEASURES/NSIDC-0756/4/1970/01/01/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"

#####
##### Dataset traits: the 500 m lattice, cell faces in metres
#####

DataWrangling.available_variables(::BedMachineAntarctica) = BedMachine_dataset_variable_names
DataWrangling.default_download_directory(::BedMachineAntarctica) = download_BedMachine_cache

Base.size(::BedMachineAntarctica, variable) = (13333, 13333, 1)
DataWrangling.x_interfaces(::BedMachineAntarctica) = (-3333250, 3333250)
DataWrangling.y_interfaces(::BedMachineAntarctica) = (-3333250, 3333250)

#####
##### Metadatum traits
#####

DataWrangling.is_three_dimensional(::BedMachineAntarcticaMetadatum) = false
DataWrangling.dataset_variable_name(data::BedMachineAntarcticaMetadatum) = BedMachine_dataset_variable_names[data.name]
DataWrangling.longitude_name(::BedMachineAntarcticaMetadatum) = "x"
DataWrangling.latitude_name(::BedMachineAntarcticaMetadatum)  = "y"
DataWrangling.default_inpainting(::BedMachineAntarcticaMetadatum) = nothing

DataWrangling.metadata_filename(::BedMachineAntarctica, name, date, region) =
    string("BedMachine_Antarctica_", name, "_", bounding_box_suffix(region), ".nc")

DataWrangling.validate_dataset_coverage(grid, metadatum::BedMachineAntarcticaMetadatum) =
    validate_projected_region(metadatum.region, metadatum.dataset)

#####
##### Download the file once, then cut each region into its own NetCDF
#####

function Downloads.download(metadatum::BedMachineAntarcticaMetadatum)
    DataWrangling.validate_dataset_coverage(nothing, metadatum)

    path = metadata_path(metadatum)
    @root if !isfile(path)
        source = earthdata_download_cached(BedMachineAntarctica_url, metadatum.dir)
        @info "Cutting BedMachine Antarctica $(metadatum.name) over $(summary(metadatum.region))..."
        window_to_netcdf(source, path, metadatum.name, metadatum.dataset, metadatum.region)
    end
    return path
end

function DataWrangling.retrieve_data(metadatum::BedMachineAntarcticaMetadatum)
    data = Dataset(metadata_path(metadatum)) do ds
        Array(ds[dataset_variable_name(metadatum)][:, :])
    end
    return data
end

"""
$(TYPEDSIGNATURES)

Check that the cell centers `x` (ascending) and `y` (descending, north to south) of a source file lie on
the lattice of `dataset`, and that `mapping` holds its polar stereographic projection.
"""
function validate_source_lattice(x, y, mapping, dataset)
    projection = (mapping["grid_mapping_name"], only(mapping["latitude_of_projection_origin"]),
                  only(mapping["standard_parallel"]), only(mapping["straight_vertical_longitude_from_pole"]))
    projection == ("polar_stereographic", -90, -71, 0) ||
        error("Expected the Antarctic polar stereographic projection (EPSG:3031), but the source declares $projection.")

    Nx, Ny, _ = size(dataset, nothing)
    x₁, x₂ = x_interfaces(dataset)
    y₁, y₂ = y_interfaces(dataset)
    Δx = (x₂ - x₁) / Nx
    Δy = (y₂ - y₁) / Ny
    expected = (x₁ + Δx / 2, Δx, y₂ - Δy / 2, -Δy)
    source = (x[1], x[2] - x[1], y[1], y[2] - y[1])
    all(isapprox.(source, expected)) ||
        error("The source lattice (x₁, Δx, y₁, Δy) = $source differs from the BedMachineAntarctica lattice $expected.")
    return nothing
end

# Rows are stored north to south and reversed here; fill values become NaN.
function window_to_netcdf(source, path, name, dataset, region)
    window = projected_window(dataset, region)
    columns = window.column_offset .+ (1:window.Nx)
    rows = window.row_offset .+ (1:window.Ny)
    variable_name = BedMachine_dataset_variable_names[name]

    data = Dataset(source) do ds
        validate_source_lattice(ds["x"][1:2], ds["y"][1:2], ds["mapping"].attrib, dataset)
        raw = ds[variable_name][columns, rows]
        return reverse(Float32[coalesce(value, NaN32) for value in raw], dims = 2)
    end

    x, y = projected_window_centers(dataset, window)
    write_projected_window(path, variable_name, x, y, data; units = get(BedMachine_units, name, "m"))
    return nothing
end

end # module BedMachine
