# Analytic stand-ins for the remote datasets. Their `download` writes a small NetCDF file, so
# `Field`, `FieldTimeSeries`, `DatasetRestoring` and `regrid_bathymetry` run their dataset code
# paths without network access, and tests can assert exact values.

using Dates
using Downloads: Downloads
using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth.Atmospheres: PrescribedPrecipitationFlux
using NumericalEarth.DataWrangling: DataWrangling, AbstractStaticBathymetry, native_grid
using Oceananigans.Grids: λnodes, φnodes, znodes

const synthetic_data_directory = mktempdir()

struct SyntheticBathymetry <: AbstractStaticBathymetry end

abstract type SyntheticDataset end
struct SyntheticOcean <: SyntheticDataset end      # 3-D `:temperature` and `:salinity`
struct SyntheticAtmosphere <: SyntheticDataset end # 2-D fields named as in JRA55

const Synthetic = Union{SyntheticBathymetry, SyntheticDataset}

DataWrangling.default_download_directory(::Synthetic) = synthetic_data_directory
DataWrangling.longitude_interfaces(::Synthetic) = (-180, 180)
DataWrangling.latitude_interfaces(::Synthetic) = (-90, 90)
DataWrangling.dataset_variable_name(metadata::Metadata{<:Synthetic}) = string(metadata.name)

Base.size(::SyntheticBathymetry) = (36, 18, 1)
DataWrangling.metadata_filename(::SyntheticBathymetry, name, date, region) = "synthetic_bathymetry.nc"

Base.size(::SyntheticOcean, name) = (36, 18, 4)
Base.size(::SyntheticAtmosphere, name) = (36, 18, 1)
DataWrangling.z_interfaces(::SyntheticOcean) = (-5000, 0)
DataWrangling.reversed_vertical_axis(::SyntheticOcean) = false
DataWrangling.is_three_dimensional(::Metadata{<:SyntheticAtmosphere}) = false
DataWrangling.all_dates(::SyntheticDataset, name) = DateTime(2000, 1, 1):Month(1):DateTime(2000, 12, 1)

DataWrangling.metadata_filename(::SyntheticDataset, name, date, region) =
    string("synthetic_", name, "_", year(date), "_", lpad(month(date), 2, '0'), ".nc")

DataWrangling.inpainted_metadata_path(metadata::Metadatum{<:SyntheticDataset}) =
    joinpath(metadata.dir, replace(metadata.filename, ".nc" => "_inpainted.jld2"))

# One continent in an otherwise global ocean
synthetic_land(λ, φ) = -60 ≤ λ ≤ 20 && -30 ≤ φ ≤ 40

synthetic_value(::SyntheticBathymetry, name, λ, φ, z) = synthetic_land(λ, φ) ? 500 : -4000

function synthetic_value(::SyntheticOcean, name, λ, φ, z)
    synthetic_land(λ, φ) && return NaN
    stratification = cosd(φ)^2 * (1 + z / 5000)
    return name == :temperature ? 2 + 20 * stratification : 34 + stratification
end

function synthetic_value(::SyntheticAtmosphere, name, λ, φ, z)
    if name == :temperature
        273 + 15 * cosd(φ)
    elseif name == :sea_level_pressure
        101325
    elseif name == :specific_humidity
        0.01
    elseif name == :eastward_velocity
        5
    elseif name == :northward_velocity
        1
    elseif name == :downwelling_shortwave_radiation
        300 * cosd(φ)
    elseif name == :downwelling_longwave_radiation
        250
    elseif name in (:rain_freshwater_flux, :snow_freshwater_flux, :river_freshwater_flux, :iceberg_freshwater_flux)
        1e-5
    else
        error("SyntheticAtmosphere has no variable $name")
    end
end

function write_synthetic_netcdf(path, metadatum)
    # The file is global whatever the region of `metadatum`: one file serves every region.
    grid = native_grid(Metadata(metadatum.name, metadatum.dataset, metadatum.dates, nothing, metadatum.dir))
    λ = λnodes(grid, Center())
    φ = φnodes(grid, Center())
    three_dimensional = metadatum.dataset isa SyntheticOcean
    z = three_dimensional ? znodes(grid, Center()) : [0]
    data = [Float32(synthetic_value(metadatum.dataset, metadatum.name, λ[i], φ[j], z[k]))
            for i in eachindex(λ), j in eachindex(φ), k in eachindex(z)]

    NCDataset(path, "c") do ds
        defDim(ds, "longitude", length(λ))
        defDim(ds, "latitude", length(φ))
        defVar(ds, "longitude", Float64, ("longitude",))[:] = λ
        defVar(ds, "latitude", Float64, ("latitude",))[:] = φ
        if three_dimensional
            defDim(ds, "z", length(z))
            defVar(ds, "z", Float64, ("z",))[:] = z
            defVar(ds, string(metadatum.name), Float32, ("longitude", "latitude", "z"))[:, :, :] = data
        else
            defVar(ds, string(metadatum.name), Float32, ("longitude", "latitude"))[:, :] = data[:, :, 1]
        end
    end

    return path
end

# Each process writes into its own directory.
function Downloads.download(metadata::Metadata{<:Synthetic})
    for metadatum in metadata
        path = metadata_path(metadatum)
        isfile(path) || write_synthetic_netcdf(path, metadatum)
    end
    return metadata_path(metadata)
end

synthetic_field_time_series(name, arch; dates = all_dates(SyntheticAtmosphere(), name), time_indices_in_memory = 2) =
    FieldTimeSeries(Metadata(name; dataset = SyntheticAtmosphere(), dates), arch;
                    time_indices_in_memory, inpainting = nothing)

function synthetic_prescribed_atmosphere(arch = CPU(); kw...)
    u  = synthetic_field_time_series(:eastward_velocity, arch; kw...)
    v  = synthetic_field_time_series(:northward_velocity, arch; kw...)
    T  = synthetic_field_time_series(:temperature, arch; kw...)
    qᵛ = synthetic_field_time_series(:specific_humidity, arch; kw...)
    p  = synthetic_field_time_series(:sea_level_pressure, arch; kw...)
    rain = synthetic_field_time_series(:rain_freshwater_flux, arch; kw...)
    snow = synthetic_field_time_series(:snow_freshwater_flux, arch; kw...)
    precipitation_flux = PrescribedPrecipitationFlux(; rain, snow)

    return PrescribedAtmosphere(u.grid, u.times;
                                source = SyntheticAtmosphere(),
                                velocities = (; u, v),
                                temperature = T,
                                specific_humidity = qᵛ,
                                pressure = p,
                                precipitation_flux)
end

synthetic_prescribed_radiation(arch = CPU(); kw...) =
    PrescribedRadiation(synthetic_field_time_series(:downwelling_shortwave_radiation, arch; kw...),
                        synthetic_field_time_series(:downwelling_longwave_radiation, arch; kw...))

synthetic_prescribed_land(arch = CPU(); kw...) =
    PrescribedLand((; rivers = synthetic_field_time_series(:river_freshwater_flux, arch; kw...),
                      icebergs = synthetic_field_time_series(:iceberg_freshwater_flux, arch; kw...)))

synthetic_bottom_height(grid; kw...) =
    regrid_bathymetry(grid, Metadatum(:bottom_height; dataset = SyntheticBathymetry()); cache = false, kw...)
