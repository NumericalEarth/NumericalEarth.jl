import Oceananigans: location
import Oceananigans.Fields: set!

using Oceananigans.Architectures: on_architecture
using Oceananigans.DistributedComputations: child_architecture

#####
##### OSPapaFluxHourly dataset type
#####

struct OSPapaFluxHourly end

const OSPapaFluxMetadata{D} = Metadata{<:OSPapaFluxHourly, D}
const OSPapaFluxMetadatum   = Metadatum{<:OSPapaFluxHourly}

metaprefix(::OSPapaFluxMetadata) = "OSPapaFluxMetadata"

default_download_directory(::OSPapaFluxHourly) = mkpath(download_OSPapa_cache)

available_variables(::OSPapaFluxHourly) = OSPapa_flux_variable_names

const OSPapa_flux_variable_names = Dict(
    :net_heat_flux                   => "QNET",
    :latent_heat_flux                => "QLAT",
    :sensible_heat_flux              => "QSEN",
    :net_shortwave_radiation         => "SWNET",
    :net_longwave_radiation          => "LWNET",
    :zonal_stress                    => "TAUX",
    :meridional_stress               => "TAUY",
    :evaporation                     => "EVAP",
    :rain                            => "RAIN",
    :evaporation_minus_precipitation => "EMP",
    :skin_temperature                => "TSK",
)

dataset_variable_name(md::OSPapaFluxMetadata) = OSPapa_flux_variable_names[md.name]

location(::OSPapaFluxMetadata) = (Center, Center, Center)
is_three_dimensional(::OSPapaFluxMetadata) = false
conversion_units(::OSPapaFluxMetadatum) = nothing
default_inpainting(::OSPapaFluxMetadata) = nothing

Base.size(::OSPapaFluxHourly, variable) = (1, 1, 1)

# The uniform hourly cache file is regenerated from the raw ERDDAP file by
# download_dataset; metadata_epoch and metadata_time_step describe the
# intended uniform axis.
metadata_epoch(::OSPapaFluxHourly)     = DateTime(2007, 6, 8)
metadata_time_step(::OSPapaFluxHourly) = 3600

# Full advertised ERDDAP coverage for the ocs_papa_flux dataset. Used only
# for `first_date`/`last_date` default-argument evaluation; users can pass
# narrower windows via start_date/end_date.
const OSPAPA_FLUX_ALL_DATES = DateTime(2007, 6, 8):Hour(1):DateTime(2022, 2, 24)

all_dates(::OSPapaFluxHourly, variable) = OSPAPA_FLUX_ALL_DATES

#####
##### Grid construction
#####

longitude_interfaces(::OSPapaFluxHourly) = (OSPAPA_LONGITUDE, OSPAPA_LONGITUDE)
latitude_interfaces(::OSPapaFluxHourly)  = (OSPAPA_LATITUDE, OSPAPA_LATITUDE)

function native_grid(::OSPapaFluxMetadata, arch=CPU(); halo=(3, 3, 3))
    return RectilinearGrid(arch; size=(), topology=(Flat, Flat, Flat))
end

#####
##### ERDDAP download + uniform-grid cache file
#####

const ERDDAP_BASE = "https://data.pmel.noaa.gov/pmel/erddap/tabledap"
const ERDDAP_FLUX_VARS = "time,QLAT,QSEN,QNET,LWNET,SWNET,TAU,TAUX,TAUY,RAIN,EVAP,EMP,TSK"

function download_ospapa_flux(; start_date, end_date, dir=download_OSPapa_cache)
    filename = "ocs_papa_flux_raw_$(Dates.format(start_date, "yyyymmddTHHMMSS"))_$(Dates.format(end_date, "yyyymmddTHHMMSS")).nc"
    filepath = joinpath(dir, filename)
    if !isfile(filepath)
        t0 = Dates.format(start_date, "yyyy-mm-ddTHH:MM:SSZ")
        t1 = Dates.format(end_date, "yyyy-mm-ddTHH:MM:SSZ")
        url = "$(ERDDAP_BASE)/ocs_papa_flux.nc?$(ERDDAP_FLUX_VARS)&time>=$(t0)&time<=$(t1)"
        @info "Downloading OS Papa flux data from ERDDAP..."
        Downloads.download(url, filepath; progress=download_progress)
    end
    return filepath
end

flux_uniform_filename(start_date, end_date) =
    "ocs_papa_flux_uniform_$(Dates.format(start_date, "yyyymmddTHHMMSS"))_$(Dates.format(end_date, "yyyymmddTHHMMSS")).nc"

metadata_filename(::OSPapaFluxHourly, name, date, bounding_box) = flux_uniform_filename(date, date)

build_filename(::OSPapaFluxHourly, name, dates::AbstractArray, bounding_box) =
    flux_uniform_filename(first(dates), last(dates))

function download_dataset(md::OSPapaFluxMetadata)
    uniform_path = joinpath(md.dir, metadata_filename(md))
    isfile(uniform_path) && return nothing

    if !(md.dates isa AbstractArray)
        error("OSPapaFluxHourly uniform cache $(uniform_path) is missing; " *
              "construct os_papa_prescribed_fluxes or a multi-date Metadata first.")
    end

    start_date = first(md.dates)
    end_date   = last(md.dates)
    raw_path = download_ospapa_flux(; start_date, end_date, dir=md.dir)
    _write_uniform_flux_file(raw_path, uniform_path, start_date, end_date)
    return nothing
end

function _write_uniform_flux_file(raw_path, uniform_path, start_date, end_date)
    ds = NCDataset(raw_path)
    raw_times = DateTime.(ds["time"][:])
    dt_to_raw_idx = Dict(t => i for (i, t) in enumerate(raw_times))

    uniform_datetimes = start_date:Hour(1):end_date
    N = length(uniform_datetimes)

    expanded = Dict{String, Vector{Float64}}()
    for ncname in values(OSPapa_flux_variable_names)
        raw = Float64.(replace(ds[ncname][:], missing => NaN))
        uniform = fill(NaN, N)
        for (j, t) in enumerate(uniform_datetimes)
            i = get(dt_to_raw_idx, t, nothing)
            isnothing(i) || (uniform[j] = raw[i])
        end
        expanded[ncname] = uniform
    end
    close(ds)

    NCDataset(uniform_path, "c") do out
        defDim(out, "X", 1)
        defDim(out, "Y", 1)
        defDim(out, "Z", 1)
        defDim(out, "TIME", N)

        time_var = defVar(out, "TIME", Float64, ("TIME",);
                          attrib=Dict("units" => "seconds since 1970-01-01 00:00:00",
                                      "calendar" => "standard"))
        time_var[:] = [Dates.datetime2unix(t) for t in uniform_datetimes]

        for (ncname, data) in expanded
            v = defVar(out, ncname, Float64, ("X", "Y", "Z", "TIME"))
            v[:, :, :, :] = reshape(data, 1, 1, 1, N)
        end
    end
    return uniform_path
end

#####
##### Data retrieval
#####

function retrieve_data(metadata::OSPapaFluxMetadatum)
    filepath = metadata_path(metadata)
    ds = NCDataset(filepath)
    varname = dataset_variable_name(metadata)

    all_times = DateTime.(ds["TIME"][:])
    t_idx = findfirst(t -> t == metadata.dates, all_times)

    if isnothing(t_idx)
        close(ds)
        error("Date $(metadata.dates) not found in OS Papa flux dataset")
    end

    raw = ds[varname][1, 1, 1, t_idx]
    close(ds)

    data = Float64(ismissing(raw) ? NaN : raw)
    return reshape([data], 1, 1, 1)
end

function set!(target_field::Field, metadata::OSPapaFluxMetadatum; kw...)
    grid = target_field.grid
    arch = child_architecture(grid)
    meta_field = Field(metadata, arch; kw...)
    parent(target_field) .= parent(meta_field)
    return target_field
end
