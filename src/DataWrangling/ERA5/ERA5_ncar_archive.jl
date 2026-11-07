#####
##### NSF NCAR's ERA5 mirror (RDA ds633.0) in the public `nsf-ncar-era5` AWS S3 bucket
#####

const NCAR_ERA5_URL = "https://nsf-ncar-era5.s3.amazonaws.com"

struct NCARAnalysis
    stream :: String
end

struct NCARForecast
    stream :: String
end

struct NCARInvariant end

# NumericalEarth name => (NCAR stream, NCAR parameter code, scale to the CDS variable's units).
# `total_precipitation` is built from the mean total precipitation rate (kg m⁻² s⁻¹), scaled by
# 3600 s / 1000 kg m⁻³ to the hourly accumulated depth (m) that CDS provides.
const NCAR_ERA5_single_level_variables = Dict(
    :temperature                     => (NCARAnalysis("e5.oper.an.sfc"), "128_167_2t", 1),
    :dewpoint_temperature            => (NCARAnalysis("e5.oper.an.sfc"), "128_168_2d", 1),
    :eastward_velocity               => (NCARAnalysis("e5.oper.an.sfc"), "128_165_10u", 1),
    :northward_velocity              => (NCARAnalysis("e5.oper.an.sfc"), "128_166_10v", 1),
    :surface_pressure                => (NCARAnalysis("e5.oper.an.sfc"), "128_134_sp", 1),
    :mean_sea_level_pressure         => (NCARAnalysis("e5.oper.an.sfc"), "128_151_msl", 1),
    :skin_temperature                => (NCARAnalysis("e5.oper.an.sfc"), "128_235_skt", 1),
    :sea_surface_temperature         => (NCARAnalysis("e5.oper.an.sfc"), "128_034_sstk", 1),
    :total_cloud_cover               => (NCARAnalysis("e5.oper.an.sfc"), "128_164_tcc", 1),
    :total_column_water_vapor        => (NCARAnalysis("e5.oper.an.sfc"), "128_137_tcwv", 1),
    :vertically_integrated_eastward_water_vapor_flux  => (NCARAnalysis("e5.oper.an.vinteg"), "162_071_viwve", 1),
    :vertically_integrated_northward_water_vapor_flux => (NCARAnalysis("e5.oper.an.vinteg"), "162_072_viwvn", 1),
    :downwelling_shortwave_radiation => (NCARForecast("e5.oper.fc.sfc.accumu"), "128_169_ssrd", 1),
    :downwelling_longwave_radiation  => (NCARForecast("e5.oper.fc.sfc.accumu"), "128_175_strd", 1),
    :evaporation                     => (NCARForecast("e5.oper.fc.sfc.accumu"), "128_182_e", 1),
    :snowfall                        => (NCARForecast("e5.oper.fc.sfc.accumu"), "128_144_sf", 1),
    :total_precipitation             => (NCARForecast("e5.oper.fc.sfc.meanflux"), "235_055_mtpr", 3.6),
    :mean_surface_momentum_flux_x    => (NCARForecast("e5.oper.fc.sfc.meanflux"), "235_041_metss", 1),
    :mean_surface_momentum_flux_y    => (NCARForecast("e5.oper.fc.sfc.meanflux"), "235_042_mntss", 1),
    :mean_surface_sensible_heat_flux => (NCARForecast("e5.oper.fc.sfc.meanflux"), "235_033_msshf", 1),
    :mean_surface_latent_heat_flux   => (NCARForecast("e5.oper.fc.sfc.meanflux"), "235_034_mslhf", 1),
    :mean_evaporation_rate           => (NCARForecast("e5.oper.fc.sfc.meanflux"), "235_043_mer", 1),
    :geopotential                    => (NCARInvariant(), "128_129_z", 1),
    :topography                      => (NCARInvariant(), "128_129_z", 1),
)

# NCAR names each variable after the upper-case ECMWF short name, prefixed when it starts with a digit
function ncar_netcdf_name(code)
    short_name = uppercase(last(split(code, '_')))
    return isdigit(first(short_name)) ? "VAR_" * short_name : short_name
end

yyyymm(date) = @sprintf("%04d%02d", Dates.year(date), Dates.month(date))

ncar_url(stream, month, code, span) = string(NCAR_ERA5_URL, "/", stream, "/", month, "/",
                                             stream, ".", code, ".ll025sc.", span, ".nc")

# Analyses: one file per month, hourly from 00 UTC on the 1st, indexed by time.
function ncar_location(s::NCARAnalysis, code, date)
    month = yyyymm(date)
    span = @sprintf("%s0100_%s%02d23", month, month, Dates.daysinmonth(date))
    t = 24 * (Dates.day(date) - 1) + Dates.hour(date) + 1
    return ncar_url(s.stream, month, code, span), (t,)
end

# Forecasts: initialized at 06 and 18 UTC, valid 1–12 hours later, in half-month files
# indexed by (forecast_hour, forecast_initial_time).
function ncar_location(s::NCARForecast, code, date)
    forecast_hour = mod(Dates.hour(date) - 7, 12) + 1
    initial_time = date - Hour(forecast_hour)
    month = yyyymm(initial_time)
    first_half = Dates.day(initial_time) ≤ 15
    span = first_half ? string(month, "0106_", month, "1606") :
                        string(month, "1606_", yyyymm(initial_time + Month(1)), "0106")
    n = 2 * (Dates.day(initial_time) - (first_half ? 1 : 16)) + (Dates.hour(initial_time) == 18) + 1
    return ncar_url(s.stream, month, code, span), (forecast_hour, n)
end

ncar_location(::NCARInvariant, code, date) =
    ncar_url("e5.oper.invariant", "197901", code, "1979010100_1979010100"), (1,)

# Each block is read with one request: analyses in time chunks, forecasts one initialization at a time.
function ncar_blocks(variable, locations)
    time_chunk = last(last(NCDatasets.chunking(variable)))
    block(index) = length(index) == 1 ? (cld(index[1], time_chunk),) : index[2:2]
    return [filter(i -> block(locations[i]) == key, eachindex(locations)) for key in unique(block.(locations))]
end

#####
##### Horizontal crop of the global 0.25° grid
#####

function ncar_crop(region::BoundingBox)
    Δ = 1/4
    p = 2  # native cells of margin around the region
    λ = something(region.longitude, (0, 360 - Δ))
    φ = something(region.latitude, (-90, 90))
    k = λ[2] - λ[1] ≥ 360 ? (0:1439) : (floor(Int, λ[1] / Δ) - p : ceil(Int, λ[2] / Δ) + p)
    j = max(1, floor(Int, (90 - φ[2]) / Δ) + 1 - p) : min(721, ceil(Int, (90 - φ[1]) / Δ) + 1 + p)
    return k, j
end

ncar_crop(column::Column) = ncar_crop(BoundingBox(longitude = (column.longitude, column.longitude),
                                                  latitude = (column.latitude, column.latitude)))

ncar_crop(::Nothing) = 0:1439, 1:721

# Longitude offsets `k` (in units of 0.25° from 0°E) may wrap the globe, so read contiguous runs.
function read_ncar_slab(variable, k, j, trailing...)
    i = mod.(k, 1440) .+ 1
    seams = [0; findall(diff(i) .!= 1); length(i)]
    runs = [i[seams[n]+1]:i[seams[n+1]] for n in 1:length(seams)-1]
    return cat((variable[run, j, trailing...] for run in runs)...; dims=1)
end

function write_era5_file(path, name, date, λ, φ, data)
    mkpath(dirname(path))
    mktemp(dirname(path)) do partial_path, io
        close(io)
        NCDatasets.Dataset(partial_path, "c") do ds
            NCDatasets.defVar(ds, "longitude", λ, ("longitude",))
            NCDatasets.defVar(ds, "latitude", φ, ("latitude",))
            NCDatasets.defVar(ds, "valid_time", [date], ("valid_time",))
            NCDatasets.defVar(ds, ERA5_netcdf_variable_names[name], reshape(data, size(data)..., 1),
                              ("longitude", "latitude", "valid_time"); fillvalue = NaN32)
        end
        mv(partial_path, path; force=true)
    end
    return nothing
end

#####
##### Downloads
#####

const ERA5NCARHourlySingleLevel = ERA5HourlySingleLevel{NCARArchive}

"""
$(TYPEDSIGNATURES)

Read `metadata` from NSF NCAR's ERA5 mirror on AWS S3 and write one file per date in the layout
the Copernicus Climate Data Store downloads produce. Only the hours and the region in `metadata`
are transferred. No account or credentials are needed.
"""
function Downloads.download(metadata::Metadata{<:ERA5NCARHourlySingleLevel}; skip_existing = true)
    haskey(NCAR_ERA5_single_level_variables, metadata.name) ||
        throw(ArgumentError("$(metadata.name) is not available from NCARArchive()."))

    paths = [metadata_path(metadatum) for metadatum in metadata]
    dates = [metadatum.dates for metadatum in metadata]
    pending = filter(n -> !(skip_existing && isfile(paths[n])), eachindex(paths))

    stream, code, scale = NCAR_ERA5_single_level_variables[metadata.name]
    k, j = ncar_crop(metadata.region)
    λ = k ./ 4
    φ = 90 .- (j .- 1) ./ 4

    locations = [ncar_location(stream, code, dates[n]) for n in pending]

    @root for url in unique(first.(locations))
        in_file = filter(n -> first(locations[n]) == url, eachindex(locations))
        @info "Reading $(metadata.name) for $(length(in_file)) dates from $url"
        NCDatasets.Dataset(url * "#mode=bytes") do ds
            variable = ds[ncar_netcdf_name(code)]
            indices = last.(locations[in_file])
            for block in ncar_blocks(variable, indices)
                first_index = minimum(indices[b][1] for b in block)
                last_index  = maximum(indices[b][1] for b in block)
                slab = read_ncar_slab(variable, k, j, first_index:last_index, indices[block[1]][2:end]...)
                for b in block
                    data = scale .* coalesce.(slab[:, :, indices[b][1] - first_index + 1], NaN32)
                    n = pending[in_file[b]]
                    write_era5_file(paths[n], metadata.name, dates[n], λ, φ, Float32.(data))
                end
            end
        end
    end

    return paths
end

Downloads.download(metadatum::Metadatum{<:ERA5NCARHourlySingleLevel}; kw...) =
    first(invoke(Downloads.download, Tuple{Metadata{<:ERA5NCARHourlySingleLevel}}, metadatum; kw...))

Downloads.download(mset::MetadataSet{<:ERA5NCARHourlySingleLevel}; kw...) =
    invoke(Downloads.download, Tuple{MetadataSet}, mset; kw...)
