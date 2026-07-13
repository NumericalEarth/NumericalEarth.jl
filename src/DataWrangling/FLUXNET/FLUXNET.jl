module FLUXNET

export FLUXNETSite
export FLUXNETPrescribedAtmosphere
export FLUXNETPrescribedRadiation
export fluxnet_flux_observations

using Dates: Dates, DateTime
using Downloads: Downloads
using Glob: glob
using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans.DistributedComputations: child_architecture
using Oceananigans.Fields: Field, interior
using Oceananigans.Grids: RectilinearGrid, Center, Flat
using Oceananigans.OutputReaders: FieldTimeSeries
using Thermodynamics: q_vap_from_RH, saturation_vapor_pressure, Liquid

using ..DataWrangling: DataWrangling, Metadata, Metadatum,
                       first_date, last_date, fill_gaps!, Celsius

using ...Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux,
                      AtmosphereThermodynamicsParameters
using ...Radiations: PrescribedRadiation, SurfaceRadiationProperties,
                     default_stefan_boltzmann_constant

download_FLUXNET_cache::String = ""

function __init__()
    global download_FLUXNET_cache = DataWrangling.download_cache("FLUXNET")
end

#####
##### Time resolution helpers
#####

# The FLUXNET2015 filename token and averaging interval (seconds) for each resolution.
resolution_token(::Val{:halfhourly}) = "HH"
resolution_token(::Val{:hourly})     = "HR"
resolution_token(::Val{:daily})      = "DD"
resolution_token(resolution::Symbol) = resolution_token(Val(resolution))

resolution_seconds(::Val{:halfhourly}) = 1800
resolution_seconds(::Val{:hourly})     = 3600
resolution_seconds(::Val{:daily})      = 86400
resolution_seconds(resolution::Symbol) = resolution_seconds(Val(resolution))

#####
##### CSV parsing (FLUXNET stores one comma-separated file per site and resolution)
#####

const FLUXNET_MISSING_VALUE = -9999.0
const FLUXNET_SUBDAILY_FORMAT = Dates.DateFormat("yyyymmddHHMM")
const FLUXNET_DAILY_FORMAT    = Dates.DateFormat("yyyymmdd")

# Timestamps are `YYYYMMDDHHMM` (subdaily) or `YYYYMMDD` (daily); use the start of
# the averaging interval (`TIMESTAMP_START`) as the sample time.
function parse_fluxnet_timestamp(field)
    s = strip(field)
    return length(s) ≥ 12 ? DateTime(s[1:12], FLUXNET_SUBDAILY_FORMAT) :
                            DateTime(s[1:8],  FLUXNET_DAILY_FORMAT)
end

parse_fluxnet_value(field) = _clean(tryparse(Float64, field))
_clean(::Nothing) = NaN
_clean(value::Float64) = ifelse(value == FLUXNET_MISSING_VALUE, NaN, value)

"""
    read_fluxnet_csv(path)

Parse a FLUXNET2015 CSV, returning `(timestamps, columns)` where `timestamps` is
a `Vector{DateTime}` built from `TIMESTAMP_START` (or `TIMESTAMP` for daily data)
and `columns` maps each remaining column name to a `Vector{Float64}` with the
`-9999` missing sentinel replaced by `NaN`.
"""
function read_fluxnet_csv(path)
    lines = readlines(path)
    length(lines) < 2 && error("FLUXNET file \"$path\" has no data rows")

    header = string.(split(strip(lines[1]), ','))
    time_index = findfirst(==("TIMESTAMP_START"), header)
    isnothing(time_index) && (time_index = findfirst(==("TIMESTAMP"), header))
    isnothing(time_index) &&
        error("FLUXNET file \"$path\" has no TIMESTAMP_START or TIMESTAMP column")

    n = length(lines) - 1
    timestamps = Vector{DateTime}(undef, n)

    is_data_column = [!(name in ("TIMESTAMP_START", "TIMESTAMP_END", "TIMESTAMP")) for name in header]
    columns = Dict{String, Vector{Float64}}(header[c] => Vector{Float64}(undef, n)
                                             for c in eachindex(header) if is_data_column[c])

    for row in 1:n
        fields = split(strip(lines[row + 1]), ',')
        timestamps[row] = parse_fluxnet_timestamp(fields[time_index])
        for c in eachindex(header)
            is_data_column[c] || continue
            columns[header[c]][row] = parse_fluxnet_value(fields[c])
        end
    end

    return timestamps, columns
end

#####
##### File discovery and parsed-file caches (keyed by resolved file path)
#####

const FLUXNET_FILE_CACHE = Dict{String, Tuple{Vector{DateTime}, Dict{String, Vector{Float64}}}}()
const FLUXNET_TIME_INDEX_CACHE = Dict{String, Dict{DateTime, Int}}()

include("FLUXNET_metadata.jl")

#####
##### Shared builder utility
#####

# Load one variable onto the single-column `grid`, filling short gaps by linear
# interpolation (FLUXNET `_F` variables are already gap-filled, so this is a safety net).
function fluxnet_field_time_series(site, name, grid; start_date, end_date, max_gap)
    md = Metadata(name; dataset = site, start_date, end_date, dir = site.dir)
    fts = FieldTimeSeries(md, grid; time_indices_in_memory = length(md))
    max_gap > 0 && fill_gaps!(fts; max_gap)
    return fts
end

include("FLUXNET_prescribed_atmosphere.jl")
include("FLUXNET_prescribed_radiation.jl")
include("FLUXNET_flux_observations.jl")

end # module FLUXNET
