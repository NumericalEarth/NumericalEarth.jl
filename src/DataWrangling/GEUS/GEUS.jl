module GEUS

# Datasets
export GEUSSolidIceDischarge

# Prescribed components
export GEUSPrescribedLand

using Dates: Dates, DateTime, Day, Month, Millisecond, firstdayofmonth, lastdayofmonth
using DocStringExtensions: TYPEDSIGNATURES
using Downloads: Downloads
using NCDatasets: Dataset
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Center, set!
using Oceananigans.Grids: Bounded, Flat, RectilinearGrid
using Oceananigans.OutputReaders: Clamp, FieldTimeSeries
using Statistics: mean

using ..DataWrangling: DataWrangling, write_atomically
using ...Lands: PrescribedLand, build_river_routing

download_GEUS_cache::String = ""

function __init__()
    global download_GEUS_cache = DataWrangling.download_cache("GEUS")
    return nothing
end

struct GEUSSolidIceDischarge
    version :: Int
end

"""
$(TYPEDSIGNATURES)

Greenland solid ice discharge across the ~270 flux gates upstream of the marine-terminating glacier
fronts, from release `version` of the GEUS Dataverse record
[Mankoff et al. (2020)](@cite Mankoff2020solid) (doi:10.22008/promice/data/ice_discharge).
The record starts in 1986 and is updated every few weeks, each update a new `version`.
"""
GEUSSolidIceDischarge(; version = 132) = GEUSSolidIceDischarge(version)

Base.summary(dataset::GEUSSolidIceDischarge) = string("GEUSSolidIceDischarge(version=", dataset.version, ")")

const GEUS_DATAVERSE = "https://dataverse.geus.dk"
const GEUS_DISCHARGE_DOI = "doi:10.22008/promice/data/ice_discharge/d/v02"

# Before the record starts, each month takes its mean over these years.
const CLIMATOLOGY_YEARS = 1987:1996

const SECONDS_PER_YEAR = 365.25 * 86400

"""
$(TYPEDSIGNATURES)

Download `gate.nc` of `dataset`'s release into `dir`, unless it is already there, and return its path.
"""
function Downloads.download(dataset::GEUSSolidIceDischarge, dir = download_GEUS_cache)
    path = joinpath(dir, string("gate_V", dataset.version, ".nc"))
    isfile(path) && return path

    listing_url = string(GEUS_DATAVERSE, "/api/datasets/:persistentId/versions/", dataset.version,
                         ".0?persistentId=", GEUS_DISCHARGE_DOI)
    listing = String(take!(Downloads.download(listing_url, IOBuffer())))
    file_id = match(r"\"id\":(\d+),[^{}]*\"filename\":\"gate\.nc\"", listing)[1]
    file_url = string(GEUS_DATAVERSE, "/api/access/datafile/", file_id)

    return write_atomically(staging_path -> Downloads.download(file_url, staging_path), path)
end

"""
$(TYPEDSIGNATURES)

Return the longitude, latitude, first days of the months, and the monthly mean discharge
(Gt yr⁻¹, months × gates) in the `gate.nc` file at `path`. The irregularly sampled discharge
is interpolated linearly to daily values, which are averaged over every complete month.
"""
function monthly_gate_discharge(path)
    λ, φ, sample_dates, discharge = Dataset(path) do ds
        Float64.(ds["mean_lon"][:]), Float64.(ds["mean_lat"][:]), DateTime.(ds["time"][:]), Float64.(ds["discharge"][:, :])
    end

    months = firstdayofmonth(first(sample_dates)):Month(1):firstdayofmonth(last(sample_dates))
    months = filter(m -> m ≥ first(sample_dates) && lastdayofmonth(m) ≤ last(sample_dates), months)

    monthly_discharge = zeros(length(months), length(λ))
    for (m, month) in enumerate(months), day in month:Day(1):lastdayofmonth(month)
        n = min(searchsortedlast(sample_dates, day), length(sample_dates) - 1)
        w = (day - sample_dates[n]) / (sample_dates[n+1] - sample_dates[n])
        monthly_discharge[m, :] .+= ((1 - w) .* discharge[n, :] .+ w .* discharge[n+1, :]) ./ Dates.daysinmonth(month)
    end

    return λ, φ, months, monthly_discharge
end

"""
$(TYPEDSIGNATURES)

Return a [`PrescribedLand`](@ref) that forces the ocean with the Greenland solid ice discharge of
[`GEUSSolidIceDischarge`](@ref), as the `icebergs` freshwater component, for the `(start_date, end_date)`
window `dates`. Each flux gate is a mouth routed onto the coastal ocean cells of `grid` like the
GloFAS rivers (see [`build_river_routing`](@ref)).

The discharge is monthly: the mean of the daily linear interpolation of the GEUS samples, and the
1987–1996 mean of each calendar month before the record starts. It is stored as a mass flux per gate
(kg s⁻¹, 1 Gt = 10¹² kg).

Keyword Arguments
=================
- `maximum_search_radius`: search distance in `grid` cells for the ocean cell receiving a gate. Default: `5`.
- `spread_radius`: radius in degrees over which each gate's discharge is divided equally. Default: `1.2`.
- `maximum_spread_cells`: cap on that footprint, nearest first. Default: `8`.
"""
function GEUSPrescribedLand(grid;
                            dates,
                            dataset = GEUSSolidIceDischarge(),
                            dir = download_GEUS_cache,
                            time_indexing = Clamp(),
                            maximum_search_radius = 5,
                            spread_radius = 1.2,
                            maximum_spread_cells = 8)

    start_date, end_date = dates
    λ, φ, months, monthly_discharge = monthly_gate_discharge(Downloads.download(dataset, dir))
    end_date ≤ lastdayofmonth(last(months)) ||
        throw(ArgumentError("$(summary(dataset)) ends on $(lastdayofmonth(last(months))), before $end_date."))

    climatology = [mean(monthly_discharge[n, :] for n in eachindex(months)
                        if Dates.month(months[n]) == calendar_month && Dates.year(months[n]) in CLIMATOLOGY_YEARS)
                   for calendar_month in 1:12]

    # One month beyond the window at each end within the record, so the window is interpolated throughout
    series_months = firstdayofmonth(start_date) - Month(1):Month(1):min(firstdayofmonth(end_date) + Month(1), last(months))
    series_discharge = [month < first(months) ? climatology[Dates.month(month)] :
                                                monthly_discharge[searchsortedfirst(months, month), :]
                        for month in series_months]

    # Mid-month times, in seconds from `start_date`
    times = [Dates.value(Millisecond(month + (lastdayofmonth(month) + Day(1) - month) / 2 - start_date)) / 1000
             for month in series_months]

    Ngates = length(λ)
    FT = eltype(grid)
    gate_grid = RectilinearGrid(architecture(grid), FT; size = Ngates, x = (0, Ngates),
                                topology = (Bounded, Flat, Flat))

    discharge = FieldTimeSeries{Center, Center, Center}(gate_grid, times; time_indexing)
    for n in eachindex(times)
        set!(discharge[n], reshape(series_discharge[n] .* 1e12 ./ SECONDS_PER_YEAR, Ngates, 1, 1))
    end

    # The discharge is a mass flux per gate, so the routing weight is 1
    routing = build_river_routing(grid, collect(1:Ngates), ones(Int, Ngates), λ, φ, ones(FT, Ngates);
                                  maximum_search_radius, spread_radius, maximum_spread_cells)

    return PrescribedLand((; icebergs = discharge); river_routing = (; icebergs = routing))
end

end # module GEUS
