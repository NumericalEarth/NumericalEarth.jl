using JLD2
using Dates
using Statistics

using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum

"""
    assemble_figure1_bundle(; scalars_file, output_path,
                              start_year = 1958, end_year = 2018,
                              hemisphere = :global,
                              include_obs = true)

Read a scalar JLD2 written by `build_en4_proxy_timeseries` or the
simulation's `:averages` writer, aggregate to annual means, optionally load
observational datasets (HadISST, ERSST, IAP OHC) and compute their annual
global means, then write everything to a single bundle JLD2 suitable for
plotting.

The bundle layout is:

```
years       :: Vector{Int}
tosga       :: Vector{Float64}      # volume-averaged ocean T [°C]
tossga      :: Vector{Float64}      # surface T [°C]
ohc300      :: Vector{Float64}      # 0–300 m OHC [J]
sossga      :: Vector{Float64}      # surface S [psu]

# only present if include_obs:
hadisst_yr  :: Vector{Int}
hadisst_sst :: Vector{Float64}
ersst_yr    :: Vector{Int}
ersst_sst   :: Vector{Float64}
iap_yr      :: Vector{Int}
iap_ohc     :: Vector{Float64}      # in the obs native units (10²² J/m²)
```

The `scripts/paper_figure_1_plot.jl` script consumes this bundle.
"""
function assemble_figure1_bundle(;
        scalars_file::AbstractString,
        output_path::AbstractString,
        start_year::Int = 1958,
        end_year::Int   = 2018,
        hemisphere::Symbol = :global,
        include_obs::Bool = true,
    )

    suf = hemisphere === :north ? "_nh" :
          hemisphere === :south ? "_sh" : ""

    (times, tosga, tossga, ohc300, sossga) =
        _read_model_timeseries(scalars_file, suf)

    yrs, tosga_a  = _annual_mean(times, tosga,  start_year, end_year)
    _,   tossga_a = _annual_mean(times, tossga, start_year, end_year)
    _,   ohc300_a = _annual_mean(times, ohc300, start_year, end_year)
    _,   sossga_a = _annual_mean(times, sossga, start_year, end_year)

    bundle = Dict{String, Any}(
        "years"    => yrs,
        "tosga"    => tosga_a,
        "tossga"   => tossga_a,
        "ohc300"   => ohc300_a,
        "sossga"   => sossga_a,
    )

    obs_status = String[]
    if include_obs
        for (label, key, loader) in (
                ("HadISST", "hadisst", () -> annual_obs_sst_series(HadISSTSST(), start_year, end_year)),
                ("ERSST",   "ersst",   () -> annual_obs_sst_series(ERSSTv5(),    start_year, end_year)),
                ("IAP OHC", "iap",     () -> annual_obs_iap_series(start_year, end_year)),
            )
            try
                (yr, val) = loader()
                bundle[key * "_yr"] = yr
                bundle[key * (label == "IAP OHC" ? "_ohc" : "_sst")] = val
                push!(obs_status, "  $label: OK")
            catch err
                push!(obs_status, "  $label: FAILED ($(sprint(showerror, err)))")
            end
        end
    end

    jldopen(output_path, "w") do f
        for (k, v) in bundle
            f[k] = v
        end
    end

    println("=" ^ 60)
    println("Figure 1 bundle written to $output_path")
    println("Bundle keys: ", join(sort(collect(keys(bundle))), ", "))
    if include_obs
        println("Observational sources:")
        foreach(println, obs_status)
    end
    println("=" ^ 60)
    return output_path
end

function _read_model_timeseries(path, suf)
    f = jldopen(path, "r")
    times   = DateTime.(f["times"])
    tosga   = f["tosga"  * suf]
    tossga  = f["tossga" * suf]
    ohc300  = f["ohc300" * suf]
    sossga  = f["sossga" * suf]
    close(f)
    return (times, tosga, tossga, ohc300, sossga)
end

function _annual_mean(times, values, y0, y1)
    years = Dates.year.(times)
    mask  = (y0 .≤ years .≤ y1)
    yrs   = unique(years[mask])
    annual = [mean(values[mask .& (years .== y)]) for y in yrs]
    return yrs, annual
end

# ---- Observational helpers ----

"""
    annual_obs_sst_series(dataset, y0, y1)

Load each monthly file of `dataset` (HadISSTSST or ERSSTv5) within
`y0..y1`, compute the global area-weighted mean, and return
`(years, values)` with one annual mean per year.
"""
function annual_obs_sst_series(dataset, y0, y1)
    yrs  = collect(y0:y1)
    vals = zeros(Float64, length(yrs))
    for (iy, y) in enumerate(yrs)
        monthly = Float64[]
        for m in 1:12
            md = Metadatum(:sea_surface_temperature;
                           dataset = dataset, date = DateTime(y, m, 1))
            NumericalEarth.DataWrangling.download_dataset(md)
            arr = NumericalEarth.DataWrangling.retrieve_data(md)
            arr2d = dropdims(arr; dims = 3)
            lat = _obs_latitudes(dataset)
            push!(monthly, _area_weighted_mean(arr2d, lat))
        end
        vals[iy] = mean(monthly)
    end
    return yrs, vals
end

function annual_obs_iap_series(y0, y1)
    ds = IAPOceanHeatContent()
    yrs  = collect(y0:y1)
    vals = zeros(Float64, length(yrs))
    for (iy, y) in enumerate(yrs)
        monthly = Float64[]
        for m in 1:12
            md = Metadatum(:ocean_heat_content;
                           dataset = ds, date = DateTime(y, m, 1))
            try
                NumericalEarth.DataWrangling.download_dataset(md)
                arr = NumericalEarth.DataWrangling.retrieve_data(md)
                arr2d = dropdims(arr; dims = 3)
                lat = _obs_latitudes(ds)
                push!(monthly, _area_weighted_mean(arr2d, lat))
            catch err
                @warn "IAP missing for $(y)-$(m)" exception = err
            end
        end
        vals[iy] = isempty(monthly) ? NaN : mean(monthly)
    end
    return yrs, vals
end

function _area_weighted_mean(arr::AbstractMatrix, lat::AbstractVector)
    Nx, Ny = size(arr)
    w = cosd.(lat)
    num = 0.0
    den = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        v = arr[i, j]
        isnan(v) && continue
        num += v * w[j]
        den += w[j]
    end
    return den > 0 ? num / den : NaN
end

_obs_latitudes(::HadISSTSST) = collect(range(-89.5, 89.5; length = 180))
_obs_latitudes(::HadISSTICE) = collect(range(-89.5, 89.5; length = 180))
_obs_latitudes(::ERSSTv5)    = collect(range(-88.0, 88.0; length = 89))
_obs_latitudes(::IAPOceanHeatContent) = collect(range(-89.5, 89.5; length = 180))
