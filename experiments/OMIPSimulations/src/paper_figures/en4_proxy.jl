using JLD2
using Dates
using Oceananigans

using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum, EN4Monthly

"""
    build_en4_proxy_timeseries(; dataset = EN4Monthly(),
                                 start_date, end_date,
                                 ρ₀ = 1025.0, cₚ = 3991.0,
                                 output_path)

Produce a JLD2 scalar time series that mimics the output of the simulation's
`:averages` writer, using EN4 gridded T / S as the data source. The 12
Figure-1 scalars (see `scalars.jl`) are written at monthly resolution with
the 1st of each month as the time coordinate.

The resulting file is readable by `plot_figure1`.

Arguments
=========

- `start_date::Date`, `end_date::Date`: inclusive month range (1st of the
  month is used).
- `dataset`: an `EN4Monthly()` (default).
- `ρ₀`, `cₚ`: reference density [kg/m³] and heat capacity [J/kg/K] used for
  OHC. Defaults match the standard Boussinesq ocean.
- `output_path`: JLD2 file to write.
"""
function build_en4_proxy_timeseries(;
        dataset = EN4Monthly(),
        start_date::Date,
        end_date::Date,
        ρ₀ = 1025.0,
        cₚ = 3991.0,
        output_path::AbstractString,
    )

    all_d = NumericalEarth.all_dates(dataset, :temperature)
    dates = filter(d -> DateTime(start_date) ≤ DateTime(d) ≤ DateTime(end_date), all_d)
    isempty(dates) && error("No EN4 dates within $start_date .. $end_date")

    keys = (
        :tosga,  :tosga_nh,  :tosga_sh,
        :tossga, :tossga_nh, :tossga_sh,
        :sossga, :sossga_nh, :sossga_sh,
        :ohc300, :ohc300_nh, :ohc300_sh,
    )
    series = Dict{Symbol, Vector{Float64}}(k => Float64[] for k in keys)
    times  = DateTime[]

    _only(op) = begin
        f = Field(op); compute!(f); only(f)
    end

    for date in dates
        T_meta = Metadatum(:temperature; dataset = dataset, date = date)
        S_meta = Metadatum(:salinity;    dataset = dataset, date = date)

        T = Field(T_meta)
        S = Field(S_meta)

        push!(series[:tosga],     _only(global_volume_T(T)))
        push!(series[:tosga_nh],  _only(global_volume_T(T; hemisphere = northern_hemisphere)))
        push!(series[:tosga_sh],  _only(global_volume_T(T; hemisphere = southern_hemisphere)))

        push!(series[:tossga],    _only(global_surface_T(T)))
        push!(series[:tossga_nh], _only(global_surface_T(T; hemisphere = northern_hemisphere)))
        push!(series[:tossga_sh], _only(global_surface_T(T; hemisphere = southern_hemisphere)))

        push!(series[:sossga],    _only(global_surface_S(S)))
        push!(series[:sossga_nh], _only(global_surface_S(S; hemisphere = northern_hemisphere)))
        push!(series[:sossga_sh], _only(global_surface_S(S; hemisphere = southern_hemisphere)))

        push!(series[:ohc300],    _only(global_ohc_300(T, ρ₀, cₚ)))
        push!(series[:ohc300_nh], _only(global_ohc_300(T, ρ₀, cₚ; hemisphere = northern_hemisphere)))
        push!(series[:ohc300_sh], _only(global_ohc_300(T, ρ₀, cₚ; hemisphere = southern_hemisphere)))

        push!(times, DateTime(date))
        @info "EN4 proxy: done $(date)"
    end

    jldopen(output_path, "w") do f
        f["times"] = times
        for (k, v) in series
            f[string(k)] = v
        end
    end

    @info "Wrote EN4 proxy scalar time series" output_path months=length(times)
    return output_path
end
