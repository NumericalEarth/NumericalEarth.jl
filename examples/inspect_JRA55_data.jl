using NumericalEarth
using CairoMakie
using Oceananigans
using Oceananigans.Units
using Printf

ℐꜜˢʷ_ts = FieldTimeSeries(Metadata(:downwelling_shortwave_radiation; dataset=RepeatYearJRA55()); time_indices_in_memory=8)
q_ts    = FieldTimeSeries(Metadata(:specific_humidity;               dataset=RepeatYearJRA55()); time_indices_in_memory=8)

times = ℐꜜˢʷ_ts.times
Nt = length(times)
n = Observable(1)

ℐꜜˢʷn = @lift ℐꜜˢʷ_ts[$n]
qn    = @lift q_ts[$n]

fig = Figure(size=(1400, 700))

axℐ = Axis3(fig[1, 1], aspect=(1, 1, 1))
axq = Axis3(fig[1, 2], aspect=(1, 1, 1))

title = @lift @sprintf("Repeat-year JRA55 forcing on year-day %.1f", times[$n] / days)

Label(fig[0, 1:2], title, fontsize=24)

sf = surface!(axℐ, ℐꜜˢʷn, colorrange=(0, 1200))

Colorbar(fig[2, 1], sf,
         vertical = false,
         width = Relative(0.5),
         flipaxis = false,
         label = "Downwelling shortwave radiation (W m⁻²)")

sf = surface!(axq, qn, colormap=:grays, colorrange=(0, 0.025))

Colorbar(fig[2, 2], sf,
         vertical = false,
         width = Relative(0.5),
         flipaxis = false,
         label = "Specific humidity (kg kg⁻¹)")

colgap!(fig.layout, 1, Relative(-0.15))
rowgap!(fig.layout, 1, Relative(-0.2))
rowgap!(fig.layout, 2, Relative(-0.2))

for ax in (axℐ, axq)
    hidedecorations!(ax)
    hidespines!(ax)
    ax.viewmode = :fit ## keeps the sphere from zooming in and out while it rotates
end

fig

snapshot_interval = 3hours ## JRA55 time resolution
rotation_period = 60days
rotation_rate = 2π / rotation_period

CairoMakie.record(fig, "JRA55_data.mp4", 1:Nt, framerate=16) do nn
    @info nn/Nt

    n[] = nn

    for ax in (axℐ, axq)
        ax.azimuth = nn * snapshot_interval * rotation_rate
    end
end
