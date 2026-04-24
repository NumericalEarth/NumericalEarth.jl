#
# Plot Iovino et al. 2023 Figure 1 from a bundle produced by
# `OMIPSimulations.assemble_figure1_bundle`. Run as a standalone script
# with CairoMakie available in the active environment.
#
# Example:
#   julia --project=.. paper_figure_1_plot.jl
#
# or from the REPL:
#   include("scripts/paper_figure_1_plot.jl")
#   plot_figure1_bundle(; bundle_file = "figure1_bundle.jld2")
#
# If the active env's Makie hits the Julia 1.12 / StaticArrays
# ambiguity bug (two duplicate `getindex` registrations), downgrade
# CairoMakie to 0.12–0.13, or run on Julia 1.11.
#

using JLD2
using CairoMakie

"""
    plot_figure1_bundle(; bundle_file, output_path = "iovino_figure1.png")

Read the JLD2 bundle written by `assemble_figure1_bundle` and render the
Iovino-style 4-panel figure.
"""
function plot_figure1_bundle(;
        bundle_file::AbstractString,
        output_path::AbstractString = "iovino_figure1.png",
    )

    f = jldopen(bundle_file, "r")
    years   = f["years"]
    tosga   = f["tosga"]
    tossga  = f["tossga"]
    ohc300  = f["ohc300"]
    sossga  = f["sossga"]
    hadisst = haskey(f, "hadisst_sst") ?
        (years = f["hadisst_yr"], sst = f["hadisst_sst"]) : nothing
    ersst   = haskey(f, "ersst_sst") ?
        (years = f["ersst_yr"],   sst = f["ersst_sst"])   : nothing
    iap     = haskey(f, "iap_ohc") ?
        (years = f["iap_yr"],     ohc = f["iap_ohc"])     : nothing
    close(f)

    fig = Figure(size = (900, 900), fontsize = 14)

    ax1 = Axis(fig[1, 1]; ylabel = "Temperature (°C)",
               title  = "Global volume averaged temperature")
    lines!(ax1, years, tosga; color = :orange, linewidth = 2, label = "model")
    axislegend(ax1; position = :rt)

    ax2 = Axis(fig[2, 1]; ylabel = "SST (°C)",
               title  = "Global average sea surface temperature")
    lines!(ax2, years, tossga; color = :orange, linewidth = 2, label = "model")
    if !isnothing(hadisst)
        lines!(ax2, hadisst.years, hadisst.sst;
               color = :mediumorchid, linewidth = 1.5, label = "HadISSTv1.1")
    end
    if !isnothing(ersst)
        lines!(ax2, ersst.years, ersst.sst;
               color = :gray, linewidth = 1.5, label = "ERSSTv5")
    end
    axislegend(ax2; position = :rt)

    ax3 = Axis(fig[3, 1]; ylabel = "HC (10²⁴ J)",
               title  = "Ocean heat content 0–300 m")
    lines!(ax3, years, ohc300 ./ 1e24;
           color = :orange, linewidth = 2, label = "model")
    if !isnothing(iap)
        # IAPv4.2 ships OHC as J/m² (column-integrated). Multiply by the
        # global ocean surface area to obtain total joules, then divide by
        # 1e24 to match the model panel.
        earth_ocean_area = 4π * 6.371e6^2 * 0.71  # ≈ 3.6e14 m²
        lines!(ax3, iap.years, iap.ohc .* earth_ocean_area ./ 1e24;
               color = :gray, linewidth = 1.5, label = "IAP OHC")
    end
    axislegend(ax3; position = :rt)

    ax4 = Axis(fig[4, 1]; xlabel = "Year", ylabel = "SSS (psu)",
               title  = "Global average sea surface salinity")
    lines!(ax4, years, sossga; color = :orange, linewidth = 2, label = "model")
    axislegend(ax4; position = :rt)

    save(output_path, fig)
    @info "Wrote" output_path
    return output_path
end
