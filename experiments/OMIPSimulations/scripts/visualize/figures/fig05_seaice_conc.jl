# Figure 5: March (row 1) and September (row 2) sea-ice concentration per case.
function fig05(caches, labels, cases)
    fig = Figure(size = (800 * length(labels), 900), fontsize = 14)
    for (i, lab) in enumerate(labels)
        sic_march     = get_field(caches[lab], :sic_march)
        sic_september = get_field(caches[lab], :sic_september)
        !isnothing(sic_march) && panel!(fig, [1, 2i-1], sic_march;
            title = "$lab: Sea-ice conc. March",
            colormap = :ice, colorrange = (0, 1), label = "fraction")
        !isnothing(sic_september) && panel!(fig, [2, 2i-1], sic_september;
            title = "$lab: Sea-ice conc. September",
            colormap = :ice, colorrange = (0, 1), label = "fraction")
    end
    savefig(fig, "fig05_seaice_conc.png")
end
