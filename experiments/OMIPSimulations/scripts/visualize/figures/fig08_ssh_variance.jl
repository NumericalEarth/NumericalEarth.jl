# Figure 8: SSH variance (m²), one panel per case.
function fig08(caches, labels, cases)
    fig = Figure(size = (800 * length(labels), 500), fontsize = 14)
    for (i, lab) in enumerate(labels)
        panel!(fig, [1, 2i-1], get_field(caches[lab], :ssh_variance);
               title = "$lab: SSH variance", colormap = :magma,
               colorrange = (0, 0.05), label = "m²")
    end
    savefig(fig, "fig08_ssh_variance.png")
end
