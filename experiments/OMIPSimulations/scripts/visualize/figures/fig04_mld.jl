# Figure 4: MLD seasonal min/max with optional dBM reference row (1° lat-lon regrid).
fig04(caches, labels, cases) =
    mld_maps(caches, labels;
             summer = (:mld_min_latlon, :mld_min_dbm_latlon, "Min MLD (summer)"),
             winter = (:mld_max_latlon, :mld_max_dbm_latlon, "Max MLD (winter)"),
             filename = "fig04_mld.png")

# Model maps of `summer`/`winter = (model_sym, dbm_sym, title)` per case, plus the dBM
# reference maps when any case carries the climatology.
function mld_maps(caches, labels; summer, winter, filename)
    ncases = length(labels)
    label_with_dbm = findfirst(lab -> !isnothing(get_field(caches[lab], summer[2])), labels)
    nrows = if isnothing(label_with_dbm)
        2
    elseif ncases >= 2
        3
    else
        4
    end
    fig = Figure(size = (800 * ncases, 450 * nrows), fontsize = 14)
    for (i, lab) in enumerate(labels)
        surface_panel!(fig, [1, 2i-1], get_field(caches[lab], summer[1]);
               title = "$lab: $(summer[3])",
               colormap = Reverse(:deep), colorrange = (0, 70), label = "m")
        surface_panel!(fig, [2, 2i-1], get_field(caches[lab], winter[1]);
               title = "$lab: $(winter[3])",
               colormap = Reverse(:deep), colorrange = (0, 500), label = "m")
    end
    if !isnothing(label_with_dbm)
        ref_label = labels[label_with_dbm]
        min_pos = [3, 1]
        max_pos = ncases >= 2 ? [3, 3] : [4, 1]
        surface_panel!(fig, min_pos, get_field(caches[ref_label], summer[2]);
               title = "dBM climatology: $(summer[3])",
               colormap = Reverse(:deep), colorrange = (0, 70), label = "m")
        surface_panel!(fig, max_pos, get_field(caches[ref_label], winter[2]);
               title = "dBM climatology: $(winter[3])",
               colormap = Reverse(:deep), colorrange = (0, 500), label = "m")
    end
    savefig(fig, filename)
end
