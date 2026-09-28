# Figure 32: Treguier et al. (2023) seasonal MLD maps — winter is the Jan–Mar mean in the
# Northern Hemisphere and Jul–Sep in the Southern, summer the reverse — with dBM reference.
fig32(caches, labels, cases) =
    mld_maps(caches, labels;
             summer = (:mld_summer_latlon, :mld_summer_dbm_latlon, "Summer MLD (JAS N / JFM S)"),
             winter = (:mld_winter_latlon, :mld_winter_dbm_latlon, "Winter MLD (JFM N / JAS S)"),
             filename = "fig32_mld_seasonal.png")
