# NaN-out samples whose FLUXNET quality flag exceeds `threshold` (0 = measured,
# 1 = good-quality gap-fill, 2 = medium, 3 = poor). Variables without a `_QC`
# companion column (e.g. USTAR, NETRAD) are left untouched.
function mask_low_quality!(fts, site, md, name, threshold)
    columns = fluxnet_columns(site)
    qc_name = FLUXNET_variable_names[name] * "_QC"
    haskey(columns, qc_name) || return fts

    qc = columns[qc_name]
    data = Array(interior(fts))
    for (t, datum) in enumerate(md)
        index = fluxnet_time_index(site, datum.dates)
        (isnothing(index) || isnan(qc[index]) || qc[index] ≤ threshold) && continue
        selectdim(data, ndims(data), t) .= NaN
    end
    copyto!(interior(fts), data)
    return fts
end

"""
    fluxnet_flux_observations(site::FLUXNETSite, architecture = CPU(), FT = Float64;
                              start_date = first_date(site, :sensible_heat_flux),
                              end_date = last_date(site, :sensible_heat_flux),
                              quality_control = nothing,
                              max_gap = 0)

Return the measured surface-energy-balance fluxes for a FLUXNET tower as a
`NamedTuple` of single-column `FieldTimeSeries`, for use as calibration targets:

- `H`: sensible heat flux `H_F_MDS` (W m⁻²)
- `LE`: latent heat flux `LE_F_MDS` (W m⁻²)
- `ustar`: friction velocity `USTAR` (m s⁻¹)
- `Rn`: net radiation `NETRAD` (W m⁻²)
- `G`: ground heat flux `G_F_MDS` (W m⁻²)

Keyword Arguments
=================
- `start_date`, `end_date`: time range to load.
- `quality_control`: if set to an integer, samples whose `_QC` flag exceeds it are
  set to `NaN` (e.g. `0` keeps only measured values, `1` also keeps good-quality
  gap-fill). Defaults to `nothing` (no masking).
- `max_gap`: maximum gap length (in time steps) to fill by linear interpolation.
  Defaults to `0` — targets are left with `NaN` gaps so a loss can skip them.
"""
function fluxnet_flux_observations(site::FLUXNETSite, architecture = CPU(), FT = Float64;
                                   start_date = first_date(site, :sensible_heat_flux),
                                   end_date = last_date(site, :sensible_heat_flux),
                                   quality_control = nothing,
                                   max_gap = 0)

    grid = RectilinearGrid(architecture, FT; size=(), topology=(Flat, Flat, Flat))

    function flux_fts(name)
        md = Metadata(name; dataset = site, start_date, end_date, dir = site.dir)
        fts = FieldTimeSeries(md, grid; time_indices_in_memory = length(md))
        isnothing(quality_control) || mask_low_quality!(fts, site, md, name, quality_control)
        max_gap > 0 && fill_gaps!(fts; max_gap)
        return fts
    end

    return (; H     = flux_fts(:sensible_heat_flux),
              LE    = flux_fts(:latent_heat_flux),
              ustar = flux_fts(:friction_velocity),
              Rn    = flux_fts(:net_radiation),
              G     = flux_fts(:ground_heat_flux))
end
