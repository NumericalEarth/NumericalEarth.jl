using Oceananigans.OutputReaders: FieldTimeSeries, InMemory
using Oceananigans.Fields: interior
using Oceananigans.Operators: Δxᶜᶠᶜ, Δyᶠᶜᶜ, Δzᶜᶠᶜ, Δzᶠᶜᶜ

# Seconds per year over m³ per km³: converts a volume flux in m³ s⁻¹ to km³ yr⁻¹, the unit every
# published Arctic freshwater transport is quoted in.
const cubic_kilometers_per_year = (365 * 24 * 3600) / 1e9

"""
    StraitSection

A rectangular strait section on a tripolar/ORCA grid.

* `i`, `j` are 1-based index ranges into the interior grid.
* `axis` is `:v` for a zonal (constant-`j`) section, where transport is
  ``∑ vₒ \\, Δx \\, Δz``, and `:u` for a meridional (constant-`i`) section,
  where transport is ``∑ uₒ \\, Δy \\, Δz``.
"""
struct StraitSection
    i :: UnitRange{Int}
    j :: UnitRange{Int}
    axis :: Symbol
end

# Per-configuration section indices. The half-degree indices are derived
# from a 720x360 TripolarGrid; the ORCA indices from ORCAGrid(ORCAOne()).
# Bering, Drake and ITF are picked at the cells closest to standard
# observational sections (Bering Strait ~66°N/169°W, Drake ~67°W/57°S,
# ITF ~110°-130°E/8.5°S).
# TODO: add `fram` and `davis` here. They must be found the same way the ORCA ones were — by walking
# the actual grid coordinates for a `j` row whose wet cells form one unbroken land-to-land run across
# the strait — not by scaling the ORCA indices, which live on a different mesh.
strait_sections(::Val{:halfdegree}) = (
    bering = StraitSection(212:218, 314:314, :v),
    drake  = StraitSection(447:447,  32:54,  :u),
    itf    = StraitSection( 83:122, 154:154, :v),
)

# Fram and Davis are the Arctic freshwater gateways. Both are complete land-to-land transects on the
# ORCA mesh — the wet cells listed form a single unbroken run bounded by Greenland to one side and by
# Svalbard (Fram) or Baffin Island (Davis) to the other, so their flux is the total exchange:
#   fram  j=275 runs 14.7°W..9.6°E at 80.1°N..78.8°N, ending against Svalbard;
#   davis j=254 runs 62.1°W..54.2°W at 65.7°N..66.9°N, essentially the observational array line.
# Rows of constant `j` slant in latitude this far north, which is why the endpoints are quoted as
# ranges; what matters for a transport is that the section closes, not that it follows a parallel.
strait_sections(::Val{:orca}) = (
    bering = StraitSection(112:118, 251:251, :v),
    drake  = StraitSection(221:221,  53:71,  :u),
    itf    = StraitSection( 39:58,  130:130, :v),
    fram   = StraitSection(268:278, 275:275, :v),
    davis  = StraitSection(234:241, 254:254, :v),
)

strait_sections(config::Symbol) = strait_sections(Val(config))

# Restrict `strait_sections(config)` to `names`. Reading a strait transport means decompressing the
# whole 3-D history to use a few hundred numbers per snapshot, so asking for only the sections you
# want is the difference between opening two velocity components and one.
function selected_strait_sections(config::Symbol, names)
    all_sections = strait_sections(config)
    for name in names
        haskey(all_sections, name) ||
            throw(ArgumentError("no section :$name for config :$config; have $(keys(all_sections))"))
    end
    return NamedTuple{Tuple(names)}(map(name -> getproperty(all_sections, name), Tuple(names)))
end

section_axes(sections) = (any(s -> s.axis == :u, values(sections)),
                          any(s -> s.axis == :v, values(sections)))

"""
    strait_transports(config::Symbol, fields_file::AbstractString;
                      backend = InMemory(10),
                      start_time = 0, stop_time = Inf,
                      sections = keys(strait_sections(config)),
                      stride = 2)

Compute time series of volume transport (Sv) through every section
`strait_sections(config)` defines, from the offline 3-D output
`fields_file` (typically `<prefix>_fields.jld2`).

Dispatches on `config`: `:halfdegree` for the 720x360 TripolarGrid,
`:orca` for the ORCAOne mesh.

Returns a `NamedTuple` with one `Vector{Float64}` of Sverdrups per
section, plus `time`. Positive is northward for a `:v` section and
eastward for a `:u` section.

The 3-D output is Zstd-compressed, so every snapshot must be decompressed
in full to read a few hundred numbers off one section — the cost is the
whole velocity history, not the section. `sections` limits which sections
are computed, and a request for zonal (`:v`) sections alone never opens
`uo`; `stride` subsamples snapshots, halving the read again. The default
of 2 still leaves 26 samples a year at the usual 15-day output cadence,
which annual means do not miss — pass `stride = 1` to read every snapshot.
"""
function strait_transports(config::Symbol, fields_file::AbstractString;
                           backend = InMemory(10),
                           start_time = 0,
                           stop_time = Inf,
                           sections = keys(strait_sections(config)),
                           stride = 2)

    selected = selected_strait_sections(config, sections)
    needs_u, needs_v = section_axes(selected)

    u_fts = needs_u ? FieldTimeSeries(fields_file, "uo"; backend = deepcopy(backend)) : nothing
    v_fts = needs_v ? FieldTimeSeries(fields_file, "vo"; backend = deepcopy(backend)) : nothing
    reference = something(u_fts, v_fts)
    grid = reference.grid

    snapshots = 1:stride:length(reference.times)
    times = collect(reference.times)[snapshots]
    transports = map(_ -> zeros(length(snapshots)), selected)

    for (m, n) in enumerate(snapshots)
        u_int = needs_u ? interior(u_fts[n]) : nothing
        v_int = needs_v ? interior(v_fts[n]) : nothing
        for (name, section) in pairs(selected)
            transports[name][m] = section_volume_flux(grid, u_int, v_int, section) * 1e-6
        end
    end

    in_window = (times .>= start_time) .& (times .<= stop_time)
    return merge(map(t -> t[in_window], transports), (; time = times[in_window]))
end

"""
    strait_freshwater_transports(config::Symbol,
                                 fields_file::AbstractString,
                                 surface_file::AbstractString;
                                 backend = InMemory(10),
                                 start_time = 0, stop_time = Inf,
                                 sections = keys(strait_sections(config)),
                                 stride = 2,
                                 reference_salinity = 34.8,
                                 ice_salinity = 4.0,
                                 ice_density = 900.0,
                                 freshwater_density = 1000.0)

Compute time series of freshwater transport (km³ yr⁻¹) through every section
`strait_sections(config)` defines, split into the liquid flux carried by the ocean and the solid
flux carried by sea ice. Reads `so`/`uo`/`vo` from the 3-D `fields_file` and
`siconc`/`sithick`/`siu`/`siv` from the 2-D `surface_file`, which are written on different
intervals — hence one time vector per component rather than a shared one.

Freshwater is measured against `reference_salinity`, the Arctic-mean value of
[Aagaard and Carmack (1989)](https://doi.org/10.1029/JC094iC10p14485) that every published Fram and
Davis estimate is quoted against: the liquid flux is `∫ v (S★ − S)/S★ dA` and the solid flux
`∫ ℵ h vⁱ (ρⁱ/ρᶠ) (S★ − Sⁱ)/S★ dl`. `ice_salinity` and `ice_density` default to the values the
sea-ice model itself carries, so the solid flux is the model's own freshwater content, not a
re-estimate of it.

**Sign convention**: positive is northward (`:v`) or eastward (`:u`), following
[`strait_transports`](@ref). Southward export out of the Arctic through Fram or Davis is therefore
*negative*. Observed magnitudes for orientation: Fram ≈ 2000–3000 km³ yr⁻¹ liquid and ≈ 2000
km³ yr⁻¹ solid, Davis ≈ 3000 km³ yr⁻¹ liquid, all southward.

`sections` and `stride` work as in [`strait_transports`](@ref) and matter more here, since the
liquid flux needs salinity on top of the velocity. `sections = (:fram, :davis, :bering)` — the
Arctic gateways and their main source, all zonal — reads two 3-D fields instead of three, and the
default `stride = 2` halves the snapshots read on top of that.

Returns `(; liquid, solid)`, each a `NamedTuple` of per-section vectors plus its own `time`.
"""
function strait_freshwater_transports(config::Symbol,
                                      fields_file::AbstractString,
                                      surface_file::AbstractString;
                                      backend = InMemory(10),
                                      start_time = 0,
                                      stop_time = Inf,
                                      sections = keys(strait_sections(config)),
                                      stride = 2,
                                      reference_salinity = 34.8,
                                      ice_salinity = 4.0,
                                      ice_density = 900.0,
                                      freshwater_density = 1000.0)

    selected = selected_strait_sections(config, sections)
    needs_u, needs_v = section_axes(selected)

    #####
    ##### Liquid, from the 3-D output. Salinity is unavoidable, but a request for zonal sections
    ##### only — which the Arctic gateways and Bering all are — never opens `uo` at all.
    #####

    u_fts = needs_u ? FieldTimeSeries(fields_file, "uo"; backend = deepcopy(backend)) : nothing
    v_fts = needs_v ? FieldTimeSeries(fields_file, "vo"; backend = deepcopy(backend)) : nothing
    S_fts = FieldTimeSeries(fields_file, "so"; backend = deepcopy(backend))
    grid  = S_fts.grid

    liquid_snapshots = 1:stride:length(S_fts.times)
    liquid_times = collect(S_fts.times)[liquid_snapshots]
    liquid = map(_ -> zeros(length(liquid_snapshots)), selected)

    for (m, n) in enumerate(liquid_snapshots)
        u_int = needs_u ? interior(u_fts[n]) : nothing
        v_int = needs_v ? interior(v_fts[n]) : nothing
        S_int = interior(S_fts[n])
        for (name, section) in pairs(selected)
            liquid[name][m] = section_liquid_freshwater_flux(grid, u_int, v_int, S_int, section,
                                                             reference_salinity) *
                              cubic_kilometers_per_year
        end
    end

    #####
    ##### Solid, from the 2-D sea-ice output
    #####

    ui_fts = FieldTimeSeries(surface_file, "siu";     backend = deepcopy(backend))
    vi_fts = FieldTimeSeries(surface_file, "siv";     backend = deepcopy(backend))
    ℵ_fts  = FieldTimeSeries(surface_file, "siconc";  backend = deepcopy(backend))
    h_fts  = FieldTimeSeries(surface_file, "sithick"; backend = deepcopy(backend))

    # Freshwater content of a unit volume of sea ice, relative to the reference salinity.
    ice_freshwater_fraction = (ice_density / freshwater_density) *
                              (reference_salinity - ice_salinity) / reference_salinity

    solid_snapshots = 1:stride:length(ui_fts.times)
    solid_times = collect(ui_fts.times)[solid_snapshots]
    solid = map(_ -> zeros(length(solid_snapshots)), selected)

    for (m, n) in enumerate(solid_snapshots)
        ui_int = interior(ui_fts[n])
        vi_int = interior(vi_fts[n])
        ℵ_int  = interior(ℵ_fts[n])
        h_int  = interior(h_fts[n])
        for (name, section) in pairs(selected)
            solid[name][m] = section_ice_freshwater_flux(grid, ui_int, vi_int, ℵ_int, h_int,
                                                         section, ice_freshwater_fraction) *
                             cubic_kilometers_per_year
        end
    end

    liquid_window = (liquid_times .>= start_time) .& (liquid_times .<= stop_time)
    solid_window  = (solid_times  .>= start_time) .& (solid_times  .<= stop_time)

    return (liquid = merge(map(t -> t[liquid_window], liquid), (; time = liquid_times[liquid_window])),
            solid  = merge(map(t -> t[solid_window],  solid),  (; time = solid_times[solid_window])))
end

function section_volume_flux(grid, u_int, v_int, section::StraitSection)
    Nz = size(grid, 3)
    total = 0.0

    if section.axis == :v
        for j in section.j, i in section.i, k in 1:Nz
            Δx = Δxᶜᶠᶜ(i, j, k, grid)
            Δz = Δzᶜᶠᶜ(i, j, k, grid)
            total += v_int[i, j, k] * Δx * Δz
        end
    elseif section.axis == :u
        for j in section.j, i in section.i, k in 1:Nz
            Δy = Δyᶠᶜᶜ(i, j, k, grid)
            Δz = Δzᶠᶜᶜ(i, j, k, grid)
            total += u_int[i, j, k] * Δy * Δz
        end
    else
        throw(ArgumentError("section.axis must be :u or :v, got $(section.axis)"))
    end

    return total
end

# Salinity interpolated from the two tracer cells straddling a velocity face. A velocity face is wet
# only when both neighbours are, so wherever the velocity is nonzero both salinities are valid ocean
# values, and dry faces contribute nothing regardless of what the output holds there.
@inline salinity_at_v_face(S_int, i, j, k) = (S_int[i, j, k] + S_int[i, j+1, k]) / 2
@inline salinity_at_u_face(S_int, i, j, k) = (S_int[i-1, j, k] + S_int[i, j, k]) / 2

# Liquid freshwater flux ∫ v (S★ − S)/S★ dA in m³ s⁻¹, positive northward/eastward.
function section_liquid_freshwater_flux(grid, u_int, v_int, S_int, section::StraitSection,
                                        reference_salinity)
    Nz = size(grid, 3)
    total = 0.0

    if section.axis == :v
        for j in section.j, i in section.i, k in 1:Nz
            S = salinity_at_v_face(S_int, i, j, k)
            isfinite(S) || continue
            Δx = Δxᶜᶠᶜ(i, j, k, grid)
            Δz = Δzᶜᶠᶜ(i, j, k, grid)
            total += v_int[i, j, k] * (reference_salinity - S) / reference_salinity * Δx * Δz
        end
    elseif section.axis == :u
        for j in section.j, i in section.i, k in 1:Nz
            S = salinity_at_u_face(S_int, i, j, k)
            isfinite(S) || continue
            Δy = Δyᶠᶜᶜ(i, j, k, grid)
            Δz = Δzᶠᶜᶜ(i, j, k, grid)
            total += u_int[i, j, k] * (reference_salinity - S) / reference_salinity * Δy * Δz
        end
    else
        throw(ArgumentError("section.axis must be :u or :v, got $(section.axis)"))
    end

    return total
end

# Solid freshwater flux ∫ ℵ h vⁱ (ρⁱ/ρᶠ) (S★ − Sⁱ)/S★ dl in m³ s⁻¹, positive northward/eastward.
# `ℵ h` is the ice volume per unit area — the same product the model integrates for `sivol` — and it
# is interpolated to the velocity face the ice is transported across.
function section_ice_freshwater_flux(grid, ui_int, vi_int, ℵ_int, h_int, section::StraitSection,
                                     ice_freshwater_fraction)
    k_surface = size(grid, 3)
    total = 0.0

    if section.axis == :v
        for j in section.j, i in section.i
            ice_volume = (ℵ_int[i, j, 1] * h_int[i, j, 1] + ℵ_int[i, j+1, 1] * h_int[i, j+1, 1]) / 2
            isfinite(ice_volume) || continue
            Δx = Δxᶜᶠᶜ(i, j, k_surface, grid)
            total += vi_int[i, j, 1] * ice_volume * Δx
        end
    elseif section.axis == :u
        for j in section.j, i in section.i
            ice_volume = (ℵ_int[i-1, j, 1] * h_int[i-1, j, 1] + ℵ_int[i, j, 1] * h_int[i, j, 1]) / 2
            isfinite(ice_volume) || continue
            Δy = Δyᶠᶜᶜ(i, j, k_surface, grid)
            total += ui_int[i, j, 1] * ice_volume * Δy
        end
    else
        throw(ArgumentError("section.axis must be :u or :v, got $(section.axis)"))
    end

    return total * ice_freshwater_fraction
end
