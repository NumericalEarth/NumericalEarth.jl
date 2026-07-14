#####
##### `CompositeSurfaceHumidity` — a single cell that evaporates from bare soil
##### *and* transpires through a canopy at the same time (ClimaLand's
##### `E_total = (1−σ) E_soil + T`). A canopy sits *above* the soil, so this is a
##### sum of two flux branches in a conductance network, not an average of two
##### humidities: soil and leaf are two saturated sources feeding one surface node
##### in parallel, both draining to the atmosphere through the aerodynamic
##### conductance `Gᵃ = Jᵃ/Δq`.
#####
#####                    Atmosphere qᵃᵗ
#####                         │  Gᵃ
#####                    surface node qˢ
#####                    ╱             ╲
#####           Gᵉ(𝒮) ╱                ╲ g_c = LAI·gₛ
#####          Soil (qᵉ)              Leaf (qᵛ⁺)
#####
##### Flux continuity `Gᵉ(qᵉ − qˢ) + g_c(qᵛ⁺ − qˢ) = Gᵃ(qˢ − qᵃᵗ)` gives the
##### conductance-weighted surface humidity, in the same `Δq`-multiplied form the
##### single-branch formulations use (so it stays finite as `Δq → 0`):
#####
#####     qˢ = ((Gᵉ qᵉ + g_c qᵛ⁺) Δq + Jᵃ qᵃᵗ) / ((Gᵉ + g_c) Δq + Jᵃ).
#####
##### The soil branch carries the [`DryLayerHumidity`](@ref) wet-branch blend: at
##### high saturation the soil skin itself saturates (`σ → 0`) and pins the
##### surface to `qᵛ⁺(Tⁱⁿ)` regardless of the canopy. The blend is applied to the
##### composite exactly as in the standalone soil model, so the limits are clean:
##### `g_c = 0` reproduces [`DryLayerHumidity`](@ref) bit-for-bit, and a fully-dry
##### soil (`σ = 1`, `Gᵉ = 0`) reproduces [`CanopyConductanceHumidity`](@ref).
#####
##### Where LAI enters (two physical channels, no blend weight): the canopy branch
##### through `g_c = LAI·gₛ`, and the soil branch through Beer–Lambert shading of
##### the available energy. Shading is an *energy* effect on the soil source
##### temperature; with a single skin temperature (leaf = soil = `Tₛ`) it cannot be
##### expressed, so both branches share `Tₛ` here. Distinct per-branch source
##### temperatures (`qᵉ = qᵛ⁺(Tⁱⁿ_soil)`, `qᵛ⁺ = qᵛ⁺(T_canopy)`) would require a
##### separate soil-skin and leaf temperature.
#####

"""
    struct CompositeSurfaceHumidity

Surface specific humidity for a vegetated soil cell that evaporates from bare
soil *and* transpires through the canopy at once — the two branches summed as
parallel conductances feeding one surface node (ClimaLand App. E). Combines a
soil humidity formulation (a [`DryLayerHumidity`](@ref)) with a canopy
formulation (a [`CanopyConductanceHumidity`](@ref)); plugs into the
`compute_interface_state` solver exactly where the single-branch formulations do.

Fields:
- `soil`   : the bare-soil branch (a [`DryLayerHumidity`](@ref)).
- `canopy` : the transpiring-canopy branch (a [`CanopyConductanceHumidity`](@ref)).
"""
struct CompositeSurfaceHumidity{S, C}
    soil   :: S
    canopy :: C
end

CompositeSurfaceHumidity(; soil, canopy) = CompositeSurfaceHumidity(soil, canopy)

Base.summary(q::CompositeSurfaceHumidity) =
    string("CompositeSurfaceHumidity(soil=", summary(q.soil), ", canopy=", summary(q.canopy), ")")
Base.show(io::IO, q::CompositeSurfaceHumidity) = print(io, summary(q))

@inline interface_phase(q::CompositeSurfaceHumidity) = interface_phase(q.soil)

# The composite reads both the surface saturation 𝒮 (soil δᵛ, canopy β) and the
# bulk land temperature (soil front temperature), so it materializes both.
@inline interface_hydrology_state(i, j, grid, ::CompositeSurfaceHumidity, land_state) =
    land_saturation(i, j, grid, land_state)
@inline interface_energy_state(i, j, grid, ::CompositeSurfaceHumidity, land_state) =
    (temperature = land_field_value(land_state.T, i, j),)

# LAI enters only through the canopy branch, so both hooks delegate to it.
@inline canopy_leaf_area_index(q::CompositeSurfaceHumidity) = canopy_leaf_area_index(q.canopy)
@inline interface_vegetation_state(i, j, grid, q::CompositeSurfaceHumidity, vegetation, time_interpolator) =
    interface_vegetation_state(i, j, grid, q.canopy, vegetation, time_interpolator)

@inline function compute_interface_humidity(q::CompositeSurfaceHumidity, Tₛ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    FT = eltype(Ψₛ)

    Gᵉ, qᵉ, σ, q_wet = dry_layer_terms(q.soil, Tₛ, Ψₛ, Ψₐ, ℙₐ)          # soil branch (+ wet-blend)
    g_c, q_leaf      = canopy_conductance_terms(q.canopy, Tₛ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)  # canopy branch

    qˢ⁻ = Ψₛ.specific_humidity
    qᵃᵗ = Ψₐ.q
    Jᵃ, Δq = atmospheric_vapor_flux(Ψₛ, Ψₐ, ℙₐ.thermodynamics_parameters)

    # Two sources (soil + canopy) in parallel behind the aerodynamic conductance.
    D      = (Gᵉ + g_c) * Δq + Jᵃ
    qˢ_dry = ifelse(D == 0, qˢ⁻, ((Gᵉ * qᵉ + g_c * q_leaf) * Δq + Jᵃ * qᵃᵗ) / D)

    # Wet-soil limit (σ → 0): the saturated soil skin pins the surface to qᵛ⁺(Tⁱⁿ),
    # so that with no canopy this reproduces `DryLayerHumidity` bit-for-bit.
    return convert(FT, q_wet + σ * (qˢ_dry - q_wet))
end

"""
    evaporation_partition(q::CompositeSurfaceHumidity, qˢ, Tₛ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)

Split the total surface vapor flux of a vegetated soil cell into its bare-soil
evaporation and canopy transpiration (mass flux, positive upward), given the
solved surface humidity `qˢ`. The soil branch draws saturated air `qᵉ` through the
dry-layer conductance `Gᵉ(𝒮)`, the canopy branch draws leaf-saturated air `qᵛ⁺`
through `g_c = LAI · gₛ`; both drain to the common surface node `qˢ`,

    E_soil = Gᵉ (qᵉ − qˢ),   T = g_c (qᵛ⁺ − qˢ),

and their sum equals the total evaporation `E = Gᵃ (qˢ − qᵃᵗ)` where the soil is
dry (`σ = 1`). Returns `(; soil_evaporation, transpiration)`.
"""
@inline function evaporation_partition(q::CompositeSurfaceHumidity, qˢ, Tₛ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)
    Gᵉ, qᵉ, σ, q_wet = dry_layer_terms(q.soil, Tₛ, Ψₛ, Ψₐ, ℙₐ)
    g_c, q_leaf      = canopy_conductance_terms(q.canopy, Tₛ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)
    soil_evaporation = Gᵉ * (qᵉ - qˢ)
    transpiration    = g_c * (q_leaf - qˢ)
    return (; soil_evaporation, transpiration)
end
