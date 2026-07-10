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
##### temperature; in today's single-source world (leaf = soil = skin temperature
##### `Tₛ`) it cannot be expressed, so v1 sums the two branches at a common skin
##### temperature. Per-branch source temperatures (`qᵉ = qᵛ⁺(Tⁱⁿ_soil)`,
##### `qᵛ⁺ = qᵛ⁺(T_canopy)`) wait on the energy-balance temperature abstraction.
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

@inline function compute_interface_humidity(q::CompositeSurfaceHumidity, Tₛ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ)
    FT = eltype(Ψₛ)

    Gᵉ, qᵉ, σ, q_wet = dry_layer_terms(q.soil, Tₛ, Ψₛ, Ψₐ, ℙₐ)   # soil branch (+ wet-blend)
    g_c, q_leaf      = canopy_conductance_terms(q.canopy, Tₛ, Ψₛ, Ψₐ, ℙₐ)  # canopy branch

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
