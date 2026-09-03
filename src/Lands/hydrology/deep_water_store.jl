#####
##### `DeepWaterStore` — a prognostic deep reservoir under a soil hydrology.
#####
##### The wrapped soil exchanges water with the reservoir through its own bottom
##### closure (a `DarcyDeepLiquidFlux`), reading the reservoir's pressure head from
##### its `deep_pressure_head` Field. This closure makes the reservoir prognostic: a
##### store `Mᵈ` (kg m⁻²) of thickness `hᵈ` that receives what the soil sends down,
##### returns capillary water when it is the wetter of the two, and drains through
##### its own bottom closure. Each step, around the delegated soil step:
#####
#####   θᵈ = Mᵈ / (ρˡ hᵈ),  𝒮ᵈ = (θᵈ − θʳ)/(ν − θʳ),  Πᵈ = Π(𝒮ᵈ)   # written into the soil's deep head
#####   (soil step: Jˡᵇ = ρˡ K (Πᵈ − Π − ℓ)/ℓ, published as `deep_liquid_flux`)
#####   Jᵈ      = deep_liquid_flux(drainage, Mᵈ, θᵈ, 𝒮ᵈ, Πᵈ, K(𝒮ᵈ), 0)      # the store's own bottom
#####   Mᵈⁿ⁺¹   = Mᵈ + Δt (Jᵈ − Jˡᵇ)
#####
##### The store shares the soil's porosity, residual fraction, retention curve and
##### conductivity curve, so the exchange timescale is set by parameters the soil
##### already carries: linearized, τ ≈ ℓ hᵈ C / K with C = dθ/dΠ.
#####

"""
    DeepWaterStore(FT = Oceananigans.defaults.FloatType;
                   soil,
                   thickness,
                   drainage = FreeDrainageFlux(FT))

Prognostic deep reservoir `Mᵈ` (kg m⁻²) of thickness `thickness` (`hᵈ`, m; a scalar or a
per-cell `Field`) under a soil hydrology `soil` — a [`VariablySaturatedHydrology`](@ref)
whose `deep_liquid_flux` is a [`DarcyDeepLiquidFlux`](@ref) and whose `deep_pressure_head`
is a `Field`, which this closure overwrites every step with the reservoir's head
`Π(𝒮ᵈ)` from the soil's retention curve. The reservoir gains what the soil drains into it,
gives back capillary rise, and loses water through `drainage`, any deep-flux closure
([`FreeDrainageFlux`](@ref), [`LinearReservoirDrainage`](@ref), [`NoDeepLiquidFlux`](@ref))
evaluated at the reservoir's own saturation; that flux is published as
`deep_drainage_flux`.

```jldoctest
julia> using Oceananigans, NumericalEarth

julia> grid = RectilinearGrid(size = 1, x = (0, 1), y = (0, 1), z = (-1, 0), topology = (Flat, Flat, Bounded));

julia> soil = VariablySaturatedHydrology(slab_depth = 0.3, porosity = 0.4, storage_height = 0.1,
                                         retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 2, pore_size_uniformity = 1.5),
                                         hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 1.5),
                                         deep_liquid_flux = DarcyDeepLiquidFlux(exchange_length = 0.5),
                                         deep_pressure_head = CenterField(grid));

julia> summary(DeepWaterStore(soil = soil, thickness = 0.7))
"DeepWaterStore(soil=VariablySaturatedHydrology(slab_depth=0.3, porosity=0.4, retention=VanGenuchtenRetention(α=2.0, n=1.5), deep=DarcyDeepLiquidFlux(exchange_length=0.5, liquid_density=1000.0), runoff=NoRunoff), thickness=0.7, drainage=FreeDrainageFlux(liquid_density=1000.0))"
```
"""
struct DeepWaterStore{S, H, D} <: AbstractHydrology
    soil      :: S
    thickness :: H
    drainage  :: D
end

function DeepWaterStore(FT::Type = Oceananigans.defaults.FloatType;
                        soil,
                        thickness,
                        drainage = FreeDrainageFlux(FT))
    soil.deep_pressure_head isa AbstractField ||
        throw(ArgumentError("DeepWaterStore needs a soil whose deep_pressure_head is a Field"))
    return DeepWaterStore(soil, normalize_property(FT, thickness), drainage)
end

Adapt.adapt_structure(to, h::DeepWaterStore) =
    DeepWaterStore(Adapt.adapt(to, h.soil), Adapt.adapt(to, h.thickness), Adapt.adapt(to, h.drainage))

prognostic_variables(h::DeepWaterStore) =
    merge_unique(prognostic_variables(h.soil), (:deep_water_storage,))

flux_variables(h::DeepWaterStore) = flux_variables(h.soil)

diagnostic_variables(h::DeepWaterStore) =
    merge_unique(diagnostic_variables(h.soil), (:deep_drainage_flux,))

initial_flux(h::DeepWaterStore, name::Symbol, grid) = initial_flux(h.soil, name, grid)
initial_diagnostic(h::DeepWaterStore, name::Symbol, grid) = initial_diagnostic(h.soil, name, grid)

update_diagnostics!(h::DeepWaterStore, land) = update_diagnostics!(h.soil, land)

saturation(h::DeepWaterStore, land) = saturation(h.soil, land)

EarthSystemModels.surface_retention_curve(h::DeepWaterStore) =
    EarthSystemModels.surface_retention_curve(h.soil)

@inline function deep_liquid_fraction(i, j, h::DeepWaterStore, Mᵈ)
    FT = typeof(Mᵈ)
    hᵈ = convert(FT, property_value(h.thickness, i, j))
    return Mᵈ / (convert(FT, h.soil.liquid_density) * hᵈ)
end

@kernel function _deep_store_head!(Πᵈ, Mᵈ, h, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        θᵈ = deep_liquid_fraction(i, j, h, Mᵈ[i, j, 1])
        𝒮ᵈ = liquid_saturation(i, j, grid, h.soil, θᵈ)
        Πᵈ[i, j, 1] = pressure_head(i, j, grid, h.soil.retention_curve, 𝒮ᵈ)
    end
end

@kernel function _deep_store_step!(Mᵈ, Jᵈ_diag, Jˡᵇ, h, grid, Δt, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Mij = Mᵈ[i, j, 1]
        θᵈ  = deep_liquid_fraction(i, j, h, Mij)
        𝒮ᵈ  = liquid_saturation(i, j, grid, h.soil, θᵈ)
        Πᵈ  = pressure_head(i, j, grid, h.soil.retention_curve, 𝒮ᵈ)
        Kᵈ  = hydraulic_conductivity(i, j, grid, h.soil.hydraulic_conductivity, 𝒮ᵈ)
        Jᵈ  = deep_liquid_flux(i, j, grid, h.drainage, Mij, θᵈ, 𝒮ᵈ, Πᵈ, Kᵈ, zero(Mij), time)
        Mᵈ[i, j, 1]      = Mij + Δt * (Jᵈ - Jˡᵇ[i, j, 1])
        Jᵈ_diag[i, j, 1] = Jᵈ
    end
end

function time_step!(h::DeepWaterStore, land, Δt, time)
    arch = architecture(land.grid)
    launch!(arch, land.grid, :xy, _deep_store_head!,
            h.soil.deep_pressure_head, land.prognostic.deep_water_storage, h, land.grid)
    time_step!(h.soil, land, Δt, time)
    launch!(arch, land.grid, :xy, _deep_store_step!,
            land.prognostic.deep_water_storage,
            land.diagnostics.deep_drainage_flux,
            land.diagnostics.deep_liquid_flux,
            h, land.grid, Δt, time)
    return nothing
end

Base.summary(h::DeepWaterStore) =
    string("DeepWaterStore(soil=", summary(h.soil),
           ", thickness=", prettysummary(h.thickness),
           ", drainage=", summary(h.drainage), ")")
