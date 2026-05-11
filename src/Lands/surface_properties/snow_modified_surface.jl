#####
##### `SnowModifiedSurface{Base}` — snow-fraction-weighted decorator.
#####
##### Wraps a base surface-property closure and blends its outputs with
##### scalar snow end-members using `state.snow_fraction` (owned by the
##### hydrology closure, typically `BucketWithSnow`):
#####
#####   prop = (1 − f_sn) · prop_base + f_sn · prop_snow
#####
##### Cross-closure read of `state.snow_fraction` is allowed because the
##### channel is the shared `state` `NamedTuple` (§2.4 of the parent
##### design doc). The `SlabLand` constructor asserts that the paired
##### hydrology declares `:snow_fraction` when this decorator is used.
#####
##### Reference: Robinson & Kukla (1985) linear-blend form, without the
##### warm-T attenuation branch RUC LSM adds.
#####

"""
    SnowModifiedSurface(base;
                        snow_albedo = 0.80,
                        snow_emissivity = 0.98,
                        snow_roughness_length = 0.011)

Decorator that blends `base`'s surface properties with scalar snow
end-members weighted by `state.snow_fraction`. The decorator owns three
output Fields (`albedo`, `emissivity`, `roughness_length`) populated by
`update_diagnostics!` after the hydrology closure refreshes
`snow_fraction`.
"""
struct SnowModifiedSurface{Base, FT, F} <: AbstractSurfaceProperties
    base                  :: Base
    snow_albedo           :: FT
    snow_emissivity       :: FT
    snow_roughness_length :: FT
    albedo                :: F   # blended output
    emissivity            :: F
    roughness_length      :: F
end

function SnowModifiedSurface(base;
                             snow_albedo = 0.80,
                             snow_emissivity = 0.98,
                             snow_roughness_length = 0.011)

    α0  = albedo(base, NamedTuple(), nothing)
    ε0  = emissivity(base, NamedTuple(), nothing)
    z₀0 = momentum_roughness_length(base, NamedTuple(), nothing)

    grid = α0.grid

    out_α  = CenterField(grid); parent(out_α)  .= parent(α0)
    out_ε  = CenterField(grid); parent(out_ε)  .= parent(ε0)
    out_z₀ = CenterField(grid); parent(out_z₀) .= parent(z₀0)

    FT = promote_type(typeof(snow_albedo),
                      typeof(snow_emissivity),
                      typeof(snow_roughness_length))
    return SnowModifiedSurface{typeof(base), FT, typeof(out_α)}(
        base,
        convert(FT, snow_albedo),
        convert(FT, snow_emissivity),
        convert(FT, snow_roughness_length),
        out_α, out_ε, out_z₀)
end

prognostic_variables(s::SnowModifiedSurface) = prognostic_variables(s.base)
flux_variables(s::SnowModifiedSurface)       = flux_variables(s.base)

initial_state(s::SnowModifiedSurface, name::Symbol, grid) = initial_state(s.base, name, grid)
initial_flux(s::SnowModifiedSurface, name::Symbol, grid)  = initial_flux(s.base, name, grid)

@kernel function _snow_blend!(out, base, snow_end, snow_fraction)
    i, j = @index(Global, NTuple)
    @inbounds begin
        f = snow_fraction[i, j, 1]
        out[i, j, 1] = (1 - f) * base[i, j, 1] + f * snow_end
    end
end

function update_diagnostics!(s::SnowModifiedSurface, state, fluxes, surface, parameters, grid)
    update_diagnostics!(s.base, state, fluxes, surface, parameters, grid)
    arch = architecture(grid)
    f_sn = state.snow_fraction
    launch!(arch, grid, :xy, _snow_blend!,
            s.albedo,           albedo(s.base, state, parameters),                    s.snow_albedo,           f_sn)
    launch!(arch, grid, :xy, _snow_blend!,
            s.emissivity,       emissivity(s.base, state, parameters),                s.snow_emissivity,       f_sn)
    launch!(arch, grid, :xy, _snow_blend!,
            s.roughness_length, momentum_roughness_length(s.base, state, parameters), s.snow_roughness_length, f_sn)
    return nothing
end

albedo(s::SnowModifiedSurface, state, parameters)                    = s.albedo
emissivity(s::SnowModifiedSurface, state, parameters)                = s.emissivity
momentum_roughness_length(s::SnowModifiedSurface, state, parameters) = s.roughness_length
scalar_roughness_length(s::SnowModifiedSurface, state, parameters)   = s.roughness_length

Base.summary(s::SnowModifiedSurface{B,FT}) where {B,FT} =
    string("SnowModifiedSurface{", B, ", $FT}(α_sn=", s.snow_albedo, ")")

# SlabLand constructor calls this after `build_state` to validate that
# the paired hydrology owns `:snow_fraction`. Without it the decorator's
# `update_diagnostics!` would error on the first time step.
function _assert_surface_state_compatible(::SnowModifiedSurface, state::NamedTuple)
    if !haskey(state, :snow_fraction)
        throw(ArgumentError("SnowModifiedSurface requires a hydrology closure " *
                            "that owns :snow_fraction (e.g. BucketWithSnow)."))
    end
    return nothing
end
