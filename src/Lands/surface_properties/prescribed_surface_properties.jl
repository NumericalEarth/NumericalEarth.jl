#####
##### `PrescribedSurfaceProperties` — per-cell radiative and aerodynamic
##### land-surface properties.
#####
##### Owns seven `CenterField`s: `albedo`, `emissivity`,
##### `roughness_length`, `vegfrac`, `lai`, `r_smin`, `is_urban`.
##### These are static within a run and are typically populated by
##### `apply_land_classifications!` from a USGS-style lookup table; they
##### can also be set per-cell after construction with `set!`.
#####
##### The four "shared" Fields (`vegfrac`, `lai`, `r_smin`, `is_urban`)
##### are read by the hydrology closure (e.g. `BucketWithSnow`) through
##### the standard `surface` argument to its kernels — the "shared
##### surface properties" channel from the parent design doc §2.4.
#####

"""
    PrescribedSurfaceProperties(grid;
                                albedo = 0.2,
                                emissivity = 0.97,
                                roughness_length = 0.1,
                                vegfrac = 0.0,
                                lai = 0.0,
                                r_smin = 100.0,
                                is_urban = 0.0)

Per-cell radiative and aerodynamic surface properties on `grid`. Each
keyword may be a `Number` (broadcast to fill a fresh `CenterField`) or
an existing `AbstractField` (used as-is).
"""
struct PrescribedSurfaceProperties{F} <: AbstractSurfaceProperties
    albedo            :: F
    emissivity        :: F
    roughness_length  :: F
    vegfrac           :: F
    lai               :: F
    r_smin            :: F
    is_urban          :: F
end

function PrescribedSurfaceProperties(grid;
                                     albedo = 0.2,
                                     emissivity = 0.97,
                                     roughness_length = 0.1,
                                     vegfrac = 0.0,
                                     lai = 0.0,
                                     r_smin = 100.0,
                                     is_urban = 0.0)

    _to_field(v::AbstractField) = v
    function _to_field(v::Number)
        f = CenterField(grid)
        fill!(f, v)
        return f
    end

    α  = _to_field(albedo)
    ε  = _to_field(emissivity)
    z₀ = _to_field(roughness_length)
    vf = _to_field(vegfrac)
    L  = _to_field(lai)
    rs = _to_field(r_smin)
    u  = _to_field(is_urban)

    F = typeof(α)
    return PrescribedSurfaceProperties{F}(α, ε, z₀, vf, L, rs, u)
end

prognostic_variables(::PrescribedSurfaceProperties) = ()
flux_variables(::PrescribedSurfaceProperties)       = ()

update_diagnostics!(::PrescribedSurfaceProperties, state, fluxes, surface, parameters, grid) = nothing

albedo(s::PrescribedSurfaceProperties, state, parameters)                    = s.albedo
emissivity(s::PrescribedSurfaceProperties, state, parameters)                = s.emissivity
momentum_roughness_length(s::PrescribedSurfaceProperties, state, parameters) = s.roughness_length
scalar_roughness_length(s::PrescribedSurfaceProperties, state, parameters)   = s.roughness_length

Base.summary(::PrescribedSurfaceProperties{F}) where F =
    string("PrescribedSurfaceProperties{", F, "}")
