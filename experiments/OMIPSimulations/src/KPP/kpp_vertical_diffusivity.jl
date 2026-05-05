# KPPVerticalDiffusivity closure type, dispatch interface, and closure-field allocation.

"""
    KPPVerticalDiffusivity{TD, P, FT}

K-Profile Parameterization vertical mixing closure (Large, McWilliams,
& Doney, 1994). Holds the calibrated [`KPPParameters`](@ref) plus
optional clamps on viscosity / diffusivity magnitude.
"""
struct KPPVerticalDiffusivity{TD, P, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 1}
    parameters          :: P
    maximum_viscosity   :: FT
    maximum_diffusivity :: FT
end

"""
    KPPVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                           FT = Float64;
                           parameters = KPPParameters(FT),
                           maximum_viscosity = Inf,
                           maximum_diffusivity = Inf)

Construct a `KPPVerticalDiffusivity` with MITgcm defaults. The radiation
profile (for SW penetration) is auto-detected at compute time via
`get_radiative_forcing(model.forcing.T)` — no closure-side argument.
"""
function KPPVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                FT::DataType        = Float64;
                                parameters          = KPPParameters(FT),
                                maximum_viscosity   = Inf,
                                maximum_diffusivity = Inf)

    TD = typeof(time_discretization)
    P  = typeof(parameters)

    return KPPVerticalDiffusivity{TD, P, FT}(parameters,
                                             convert(FT, maximum_viscosity),
                                             convert(FT, maximum_diffusivity))
end

KPPVerticalDiffusivity(FT::DataType; kw...) =
    KPPVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

Adapt.adapt_structure(to, clo::KPPVerticalDiffusivity{TD, P, FT}) where {TD, P, FT} =
    KPPVerticalDiffusivity{TD, P, FT}(adapt(to, clo.parameters),
                                       clo.maximum_viscosity,
                                       clo.maximum_diffusivity)

#####
##### Type aliases and dispatch
#####

const KPPVD        = KPPVerticalDiffusivity
const KPPVDArray   = AbstractArray{<:KPPVD}
const FlavorOfKPP  = Union{KPPVD, KPPVDArray}

@inline viscosity_location(::FlavorOfKPP)   = (Center(), Center(), Face())
@inline diffusivity_location(::FlavorOfKPP) = (Center(), Center(), Face())

@inline viscosity(::FlavorOfKPP, diffusivities)        = diffusivities.κu
@inline diffusivity(::FlavorOfKPP, diffusivities, id)  = diffusivities.κc

with_tracers(tracers, closure::FlavorOfKPP) = closure

@inline time_discretization(::KPPVerticalDiffusivity{TD}) where TD = TD()

#####
##### Closure-field allocation
#####

function build_closure_fields(grid, clock, tracer_names, bcs, closure::FlavorOfKPP)
    # 3D fields filled by the per-interface kernel.
    κu  = Field((Center(), Center(), Face()), grid)
    κc  = Field((Center(), Center(), Face()), grid)
    γ   = Field((Center(), Center(), Face()), grid)

    # 2D column-level fields filled by the (i, j) kernel and read by the 3D one.
    hbl    = Field{Center, Center, Nothing}(grid)
    u★     = Field{Center, Center, Nothing}(grid)
    Bo     = Field{Center, Center, Nothing}(grid)
    Kint_u = Field{Center, Center, Nothing}(grid)
    Kint_c = Field{Center, Center, Nothing}(grid)

    # Cache surface tracer BCs for the nonlocal-flux override; per-tracer Q₀
    # is fetched via `getbc(top_tracer_bcs.<id>, ...)` in `diffusive_flux_z`.
    top_tracer_bcs = NamedTuple(name => bcs[name].top for name in tracer_names)

    return (; κu, κc, γ, hbl, u★, Bo, Kint_u, Kint_c, top_tracer_bcs)
end

#####
##### Show
#####

function Base.summary(closure::KPPVerticalDiffusivity)
    TD = nameof(typeof(time_discretization(closure)))
    return string("KPPVerticalDiffusivity{", TD, "}")
end

function Base.show(io::IO, closure::KPPVerticalDiffusivity)
    p = closure.parameters
    print(io, summary(closure), '\n',
              "├── Riᶜ:  ", prettysummary(p.Riᶜ),  '\n',
              "├── Ri∞:  ", prettysummary(p.Ri∞),  '\n',
              "├── κᵥ:   ", prettysummary(p.κᵥ),   '\n',
              "├── ν₀ˢʰ: ", prettysummary(p.ν₀ˢʰ), '\n',
              "├── κ₀ˢʰ: ", prettysummary(p.κ₀ˢʰ), '\n',
              "├── Cᵍ:   ", prettysummary(p.Cᵍ),   '\n',
              "├── maximum_viscosity:   ", prettysummary(closure.maximum_viscosity),   '\n',
              "└── maximum_diffusivity: ", prettysummary(closure.maximum_diffusivity))
end
