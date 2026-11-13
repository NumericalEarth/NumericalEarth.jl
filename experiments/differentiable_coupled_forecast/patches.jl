# Temporary monkey patches for the coupled forecast on ReactantState, in the form of test/reactant/patches.jl
# (NumericalEarth/NumericalEarth.jl#756): each stands in for an upstream fix that has not landed and names what
# retires it. When that lands, delete the patch and rerun `coupled_forecast_gradient.jl`; the first error is
# the next blocker. Delete this file when no patch is left.

using Reactant: Reactant
using Breeze: Breeze
using Oceananigans.Fields: Field, interior
using Breeze.AtmosphereModels: condensate_field_names, liquid_mass_fraction, ice_mass_fraction

# PATCH 1. `set!(model; ρ = HydrostaticallyBalancedDensity())`, which `atmosphere_model` runs on construction,
# refuses condensate by reducing every condensate field with `all(iszero, interior(field))`: element by element
# on the host, which a device array refuses ("Scalar indexing is disallowed"). Reactant reads a CPU buffer in
# place, so the CPU backend passes and only the GPU stops. Here each reduction runs on one host copy of the field.
# Retire when: a Breeze release carries NumericalEarth/Breeze.jl#1261 (fixes NumericalEarth/Breeze.jl#1260), whose
# Reactant extension reduces a host copy the same way.
host_interior(field) = (v = interior(field); view(Array(parent(v)), v.indices...))

function Breeze.AtmosphereModels.has_nonzero_mass_fraction(mass_fraction)
    field = mass_fraction isa Field ? mass_fraction : Field(mass_fraction)
    return !all(iszero, host_interior(field))
end

function Breeze.AtmosphereModels.has_condensate(model)
    for name in condensate_field_names(model.microphysics)
        all(iszero, host_interior(model.microphysical_fields[name])) || return true
    end

    if applicable(liquid_mass_fraction, model.microphysics, model)
        Breeze.AtmosphereModels.has_nonzero_mass_fraction(liquid_mass_fraction(model.microphysics, model)) && return true
    end

    if applicable(ice_mass_fraction, model.microphysics, model)
        Breeze.AtmosphereModels.has_nonzero_mass_fraction(ice_mass_fraction(model.microphysics, model)) && return true
    end

    return false
end

# PATCH 2. Leave the Enzyme-JAX patterns named in EXCLUDED_PATTERNS (comma separated) out of every compile of
# the session: every compile builds its pattern list through `optimization_passes(compile_options; ...)`, so the
# exclusions are added there and the original method is called in the world before this redefinition. In the
# hailmary gradients `greedy_while_loop_batch_fission` was quadratic in the loop body after reverse-mode AD and
# `while_deadresult` multiplied the post-AD IR of nested loops (EnzymeAD/Reactant.jl#3573); both are candidates
# here, where the acoustic substep loop nests inside the forecast loop. Off unless EXCLUDED_PATTERNS is set; read
# at the first include.
# Retire when: a gradient compile on Reactant ≥ 0.2.304 (Reactant_jll ≥ 0.0.438, with EnzymeAD/Enzyme-JAX#3733
# and EnzymeAD/Enzyme-JAX#3734, which closed EnzymeAD/Reactant.jl#3573) finishes in reasonable time without it.
if !(@isdefined excluded_patterns)
    excluded_patterns = String.(filter(!isempty, split(get(ENV, "EXCLUDED_PATTERNS", ""), ",")))
    if !isempty(excluded_patterns)
        world_before_pattern_exclusion = Base.get_world_counter()
        function Reactant.Compiler.optimization_passes(compile_options::Reactant.CompileOptions; kwargs...)
            excluded = unique(vcat(compile_options.excluded_passes, excluded_patterns))
            fields = (name === :excluded_passes ? excluded : getfield(compile_options, name) for name in fieldnames(Reactant.CompileOptions))
            return Base.invoke_in_world(world_before_pattern_exclusion, Reactant.Compiler.optimization_passes,
                                        Reactant.CompileOptions(fields...); kwargs...)
        end
        @info "Enzyme-JAX patterns excluded from every compile" excluded_patterns
    end
end
