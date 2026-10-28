# An opt-in workaround for the gradient compile, off unless EXCLUDED_PATTERNS is set. It changes a Reactant
# internal, so delete this file once the Enzyme-JAX pattern below is fixed.

using Reactant

# Leave the Enzyme-JAX patterns named in EXCLUDED_PATTERNS (comma separated) out of every compile of
# the session: every compile builds its pattern list through `optimization_passes(compile_options; ...)`, so
# the exclusions are added there and the original method is called in the world before this redefinition.
# In the hailmary gradients `greedy_while_loop_batch_fission` was quadratic in the loop body after reverse-mode
# AD and `while_deadresult` multiplied the post-AD IR of nested loops (EnzymeAD/Reactant.jl#3573); both are
# candidates here, where the acoustic substep loop nests inside the forecast loop. Read at the first include.
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
