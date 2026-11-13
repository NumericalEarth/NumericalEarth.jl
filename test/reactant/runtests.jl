# The Reactant tests, one file per group, each also runnable on its own:
#
#   unit             one piece at a time: clocks, flux assembly, data moved to ReactantState
#   coupled          coupled models: an ocean under a prescribed atmosphere, an atmosphere over land
#   nested           a nested Breeze atmosphere: compiled initialization and time_step!
#   nested_gradient  the gradient through one hour of a nested Breeze atmosphere (GPU only; skipped elsewhere)
#
#   julia --project=test/reactant test/reactant/runtests.jl                 # every group
#   julia --project=test/reactant test/reactant/runtests.jl unit nested     # some of them
#   julia --project=test/reactant test/reactant/test_reactant_unit.jl       # one file

reactant_test_groups = ("unit", "coupled", "nested", "nested_gradient")
selected_groups = isempty(ARGS) ? reactant_test_groups : ARGS

for group in selected_groups
    group in reactant_test_groups ||
        error("Unknown Reactant test group \"$group\"; the groups are $(join(reactant_test_groups, ", "))")
end

for group in selected_groups
    @info "Reactant tests: $group"
    include(joinpath(@__DIR__, "test_reactant_$group.jl"))
end
