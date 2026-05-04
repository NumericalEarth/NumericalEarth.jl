function ComponentExchanger(land::PrescribedLand, grid)
    state = (; freshwater_flux = Field{Center, Center, Nothing}(grid))
    return ComponentExchanger(state, nothing)
end

# No initialization needed for land (uses _node interpolation)
initialize!(::ComponentExchanger, grid, land::PrescribedLand) = nothing
